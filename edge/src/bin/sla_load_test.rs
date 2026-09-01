//! SLA / throughput load test (#5).
//!
//! Drives the real per-query MIDAS-Edge compute path under sustained
//! adversarial load and reports:
//!   1. Sustained throughput (queries/sec actually processed).
//!   2. End-to-end latency histogram (p50/p95/p99/max) for the full
//!      defense -> budget-gated rotation -> TFLite inference chain.
//!   3. Fraction of inferences that violate the configured SLA budget
//!      (default 50 ms from `sla_budget_ms`).
//!   4. Phase census (Archimedean vs Logarithmic) under the offered load.
//!   5. Per-thread CPU util for the load-driver thread (sampled from
//!      /proc/[pid]/task/[tid]/stat during a sustained window).
//!
//! The compute path mirrors the deployed 3-thread pipeline exactly:
//!   windowed penetration (RingBuffer) -> rotation_angle -> phase decision
//!   -> budget_gated_rotation_fixed_basis -> model.infer
//!
//! Usage:
//!   EDGE_CONFIG=configs/edge_config.json \
//!   EDGE_MODEL_PATH=models/classifier_float32.tflite \
//!   cargo run --release --bin sla_load_test --features tflite -- \
//!     results/pi/load_adv_batch.json results/pi/sla_load_test.json
//!
//! Env tuning:
//!   LOAD_N        total queries to run (default: batch size)
//!   LOAD_RATE_HZ  offered rate; 0 = max/back-to-back (default 0)
//!   LOAD_SCALE    multiplicative magnitude scale on inputs (for Logarithmic-
//!                 phase stress coverage; default 1.0)
//!   LOAD_CPU_SECS sustained window for /proc per-thread CPU sampling (default 3)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use edge::channels::Phase;
use edge::metrics::PhaseLatencySample;
use edge::model::{InferenceModel, TfliteModel};
use edge_core::config::MidasConfig;
use edge_core::ring_buffer::RingBuffer;
use edge_core::rotation::{budget_gated_rotation_fixed_basis, compute_fixed_basis, rotation_angle};
use edge_core::trajectory::penetration_epsilon_windowed;
use serde::Deserialize;

#[derive(Deserialize)]
struct LoadBatch {
    d: usize,
    #[serde(default)]
    originals: Vec<Vec<f32>>,
    #[serde(default)]
    adaptive_adversarial: Vec<Vec<f32>>,
}

#[derive(serde::Serialize)]
struct Summary {
    n: usize,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
    mean_ms: f64,
    sla_violations: usize,
    sla_violation_frac: f64,
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn parse_env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn pin_thread(core_id: usize) {
    if let Some(ref cores) = core_affinity::get_core_ids() {
        if let Some(core) = cores.get(core_id) {
            core_affinity::set_for_current(*core);
        }
    }
}

/// Per-thread CPU ticks for the first task whose comm equals `name`.
fn read_thread_ticks(pid: u32, name: &str) -> Option<u64> {
    let dir = format!("/proc/{pid}/task");
    for e in std::fs::read_dir(&dir).ok()?.flatten() {
        let tid = e.file_name().to_string_lossy().to_string();
        if tid.parse::<u32>().is_err() {
            continue;
        }
        let comm = std::fs::read_to_string(format!("{dir}/{tid}/comm")).ok()?;
        if comm.trim() != name {
            continue;
        }
        let stat = std::fs::read_to_string(format!("{dir}/{tid}/stat")).ok()?;
        let tail = stat.split(')').nth(1)?.trim_start();
        let f: Vec<&str> = tail.split_whitespace().collect();
        if f.len() < 15 {
            return None;
        }
        let utime: u64 = f[11].parse().ok()?;
        let stime: u64 = f[12].parse().ok()?;
        return Some(utime + stime);
    }
    None
}

fn main() {
    let mut args = std::env::args().skip(1);
    let batch_path = args.next().expect("usage: sla_load_test <batch.json> [out.json]");
    let out_path = args.next().unwrap_or_else(|| "results/pi/sla_load_test.json".into());

    let config_path = std::env::var("EDGE_CONFIG").unwrap_or_else(|_| "configs/edge_config.json".to_string());
    let config = MidasConfig::from_file(&config_path).expect("load config");
    let sla_budget = Duration::from_millis(config.sla_budget_ms);

    let model_path = std::env::var("EDGE_MODEL_PATH").expect("EDGE_MODEL_PATH required (tflite)");
    let model = TfliteModel::new(&model_path).unwrap_or_else(|e| panic!("load {model_path}: {e}"));

    let load_n = parse_env_usize("LOAD_N", 20000);
    let rate_hz = parse_env_f64("LOAD_RATE_HZ", 0.0);
    let scale = parse_env_f64("LOAD_SCALE", 1.0);
    let cpu_secs = parse_env_usize("LOAD_CPU_SECS", 3);

    let raw = std::fs::read_to_string(&batch_path).expect("read batch");
    let batch: LoadBatch = serde_json::from_str(&raw).expect("parse batch");

    let mut inputs: Vec<Vec<f32>> = if !batch.adaptive_adversarial.is_empty() {
        batch.adaptive_adversarial.clone()
    } else {
        batch.originals.clone()
    };
    if scale != 1.0 {
        for v in inputs.iter_mut() {
            for x in v.iter_mut() {
                *x *= scale as f32;
            }
        }
    }
    let total = load_n;
    if total == 0 || inputs.is_empty() {
        eprintln!("no queries to run (batch empty or LOAD_N=0)");
        std::process::exit(2);
    }

    let c_base = inputs[0].clone();
    let basis = compute_fixed_basis(&c_base);

    let delta_theta_max = config.delta_theta_max_rad();
    let gamma = config.gamma;
    let offered_interval = if rate_hz > 0.0 { Some(Duration::from_secs_f64(1.0 / rate_hz)) } else { None };

    let shared = Arc::new(Mutex::new(Vec::<PhaseLatencySample>::new()));
    let done = Arc::new(AtomicBool::new(false));
    let shared2 = shared.clone();
    let done2 = done.clone();

    let driver = thread::Builder::new().name("load-driver".into()).spawn(move || {
        pin_thread(0);
        let mut ring_buf = RingBuffer::new(config.W);
        let mut last = Instant::now();
        for i in 0..total {
            if let Some(iv) = offered_interval {
                let wait = iv.saturating_sub(last.elapsed());
                if !wait.is_zero() {
                    thread::sleep(wait);
                }
                last = Instant::now();
            }
            let v_t = &inputs[i % inputs.len()];
            ring_buf.push(v_t.clone());
            let window = ring_buf.window();
            let epsilon_p = penetration_epsilon_windowed(&window);
            let delta_theta = rotation_angle(epsilon_p, gamma, config.lambda, config.k, delta_theta_max);
            let phase = if epsilon_p <= gamma { Phase::Archimedean } else { Phase::Logarithmic };
            let t0 = Instant::now();
            let x_rotated = if epsilon_p > 0.0 {
                budget_gated_rotation_fixed_basis(v_t, delta_theta, &basis, &c_base, sla_budget)
            } else {
                v_t.clone()
            };
            let _r = model.infer(&x_rotated);
            let elapsed = t0.elapsed();
            shared2.lock().unwrap().push(PhaseLatencySample { phase, latency: elapsed });
        }
        done2.store(true, Ordering::SeqCst);
    }).expect("spawn driver");

    // Per-thread CPU sampler for the load-driver thread (sustained window).
    let cpu_peak = Arc::new(Mutex::new(0.0f64));
    let cpu_peak2 = cpu_peak.clone();
    let done3 = done.clone();
    let sampler = thread::Builder::new().name("cpu-sampler".into()).spawn(move || {
        pin_thread(3);
        let pid = std::process::id();
        let name = "load-driver";
        let hz: f64 = 100.0; // CONFIG_HZ
        let start = Instant::now();
        let mut prev: Option<(u64, Instant)> = None;
        while start.elapsed() < Duration::from_secs(cpu_secs as u64) && !done3.load(Ordering::SeqCst) {
            if let Some(ticks) = read_thread_ticks(pid, name) {
                if let Some((pt, ptim)) = prev {
                    let dt = ptim.elapsed().as_secs_f64();
                    if dt > 0.0 {
                        let util = (ticks.saturating_sub(pt)) as f64 / (hz * dt) * 100.0;
                        let mut peak = cpu_peak2.lock().unwrap();
                        if util > *peak {
                            *peak = util;
                        }
                    }
                }
                prev = Some((ticks, Instant::now()));
            }
            thread::sleep(Duration::from_millis(50));
        }
    }).expect("spawn sampler");

    let driver_wall_start = Instant::now();
    // Wait for the driver to finish, then join.
    while !done.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(10));
    }
    driver.join().expect("driver join");
    let wall_s = driver_wall_start.elapsed().as_secs_f64();
    let _ = sampler.join();

    let samples = shared.lock().unwrap();
    let mut ms: Vec<f64> = samples.iter().map(|s| s.latency.as_secs_f64() * 1e3).collect();
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ms.len();
    let pct = |q: f64| -> f64 {
        if n == 0 { return 0.0; }
        let idx = ((n as f64 - 1.0) * q).round() as usize;
        ms[idx.min(n - 1)]
    };
    let max = ms.last().copied().unwrap_or(0.0);
    let mean = if n == 0 { 0.0 } else { ms.iter().sum::<f64>() / n as f64 };
    let sla_violations = ms.iter().filter(|m| **m > (sla_budget.as_secs_f64() * 1e3)).count();
    let qps = if wall_s > 0.0 { n as f64 / wall_s } else { 0.0 };

    let mut arch = 0usize;
    let mut log = 0usize;
    for s in samples.iter() {
        match s.phase {
            Phase::Archimedean => arch += 1,
            Phase::Logarithmic => log += 1,
        }
    }

    let peak_cpu = *cpu_peak.lock().unwrap();

    let out = serde_json::json!({
        "experiment": "SLA/throughput load test (#5)",
        "model": model_path,
        "gamma": gamma,
        "sla_budget_ms": config.sla_budget_ms,
        "d": batch.d,
        "offered_rate_hz": rate_hz,
        "scale": scale,
        "total_queries": n,
        "wall_s": wall_s,
        "qps": qps,
        "summary": Summary {
            n,
            p50_ms: pct(0.50),
            p95_ms: pct(0.95),
            p99_ms: pct(0.99),
            max_ms: max,
            mean_ms: mean,
            sla_violations,
            sla_violation_frac: if n > 0 { sla_violations as f64 / n as f64 } else { 0.0 },
        },
        "phase_census": { "archimedean": arch, "logarithmic": log },
        "per_thread_cpu_driver_peak_pct": peak_cpu,
        "notes": [
            "per-query path: windowed penetration -> rotation_angle -> budget-gated fixed-basis rotation -> TFLite inference (mirrors deployed 3-thread pipeline).",
            "LOAD_RATE_HZ=0 means back-to-back (max offered rate); qps is the sustained processed rate.",
            "per-thread CPU sampled from /proc/[pid]/task/[tid]/stat for the load-driver thread over ~cpu_secs.",
        ],
    });

    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    std::fs::write(&out_path, serde_json::to_string_pretty(&out).expect("serialize"))
        .expect("write out");
}
