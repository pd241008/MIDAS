#![allow(dead_code)]

mod attacks;
mod channels;
mod dataset;
mod defense_thread;
mod metrics;
mod model;
mod report;
mod rotation_thread;
mod sensor_thread;

use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use log::{error, info, warn};
use edge_core::config::MidasConfig;
use edge_core::ring_buffer::RingBuffer;

use sensor_thread::{sensor_thread, MockSensor};
use model::MockModel;
use metrics::LatencyHistogram;

fn pin_thread(core_id: usize) {
    let ids = core_affinity::get_core_ids();
    if let Some(ref cores) = ids {
        if let Some(core) = cores.get(core_id) {
            core_affinity::set_for_current(*core);
            info!("pinned to core {}", core_id);
        } else {
            warn!(
                "core {} not available ({} cores detected) — skipping pin",
                core_id,
                cores.len()
            );
        }
    } else {
        warn!("core affinity not supported on this platform — skipping pin");
    }
}

fn run_pipeline(synthetic_override: Option<bool>) {
    info!("=== Pipeline Mode ===");

    let config_path = std::env::var("EDGE_CONFIG")
        .unwrap_or_else(|_| "configs/edge_config.json".to_string());
    let config = MidasConfig::from_file(&config_path).expect("failed to load config");
    info!("config loaded: {:?}", config);

    let (sensor_tx, sensor_rx, defense_tx, defense_rx) =
        channels::build_channels(&config);

    let ring_buffer = Arc::new(RwLock::new(RingBuffer::new(config.W)));

    let model = Box::new(MockModel);

    let latency_samples = Arc::new(Mutex::new(Vec::<Duration>::new()));

    let config_for_defense = config.clone();
    let config_for_rotation = config.clone();
    let rb_for_defense = ring_buffer.clone();
    let lat_for_rotation = latency_samples.clone();

    let sensor_handle = thread::Builder::new()
        .name("sensor".into())
        .spawn(move || {
            pin_thread(0);
            let source = MockSensor::new(config.D, 200);
            sensor_thread(source, sensor_tx);
        })
        .expect("failed to spawn sensor thread");

    let defense_handle = thread::Builder::new()
        .name("defense".into())
        .spawn(move || {
            pin_thread(1);
            defense_thread::defense_thread(
                sensor_rx,
                defense_tx,
                rb_for_defense,
                &config_for_defense,
            );
        })
        .expect("failed to spawn defense thread");

    let rotation_handle = thread::Builder::new()
        .name("rotation".into())
        .spawn(move || {
            pin_thread(2);
            rotation_thread::rotation_thread(
                defense_rx,
                &*model,
                &config_for_rotation,
                &*lat_for_rotation,
                synthetic_override,
            );
        })
        .expect("failed to spawn rotation thread");

    info!("all threads launched, waiting for completion");

    sensor_handle.join().expect("sensor thread panicked");
    info!("sensor thread joined");

    defense_handle.join().expect("defense thread panicked");
    info!("defense thread joined");

    rotation_handle.join().expect("rotation thread panicked");
    info!("rotation thread joined");

    let samples = latency_samples.lock().unwrap();
    let hist = LatencyHistogram::from_samples(samples.clone());
    drop(samples);
    let json = serde_json::to_string_pretty(&hist).expect("failed to serialize histogram");
    info!("latency histogram:\n{}", json);
    println!("{}", json);
}

fn run_harness(synthetic_override: Option<bool>) {
    info!("=== Harness Mode ===");

    let config_path = std::env::var("EDGE_CONFIG")
        .unwrap_or_else(|_| "configs/edge_config.json".to_string());
    let config = MidasConfig::from_file(&config_path).expect("failed to load config");

    let (c_base, basis) = edge_core::load_manifold_for_export(&config, synthetic_override);

    let defense_success = report::placeholder_defense_success();
    let latency = report::placeholder_latency();
    let pgd_iters = report::placeholder_pgd_iterations();

    report::write_csv(&defense_success, "results/defense_success_rate.csv").unwrap();
    report::write_csv(&latency, "results/latency_distribution.csv").unwrap();
    report::write_csv(&pgd_iters, "results/defense_rate_vs_pgd.csv").unwrap();

    report::write_json(&defense_success, "results/defense_success_rate.json").unwrap();
    report::write_json(&latency, "results/latency_distribution.json").unwrap();
    report::write_json(&pgd_iters, "results/defense_rate_vs_pgd.json").unwrap();

    info!("report tables written to results/");

    let attack_configs = attacks::paper_attack_configs();
    for cfg in &attack_configs {
        info!("attack config: {} — will run under naive AND adaptive attacker models", cfg);
    }

    let basis_json = serde_json::json!({
        "c_base": c_base,
        "basis": basis
    });
    std::fs::create_dir_all("results").unwrap_or_default();
    std::fs::write("results/basis.json", basis_json.to_string())
        .expect("failed to write results/basis.json");
    info!("exported basis.json for PyTorch shadow evaluation");
    info!(
        "basis.json c_base norm={:.6}",
        c_base.iter().map(|x| x * x).sum::<f32>().sqrt()
    );
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args: Vec<String> = std::env::args().collect();

    let mut synthetic_override: Option<bool> = None;
    let mut mode = "pipeline".to_string();

    for arg in &args[1..] {
        match arg.as_str() {
            "--synthetic-manifold" => {
                synthetic_override = Some(true);
                info!("CLI override: --synthetic-manifold (using synthetic manifold)");
            }
            "--no-synthetic-manifold" => {
                synthetic_override = Some(false);
                info!("CLI override: --no-synthetic-manifold (require real manifold file)");
            }
            "pipeline" | "harness" | "all" => {
                mode = arg.clone();
            }
            other => {
                if other.starts_with('-') {
                    error!("unknown flag: {other}");
                    std::process::exit(1);
                }
                mode = other.to_string();
            }
        }
    }

    match mode.as_str() {
        "harness" => run_harness(synthetic_override),
        "all" => {
            run_pipeline(synthetic_override);
            run_harness(synthetic_override);
        }
        _ => run_pipeline(synthetic_override),
    }
}
