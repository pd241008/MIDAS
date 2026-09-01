//! Phase-split latency measurement (#6).
//!
//! Drives the real 3-thread pipeline (sensor -> defense -> rotation) with the
//! PhaseMixSensor, which emits alternating benign and adversarial segments so
//! the defense produces both Archimedean and Logarithmic phase windows. Reports
//! the per-phase latency histogram (p50/p95/p99/max) for the paper's
//! Benign/Archimedean vs Adversarial/Logarithmic latency rows.
//!
//! Usage:
//!   EDGE_CONFIG=configs/edge_config.json cargo run --release --bin phase_latency
//!   PHASE_BENIGN=200 PHASE_ADV=200 PHASE_SEGMENTS=20 cargo run --release --bin phase_latency

use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use edge_core::config::MidasConfig;
use edge_core::ring_buffer::RingBuffer;

use edge::{channels, defense_thread, metrics::PhaseLatencySummary, rotation_thread};
use edge::model::{InferenceModel, MockModel};
#[cfg(feature = "tflite")]
use edge::model::TfliteModel;
use edge::sensor_thread::{PhaseMixSensor, sensor_thread};

fn pin_thread(core_id: usize) {
    if let Some(ref cores) = core_affinity::get_core_ids() {
        if let Some(core) = cores.get(core_id) {
            core_affinity::set_for_current(*core);
        }
    }
}

fn parse_env_var(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn main() {
    let config_path = std::env::var("EDGE_CONFIG")
        .unwrap_or_else(|_| "configs/edge_config.json".to_string());
    let config = MidasConfig::from_file(&config_path).expect("failed to load config");

    let benign_len = parse_env_var("PHASE_BENIGN", 200);
    let adv_len = parse_env_var("PHASE_ADV", 200);
    let segments = parse_env_var("PHASE_SEGMENTS", 20);

    println!(
        "phase_latency: d={} gamma={} W={} benign={} adv={} segments={}",
        config.D, config.gamma, config.W, benign_len, adv_len, segments
    );

    let (sensor_tx, sensor_rx, defense_tx, defense_rx) = channels::build_channels(&config);
    let ring_buffer = Arc::new(RwLock::new(RingBuffer::new(config.W)));
    let model: Box<dyn InferenceModel> = if std::env::var("EDGE_MODEL_PATH").is_ok() {
        #[cfg(feature = "tflite")]
        {
            let path = std::env::var("EDGE_MODEL_PATH").unwrap();
            match TfliteModel::new(&path) {
                Ok(m) => Box::new(m),
                Err(e) => {
                    eprintln!("failed to load TFLite model {path}: {e} — using MockModel");
                    Box::new(MockModel)
                }
            }
        }
        #[cfg(not(feature = "tflite"))]
        {
            eprintln!("EDGE_MODEL_PATH set but built without --features tflite — using MockModel");
            Box::new(MockModel)
        }
    } else {
        Box::new(MockModel)
    };
    let latency_samples = Arc::new(Mutex::new(Vec::<edge::metrics::PhaseLatencySample>::new()));

    let config_for_defense = config.clone();
    let config_for_rotation = config.clone();
    let rb_for_defense = ring_buffer.clone();
    let lat_for_rotation = latency_samples.clone();

    let sensor = thread::Builder::new()
        .name("sensor".into())
        .spawn(move || {
            pin_thread(0);
            let source = PhaseMixSensor::new(config.D, benign_len, adv_len, segments);
            sensor_thread(source, sensor_tx);
        })
        .unwrap();

    let defense = thread::Builder::new()
        .name("defense".into())
        .spawn(move || {
            pin_thread(1);
            defense_thread::defense_thread(sensor_rx, defense_tx, rb_for_defense, &config_for_defense);
        })
        .unwrap();

    let rotation = thread::Builder::new()
        .name("rotation".into())
        .spawn(move || {
            pin_thread(2);
            rotation_thread::rotation_thread(defense_rx, model.as_ref(), &config_for_rotation, &*lat_for_rotation, None);
        })
        .unwrap();

    sensor.join().unwrap();
    defense.join().unwrap();
    rotation.join().unwrap();

    let samples = latency_samples.lock().unwrap();
    let split = PhaseLatencySummary::from_samples(&samples);
    println!("{}", serde_json::to_string_pretty(&split).unwrap());
}
