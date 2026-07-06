use std::sync::Mutex;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use log::{info, warn};
use edge_core::config::MidasConfig;
use edge_core::rotation::budget_gated_rotation;

use crate::channels::DefenseCommand;
use crate::model::InferenceModel;

pub fn rotation_thread(
    rx: Receiver<DefenseCommand>,
    model: &dyn InferenceModel,
    config: &MidasConfig,
    hist: &Mutex<Vec<Duration>>,
) {
    info!("rotation thread started on core 2");

    let sla_budget = Duration::from_millis(config.sla_budget_ms);
    let c_base = load_manifold(&config.manifold_path);

    loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => {
                info!("rotation channel closed, shutting down");
                break;
            }
        };

        let start = Instant::now();

        let x_rotated = if cmd.epsilon_p > 0.0 {
            budget_gated_rotation(&cmd.v_t, cmd.delta_theta, &c_base, sla_budget)
        } else {
            cmd.v_t.clone()
        };

        let result = model.infer(&x_rotated);

        if cmd.phase == crate::channels::Phase::Logarithmic {
            info!(
                "inference: label={}, conf={:.3}, latency={:?}, phase=Logarithmic",
                result.label, result.confidence, result.latency
            );
        }

        let elapsed = start.elapsed();
        hist.lock().unwrap().push(elapsed);
    }

    info!("rotation thread finished");
}

fn load_manifold(path: &str) -> Vec<f32> {
    // TODO: load actual manifold centroid from path
    warn!("using synthetic manifold centroid (zeros) — replace with real manifold file at {path}");
    vec![0.0; 10]
}
