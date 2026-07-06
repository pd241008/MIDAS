use std::sync::Mutex;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use log::{info, warn};
use edge_core::config::MidasConfig;
use edge_core::rotation::{budget_gated_rotation_fixed_basis, compute_fixed_basis};

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

    // Compute the fixed rotation basis ONCE at startup from the base manifold.
    // This basis is held fixed for the life of the deployment and MUST NOT
    // depend on per-query state (security-motivated: prevents attacker from
    // backpropagating through the rotation plane).
    info!(
        "computing fixed rotation basis from manifold (dim={})",
        c_base.len()
    );
    let basis = compute_fixed_basis(&c_base);
    info!(
        "fixed rotation basis ready: b0 norm={:.4}, b1 norm={:.4}, dot={:.4}",
        basis[0].iter().map(|x| x * x).sum::<f32>().sqrt(),
        basis[1].iter().map(|x| x * x).sum::<f32>().sqrt(),
        basis[0].iter().zip(&basis[1]).map(|(a, b)| a * b).sum::<f32>(),
    );

    loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => {
                info!("rotation channel closed, shutting down");
                break;
            }
        };

        let start = Instant::now();

        // Apply budget-gated rotation using the FIXED (non-attacker-coupled) basis.
        // Only delta_theta's magnitude depends on the live trajectory via epsilon_p;
        // the rotation plane itself is independent of the attacker's query.
        let x_rotated = if cmd.epsilon_p > 0.0 {
            budget_gated_rotation_fixed_basis(
                &cmd.v_t,
                cmd.delta_theta,
                &basis,
                &c_base,
                sla_budget,
            )
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
