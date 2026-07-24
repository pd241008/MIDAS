use std::sync::Mutex;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use log::{error, info, warn};
use edge_core::config::MidasConfig;
use edge_core::manifold;
use edge_core::rotation::{budget_gated_rotation_fixed_basis, compute_fixed_basis};

use crate::channels::DefenseCommand;
use crate::model::InferenceModel;

fn load_manifold_for_rotation(
    config: &MidasConfig,
    synthetic_override: Option<bool>,
) -> (manifold::Manifold, bool) {
    let use_synthetic = synthetic_override.unwrap_or(config.use_synthetic_manifold);

    if use_synthetic {
        warn!(
            "WARNING: using synthetic manifold, not loaded from disk — \
             this is a local dev configuration, NOT a real deployment"
        );
        (manifold::generate_synthetic_manifold(config.D, 200, 42), true)
    } else {
        match manifold::load_manifold(&config.manifold_path, config.D) {
            Ok(m) => {
                info!(
                    "loaded real manifold from {}: {} samples, dim={}",
                    config.manifold_path, m.n_samples, m.dim
                );
                (m, false)
            }
            Err(e) => {
                error!("FATAL: failed to load manifold from {}: {e}", config.manifold_path);
                panic!("manifold loading failed: {e}");
            }
        }
    }
}

fn compute_basis(manifold: &manifold::Manifold, c_base: &[f32]) -> Vec<Vec<f32>> {
    if manifold.n_samples >= 2 {
        info!("computing PCA-2 fixed rotation basis from {} manifold samples", manifold.n_samples);
        match manifold::pca2(manifold) {
            Ok(b) => {
                let dot: f32 = b[0].iter().zip(&b[1]).map(|(a, c)| a * c).sum();
                info!(
                    "PCA-2 basis ready: b0 norm={:.4}, b1 norm={:.4}, dot={:.4}",
                    b[0].iter().map(|x| x * x).sum::<f32>().sqrt(),
                    b[1].iter().map(|x| x * x).sum::<f32>().sqrt(),
                    dot,
                );
                b
            }
            Err(e) => {
                warn!("PCA-2 failed ({e}), falling back to Gram-Schmidt from centroid");
                compute_fixed_basis(c_base)
            }
        }
    } else {
        info!("fewer than 2 manifold samples, using Gram-Schmidt basis from centroid");
        compute_fixed_basis(c_base)
    }
}

pub fn rotation_thread(
    rx: Receiver<DefenseCommand>,
    model: &dyn InferenceModel,
    config: &MidasConfig,
    hist: &Mutex<Vec<Duration>>,
    synthetic_override: Option<bool>,
) {
    info!("rotation thread started on core 2");

    let sla_budget = Duration::from_millis(config.sla_budget_ms);

    let (manifold, is_synthetic) = load_manifold_for_rotation(config, synthetic_override);

    info!(
        "manifold: {} samples, dim={}, centroid_norm={:.6}",
        manifold.n_samples,
        manifold.dim,
        manifold.centroid.iter().map(|x| x * x).sum::<f32>().sqrt(),
    );

    let c_base = manifold.centroid.clone();
    let basis = compute_basis(&manifold, &c_base);

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

    if is_synthetic {
        warn!(
            "rotation thread finished — NOTE: ran with synthetic manifold, \
             not real data"
        );
    } else {
        info!("rotation thread finished");
    }
}
