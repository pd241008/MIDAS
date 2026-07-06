use std::sync::{Arc, RwLock};

use crossbeam_channel::{Receiver, Sender};
use log::info;
use edge_core::ring_buffer::RingBuffer;
use edge_core::trajectory::penetration_epsilon_windowed;
use edge_core::rotation::rotation_angle;
use edge_core::config::MidasConfig;

use crate::channels::{DefenseCommand, Phase};

pub fn defense_thread(
    rx: Receiver<Vec<f32>>,
    tx: Sender<DefenseCommand>,
    ring_buffer: Arc<RwLock<RingBuffer>>,
    config: &MidasConfig,
) {
    info!("defense thread started on core 1");

    let delta_theta_max_rad = config.delta_theta_max_rad();

    loop {
        let v_t = match rx.recv() {
            Ok(v) => v,
            Err(_) => {
                info!("defense channel closed, shutting down");
                break;
            }
        };

        let epsilon_p = {
            let mut rb = ring_buffer.write().unwrap();
            rb.push(v_t.clone());

            // Use the windowed penetration epsilon (window averaged, not single-step)
            // This accumulates sustained aligned magnitude and resists the
            // convergent-attacker failure mode.
            let window = rb.window();
            penetration_epsilon_windowed(&window)
        };

        let delta_theta = rotation_angle(
            epsilon_p,
            config.gamma,
            config.lambda,
            config.k,
            delta_theta_max_rad,
        );

        let phase = if epsilon_p <= config.gamma {
            Phase::Archimedean
        } else {
            info!(
                "phase transition: Logarithmic (epsilon_p={:.6}, gamma={})",
                epsilon_p, config.gamma
            );
            Phase::Logarithmic
        };

        let cmd = DefenseCommand {
            v_t,
            delta_theta,
            epsilon_p,
            phase,
        };

        if tx.send(cmd).is_err() {
            info!("defense output channel closed, shutting down");
            break;
        }
    }

    info!("defense thread finished");
}
