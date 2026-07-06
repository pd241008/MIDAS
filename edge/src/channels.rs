use edge_core::config::MidasConfig;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct DefenseCommand {
    pub v_t: Vec<f32>,
    pub delta_theta: f32,
    pub epsilon_p: f32,
    pub phase: Phase,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum Phase {
    Archimedean,
    Logarithmic,
}

pub fn build_channels(
    config: &MidasConfig,
) -> (
    crossbeam_channel::Sender<Vec<f32>>,
    crossbeam_channel::Receiver<Vec<f32>>,
    crossbeam_channel::Sender<DefenseCommand>,
    crossbeam_channel::Receiver<DefenseCommand>,
) {
    let (sensor_tx, sensor_rx) =
        crossbeam_channel::bounded(config.channel_capacity);
    let (defense_tx, defense_rx) =
        crossbeam_channel::bounded(config.channel_capacity);

    (sensor_tx, sensor_rx, defense_tx, defense_rx)
}
