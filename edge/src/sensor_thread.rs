use crossbeam_channel::Sender;
use log::info;

pub trait SensorSource: Send + 'static {
    fn read(&mut self) -> Option<Vec<f32>>;
}

pub struct MockSensor {
    dim: usize,
    count: usize,
    max_count: usize,
}

impl MockSensor {
    pub fn new(dim: usize, max_count: usize) -> Self {
        Self {
            dim,
            count: 0,
            max_count,
        }
    }
}

impl SensorSource for MockSensor {
    fn read(&mut self) -> Option<Vec<f32>> {
        if self.count >= self.max_count {
            return None;
        }
        self.count += 1;
        let t = self.count as f32;
        Some(
            (0..self.dim)
                .map(|i| {
                    let base = (t * 0.1 + i as f32).sin();
                    base * (1.0 + t * 0.02).min(10.0)
                })
                .collect(),
        )
    }
}

pub fn sensor_thread(mut source: impl SensorSource, tx: Sender<Vec<f32>>) {
    info!("sensor thread started on core 0");
    while let Some(vec) = source.read() {
        if tx.send(vec).is_err() {
            info!("sensor channel closed, shutting down");
            break;
        }
    }
    info!("sensor thread finished");
}

/// Emits traffic split into benign and adversarial segments so the pipeline
/// produces both Archimedean and Logarithmic phase windows for phase-split
/// latency measurement.
///
/// - benign_len vectors of small magnitude (low alignment -> low epsilon_p ->
///   Archimedean when epsilon_p <= gamma).
/// - adv_len vectors of large, aligned magnitude (high epsilon_p -> Logarithmic
///   when epsilon_p > gamma).
pub struct PhaseMixSensor {
    dim: usize,
    benign_len: usize,
    adv_len: usize,
    segments: usize,
    seg: usize,
    idx: usize,
}

impl PhaseMixSensor {
    pub fn new(dim: usize, benign_len: usize, adv_len: usize, segments: usize) -> Self {
        Self {
            dim,
            benign_len,
            adv_len,
            segments,
            seg: 0,
            idx: 0,
        }
    }
}

impl SensorSource for PhaseMixSensor {
    fn read(&mut self) -> Option<Vec<f32>> {
        if self.seg >= self.segments {
            return None;
        }
        let len = if self.seg % 2 == 0 { self.benign_len } else { self.adv_len };
        if self.idx >= len {
            self.seg += 1;
            self.idx = 0;
            return self.read();
        }
        self.idx += 1;
        let is_adv = self.seg % 2 == 1;
        let phase_idx = self.idx as f32;
        Some(
            (0..self.dim)
                .map(|i| {
                    let mut v = (phase_idx * 0.1 + i as f32 * 0.7).sin();
                    if is_adv {
                        v *= 10.0;
                    } else {
                        v *= 0.05;
                    }
                    v
                })
                .collect(),
        )
    }
}
