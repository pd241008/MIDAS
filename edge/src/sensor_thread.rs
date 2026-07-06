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
