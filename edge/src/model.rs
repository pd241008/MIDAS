use std::time::Duration;

#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub label: u32,
    pub confidence: f32,
    pub latency: Duration,
}

pub trait InferenceModel: Send {
    fn infer(&self, x: &[f32]) -> InferenceResult;
}

pub struct MockModel;

impl InferenceModel for MockModel {
    fn infer(&self, x: &[f32]) -> InferenceResult {
        let start = std::time::Instant::now();
        let sum: f32 = x.iter().sum();
        let label = (sum.abs() as u32) % 10;
        std::thread::sleep(Duration::from_micros(50));
        InferenceResult {
            label,
            confidence: 0.95,
            latency: start.elapsed(),
        }
    }
}

pub struct TfliteModel;

impl TfliteModel {
    pub fn new(_model_path: &str) -> Result<Self, String> {
        // TODO: real INT8 model
        // FFI handle to loaded TFLite C++ interpreter via libtensorflowlite_c.so
        Err("TfliteModel not yet implemented — build with --features tflite and provide libtensorflowlite_c.so at link time".into())
    }
}

impl InferenceModel for TfliteModel {
    fn infer(&self, _x: &[f32]) -> InferenceResult {
        unimplemented!("TfliteModel::infer — implement FFI call to TFLite C++ runtime")
    }
}
