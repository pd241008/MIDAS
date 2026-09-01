#[cfg(feature = "tflite")]
use std::ffi::CString;
use std::time::{Duration, Instant};

#[cfg(feature = "tflite")]
use crate::tflite_ffi as ffi;

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

/// RAII guard for a loaded TFLite model + interpreter pair.
///
/// The interpreter is not thread-safe, but the three-thread pipeline calls
/// `infer` from the single rotation thread only; `unsafe impl Send` reflects
/// that ownership discipline (the raw pointers never leave this struct).
#[cfg(feature = "tflite")]
pub struct TfliteModel {
    _model: *mut ffi::TfLiteModel,
    _options: *mut ffi::TfLiteInterpreterOptions,
    interpreter: *mut ffi::TfLiteInterpreter,
    input_tensor: *mut ffi::TfLiteTensor,
    output_tensor: *const ffi::TfLiteTensor,
    input_len: usize,
    input_scale: f32,
    input_zp: i32,
    output_scale: f32,
    output_zp: i32,
    int8_input: bool,
}

// SAFETY: all raw pointers are owned exclusively by this struct and only used
// from the rotation thread (see trait contract on InferenceModel).
#[cfg(feature = "tflite")]
unsafe impl Send for TfliteModel {}

#[cfg(feature = "tflite")]
impl TfliteModel {
    /// Load a `.tflite` model and allocate tensors.
    ///
    /// Supports both float32 and int8-quantized models; quantization params
    /// are read from the model itself at load time.
    pub fn new(model_path: &str) -> Result<Self, String> {
        let cpath = CString::new(model_path)
            .map_err(|_| format!("model path contains NUL byte: {model_path:?}"))?;

        // SAFETY: cpath outlives the call; returned handle stored in self.
        let model = unsafe { ffi::TfLiteModelCreateFromFile(cpath.as_ptr()) };
        if model.is_null() {
            return Err(format!("TfLiteModelCreateFromFile failed for {model_path:?}"));
        }

        let options = unsafe { ffi::TfLiteInterpreterOptionsCreate() };
        unsafe { ffi::TfLiteInterpreterOptionsSetNumThreads(options, 1) };

        let interpreter = unsafe { ffi::TfLiteInterpreterCreate(model, options) };
        if interpreter.is_null() {
            unsafe {
                ffi::TfLiteInterpreterOptionsDelete(options);
                ffi::TfLiteModelDelete(model);
            }
            return Err("TfLiteInterpreterCreate failed".into());
        }

        if unsafe { ffi::TfLiteInterpreterAllocateTensors(interpreter) } != ffi::TFLITE_OK {
            unsafe {
                ffi::TfLiteInterpreterDelete(interpreter);
                ffi::TfLiteInterpreterOptionsDelete(options);
                ffi::TfLiteModelDelete(model);
            }
            return Err("TfLiteInterpreterAllocateTensors failed".into());
        }

        if unsafe { ffi::TfLiteInterpreterGetInputTensorCount(interpreter) } != 1 {
            return Err("expected exactly one input tensor".into());
        }
        let input_tensor = unsafe { ffi::TfLiteInterpreterGetInputTensor(interpreter, 0) };
        let output_tensor = unsafe { ffi::TfLiteInterpreterGetOutputTensor(interpreter, 0) };

        let input_type = unsafe { ffi::TfLiteTensorType(input_tensor) };
        let int8_input = match input_type {
            t if t == ffi::K_TFLITE_FLOAT32 => false,
            t if t == ffi::K_TFLITE_INT8 => true,
            other => return Err(format!("unsupported input tensor type id {other}")),
        };

        let in_params = unsafe { ffi::TfLiteTensorQuantizationParams(input_tensor) };
        let out_params = unsafe { ffi::TfLiteTensorQuantizationParams(output_tensor) };
        let input_len = unsafe { ffi::TfLiteTensorByteSize(input_tensor) };

        Ok(TfliteModel {
            _model: model,
            _options: options,
            interpreter,
            input_tensor,
            output_tensor,
            input_len,
            input_scale: in_params.scale,
            input_zp: in_params.zero_point,
            output_scale: out_params.scale,
            output_zp: out_params.zero_point,
            int8_input,
        })
    }

    fn invoke_raw(&self, x: &[f32], out: &mut [f32]) -> Result<(), String> {
        if self.int8_input {
            let s = self.input_scale;
            let zp = self.input_zp as f32;
            let q: Vec<u8> = x
                .iter()
                .map(|&v| {
                    let qi = (v / s + zp).round();
                    qi.clamp(-128.0, 127.0) as i8 as u8
                })
                .collect();
            // SAFETY: q.len() == input_len verified in predict(); pointer valid for the call.
            unsafe {
                ffi::TfLiteTensorCopyFromBuffer(
                    self.input_tensor,
                    q.as_ptr() as *const _,
                    q.len(),
                )
            }
        } else {
            // SAFETY: x outlives the call; element size f32 matches tensor type.
            unsafe {
                ffi::TfLiteTensorCopyFromBuffer(
                    self.input_tensor,
                    x.as_ptr() as *const _,
                    x.len() * 4,
                )
            }
        }
        .check("TfLiteTensorCopyFromBuffer")?;

        // SAFETY: interpreter is exclusively borrowed behind &self per the
        // single-caller contract documented on this struct.
        unsafe { ffi::TfLiteInterpreterInvoke(self.interpreter) }.check("TfLiteInterpreterInvoke")?;

        if self.int8_input {
            let mut qo = vec![0u8; out.len()];
            // SAFETY: output buffer sized to match tensor byte size.
            unsafe {
                ffi::TfLiteTensorCopyToBuffer(
                    self.output_tensor,
                    qo.as_mut_ptr() as *mut _,
                    qo.len(),
                )
            }
            .check("TfLiteTensorCopyToBuffer")?;
            for (dst, &b) in out.iter_mut().zip(qo.iter()) {
                *dst = (b as i8 as f32 - self.output_zp as f32) * self.output_scale;
            }
        } else {
            // SAFETY: as above, float32 output path.
            unsafe {
                ffi::TfLiteTensorCopyToBuffer(
                    self.output_tensor,
                    out.as_mut_ptr() as *mut _,
                    out.len() * 4,
                )
            }
            .check("TfLiteTensorCopyToBuffer")?;
        }
        Ok(())
    }

    /// Convenience scalar-output inference returning the dequantized sigmoid
    /// probability (single-output models).
    pub fn predict(&self, x: &[f32]) -> Result<f32, String> {
        let want = if self.int8_input { self.input_len } else { self.input_len / 4 };
        if x.len() != want {
            return Err(format!("input dim mismatch: got {}, want {}", x.len(), want));
        }
        let mut out = [0f32; 1];
        self.invoke_raw(x, &mut out)?;
        Ok(out[0])
    }
}

#[cfg(feature = "tflite")]
trait StatusExt {
    fn check(self, what: &'static str) -> Result<(), String>;
}
#[cfg(feature = "tflite")]
impl StatusExt for std::os::raw::c_int {
    fn check(self, what: &'static str) -> Result<(), String> {
        if self == ffi::TFLITE_OK {
            Ok(())
        } else {
            Err(format!("{what} failed with TfLiteStatus={self}"))
        }
    }
}

#[cfg(feature = "tflite")]
impl Drop for TfliteModel {
    fn drop(&mut self) {
        // SAFETY: each handle was created in new() exactly once.
        unsafe {
            ffi::TfLiteInterpreterDelete(self.interpreter);
            ffi::TfLiteInterpreterOptionsDelete(self._options);
            ffi::TfLiteModelDelete(self._model);
        }
    }
}

#[cfg(feature = "tflite")]
impl InferenceModel for TfliteModel {
    fn infer(&self, x: &[f32]) -> InferenceResult {
        let start = Instant::now();
        match self.predict(x) {
            Ok(p) => InferenceResult {
                label: u32::from(p >= 0.5),
                confidence: p.max(1.0 - p),
                latency: start.elapsed(),
            },
            Err(_e) => InferenceResult {
                label: 0,
                confidence: 0.0,
                latency: start.elapsed(),
            },
        }
    }
}
