//! Hand-written FFI bindings to the TensorFlow Lite C API (`libtensorflowlite_c.so`).
//!
//! Only the surface needed for single-input / scalar-output classifiers is bound.
//! Symbol names and semantics follow `tensorflow/lite/c/c_api.h` (TF 2.17).

use std::os::raw::{c_char, c_int};

/// Opaque handle types (never dereferenced from Rust).
#[repr(C)]
pub struct TfLiteModel {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TfLiteInterpreterOptions {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TfLiteInterpreter {
    _private: [u8; 0],
}
#[repr(C)]
pub struct TfLiteTensor {
    _private: [u8; 0],
}

/// `TfLiteStatus` — `kTfLiteOk = 0`, `kTfLiteError = 1`.
pub type TfLiteStatus = c_int;
pub const TFLITE_OK: TfLiteStatus = 0;

/// `TfLiteType` values (subset).
pub const K_TFLITE_FLOAT32: c_int = 1;
pub const K_TFLITE_INT8: c_int = 9;

/// `TfLiteQuantizationParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TfLiteQuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
}

extern "C" {
    // Model
    pub fn TfLiteModelCreateFromFile(model_path: *const c_char) -> *mut TfLiteModel;
    pub fn TfLiteModelDelete(model: *mut TfLiteModel);

    // Interpreter options
    pub fn TfLiteInterpreterOptionsCreate() -> *mut TfLiteInterpreterOptions;
    pub fn TfLiteInterpreterOptionsSetNumThreads(options: *mut TfLiteInterpreterOptions, num_threads: i32);
    pub fn TfLiteInterpreterOptionsDelete(options: *mut TfLiteInterpreterOptions);

    // Interpreter
    pub fn TfLiteInterpreterCreate(
        model: *const TfLiteModel,
        options: *const TfLiteInterpreterOptions,
    ) -> *mut TfLiteInterpreter;
    pub fn TfLiteInterpreterDelete(interpreter: *mut TfLiteInterpreter);
    pub fn TfLiteInterpreterAllocateTensors(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;
    pub fn TfLiteInterpreterInvoke(interpreter: *mut TfLiteInterpreter) -> TfLiteStatus;

    // Tensors
    pub fn TfLiteInterpreterGetInputTensorCount(interpreter: *const TfLiteInterpreter) -> i32;
    pub fn TfLiteInterpreterGetInputTensor(
        interpreter: *mut TfLiteInterpreter,
        index: i32,
    ) -> *mut TfLiteTensor;
    pub fn TfLiteInterpreterGetOutputTensor(
        interpreter: *const TfLiteInterpreter,
        index: i32,
    ) -> *const TfLiteTensor;
    pub fn TfLiteTensorType(tensor: *const TfLiteTensor) -> c_int;
    pub fn TfLiteTensorByteSize(tensor: *const TfLiteTensor) -> usize;
    pub fn TfLiteTensorNumDims(tensor: *const TfLiteTensor) -> i32;
    pub fn TfLiteTensorDim(tensor: *const TfLiteTensor, dim_index: i32) -> i32;
    pub fn TfLiteTensorCopyFromBuffer(
        tensor: *mut TfLiteTensor,
        input_data: *const std::ffi::c_void,
        input_data_size: usize,
    ) -> TfLiteStatus;
    pub fn TfLiteTensorCopyToBuffer(
        tensor: *const TfLiteTensor,
        output_data: *mut std::ffi::c_void,
        output_data_size: usize,
    ) -> TfLiteStatus;
    pub fn TfLiteTensorQuantizationParams(tensor: *const TfLiteTensor) -> TfLiteQuantizationParams;
}
