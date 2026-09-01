//! EDGE library crate: pipeline threads, defense math glue, models, attacks.
//!
//! The `edge` binary (src/main.rs) drives these; the `tflite_validate` bin
//! uses the model layer for FFI validation.

pub mod attacks;
pub mod channels;
pub mod dataset;
pub mod defense_thread;
pub mod metrics;
pub mod model;
pub mod report;
pub mod rotation_thread;
pub mod sensor_thread;

#[cfg(feature = "tflite")]
pub mod tflite_ffi;
