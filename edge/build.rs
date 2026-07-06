fn main() {
    // TFLite C++ FFI linking setup (behind `tflite` feature flag).
    //
    // When the `tflite` feature is enabled, this build.rs links against
    // `libtensorflowlite_c.so` (the TensorFlow Lite C API library).
    //
    // On Raspberry Pi 5 / aarch64:
    //   - Build TFLite C++ library via CMake/Bazel targeting aarch64, or
    //   - Vendor prebuilt `.so` from https://github.com/nicktajzs/TensorFlowLiteC-prebuilt
    //
    // The C API header (`tensorflow/lite/c/c_api.h`) is the stable FFI boundary.
    //
    // TODO: uncomment when real TFLite build is set up:
    //
    // #[cfg(feature = "tflite")]
    // {
    //     println!("cargo:rustc-link-lib=tensorflowlite_c");
    //     println!("cargo:rustc-link-search=/usr/local/lib");
    //     println!("cargo:rerun-if-changed=build.rs");
    // }
    //
    // For now, this is a no-op build.rs so the scaffolding compiles without
    // requiring the TFLite C++ runtime to be installed.
}
