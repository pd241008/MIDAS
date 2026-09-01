use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=TFLITE_LIB_DIR");

    // TFLite C API linking, enabled only under the `tflite` feature.
    // Vendored prebuilt libtensorflowlite_c.so v2.17.1 (tphakala/tflite_c,
    // built with XNNPACK) lives in vendor/tflite/<target-arch>/.
    if std::env::var("CARGO_FEATURE_TFLITE").is_ok() {
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
        let vendored = manifest.join("vendor").join("tflite").join(&arch);

        let dir = std::env::var("TFLITE_LIB_DIR")
            .map(PathBuf::from)
            .unwrap_or(vendored);

        if !dir.join("libtensorflowlite_c.so").exists()
            && !dir.join("libtensorflowlite_c.so.2.17.1").exists()
        {
            panic!(
                "feature `tflite` enabled but libtensorflowlite_c.so not found in {} \
                 (set TFLITE_LIB_DIR to override)",
                dir.display()
            );
        }

        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-lib=dylib=tensorflowlite_c");

        // Runtime search: absolute dev path first, then a deploy-friendly
        // path relative to the binary ($ORIGIN/../vendor/tflite/<arch> for
        // target/{debug,release}/... layouts).
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        println!(
            "cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../../vendor/tflite/{arch}"
        );
    }
}
