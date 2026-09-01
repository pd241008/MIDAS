//! FFI validation harness: feed the same vectors through the Rust TFLite C API
//! binding and compare against the PyTorch reference produced by
//! `shadow/gen_ffit_vectors.py`.
//!
//! Usage:
//!   cargo run --bin tflite_validate --features tflite -- \
//!       --model models/classifier_float32.tflite \
//!       --vectors results/ffi_vectors.json \
//!       --out results/ffi_validation.json

use edge::model::TfliteModel;

#[derive(serde::Deserialize)]
struct Vectors {
    input_dim: usize,
    inputs: Vec<Vec<f32>>,
    #[serde(default)]
    #[allow(dead_code)]
    input_names: Vec<String>,
    reference: Vec<f32>,
    source_model: String,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: tflite_validate <model> [vectors] [out]");
    let vectors_path = args.next().unwrap_or_else(|| "results/ffi_vectors.json".into());
    let out_path = args.next().unwrap_or_else(|| "results/ffi_validation.json".into());

    let raw = std::fs::read_to_string(&vectors_path).expect("read vectors");
    let v: Vectors = serde_json::from_str(&raw).expect("parse vectors");

    let model =
        TfliteModel::new(&model_path).unwrap_or_else(|e| panic!("load {model_path}: {e}"));

    let mut max_abs = 0f64;
    let mut sum_abs = 0f64;
    let mut flips = 0u32;
    let mut lat_us: Vec<f64> = Vec::with_capacity(v.inputs.len());

    for (x, &ref_p) in v.inputs.iter().zip(v.reference.iter()) {
        let p = model.predict(x).expect("predict");
        let d = (p - ref_p).abs() as f64;
        max_abs = max_abs.max(d);
        sum_abs += d;
        if (p >= 0.5) != (ref_p >= 0.5) {
            flips += 1;
        }
        // latency measured separately to keep the numeric path hot-loop clean
        let t = std::time::Instant::now();
        let _ = model.predict(x);
        lat_us.push(t.elapsed().as_nanos() as f64 / 1000.0);
    }

    let n = v.inputs.len();
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f64 {
        let i = ((p / 100.0) * (lat_us.len() - 1) as f64).round() as usize;
        lat_us[i]
    };

    let report = serde_json::json!({
        "model": model_path,
        "source_model_pt": v.source_model,
        "vectors": n,
        "input_dim": v.input_dim,
        "max_abs_diff": max_abs,
        "mean_abs_diff": sum_abs / n as f64,
        "label_flips": flips,
        "label_flip_rate": flips as f64 / n as f64,
        "latency_us_p50": pct(50.0),
        "latency_us_p99": pct(99.0),
        "latency_us_max": *lat_us.last().unwrap(),
    });

    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    std::fs::write(&out_path, serde_json::to_string_pretty(&report).unwrap()).expect("write out");
}
