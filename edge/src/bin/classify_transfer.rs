//! Transfer verification (#7): classify surrogate-generated adversarial batches
//! with the real FFI-backed TFLite classifier and report per-group ASR (flip rate
//! vs the surrogate natural label).
//!
//! Usage:
//!   caffe build --bin classify_transfer --features tflite
//!   cargo run --release --bin classify_transfer --features tflite -- \
//!       models/classifier_float32.tflite results/pi/transfer_batch.json results/pi/transfer_ffi.json

use edge::model::TfliteModel;
use serde::Deserialize;

#[derive(Deserialize)]
struct Batch {
    d: usize,
    #[serde(default)]
    epss_note: String,
    labels: Vec<f32>,
    #[serde(default)]
    originals: Vec<Vec<f32>>,
    naive_adversarial: Vec<Vec<f32>>,
    adaptive_adversarial: Vec<Vec<f32>>,
}

#[derive(serde::Serialize)]
struct Out {
    model: String,
    n: usize,
    naive_asr: f64,
    adaptive_asr: f64,
    naive_flips: usize,
    adaptive_flips: usize,
    reference_asr_originals: f64,
    label_balance: Vec<f64>,
}

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().expect("usage: classify_transfer <model> <batch.json> [out.json]");
    let batch_path = args.next().expect("missing batch json");
    let out_path = args.next().unwrap_or_else(|| "results/pi/transfer_ffi.json".into());

    let raw = std::fs::read_to_string(&batch_path).expect("read batch");
    let b: Batch = serde_json::from_str(&raw).expect("parse batch");

    let model = TfliteModel::new(&model_path).unwrap_or_else(|e| panic!("load {model_path}: {e}"));

    let n = b.labels.len();
    if b.naive_adversarial.len() != n || b.adaptive_adversarial.len() != n {
        panic!("batch length mismatch (labels {n}, naive {}, adaptive {})",
               b.naive_adversarial.len(), b.adaptive_adversarial.len());
    }

    let predict = |x: &[f32]| -> f32 { model.predict(x).unwrap_or_else(|e| panic!("predict: {e}")) };

    let mut naive_flips = 0usize;
    let mut adaptive_flips = 0usize;
    let mut orig_flips = 0usize;
    let mut label_balance = vec![0f64; 2];
    for i in 0..n {
        let y = b.labels[i];
        label_balance[y as usize] += 1.0;
        let pn = predict(&b.naive_adversarial[i]);
        let pa = predict(&b.adaptive_adversarial[i]);
        if (pn >= 0.5) != (y >= 0.5) { naive_flips += 1; }
        if (pa >= 0.5) != (y >= 0.5) { adaptive_flips += 1; }
        if !b.originals.is_empty() {
            let po = predict(&b.originals[i]);
            if (po >= 0.5) != (y >= 0.5) { orig_flips += 1; }
        }
    }

    let out = Out {
        model: model_path.clone(),
        n,
        naive_asr: naive_flips as f64 / n as f64,
        adaptive_asr: adaptive_flips as f64 / n as f64,
        naive_flips,
        adaptive_flips,
        reference_asr_originals: if b.originals.is_empty() { f64::NAN } else { orig_flips as f64 / n as f64 },
        label_balance,
    };
    println!("naive ASR (real classifier)   = {:.3}", out.naive_asr);
    println!("adaptive ASR (real classifier)= {:.3}", out.adaptive_asr);
    println!("original flip (sanity)        = {:.3}", out.reference_asr_originals);

    std::fs::write(&out_path, serde_json::to_string_pretty(&out).expect("serialize out"))
        .expect("write out");
}
