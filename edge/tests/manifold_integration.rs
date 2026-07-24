use std::io::Cursor;
use std::time::Duration;

use edge_core::config::MidasConfig;
use edge_core::manifold::{self, pca2};
use edge_core::ring_buffer::RingBuffer;
use edge_core::rotation::{budget_gated_rotation_fixed_basis, compute_fixed_basis};

fn test_config(use_synthetic: bool) -> MidasConfig {
    MidasConfig {
        W: 10,
        D: 10,
        gamma: 0.292,
        lambda: 1.0,
        k: 2.0,
        tau: 0.75,
        delta_theta_max_deg: 45.0,
        sla_budget_ms: 10,
        channel_capacity: 4,
        model_path: "models/classifier.tflite".into(),
        manifold_path: "/nonexistent/test_manifold.npy".into(),
        use_synthetic_manifold: use_synthetic,
    }
}

#[test]
fn test_synthetic_manifold_pipeline_end_to_end() {
    let config = test_config(true);

    let m = manifold::generate_synthetic_manifold(config.D, 200, 42);
    assert_eq!(m.dim, config.D);
    assert_eq!(m.n_samples, 200);

    let c_base = m.centroid.clone();
    let basis = pca2(&m).expect("PCA-2 should succeed on 200 samples");
    assert_eq!(basis.len(), 2);
    let n0: f32 = basis[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    let n1: f32 = basis[1].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n0 - 1.0).abs() < 1e-4, "b0 not unit: {n0}");
    assert!((n1 - 1.0).abs() < 1e-4, "b1 not unit: {n1}");

    let basis_json = serde_json::json!({
        "c_base": c_base,
        "basis": basis
    });
    let parsed: serde_json::Value = serde_json::from_str(&basis_json.to_string()).unwrap();
    assert_eq!(parsed["c_base"].as_array().unwrap().len(), 10);
    assert_eq!(parsed["basis"].as_array().unwrap().len(), 2);

    let sla_budget = Duration::from_millis(config.sla_budget_ms);
    let v_t: Vec<f32> = (0..config.D).map(|i| i as f32 * 0.1).collect();
    let delta_theta = 0.1;
    let rotated = budget_gated_rotation_fixed_basis(&v_t, delta_theta, &basis, &c_base, sla_budget);
    assert_eq!(rotated.len(), config.D);

    let mut ring_buffer = RingBuffer::new(config.W);
    for i in 0..5 {
        let sample: Vec<f32> = (0..config.D)
            .map(|j| (i as f32 * 0.1 + j as f32).sin())
            .collect();
        ring_buffer.push(sample);
    }
    let window = ring_buffer.window();
    assert_eq!(window.len(), 5);

    let basis_from_centroid = compute_fixed_basis(&c_base);
    assert_eq!(basis_from_centroid.len(), 2);
    let n0c: f32 = basis_from_centroid[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n0c - 1.0).abs() < 1e-6);
}

#[test]
fn test_real_npy_file_loading_and_pca2() {
    let samples = vec![
        vec![0.0, 0.0, 0.0],
        vec![1.0, 0.01, -0.01],
        vec![-1.0, -0.01, 0.01],
        vec![0.5, 0.02, -0.02],
        vec![-0.5, -0.02, 0.02],
    ];
    let mut buf = Cursor::new(Vec::new());
    manifold::write_npy(&mut buf, &samples).unwrap();
    let bytes = buf.into_inner();

    let tmp = std::env::temp_dir().join("test_midas_manifold.npy");
    std::fs::write(&tmp, &bytes).unwrap();

    let m = manifold::load_manifold(tmp.to_str().unwrap(), 3).unwrap();
    assert_eq!(m.n_samples, 5);
    assert_eq!(m.dim, 3);

    let expected_centroid = vec![0.0, 0.0, 0.0];
    for (a, b) in m.centroid.iter().zip(&expected_centroid) {
        assert!(
            (a - b).abs() < 1e-5,
            "centroid mismatch: got {:?}, expected {:?}",
            m.centroid,
            expected_centroid
        );
    }

    let basis = pca2(&m).unwrap();
    assert_eq!(basis.len(), 2);
    let n0: f32 = basis[0].iter().map(|x| x * x).sum::<f32>().sqrt();
    let n1: f32 = basis[1].iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n0 - 1.0).abs() < 1e-4);
    assert!((n1 - 1.0).abs() < 1e-4);
    let dot: f32 = basis[0].iter().zip(&basis[1]).map(|(a, b)| a * b).sum();
    assert!(dot.abs() < 1e-4);

    let b0_along_x: f32 = basis[0][0].abs();
    assert!(
        b0_along_x > 0.99,
        "b0 should align with x-axis (dominant variance), got b0[0]={}",
        basis[0][0]
    );

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn test_fast_fail_missing_file() {
    let result = manifold::load_manifold("/nonexistent/path/does_not_exist.npy", 10);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, edge_core::ManifoldError::FileNotFound(_)),
        "expected FileNotFound, got: {err}"
    );
}

#[test]
fn test_fast_fail_shape_mismatch() {
    let samples = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let mut buf = Cursor::new(Vec::new());
    manifold::write_npy(&mut buf, &samples).unwrap();
    let bytes = buf.into_inner();

    let result = manifold::parse_npy(&bytes, 5);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, edge_core::ManifoldError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got: {err}"
    );
}

#[test]
fn test_fast_fail_malformed_file() {
    let result = manifold::parse_npy(b"definitely not npy", 10);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, edge_core::ManifoldError::InvalidMagic),
        "expected InvalidMagic, got: {err}"
    );
}

#[test]
fn test_synthetic_vs_gram_schmidt_basis_consistency() {
    let m = manifold::generate_synthetic_manifold(10, 200, 42);
    let pca_basis = pca2(&m).unwrap();
    let gs_basis = compute_fixed_basis(&m.centroid);

    assert_eq!(pca_basis.len(), 2);
    assert_eq!(gs_basis.len(), 2);

    for (i, b) in pca_basis.iter().enumerate() {
        let n: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-4, "pca basis[{i}] not unit: {n}");
    }
    for (i, b) in gs_basis.iter().enumerate() {
        let n: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "gs basis[{i}] not unit: {n}");
    }

    let pca_dot: f32 = pca_basis[0].iter().zip(&pca_basis[1]).map(|(a, b)| a * b).sum();
    let gs_dot: f32 = gs_basis[0].iter().zip(&gs_basis[1]).map(|(a, b)| a * b).sum();
    assert!(pca_dot.abs() < 1e-4);
    assert!(gs_dot.abs() < 1e-6);
}

#[test]
fn test_pipeline_uses_manifold_not_placeholder() {
    let config = test_config(true);
    let m = manifold::generate_synthetic_manifold(config.D, 200, 42);
    let c_base = m.centroid.clone();
    let basis = pca2(&m).unwrap();

    let is_nonzero_centroid = c_base.iter().any(|x| x.abs() > 1e-6);
    assert!(
        is_nonzero_centroid,
        "synthetic centroid should be non-zero, got all near zero: {c_base:?}"
    );

    let is_nonzero_basis = basis[0].iter().any(|x| x.abs() > 1e-6)
        && basis[1].iter().any(|x| x.abs() > 1e-6);
    assert!(
        is_nonzero_basis,
        "PCA-2 basis should be non-trivial, got: {basis:?}"
    );

    let v_t: Vec<f32> = vec![0.5; config.D];
    let rotated = budget_gated_rotation_fixed_basis(&v_t, 0.1, &basis, &c_base, Duration::from_millis(10));
    let diff: f32 = v_t
        .iter()
        .zip(&rotated)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        diff > 0.001,
        "rotation with non-trivial basis should change the vector (diff={diff})"
    );
}
