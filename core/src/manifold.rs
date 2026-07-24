use std::fmt;
use std::fs;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct Manifold {
    pub samples: Vec<Vec<f32>>,
    pub centroid: Vec<f32>,
    pub dim: usize,
    pub n_samples: usize,
}

#[derive(Debug)]
pub enum ManifoldError {
    FileNotFound(String),
    IoError(String),
    InvalidMagic,
    UnsupportedVersion(u8, u8),
    InvalidHeader(String),
    ShapeMismatch { expected: usize, actual: usize },
    UnsupportedDtype(String),
    EmptyData,
    ZeroNorm,
}

impl fmt::Display for ManifoldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileNotFound(p) => write!(f, "manifold file not found: {p}"),
            Self::IoError(msg) => write!(f, "I/O error reading manifold: {msg}"),
            Self::InvalidMagic => write!(f, "not a valid .npy file (bad magic bytes)"),
            Self::UnsupportedVersion(maj, min) => {
                write!(f, "unsupported npy version: {maj}.{min}")
            }
            Self::InvalidHeader(msg) => write!(f, "invalid npy header: {msg}"),
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "manifold dimension mismatch: expected D={expected}, got {actual}"
                )
            }
            Self::UnsupportedDtype(d) => write!(f, "unsupported npy dtype: {d}"),
            Self::EmptyData => write!(f, "manifold file contains no data"),
            Self::ZeroNorm => write!(f, "manifold centroid has zero norm"),
        }
    }
}

impl std::error::Error for ManifoldError {}

pub fn load_manifold(path: &str, expected_dim: usize) -> Result<Manifold, ManifoldError> {
    let bytes = fs::read(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            ManifoldError::FileNotFound(path.to_string())
        } else {
            ManifoldError::IoError(format!("{path}: {e}"))
        }
    })?;
    parse_npy(&bytes, expected_dim)
}

pub fn parse_npy(bytes: &[u8], expected_dim: usize) -> Result<Manifold, ManifoldError> {
    if bytes.len() < 128 {
        return Err(ManifoldError::InvalidMagic);
    }
    if &bytes[0..6] != b"\x93NUMPY" {
        return Err(ManifoldError::InvalidMagic);
    }

    let major = bytes[6];
    let minor = bytes[7];

    if major > 2 {
        return Err(ManifoldError::UnsupportedVersion(major, minor));
    }

    let (header_len, data_offset) = if major == 1 {
        let hl = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        (hl, 10)
    } else {
        let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        (hl, 12)
    };

    let header_end = data_offset + header_len;
    if header_end > bytes.len() {
        return Err(ManifoldError::InvalidHeader("truncated header".into()));
    }

    let header_str = std::str::from_utf8(&bytes[data_offset..header_end])
        .map_err(|e| ManifoldError::InvalidHeader(e.to_string()))?;

    let descr = parse_header_field(header_str, "descr")
        .ok_or_else(|| ManifoldError::InvalidHeader("missing 'descr'".into()))?;
    let fortran_order = parse_header_field(header_str, "fortran_order")
        .map(|s| s == "True")
        .unwrap_or(false);
    let shape_str = parse_header_field(header_str, "shape")
        .ok_or_else(|| ManifoldError::InvalidHeader("missing 'shape'".into()))?;

    let shape = parse_shape(&shape_str)?;

    if fortran_order {
        return Err(ManifoldError::InvalidHeader(
            "Fortran order not supported".into(),
        ));
    }

    let descr_clean = descr.trim().trim_matches('\'').trim_matches('"');
    let (elem_size, is_double) = match descr_clean {
        "<f4" | "f4" => (4, false),
        "<f8" | "f8" => (8, true),
        _ => return Err(ManifoldError::UnsupportedDtype(descr)),
    };

    let data_bytes = &bytes[header_end..];

    match shape.len() {
        1 => {
            let d = shape[0];
            if d != expected_dim {
                return Err(ManifoldError::ShapeMismatch {
                    expected: expected_dim,
                    actual: d,
                });
            }
            let expected_bytes = d * elem_size;
            if data_bytes.len() < expected_bytes {
                return Err(ManifoldError::InvalidHeader(
                    "data too short for declared shape".into(),
                ));
            }
            let values = read_float_slice(data_bytes, d, elem_size, is_double);
            Ok(Manifold {
                samples: vec![values.clone()],
                centroid: values,
                dim: d,
                n_samples: 1,
            })
        }
        2 => {
            let n = shape[0];
            let d = shape[1];
            if d != expected_dim {
                return Err(ManifoldError::ShapeMismatch {
                    expected: expected_dim,
                    actual: d,
                });
            }
            if n == 0 {
                return Err(ManifoldError::EmptyData);
            }
            let expected_bytes = n * d * elem_size;
            if data_bytes.len() < expected_bytes {
                return Err(ManifoldError::InvalidHeader(
                    "data too short for declared shape".into(),
                ));
            }

            let mut samples = Vec::with_capacity(n);
            let mut centroid = vec![0.0f32; d];

            for row in 0..n {
                let offset = row * d * elem_size;
                let sample = read_float_slice(&data_bytes[offset..], d, elem_size, is_double);
                for (c, s) in centroid.iter_mut().zip(&sample) {
                    *c += s;
                }
                samples.push(sample);
            }

            for c in centroid.iter_mut() {
                *c /= n as f32;
            }

            Ok(Manifold {
                samples,
                centroid,
                dim: d,
                n_samples: n,
            })
        }
        _ => Err(ManifoldError::InvalidHeader(format!(
            "expected 1D or 2D array, got {}D",
            shape.len()
        ))),
    }
}

fn read_float_slice(data: &[u8], count: usize, elem_size: usize, is_double: bool) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let off = i * elem_size;
            if is_double {
                f64::from_le_bytes([
                    data[off],
                    data[off + 1],
                    data[off + 2],
                    data[off + 3],
                    data[off + 4],
                    data[off + 5],
                    data[off + 6],
                    data[off + 7],
                ]) as f32
            } else {
                f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
            }
        })
        .collect()
}

fn parse_header_field(header: &str, field: &str) -> Option<String> {
    let key = format!("'{field}'");
    let pos = header.find(&key)?;
    let rest = &header[pos + key.len()..];
    let rest = rest.trim_start().trim_start_matches(':').trim_start();
    let mut depth = 0i32;
    let mut end = rest.len();
    for (i, c) in rest.char_indices() {
        match c {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' if depth == 0 => {
                end = i;
                break;
            }
            '}' if depth == 0 => {
                end = i;
                break;
            }
            _ => {}
        }
    }
    Some(rest[..end].trim().to_string())
}

fn parse_shape(s: &str) -> Result<Vec<usize>, ManifoldError> {
    let s = s.trim();
    let s = s.trim_start_matches('(').trim_end_matches(')');
    if s.is_empty() {
        return Ok(vec![]);
    }
    s.split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| {
            x.trim()
                .parse::<usize>()
                .map_err(|_| ManifoldError::InvalidHeader(format!("bad shape element: {x}")))
        })
        .collect()
}

pub fn pca2(manifold: &Manifold) -> Result<Vec<Vec<f32>>, ManifoldError> {
    let d = manifold.dim;
    let n = manifold.n_samples;

    if d == 0 {
        return Ok(vec![vec![1.0], vec![0.0]]);
    }

    if n < 2 {
        return Ok(fixed_basis_from_centroid(&manifold.centroid));
    }

    let centered: Vec<Vec<f32>> = manifold
        .samples
        .iter()
        .map(|s| {
            s.iter()
                .zip(&manifold.centroid)
                .map(|(x, c)| x - c)
                .collect()
        })
        .collect();

    let mut cov = vec![vec![0.0f32; d]; d];
    let scale = 1.0 / (n - 1) as f32;
    for i in 0..d {
        for j in i..d {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += centered[k][i] * centered[k][j];
            }
            let val = sum * scale;
            cov[i][j] = val;
            cov[j][i] = val;
        }
    }

    let b0 = power_iteration(&cov, 1000);

    let eigenvalue: f32 = cov
        .iter()
        .enumerate()
        .map(|(i, row)| row.iter().zip(&b0).map(|(c, b)| c * b).sum::<f32>() * b0[i])
        .sum();
    for i in 0..d {
        for j in 0..d {
            cov[i][j] -= eigenvalue * b0[i] * b0[j];
        }
    }

    let b1 = power_iteration(&cov, 1000);

    let dot: f32 = b0.iter().zip(&b1).map(|(a, b)| a * b).sum();
    let mut b1_orth: Vec<f32> = b1.iter().zip(&b0).map(|(bi, b0i)| bi - dot * b0i).collect();
    let n1: f32 = b1_orth.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n1 > f32::EPSILON {
        for x in b1_orth.iter_mut() {
            *x /= n1;
        }
    } else {
        b1_orth = fallback_second_basis(&b0);
    }

    Ok(vec![b0, b1_orth])
}

fn power_iteration(matrix: &[Vec<f32>], max_iter: usize) -> Vec<f32> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }

    let mut b: Vec<f32> = vec![0.0; n];
    b[0] = 1.0;

    for _ in 0..max_iter {
        let mut b_new = vec![0.0f32; n];
        for i in 0..n {
            b_new[i] = matrix[i].iter().zip(&b).map(|(a, x)| a * x).sum();
        }

        let norm: f32 = b_new.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < f32::EPSILON {
            break;
        }

        let dot: f32 = b
            .iter()
            .zip(&b_new)
            .map(|(old, new)| old * new / norm)
            .sum();
        for x in b.iter_mut() {
            *x = 0.0;
        }
        for (bi, ni) in b.iter_mut().zip(&b_new) {
            *bi = ni / norm;
        }

        if (dot.abs() - 1.0).abs() < 1e-6 {
            break;
        }
    }

    b
}

fn fallback_second_basis(b0: &[f32]) -> Vec<f32> {
    let dim = b0.len();
    let mut best = vec![0.0f32; dim];
    if dim > 1 {
        best[1] = 1.0;
    } else if dim > 0 {
        best[0] = 1.0;
    }
    let dot: f32 = best.iter().zip(b0).map(|(a, b)| a * b).sum();
    for (bi, b0i) in best.iter_mut().zip(b0) {
        *bi -= dot * b0i;
    }
    let n: f32 = best.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > f32::EPSILON {
        for x in best.iter_mut() {
            *x /= n;
        }
    }
    best
}

fn fixed_basis_from_centroid(c_base: &[f32]) -> Vec<Vec<f32>> {
    let dim = c_base.len();
    if dim == 0 {
        return vec![vec![1.0], vec![0.0]];
    }
    let norm: f32 = c_base.iter().map(|x| x * x).sum::<f32>().sqrt();
    let b0: Vec<f32> = if norm > f32::EPSILON {
        c_base.iter().map(|x| x / norm).collect()
    } else {
        let mut v = vec![0.0_f32; dim];
        if dim > 0 {
            v[0] = 1.0;
        }
        v
    };
    vec![b0.clone(), fallback_second_basis(&b0)]
}

pub fn generate_synthetic_manifold(
    dim: usize,
    n_samples: usize,
    seed: u64,
) -> Manifold {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut samples = Vec::with_capacity(n_samples);
    let mut centroid = vec![0.0f32; dim];

    for _ in 0..n_samples {
        let sample: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        for (c, s) in centroid.iter_mut().zip(&sample) {
            *c += s;
        }
        samples.push(sample);
    }

    for c in centroid.iter_mut() {
        *c /= n_samples as f32;
    }

    Manifold {
        samples,
        centroid,
        dim,
        n_samples,
    }
}

pub fn write_npy<W: std::io::Write>(w: &mut W, data: &[Vec<f32>]) -> std::io::Result<()> {
    let n = data.len();
    let d = if n > 0 { data[0].len() } else { 0 };

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1, 0])?;

    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({n}, {d}), }}"
    );
    let min_len = header.len() + 1;
    let aligned_total = ((10 + min_len + 63) / 64) * 64;
    let padded_len = aligned_total - 10;
    let pad = padded_len - header.len() - 1;

    let hl_bytes = (padded_len as u16).to_le_bytes();
    w.write_all(&hl_bytes)?;

    w.write_all(header.as_bytes())?;
    for _ in 0..pad {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;

    for row in data {
        for val in row {
            w.write_all(&val.to_le_bytes())?;
        }
    }

    Ok(())
}

pub fn write_npy_1d<W: std::io::Write>(w: &mut W, data: &[f32]) -> std::io::Result<()> {
    let d = data.len();

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1, 0])?;

    let header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d},), }}"
    );
    let min_len = header.len() + 1;
    let aligned_total = ((10 + min_len + 63) / 64) * 64;
    let padded_len = aligned_total - 10;
    let pad = padded_len - header.len() - 1;

    let hl_bytes = (padded_len as u16).to_le_bytes();
    w.write_all(&hl_bytes)?;

    w.write_all(header.as_bytes())?;
    for _ in 0..pad {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;

    for val in data {
        w.write_all(&val.to_le_bytes())?;
    }

    Ok(())
}

pub fn load_manifold_for_export(
    config: &crate::config::MidasConfig,
    synthetic_override: Option<bool>,
) -> (Vec<f32>, Vec<Vec<f32>>) {
    use crate::rotation::compute_fixed_basis;
    use log::{error, info, warn};

    let use_synthetic = synthetic_override.unwrap_or(config.use_synthetic_manifold);

    let (manifold, is_synthetic) = if use_synthetic {
        warn!(
            "WARNING: using synthetic manifold for basis export — \
             not loaded from disk"
        );
        (generate_synthetic_manifold(config.D, 200, 42), true)
    } else {
        match load_manifold(&config.manifold_path, config.D) {
            Ok(m) => {
                info!(
                    "loaded manifold for basis export from {}: {} samples, dim={}",
                    config.manifold_path, m.n_samples, m.dim
                );
                (m, false)
            }
            Err(e) => {
                error!("FATAL: failed to load manifold for basis export from {}: {e}", config.manifold_path);
                panic!("manifold loading failed for basis export: {e}");
            }
        }
    };

    let c_base = manifold.centroid.clone();
    let basis = if manifold.n_samples >= 2 {
        match pca2(&manifold) {
            Ok(b) => b,
            Err(e) => {
                warn!("PCA-2 failed for basis export ({e}), falling back to Gram-Schmidt");
                compute_fixed_basis(&c_base)
            }
        }
    } else {
        compute_fixed_basis(&c_base)
    };

    if is_synthetic {
        warn!("basis export: using synthetic manifold basis");
    }

    (c_base, basis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_write_read_roundtrip_2d() {
        let original = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let mut buf = Cursor::new(Vec::new());
        write_npy(&mut buf, &original).unwrap();
        let bytes = buf.into_inner();

        let manifold = parse_npy(&bytes, 3).unwrap();
        assert_eq!(manifold.dim, 3);
        assert_eq!(manifold.n_samples, 3);
        for (i, row) in manifold.samples.iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                assert!(
                    (val - original[i][j]).abs() < 1e-6,
                    "mismatch at [{i}][{j}]: {val} != {}",
                    original[i][j]
                );
            }
        }
        assert!((manifold.centroid[0] - 4.0).abs() < 1e-6);
        assert!((manifold.centroid[1] - 5.0).abs() < 1e-6);
        assert!((manifold.centroid[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_write_read_roundtrip_1d() {
        let original = vec![0.5f32, -1.0, 2.5];
        let mut buf = Cursor::new(Vec::new());
        write_npy_1d(&mut buf, &original).unwrap();
        let bytes = buf.into_inner();

        let manifold = parse_npy(&bytes, 3).unwrap();
        assert_eq!(manifold.dim, 3);
        assert_eq!(manifold.n_samples, 1);
        assert_eq!(manifold.samples.len(), 1);
        for (a, b) in manifold.centroid.iter().zip(&original) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_shape_mismatch_fails_fast() {
        let original = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let mut buf = Cursor::new(Vec::new());
        write_npy(&mut buf, &original).unwrap();
        let bytes = buf.into_inner();

        let result = parse_npy(&bytes, 5);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::ShapeMismatch { expected, actual } => {
                assert_eq!(expected, 5);
                assert_eq!(actual, 3);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn test_missing_file_fails_fast() {
        let result = load_manifold("/nonexistent/path/c_base.npy", 10);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::FileNotFound(p) => assert!(p.contains("nonexistent")),
            other => panic!("expected FileNotFound, got {other:?}"),
        }
    }

    #[test]
    fn test_malformed_file_fails_fast() {
        let bad_bytes = b"NOT_AN_NPY_FILE";
        let result = parse_npy(bad_bytes, 10);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::InvalidMagic => {}
            other => panic!("expected InvalidMagic, got {other:?}"),
        }
    }

    #[test]
    fn test_empty_data_fails() {
        let original: Vec<Vec<f32>> = vec![];
        let mut buf = Cursor::new(Vec::new());
        write_npy(&mut buf, &original).unwrap();
        let bytes = buf.into_inner();

        let result = parse_npy(&bytes, 3);
        assert!(result.is_err());
        match result.unwrap_err() {
            ManifoldError::ShapeMismatch { expected, actual } => {
                assert_eq!(expected, 3);
                assert_eq!(actual, 0);
            }
            ManifoldError::EmptyData => {}
            other => panic!("expected ShapeMismatch or EmptyData, got {other:?}"),
        }
    }

    #[test]
    fn test_pca2_known_data() {
        let mut samples = Vec::new();
        for i in 0..100 {
            let x = i as f32 * 0.1;
            samples.push(vec![x, x * 0.5, 0.0]);
        }
        let mut centroid = vec![0.0f32; 3];
        for s in &samples {
            for (c, v) in centroid.iter_mut().zip(s) {
                *c += v;
            }
        }
        for c in centroid.iter_mut() {
            *c /= 100.0;
        }
        let manifold = Manifold {
            samples,
            centroid,
            dim: 3,
            n_samples: 100,
        };

        let basis = pca2(&manifold).unwrap();
        assert_eq!(basis.len(), 2);
        let n0: f32 = basis[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1: f32 = basis[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-4, "b0 not unit: {n0}");
        assert!((n1 - 1.0).abs() < 1e-4, "b1 not unit: {n1}");
        let dot: f32 = basis[0].iter().zip(&basis[1]).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-4, "b0, b1 not orthogonal: {dot}");

        let b0_dir = &basis[0];
        let data_dir = vec![1.0f32, 0.5, 0.0];
        let data_norm: f32 = data_dir.iter().map(|x| x * x).sum::<f32>().sqrt();
        let data_unit: Vec<f32> = data_dir.iter().map(|x| x / data_norm).collect();
        let alignment: f32 = b0_dir.iter().zip(&data_unit).map(|(a, b)| a * b).sum();
        assert!(
            alignment.abs() > 0.99,
            "b0 should align with dominant direction, got alignment={alignment}"
        );
    }

    #[test]
    fn test_pca2_single_sample_falls_back() {
        let manifold = Manifold {
            samples: vec![vec![1.0, 2.0, 3.0]],
            centroid: vec![1.0, 2.0, 3.0],
            dim: 3,
            n_samples: 1,
        };
        let basis = pca2(&manifold).unwrap();
        assert_eq!(basis.len(), 2);
        let n0: f32 = basis[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pca2_zero_centroid() {
        let manifold = Manifold {
            samples: vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]],
            centroid: vec![0.0, 0.0, 0.0],
            dim: 3,
            n_samples: 2,
        };
        let basis = pca2(&manifold).unwrap();
        assert_eq!(basis.len(), 2);
        let n0: f32 = basis[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_synthetic_manifold_deterministic() {
        let m1 = generate_synthetic_manifold(10, 50, 42);
        let m2 = generate_synthetic_manifold(10, 50, 42);
        assert_eq!(m1.dim, m2.dim);
        assert_eq!(m1.n_samples, m2.n_samples);
        for (s1, s2) in m1.samples.iter().zip(&m2.samples) {
            for (a, b) in s1.iter().zip(s2) {
                assert!((a - b).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_synthetic_manifold_shape() {
        let m = generate_synthetic_manifold(10, 200, 42);
        assert_eq!(m.dim, 10);
        assert_eq!(m.n_samples, 200);
        assert_eq!(m.samples.len(), 200);
        assert_eq!(m.centroid.len(), 10);
        for s in &m.samples {
            assert_eq!(s.len(), 10);
        }
    }

    #[test]
    fn test_synthetic_pca2_produces_valid_basis() {
        let m = generate_synthetic_manifold(10, 200, 42);
        let basis = pca2(&m).unwrap();
        assert_eq!(basis.len(), 2);
        let n0: f32 = basis[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1: f32 = basis[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-4);
        assert!((n1 - 1.0).abs() < 1e-4);
        let dot: f32 = basis[0].iter().zip(&basis[1]).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-4);
    }

    #[test]
    fn test_basis_matches_export_format() {
        let m = generate_synthetic_manifold(10, 200, 42);
        let basis = pca2(&m).unwrap();
        let json = serde_json::json!({
            "c_base": m.centroid,
            "basis": basis
        });
        let s = json.to_string();
        assert!(s.contains("c_base"));
        assert!(s.contains("basis"));
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert!(parsed["c_base"].is_array());
        assert!(parsed["basis"].is_array());
        assert_eq!(parsed["c_base"].as_array().unwrap().len(), 10);
        assert_eq!(parsed["basis"].as_array().unwrap().len(), 2);
    }
}
