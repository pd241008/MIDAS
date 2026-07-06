pub fn compute_momentum(v_t: &[f32], v_t_minus_1: &[f32]) -> f32 {
    if v_t.len() != v_t_minus_1.len() {
        return 0.0;
    }

    let dot: f32 = v_t
        .iter()
        .zip(v_t_minus_1)
        .map(|(a, b)| a * b)
        .sum();

    let norm_t: f32 = v_t.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_tm1: f32 = v_t_minus_1.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_t < f32::EPSILON || norm_tm1 < f32::EPSILON {
        return 0.0;
    }

    (dot / (norm_t * norm_tm1)).clamp(-1.0, 1.0)
}

pub fn penetration_epsilon(v_t: &[f32], v_t_minus_1: &[f32], m_t: f32) -> f32 {
    let diff_norm: f32 = v_t
        .iter()
        .zip(v_t_minus_1)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    diff_norm * m_t.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_momentum_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let m = compute_momentum(&v, &v);
        let expected = 1.0;
        assert!((m - expected).abs() < 1e-6, "expected {expected}, got {m}");
    }

    #[test]
    fn test_compute_momentum_orthogonal() {
        let v_t = vec![1.0, 0.0];
        let v_tm1 = vec![0.0, 1.0];
        let m = compute_momentum(&v_t, &v_tm1);
        assert!((m - 0.0).abs() < 1e-6, "expected 0, got {m}");
    }

    #[test]
    fn test_compute_momentum_anti_aligned() {
        let v_t = vec![1.0, 0.0];
        let v_tm1 = vec![-1.0, 0.0];
        let m = compute_momentum(&v_t, &v_tm1);
        assert!((m - (-1.0)).abs() < 1e-6, "expected -1, got {m}");
    }

    #[test]
    fn test_compute_momentum_zero_vector() {
        let v_t = vec![0.0, 0.0];
        let v_tm1 = vec![1.0, 2.0];
        let m = compute_momentum(&v_t, &v_tm1);
        assert!((m - 0.0).abs() < 1e-6, "expected 0 for zero-norm, got {m}");
    }

    #[test]
    fn test_penetration_epsilon_positive_mt() {
        let v_t = vec![3.0, 4.0];
        let v_tm1 = vec![0.0, 0.0];
        let m_t = 1.0;
        let eps = penetration_epsilon(&v_t, &v_tm1, m_t);
        let expected = 5.0;
        assert!(
            (eps - expected).abs() < 1e-6,
            "expected {expected}, got {eps}"
        );
    }

    #[test]
    fn test_penetration_epsilon_negative_mt_gives_zero() {
        let v_t = vec![1.0, 0.0];
        let v_tm1 = vec![-1.0, 0.0];
        let m_t = -1.0;
        let eps = penetration_epsilon(&v_t, &v_tm1, m_t);
        assert!(
            (eps - 0.0).abs() < 1e-6,
            "expected 0 due to max(0,·) gate, got {eps}"
        );
    }
}
