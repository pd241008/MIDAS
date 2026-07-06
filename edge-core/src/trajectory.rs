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

/// Windowed penetration epsilon:
///   epsilon_p = (1 / |W|) * sum_{i in W} ||v_i||_2 * max(0, M_i)
///
/// Averaged over the current ring-buffer window. Replaces the single-step form
/// `||v_t - v_{t-1}||_2 * max(0, M_t)` which has a critical failure mode:
/// a well-converged attacker taking consistent steps drives v_t ≈ v_{t-1},
/// so ||v_t - v_{t-1}|| → 0 exactly as M_t → 1 — the signal vanishes precisely
/// when the attack is most dangerous. The windowed form accumulates sustained
/// aligned magnitude instead.
///
/// Takes a window of feature vectors (newest last). Returns 0 if fewer than 2 vectors.
pub fn penetration_epsilon_windowed(window: &[Vec<f32>]) -> f32 {
    if window.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0_f32;
    let mut pairs = 0;

    for i in 0..window.len() - 1 {
        let v_cur = &window[i + 1];
        let v_prev = &window[i];
        let m = compute_momentum(v_cur, v_prev);
        let norm_cur: f32 = v_cur.iter().map(|x| x * x).sum::<f32>().sqrt();
        sum += norm_cur * m.max(0.0);
        pairs += 1;
    }

    sum / pairs as f32
}

/// Single-step penetration epsilon (used only for regression-test comparison).
/// Retained as a private reference to demonstrate why the windowed form is necessary.
#[allow(dead_code)]
fn penetration_epsilon_single_step(v_t: &[f32], v_t_minus_1: &[f32], m_t: f32) -> f32 {
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

    // --- momentum tests ---

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

    // --- windowed penetration epsilon tests ---

    #[test]
    fn test_penetration_epsilon_windowed_basic() {
        let window = vec![
            vec![0.0, 0.0],
            vec![3.0, 4.0],
        ];
        let eps = penetration_epsilon_windowed(&window);
        // M = cos_sim((3,4), (0,0)) = 0 (due to zero-norm guard)
        // ||v_cur|| = 5, max(0, 0) = 0, contribution = 0
        // avg = 0 / 1 = 0
        assert!((eps - 0.0).abs() < 1e-6, "expected 0, got {eps}");
    }

    #[test]
    fn test_penetration_epsilon_windowed_aligned() {
        let window = vec![
            vec![1.0, 0.0],
            vec![3.0, 0.0],
        ];
        let eps = penetration_epsilon_windowed(&window);
        // M = cos_sim((3,0), (1,0)) = 1.0
        // ||v_cur|| = 3, max(0, 1) = 1, contribution = 3
        // avg = 3 / 1 = 3
        assert!((eps - 3.0).abs() < 1e-6, "expected 3, got {eps}");
    }

    #[test]
    fn test_penetration_epsilon_windowed_multi_pair() {
        let window = vec![
            vec![1.0, 0.0],
            vec![3.0, 0.0],
            vec![6.0, 0.0],
        ];
        let eps = penetration_epsilon_windowed(&window);
        // pair 1: M=1, ||v||=3 → contribution 3
        // pair 2: M=1, ||v||=6 → contribution 6
        // avg = (3 + 6) / 2 = 4.5
        assert!((eps - 4.5).abs() < 1e-6, "expected 4.5, got {eps}");
    }

    #[test]
    fn test_penetration_epsilon_windowed_insufficient() {
        let eps = penetration_epsilon_windowed(&[]);
        assert!((eps - 0.0).abs() < 1e-6, "expected 0 for empty window");
        let eps = penetration_epsilon_windowed(&[vec![1.0]]);
        assert!((eps - 0.0).abs() < 1e-6, "expected 0 for single-vector window");
    }

    #[test]
    fn test_penetration_epsilon_negative_mt_gives_zero() {
        let window = vec![
            vec![1.0, 0.0],
            vec![-1.0, 0.0],
        ];
        let eps = penetration_epsilon_windowed(&window);
        // M = cos_sim((-1,0), (1,0)) = -1, max(0, -1) = 0
        // contribution = 0, avg = 0
        assert!(
            (eps - 0.0).abs() < 1e-6,
            "expected 0 due to max(0,·) gate, got {eps}"
        );
    }

    // --- convergent-attacker regression test ---
    //
    // A well-converged attacker taking consistent small steps drives
    // v_t ≈ v_{t-1} so ||v_t - v_{t-1}|| → 0 even as M_t → 1.
    // The single-step form collapses to ~0; the windowed form must not.

    #[test]
    fn test_windowed_epsilon_resists_convergent_attack() {
        // Construct a trajectory: start at origin, take 10 tiny steps in the
        // same direction. Each step moves 0.01 along axis 0.
        let mut window = Vec::new();
        window.push(vec![0.0_f32, 0.0]);
        for i in 0..10 {
            window.push(vec![(i as f32 + 1.0) * 0.01, 0.0]);
        }

        let windowed_eps = penetration_epsilon_windowed(&window);

        // The windowed form accumulates sustained aligned magnitude.
        // With 10 steps of increasing magnitude in same direction, epsilon
        // should be clearly non-zero (each step's contribution is ||v_cur|| * 1).
        // Average of {0.01, 0.02, ..., 0.10} = 0.055
        assert!(
            windowed_eps > 0.05,
            "windowed epsilon collapsed to {windowed_eps}, expected > 0.05"
        );

        // Demonstrate the single-step form would fail.
        let last = window.last().unwrap();
        let second_last = window.get(window.len() - 2).unwrap();
        let m = compute_momentum(last, second_last);
        let single_eps = penetration_epsilon_single_step(last, second_last, m);

        assert!(
            single_eps < 0.01,
            "single-step epsilon = {single_eps}, should be near 0 for convergent attacker"
        );
    }
}
