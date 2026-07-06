use std::time::{Duration, Instant};

pub fn rotation_angle(
    epsilon_p: f32,
    gamma: f32,
    lambda: f32,
    k: f32,
    delta_theta_max: f32,
) -> f32 {
    if epsilon_p <= gamma {
        lambda * epsilon_p
    } else {
        (lambda * (k * (epsilon_p - gamma)).exp()).min(delta_theta_max)
    }
}

pub fn rotate_manifold_givens(x: &[f32], theta: f32) -> Vec<f32> {
    if x.len() < 2 {
        return x.to_vec();
    }
    let c = theta.cos();
    let s = theta.sin();
    let mut rotated = x.to_vec();
    let x0 = rotated[0];
    let x1 = rotated[1];
    rotated[0] = c * x0 - s * x1;
    rotated[1] = s * x0 + c * x1;
    rotated
}

pub fn project_to_manifold(x_adv: &[f32], c_base: &[f32]) -> Vec<f32> {
    let mut projected = x_adv.to_vec();
    if x_adv.len() != c_base.len() || x_adv.is_empty() {
        return projected;
    }
    let diff: f32 = x_adv
        .iter()
        .zip(c_base)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();
    if diff <= f32::EPSILON {
        return projected;
    }
    for (p, c) in projected.iter_mut().zip(c_base) {
        let d = *p - c;
        *p -= d * 0.5;
    }
    projected
}

pub fn budget_gated_rotation(
    x_adv: &[f32],
    theta: f32,
    c_base: &[f32],
    sla_budget: Duration,
) -> Vec<f32> {
    let deadline = Instant::now() + sla_budget;
    let rotated = rotate_manifold_givens(x_adv, theta);
    if Instant::now() > deadline {
        project_to_manifold(x_adv, c_base)
    } else {
        rotated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_angle_archimedean() {
        let theta = rotation_angle(0.1, 1.0, 0.5, 1.0, 3.14);
        let expected = 0.5 * 0.1;
        assert!(
            (theta - expected).abs() < 1e-6,
            "Archimedean: expected {expected}, got {theta}"
        );
    }

    #[test]
    fn test_rotation_angle_logarithmic() {
        let theta = rotation_angle(2.0, 1.0, 0.5, 1.0, 10.0);
        let expected = 0.5_f32 * (1.0_f32 * (2.0 - 1.0)).exp();
        assert!(
            (theta - expected).abs() < 1e-6,
            "Logarithmic: expected {expected}, got {theta}"
        );
    }

    #[test]
    fn test_rotation_angle_logarithmic_clamped() {
        let theta = rotation_angle(10.0, 1.0, 0.5, 2.0, 1.0);
        assert!(
            (theta - 1.0).abs() < 1e-6,
            "Clamped: expected 1.0, got {theta}"
        );
    }

    #[test]
    fn test_rotation_angle_at_boundary() {
        let theta_below = rotation_angle(0.999, 1.0, 0.5, 1.0, 10.0);
        let theta_above = rotation_angle(1.001, 1.0, 0.5, 1.0, 10.0);
        assert!(
            theta_below > 0.0,
            "Below gamma should still produce positive angle"
        );
        assert!(
            theta_above > theta_below,
            "Above gamma should jump due to exp factor"
        );
    }

    #[test]
    fn test_givens_rotation_identity_when_zero() {
        let x = vec![1.0, 0.0, 3.0];
        let rotated = rotate_manifold_givens(&x, 0.0);
        for (a, b) in x.iter().zip(&rotated) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_givens_rotation_90_deg() {
        let x = vec![1.0, 0.0];
        let rotated = rotate_manifold_givens(&x, std::f32::consts::FRAC_PI_2);
        assert!((rotated[0] - 0.0).abs() < 1e-6);
        assert!((rotated[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_budget_gated_fallback() {
        let x = vec![1.0, 2.0, 3.0];
        let c_base = vec![0.0, 0.0, 0.0];
        let result = budget_gated_rotation(&x, 1.0, &c_base, Duration::from_nanos(0));
        let projected = project_to_manifold(&x, &c_base);
        for (a, b) in result.iter().zip(&projected) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_project_to_manifold_shortens() {
        let x = vec![2.0, 2.0];
        let c = vec![0.0, 0.0];
        let p = project_to_manifold(&x, &c);
        let orig_norm = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        let proj_norm = p.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(proj_norm < orig_norm, "projection should reduce magnitude");
    }
}
