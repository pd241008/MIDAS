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

/// Apply a Givens rotation in the (0, 1) coordinate plane by angle `theta`.
/// Used only internally and in tests; the deployment path should use
/// `rotate_manifold_givens_fixed_basis` instead.
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

/// Rotate x by `theta` in the 2D subspace spanned by `basis` (two orthonormal
/// D-vectors). The basis is held fixed for the life of the deployment — it is
/// computed offline via PCA-2 of `c_base` and MUST NOT depend on per-query state.
///
/// This is a security-motivated design decision: an attacker-coupled basis
/// (rotation plane built from the attacker's own step) can be exploited by an
/// adversary who backpropagates through the rotation. With a fixed basis, the
/// attacker's differentiable path runs only through `delta_theta`'s magnitude,
/// not the rotation plane itself.
pub fn rotate_manifold_givens_fixed_basis(
    x: &[f32],
    basis: &[Vec<f32>],
    theta: f32,
) -> Vec<f32> {
    if x.len() < 2 || basis.len() < 2 || basis[0].len() != x.len() || basis[1].len() != x.len() {
        return x.to_vec();
    }
    let c = theta.cos();
    let s = theta.sin();
    // Project x onto the fixed basis
    let coord0: f32 = x.iter().zip(&basis[0]).map(|(a, b)| a * b).sum();
    let coord1: f32 = x.iter().zip(&basis[1]).map(|(a, b)| a * b).sum();
    // Rotate the coordinates
    let rot0 = c * coord0 - s * coord1;
    let rot1 = s * coord0 + c * coord1;
    // Reconstruct in original space
    let mut rotated = x.to_vec();
    for i in 0..rotated.len() {
        rotated[i] += (rot0 - coord0) * basis[0][i] + (rot1 - coord1) * basis[1][i];
    }
    rotated
}

/// Compute the fixed rotation basis (top-2 principal components) from the base
/// manifold centroid `c_base`.
///
/// TODO: replace with real PCA-2 computation once manifold samples are available.
/// Currently returns a placeholder: b0 = c_base normalized (or [1,0,0,...] if
/// zero-norm), b1 = a Gram-Schmidt orthogonalized standard basis vector.
pub fn compute_fixed_basis(c_base: &[f32]) -> Vec<Vec<f32>> {
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
    // b1: standard basis vector e_1, Gram-Schmidt against b0
    let mut b1 = vec![0.0_f32; dim];
    if dim > 1 {
        b1[1] = 1.0;
    } else {
        b1[0] = 1.0;
    }
    // Subtract b0 projection
    let dot_b1_b0: f32 = b1.iter().zip(&b0).map(|(a, b)| a * b).sum();
    for i in 0..dim {
        b1[i] -= dot_b1_b0 * b0[i];
    }
    let n1: f32 = b1.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n1 > f32::EPSILON {
        for x in b1.iter_mut() {
            *x /= n1;
        }
    }
    vec![b0, b1]
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

/// Budget-gated rotation using the fixed basis (deployment path).
/// 1. Record deadline = now + sla_budget
/// 2. Rotate using the fixed (non-attacker-coupled) basis
/// 3. If deadline exceeded, fall back to project_to_manifold(x_adv, c_base)
pub fn budget_gated_rotation_fixed_basis(
    x_adv: &[f32],
    theta: f32,
    basis: &[Vec<f32>],
    c_base: &[f32],
    sla_budget: Duration,
) -> Vec<f32> {
    let deadline = Instant::now() + sla_budget;
    let rotated = rotate_manifold_givens_fixed_basis(x_adv, basis, theta);
    if Instant::now() > deadline {
        project_to_manifold(x_adv, c_base)
    } else {
        rotated
    }
}

/// Budget-gated rotation using the simple (0,1)-plane rotation.
/// Kept for backward compatibility; new code should use the fixed-basis variant.
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
    fn test_fixed_basis_givens_rotation() {
        // The standard basis should give the same result as rotate_manifold_givens
        let x = vec![1.0, 2.0, 3.0];
        let basis = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let theta = std::f32::consts::FRAC_PI_4;
        let expected = rotate_manifold_givens(&x, theta);
        let got = rotate_manifold_givens_fixed_basis(&x, &basis, theta);
        for (a, b) in expected.iter().zip(&got) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
    }

    #[test]
    fn test_compute_fixed_basis_orthonormal() {
        let c_base = vec![3.0, 1.0, 4.0];
        let basis = compute_fixed_basis(&c_base);
        assert_eq!(basis.len(), 2);
        // Check orthonormality
        let dot: f32 = basis[0].iter().zip(&basis[1]).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-6, "basis vectors not orthogonal: dot = {dot}");
        let n0: f32 = basis[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1: f32 = basis[1].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-6, "b0 not unit: {n0}");
        assert!((n1 - 1.0).abs() < 1e-6, "b1 not unit: {n1}");
    }

    #[test]
    fn test_fixed_basis_preserves_invariant_subspace() {
        // A vector entirely in the orthogonal complement of the basis should be unchanged
        let _dim = 4;
        let c_base = vec![1.0, 0.0, 0.0, 0.0];
        let basis = compute_fixed_basis(&c_base);
        // x is along axis 2 (orthogonal to both b0=[1,0,0,0] and b1=[0,1,0,0])
        let x = vec![0.0, 0.0, 5.0, 0.0];
        let rotated = rotate_manifold_givens_fixed_basis(&x, &basis, 1.0);
        for (a, b) in x.iter().zip(&rotated) {
            assert!((a - b).abs() < 1e-6, "expected {a}, got {b}");
        }
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
