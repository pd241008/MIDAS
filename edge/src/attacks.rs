use std::fmt;

pub struct AttackConfig {
    pub name: String,
    pub params: AttackParams,
}

pub enum AttackParams {
    Pgd { alpha: f32, t: u32, epsilon: f32 },
    Fgsm { epsilon: f32 },
    CwL2 { kappa: f32, binary_search_steps: u32 },
}

impl fmt::Display for AttackConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.params {
            AttackParams::Pgd { alpha, t, epsilon } => {
                write!(f, "PGD(alpha={alpha}, T={t}, eps={epsilon})")
            }
            AttackParams::Fgsm { epsilon } => {
                write!(f, "FGSM(eps={epsilon})")
            }
            AttackParams::CwL2 {
                kappa,
                binary_search_steps,
            } => {
                write!(f, "C&W-L2(kappa={kappa}, bsteps={binary_search_steps})")
            }
        }
    }
}

pub fn paper_attack_configs() -> Vec<AttackConfig> {
    let mut configs = Vec::new();

    for epsilon in [0.05_f32, 0.1, 0.2] {
        for t in [20_u32, 50, 100] {
            configs.push(AttackConfig {
                name: format!("PGD_T{t}_eps{:.2}", epsilon),
                params: AttackParams::Pgd {
                    alpha: 0.01,
                    t,
                    epsilon,
                },
            });
        }
    }

    configs.push(AttackConfig {
        name: "CW_L2".into(),
        params: AttackParams::CwL2 {
            kappa: 0.0,
            binary_search_steps: 9,
        },
    });

    for epsilon in [0.05_f32, 0.1, 0.2] {
        configs.push(AttackConfig {
            name: format!("FGSM_eps{:.2}", epsilon),
            params: AttackParams::Fgsm { epsilon },
        });
    }

    configs
}

/// Attacker model enum: every attack config is run under both variants.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttackerModel {
    /// Treats delta_theta and the rotation basis as fixed constants with respect
    /// to x_t. Gradients are computed against the defended output without
    /// differentiating through the trajectory/rotation computation.
    Naive,
    /// Backpropagates through the entire trajectory-to-rotation pipeline end to end:
    ///   v_t -> M_t -> epsilon_p -> delta_theta -> R(delta_theta)
    /// with no detaching or substitution anywhere in that chain.
    /// Since the rotation basis is fixed (not attacker-coupled), the differentiable
    /// path runs only through delta_theta's magnitude — not the plane itself.
    ///
    /// Implementation approach (decided): host-side PyTorch shadow model.
    /// See `adaptivetools/` (expected location for the Python differentiable mirror).
    Adaptive,
}

impl fmt::Display for AttackerModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttackerModel::Naive => write!(f, "naive"),
            AttackerModel::Adaptive => write!(f, "adaptive"),
        }
    }
}

// --- Naive attackers (pure Rust, finite-difference gradient estimation) ---

pub fn pgd_attack(
    x: &[f32],
    alpha: f32,
    epsilon: f32,
    t: u32,
    model: &dyn Fn(&[f32]) -> u32,
) -> Vec<f32> {
    let mut adv = x.to_vec();
    let mut rng = rand::thread_rng();
    use rand::Rng;

    for a in adv.iter_mut() {
        *a += rng.gen_range(-epsilon..epsilon);
        *a = a.clamp(0.0, 1.0);
    }

    for _step in 0..t {
        let _label = model(&adv);
        let mut grad = vec![0.0_f32; adv.len()];
        let eps_fd = 1e-3;
        for i in 0..adv.len() {
            let mut x_plus = adv.clone();
            x_plus[i] += eps_fd;
            let l_plus = model(&x_plus);

            let mut x_minus = adv.clone();
            x_minus[i] -= eps_fd;
            let l_minus = model(&x_minus);

            grad[i] = (l_plus as f32 - l_minus as f32) / (2.0 * eps_fd);
        }

        for i in 0..adv.len() {
            adv[i] += alpha * grad[i].signum();
            let diff = adv[i] - x[i];
            let clamped = diff.clamp(-epsilon, epsilon);
            adv[i] = x[i] + clamped;
            adv[i] = adv[i].clamp(0.0, 1.0);
        }
    }

    adv
}

pub fn fgsm_attack(
    x: &[f32],
    epsilon: f32,
    model: &dyn Fn(&[f32]) -> u32,
) -> Vec<f32> {
    let mut adv = x.to_vec();
    let _label = model(x);

    let mut grad = vec![0.0_f32; adv.len()];
    let eps_fd = 1e-3;
    for i in 0..adv.len() {
        let mut x_plus = adv.clone();
        x_plus[i] += eps_fd;
        let l_plus = model(&x_plus);

        let mut x_minus = adv.clone();
        x_minus[i] -= eps_fd;
        let l_minus = model(&x_minus);

        grad[i] = (l_plus as f32 - l_minus as f32) / (2.0 * eps_fd);
    }

    for i in 0..adv.len() {
        adv[i] += epsilon * grad[i].signum();
        adv[i] = adv[i].clamp(0.0, 1.0);
    }

    adv
}

pub fn cw_l2_attack(
    x: &[f32],
    _kappa: f32,
    _binary_search_steps: u32,
    _model: &dyn Fn(&[f32]) -> u32,
) -> Vec<f32> {
    // TODO: full C&W L2 implementation
    x.to_vec()
}

// --- Adaptive attacker (PyTorch shadow model) ---
//
// DESIGN DECISION (per spec): The adaptive attacker requires a differentiable
// shadow implementation of the rotation/projection math, since backpropagating
// through the real Rust pipeline over TCP won't give gradients.
//
// Approach: Host-side PyTorch shadow model.
//   Location: `adaptivetools/` at workspace root (Python package, not in Rust tree).
//   Contents:
//     - PyTorch module mirroring edge-core's rotation math
//       (compute_momentum, penetration_epsilon_windowed, rotation_angle,
//        rotate_manifold_givens_fixed_basis, project_to_manifold)
//     - Finite-difference gradient checker to verify analytic gradients
//       match numerical gradients before using any adaptive-attacker number.
//     - Attack loop: PGD/FGSM/C&W against the differentiable shadow,
//       reporting success rates under AttackerModel::Adaptive.
//
// Verification requirement (not optional):
//   Before trusting any adaptive-attacker success-rate number, verify gradient
//   flow is unbroken end-to-end via a finite-difference check against the
//   analytic gradient path. A broken gradient graph will silently understate
//   attack success and produce a falsely reassuring result.
//
// Results table requirement:
//   Every attack config produces TWO rows:
//     "MIDAS-Edge, naive attacker"
//     "MIDAS-Edge, adaptive attacker"
//   Never report only one.
//
// Numerically-differentiable fallback:
//   If the PyTorch shadow is not available, a finite-difference gradient
//   estimator (central differences through the real Rust pipeline over TCP)
//   can serve as a sanity check, at O(D) forward passes per gradient step.
