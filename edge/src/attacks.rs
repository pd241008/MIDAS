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
