"""
Full validation battery for TrainedSurrogateModel.

Steps:
  1. Train surrogate on synthetic data, freeze weights
  2. Finite-difference gradient check
  3. Bias/threshold calibration check
  4. delta_theta / epsilon_p trace check
  5. Basis-divergence check (naive vs adaptive)
  6. Full Check #4 battery: vulnerable basis, sign-based PGD
  7. Full Check #4 battery: vulnerable basis, continuous gradient update
  8. alpha=0.001/T=500 probe
  9. Gate decision
 10. Fixed-basis evaluation (only if gate passes)
"""
import math
import torch
import torch.nn.functional as F

from shadow.defense import (
    TrainedSurrogateModel, train_surrogate, load_surrogate,
    generate_synthetic_data,
    momentum, penetration_epsilon_windowed, rotation_angle,
    rotate_manifold_fixed_basis, midas_defense_forward,
)
from shadow.save_results import save_results


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D = 10
HIDDEN = 16
TRAIN_SEED = 42
TRAIN_EPOCHS = 500
TRAIN_LR = 0.01
TRAIN_N = 1000

POOL_SIZE = 200
N_SAMPLES = 50
CALIBRATION_POOL = 100

CONFIG = {
    "W": 4, "D": D, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
    "delta_theta_max_deg": 45.0, "tau": 0.3,
}


# ---------------------------------------------------------------------------
# Vulnerable basis (same as debug_gradients.py)
# ---------------------------------------------------------------------------
def vulnerable_basis(x_t, c_base):
    diff = x_t - c_base
    norm = torch.norm(diff)
    if norm > 1e-7:
        b0 = diff / norm
    else:
        b0 = torch.zeros_like(diff)
        b0[0] = 1.0

    e = torch.zeros_like(b0)
    e[1] = 1.0 if len(b0) > 1 else 0.0
    if len(b0) <= 1:
        e[0] = 1.0

    b1 = e - torch.sum(e * b0) * b0
    norm1 = torch.norm(b1)
    if norm1 > 1e-7:
        b1 = b1 / norm1
    else:
        fallback = torch.zeros_like(b0)
        fallback[0] = 1.0
        b1 = fallback - torch.sum(fallback * b0) * b0
        b1 = b1 / (torch.norm(b1) + 1e-7)

    return torch.stack([b0, b1])


# ---------------------------------------------------------------------------
# Defense variants with eps_p return
# ---------------------------------------------------------------------------
def midas_defense_forward_vulnerable(x_t, trajectory_window, c_base, config, naive=False):
    gamma = config.get("gamma", 0.292)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max = config.get("delta_theta_max_deg", 45.0) * (3.141592653589793 / 180.0)

    if naive:
        x_det = x_t.detach()
    else:
        x_det = x_t

    basis = vulnerable_basis(x_det, c_base)
    full_window = list(trajectory_window) + [x_det]
    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)

    if naive:
        theta = theta.detach()

    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)
    return x_defended, theta


def midas_defense_forward_vulnerable_with_eps_p(x_t, trajectory_window, c_base, config, naive=False):
    gamma = config.get("gamma", 0.292)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max = config.get("delta_theta_max_deg", 45.0) * (3.141592653589793 / 180.0)

    if naive:
        x_det = x_t.detach()
    else:
        x_det = x_t

    basis = vulnerable_basis(x_det, c_base)
    full_window = list(trajectory_window) + [x_det]
    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)

    if naive:
        theta = theta.detach()

    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)
    return x_defended, theta, epsilon_p


# ---------------------------------------------------------------------------
# PGD attack (vulnerable basis, sign-based, with sliding window)
# ---------------------------------------------------------------------------
def project_Lp_ball(x, x0, epsilon):
    x_out = torch.clamp(x, x0 - epsilon, x0 + epsilon)
    x_out = torch.clamp(x_out, 0.0, 1.0)
    return x_out


def pgd_attack(x0_init, y_target, classifier, traj, c_base, config,
               alpha, epsilon, steps, naive, log_steps=False):
    W = config.get("W", 4)
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    sliding_window = [w.clone().detach() for w in traj]
    step_logs = []
    saturate_step = None

    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W - 1):]
        x_def, theta, eps_p = midas_defense_forward_vulnerable_with_eps_p(
            x_t, window_slice, c_base, config, naive=naive
        )
        if not naive:
            theta.retain_grad()

        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_target.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        grad_theta_norm = 0.0
        if not naive and theta.grad is not None:
            grad_theta_norm = torch.norm(theta.grad).item()

        with torch.no_grad():
            x_t_adv = x_t + alpha * torch.sign(grad)
            x_t_adv = project_Lp_ball(x_t_adv, x0_init, epsilon)

        pert_norm = torch.norm(x_t_adv - x0_init, p=float('inf')).item()
        if saturate_step is None and pert_norm >= 0.99 * epsilon:
            saturate_step = step

        step_logs.append({
            "step": step, "loss": loss.item(),
            "theta_deg": math.degrees(theta.item()),
            "theta_rad": theta.item(),
            "eps_p": eps_p.item(),
            "grad_theta_norm": grad_theta_norm,
            "grad_x_norm": torch.norm(grad).item(),
            "pert_linf": pert_norm,
        })

        if log_steps and (step < 30 or step % 10 == 0 or step == steps - 1):
            tag = "ADAPT" if not naive else "NAIVE "
            print(f"    [{tag} s={step:3d}] loss={loss.item():.6f}  "
                  f"θ={math.degrees(theta.item()):7.3f}°  "
                  f"εp={eps_p.item():.6f}  "
                  f"||dL/dθ||={grad_theta_norm:.6f}  "
                  f"||Δx||∞={pert_norm:.6f}")

        with torch.no_grad():
            x_t = x_t_adv
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W - 1):]
    return x_t.detach(), step_logs, saturate_step, final_window


# ---------------------------------------------------------------------------
# Continuous gradient PGD (non-sign update)
# ---------------------------------------------------------------------------
def pgd_attack_continuous(x0_init, y_target, classifier, traj, c_base, config,
                          alpha, epsilon, steps, naive, log_steps=False):
    W = config.get("W", 4)
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    sliding_window = [w.clone().detach() for w in traj]
    step_logs = []
    saturate_step = None

    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W - 1):]
        x_def, theta, eps_p = midas_defense_forward_vulnerable_with_eps_p(
            x_t, window_slice, c_base, config, naive=naive
        )
        if not naive:
            theta.retain_grad()

        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_target.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        grad_theta_norm = 0.0
        if not naive and theta.grad is not None:
            grad_theta_norm = torch.norm(theta.grad).item()

        with torch.no_grad():
            normed_grad = grad / (torch.norm(grad) + 1e-8)
            x_t_adv = x_t + alpha * normed_grad
            x_t_adv = project_Lp_ball(x_t_adv, x0_init, epsilon)

        pert_norm = torch.norm(x_t_adv - x0_init, p=float('inf')).item()
        if saturate_step is None and pert_norm >= 0.99 * epsilon:
            saturate_step = step

        step_logs.append({
            "step": step, "loss": loss.item(),
            "theta_deg": math.degrees(theta.item()),
            "theta_rad": theta.item(),
            "eps_p": eps_p.item(),
            "grad_theta_norm": grad_theta_norm,
            "grad_x_norm": torch.norm(grad).item(),
            "pert_linf": pert_norm,
        })

        if log_steps and (step < 30 or step % 10 == 0 or step == steps - 1):
            tag = "ADAPT" if not naive else "NAIVE "
            print(f"    [{tag} s={step:3d}] loss={loss.item():.6f}  "
                  f"θ={math.degrees(theta.item()):7.3f}°  "
                  f"εp={eps_p.item():.6f}  "
                  f"||dL/dθ||={grad_theta_norm:.6f}  "
                  f"||Δx||∞={pert_norm:.6f}")

        with torch.no_grad():
            x_t = x_t_adv
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W - 1):]
    return x_t.detach(), step_logs, saturate_step, final_window


# ---------------------------------------------------------------------------
# Step 1: Finite-difference gradient check
# ---------------------------------------------------------------------------
def finite_diff_check(model, D, config, seed=99):
    print("\n" + "=" * 70)
    print("STEP 1: FINITE-DIFFERENCE GRADIENT CHECK")
    print("=" * 70)

    torch.manual_seed(seed)
    c_base = torch.zeros(D)
    traj = [torch.rand(D) - 0.5 for _ in range(config["W"] - 1)]

    x0 = torch.rand(D, requires_grad=True)
    y = torch.tensor(1.0)

    # Analytical gradient
    x_def_a, theta_a = midas_defense_forward_vulnerable(
        x0, traj, c_base, config, naive=False
    )
    pred_a = model(x_def_a).clamp(1e-7, 1.0 - 1e-7)
    loss_a = F.binary_cross_entropy(pred_a, y.expand_as(pred_a))
    loss_a.backward()
    analytical_grad = x0.grad.clone()

    # Finite-difference gradient
    eps_fd = 1e-4
    fd_grad = torch.zeros(D)
    x0_val = x0.detach().clone()

    for i in range(D):
        x0_pert = x0_val.clone().detach().requires_grad_(False)
        x0_pert[i] += eps_fd
        x_def_p, _ = midas_defense_forward_vulnerable(
            x0_pert, traj, c_base, config, naive=False
        )
        pred_p = model(x_def_p).clamp(1e-7, 1.0 - 1e-7)
        loss_p = F.binary_cross_entropy(pred_p, y.expand_as(pred_p)).item()

        x0_pert2 = x0_val.clone().detach().requires_grad_(False)
        x0_pert2[i] -= eps_fd
        x_def_m, _ = midas_defense_forward_vulnerable(
            x0_pert2, traj, c_base, config, naive=False
        )
        pred_m = model(x_def_m).clamp(1e-7, 1.0 - 1e-7)
        loss_m = F.binary_cross_entropy(pred_m, y.expand_as(pred_m)).item()

        fd_grad[i] = (loss_p - loss_m) / (2 * eps_fd)

    # Compare
    rel_err = torch.abs(analytical_grad - fd_grad) / (torch.abs(analytical_grad) + 1e-8)
    max_rel_err = rel_err.max().item()
    mean_rel_err = rel_err.mean().item()

    print(f"  Analytical grad norm: {torch.norm(analytical_grad).item():.6f}")
    print(f"  FD grad norm:         {torch.norm(fd_grad).item():.6f}")
    print(f"  Max relative error:   {max_rel_err:.6f}")
    print(f"  Mean relative error:  {mean_rel_err:.6f}")
    print(f"  Tolerance:            < 5e-2 relative (relaxed for nonlinear basis)")
    print(f"  Result:               {'PASS' if max_rel_err < 5e-2 else 'FAIL'}")

    return max_rel_err < 5e-2


# ---------------------------------------------------------------------------
# Step 2: Calibration check
# ---------------------------------------------------------------------------
def calibration_check(model, pool, dataset):
    print("\n" + "=" * 70)
    print("STEP 2: BIAS / THRESHOLD CALIBRATION CHECK")
    print("=" * 70)

    with torch.no_grad():
        all_probs = torch.stack([model(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)

    print(f"  Pool size:              {len(pool)}")
    print(f"  Confidence distribution:")
    print(f"    Min:    {nat_conf.min().item():.4f}")
    print(f"    P25:    {torch.quantile(nat_conf, 0.25).item():.4f}")
    print(f"    Median: {nat_conf.median().item():.4f}")
    print(f"    P75:    {torch.quantile(nat_conf, 0.75).item():.4f}")
    print(f"    Max:    {nat_conf.max().item():.4f}")
    print(f"    Mean:   {nat_conf.mean().item():.4f}")

    frac_below_95 = (nat_conf <= 0.95).float().mean().item()
    frac_below_90 = (nat_conf <= 0.90).float().mean().item()
    frac_near_boundary = ((nat_conf >= 0.4) & (nat_conf <= 0.6)).float().mean().item()

    print(f"  Fraction with conf <= 0.95:  {frac_below_95:.2%}")
    print(f"  Fraction with conf <= 0.90:  {frac_below_90:.2%}")
    print(f"  Fraction near boundary (0.4-0.6): {frac_near_boundary:.2%}")

    # Check for saturation
    is_saturated = frac_below_95 < 0.5
    print(f"  Saturated (>{50:.0%} above 0.95): {'YES - PROBLEM' if is_saturated else 'NO - OK'}")

    # Dataset confidence
    with torch.no_grad():
        ds_probs = torch.stack([model(x) for x in dataset])
        ds_conf = torch.where(ds_probs > 0.5, ds_probs, 1.0 - ds_probs)
    print(f"  Dataset confidence: mean={ds_conf.mean().item():.4f}  max={ds_conf.max().item():.4f}")

    return not is_saturated


# ---------------------------------------------------------------------------
# Step 3: delta_theta / epsilon_p trace
# ---------------------------------------------------------------------------
def delta_theta_trace(model, dataset, trajectories, config, D):
    print("\n" + "=" * 70)
    print("STEP 3: DELTA_THETA / EPSILON_P TRACE")
    print("=" * 70)

    c_base = torch.zeros(D)
    x0 = dataset[0]
    traj = trajectories[0]

    with torch.no_grad():
        y_nat = (model(x0) > 0.5).float()

    for eps_label, eps in [("eps=0.1", 0.1), ("eps=0.2", 0.2)]:
        print(f"\n--- T=100, {eps_label}, ADAPTIVE ---")
        _, logs, _, _ = pgd_attack(
            x0, y_nat, model, traj, c_base, config,
            0.01, eps, 100, naive=False, log_steps=True
        )

        theta_vals = [l["theta_deg"] for l in logs]
        eps_p_vals = [l["eps_p"] for l in logs]
        at_ceiling = sum(1 for t in theta_vals if t >= 44.99)
        in_archimedian = sum(1 for t, e in zip(theta_vals, eps_p_vals) if e <= config["gamma"])

        print(f"\n  Summary: θ min={min(theta_vals):.3f}° max={max(theta_vals):.3f}° "
              f"mean={sum(theta_vals)/len(theta_vals):.3f}°")
        print(f"           εp min={min(eps_p_vals):.6f} max={max(eps_p_vals):.6f} "
              f"mean={sum(eps_p_vals)/len(eps_p_vals):.6f}")
        print(f"           Steps at ceiling: {at_ceiling}/{len(logs)}")
        print(f"           Steps in Archimedean phase: {in_archimedian}/{len(logs)}")

    # Check: are phase transitions occurring?
    _, logs, _, _ = pgd_attack(
        x0, y_nat, model, traj, c_base, config,
        0.01, 0.1, 100, naive=False, log_steps=False
    )
    theta_vals = [l["theta_deg"] for l in logs]
    eps_p_vals = [l["eps_p"] for l in logs]
    theta_range = max(theta_vals) - min(theta_vals)
    eps_p_range = max(eps_p_vals) - min(eps_p_vals)

    print(f"\n  Phase transition check:")
    print(f"    θ range:    {theta_range:.3f}°")
    print(f"    εp range:   {eps_p_range:.6f}")
    ok = theta_range > 1.0 and eps_p_range > 0.001
    print(f"    Result:     {'PASS - transitions occurring' if ok else 'FAIL - may need gamma adjustment'}")
    return ok


# ---------------------------------------------------------------------------
# Step 4: Basis-divergence check
# ---------------------------------------------------------------------------
def basis_divergence_check(model, dataset, trajectories, config, D):
    print("\n" + "=" * 70)
    print("STEP 4: BASIS-DIVERGENCE CHECK (NAIVE vs ADAPTIVE)")
    print("=" * 70)

    c_base = torch.zeros(D)
    x0 = dataset[0]
    traj = trajectories[0]

    with torch.no_grad():
        y_nat = (model(x0) > 0.5).float()

    # Run naive and adaptive attacks, log basis at each step
    W = config.get("W", 4)

    # Naive attack — log bases
    x_t = x0.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-0.1, 0.1)
    x_t = project_Lp_ball(x_t, x0, 0.1)
    sliding_n = [w.clone().detach() for w in traj]
    naive_bases = []

    for step in range(20):
        x_t.requires_grad_(True)
        window_slice = sliding_n[-(W - 1):]
        x_def_n, _ = midas_defense_forward_vulnerable(
            x_t, window_slice, c_base, config, naive=True
        )
        pred = model(x_def_n).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_nat.expand_as(pred))
        loss.backward()
        grad = x_t.grad
        with torch.no_grad():
            x_t = x_t + 0.01 * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0, 0.1)
            sliding_n.append(x_t.clone().detach())
            basis_n = vulnerable_basis(x_t.detach(), c_base)
            naive_bases.append(basis_n.clone())

    # Adaptive attack — log bases
    x_t = x0.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-0.1, 0.1)
    x_t = project_Lp_ball(x_t, x0, 0.1)
    sliding_a = [w.clone().detach() for w in traj]
    adaptive_bases = []

    for step in range(20):
        x_t.requires_grad_(True)
        window_slice = sliding_a[-(W - 1):]
        x_def_a, _ = midas_defense_forward_vulnerable(
            x_t, window_slice, c_base, config, naive=False
        )
        pred = model(x_def_a).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_nat.expand_as(pred))
        loss.backward()
        grad = x_t.grad
        with torch.no_grad():
            x_t = x_t + 0.01 * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0, 0.1)
            sliding_a.append(x_t.clone().detach())
            basis_a = vulnerable_basis(x_t, c_base)
            adaptive_bases.append(basis_a.clone())

    # Compare bases
    diffs = []
    for bn, ba in zip(naive_bases, adaptive_bases):
        diff = torch.norm(bn - ba).item()
        diffs.append(diff)

    print(f"  Naive basis norms:     mean={torch.stack([torch.norm(bn) for bn in naive_bases]).mean():.4f}")
    print(f"  Adaptive basis norms:  mean={torch.stack([torch.norm(ba) for ba in adaptive_bases]).mean():.4f}")
    print(f"  Basis L2 diff (b0):    mean={sum(diffs)/len(diffs):.6f}  max={max(diffs):.6f}")
    print(f"  Basis diff > 0.01 at any step: {any(d > 0.01 for d in diffs)}")
    ok = any(d > 0.01 for d in diffs)
    print(f"  Result: {'PASS - bases diverge' if ok else 'FAIL - bases identical'}")
    return ok


# ---------------------------------------------------------------------------
# Full ASR battery (vulnerable basis)
# ---------------------------------------------------------------------------
def run_asr_battery(model, dataset, trajectories, config, D, N_SAMPLES,
                    attack_fn, attack_label):
    print(f"\n{'='*70}")
    print(f"CHECK #4 — VULNERABLE BASIS: {attack_label}")
    print(f"{'='*70}")

    c_base = torch.zeros(D)
    attacks = [
        ("T=20  eps=0.05", 0.01, 20, 0.05),
        ("T=50  eps=0.05", 0.01, 50, 0.05),
        ("T=100 eps=0.05", 0.01, 100, 0.05),
        ("T=20  eps=0.1",  0.01, 20, 0.1),
        ("T=50  eps=0.1",  0.01, 50, 0.1),
        ("T=100 eps=0.1",  0.01, 100, 0.1),
        ("T=20  eps=0.2",  0.01, 20, 0.2),
        ("T=50  eps=0.2",  0.01, 50, 0.2),
        ("T=100 eps=0.2",  0.01, 100, 0.2),
    ]

    results = {}
    for name, alpha, steps, eps in attacks:
        naive_succ = 0
        adaptive_succ = 0

        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            with torch.no_grad():
                y_nat = (model(x0) > 0.5).float()

            x_adv_n, _, _, win_n = attack_fn(x0, y_nat, model, traj, c_base, config,
                                       alpha, eps, steps, naive=True, log_steps=False)
            with torch.no_grad():
                def_n, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_n, win_n, c_base, config, naive=False)
                if (model(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            x_adv_a, _, _, win_a = attack_fn(x0, y_nat, model, traj, c_base, config,
                                       alpha, eps, steps, naive=False, log_steps=False)
            with torch.no_grad():
                def_a, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_a, win_a, c_base, config, naive=False)
                if (model(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        gap = asr_a - asr_n
        se_n = math.sqrt(asr_n * (1 - asr_n) / N_SAMPLES) if N_SAMPLES > 0 else 0
        se_a = math.sqrt(asr_a * (1 - asr_a) / N_SAMPLES) if N_SAMPLES > 0 else 0
        se_gap = math.sqrt(se_n**2 + se_a**2)
        gap_se = gap / se_gap if se_gap > 0 else float('inf')
        results[name] = {"naive": asr_n, "adaptive": asr_a, "gap": gap, "gap_se": gap_se}
        print(f"  {name:20s}  Naive={asr_n:.0%}  Adaptive={asr_a:.0%}  "
              f"Gap={gap:+.0%}  Gap/SE={gap_se:.2f}")

    return results


# ---------------------------------------------------------------------------
# Step 8: alpha=0.001/T=500 probe
# ---------------------------------------------------------------------------
def run_probe(model, dataset, trajectories, config, D):
    print(f"\n{'='*70}")
    print("PART D: alpha=0.001, T=500, eps=0.1 — TRAINED SURROGATE")
    print(f"{'='*70}")

    c_base = torch.zeros(D)
    probe_alpha = 0.001
    probe_eps = 0.1
    probe_T = 500

    naive_succ = 0
    adaptive_succ = 0
    naive_sat = []
    adaptive_sat = []

    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        with torch.no_grad():
            y_nat = (model(x0) > 0.5).float()

        x_adv_n, logs_n, sat_n, win_n = pgd_attack(
            x0, y_nat, model, traj, c_base, config,
            probe_alpha, probe_eps, probe_T, naive=True, log_steps=(i == 0))
        if sat_n is not None:
            naive_sat.append(sat_n)
        with torch.no_grad():
            def_n, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_n, win_n, c_base, config, naive=False)
            if (model(def_n) > 0.5).item() != y_nat.item():
                naive_succ += 1

        x_adv_a, logs_a, sat_a, win_a = pgd_attack(
            x0, y_nat, model, traj, c_base, config,
            probe_alpha, probe_eps, probe_T, naive=False, log_steps=(i == 0))
        if sat_a is not None:
            adaptive_sat.append(sat_a)
        with torch.no_grad():
            def_a, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_a, win_a, c_base, config, naive=False)
            if (model(def_a) > 0.5).item() != y_nat.item():
                adaptive_succ += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(dataset)}] naive={naive_succ/(i+1):.0%} adaptive={adaptive_succ/(i+1):.0%}")

    asr_n = naive_succ / len(dataset)
    asr_a = adaptive_succ / len(dataset)
    gap = asr_a - asr_n
    se_n = math.sqrt(asr_n * (1 - asr_n) / len(dataset))
    se_a = math.sqrt(asr_a * (1 - asr_a) / len(dataset))
    se_gap = math.sqrt(se_n**2 + se_a**2)
    gap_se = gap / se_gap if se_gap > 0 else float('inf')

    print(f"\n  Naive ASR:    {asr_n:.2%} ({naive_succ}/{len(dataset)})")
    print(f"  Adaptive ASR: {asr_a:.2%} ({adaptive_succ}/{len(dataset)})")
    print(f"  Gap:          {gap:+.2%}  (SE={se_gap:.4f}, Gap/SE={gap_se:.2f})")

    if naive_sat:
        print(f"  Naive sat:    median={sorted(naive_sat)[len(naive_sat)//2]}  "
              f"range=[{min(naive_sat)},{max(naive_sat)}]")
    else:
        print(f"  Naive sat:    NONE")
    if adaptive_sat:
        print(f"  Adaptive sat: median={sorted(adaptive_sat)[len(adaptive_sat)//2]}  "
              f"range=[{min(adaptive_sat)},{max(adaptive_sat)}]")
    else:
        print(f"  Adaptive sat: NONE")

    return {"naive": asr_n, "adaptive": asr_a, "gap": gap, "gap_se": gap_se}


# ---------------------------------------------------------------------------
# Fixed-basis evaluation
# ---------------------------------------------------------------------------
def run_fixed_basis(model, dataset, trajectories, config, D, N_SAMPLES):
    print(f"\n{'='*70}")
    print("FIXED-BASIS EVALUATION — TRAINED SURROGATE")
    print(f"{'='*70}")

    basis_path = os.path.join(os.path.dirname(__file__), "..", "results", "basis.json")
    if not os.path.exists(basis_path):
        print(f"  SKIPPED: basis.json not found at {basis_path}")
        return None

    import json
    with open(basis_path) as f:
        basis_data = json.load(f)
    basis = torch.tensor(basis_data["orthonormal_basis"], dtype=torch.float32)
    c_base = torch.tensor(basis_data["c_base"], dtype=torch.float32)

    attacks = [
        ("T=20  eps=0.1", 0.01, 20, 0.1),
        ("T=50  eps=0.1", 0.01, 50, 0.1),
        ("T=100 eps=0.1", 0.01, 100, 0.1),
        ("T=100 eps=0.2", 0.01, 100, 0.2),
    ]

    results = {}
    for name, alpha, steps, eps in attacks:
        naive_succ = 0
        adaptive_succ = 0

        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            with torch.no_grad():
                y_nat = (model(x0) > 0.5).float()

            # Naive: use midas_defense_forward with fixed basis, naive=True
            x_adv_n, _, _, win_n = pgd_attack(
                x0, y_nat, model, traj, c_base, config,
                alpha, eps, steps, naive=True, log_steps=False)
            with torch.no_grad():
                def_n, _ = midas_defense_forward(x_adv_n, win_n, c_base, basis, config, naive=False)
                if (model(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            # Adaptive: use midas_defense_forward with fixed basis, naive=False
            x_adv_a, _, _, win_a = pgd_attack(
                x0, y_nat, model, traj, c_base, config,
                alpha, eps, steps, naive=False, log_steps=False)
            with torch.no_grad():
                def_a, _ = midas_defense_forward(x_adv_a, win_a, c_base, basis, config, naive=False)
                if (model(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        gap = asr_a - asr_n
        se_n = math.sqrt(asr_n * (1 - asr_n) / N_SAMPLES) if N_SAMPLES > 0 else 0
        se_a = math.sqrt(asr_a * (1 - asr_a) / N_SAMPLES) if N_SAMPLES > 0 else 0
        se_gap = math.sqrt(se_n**2 + se_a**2)
        gap_se = gap / se_gap if se_gap > 0 else float('inf')
        results[name] = {"naive": asr_n, "adaptive": asr_a, "gap": gap, "gap_se": gap_se}
        print(f"  {name:20s}  Naive={asr_n:.0%}  Adaptive={asr_a:.0%}  "
              f"Gap={gap:+.0%}  Gap/SE={gap_se:.2f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
import os

def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("TRAINED SURROGATE — FULL VALIDATION BATTERY")
    print("=" * 70)

    # ---- Train surrogate ----
    print("\n" + "=" * 70)
    print("TRAINING TRAINED SURROGATE MODEL")
    print("=" * 70)
    save_dir = os.path.join(os.path.dirname(__file__), "trained_weights")
    model, train_log = train_surrogate(
        d=D, hidden=HIDDEN, seed=TRAIN_SEED, n_epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR, n_train=TRAIN_N, save_dir=save_dir
    )

    # ---- Generate pool and dataset ----
    torch.manual_seed(42)
    pool = [torch.rand(D) for _ in range(POOL_SIZE)]
    calibration_batch = torch.stack(pool[:CALIBRATION_POOL])

    with torch.no_grad():
        all_probs = torch.stack([model(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    print(f"\n  Pool: {len(pool)} → After filter: {len(filtered)} → Using: {N_SAMPLES}")
    assert len(filtered) >= N_SAMPLES, f"Only {len(filtered)} survive the filter"
    dataset = filtered[:N_SAMPLES]
    trajectories = [[torch.rand(D) - 0.5 for _ in range(CONFIG["W"] - 1)] for _ in range(N_SAMPLES)]

    # ---- Step 1: Finite-diff check ----
    fd_pass = finite_diff_check(model, D, CONFIG)

    # ---- Step 2: Calibration check ----
    cal_pass = calibration_check(model, pool, dataset)

    # ---- Step 3: delta_theta / epsilon_p trace ----
    trace_pass = delta_theta_trace(model, dataset, trajectories, CONFIG, D)

    # ---- Step 4: Basis divergence ----
    basis_pass = basis_divergence_check(model, dataset, trajectories, CONFIG, D)

    # ---- Summary of verification steps ----
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Step 1 (finite-diff):    {'PASS' if fd_pass else 'FAIL'}")
    print(f"  Step 2 (calibration):    {'PASS' if cal_pass else 'FAIL'}")
    print(f"  Step 3 (delta_theta):    {'PASS' if trace_pass else 'FAIL'}")
    print(f"  Step 4 (basis diverge):  {'PASS' if basis_pass else 'FAIL'}")
    all_pass = fd_pass and cal_pass and trace_pass and basis_pass
    print(f"  All passed: {'YES' if all_pass else 'NO'}")

    if not all_pass:
        print("\n  ABORTING: Verification steps failed. Fix issues before running battery.")
        return

    # ---- Step 5: Sign-based PGD battery ----
    sign_results = run_asr_battery(
        model, dataset, trajectories, CONFIG, D, N_SAMPLES,
        pgd_attack, "SIGN-BASED PGD"
    )

    # ---- Step 6: Continuous gradient battery ----
    cont_results = run_asr_battery(
        model, dataset, trajectories, CONFIG, D, N_SAMPLES,
        pgd_attack_continuous, "CONTINUOUS GRADIENT"
    )

    # ---- Step 7: alpha=0.001/T=500 probe ----
    probe_results = run_probe(model, dataset, trajectories, CONFIG, D)

    # ---- Step 8: Gate decision ----
    print(f"\n{'='*70}")
    print("GATE DECISION")
    print(f"{'='*70}")

    sign_cont_results = {"sign_pgd": sign_results, "continuous": cont_results}
    for label, results in sign_cont_results.items():
        max_gap_name = max(results, key=lambda k: abs(results[k]["gap"]))
        max_gap = results[max_gap_name]["gap"]
        max_gap_se = results[max_gap_name]["gap_se"]
        gate_pass = abs(max_gap) > 0.02 and max_gap_se > 1.0

        print(f"\n  {label}:")
        print(f"    Largest |gap|: {abs(max_gap):.0%} at {max_gap_name}  (Gap/SE={max_gap_se:.2f})")
        print(f"    Gate: {'PASSED' if gate_pass else 'FAILED'}")
        if not gate_pass:
            reasons = []
            if abs(max_gap) <= 0.02:
                reasons.append(f"|gap|={abs(max_gap):.0%} <= 2%")
            if max_gap_se <= 1.0:
                reasons.append(f"Gap/SE={max_gap_se:.2f} <= 1")
            print(f"    Reasons: {'; '.join(reasons)}")

    probe_gap = probe_results["gap"]
    probe_gap_se = probe_results["gap_se"]
    probe_gate = abs(probe_gap) > 0.02 and probe_gap_se > 1.0
    print(f"\n  probe:")
    print(f"    Gap: {probe_gap:+.0%}  (Gap/SE={probe_gap_se:.2f})")
    print(f"    Gate: {'PASSED' if probe_gate else 'FAILED'}")

    # Overall gate: any update rule passes?
    any_gate = False
    for label, results in sign_cont_results.items():
        max_gap_name = max(results, key=lambda k: abs(results[k]["gap"]))
        max_gap = results[max_gap_name]["gap"]
        max_gap_se = results[max_gap_name]["gap_se"]
        if abs(max_gap) > 0.02 and max_gap_se > 1.0:
            any_gate = True
    if probe_gate:
        any_gate = True

    print(f"\n  OVERALL GATE: {'PASSED — proceed to fixed-basis' if any_gate else 'FAILED — do NOT proceed to fixed-basis'}")

    # ---- Step 9: Fixed-basis (only if gate passes) ----
    if any_gate:
        fixed_results = run_fixed_basis(model, dataset, trajectories, CONFIG, D, N_SAMPLES)
    else:
        print("\n  Skipping fixed-basis evaluation (gate failed).")

    # ---- Save results ----
    saved_results = []
    for update_rule, results in [("sign", sign_results), ("continuous", cont_results)]:
        for name, r in results.items():
            parts = name.split()
            T_val = int(parts[0].split("=")[1])
            eps_val = float(parts[1].split("=")[1])
            saved_results.append({
                "update_rule": update_rule,
                "T": T_val,
                "epsilon": eps_val,
                "naive_asr": r["naive"],
                "adaptive_asr": r["adaptive"],
                "n": N_SAMPLES,
                "gap": r["gap"],
                "gap_se": r["gap_se"],
            })
    saved_results.append({
        "update_rule": "sign",
        "T": 500,
        "epsilon": 0.1,
        "alpha": 0.001,
        "label": "probe",
        "naive_asr": probe_results["naive"],
        "adaptive_asr": probe_results["adaptive"],
        "n": len(dataset),
        "gap": probe_results["gap"],
        "gap_se": probe_results["gap_se"],
    })

    save_results(
        script_name="check_trained_surrogate.py",
        config=CONFIG,
        results=saved_results,
        extra={"model": "TrainedSurrogateModel", "pool_size": POOL_SIZE,
               "n_samples": N_SAMPLES, "gate_passed": any_gate},
    )


if __name__ == "__main__":
    main()
