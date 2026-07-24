"""
Check #4 pass with paper-aligned config: W=10, tau=0.75.

Steps:
  1. Load pre-trained TrainedSurrogateModel
  2. Finite-difference gradient check (W=10)
  3. epsilon_p distribution under W=10 on benign trajectories → gamma recalibration check
  4. delta_theta / epsilon_p trace (W=10)
  5. 9-config sign-based PGD sweep, N=50, vulnerable basis
  6. Save results
"""
import math
import os
import json
import torch
import torch.nn.functional as F

from shadow.defense import (
    TrainedSurrogateModel, load_surrogate,
    momentum, penetration_epsilon_windowed, rotation_angle,
    rotate_manifold_fixed_basis,
)
from shadow.save_results import save_results


# ---------------------------------------------------------------------------
# Config — paper-aligned
# ---------------------------------------------------------------------------
D = 10
HIDDEN = 16
POOL_SIZE = 200
N_SAMPLES = 50
CALIBRATION_POOL = 100

CONFIG = {
    "W": 10, "D": D, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
    "delta_theta_max_deg": 45.0, "tau": 0.75,
}


# ---------------------------------------------------------------------------
# Vulnerable basis
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
# Defense forward (vulnerable basis) with eps_p return
# ---------------------------------------------------------------------------
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


def midas_defense_forward_vulnerable(x_t, trajectory_window, c_base, config, naive=False):
    x_def, theta, _ = midas_defense_forward_vulnerable_with_eps_p(
        x_t, trajectory_window, c_base, config, naive=naive
    )
    return x_def, theta


# ---------------------------------------------------------------------------
# PGD attack (vulnerable basis, sign-based, sliding window)
# ---------------------------------------------------------------------------
def project_Lp_ball(x, x0, epsilon):
    x_out = torch.clamp(x, x0 - epsilon, x0 + epsilon)
    x_out = torch.clamp(x_out, 0.0, 1.0)
    return x_out


def pgd_attack(x0_init, y_target, classifier, traj, c_base, config,
               alpha, epsilon, steps, naive, log_steps=False):
    W = config.get("W", 10)
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
# Step 1: Finite-difference gradient check
# ---------------------------------------------------------------------------
def finite_diff_check(model, config, seed=99):
    print("\n" + "=" * 70)
    print("STEP 1: FINITE-DIFFERENCE GRADIENT CHECK (W=10)")
    print("=" * 70)

    W = config["W"]
    torch.manual_seed(seed)
    c_base = torch.zeros(D)
    traj = [torch.rand(D) - 0.5 for _ in range(W - 1)]

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
# Step 2: epsilon_p distribution under W=10 on benign trajectories
# ---------------------------------------------------------------------------
def epsilon_p_distribution_check(model, dataset, trajectories, config):
    print("\n" + "=" * 70)
    print("STEP 2: EPSILON_P DISTRIBUTION UNDER W=10")
    print("=" * 70)

    c_base = torch.zeros(D)
    gamma_old = config["gamma"]

    # Collect epsilon_p values from benign (no-attack) trajectories under W=10
    eps_p_values = []
    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        with torch.no_grad():
            full_window = list(traj) + [x0]
            ep = penetration_epsilon_windowed(full_window)
            eps_p_values.append(ep.item())

    eps_p_tensor = torch.tensor(eps_p_values)
    p95 = torch.quantile(eps_p_tensor, 0.95).item()
    p99 = torch.quantile(eps_p_tensor, 0.99).item()
    median = eps_p_tensor.median().item()
    mean = eps_p_tensor.mean().item()

    print(f"  epsilon_p distribution (W=10, benign trajectories):")
    print(f"    Min:    {eps_p_tensor.min().item():.6f}")
    print(f"    P25:    {torch.quantile(eps_p_tensor, 0.25).item():.6f}")
    print(f"    Median: {median:.6f}")
    print(f"    Mean:   {mean:.6f}")
    print(f"    P75:    {torch.quantile(eps_p_tensor, 0.75).item():.6f}")
    print(f"    P95:    {p95:.6f}")
    print(f"    P99:    {p99:.6f}")
    print(f"    Max:    {eps_p_tensor.max().item():.6f}")
    print(f"  Current gamma: {gamma_old:.6f}")

    # With W=10, the windowed average divides by more pairs (W-1=9 pairs vs W-1=3),
    # so epsilon_p values will generally be lower. Check if gamma is still meaningful.
    # A good gamma should be above most benign epsilon_p but below the adversarial ones.
    # Use ~95th percentile of the adversarial epsilon_p (from a short attack) to calibrate.
    # For now, report the distribution and whether the old gamma still sits above the
    # median benign epsilon_p.

    frac_below_gamma = (eps_p_tensor <= gamma_old).float().mean().item()
    print(f"  Fraction of benign epsilon_p <= gamma ({gamma_old:.3f}): {frac_below_gamma:.1%}")

    if p95 > gamma_old:
        print(f"  WARNING: P95 of benign epsilon_p ({p95:.6f}) > gamma ({gamma_old:.6f})")
        print(f"  gamma should be recalibrated. Using P95 = {p95:.6f} as new gamma.")
        new_gamma = p95
    else:
        print(f"  OK: gamma ({gamma_old:.6f}) is above P95 of benign epsilon_p ({p95:.6f})")
        print(f"  No recalibration needed.")
        new_gamma = gamma_old

    return new_gamma, eps_p_tensor


# ---------------------------------------------------------------------------
# Step 3: delta_theta / epsilon_p trace
# ---------------------------------------------------------------------------
def delta_theta_trace(model, dataset, trajectories, config):
    print("\n" + "=" * 70)
    print("STEP 3: DELTA_THETA / EPSILON_P TRACE (W=10)")
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
# Step 4: 9-config sign-based PGD sweep (vulnerable basis)
# ---------------------------------------------------------------------------
def run_sign_pgd_sweep(model, dataset, trajectories, config, N_SAMPLES):
    print(f"\n{'='*70}")
    print("CHECK #4 — VULNERABLE BASIS: SIGN-BASED PGD (W=10, tau=0.75)")
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

            # Naive attack (theta detached from graph)
            x_adv_n, _, _, win_n = pgd_attack(
                x0, y_nat, model, traj, c_base, config,
                alpha, eps, steps, naive=True, log_steps=False
            )
            with torch.no_grad():
                def_n, _, _ = midas_defense_forward_vulnerable_with_eps_p(
                    x_adv_n, win_n, c_base, config, naive=False
                )
                if (model(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            # Adaptive attack (gradients flow through theta)
            x_adv_a, _, _, win_a = pgd_attack(
                x0, y_nat, model, traj, c_base, config,
                alpha, eps, steps, naive=False, log_steps=False
            )
            with torch.no_grad():
                def_a, _, _ = midas_defense_forward_vulnerable_with_eps_p(
                    x_adv_a, win_a, c_base, config, naive=False
                )
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
def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("CHECK #4 — PAPER-ALIGNED CONFIG: W=10, tau=0.75, gamma=0.292")
    print("Trained surrogate (pre-trained, frozen weights)")
    print("=" * 70)

    print(f"\n  Config: {json.dumps(CONFIG, indent=2)}")

    # ---- Load pre-trained surrogate ----
    print("\n" + "=" * 70)
    print("LOADING PRE-TRAINED SURROGATE")
    print("=" * 70)
    weights_path = os.path.join(os.path.dirname(__file__), "trained_weights", "trained_surrogate.pt")
    meta_path = os.path.join(os.path.dirname(__file__), "trained_weights", "trained_surrogate_meta.json")
    model = load_surrogate(d=D, hidden=HIDDEN, weights_path=weights_path)
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  Loaded: {meta}")

    # ---- Generate pool and dataset ----
    torch.manual_seed(42)
    pool = [torch.rand(D) for _ in range(POOL_SIZE)]

    with torch.no_grad():
        all_probs = torch.stack([model(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    print(f"\n  Pool: {len(pool)} → After filter: {len(filtered)} → Using: {N_SAMPLES}")
    assert len(filtered) >= N_SAMPLES, f"Only {len(filtered)} survive the filter"
    dataset = filtered[:N_SAMPLES]

    # Trajectories: W-1 = 9 samples per trajectory for W=10
    trajectories = [[torch.rand(D) - 0.5 for _ in range(CONFIG["W"] - 1)] for _ in range(N_SAMPLES)]

    # ---- Step 1: Finite-diff check ----
    fd_pass = finite_diff_check(model, CONFIG)

    # ---- Step 2: epsilon_p distribution → gamma check ----
    new_gamma, eps_p_tensor = epsilon_p_distribution_check(model, dataset, trajectories, CONFIG)

    if new_gamma != CONFIG["gamma"]:
        print(f"\n  >>> RECALIBRATING gamma: {CONFIG['gamma']:.6f} → {new_gamma:.6f}")
        CONFIG["gamma"] = new_gamma
        print(f"  Updated config: {json.dumps(CONFIG, indent=2)}")
    else:
        print(f"\n  >>> gamma unchanged at {CONFIG['gamma']:.6f}")

    # ---- Step 3: delta_theta trace ----
    trace_pass = delta_theta_trace(model, dataset, trajectories, CONFIG)

    # ---- Sanity summary ----
    print(f"\n{'='*70}")
    print("SANITY CHECK SUMMARY (W=10, tau=0.75)")
    print(f"{'='*70}")
    print(f"  Finite-diff gradient:  {'PASS' if fd_pass else 'FAIL'}")
    print(f"  Phase transitions:     {'PASS' if trace_pass else 'FAIL'}")
    print(f"  gamma:                 {CONFIG['gamma']:.6f}")
    sanity_ok = fd_pass and trace_pass
    print(f"  All OK: {'YES' if sanity_ok else 'NO'}")

    if not sanity_ok:
        print("\n  ABORTING: Sanity checks failed.")
        return

    # ---- Step 4: 9-config sign-PGD sweep ----
    results = run_sign_pgd_sweep(model, dataset, trajectories, CONFIG, N_SAMPLES)

    # ---- Gate decision ----
    print(f"\n{'='*70}")
    print("GATE DECISION")
    print(f"{'='*70}")

    max_gap_name = max(results, key=lambda k: abs(results[k]["gap"]))
    max_gap = results[max_gap_name]["gap"]
    max_gap_se = results[max_gap_name]["gap_se"]
    gate_pass = abs(max_gap) > 0.02 and max_gap_se > 1.0

    print(f"  Largest |gap|: {abs(max_gap):.0%} at {max_gap_name}  (Gap/SE={max_gap_se:.2f})")
    print(f"  Gate: {'PASSED' if gate_pass else 'FAILED'}")
    if not gate_pass:
        reasons = []
        if abs(max_gap) <= 0.02:
            reasons.append(f"|gap|={abs(max_gap):.0%} <= 2%")
        if max_gap_se <= 1.0:
            reasons.append(f"Gap/SE={max_gap_se:.2f} <= 1")
        print(f"  Reasons: {'; '.join(reasons)}")

    # ---- Save results ----
    saved_results = []
    for name, r in results.items():
        parts = name.split()
        T_val = int(parts[0].split("=")[1])
        eps_val = float(parts[1].split("=")[1])
        saved_results.append({
            "update_rule": "sign_pgd",
            "T": T_val,
            "epsilon": eps_val,
            "naive_asr": r["naive"],
            "adaptive_asr": r["adaptive"],
            "n": N_SAMPLES,
            "gap": r["gap"],
            "gap_se": r["gap_se"],
        })

    saved_path = save_results(
        script_name="check4_w10_tau075.py",
        config=CONFIG,
        results=saved_results,
        extra={
            "model": "TrainedSurrogateModel",
            "pool_size": POOL_SIZE,
            "n_samples": N_SAMPLES,
            "gate_passed": gate_pass,
            "max_gap_config": max_gap_name,
            "max_gap": max_gap,
            "max_gap_se": max_gap_se,
            "gamma_original": 0.292,
            "gamma_used": CONFIG["gamma"],
            "sanity_fd_pass": fd_pass,
            "sanity_trace_pass": trace_pass,
        },
    )

    print(f"\n  Final config: {json.dumps(CONFIG, indent=2)}")
    print(f"  Gate: {'PASSED — proceed to N=200 recheck and fixed-basis evaluation' if gate_pass else 'FAILED — report as scoped null with W=10/tau=0.75'}")


if __name__ == "__main__":
    main()
