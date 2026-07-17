"""
Diagnostic: Isolate the basis-coupling channel in Check #4.

Check 1: Compare actual basis vectors between naive and adaptive runs.
Check 2: Raise gamma to avoid theta saturation, re-run single config.
"""
import json
import math
import os
import torch
import torch.nn.functional as F

from shadow.defense import (
    SmoothMockModel, SmoothMLPModel,
    penetration_epsilon_windowed, rotation_angle, rotate_manifold_fixed_basis,
)
from shadow.scratch.debug_gradients import vulnerable_basis


def project_Lp_ball(x, x0, epsilon):
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    return torch.clamp(x0 + diff, min=0.0, max=1.0)


def midas_defense_forward_vulnerable_with_eps_p(x_t, trajectory_window, c_base, config, naive=False):
    gamma = config.get("gamma", 0.292)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max = config.get("delta_theta_max_deg", 45.0) * (math.pi / 180.0)

    full_window = trajectory_window + [x_t]
    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)

    if naive:
        theta = theta.detach()
        basis = vulnerable_basis(x_t.detach(), c_base)
    else:
        basis = vulnerable_basis(x_t, c_base)

    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)
    return x_defended, theta, epsilon_p, basis


def pgd_attack_with_basis_logging(x0_init, y_target, classifier, traj, c_base, config,
                                   alpha, epsilon, steps, naive, label=""):
    """PGD attack that logs basis vectors at key steps."""
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    W = config.get("W", 4)
    sliding_window = [w.clone().detach() for w in traj]

    check_steps = {0, 5, 10, 20, 49}
    basis_log = []

    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W - 1):]
        x_def, theta, eps_p, basis = midas_defense_forward_vulnerable_with_eps_p(
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

        if step in check_steps:
            basis_log.append({
                "step": step,
                "basis_b0": basis[0].clone().detach(),
                "basis_b1": basis[1].clone().detach(),
                "theta_deg": math.degrees(theta.item()),
                "eps_p": eps_p.item(),
                "loss": loss.item(),
                "grad_x_norm": torch.norm(grad).item(),
                "grad_theta_norm": grad_theta_norm,
            })

        with torch.no_grad():
            x_t_adv = x_t + alpha * torch.sign(grad)
            x_t_adv = project_Lp_ball(x_t_adv, x0_init, epsilon)
            x_t = x_t_adv
            sliding_window.append(x_t.clone().detach())

    return x_t.detach(), basis_log, sliding_window[-(W - 1):]


def make_dataset(D, seed=42):
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    torch.manual_seed(seed)
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])
    return pool, calibration_batch


def check1_basis_comparison():
    """Check 1: Compare basis vectors between naive and adaptive, same sample."""
    print("=" * 70)
    print("CHECK 1: Basis vector comparison — naive vs adaptive")
    print("=" * 70)

    torch.manual_seed(42)
    D = 10
    config = {"W": 4, "D": D, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
              "delta_theta_max_deg": 45.0, "tau": 0.3}
    pool, calibration_batch = make_dataset(D)
    c_base = torch.zeros(D)

    for cls_name, classifier in [
        ("SmoothMockModel", SmoothMockModel(d=D, calibration_batch=calibration_batch)),
        ("SmoothMLPModel", SmoothMLPModel(d=D, hidden=16, calibration_batch=calibration_batch, seed=1337)),
    ]:
        print(f"\n--- {cls_name} ---")
        x0 = pool[42]
        traj = [torch.rand(D) - 0.5 for _ in range(3)]
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()

        _, naive_log, win_n = pgd_attack_with_basis_logging(
            x0, y_nat, classifier, traj, c_base, config,
            0.01, 0.1, 50, naive=True, label="NAIVE"
        )
        _, adaptive_log, win_a = pgd_attack_with_basis_logging(
            x0, y_nat, classifier, traj, c_base, config,
            0.01, 0.1, 50, naive=False, label="ADAPTIVE"
        )

        print(f"\n  {'step':>4s}  {'theta_n':>8s}  {'theta_a':>8s}  {'theta_diff':>10s}  "
              f"{'b0_l2diff':>10s}  {'b1_l2diff':>10s}  {'loss_n':>8s}  {'loss_a':>8s}")
        print(f"  {'':->4s}  {'':->8s}  {'':->8s}  {'':->10s}  {'':->10s}  {'':->10s}  {'':->8s}  {'':->8s}")

        for n, a in zip(naive_log, adaptive_log):
            assert n["step"] == a["step"]
            b0_diff = torch.norm(n["basis_b0"] - a["basis_b0"]).item()
            b1_diff = torch.norm(n["basis_b1"] - a["basis_b1"]).item()
            theta_diff = abs(n["theta_deg"] - a["theta_deg"])
            print(f"  {n['step']:4d}  {n['theta_deg']:8.3f}  {a['theta_deg']:8.3f}  {theta_diff:10.6f}  "
                  f"{b0_diff:10.8f}  {b1_diff:10.8f}  {n['loss']:8.6f}  {a['loss']:8.6f}")

        print(f"\n  Gradient norms (adaptive only):")
        for n, a in zip(naive_log, adaptive_log):
            print(f"    step={n['step']:3d}  ||dL/dx||={a['grad_x_norm']:.6f}  "
                  f"||dL/dθ||={a['grad_theta_norm']:.6f}")

        b0_all_same = all(
            torch.norm(n["basis_b0"] - a["basis_b0"]).item() < 1e-6
            for n, a in zip(naive_log, adaptive_log)
        )
        b1_all_same = all(
            torch.norm(n["basis_b1"] - a["basis_b1"]).item() < 1e-6
            for n, a in zip(naive_log, adaptive_log)
        )
        print(f"\n  Basis vectors identical (float32): b0={b0_all_same}, b1={b1_all_same}")
        if b0_all_same and b1_all_same:
            print(f"  -> Bases are NUMERICALLY IDENTICAL. detach() vs non-detached x_t")
            print(f"     produces same values (as expected — detach only affects grad tracking).")
            print(f"     The basis-coupling channel's effect is purely in the GRADIENT PATH.")
        else:
            print(f"  -> Bases DIFFER! This is unexpected and needs investigation.")


def check2_raised_gamma():
    """Check 2: Raise gamma so theta doesn't saturate, re-run single config."""
    print("\n" + "=" * 70)
    print("CHECK 2: Raised gamma — isolate basis-coupling channel")
    print("=" * 70)

    torch.manual_seed(42)
    D = 10
    pool, calibration_batch = make_dataset(D)
    c_base = torch.zeros(D)

    config_base = {"W": 4, "D": D, "lambda": 1.0, "k": 2.0,
                   "delta_theta_max_deg": 45.0, "tau": 0.3}

    # From last run, eps_p reaches ~1.87 by step 10.
    # Set gamma=2.0 so eps_p stays below gamma for ~10 steps.
    # But theta = lambda * eps_p still hits ceiling (45° = 0.785 rad) when eps_p > 0.785.
    # So also try lambda=0.1 to keep theta below ceiling for the full run.
    gamma_configs = [
        ("gamma=2.0 lambda=1.0", 2.0, 1.0),   # Archimedean until eps_p>2.0, but ceiling at eps_p=0.785
        ("gamma=10.0 lambda=0.3", 10.0, 0.3),  # Archimedean always, ceiling at eps_p=2.62
        ("gamma=10.0 lambda=0.1", 10.0, 0.1),  # Archimedean always, ceiling at eps_p=7.85 (never hit)
    ]

    for cls_name, classifier in [
        ("SmoothMockModel", SmoothMockModel(d=D, calibration_batch=calibration_batch)),
        ("SmoothMLPModel", SmoothMLPModel(d=D, hidden=16, calibration_batch=calibration_batch, seed=1337)),
    ]:
        print(f"\n{'='*60}")
        print(f"  {cls_name}")
        print(f"{'='*60}")

        for cfg_label, gamma, lambda_ in gamma_configs:
            config = {**config_base, "gamma": gamma, "lambda": lambda_}
            delta_theta_max_rad = config["delta_theta_max_deg"] * math.pi / 180.0

            print(f"\n  --- {cfg_label} ---")
            print(f"      Archimedean ceiling: eps_p = {delta_theta_max_rad / lambda_:.3f}")

            naive_succ = 0
            adaptive_succ = 0
            N_SAMPLES = 50

            for i in range(N_SAMPLES):
                x0 = pool[i]
                traj = [torch.rand(D) - 0.5 for _ in range(3)]
                with torch.no_grad():
                    y_nat = (classifier(x0) > 0.5).float()

                x_adv_n, n_log, win_n = pgd_attack_with_basis_logging(
                    x0, y_nat, classifier, traj, c_base, config,
                    0.01, 0.1, 50, naive=True
                )
                with torch.no_grad():
                    _, theta_n, _, basis_n = midas_defense_forward_vulnerable_with_eps_p(
                        x_adv_n, win_n, c_base, config, naive=False
                    )
                    def_n = rotate_manifold_fixed_basis(x_adv_n, basis_n, theta_n)
                    if (classifier(def_n) > 0.5).item() != y_nat.item():
                        naive_succ += 1

                x_adv_a, a_log, win_a = pgd_attack_with_basis_logging(
                    x0, y_nat, classifier, traj, c_base, config,
                    0.01, 0.1, 50, naive=False
                )
                with torch.no_grad():
                    _, theta_a, _, basis_a = midas_defense_forward_vulnerable_with_eps_p(
                        x_adv_a, win_a, c_base, config, naive=False
                    )
                    def_a = rotate_manifold_fixed_basis(x_adv_a, basis_a, theta_a)
                    if (classifier(def_a) > 0.5).item() != y_nat.item():
                        adaptive_succ += 1

            asr_n = naive_succ / N_SAMPLES
            asr_a = adaptive_succ / N_SAMPLES
            gap = asr_a - asr_n
            se_n = math.sqrt(asr_n * (1 - asr_n) / N_SAMPLES) if N_SAMPLES > 0 else 0
            se_a = math.sqrt(asr_a * (1 - asr_a) / N_SAMPLES) if N_SAMPLES > 0 else 0
            se_gap = math.sqrt(se_n**2 + se_a**2)
            gap_se = gap / se_gap if se_gap > 0 else float('inf')
            print(f"      Naive={asr_n:.0%}  Adaptive={asr_a:.0%}  "
                  f"Gap={gap:+.0%}  Gap/SE={gap_se:.2f}")

            # Show theta trace for sample 0
            print(f"      Sample 0 theta trace (adaptive):")
            for entry in a_log:
                print(f"        step={entry['step']:3d}  θ={entry['theta_deg']:7.3f}°  "
                      f"εp={entry['eps_p']:.6f}")


if __name__ == "__main__":
    check1_basis_comparison()
    check2_raised_gamma()
