"""
Final check: Is sign-based PGD masking the gap?

Repeats vulnerable-basis, eps=0.1, T=50 with raised gamma (10.0, lambda=0.3)
using continuous gradient ascent instead of sign-based PGD.

If a gap appears under continuous updates, the prior null results were an
artifact of sign-PGD's corner-seeking behavior against fixed-weight surrogates.
If the gap is still ~0, the structural-null conclusion is well-supported.
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
    return x_defended, theta, epsilon_p


def pgd_continuous(x0_init, y_target, classifier, traj, c_base, config,
                   alpha, epsilon, steps, naive, log_steps=False):
    """PGD with continuous (normalized) gradient ascent instead of sign step."""
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    W = config.get("W", 4)
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
            grad_norm = torch.norm(grad).item()
            grad_dir = grad / (grad_norm + 1e-8)
            x_t_adv = x_t + alpha * grad_dir
            x_t_adv = project_Lp_ball(x_t_adv, x0_init, epsilon)

        pert_norm = torch.norm(x_t_adv - x0_init, p=float('inf')).item()

        if saturate_step is None and pert_norm >= 0.99 * epsilon:
            saturate_step = step

        step_logs.append({
            "step": step, "loss": loss.item(),
            "theta_deg": math.degrees(theta.item()),
            "eps_p": eps_p.item(),
            "grad_theta_norm": grad_theta_norm,
            "grad_x_norm": grad_norm,
            "pert_linf": pert_norm,
        })

        if log_steps and (step < 30 or step % 10 == 0 or step == steps - 1):
            tag = "ADAPT" if not naive else "NAIVE "
            print(f"    [{tag} s={step:3d}] loss={loss.item():.6f}  "
                  f"θ={math.degrees(theta.item()):7.3f}°  "
                  f"εp={eps_p.item():.6f}  "
                  f"||dL/dx||={grad_norm:.6f}  "
                  f"||dL/dθ||={grad_theta_norm:.6f}  "
                  f"||Δx||∞={pert_norm:.6f}")

        with torch.no_grad():
            x_t = x_t_adv
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W - 1):]
    return x_t.detach(), step_logs, saturate_step, final_window


def make_dataset(D, seed=42):
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    torch.manual_seed(seed)
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])
    return pool, calibration_batch


def main():
    torch.manual_seed(42)
    D = 10

    # Raised gamma config: theta stays ~29°, well below 45° ceiling
    config = {
        "W": 4, "D": D, "gamma": 10.0, "lambda": 0.3, "k": 2.0,
        "delta_theta_max_deg": 45.0, "tau": 0.3,
    }

    pool, calibration_batch = make_dataset(D)
    c_base = torch.zeros(D)

    # alpha=0.01 for continuous updates (same scale as sign-based;
    # continuous moves alpha in the gradient direction, sign moves alpha per-dim)
    alpha = 0.01
    epsilon = 0.1
    steps = 50
    N_SAMPLES = 50

    print("=" * 70)
    print("CONTINUOUS GRADIENT ASCENT — isolated basis-coupling check")
    print("=" * 70)
    print(f"  Config: gamma={config['gamma']}, lambda={config['lambda']}, "
          f"alpha={alpha}, eps={epsilon}, T={steps}")
    print(f"  Update rule: x += alpha * grad / (||grad|| + 1e-8)")
    print(f"  (NOT sign-based PGD)")

    for cls_name, classifier in [
        ("SmoothMockModel", SmoothMockModel(d=D, calibration_batch=calibration_batch)),
        ("SmoothMLPModel", SmoothMLPModel(d=D, hidden=16, calibration_batch=calibration_batch, seed=1337)),
    ]:
        print(f"\n{'='*60}")
        print(f"  {cls_name} — CONTINUOUS UPDATE")
        print(f"{'='*60}")

        naive_succ = 0
        adaptive_succ = 0
        naive_sat = []
        adaptive_sat = []

        for i in range(N_SAMPLES):
            x0 = pool[i]
            traj = [torch.rand(D) - 0.5 for _ in range(3)]
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            x_adv_n, logs_n, sat_n, win_n = pgd_continuous(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, epsilon, steps, naive=True, log_steps=(i == 0)
            )
            if sat_n is not None:
                naive_sat.append(sat_n)
            with torch.no_grad():
                _, theta_n, _ = midas_defense_forward_vulnerable_with_eps_p(
                    x_adv_n, win_n, c_base, config, naive=False
                )
                basis_n = vulnerable_basis(x_adv_n, c_base)
                def_n = rotate_manifold_fixed_basis(x_adv_n, basis_n, theta_n)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            x_adv_a, logs_a, sat_a, win_a = pgd_continuous(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, epsilon, steps, naive=False, log_steps=(i == 0)
            )
            if sat_a is not None:
                adaptive_sat.append(sat_a)
            with torch.no_grad():
                _, theta_a, _ = midas_defense_forward_vulnerable_with_eps_p(
                    x_adv_a, win_a, c_base, config, naive=False
                )
                basis_a = vulnerable_basis(x_adv_a, c_base)
                def_a = rotate_manifold_fixed_basis(x_adv_a, basis_a, theta_a)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{N_SAMPLES}] naive={naive_succ/(i+1):.0%} adaptive={adaptive_succ/(i+1):.0%}")

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        gap = asr_a - asr_n
        se_n = math.sqrt(asr_n * (1 - asr_n) / N_SAMPLES) if N_SAMPLES > 0 else 0
        se_a = math.sqrt(asr_a * (1 - asr_a) / N_SAMPLES) if N_SAMPLES > 0 else 0
        se_gap = math.sqrt(se_n**2 + se_a**2)
        gap_se = gap / se_gap if se_gap > 0 else float('inf')

        print(f"\n  Naive ASR:    {asr_n:.2%} ({naive_succ}/{N_SAMPLES})")
        print(f"  Adaptive ASR: {asr_a:.2%} ({adaptive_succ}/{N_SAMPLES})")
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

    # ---- Also run sign-based PGD for direct comparison on same samples ----
    print(f"\n{'='*70}")
    print("SIGN-BASED PGD — same config for direct comparison")
    print(f"{'='*70}")

    from shadow.check4_corrected import pgd_attack

    for cls_name, classifier in [
        ("SmoothMockModel", SmoothMockModel(d=D, calibration_batch=calibration_batch)),
        ("SmoothMLPModel", SmoothMLPModel(d=D, hidden=16, calibration_batch=calibration_batch, seed=1337)),
    ]:
        print(f"\n--- {cls_name} — SIGN-BASED PGD ---")

        naive_succ = 0
        adaptive_succ = 0

        for i in range(N_SAMPLES):
            x0 = pool[i]
            traj = [torch.rand(D) - 0.5 for _ in range(3)]
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            x_adv_n, _, _, win_n = pgd_attack(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, epsilon, steps, naive=True, log_steps=False
            )
            with torch.no_grad():
                _, theta_n, _ = midas_defense_forward_vulnerable_with_eps_p(
                    x_adv_n, win_n, c_base, config, naive=False
                )
                basis_n = vulnerable_basis(x_adv_n, c_base)
                def_n = rotate_manifold_fixed_basis(x_adv_n, basis_n, theta_n)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            x_adv_a, _, _, win_a = pgd_attack(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, epsilon, steps, naive=False, log_steps=False
            )
            with torch.no_grad():
                _, theta_a, _ = midas_defense_forward_vulnerable_with_eps_p(
                    x_adv_a, win_a, c_base, config, naive=False
                )
                basis_a = vulnerable_basis(x_adv_a, c_base)
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
        print(f"  Naive ASR:    {asr_n:.2%} ({naive_succ}/{N_SAMPLES})")
        print(f"  Adaptive ASR: {asr_a:.2%} ({adaptive_succ}/{N_SAMPLES})")
        print(f"  Gap:          {gap:+.2%}  (SE={se_gap:.4f}, Gap/SE={gap_se:.2f})")


if __name__ == "__main__":
    main()
