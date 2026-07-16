"""
Diagnostic probe: alpha=0.001, T=500, eps=0.1, vulnerable basis.

Tests whether the ~0 naive-vs-adaptive gap is saturation (parameter choice)
or structural (linear surrogate + sign-based PGD).

Reports:
  1. Naive ASR vs Adaptive ASR for this single config
  2. Per-step loss and ||dL/dtheta|| logs for sample 0
  3. ||delta_x||_inf trajectory to confirm non-saturation behavior
  4. Comparison: where does saturation hit vs alpha=0.01?
"""
import json
import math
import torch
import torch.nn.functional as F
import os

from shadow.defense import SmoothMockModel
from shadow.scratch.debug_gradients import (
    midas_defense_forward_vulnerable,
    vulnerable_basis,
)


def project_Lp_ball(x, x0, epsilon):
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    return torch.clamp(x0 + diff, min=0.0, max=1.0)


def pgd_probe(x0_init, y_target, classifier, traj, c_base, config,
              alpha, epsilon, steps, naive, sample_idx, log_steps=True):
    """PGD on vulnerable basis with full per-step logging for the probe."""
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    step_logs = []
    saturate_step = None  # first step where ||delta_x||_inf >= 0.99 * epsilon

    for step in range(steps):
        x_t.requires_grad_(True)
        window_cloned = [w.clone().detach() for w in traj]
        x_def, theta_vuln = midas_defense_forward_vulnerable(
            x_t, window_cloned, c_base, config, naive=naive
        )
        if not naive:
            theta_vuln.retain_grad()
        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_target.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        grad_theta_norm = 0.0
        if not naive and theta_vuln.grad is not None:
            grad_theta_norm = torch.norm(theta_vuln.grad).item()

        with torch.no_grad():
            x_t_adv = x_t + alpha * torch.sign(grad)
            x_t_adv = project_Lp_ball(x_t_adv, x0_init, epsilon)

        pert_norm = torch.norm(x_t_adv - x0_init, p=float('inf')).item()

        if saturate_step is None and pert_norm >= 0.99 * epsilon:
            saturate_step = step

        step_logs.append({
            "step": step,
            "loss": loss.item(),
            "grad_theta_norm": grad_theta_norm,
            "grad_x_norm": torch.norm(grad).item(),
            "pert_linf": pert_norm,
        })

        if log_steps and (step < 30 or step % 50 == 0 or step == steps - 1):
            tag = "ADAPT" if not naive else "NAIVE "
            print(f"    [{tag} s={step:3d}] loss={loss.item():.6f}  "
                  f"||dL/d\u03b8||={grad_theta_norm:.8f}  "
                  f"||dL/dx||={torch.norm(grad).item():.6f}  "
                  f"||\u0394x||\u221e={pert_norm:.6f}")

        with torch.no_grad():
            x_t = x_t_adv

    return x_t.detach(), step_logs, saturate_step


def main():
    torch.manual_seed(42)

    # ---- Load basis ----
    if not os.path.exists("results/basis.json"):
        raise RuntimeError("results/basis.json not found!")
    with open("results/basis.json") as f:
        basis_data = json.load(f)
    c_base = torch.tensor(basis_data["c_base"], dtype=torch.float32)

    config = {
        "W": 4, "D": 10, "gamma": 0.5, "lambda": 1.0, "k": 2.0,
        "delta_theta_max_deg": 45.0, "tau": 0.3,
        "sla_budget_ms": 10, "channel_capacity": 4,
    }
    D = config["D"]

    # ---- Same calibration + filtering as diagnose_check4.py ----
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])
    classifier = SmoothMockModel(d=D, calibration_batch=calibration_batch)

    with torch.no_grad():
        all_probs = torch.stack([classifier(x) for x in pool])
        nat_labels = (all_probs > 0.5).float()
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    print(f"  Pool size: {len(pool)}")
    print(f"  After confidence filter (<=0.95): {len(filtered)}")
    print(f"  N_SAMPLES: {N_SAMPLES}")

    if len(filtered) < N_SAMPLES:
        print(f"  *** FATAL: Only {len(filtered)} samples survive ***")
        return

    dataset = filtered[:N_SAMPLES]
    trajectories = [
        [torch.randn(D) for _ in range(config["W"] - 1)] for _ in range(N_SAMPLES)
    ]

    # ================================================================
    # PROBE CONFIG: eps=0.1, alpha=0.001, T=500
    # ================================================================
    probe_alpha = 0.001
    probe_eps = 0.1
    probe_T = 500

    print("\n" + "=" * 70)
    print(f"PROBE: eps={probe_eps}, alpha={probe_alpha}, T={probe_T}, vulnerable basis")
    print("=" * 70)

    naive_succ = 0
    adaptive_succ = 0
    naive_saturate_steps = []
    adaptive_saturate_steps = []

    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()

        # Naive
        x_adv_n, logs_n, sat_n = pgd_probe(
            x0, y_nat, classifier, traj, c_base, config,
            probe_alpha, probe_eps, probe_T, naive=True, sample_idx=i, log_steps=(i == 0)
        )
        if sat_n is not None:
            naive_saturate_steps.append(sat_n)
        with torch.no_grad():
            def_n, _ = midas_defense_forward_vulnerable(x_adv_n, traj, c_base, config, naive=False)
            if (classifier(def_n) > 0.5).item() != y_nat.item():
                naive_succ += 1

        # Adaptive
        x_adv_a, logs_a, sat_a = pgd_probe(
            x0, y_nat, classifier, traj, c_base, config,
            probe_alpha, probe_eps, probe_T, naive=False, sample_idx=i, log_steps=(i == 0)
        )
        if sat_a is not None:
            adaptive_saturate_steps.append(sat_a)
        with torch.no_grad():
            def_a, _ = midas_defense_forward_vulnerable(x_adv_a, traj, c_base, config, naive=False)
            if (classifier(def_a) > 0.5).item() != y_nat.item():
                adaptive_succ += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_SAMPLES}] running naive={naive_succ/(i+1):.0%} adaptive={adaptive_succ/(i+1):.0%}")

    asr_n = naive_succ / N_SAMPLES
    asr_a = adaptive_succ / N_SAMPLES
    gap = asr_a - asr_n

    print(f"\n--- RESULTS ---")
    print(f"  Naive ASR:   {asr_n:.2%} ({naive_succ}/{N_SAMPLES})")
    print(f"  Adaptive ASR:{asr_a:.2%} ({adaptive_succ}/{N_SAMPLES})")
    print(f"  Gap:         {gap:+.2%}")

    # SE on proportions
    se_n = math.sqrt(asr_n * (1 - asr_n) / N_SAMPLES)
    se_a = math.sqrt(asr_a * (1 - asr_a) / N_SAMPLES)
    se_gap = math.sqrt(se_n**2 + se_a**2)
    print(f"\n  SE(Naive):    {se_n:.4f}")
    print(f"  SE(Adaptive): {se_a:.4f}")
    print(f"  SE(Gap):      {se_gap:.4f}")
    print(f"  Gap / SE:     {gap / se_gap:.2f}" if se_gap > 0 else "  Gap / SE:     inf")

    # Saturation analysis
    print(f"\n--- SATURATION ANALYSIS ---")
    if naive_saturate_steps:
        print(f"  Naive: median saturation step = {sorted(naive_saturate_steps)[len(naive_saturate_steps)//2]}")
        print(f"    range: [{min(naive_saturate_steps)}, {max(naive_saturate_steps)}]")
        print(f"    fraction saturating: {len(naive_saturate_steps)}/{N_SAMPLES}")
    else:
        print(f"  Naive: NO samples saturated (perturbation never hit epsilon boundary)")

    if adaptive_saturate_steps:
        print(f"  Adaptive: median saturation step = {sorted(adaptive_saturate_steps)[len(adaptive_saturate_steps)//2]}")
        print(f"    range: [{min(adaptive_saturate_steps)}, {max(adaptive_saturate_steps)}]")
        print(f"    fraction saturating: {len(adaptive_saturate_steps)}/{N_SAMPLES}")
    else:
        print(f"  Adaptive: NO samples saturated (perturbation never hit epsilon boundary)")

    # Comparison with alpha=0.01
    print(f"\n--- COMPARISON: alpha=0.01 saturation step (eps=0.1, T=100) ---")
    # For alpha=0.01, eps=0.1, the max steps to saturate is ceil(0.1/0.01) = 10
    # (in L-inf PGD with sign update, each step moves at most alpha, so saturate at step >= eps/alpha)
    max_steps_to_saturate_01 = math.ceil(probe_eps / 0.01)
    max_steps_to_saturate_001 = math.ceil(probe_eps / probe_alpha)
    print(f"  alpha=0.01: max {max_steps_to_saturate_01} steps to saturate (eps/alpha = {probe_eps/0.01:.0f})")
    print(f"  alpha=0.001: max {max_steps_to_saturate_001} steps to saturate (eps/alpha = {probe_eps/probe_alpha:.0f})")
    print(f"  At T=500 with alpha=0.001: {probe_T - max_steps_to_saturate_001} steps post-saturation")
    print(f"  At T=100 with alpha=0.01:  {100 - max_steps_to_saturate_01} steps post-saturation")

    # ================================================================
    # REFERENCE: alpha=0.01, T=100, eps=0.1 for direct comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("REFERENCE: alpha=0.01, T=100, eps=0.1, vulnerable basis (same probe framework)")
    print("=" * 70)

    ref_alpha = 0.01
    ref_T = 100
    ref_naive_succ = 0
    ref_adaptive_succ = 0

    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()

        x_adv_n, _, _ = pgd_probe(
            x0, y_nat, classifier, traj, c_base, config,
            ref_alpha, probe_eps, ref_T, naive=True, sample_idx=i, log_steps=(i == 0)
        )
        with torch.no_grad():
            def_n, _ = midas_defense_forward_vulnerable(x_adv_n, traj, c_base, config, naive=False)
            if (classifier(def_n) > 0.5).item() != y_nat.item():
                ref_naive_succ += 1

        x_adv_a, _, _ = pgd_probe(
            x0, y_nat, classifier, traj, c_base, config,
            ref_alpha, probe_eps, ref_T, naive=False, sample_idx=i, log_steps=(i == 0)
        )
        with torch.no_grad():
            def_a, _ = midas_defense_forward_vulnerable(x_adv_a, traj, c_base, config, naive=False)
            if (classifier(def_a) > 0.5).item() != y_nat.item():
                ref_adaptive_succ += 1

    ref_asr_n = ref_naive_succ / N_SAMPLES
    ref_asr_a = ref_adaptive_succ / N_SAMPLES
    ref_gap = ref_asr_a - ref_asr_n

    print(f"\n  Naive ASR:   {ref_asr_n:.2%} ({ref_naive_succ}/{N_SAMPLES})")
    print(f"  Adaptive ASR:{ref_asr_a:.2%} ({ref_adaptive_succ}/{N_SAMPLES})")
    print(f"  Gap:         {ref_gap:+.2%}")


if __name__ == "__main__":
    main()
