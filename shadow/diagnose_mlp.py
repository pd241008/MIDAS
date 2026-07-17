"""
MLP Surrogate — Check #4 vulnerable-basis sweep + alpha=0.001/T=500 probe.

Uses SmoothMLPModel (fixed-weight 2-layer MLP) instead of SmoothMockModel.
Same seeds, same pool construction, same filtering — results are directly comparable.
"""
import json
import math
import torch
import torch.nn.functional as F
import os

from shadow.defense import SmoothMLPModel
from shadow.scratch.debug_gradients import (
    midas_defense_forward_vulnerable,
    vulnerable_basis,
)


def project_Lp_ball(x, x0, epsilon):
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    return torch.clamp(x0 + diff, min=0.0, max=1.0)


def pgd_vuln(x0_init, y_target, classifier, traj, c_base, config,
             alpha, epsilon, steps, naive, log_steps=False):
    """PGD on vulnerable basis with optional per-step logging."""
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
        x_def, theta_vuln = midas_defense_forward_vulnerable(
            x_t, window_slice, c_base, config, naive=naive
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
            "step": step, "loss": loss.item(),
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
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W - 1):]
    return x_t.detach(), step_logs, saturate_step, final_window


def make_dataset(D, seed=42):
    """Same pool/calibration/filtering as every prior script."""
    CALIBRATION_POOL = 200
    N_SAMPLES = 50

    torch.manual_seed(seed)
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])
    classifier = SmoothMLPModel(d=D, hidden=16, calibration_batch=calibration_batch, seed=1337)

    with torch.no_grad():
        all_probs = torch.stack([classifier(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    print(f"  Pool size: {len(pool)}")
    print(f"  After confidence filter (<=0.95): {len(filtered)}")
    print(f"  N_SAMPLES: {N_SAMPLES}")

    assert len(filtered) >= N_SAMPLES, f"Only {len(filtered)} survive"

    dataset = filtered[:N_SAMPLES]
    trajectories = [
        [torch.rand(D) - 0.5 for _ in range(3)] for _ in range(N_SAMPLES)
    ]

    return classifier, dataset, trajectories, N_SAMPLES


def log_confidence(classifier, samples, label=""):
    with torch.no_grad():
        probs = torch.stack([classifier(x) for x in samples])
    nat_probs = torch.where(probs > 0.5, probs, 1.0 - probs)
    print(f"  [{label}] min={nat_probs.min():.4f}  median={nat_probs.median():.4f}  "
          f"max={nat_probs.max():.4f}  mean={nat_probs.mean():.4f}  "
          f"frac>0.95={(nat_probs > 0.95).float().mean():.2%}")


def main():
    config = {
        "W": 4, "D": 10, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
        "delta_theta_max_deg": 45.0, "tau": 0.3,
        "sla_budget_ms": 10, "channel_capacity": 4,
    }
    D = config["D"]

    if not os.path.exists("results/basis.json"):
        raise RuntimeError("results/basis.json not found!")
    with open("results/basis.json") as f:
        basis_data = json.load(f)
    c_base = torch.tensor(basis_data["c_base"], dtype=torch.float32)

    classifier, dataset, trajectories, N_SAMPLES = make_dataset(D)

    print("\n--- Post-calibration confidence (attack set) ---")
    log_confidence(classifier, dataset, "MLP attack set")

    # ================================================================
    # PART 1: Full 3x3 ASR table — vulnerable basis, MLP surrogate
    # ================================================================
    print("\n" + "=" * 70)
    print("CHECK #4 — VULNERABLE BASIS: MLP SURROGATE (3x3 ASR TABLE)")
    print("=" * 70)

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
                y_nat = (classifier(x0) > 0.5).float()

            x_adv_n, _, _, win_n = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                       alpha, eps, steps, naive=True, log_steps=False)
            with torch.no_grad():
                def_n, _ = midas_defense_forward_vulnerable(x_adv_n, win_n, c_base, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            x_adv_a, _, _, win_a = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                       alpha, eps, steps, naive=False, log_steps=False)
            with torch.no_grad():
                def_a, _ = midas_defense_forward_vulnerable(x_adv_a, win_a, c_base, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
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

    # ================================================================
    # PART 2: Per-step logs — sample 0, T=100, eps=0.2 and eps=0.1
    # ================================================================
    print("\n" + "=" * 70)
    print("PER-STEP LOGS — SAMPLE 0, VULNERABLE BASIS, MLP SURROGATE")
    print("=" * 70)

    x0 = dataset[0]
    traj = trajectories[0]
    with torch.no_grad():
        y_nat = (classifier(x0) > 0.5).float()

    for eps_label, eps in [("eps=0.2", 0.2), ("eps=0.1", 0.1)]:
        print(f"\n--- T=100, {eps_label}, NAIVE ---")
        pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                 0.01, eps, 100, naive=True, log_steps=True)

        print(f"\n--- T=100, {eps_label}, ADAPTIVE ---")
        pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                 0.01, eps, 100, naive=False, log_steps=True)

    # ================================================================
    # PART 3: Saturation check
    # ================================================================
    print("\n" + "=" * 70)
    print("SATURATION CHECK — MLP SURROGATE")
    print("=" * 70)

    for eps_label, eps in [("eps=0.05", 0.05), ("eps=0.1", 0.1), ("eps=0.2", 0.2)]:
        print(f"\n--- {eps_label}, ADAPTIVE ---")
        _, logs, sat, _ = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                0.01, eps, 100, naive=False, log_steps=False)
        key_steps = [0, 4, 9, 19, 49, 99]
        for s in key_steps:
            if s < len(logs):
                l = logs[s]
                print(f"  step={l['step']:3d}  ||\u0394x||\u221e={l['pert_linf']:.6f}  "
                      f"loss={l['loss']:.6f}  ||dL/d\u03b8||={l['grad_theta_norm']:.8f}")
        print(f"  Saturation step: {sat}")

    # ================================================================
    # GATE VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("GATE VERDICT — MLP SURROGATE")
    print("=" * 70)
    max_gap_name = max(results, key=lambda k: abs(results[k]["gap"]))
    max_gap = results[max_gap_name]["gap"]
    max_gap_se = results[max_gap_name]["gap_se"]
    print(f"  Largest |gap|: {abs(max_gap):.0%} at {max_gap_name}  (Gap/SE={max_gap_se:.2f})")
    print(f"  Gate threshold: >2% on any config AND Gap/SE > 1")
    gate_pass = abs(max_gap) > 0.02 and max_gap_se > 1.0
    if gate_pass:
        print(f"  GATE: PASSED")
    else:
        print(f"  GATE: FAILED")
        reasons = []
        if abs(max_gap) <= 0.02:
            reasons.append(f"|gap|={abs(max_gap):.0%} <= 2%")
        if max_gap_se <= 1.0:
            reasons.append(f"Gap/SE={max_gap_se:.2f} <= 1")
        print(f"  Reasons: {'; '.join(reasons)}")

    # ================================================================
    # PART 4: alpha=0.001, T=500 probe — vulnerable basis
    # ================================================================
    print("\n" + "=" * 70)
    print("PROBE: eps=0.1, alpha=0.001, T=500 — MLP SURROGATE")
    print("=" * 70)

    probe_alpha = 0.001
    probe_eps = 0.1
    probe_T = 500

    naive_succ = 0
    adaptive_succ = 0
    naive_sat_steps = []
    adaptive_sat_steps = []

    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()

        x_adv_n, logs_n, sat_n, win_n = pgd_vuln(
            x0, y_nat, classifier, traj, c_base, config,
            probe_alpha, probe_eps, probe_T, naive=True, log_steps=(i == 0))
        if sat_n is not None:
            naive_sat_steps.append(sat_n)
        with torch.no_grad():
            def_n, _ = midas_defense_forward_vulnerable(x_adv_n, win_n, c_base, config, naive=False)
            if (classifier(def_n) > 0.5).item() != y_nat.item():
                naive_succ += 1

        x_adv_a, logs_a, sat_a, win_a = pgd_vuln(
            x0, y_nat, classifier, traj, c_base, config,
            probe_alpha, probe_eps, probe_T, naive=False, log_steps=(i == 0))
        if sat_a is not None:
            adaptive_sat_steps.append(sat_a)
        with torch.no_grad():
            def_a, _ = midas_defense_forward_vulnerable(x_adv_a, win_a, c_base, config, naive=False)
            if (classifier(def_a) > 0.5).item() != y_nat.item():
                adaptive_succ += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{N_SAMPLES}] naive={naive_succ/(i+1):.0%} adaptive={adaptive_succ/(i+1):.0%}")

    asr_n = naive_succ / N_SAMPLES
    asr_a = adaptive_succ / N_SAMPLES
    gap = asr_a - asr_n

    se_n = math.sqrt(asr_n * (1 - asr_n) / N_SAMPLES)
    se_a = math.sqrt(asr_a * (1 - asr_a) / N_SAMPLES)
    se_gap = math.sqrt(se_n**2 + se_a**2)
    gap_se = gap / se_gap if se_gap > 0 else float('inf')

    print(f"\n--- PROBE RESULTS ---")
    print(f"  Naive ASR:    {asr_n:.2%} ({naive_succ}/{N_SAMPLES})")
    print(f"  Adaptive ASR: {asr_a:.2%} ({adaptive_succ}/{N_SAMPLES})")
    print(f"  Gap:          {gap:+.2%}")
    print(f"  SE(Gap):      {se_gap:.4f}")
    print(f"  Gap/SE:       {gap_se:.2f}")

    print(f"\n--- SATURATION (probe) ---")
    if naive_sat_steps:
        print(f"  Naive:   median={sorted(naive_sat_steps)[len(naive_sat_steps)//2]}  "
              f"range=[{min(naive_sat_steps)},{max(naive_sat_steps)}]  "
              f"frac={len(naive_sat_steps)}/{N_SAMPLES}")
    else:
        print(f"  Naive:   NO samples saturated")
    if adaptive_sat_steps:
        print(f"  Adaptive: median={sorted(adaptive_sat_steps)[len(adaptive_sat_steps)//2]}  "
              f"range=[{min(adaptive_sat_steps)},{max(adaptive_sat_steps)}]  "
              f"frac={len(adaptive_sat_steps)}/{N_SAMPLES}")
    else:
        print(f"  Adaptive: NO samples saturated")

    max_steps_to_saturate_001 = math.ceil(probe_eps / probe_alpha)
    print(f"  Theoretical max steps to saturate (eps/alpha): {max_steps_to_saturate_001}")
    print(f"  Post-saturation steps (T=500): {probe_T - max_steps_to_saturate_001}")

    # ================================================================
    # REFERENCE: alpha=0.01, T=100, eps=0.1 — same probe framework
    # ================================================================
    print("\n" + "=" * 70)
    print("REFERENCE: alpha=0.01, T=100, eps=0.1 — MLP SURROGATE")
    print("=" * 70)

    ref_naive_succ = 0
    ref_adaptive_succ = 0

    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()

        x_adv_n, _, _, win_n = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                   0.01, probe_eps, 100, naive=True, log_steps=False)
        with torch.no_grad():
            def_n, _ = midas_defense_forward_vulnerable(x_adv_n, win_n, c_base, config, naive=False)
            if (classifier(def_n) > 0.5).item() != y_nat.item():
                ref_naive_succ += 1

        x_adv_a, _, _, win_a = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                   0.01, probe_eps, 100, naive=False, log_steps=False)
        with torch.no_grad():
            def_a, _ = midas_defense_forward_vulnerable(x_adv_a, win_a, c_base, config, naive=False)
            if (classifier(def_a) > 0.5).item() != y_nat.item():
                ref_adaptive_succ += 1

    ref_asr_n = ref_naive_succ / N_SAMPLES
    ref_asr_a = ref_adaptive_succ / N_SAMPLES
    ref_gap = ref_asr_a - ref_asr_n

    print(f"  Naive ASR:    {ref_asr_n:.2%} ({ref_naive_succ}/{N_SAMPLES})")
    print(f"  Adaptive ASR: {ref_asr_a:.2%} ({ref_adaptive_succ}/{N_SAMPLES})")
    print(f"  Gap:          {ref_gap:+.2%}")


if __name__ == "__main__":
    main()
