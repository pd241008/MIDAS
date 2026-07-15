"""
Diagnostic re-run of Check #4: vulnerable (attacker-coupled) basis only.

Outputs:
  1. Sample set sizes (pre-filter pool, post-filter attack set)
  2. Full 3x3 ASR table for naive and adaptive on vulnerable basis
  3. Per-step loss and ||d(loss)/d(delta_theta)|| for first sample, T=100, eps=0.2
  4. Per-step loss and ||d(loss)/d(delta_theta)|| for first sample, T=100, eps=0.05
  5. Saturation check: final perturbation norm vs epsilon at step 5/10/20/50/100
"""
import json
import torch
import torch.nn.functional as F
import os

from shadow.defense import SmoothMockModel
from shadow.scratch.debug_gradients import (
    midas_defense_forward_vulnerable,
    vulnerable_basis,
)


def log_confidence_distribution(classifier, samples, label=""):
    with torch.no_grad():
        probs = torch.stack([classifier(x) for x in samples])
    nat_probs = torch.where(probs > 0.5, probs, 1.0 - probs)
    print(f"  [{label}] Natural confidence: "
          f"min={nat_probs.min():.4f}  median={nat_probs.median():.4f}  "
          f"max={nat_probs.max():.4f}  mean={nat_probs.mean():.4f}  "
          f"frac>0.95={(nat_probs > 0.95).float().mean():.2%}")
    return nat_probs


def project_Lp_ball(x, x0, epsilon):
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    return torch.clamp(x0 + diff, min=0.0, max=1.0)


def pgd_vuln_diagnostic(x0_init, y_target, classifier, traj, c_base, config,
                         alpha, epsilon, steps, naive, sample_idx, log_steps=True):
    """PGD on vulnerable basis with full per-step logging."""
    from shadow.attacks import project_Lp_ball

    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    step_logs = []

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

        step_logs.append({
            "step": step,
            "loss": loss.item(),
            "grad_theta_norm": grad_theta_norm,
            "grad_x_norm": torch.norm(grad).item(),
            "pert_linf": pert_norm,
        })

        if log_steps and step < 30 or step % 10 == 0 or step == steps - 1:
            tag = "ADAPT" if not naive else "NAIVE "
            print(f"    [{tag} s={step:3d}] loss={loss.item():.6f}  "
                  f"||dL/dθ||={grad_theta_norm:.8f}  "
                  f"||dL/dx||={torch.norm(grad).item():.6f}  "
                  f"||Δx||∞={pert_norm:.6f}")

        with torch.no_grad():
            x_t = x_t_adv

    return x_t.detach(), step_logs


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

    # ---- Calibration + filtering ----
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])
    classifier = SmoothMockModel(d=D, calibration_batch=calibration_batch)

    print("=" * 70)
    print("SAMPLE SET DIAGNOSTICS")
    print("=" * 70)
    log_confidence_distribution(classifier, pool, "pre-calib full pool")

    with torch.no_grad():
        all_probs = torch.stack([classifier(x) for x in pool])
        nat_labels = (all_probs > 0.5).float()
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    print(f"\n  Pool size: {len(pool)}")
    print(f"  After confidence filter (<=0.95): {len(filtered)}")
    print(f"  Removed by filter: {len(pool) - len(filtered)}")
    print(f"  N_SAMPLES (attack set): {N_SAMPLES}")
    print(f"  Fraction of pool used: {N_SAMPLES}/{len(pool)} = {N_SAMPLES/len(pool):.1%}")

    if len(filtered) < N_SAMPLES:
        print(f"\n  *** FATAL: Only {len(filtered)} samples survive, need {N_SAMPLES} ***")
        return

    dataset = filtered[:N_SAMPLES]
    trajectories = [
        [torch.rand(D) for _ in range(config["W"] - 1)] for _ in range(N_SAMPLES)
    ]

    print("\n--- Post-calibration confidence distribution (attack set) ---")
    log_confidence_distribution(classifier, dataset, "post-calib attack set")

    # ================================================================
    # PART 1: Full 3x3 ASR table on vulnerable basis
    # ================================================================
    print("\n" + "=" * 70)
    print("CHECK #4 — VULNERABLE BASIS: FULL 3x3 ASR TABLE")
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

            # Naive
            x_adv_n, _ = pgd_vuln_diagnostic(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, eps, steps, naive=True, sample_idx=i, log_steps=False
            )
            with torch.no_grad():
                def_n, _ = midas_defense_forward_vulnerable(x_adv_n, traj, c_base, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            # Adaptive
            x_adv_a, _ = pgd_vuln_diagnostic(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, eps, steps, naive=False, sample_idx=i, log_steps=False
            )
            with torch.no_grad():
                def_a, _ = midas_defense_forward_vulnerable(x_adv_a, traj, c_base, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        gap = asr_a - asr_n
        results[name] = {"naive": asr_n, "adaptive": asr_a, "gap": gap}
        print(f"  {name:20s}  Naive={asr_n:.0%}  Adaptive={asr_a:.0%}  Gap={gap:+.0%}")

    # ================================================================
    # PART 2: Per-step logs for sample 0, T=100
    # ================================================================
    print("\n" + "=" * 70)
    print("PER-STEP LOGS — SAMPLE 0, VULNERABLE BASIS")
    print("=" * 70)

    x0 = dataset[0]
    traj = trajectories[0]
    with torch.no_grad():
        y_nat = (classifier(x0) > 0.5).float()

    for eps_label, eps in [("eps=0.2", 0.2), ("eps=0.05", 0.05)]:
        print(f"\n--- T=100, {eps_label}, NAIVE ---")
        _, naive_logs = pgd_vuln_diagnostic(
            x0, y_nat, classifier, traj, c_base, config,
            0.01, eps, 100, naive=True, sample_idx=0, log_steps=True
        )

        print(f"\n--- T=100, {eps_label}, ADAPTIVE ---")
        _, adapt_logs = pgd_vuln_diagnostic(
            x0, y_nat, classifier, traj, c_base, config,
            0.01, eps, 100, naive=False, sample_idx=0, log_steps=True
        )

    # ================================================================
    # PART 3: Saturation check — does perturbation stop growing?
    # ================================================================
    print("\n" + "=" * 70)
    print("SATURATION CHECK — perturbation norm at key steps")
    print("=" * 70)

    for eps_label, eps in [("eps=0.05", 0.05), ("eps=0.1", 0.1), ("eps=0.2", 0.2)]:
        print(f"\n--- {eps_label}, ADAPTIVE ---")
        _, logs = pgd_vuln_diagnostic(
            x0, y_nat, classifier, traj, c_base, config,
            0.01, eps, 100, naive=False, sample_idx=0, log_steps=False
        )
        key_steps = [0, 4, 9, 19, 49, 99]
        for s in key_steps:
            if s < len(logs):
                l = logs[s]
                print(f"  step={l['step']:3d}  ||Δx||∞={l['pert_linf']:.6f}  "
                      f"loss={l['loss']:.6f}  ||dL/dθ||={l['grad_theta_norm']:.8f}")

    # ================================================================
    # GATE VERDICT
    # ================================================================
    print("\n" + "=" * 70)
    print("GATE VERDICT")
    print("=" * 70)
    max_gap_name = max(results, key=lambda k: abs(results[k]["gap"]))
    max_gap = results[max_gap_name]["gap"]
    print(f"  Largest |gap|: {abs(max_gap):.0%} at {max_gap_name}")
    print(f"  Gate threshold: >2% on any config")
    if abs(max_gap) > 0.02:
        print(f"  GATE: PASSED (technically, {max_gap_name} shows {max_gap:+.0%})")
    else:
        print(f"  GATE: FAILED (no config exceeds 2% gap)")
    print(f"  Qualitative assessment: {'Real gap' if abs(max_gap) > 0.05 else 'Marginal or absent gap'}")


if __name__ == "__main__":
    main()
