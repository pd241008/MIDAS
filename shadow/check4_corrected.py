"""
Check #4 — Corrected trajectory generation + recalibrated gamma.

Changes from prior rounds:
1. Pool points: torch.rand(D) (in [0,1]^D, so project_Lp_ball clamping is a no-op)
2. Trajectory vectors: torch.rand(D) - 0.5 (zero-mean, satisfying Clean Traffic Axiom)
3. gamma = 0.292 (95th pct of corrected epsilon_p distribution)

Runs:
  A. Phase transition confirmation (delta_theta trace, sample 0)
  B. Full 9-config ASR table for SmoothMockModel (vulnerable basis)
  C. Full 9-config ASR table for SmoothMLPModel (vulnerable basis)
  D. alpha=0.001/T=500 probe for both surrogates
  E. Gate verdict
"""
import json
import math
import os
import torch
import torch.nn.functional as F

from shadow.defense import SmoothMockModel, SmoothMLPModel, penetration_epsilon_windowed, rotation_angle
from shadow.scratch.debug_gradients import vulnerable_basis


def project_Lp_ball(x, x0, epsilon):
    diff = x - x0
    diff = torch.clamp(diff, min=-epsilon, max=epsilon)
    return torch.clamp(x0 + diff, min=0.0, max=1.0)


def midas_defense_forward_vulnerable_with_eps_p(x_t, trajectory_window, c_base, config, naive=False):
    gamma = config.get("gamma", 0.292)
    lambda_ = config.get("lambda", 1.0)
    k = config.get("k", 2.0)
    delta_theta_max = config.get("delta_theta_max_deg", 45.0) * (3.141592653589793 / 180.0)

    full_window = trajectory_window + [x_t]
    epsilon_p = penetration_epsilon_windowed(full_window)
    theta = rotation_angle(epsilon_p, gamma, lambda_, k, delta_theta_max)

    if naive:
        theta = theta.detach()
        basis = vulnerable_basis(x_t.detach(), c_base)
    else:
        basis = vulnerable_basis(x_t, c_base)

    from shadow.defense import rotate_manifold_fixed_basis
    x_defended = rotate_manifold_fixed_basis(x_t, basis, theta)
    return x_defended, theta, epsilon_p


def pgd_attack(x0_init, y_target, classifier, traj, c_base, config,
               alpha, epsilon, steps, naive, log_steps=False):
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    step_logs = []
    saturate_step = None

    for step in range(steps):
        x_t.requires_grad_(True)
        window_cloned = [w.clone().detach() for w in traj]
        x_def, theta, eps_p = midas_defense_forward_vulnerable_with_eps_p(
            x_t, window_cloned, c_base, config, naive=naive
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

    return x_t.detach(), step_logs, saturate_step


def make_dataset(D, seed=42):
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    torch.manual_seed(seed)
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])
    return pool, calibration_batch


def run_check4_for_classifier(classifier_name, classifier, pool, config, D, N_SAMPLES=50):
    """Run full Check #4 for one classifier."""
    c_base = torch.zeros(D)

    with torch.no_grad():
        all_probs = torch.stack([classifier(x) for x in pool])
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    print(f"\n  Pool: {len(pool)} → After filter: {len(filtered)} → Using: {N_SAMPLES}")
    assert len(filtered) >= N_SAMPLES, f"Only {len(filtered)} survive"
    dataset = filtered[:N_SAMPLES]
    trajectories = [[torch.rand(D) - 0.5 for _ in range(config["W"] - 1)] for _ in range(N_SAMPLES)]

    # ---- PART A: Phase transition confirmation ----
    print(f"\n{'='*70}")
    print(f"PART A: PHASE TRANSITION CONFIRMATION — {classifier_name}")
    print(f"{'='*70}")

    x0 = dataset[0]
    traj = trajectories[0]
    with torch.no_grad():
        y_nat = (classifier(x0) > 0.5).float()

    for eps_label, eps in [("eps=0.1", 0.1), ("eps=0.2", 0.2)]:
        print(f"\n--- T=100, {eps_label}, ADAPTIVE ---")
        _, logs, _ = pgd_attack(
            x0, y_nat, classifier, traj, c_base, config,
            0.01, eps, 100, naive=False, log_steps=True
        )

        theta_vals = [l["theta_deg"] for l in logs]
        eps_p_vals = [l["eps_p"] for l in logs]
        at_ceiling = sum(1 for t in theta_vals if t >= 44.99)
        in_archimedian = sum(1 for t, e in zip(theta_vals, eps_p_vals) if e <= config["gamma"])

        print(f"\n  Summary: θ min={min(theta_vals):.3f}° max={max(theta_vals):.3f}° mean={sum(theta_vals)/len(theta_vals):.3f}°")
        print(f"           εp min={min(eps_p_vals):.6f} max={max(eps_p_vals):.6f} mean={sum(eps_p_vals)/len(eps_p_vals):.6f}")
        print(f"           Steps at ceiling: {at_ceiling}/{len(logs)}")
        print(f"           Steps in Archimedean phase: {in_archimedian}/{len(logs)}")

    # ---- PART B/C: Full 9-config ASR table ----
    print(f"\n{'='*70}")
    print(f"CHECK #4 — VULNERABLE BASIS: {classifier_name}")
    print(f"{'='*70}")

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

            x_adv_n, _, _ = pgd_attack(x0, y_nat, classifier, traj, c_base, config,
                                       alpha, eps, steps, naive=True, log_steps=False)
            with torch.no_grad():
                def_n, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_n, traj, c_base, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            x_adv_a, _, _ = pgd_attack(x0, y_nat, classifier, traj, c_base, config,
                                       alpha, eps, steps, naive=False, log_steps=False)
            with torch.no_grad():
                def_a, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_a, traj, c_base, config, naive=False)
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

    return results, dataset, trajectories


def main():
    torch.manual_seed(42)
    D = 10

    # Recalibrated config: gamma = 95th pct of corrected epsilon_p distribution (0.292)
    config = {
        "W": 4, "D": D, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
        "delta_theta_max_deg": 45.0, "tau": 0.3,
    }

    print("=" * 70)
    print("CHECK #4 — CORRECTED TRAJECTORY GENERATION")
    print("=" * 70)
    print(f"\n  gamma:       {config['gamma']}  (95th pct of corrected εp distribution)")
    print(f"  lambda:      {config['lambda']}")
    print(f"  k:           {config['k']}")
    print(f"  θ_max:       {config['delta_theta_max_deg']}°")
    print(f"  Pool:        torch.rand(D)  (in [0,1]^D, clamping no-op)")
    print(f"  Trajectory:  torch.rand(D)-0.5  (zero-mean, Clean Traffic Axiom)")

    pool, calibration_batch = make_dataset(D)

    # ================================================================
    # SMOOTH MOCK MODEL (linear)
    # ================================================================
    classifier_mock = SmoothMockModel(d=D, calibration_batch=calibration_batch)
    mock_results, mock_dataset, mock_trajectories = run_check4_for_classifier(
        "SmoothMockModel (linear)", classifier_mock, pool, config, D
    )

    # ================================================================
    # SMOOTH MLP MODEL (16-unit tanh)
    # ================================================================
    classifier_mlp = SmoothMLPModel(d=D, hidden=16, calibration_batch=calibration_batch, seed=1337)
    mlp_results, mlp_dataset, mlp_trajectories = run_check4_for_classifier(
        "SmoothMLPModel (16-unit tanh)", classifier_mlp, pool, config, D
    )

    # ================================================================
    # PART D: alpha=0.001/T=500 probe — both surrogates
    # ================================================================
    print(f"\n{'='*70}")
    print("PART D: alpha=0.001, T=500, eps=0.1 — BOTH SURROGATES")
    print(f"{'='*70}")

    probe_alpha = 0.001
    probe_eps = 0.1
    probe_T = 500
    c_base = torch.zeros(D)

    for cls_name, classifier, dataset, trajectories in [
        ("SmoothMockModel", classifier_mock, mock_dataset, mock_trajectories),
        ("SmoothMLPModel", classifier_mlp, mlp_dataset, mlp_trajectories),
    ]:
        print(f"\n--- {cls_name} ---")
        naive_succ = 0
        adaptive_succ = 0
        naive_sat = []
        adaptive_sat = []

        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            x_adv_n, logs_n, sat_n = pgd_attack(
                x0, y_nat, classifier, traj, c_base, config,
                probe_alpha, probe_eps, probe_T, naive=True, log_steps=(i == 0))
            if sat_n is not None:
                naive_sat.append(sat_n)
            with torch.no_grad():
                def_n, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_n, traj, c_base, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

            x_adv_a, logs_a, sat_a = pgd_attack(
                x0, y_nat, classifier, traj, c_base, config,
                probe_alpha, probe_eps, probe_T, naive=False, log_steps=(i == 0))
            if sat_a is not None:
                adaptive_sat.append(sat_a)
            with torch.no_grad():
                def_a, _, _ = midas_defense_forward_vulnerable_with_eps_p(x_adv_a, traj, c_base, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
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

    # ================================================================
    # PART E: Gate verdict
    # ================================================================
    print(f"\n{'='*70}")
    print("GATE VERDICT")
    print(f"{'='*70}")

    for cls_name, results in [("SmoothMockModel", mock_results), ("SmoothMLPModel", mlp_results)]:
        max_gap_name = max(results, key=lambda k: abs(results[k]["gap"]))
        max_gap = results[max_gap_name]["gap"]
        max_gap_se = results[max_gap_name]["gap_se"]
        gate_pass = abs(max_gap) > 0.02 and max_gap_se > 1.0

        print(f"\n  {cls_name}:")
        print(f"    Largest |gap|: {abs(max_gap):.0%} at {max_gap_name}  (Gap/SE={max_gap_se:.2f})")
        print(f"    Gate: {'PASSED' if gate_pass else 'FAILED'}")
        if not gate_pass:
            reasons = []
            if abs(max_gap) <= 0.02:
                reasons.append(f"|gap|={abs(max_gap):.0%} <= 2%")
            if max_gap_se <= 1.0:
                reasons.append(f"Gap/SE={max_gap_se:.2f} <= 1")
            print(f"    Reasons: {'; '.join(reasons)}")


if __name__ == "__main__":
    main()
