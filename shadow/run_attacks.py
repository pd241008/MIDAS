import json
import torch
import torch.nn.functional as F
import os
from shadow.defense import SmoothMockModel, midas_defense_forward
from shadow.attacks import naive_pgd_attack, adaptive_pgd_attack, project_Lp_ball
from shadow.report import generate_report
from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable


def log_confidence_distribution(classifier, samples, label=""):
    """Log the natural-class confidence distribution for a set of samples."""
    with torch.no_grad():
        probs = torch.stack([classifier(x) for x in samples])
    nat_probs = torch.where(probs > 0.5, probs, 1.0 - probs)
    print(f"  [{label}] Natural confidence: "
          f"min={nat_probs.min():.4f}  median={nat_probs.median():.4f}  "
          f"max={nat_probs.max():.4f}  mean={nat_probs.mean():.4f}  "
          f"frac>0.95={(nat_probs > 0.95).float().mean():.2%}")
    return nat_probs


def pgd_vuln(x0_init, y_target, classifier, traj, c_base, config,
             alpha, epsilon, steps, naive, first_sample_flag):
    """Vulnerable-basis (attacker-coupled) PGD attack — sign-based, matching fixed-basis."""
    from shadow.attacks import project_Lp_ball

    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    for step in range(steps):
        x_t.requires_grad_(True)
        window_cloned = [w.clone().detach() for w in traj]
        x_def, theta_vuln = midas_defense_forward_vulnerable(x_t, window_cloned, c_base, config, naive=naive)
        if not naive:
            theta_vuln.retain_grad()
        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y_target.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        if first_sample_flag:
            if not naive and theta_vuln.grad is not None:
                grad_theta_norm = torch.norm(theta_vuln.grad).item()
            else:
                grad_theta_norm = 0.0
            tag = "VULN-ADAPT" if not naive else "VULN-NAIVE"
            print(f"    [{tag} Step {step:3d}] ||d(loss)/d(delta_theta)|| = {grad_theta_norm:.8f}  loss = {loss.item():.6f}")

        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0_init, epsilon)

    return x_t.detach()


def evaluate_attack_set(classifier, dataset, trajectories, c_base, basis, config,
                        attacks, N_SAMPLES, mode="vuln"):
    """
    Run a full attack sweep (all epsilon/T combos) for either vulnerable or fixed basis.
    Returns results dict. Prints per-config summary.
    """
    results = {}

    for name, alpha, steps, eps in attacks:
        print(f"\n{'='*60}")
        print(f"Evaluating {name}  [{mode.upper()} BASIS]")
        print(f"{'='*60}")

        naive_succ = 0
        adaptive_succ = 0

        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            is_first = (i == 0)

            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            if mode == "vuln":
                x_adv_n = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                    alpha, eps, steps, naive=True, first_sample_flag=is_first)
                def_n, _ = midas_defense_forward_vulnerable(x_adv_n, traj, c_base, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

                if is_first:
                    print()
                x_adv_a = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                    alpha, eps, steps, naive=False, first_sample_flag=is_first)
                def_a, _ = midas_defense_forward_vulnerable(x_adv_a, traj, c_base, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1
            else:
                if is_first:
                    print("\n  --- Fixed Basis: Naive ---")
                x_adv_n = naive_pgd_attack(x0, y_nat, classifier, traj, c_base, basis, config,
                                           alpha, eps, steps, first_sample=is_first)
                def_n, _ = midas_defense_forward(x_adv_n, traj, c_base, basis, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

                if is_first:
                    print("\n  --- Fixed Basis: Adaptive ---")
                x_adv_a = adaptive_pgd_attack(x0, y_nat, classifier, traj, c_base, basis, config,
                                              alpha, eps, steps, first_sample=is_first)
                def_a, _ = midas_defense_forward(x_adv_a, traj, c_base, basis, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        results[name] = {"naive": asr_n, "adaptive": asr_a}
        gap = asr_a - asr_n
        print(f"\n  Naive ASR: {asr_n:.2%}  Adaptive ASR: {asr_a:.2%}  Gap: {gap:+.2%}"
              f"  {'✓ real gap' if abs(gap) > 0.02 else '— flat'}")

    return results


def run_all_attacks():
    print("Loading basis from results/basis.json...")
    if not os.path.exists("results/basis.json"):
        raise RuntimeError("results/basis.json not found! Run the Rust harness first to export the basis.")

    with open("results/basis.json", "r") as f:
        basis_data = json.load(f)

    c_base = torch.tensor(basis_data["c_base"], dtype=torch.float32)
    basis = torch.tensor(basis_data["basis"], dtype=torch.float32)

    config = {
        "W": 4,
        "D": 10,
        "gamma": 0.5,
        "lambda": 1.0,
        "k": 2.0,
        "delta_theta_max_deg": 45.0,
        "tau": 0.3,
        "sla_budget_ms": 10,
        "channel_capacity": 4
    }

    D = config["D"]

    # ---- Calibration + sample-set construction ----
    # Generate a pool of candidate points; calibration batch is drawn from the same pool.
    torch.manual_seed(42)
    CALIBRATION_POOL = 200
    N_SAMPLES = 50
    pool = [torch.rand(D) for _ in range(CALIBRATION_POOL)]
    calibration_batch = torch.stack(pool[:100])

    classifier = SmoothMockModel(d=D, calibration_batch=calibration_batch)

    # Pre-calibration confidence distribution
    print("\n--- Pre-calibration confidence distribution (full pool) ---")
    log_confidence_distribution(classifier, pool, "pre-calib full pool")

    # Filter: keep only correctly-classified, non-trivially-overconfident points
    with torch.no_grad():
        all_probs = torch.stack([classifier(x) for x in pool])
        nat_labels = (all_probs > 0.5).float()
        nat_conf = torch.where(all_probs > 0.5, all_probs, 1.0 - all_probs)
        # Keep points where natural confidence <= 0.95 (not trivially overconfident)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered = [pool[i] for i in range(len(pool)) if keep_mask[i]]

    print(f"\n  Pool size: {len(pool)}  →  After confidence filter (≤0.95): {len(filtered)}")

    if len(filtered) < N_SAMPLES:
        raise RuntimeError(f"Only {len(filtered)} samples survive the confidence filter, "
                           f"need {N_SAMPLES}. Reduce N_SAMPLES or increase CALIBRATION_POOL.")

    dataset = filtered[:N_SAMPLES]
    trajectories = [
        [torch.randn(D) for _ in range(config["W"] - 1)] for _ in range(N_SAMPLES)
    ]

    # Post-calibration confidence distribution on the actual attack set
    print("\n--- Post-calibration confidence distribution (attack set) ---")
    nat_probs = log_confidence_distribution(classifier, dataset, "post-calib attack set")

    # ---- Paper's declared epsilon × T sweep ----
    attacks = [
        # (name, alpha, steps, epsilon)
        ("PGD T=20  eps=0.05", 0.01, 20, 0.05),
        ("PGD T=50  eps=0.05", 0.01, 50, 0.05),
        ("PGD T=100 eps=0.05", 0.01, 100, 0.05),
        ("PGD T=20  eps=0.1",  0.01, 20, 0.1),
        ("PGD T=50  eps=0.1",  0.01, 50, 0.1),
        ("PGD T=100 eps=0.1",  0.01, 100, 0.1),
        ("PGD T=20  eps=0.2",  0.01, 20, 0.2),
        ("PGD T=50  eps=0.2",  0.01, 50, 0.2),
        ("PGD T=100 eps=0.2",  0.01, 100, 0.2),
    ]

    # ================================================================
    # STEP 2: Check #4 — vulnerable (attacker-coupled) basis first
    # ================================================================
    print(f"\n{'#'*60}")
    print("# CHECK #4 — Vulnerable (attacker-coupled) basis")
    print(f"{'#'*60}")
    vuln_results = evaluate_attack_set(
        classifier, dataset, trajectories, c_base, basis, config,
        attacks, N_SAMPLES, mode="vuln"
    )

    # Gate: check if any config shows a real naive-vs-adaptive gap
    has_gap = any(
        abs(v["adaptive"] - v["naive"]) > 0.02
        for v in vuln_results.values()
    )

    if not has_gap:
        print("\n" + "="*60)
        print("GATE FAILED: No meaningful naive-vs-adaptive gap on vulnerable basis.")
        print("Do NOT proceed to fixed-basis evaluation.")
        print("="*60)
        return

    # ================================================================
    # STEP 4: Fixed-basis evaluation (only if gap was found)
    # ================================================================
    print(f"\n{'#'*60}")
    print("# FIXED-BASIS evaluation (gap confirmed on vulnerable basis)")
    print(f"{'#'*60}")
    fixed_results = evaluate_attack_set(
        classifier, dataset, trajectories, c_base, basis, config,
        attacks, N_SAMPLES, mode="fixed"
    )

    # ---- Report generation ----
    # Merge results for the report (only PGD eps=0.1 for the paper table)
    report_results = {}
    for key in ["PGD T=20  eps=0.1", "PGD T=50  eps=0.1", "PGD T=100 eps=0.1"]:
        if key in fixed_results:
            report_results[key] = fixed_results[key]

    print("\nGenerating report...")
    try:
        generate_report(report_results, "results/defense_success_rate.json", "results/adaptive_defense_success_rate.json")
        print("Wrote results/adaptive_defense_success_rate.json!")
    except Exception as e:
        print(f"Skipping JSON report generation: {e}")


if __name__ == "__main__":
    run_all_attacks()
