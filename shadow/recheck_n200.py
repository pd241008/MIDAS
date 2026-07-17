"""
N=200 Re-check: Two borderline configs from check_trained_surrogate.py.

Configs:
  1. Sign-based PGD, T=20, epsilon=0.2 (N=50 result: Naive=18%, Adaptive=22%, Gap/SE=0.50)
  2. Continuous gradient, T=50, epsilon=0.2 (N=50 result: Naive=22%, Adaptive=16%, Gap/SE=-0.77)

Uses TrainedSurrogateModel, vulnerable basis, fresh N=200 pool.
"""
import math
import torch
import torch.nn.functional as F

from shadow.defense import (
    TrainedSurrogateModel, train_surrogate,
    penetration_epsilon_windowed, rotation_angle,
    rotate_manifold_fixed_basis, midas_defense_forward,
)
from shadow.check_trained_surrogate import (
    vulnerable_basis, project_Lp_ball,
    pgd_attack, pgd_attack_continuous,
    midas_defense_forward_vulnerable_with_eps_p,
    D, HIDDEN, TRAIN_SEED, TRAIN_EPOCHS, TRAIN_LR, TRAIN_N,
)
from shadow.save_results import save_results


CONFIG = {
    "W": 4, "D": D, "gamma": 0.292, "lambda": 1.0, "k": 2.0,
    "delta_theta_max_deg": 45.0, "tau": 0.3,
}

N_SAMPLES = 200
POOL_SIZE = 500


def run_single_config(model, dataset, trajectories, config, D, N_SAMPLES,
                      attack_fn, alpha, epsilon, steps, update_rule_label):
    """Run one config across the full dataset, return naive/adaptive ASR."""
    c_base = torch.zeros(D)
    naive_succ = 0
    adaptive_succ = 0

    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        with torch.no_grad():
            y_nat = (model(x0) > 0.5).float()

        x_adv_n, _, _, win_n = attack_fn(
            x0, y_nat, model, traj, c_base, config,
            alpha, epsilon, steps, naive=True, log_steps=False)
        with torch.no_grad():
            def_n, _, _ = midas_defense_forward_vulnerable_with_eps_p(
                x_adv_n, win_n, c_base, config, naive=False)
            if (model(def_n) > 0.5).item() != y_nat.item():
                naive_succ += 1

        x_adv_a, _, _, win_a = attack_fn(
            x0, y_nat, model, traj, c_base, config,
            alpha, epsilon, steps, naive=False, log_steps=False)
        with torch.no_grad():
            def_a, _, _ = midas_defense_forward_vulnerable_with_eps_p(
                x_adv_a, win_a, c_base, config, naive=False)
            if (model(def_a) > 0.5).item() != y_nat.item():
                adaptive_succ += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_SAMPLES}] naive={naive_succ/(i+1):.1%} "
                  f"adaptive={adaptive_succ/(i+1):.1%}")

    asr_n = naive_succ / N_SAMPLES
    asr_a = adaptive_succ / N_SAMPLES
    gap = asr_a - asr_n
    se_n = math.sqrt(asr_n * (1 - asr_n) / N_SAMPLES) if N_SAMPLES > 0 else 0
    se_a = math.sqrt(asr_a * (1 - asr_a) / N_SAMPLES) if N_SAMPLES > 0 else 0
    se_gap = math.sqrt(se_n**2 + se_a**2)
    gap_se = gap / se_gap if se_gap > 0 else float('inf')

    return {
        "naive_asr": asr_n,
        "adaptive_asr": asr_a,
        "n": N_SAMPLES,
        "gap": gap,
        "gap_se": gap_se,
        "naive_count": naive_succ,
        "adaptive_count": adaptive_succ,
    }


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("N=200 RE-CHECK — TWO BORDERLINE CONFIGS")
    print("=" * 70)
    print(f"  Config 1: Sign-based PGD, T=20, eps=0.2")
    print(f"  Config 2: Continuous gradient, T=50, eps=0.2")
    print(f"  N_SAMPLES: {N_SAMPLES}")
    print(f"  POOL_SIZE: {POOL_SIZE}")

    # ---- Train surrogate ----
    print("\n" + "=" * 70)
    print("TRAINING SURROGATE")
    print("=" * 70)
    import os
    save_dir = os.path.join(os.path.dirname(__file__), "trained_weights")
    model, train_log = train_surrogate(
        d=D, hidden=HIDDEN, seed=TRAIN_SEED, n_epochs=TRAIN_EPOCHS,
        lr=TRAIN_LR, n_train=TRAIN_N, save_dir=save_dir
    )

    # ---- Generate larger pool, filter, take N=200 ----
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
    trajectories = [[torch.rand(D) - 0.5 for _ in range(CONFIG["W"] - 1)]
                    for _ in range(N_SAMPLES)]

    # ---- Config 1: Sign-based PGD, T=20, eps=0.2 ----
    print(f"\n{'='*70}")
    print("CONFIG 1: SIGN-BASED PGD, T=20, EPS=0.2")
    print(f"{'='*70}")
    r1 = run_single_config(model, dataset, trajectories, CONFIG, D, N_SAMPLES,
                           pgd_attack, alpha=0.01, epsilon=0.2, steps=20,
                           update_rule_label="sign")
    print(f"\n  Naive ASR:    {r1['naive_asr']:.2%} ({int(r1['naive_count'])}/{N_SAMPLES})")
    print(f"  Adaptive ASR: {r1['adaptive_asr']:.2%} ({int(r1['adaptive_count'])}/{N_SAMPLES})")
    print(f"  Gap:          {r1['gap']:+.2%}")
    print(f"  Gap/SE:       {r1['gap_se']:.2f}")

    # ---- Config 2: Continuous gradient, T=50, eps=0.2 ----
    print(f"\n{'='*70}")
    print("CONFIG 2: CONTINUOUS GRADIENT, T=50, EPS=0.2")
    print(f"{'='*70}")
    r2 = run_single_config(model, dataset, trajectories, CONFIG, D, N_SAMPLES,
                           pgd_attack_continuous, alpha=0.01, epsilon=0.2, steps=50,
                           update_rule_label="continuous")
    print(f"\n  Naive ASR:    {r2['naive_asr']:.2%} ({int(r2['naive_count'])}/{N_SAMPLES})")
    print(f"  Adaptive ASR: {r2['adaptive_asr']:.2%} ({int(r2['adaptive_count'])}/{N_SAMPLES})")
    print(f"  Gap:          {r2['gap']:+.2%}")
    print(f"  Gap/SE:       {r2['gap_se']:.2f}")

    # ---- Save results ----
    saved_results = [
        {
            "update_rule": "sign",
            "T": 20,
            "epsilon": 0.2,
            "naive_asr": r1["naive_asr"],
            "adaptive_asr": r1["adaptive_asr"],
            "n": N_SAMPLES,
            "gap": r1["gap"],
            "gap_se": r1["gap_se"],
            "label": "borderline_sign_pgd",
        },
        {
            "update_rule": "continuous",
            "T": 50,
            "epsilon": 0.2,
            "naive_asr": r2["naive_asr"],
            "adaptive_asr": r2["adaptive_asr"],
            "n": N_SAMPLES,
            "gap": r2["gap"],
            "gap_se": r2["gap_se"],
            "label": "borderline_continuous",
        },
    ]
    save_results(
        script_name="recheck_n200.py",
        config=CONFIG,
        results=saved_results,
        extra={"model": "TrainedSurrogateModel", "pool_size": POOL_SIZE,
               "n_samples": N_SAMPLES, "purpose": "N=200 re-check of borderline configs"},
    )

    # ---- Gate check ----
    print(f"\n{'='*70}")
    print("GATE CHECK (N=200)")
    print(f"{'='*70}")
    for label, r in [("Config 1 (sign, T=20, eps=0.2)", r1),
                     ("Config 2 (continuous, T=50, eps=0.2)", r2)]:
        gate = abs(r["gap"]) > 0.02 and r["gap_se"] > 1.0
        print(f"  {label}:")
        print(f"    Gap/SE = {r['gap_se']:.2f}, |Gap| = {abs(r['gap']):.0%}")
        print(f"    Gate: {'PASSED' if gate else 'FAILED'}")


if __name__ == "__main__":
    main()
