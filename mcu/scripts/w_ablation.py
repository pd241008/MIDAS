"""
Ring-buffer W ablation for the MCU adaptive-attacker gate.

Sweeps the defense window size W (the number of trajectory rows the delta-theta
defense retains, W-1 prior + the current row) and re-runs the adaptive-attacker
gate on the deployed INT8 specialists per source. The attacked benign sample set
is held FIXED across W (sampled once at W_max), so any gate difference is purely
the ring-buffer/window effect, not sample re-sampling.

Under test per (source, W):
  - vulnerable (attacker-coupled) basis: the known-vulnerable control. The gate
    PASSES when |naive-adaptive gap| > 0.02 on at least one attack config.
  - fixed basis (deployed rotation): the adaptation gap should collapse.

Outputs: mcu/features/w_ablation_report.json + timestamped
mcu/results/w_ablation_*.json (mirror save_results convention).
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_attacker_gate import (  # noqa: E402
    Int8SpecialistReplica, load_source, load_manifold, load_gamma,
    defense_forward, pgd_attack, DELTA_THETA_MAX_DEG,
)
from save_results import save_results  # noqa: E402

W_MAX = 10
D = 12
W_SWEEP = [2, 3, 4, 6, 8, 10]
N_SAMPLES = int(os.environ.get("WA_N", "200"))
ATTACK_CFGS = [("eps0.05_20", 0.01, 20, 0.05), ("eps0.1_50", 0.01, 50, 0.1),
               ("eps0.2_100", 0.01, 100, 0.2)]
FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")


def build_fixed_dataset(Xs, ys, classifier, n_samples, w_max):
    """One-time sample set (W=w_max trajectories); per-W windows are sliced later."""
    benign_idx = torch.where(ys == 0)[0]
    with torch.no_grad():
        probs = classifier(Xs[benign_idx])
    nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
    keep = nat_conf.squeeze() <= 0.95
    benign_idx = benign_idx[keep]

    dataset, traj_wmax = [], []
    for k in range(len(benign_idx)):
        i = int(benign_idx[k].item())
        if i < w_max - 1:
            continue
        dataset.append(Xs[i])
        traj_wmax.append([Xs[j] for j in range(i - (w_max - 1), i)])
        if len(dataset) >= n_samples:
            break
    return dataset, traj_wmax


def slice_traj(traj_wmax, w):
    if w <= 1:
        return []
    return traj_wmax[-(w - 1):] if w - 1 <= len(traj_wmax) else traj_wmax


def run_battery_w(classifier, dataset, traj_wmax, c_base, basis, cfg, attacks, w, vulnerable):
    results = {}
    for name, alpha, steps, eps in attacks:
        n_succ = a_succ = 0
        for x0, traj0 in zip(dataset, traj_wmax):
            traj = slice_traj(traj0, w)
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()
            xa_n, wn = pgd_attack(x0, y_nat, classifier, traj, c_base, basis, cfg,
                                  alpha, eps, steps, naive=True, vulnerable=vulnerable)
            xd_n, _ = defense_forward(xa_n, wn, c_base, basis, cfg, vulnerable=vulnerable)
            if (classifier(xd_n) > 0.5).item() != y_nat.item():
                n_succ += 1
            xa_a, wa = pgd_attack(x0, y_nat, classifier, traj, c_base, basis, cfg,
                                  alpha, eps, steps, naive=False, vulnerable=vulnerable)
            xd_a, _ = defense_forward(xa_a, wa, c_base, basis, cfg, vulnerable=vulnerable)
            if (classifier(xd_a) > 0.5).item() != y_nat.item():
                a_succ += 1
        asr_n = n_succ / len(dataset)
        asr_a = a_succ / len(dataset)
        results[name] = {"naive_asr": round(asr_n, 4), "adaptive_asr": round(asr_a, 4),
                         "gap": round(asr_a - asr_n, 4)}
    return results


def main():
    torch.manual_seed(42)
    gammas = load_gamma()
    basis, c_base = load_manifold()
    report = {
        "experiment": "Ring-buffer W ablation on the MCU adaptive-attacker gate",
        "note": "same attacked sample set across W (sampled at W_max); window = W-1 prior "
                "rows + current. gate PASS on vulnerable when any |gap| > 0.02.",
        "d": D, "W_max": W_MAX, "n_samples": N_SAMPLES,
        "attack_configs": ["eps0.05_20", "eps0.1_50", "eps0.2_100"],
    }

    for tag, prov in [("nsl", 0), ("unsw", 1)]:
        gamma = gammas[tag]
        classifier = Int8SpecialistReplica(tag, gamma)
        classifier.eval()
        Xs, ys = load_source(prov)
        dataset, traj_wmax = build_fixed_dataset(Xs, ys, classifier, N_SAMPLES, W_MAX)
        n = len(dataset)
        print(f"\n===== {tag.upper()} (gamma={gamma:.4f}) — {n} samples, W sweep "
              f"{W_SWEEP} =====")

        src = {"gamma": gamma, "n": n, "by_W": {}}
        for w in W_SWEEP:
            cfg = {"W": w, "gamma": gamma, "lambda": 1.0, "k": 2.0,
                   "delta_theta_max": DELTA_THETA_MAX_DEG * (3.141592653589793 / 180.0)}
            vuln = run_battery_w(classifier, dataset, traj_wmax, c_base, None, cfg,
                                 ATTACK_CFGS, w, vulnerable=True)
            has_gap = any(abs(v["gap"]) > 0.02 for v in vuln.values())
            fixed = run_battery_w(classifier, dataset, traj_wmax, c_base, basis, cfg,
                                  ATTACK_CFGS, w, vulnerable=False)
            entry = {"vulnerable_basis": vuln, "gate_passed": bool(has_gap),
                     "fixed_basis": fixed}
            src["by_W"][str(w)] = entry
            gv = [v["gap"] for v in vuln.values()]
            gf = [fixed[k]["gap"] for k, *_ in ATTACK_CFGS]
            print(f"  W={w:2d}  vuln gaps={gv}  pass={entry['gate_passed']}  fixed gaps={gf}")
        report[tag] = src

    with open(os.path.join(FEATURES_DIR, "w_ablation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {os.path.join(FEATURES_DIR, 'w_ablation_report.json')}")

    save_results(
        "w_ablation.py",
        config={"W_sweep": W_SWEEP, "W_max": W_MAX, "d": D, "n": N_SAMPLES,
                "attacks": ATTACK_CFGS},
        results=report,
    )


if __name__ == "__main__":
    main()