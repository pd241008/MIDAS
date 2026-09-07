"""
Part B (MCU tier) high-N gate recheck — mirror of shadow/recheck_n200.py.

Edge discipline: a config only counts toward a gate verdict if, on a FRESH
pool at higher N, it satisfies BOTH |gap| > 0.02 AND gap/SE > 1.0 (see
shadow/recheck_n200.py "GATE CHECK"). We re-run every config from the N=200
gate (mcu/features/attacker_gate_report.json) on fresh seeded pools at
N_SAMPLES=800, computing exact binomial SEs for naive/adaptive ASR and Gap/SE.

Purpose per source:
  - NSL coupled-basis: positive control — a known vulnerability must REGISTER
    (large positive gap, Gap/SE >> 1). If it does not, the harness cannot be
    trusted to report an absence elsewhere.
  - NSL fixed-basis (0.045/0.065 at N=200): must COLLAPSE toward 0 if noise.
  - UNSW coupled-basis (max 0.065 at N=200): decide whether the weak signal
    holds or was noise. Weak-but-persistent = real reportable finding; either
    way it does NOT clear the "known-vulnerability demonstrated" bar.

Emits:
  - mcu/features/gate_recheck_report.json      (stable merged per-config)
  - mcu/results/recheck_gate_n800_<timestamp>.json   (git-hashed capture)
"""
import argparse
import importlib.util
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")

N_SAMPLES = 800
SEED = 42

CONFIGS = [("eps0.05_20", 0.01, 20, 0.05),
           ("eps0.1_50", 0.01, 50, 0.1),
           ("eps0.2_100", 0.01, 100, 0.2)]


def load_aag():
    spec = importlib.util.spec_from_file_location(
        "aag", os.path.join(HERE, "adaptive_attacker_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_fresh_pool(aag, classifier, Xs, ys, n_samples):
    """Seeded random benign sample (mirror recheck_n200 fresh-pool semantics)."""
    benign_idx = torch.where(ys == 0)[0]
    perm = torch.randperm(len(benign_idx))
    candidates = benign_idx[perm]

    with torch.no_grad():
        probs = classifier(Xs[candidates])
    nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
    keep = nat_conf.squeeze() <= 0.95
    candidates = candidates[keep]

    dataset, trajectories = [], []
    for k in range(len(candidates)):
        i = int(candidates[k].item())
        if i < aag.W - 1:
            continue
        traj = [Xs[j] for j in range(i - (aag.W - 1), i)]
        dataset.append(Xs[i])
        trajectories.append(traj)
        if len(dataset) >= n_samples:
            break
    return dataset, trajectories, len(candidates)


def run_single_config(aag, classifier, dataset, trajectories, c_base, basis,
                      cfg, alpha, eps, steps, vulnerable):
    n = 0
    naive_succ = 0
    adaptive_succ = 0
    for x0, traj in zip(dataset, trajectories):
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()
        xa_n, wn = aag.pgd_attack(x0, y_nat, classifier, traj, c_base, basis, cfg,
                                  alpha, eps, steps, naive=True, vulnerable=vulnerable)
        xd_n, _ = aag.defense_forward(xa_n, wn, c_base, basis, cfg, vulnerable=vulnerable)
        if (classifier(xd_n) > 0.5).item() != y_nat.item():
            naive_succ += 1
        xa_a, wa = aag.pgd_attack(x0, y_nat, classifier, traj, c_base, basis, cfg,
                                  alpha, eps, steps, naive=False, vulnerable=vulnerable)
        xd_a, _ = aag.defense_forward(xa_a, wa, c_base, basis, cfg, vulnerable=vulnerable)
        if (classifier(xd_a) > 0.5).item() != y_nat.item():
            adaptive_succ += 1
        n += 1
        if n % 100 == 0:
            print(f"    [{n}/{len(dataset)}] naive={naive_succ/n:.3f} adaptive={adaptive_succ/n:.3f}")

    asr_n = naive_succ / n
    asr_a = adaptive_succ / n
    gap = asr_a - asr_n
    se_n = math.sqrt(asr_n * (1 - asr_n) / n) if n else 0.0
    se_a = math.sqrt(asr_a * (1 - asr_a) / n) if n else 0.0
    se_gap = math.sqrt(se_n ** 2 + se_a ** 2)
    gap_se = gap / se_gap if se_gap > 0 else float("inf")
    return {"naive_asr": round(asr_n, 4), "adaptive_asr": round(asr_a, 4),
            "n": n, "gap": round(gap, 4), "gap_se": round(gap_se, 4),
            "naive_count": naive_succ, "adaptive_count": adaptive_succ,
            "gate": bool(abs(gap) > 0.02 and gap_se > 1.0)}


def main():
    args = argparse.ArgumentParser(description="MCU high-N gate recheck")
    args.add_argument("--source", default="all", choices=["all", "nsl", "unsw"])
    args.add_argument("--basis", default="all", choices=["all", "coupled", "fixed"])
    args.add_argument("--n", type=int, default=N_SAMPLES)
    args = args.parse_args()

    torch.manual_seed(SEED)
    aag = load_aag()
    gammas = aag.load_gamma()
    basis, c_base = aag.load_manifold()
    cfg = {"W": aag.W, "gamma": 0.0, "lambda": 1.0, "k": 2.0,
           "delta_theta_max": aag.DELTA_THETA_MAX_DEG * (3.141592653589793 / 180.0)}
    n = args.n

    report = {"experiment": "MCU high-N gate recheck (mirror of shadow/recheck_n200.py)",
              "W": aag.W, "d": aag.D, "n_samples": n, "seed": SEED,
              "gate_rule": "|gap| > 0.02 AND gap/SE > 1.0 on fresh pool",
              "n200_source": "mcu/features/attacker_gate_report.json",
              "configs": {}}

    for tag, prov in [("nsl", 0), ("unsw", 1)]:
        if args.source not in ("all", tag):
            continue
        gamma = gammas[tag]
        cfg["gamma"] = gamma
        print(f"\n===== {tag.upper()} (gamma={gamma:.4f}) N={n} =====")
        classifier = aag.Int8SpecialistReplica(tag, gamma).eval()
        Xs, ys = aag.load_source(prov)
        dataset, trajectories, pool_n = build_fresh_pool(aag, classifier, Xs, ys, n)
        print(f"  fresh pool: {pool_n} benign after conf filter; using {len(dataset)}")
        if len(dataset) < n:
            print(f"  !! WARNING: only {len(dataset)} samples available (need {n})")

        report["configs"].setdefault(tag, {})
        for basis_mode in (["coupled"] if args.basis == "coupled"
                           else ["fixed"] if args.basis == "fixed"
                           else ["coupled", "fixed"]):
            vuln = basis_mode == "coupled"
            basis_used = None if vuln else basis
            report["configs"][tag].setdefault(basis_mode, {})
            for name, alpha, steps, eps in CONFIGS:
                print(f"\n  [{tag} {basis_mode} {name}]: naive/adaptive PGD "
                      f"(alpha={alpha}, T={steps}, eps={eps})")
                r = run_single_config(aag, classifier, dataset, trajectories,
                                      c_base, basis_used, cfg, alpha, eps, steps, vuln)
                r["config"] = name
                report["configs"][tag][basis_mode][name] = r
                print(f"    N={r['n']}  naive={r['naive_asr']:.4f} "
                      f"adaptive={r['adaptive_asr']:.4f}  "
                      f"gap={r['gap']:+.4f}  gap/SE={r['gap_se']:.2f}  "
                      f"gate={'PASS' if r['gate'] else 'FAIL'}")

    # Stable merged copy (survives chunked runs)
    stable_path = os.path.join(FEATURES_DIR, "gate_recheck_report.json")
    stable = {}
    if os.path.exists(stable_path):
        with open(stable_path) as f:
            stable = json.load(f)
    stable["experiment"] = report["experiment"]
    stable["n_samples"] = n
    stable["seed"] = SEED
    stable["gate_rule"] = report["gate_rule"]
    for tag in report["configs"]:
        stable.setdefault("configs", {}).setdefault(tag, {})
        for bm in report["configs"][tag]:
            stable["configs"][tag].setdefault(bm, {})
            for name, r in report["configs"][tag][bm].items():
                stable["configs"][tag][bm][name] = r
    with open(stable_path, "w") as f:
        json.dump(stable, f, indent=2)
    print(f"\nsaved {stable_path}")

    save_results("recheck_gate_n800.py",
                 config={"W": aag.W, "D": aag.D, "n_samples": n, "seed": SEED,
                         "gate_rule": report["gate_rule"]},
                 results={"configs": report["configs"]},
                 extra={"n200_source": "mcu/features/attacker_gate_report.json"})


if __name__ == "__main__":
    main()