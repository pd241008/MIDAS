"""
Part B (MCU tier) items 9+10: recalibrate gamma on the MCU-tier data.

Edge's gamma=2.3061 (benign P95 of epsilon_p, d=42, W=10, 42-dim surrogate) does
NOT transfer to the MCU tier: different d (12), different W dimension, different
(normalized) feature distribution, different per-source specialist classifiers.

Matching the Edge method (shadow/defense.py penetration_epsilon_windowed +
shadow/compute_fpr.py):
  - Trajectory vectors v_i are normalized 12-dim feature rows
  - epsilon_p = (1/|W|) * sum_i ||v_i||_2 * max(0, momentum(v_i, v_{i+1}))
  - A window counts toward the benign epsilon_p distribution only if ALL W rows
    are benign (clean benign-only windows, W=10)
  - gamma = P95 of clean benign epsilon_p  (paper rule)
  - FPR = fraction of clean benign windows with epsilon_p > gamma

Because deployment is provenance-routed (two per-source specialists on the shared
unified 12-dim space), gamma is recalibrated PER SOURCE (nsl / unsw).

Emits:
  - mcu/features/gamma_report.json   gamma + FPR + quantiles per source
  - mcu/features/gamma_params.json   params for firmware / README
"""
import json
import os

import numpy as np
import torch

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")
RESULTS_DIR = os.path.join(MCU_DIR, "results")

D = 12
W = 10   # matches Edge tier window (and firmware ring buffer depth)


def penetration_epsilon_windowed_np(window_rows: np.ndarray) -> float:
    """NumPy replica of defense.py penetration_epsilon_windowed.
    window_rows: (W, d) normalized feature rows, consecutive trajectory vectors."""
    if window_rows.shape[0] < 2:
        return 0.0
    sum_val = 0.0
    pairs = window_rows.shape[0] - 1
    for i in range(pairs):
        v_prev = window_rows[i]
        v_cur = window_rows[i + 1]
        norm_t = np.linalg.norm(v_cur)
        norm_tm1 = np.linalg.norm(v_prev)
        if norm_t < 1e-7 or norm_tm1 < 1e-7:
            m = 0.0
        else:
            m = float(np.dot(v_cur, v_prev) / (norm_t * norm_tm1))
            m = max(-1.0, min(1.0, m))
        m = max(0.0, m)
        sum_val += norm_t * m
    return sum_val / pairs


def compute_source(prov_value, name, X, y, p):
    mask = p == prov_value
    Xs = X[mask]
    ys = y[mask]

    # Clean benign-only windows: all W consecutive rows benign (label 0)
    benign_mask = ys == 0
    clean_eps = []
    for j in range(W - 1, len(Xs)):
        if not benign_mask[j - W + 1: j + 1].all():
            continue
        window = Xs[j - W + 1: j + 1]
        clean_eps.append(penetration_epsilon_windowed_np(window))

    clean_eps_t = torch.tensor(clean_eps, dtype=torch.float32)
    n_clean = len(clean_eps)

    p50 = torch.quantile(clean_eps_t, 0.50).item()
    p75 = torch.quantile(clean_eps_t, 0.75).item()
    p90 = torch.quantile(clean_eps_t, 0.90).item()
    p95 = torch.quantile(clean_eps_t, 0.95).item()
    gamma = p95  # operating point per paper rule

    def fpr(t):
        return (clean_eps_t > t).float().mean().item()

    return {
        "name": name,
        "d": D,
        "w": W,
        "n_rows": int(len(Xs)),
        "n_clean_benign_windows": n_clean,
        "gamma_operating_p95": round(gamma, 6),
        "fpr_at_operating_gamma": round(fpr(gamma), 6),
        "benign_epsilon_p_quantiles": {
            "p50": round(p50, 6),
            "p75": round(p75, 6),
            "p90": round(p90, 6),
            "p95": round(p95, 6),
        },
    }


def main():
    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy")).astype(np.float32)
    y = np.load(os.path.join(FEATURES_DIR, "unified_labels.npy")).astype(np.float32)
    p = np.load(os.path.join(FEATURES_DIR, "unified_provenance.npy"))

    print(f"MCU-tier gamma recalibration (d={D}, W={W}, per-source specialists)")
    sources = [
        compute_source(0, "nsl", X, y, p),
        compute_source(1, "unsw", X, y, p),
    ]

    # Merged (joint space, provenance-blind) — for reference/paper completeness
    joint_mask = np.ones_like(p, dtype=bool)
    joint = compute_source(0, "__joint_prov0__", X[~joint_mask], y, p) if False else None

    report = {
        "experiment": "Items 9+10: gamma recalibration + FPR on MCU-tier data",
        "note": ("Edge gamma=2.3061 does not transfer (d=12 not 42, normalized overlap features, "
                 "per-source specialists). gamma = clean benign epsilon_p P95 per source, W=10."),
        "config": {"d": D, "W": W},
        "edge_operating_gamma_for_ref": 2.3061,
        "sources": sources,
        "fpr_note": "FPR computed on clean benign-only windows (all 10 rows benign), the paper-rule-consistent definition.",
    }

    print(json.dumps(report, indent=2))

    # Params file for firmware / downstream (stable name, non-timestamped)
    params = {
        "d": D,
        "w": W,
        "gamma_per_source": {s["name"]: s["gamma_operating_p95"] for s in sources},
    }
    os.makedirs(FEATURES_DIR, exist_ok=True)
    with open(os.path.join(FEATURES_DIR, "gamma_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nsaved {os.path.join(FEATURES_DIR, 'gamma_params.json')}")

    # Timestamped result (mirror shadow/results convention)
    save_results(
        "recalibrate_gamma.py",
        config={"d": D, "W": W, "edge_operating_gamma_for_ref": 2.3061},
        results=report,
        extra={"params": params},
    )


if __name__ == "__main__":
    main()
