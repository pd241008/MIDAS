"""
Run naive-vs-adaptive PGD/C&W battery against the production classifier.

The production float32 TFLite model (classifier_float32.tflite) is a direct
export of the PyTorch TrainedSurrogateModel trained on UNSW-NB15 (d=42,
hidden=16, tanh, sigmoid).  For float32 the two are bit-exact (cross-language
validation: max |Δp| = 1.9e-6, zero label flips), so the adaptive attack
(which requires backpropagation) runs in PyTorch with the exact production
weights, and the resulting adversarial examples are verified against the real
TFLite model via classify_transfer.rs.

Spec:
  model     : classifier_float32.tflite  (TrainedSurrogateModel + TFLite verify)
  basis     : fixed PCA-2 (deployed) + attacker-coupled (table-format match)
  attacks   : PGD, α=0.01, T∈{20,50,100}, ε∈{0.05,0.1,0.2}
  c&w       : iters=100, c_steps=3, lr=0.01, initial_c=1.0
  attackers : naive and adaptive (backprop through trajectory→rotation)
  N         : 200 per config
  data      : UNSW-NB15 test set, benign-filtered (nat_conf ≤ 0.95), W=10
  gamma     : 2.3061 (benign P95, recalibrated)
"""
import sys
import os
import json
import time
import subprocess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.defense import (
    TrainedSurrogateModel,
    midas_defense_forward,
)
from shadow.attacks import (
    naive_pgd_attack,
    adaptive_pgd_attack,
    project_Lp_ball,
)
from shadow.cw_attack import cw_l2_attack
from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
D = 42
W = 10
GAMMA = 2.3061
N_SAMPLES = 200

CONFIG = {
    "W": W,
    "D": D,
    "gamma": GAMMA,
    "lambda": 1.0,
    "k": 2.0,
    "delta_theta_max_deg": 45.0,
    "tau": 0.75,
    "sla_budget_ms": 10,
    "channel_capacity": 4,
}

PGD_ATTACKS = [
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

CW_KWARGS = {
    "iters": 100,
    "c_steps": 3,
    "lr": 0.01,
    "initial_c": 1.0,
}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")
WEIGHTS_DIR = os.path.join(REPO, "shadow", "trained_weights")
MODEL_PATH = os.path.join(REPO, "models", "classifier_float32.tflite")
OUT_DIR = os.path.join(REPO, "results", "pi")


# ---------------------------------------------------------------------------
# Model / data loading
# ---------------------------------------------------------------------------
def load_production_classifier():
    """Load the exact production classifier weights into PyTorch."""
    model = TrainedSurrogateModel(D, hidden=16)
    path = os.path.join(WEIGHTS_DIR, "trained_surrogate_real.pt")
    model.load_state_dict(torch.load(path, weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def load_manifold():
    with open(os.path.join(REPO, "results", "basis.json")) as f:
        data = json.load(f)
    return (
        torch.tensor(data["basis"], dtype=torch.float32),
        torch.tensor(data["c_base"], dtype=torch.float32),
    )


def load_real_data():
    test_X = torch.tensor(
        np.load(os.path.join(DATA_DIR, "test_features.npy")), dtype=torch.float32
    )
    test_y = torch.tensor(
        np.load(os.path.join(DATA_DIR, "test_labels.npy")), dtype=torch.float32
    )
    return test_X, test_y


def build_dataset(test_X, test_y, N, W, classifier):
    benign_mask = test_y == 0
    benign_X = test_X[benign_mask]
    with torch.no_grad():
        probs = classifier(benign_X)
        nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered_X = benign_X[keep_mask]
        benign_indices = torch.where(benign_mask)[0]
        filtered_orig_indices = benign_indices[keep_mask]

    print(f"  benign pool: {len(benign_X)} → after ≤0.95 filter: {len(filtered_X)}")

    if len(filtered_X) < N:
        print(f"  WARNING: only {len(filtered_X)} survive filter; using all")
        N = len(filtered_X)

    dataset = filtered_X[:N]
    trajectories = []
    for i in range(N):
        orig_idx = filtered_orig_indices[i].item()
        start = max(0, orig_idx - (W - 1))
        traj = [test_X[j] for j in range(start, orig_idx)]
        while len(traj) < W - 1:
            traj.insert(0, filtered_X[0])
        trajectories.append(traj)

    return dataset, trajectories, N


# ---------------------------------------------------------------------------
# PGD attack loops (fixed + vulnerable basis)
# ---------------------------------------------------------------------------
def pgd_vuln(x0, y, classifier, traj, c_base, config,
             alpha, epsilon, steps, naive, first_sample):
    x_t = x0.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0, epsilon)

    W_local = config.get("W", W)
    sliding_window = [w.clone().detach() for w in traj]

    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W_local - 1):]
        x_def, theta = midas_defense_forward_vulnerable(
            x_t, window_slice, c_base, config, naive=naive
        )
        if not naive:
            theta.retain_grad()
        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y.expand_as(pred))
        loss.backward()
        grad = x_t.grad

        if first_sample and not naive and theta.grad is not None:
            print(f"    [VULN-ADAPT Step {step:3d}] "
                  f"||d(loss)/d(delta_theta)|| = {torch.norm(theta.grad).item():.8f}  "
                  f"loss = {loss.item():.6f}")

        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0, epsilon)
            sliding_window.append(x_t.clone().detach())

    return x_t.detach(), sliding_window[-(W_local - 1):]


def run_pgd_battery(classifier, dataset, trajectories, c_base, basis, config):
    results = {}

    # --- Fixed basis ---
    fixed = {}
    for name, alpha, steps, eps in PGD_ATTACKS:
        print(f"\n{'='*60}")
        print(f"  {name}  [FIXED BASIS]")
        print(f"{'='*60}")

        naive_succ = 0
        adaptive_succ = 0

        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            is_first = (i == 0)
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            if is_first:
                print("\n  --- Fixed Basis: Naive ---")
            x_adv_n, win_n = naive_pgd_attack(
                x0, y_nat, classifier, traj, c_base, basis, config,
                alpha, eps, steps, first_sample=is_first
            )
            def_n, _ = midas_defense_forward(
                x_adv_n, win_n, c_base, basis, config, naive=False
            )
            if (classifier(def_n) > 0.5).item() != y_nat.item():
                naive_succ += 1

            if is_first:
                print("\n  --- Fixed Basis: Adaptive ---")
            x_adv_a, win_a = adaptive_pgd_attack(
                x0, y_nat, classifier, traj, c_base, basis, config,
                alpha, eps, steps, first_sample=is_first
            )
            def_a, _ = midas_defense_forward(
                x_adv_a, win_a, c_base, basis, config, naive=False
            )
            if (classifier(def_a) > 0.5).item() != y_nat.item():
                adaptive_succ += 1

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        gap = asr_a - asr_n
        fixed[name] = {
            "naive": round(asr_n, 6),
            "adaptive": round(asr_a, 6),
            "gap": round(gap, 6),
            "n": N_SAMPLES,
        }
        print(f"\n  Naive ASR: {asr_n:.2%}  Adaptive ASR: {asr_a:.2%}  "
              f"Gap: {gap:+.2%}  "
              f"{'✓ real gap' if abs(gap) > 0.02 else '— flat'}")

    # --- Vulnerable (attacker-coupled) basis ---
    vuln = {}
    for name, alpha, steps, eps in PGD_ATTACKS:
        print(f"\n{'='*60}")
        print(f"  {name}  [VULNERABLE BASIS]")
        print(f"{'='*60}")

        naive_succ = 0
        adaptive_succ = 0

        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            is_first = (i == 0)
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            if is_first:
                print("\n  --- Vulnerable Basis: Naive ---")
            x_adv_n, win_n = pgd_vuln(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, eps, steps, naive=True, first_sample=is_first
            )
            def_n, _ = midas_defense_forward_vulnerable(
                x_adv_n, win_n, c_base, config, naive=False
            )
            if (classifier(def_n) > 0.5).item() != y_nat.item():
                naive_succ += 1

            if is_first:
                print("\n  --- Vulnerable Basis: Adaptive ---")
                print()
            x_adv_a, win_a = pgd_vuln(
                x0, y_nat, classifier, traj, c_base, config,
                alpha, eps, steps, naive=False, first_sample=is_first
            )
            def_av, _ = midas_defense_forward_vulnerable(
                x_adv_a, win_a, c_base, config, naive=False
            )
            if (classifier(def_av) > 0.5).item() != y_nat.item():
                adaptive_succ += 1

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        gap = asr_a - asr_n
        vuln[name] = {
            "naive": round(asr_n, 6),
            "adaptive": round(asr_a, 6),
            "gap": round(gap, 6),
            "n": N_SAMPLES,
        }
        print(f"\n  Naive ASR: {asr_n:.2%}  Adaptive ASR: {asr_a:.2%}  "
              f"Gap: {gap:+.2%}  "
              f"{'✓ real gap' if abs(gap) > 0.02 else '— flat'}")

    return {"fixed": fixed, "vulnerable": vuln}


# ---------------------------------------------------------------------------
# C&W-L2 attack loops
# ---------------------------------------------------------------------------
def run_cw_battery(classifier, dataset, trajectories, c_base, basis, config):
    results = {}
    for coupled in (False, True):
        basis_name = "fixed" if not coupled else "attacker-coupled"
        for naive in (True, False):
            attacker = "naive" if naive else "adaptive"
            label = f"C&W-L2 {basis_name} / {attacker}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            succ = 0
            dists = []
            for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
                is_first = (i == 0)
                with torch.no_grad():
                    y_nat = (classifier(x0) > 0.5).float()

                success, _, dist, best_c = cw_l2_attack(
                    x0, y_nat, classifier, traj, c_base, basis, config,
                    naive=naive, coupled_basis=coupled,
                    first_sample=is_first, **CW_KWARGS
                )
                if success:
                    succ += 1
                if dist is not None:
                    dists.append(dist)

            asr = succ / N_SAMPLES
            med_dist = float(np.median(dists)) if dists else None
            mean_dist = float(np.mean(dists)) if dists else None
            key = f"C&W-L2 {basis_name} {attacker}"
            results[key] = {
                "asr": round(asr, 6),
                "n": N_SAMPLES,
                "median_l2": round(med_dist, 6) if med_dist is not None else None,
                "mean_l2": round(mean_dist, 6) if mean_dist is not None else None,
                **CW_KWARGS,
            }
            print(f"  ASR = {asr:.3f}  median L2 = {med_dist:.5f}  "
                  f"mean L2 = {mean_dist:.5f}")

    return results


# ---------------------------------------------------------------------------
# TFLite verification via classify_transfer.rs
# ---------------------------------------------------------------------------
def verify_with_tflite(batch_path, model_path, out_path):
    """Run classify_transfer.rs to measure ASR on the real TFLite model."""
    if not os.path.exists(model_path):
        print(f"  WARNING: TFLite model not found at {model_path}; skipping verification")
        return None

    cmd = [
        "cargo", "run", "--release", "--features", "tflite",
        "--bin", "classify_transfer", "--",
        model_path, batch_path, out_path,
    ]
    print(f"\nVerifying adversarial batch against real TFLite model...")
    print(f"  cmd: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, cwd=os.path.join(REPO, "edge"),
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  classify_transfer failed: {result.stderr[:500]}")
            return None
        print(result.stdout)
        with open(out_path) as f:
            return json.load(f)
    except Exception as e:
        print(f"  Verification failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Adversarial batch for TFLite verification
# ---------------------------------------------------------------------------
def collect_representative_batch(classifier, dataset, trajectories, c_base, basis, config):
    """Collect adversarial examples for one representative config to verify on TFLite."""
    reps = {}
    for (basis_label, coupled, attack_fn_name, attack_fn) in [
        ("fixed", False, "naive", naive_pgd_attack),
        ("fixed", False, "adaptive", adaptive_pgd_attack),
        ("vulnerable", True, "naive", None),  # pgd_vuln handles both
        ("vulnerable", True, "adaptive", None),
    ]:
        print(f"  Collecting batch: {basis_label} / {attack_fn_name}")
        x0 = dataset[0]
        traj = trajectories[0]
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()

        if attack_fn is None:
            naive_flag = (attack_fn_name == "naive")
            x_adv, win = pgd_vuln(
                x0, y_nat, classifier, traj, c_base, config,
                0.01, 0.1, 50, naive=naive_flag, first_sample=True
            )
        else:
            x_adv, win = attack_fn(
                x0, y_nat, classifier, traj, c_base, basis, config,
                0.01, 0.1, 50, first_sample=True
            )
        reps[f"{basis_label}_{attack_fn_name}"] = x_adv.squeeze().tolist()
    return reps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=" * 60)
    print("PRODUCTION CLASSIFIER ATTACK BATTERY")
    print(f"  model: classifier_float32.tflite (TrainedSurrogateModel weights)")
    print(f"  data: UNSW-NB15 d={D} W={W} gamma={GAMMA}")
    print("=" * 60)

    classifier = load_production_classifier()
    basis, c_base = load_manifold()
    test_X, test_y = load_real_data()
    dataset, trajectories, n_samples = build_dataset(
        test_X, test_y, N_SAMPLES, W, classifier
    )
    print(f"  N_SAMPLES={n_samples}  dataset: {len(dataset)} samples, "
          f"trajectories: {len(trajectories)}")

    with torch.no_grad():
        y_nat = (classifier(dataset) > 0.5).float()
    labels = y_nat.squeeze()

    # --- PGD battery ---
    pgd_results = run_pgd_battery(
        classifier, dataset, trajectories, c_base, basis, CONFIG
    )

    # --- C&W-L2 battery ---
    cw_results = run_cw_battery(
        classifier, dataset, trajectories, c_base, basis, CONFIG
    )

    dt = time.time() - t0

    # --- Representative batch for TFLite verification ---
    rep_batch = collect_representative_batch(
        classifier, dataset, trajectories, c_base, basis, CONFIG
    )
    batch_path = os.path.join(OUT_DIR, "tflite_attack_batch.json")
    with open(batch_path, "w") as f:
        json.dump({
            "d": D,
            "W": W,
            "n": 1,
            "labels": [labels[0].item()],
            "originals": [dataset[0].tolist()],
            "naive_adversarial": rep_batch["fixed_naive"],
            "adaptive_adversarial": rep_batch["fixed_adaptive"],
            **{k: [v] for k, v in rep_batch.items()},
        }, f, indent=2)
    print(f"Saved representative batch to {batch_path}")

    # --- Save raw results ---
    output = {
        "table": "Production classifier attack battery (classifier_float32.tflite)",
        "model": "classifier_float32.tflite",
        "model_note": (
            "float32 TFLite export of TrainedSurrogateModel (d=42, hidden=16, "
            "tanh, sigmoid). PyTorch and float32 TFLite are bit-exact "
            "(max |Δp| = 1.9e-6, zero label flips)."
        ),
        "basis": "fixed PCA-2 (deployed) + attacker-coupled (gate)",
        "gamma": GAMMA,
        "d": D,
        "w": W,
        "n_samples": n_samples,
        "data_source": "UNSW-NB15",
        "elapsed_seconds": round(dt, 1),
        "pgd": pgd_results,
        "cw": cw_results,
    }
    out_path = os.path.join(OUT_DIR, "check4_tflite.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved raw results to {out_path} ({dt:.1f}s)")

    # --- Build paper table ---
    build_paper_table(pgd_results, cw_results, n_samples, GAMMA)

    # --- TFLite verification ---
    print("\n" + "=" * 60)
    print("TFLite verification")
    print("=" * 60)
    verify_with_tflite(
        batch_path,
        MODEL_PATH,
        os.path.join(OUT_DIR, "tflite_verification.json"),
    )


def build_paper_table(pgd_results, cw_results, n, gamma):
    """Build results/pi/defense_success_rate_tflite.json matching Table XVII."""
    rows = []
    for name, res in pgd_results["fixed"].items():
        rows.append({
            "config": name,
            "basis": "fixed",
            "naive_ASR": res["naive"],
            "adaptive_ASR": res["adaptive"],
            "naive_defense_success=1-ASR": round(1 - res["naive"], 4),
            "adaptive_defense_success=1-ASR": round(1 - res["adaptive"], 4),
            "n": n,
        })
    for name, res in pgd_results["vulnerable"].items():
        rows.append({
            "config": name,
            "basis": "attacker-coupled",
            "naive_ASR": res["naive"],
            "adaptive_ASR": res["adaptive"],
            "naive_defense_success=1-ASR": round(1 - res["naive"], 4),
            "adaptive_defense_success=1-ASR": round(1 - res["adaptive"], 4),
            "n": n,
        })

    table = {
        "table": "On-device adaptive-robustness - PGD sweep (production classifier)",
        "note": (
            f"Attack-success = prediction flipped by the defended classifier. "
            f"'fixed' = deployed fixed PCA-2 basis; 'attacker-coupled' = basis an "
            f"adaptive attacker who knows the rotation could exploit (gate basis). "
            f"Production classifier_float32.tflite (d=42 W=10), n={n}, gamma={gamma} "
            f"(recalibrated benign P95). PyTorch attack with exact production weights; "
            f"float32 TFLite is bit-exact to PyTorch (max |Δp| = 1.9e-6)."
        ),
        "columns": [
            "config", "basis",
            "naive_ASR", "adaptive_ASR",
            "naive_defense_success=1-ASR", "adaptive_defense_success=1-ASR",
            "n",
        ],
        "rows": rows,
        "sources": ["results/pi/check4_tflite.json"],
    }

    out_path = os.path.join(OUT_DIR, "defense_success_rate_tflite.json")
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"Built paper table: {out_path} ({len(rows)} rows)")

    # C&W table
    cw_rows = []
    for key, res in cw_results.items():
        parts = key.split()
        basis = parts[1] if len(parts) > 1 else "unknown"
        attacker = parts[2] if len(parts) > 2 else "unknown"
        cw_rows.append({
            "config": key,
            "basis": basis,
            "attacker": attacker,
            "ASR": res["asr"],
            "median_L2": res["median_l2"],
            "mean_L2": res["mean_l2"],
            "n": n,
            **{k: res[k] for k in ("iters", "c_steps", "lr", "initial_c") if k in res},
        })

    cw_table = {
        "table": "Production classifier C&W-L2 battery",
        "note": (
            f"C&W-L2 (untargeted) against production classifier_float32.tflite. "
            f"iters=100, c_steps=3, lr=0.01, initial_c=1.0. n={n}, gamma={gamma}."
        ),
        "columns": ["config", "basis", "attacker", "ASR", "median_L2", "mean_L2", "n"],
        "rows": cw_rows,
        "sources": ["results/pi/check4_tflite.json"],
    }
    cw_path = os.path.join(OUT_DIR, "cw_l2_tflite.json")
    with open(cw_path, "w") as f:
        json.dump(cw_table, f, indent=2)
    print(f"Built C&W table: {cw_path} ({len(cw_rows)} rows)")


if __name__ == "__main__":
    main()
