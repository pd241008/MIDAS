"""
Check #4 battery on real UNSW-NB15 data (d=42).
- Loads the real trained surrogate (not SmoothMockModel)
- Uses real UNSW-NB15 samples + trajectories
- Recalibrated gamma from real epsilon_p distribution
- 9-config PGD sweep: eps∈{0.05,0.1,0.2} × T∈{20,50,100}
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow.defense import (
    TrainedSurrogateModel, midas_defense_forward,
    penetration_epsilon_windowed, rotation_angle, project_to_manifold,
    rotate_manifold_fixed_basis,
)
from shadow.attacks import naive_pgd_attack, adaptive_pgd_attack, project_Lp_ball
from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shadow", "trained_weights")

D = 42
W = 10
GAMMA_RECALIBRATED = 2.3061  # benign P95 (recalibrated, was 1.984 P75)  # 95th percentile of real epsilon_p

CONFIG = {
    "W": W,
    "D": D,
    "gamma": GAMMA_RECALIBRATED,
    "lambda": 1.0,
    "k": 2.0,
    "delta_theta_max_deg": 45.0,
    "tau": 0.75,
    "sla_budget_ms": 10,
    "channel_capacity": 4,
}


def load_real_surrogate():
    model = TrainedSurrogateModel(D, hidden=16)
    path = os.path.join(WEIGHTS_DIR, "trained_surrogate_real.pt")
    model.load_state_dict(torch.load(path, weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_manifold():
    with open("results/basis.json") as f:
        data = json.load(f)
    return (torch.tensor(data["basis"], dtype=torch.float32),
            torch.tensor(data["c_base"], dtype=torch.float32))


def load_real_data():
    train_X = torch.tensor(np.load(f"{DATA_DIR}/train_features.npy"), dtype=torch.float32)
    train_y = torch.tensor(np.load(f"{DATA_DIR}/train_labels.npy"), dtype=torch.float32)
    test_X = torch.tensor(np.load(f"{DATA_DIR}/test_features.npy"), dtype=torch.float32)
    test_y = torch.tensor(np.load(f"{DATA_DIR}/test_labels.npy"), dtype=torch.float32)
    return train_X, train_y, test_X, test_y


def build_dataset_and_trajectories(test_X, test_y, N_SAMPLES, W, classifier):
    """
    Build dataset (benign samples to attack) and trajectories (sequential benign samples).
    Applies >0.95 confidence filter to exclude trivially-overconfident samples.
    Uses the first N_SAMPLES benign samples from the FILTERED set.
    """
    benign_mask = test_y == 0
    benign_X = test_X[benign_mask]

    # Apply >0.95 confidence filter
    with torch.no_grad():
        probs = classifier(benign_X)
        nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered_X = benign_X[keep_mask]
        # Also need original indices for trajectory building
        benign_indices = torch.where(benign_mask)[0]
        filtered_orig_indices = benign_indices[keep_mask]

    print(f"  Benign pool: {len(benign_X)} → After >0.95 filter: {len(filtered_X)}")
    print(f"  Confidence dist of filtered: "
          f"min={nat_conf[keep_mask].min():.4f} median={nat_conf[keep_mask].median():.4f} "
          f"max={nat_conf[keep_mask].max():.4f} mean={nat_conf[keep_mask].mean():.4f}")

    if len(filtered_X) < N_SAMPLES:
        print(f"  WARNING: Only {len(filtered_X)} samples survive filter, need {N_SAMPLES}. Using all.")
        N_SAMPLES = len(filtered_X)

    dataset = filtered_X[:N_SAMPLES]

    # Build trajectories using original test_X indices
    trajectories = []
    for i in range(N_SAMPLES):
        orig_idx = filtered_orig_indices[i].item()
        start = max(0, orig_idx - (W - 1))
        traj = [test_X[j] for j in range(start, orig_idx)]
        while len(traj) < W - 1:
            traj.insert(0, filtered_X[0])
        trajectories.append(traj)

    return dataset, trajectories, N_SAMPLES


def pgd_vuln(x0_init, y_target, classifier, traj, c_base, config,
             alpha, epsilon, steps, naive, first_sample_flag):
    """Vulnerable-basis PGD attack (sign-based), matching fixed-basis."""
    x_t = x0_init.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0_init, epsilon)

    W_local = config.get("W", W)
    sliding_window = [w.clone().detach() for w in traj]

    for step in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(W_local - 1):]
        x_def, theta_vuln = midas_defense_forward_vulnerable(
            x_t, window_slice, c_base, config, naive=naive
        )
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
            sliding_window.append(x_t.clone().detach())

    final_window = sliding_window[-(W_local - 1):]
    return x_t.detach(), final_window


def run_battery(classifier, dataset, trajectories, c_base, basis, config, attacks, N_SAMPLES, mode="vuln"):
    results = {}
    for name, alpha, steps, eps in attacks:
        print(f"\n{'='*60}")
        print(f"  {name}  [{mode.upper()} BASIS]")
        print(f"{'='*60}")

        naive_succ = 0
        adaptive_succ = 0

        for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
            is_first = (i == 0)
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()

            if mode == "vuln":
                x_adv_n, win_n = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                    alpha, eps, steps, naive=True, first_sample_flag=is_first)
                def_n, _ = midas_defense_forward_vulnerable(x_adv_n, win_n, c_base, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

                if is_first:
                    print()
                x_adv_a, win_a = pgd_vuln(x0, y_nat, classifier, traj, c_base, config,
                                    alpha, eps, steps, naive=False, first_sample_flag=is_first)
                def_a, _ = midas_defense_forward_vulnerable(x_adv_a, win_a, c_base, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1
            else:
                if is_first:
                    print("\n  --- Fixed Basis: Naive ---")
                x_adv_n, win_n = naive_pgd_attack(x0, y_nat, classifier, traj, c_base, basis, config,
                                           alpha, eps, steps, first_sample=is_first)
                def_n, _ = midas_defense_forward(x_adv_n, win_n, c_base, basis, config, naive=False)
                if (classifier(def_n) > 0.5).item() != y_nat.item():
                    naive_succ += 1

                if is_first:
                    print("\n  --- Fixed Basis: Adaptive ---")
                x_adv_a, win_a = adaptive_pgd_attack(x0, y_nat, classifier, traj, c_base, basis, config,
                                              alpha, eps, steps, first_sample=is_first)
                def_a, _ = midas_defense_forward(x_adv_a, win_a, c_base, basis, config, naive=False)
                if (classifier(def_a) > 0.5).item() != y_nat.item():
                    adaptive_succ += 1

        asr_n = naive_succ / N_SAMPLES
        asr_a = adaptive_succ / N_SAMPLES
        gap = asr_a - asr_n
        results[name] = {"naive": asr_n, "adaptive": asr_a, "gap": gap, "n": N_SAMPLES}
        print(f"\n  Naive ASR: {asr_n:.2%}  Adaptive ASR: {asr_a:.2%}  Gap: {gap:+.2%}"
              f"  {'✓ real gap' if abs(gap) > 0.02 else '— flat'}")

    return results


def save_results(results, filename):
    os.makedirs("results", exist_ok=True)
    path = f"results/{filename}"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {path}")


def main():
    N_SAMPLES = 200
    print("="*60)
    print("CHECK #4 BATTERY — Real UNSW-NB15 data (d=42)")
    print(f"  gamma={GAMMA_RECALIBRATED} (recalibrated from real epsilon_p)")
    print(f"  N_SAMPLES={N_SAMPLES}, W={W}")
    print("="*60)

    classifier = load_real_surrogate()
    basis, c_base = load_manifold()
    train_X, train_y, test_X, test_y = load_real_data()

    dataset, trajectories, N_SAMPLES = build_dataset_and_trajectories(test_X, test_y, N_SAMPLES, W, classifier)
    print(f"  Dataset: {len(dataset)} benign samples from test set (filtered)")
    print(f"  Trajectories: {len(trajectories)} windows of length {W-1}")

    attacks = [
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

    # Vulnerable basis first
    print(f"\n{'#'*60}")
    print("# CHECK #4 — Vulnerable (attacker-coupled) basis")
    print(f"{'#'*60}")
    vuln_results = run_battery(
        classifier, dataset, trajectories, c_base, basis, CONFIG,
        attacks, N_SAMPLES, mode="vuln"
    )

    # Gate check
    has_gap = any(abs(v["gap"]) > 0.02 for v in vuln_results.values())
    if not has_gap:
        print("\n" + "="*60)
        print("GATE FAILED: No meaningful naive-vs-adaptive gap on vulnerable basis.")
        print("Do NOT proceed to fixed-basis evaluation.")
        print("="*60)
        save_results({"vuln": vuln_results, "gate_passed": False}, "check4_real_data.json")
        return

    # Fixed basis
    print(f"\n{'#'*60}")
    print("# FIXED-BASIS evaluation (gap confirmed on vulnerable basis)")
    print(f"{'#'*60}")
    fixed_results = run_battery(
        classifier, dataset, trajectories, c_base, basis, CONFIG,
        attacks, N_SAMPLES, mode="fixed"
    )

    save_results({
        "vuln": vuln_results,
        "fixed": fixed_results,
        "gate_passed": True,
        "gamma": GAMMA_RECALIBRATED,
        "d": D,
        "w": W,
        "n_samples": N_SAMPLES,
        "data_source": "UNSW-NB15",
    }, "check4_real_data.json")


if __name__ == "__main__":
    main()
