"""
C&W-L2 attacker battery on real UNSW-NB15 data (d=42).
- Loads the real trained surrogate (d=42, hidden=16)
- Real UNSW-NB15 benign samples + trajectories
- Naive vs adaptive attacker, on both fixed PCA-2 basis and attacker-coupled basis
- ASR = fraction of samples whose defended prediction flips after the C&W-L2 perturbation
"""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.defense import TrainedSurrogateModel
from shadow.cw_attack import cw_l2_attack

D = 42
W = 10
GAMMA_RECALIBRATED = 2.3061  # benign P95 (recalibrated, was 1.984 P75)

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

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shadow", "trained_weights")


def load_real_surrogate():
    model = TrainedSurrogateModel(D, hidden=16)
    path = os.path.join(WEIGHTS_DIR, "trained_surrogate_real.pt")
    model.load_state_dict(torch.load(path, weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def load_manifold():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "results", "basis.json")) as f:
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
    benign_mask = test_y == 0
    benign_X = test_X[benign_mask]
    with torch.no_grad():
        probs = classifier(benign_X)
        nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
        keep_mask = nat_conf.squeeze() <= 0.95
        filtered_X = benign_X[keep_mask]
        benign_indices = torch.where(benign_mask)[0]
        filtered_orig_indices = benign_indices[keep_mask]

    if len(filtered_X) < N_SAMPLES:
        print(f"  WARNING: only {len(filtered_X)} survive filter, need {N_SAMPLES}; using all")
        N_SAMPLES = len(filtered_X)

    dataset = filtered_X[:N_SAMPLES]
    trajectories = []
    for i in range(N_SAMPLES):
        orig_idx = filtered_orig_indices[i].item()
        start = max(0, orig_idx - (W - 1))
        traj = [test_X[j] for j in range(start, orig_idx)]
        while len(traj) < W - 1:
            traj.insert(0, filtered_X[0])
        trajectories.append(traj)
    return dataset, trajectories, N_SAMPLES


def run_battery(classifier, dataset, trajectories, c_base, basis, config,
                N_SAMPLES, coupled_basis, naive, cw_kwargs):
    succ = 0
    dists = []
    for i, (x0, traj) in enumerate(zip(dataset, trajectories)):
        is_first = (i == 0)
        with torch.no_grad():
            y_nat = (classifier(x0) > 0.5).float()
        success, _, dist, best_c = cw_l2_attack(
            x0, y_nat, classifier, traj, c_base, basis, config,
            naive=naive, coupled_basis=coupled_basis,
            first_sample=is_first, **cw_kwargs
        )
        if success:
            succ += 1
        if dist is not None:
            dists.append(dist)
    return succ / N_SAMPLES, dists


def main():
    N_SAMPLES = int(os.environ.get("CW_N_SAMPLES", "60"))
    ITERS = int(os.environ.get("CW_ITERS", "100"))
    C_STEPS = int(os.environ.get("CW_C_STEPS", "3"))
    LR = float(os.environ.get("CW_LR", "0.01"))
    INITIAL_C = float(os.environ.get("CW_INITIAL_C", "1.0"))
    cw_kwargs = {"iters": ITERS, "c_steps": C_STEPS, "lr": LR, "initial_c": INITIAL_C}

    print("="*60)
    print(f"C&W-L2 battery — real UNSW-NB15 d=42")
    print(f"  N_SAMPLES={N_SAMPLES} W={W} gamma={GAMMA_RECALIBRATED}")
    print(f"  iters={ITERS} c_steps={C_STEPS} lr={LR} initial_c={INITIAL_C}")
    print("="*60)

    classifier = load_real_surrogate()
    basis, c_base = load_manifold()
    train_X, train_y, test_X, test_y = load_real_data()
    dataset, trajectories, N_SAMPLES = build_dataset_and_trajectories(test_X, test_y, N_SAMPLES, W, classifier)
    print(f"  dataset: {len(dataset)} benign samples (filtered)")

    results = {}
    for coupled in (True, False):
        basis_name = "attacker-coupled" if coupled else "fixed"
        for naive in (True, False):
            attacker = "naive" if naive else "adaptive"
            label = f"{basis_name} / {attacker}"
            print(f"\n--- {label} ---")
            t0 = time.time()
            asr, dists = run_battery(classifier, dataset, trajectories, c_base, basis,
                                     CONFIG, N_SAMPLES, coupled, naive, cw_kwargs)
            dt = time.time() - t0
            med_dist = float(np.median(dists)) if dists else None
            mean_dist = float(np.mean(dists)) if dists else None
            print(f"  ASR = {asr:.3f}  median_L2={med_dist:.5f}  mean_L2={mean_dist:.5f}  ({dt:.1f}s)")
            results[label] = {
                "asr": asr, "n": N_SAMPLES,
                "median_l2_dist": med_dist, "mean_l2_dist": mean_dist,
                "iters": ITERS, "c_steps": C_STEPS, "lr": LR, "initial_c": INITIAL_C,
            }

    out_name = os.environ.get("CW_OUT", "results/pi/cw_l2_pi.json")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    with open(out_name, "w") as f:
        json.dump({"gamma": GAMMA_RECALIBRATED, "d": D, "w": W,
                   "n_samples": N_SAMPLES, "data_source": "UNSW-NB15",
                   "attacker": "adaptive through-defense (coupled) gradient",
                   "results": results}, f, indent=2)
    print(f"\nSaved to {out_name}")


if __name__ == "__main__":
    main()
