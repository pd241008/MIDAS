"""
Generate transfer-adversarial batches for #7:
- Reuses the real surrogate (d=42, W=10) PGD harness (attacker-coupled basis)
- For naive and adaptive attackers, craft L-inf PGD adversarial inputs on a
  sample subset, and export {samples, labels, group} as JSON for the on-device
  FFI classifier to evaluate (transfer verification vs the production model).
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.defense import TrainedSurrogateModel
from shadow.attacks import project_Lp_ball
from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable

D = 42
W = 10
GAMMA_RECALIBRATED = 2.3061  # benign P95 (recalibrated, was 1.984 P75)
CONFIG = {
    "W": W, "D": D, "gamma": GAMMA_RECALIBRATED, "lambda": 1.0, "k": 2.0,
    "delta_theta_max_deg": 45.0, "tau": 0.75, "sla_budget_ms": 10,
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


def load_real_data():
    test_X = torch.tensor(np.load(f"{DATA_DIR}/test_features.npy"), dtype=torch.float32)
    test_y = torch.tensor(np.load(f"{DATA_DIR}/test_labels.npy"), dtype=torch.float32)
    return test_X, test_y


def build_dataset(test_X, test_y, N, W, classifier):
    benign_mask = test_y == 0
    benign_X = test_X[benign_mask]
    with torch.no_grad():
        probs = classifier(benign_X)
        nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
        keep = nat_conf.squeeze() <= 0.95
        filtered_X = benign_X[keep]
        indices = torch.where(benign_mask)[0][keep]
    ds = filtered_X[:N]
    trajs, labs = [], []
    for i in range(N):
        oi = indices[i].item()
        start = max(0, oi - (W - 1))
        traj = [test_X[j] for j in range(start, oi)]
        while len(traj) < W - 1:
            traj.insert(0, filtered_X[0])
        trajs.append(traj)
        labs.append((classifier(ds[i]) > 0.5).float().item())
    return ds, trajs, torch.tensor(labs)


def pgd_coupled(x0, y, classifier, traj, c_base, config, alpha, epsilon, steps, naive):
    """mirror pgd_vuln; returns adversarial x (detached)."""
    x_t = x0.clone().detach()
    x_t = x_t + torch.empty_like(x_t).uniform_(-epsilon, epsilon)
    x_t = project_Lp_ball(x_t, x0, epsilon)
    sliding_window = [w.clone().detach() for w in traj]
    Wl = config.get("W", W)
    for _ in range(steps):
        x_t.requires_grad_(True)
        window_slice = sliding_window[-(Wl - 1):]
        x_def, _ = midas_defense_forward_vulnerable(x_t, window_slice, c_base, config, naive=naive)
        pred = classifier(x_def).clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(pred, y)
        loss.backward()
        grad = x_t.grad
        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(grad)
            x_t = project_Lp_ball(x_t, x0, epsilon)
            sliding_window.append(x_t.clone().detach())
    return x_t.detach()


def main():
    N = int(os.environ.get("XFER_N", "100"))
    EPS = float(os.environ.get("XFER_EPS", "0.1"))
    T = int(os.environ.get("XFER_T", "50"))
    ALPHA = float(os.environ.get("XFER_ALPHA", "0.01"))
    out = os.environ.get("XFER_OUT", "results/pi/transfer_batch.json")

    print(f"Generating transfer batch: N={N} eps={EPS} T={T} alpha={ALPHA} (attacker-coupled basis)")
    classifier = load_real_surrogate()
    test_X, test_y = load_real_data()
    c_base = torch.tensor(np.load(f"{DATA_DIR}/c_base_real.npy"), dtype=torch.float32)

    ds, trajs, labs = build_dataset(test_X, test_y, N, W, classifier)
    print(f"  dataset: {N} benign samples (labels sum={labs.sum().item()})")

    torch.manual_seed(7)
    samples = {"naive": [], "adaptive": [], "labels": [], "originals": []}
    for i in range(N):
        y = labs[i].float()
        for naive in (True, False):
            x_adv = pgd_coupled(ds[i], y, classifier, trajs[i], c_base, CONFIG,
                                ALPHA, EPS, T, naive)
            samples["naive" if naive else "adaptive"].append(x_adv.tolist())
        samples["labels"].append(labs[i].item())
        samples["originals"].append(ds[i].tolist())

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "d": D, "W": W, "eps": EPS, "T": T, "alpha": ALPHA,
            "n": N, "c_base": c_base.tolist(),
            "labels": samples["labels"],
            "originals": samples["originals"],
            "naive_adversarial": samples["naive"],
            "adaptive_adversarial": samples["adaptive"],
        }, f)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
