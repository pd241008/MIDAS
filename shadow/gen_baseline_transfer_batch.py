"""
#8 baseline FFI transfer re-validation (mirrors #7 gen_transfer_batch.py).
For each baseline defense (input_smoothing, chen_query_blinding, adv_training),
craft L-inf PGD adversarial inputs (T=50, eps=0.1, alpha=0.01) for naive and
adaptive attackers against that baseline's OWN defended forward, and emit ONE
JSON file per baseline in the exact schema classify_transfer.rs consumes
(d, labels, originals, naive_adversarial, adaptive_adversarial).

Each resulting file is then classified on-device by classify_transfer.rs,
giving the per-baseline ASR on the production FFI TFLite classifier.
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.defense import TrainedSurrogateModel
from shadow.baselines import SmoothedClassifier, ChenBlindingClassifier
from shadow.attacks import project_Lp_ball

D = 42
W = 10
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shadow", "trained_weights")
_T0 = time.time()


def load_real_surrogate():
    model = TrainedSurrogateModel(D, hidden=16)
    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "trained_surrogate_real.pt"), weights_only=True))
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
    trajs = []
    for i in range(N):
        oi = indices[i].item()
        start = max(0, oi - (W - 1))
        traj = [test_X[j] for j in range(start, oi)]
        while len(traj) < W - 1:
            traj.insert(0, filtered_X[0])
        trajs.append(traj)
    labs = (classifier(ds) > 0.5).float()
    return ds, trajs, labs


def forward_for(baseline, classifier, smoothing, blinding, adv_model):
    if baseline == "input_smoothing":
        return lambda x, naive: (classifier(x).clamp(1e-7, 1 - 1e-7) if naive else smoothing(x))
    if baseline == "chen_query_blinding":
        return lambda x, naive: blinding(x, naive)
    if baseline == "adv_training":
        return lambda x, naive: adv_model(x)
    raise ValueError(baseline)


def pgd(x0, y, forward, eps, alpha, steps, naive):
    x_t = x0.clone().detach() + torch.empty_like(x0).uniform_(-eps, eps)
    x_t = project_Lp_ball(x_t, x0, eps)
    for _ in range(steps):
        x_t = x_t.clone().detach().requires_grad_(True)
        pred = forward(x_t, naive).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(pred, y)
        loss.backward()
        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(x_t.grad)
            x_t = project_Lp_ball(x_t, x0, eps)
    return x_t.detach()


def main():
    N = int(os.environ.get("BT_N", "200"))
    EPS = float(os.environ.get("BT_EPS", "0.1"))
    T = int(os.environ.get("BT_T", "50"))
    ALPHA = float(os.environ.get("BT_ALPHA", "0.01"))
    OUTDIR = os.environ.get("BT_OUTDIR", "results/pi/")

    classifier = load_real_surrogate()
    test_X, test_y = load_real_data()
    ds, trajs, labs = build_dataset(test_X, test_y, N, W, classifier)

    smoothing = SmoothedClassifier(classifier, sigma=0.1, K=32)
    blinding = ChenBlindingClassifier(classifier, tau=0.75)
    adv_model = None
    adv_path = os.path.join(WEIGHTS_DIR, "trained_adv_real.pt")
    if os.path.exists(adv_path):
        m = TrainedSurrogateModel(D, hidden=16)
        m.load_state_dict(torch.load(adv_path, weights_only=True))
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()
        adv_model = m
        print("  adv_model loaded from trained_adv_real.pt")

    torch.manual_seed(7)
    names = ["input_smoothing", "chen_query_blinding", "adv_training"]
    if adv_model is None:
        print("  WARNING: trained_adv_real.pt missing -> skipping adv_training batch")
        names.remove("adv_training")
    os.makedirs(OUTDIR, exist_ok=True)
    for bname in names:
        fwd = forward_for(bname, classifier, smoothing, blinding, adv_model)
        naive = []
        adaptive = []
        for i in range(N):
            y = labs[i].float()
            naive.append(pgd(ds[i], y, fwd, EPS, ALPHA, T, True).tolist())
            adaptive.append(pgd(ds[i], y, fwd, EPS, ALPHA, T, False).tolist())
        payload = {
            "d": D, "W": W, "eps": EPS, "T": T, "alpha": ALPHA,
            "n": N,
            "labels": labs.tolist(),
            "originals": ds.tolist(),
            "naive_adversarial": naive,
            "adaptive_adversarial": adaptive,
        }
        fpath = os.path.join(OUTDIR, f"baseline_transfer_{bname}.json")
        with open(fpath, "w") as f:
            json.dump(payload, f)
        print(f"  {bname}: {N} samples -> {fpath} ({time.time()-_T0:.1f}s)")


if __name__ == "__main__":
    main()
