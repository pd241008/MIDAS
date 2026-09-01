"""
#8 Baselines -- measure defense-success (= 1 - ASR) of three SOTA baselines
against the real d=42 UNSW-NB15 surrogate, under the SAME PGD and C&W-L2 attack
suites and SAME eps/gamma used for MIDAS-Edge.

Baselines (shadow/baselines.py):
  - input_smoothing      : SmoothedClassifier(sigma=0.1, K=32)
  - chen_query_blinding  : ChenBlindingClassifier(tau=0.75)
  - adv_training         : adversarially-trained surrogate (PGD eps=0.1 T=50)

Protocol mirrors run_cw_real.py / gen_transfer_batch.py:
  - real d=42 surrogate as the attack target; benign-subset filtered to
    nat_conf <= 0.95; W=10 trajectories; gamma=2.3061.
  - naive/adaptive attacker where the defense is differentiable (MIDAS rotation,
    input-smoothing expectation); adv-training/blinding have no rotational
    parameter -> naive == adaptive (reported as one value).
  - attacker-coupled basis applies only to MIDAS (rotation defense):
    for the baselines there is no manifold rotation to couple to, so we evaluate
    on the input-space attack (no basis). This is the correct fair comparison:
    a baseline has no coupled-basis vulnerability.
  - ASR = fraction of samples whose defended prediction flips vs natural label.
    (Blinding: a rejected/withheld sample is a defense SUCCESS, not a flip.)
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.defense import TrainedSurrogateModel
from shadow.baselines import SmoothedClassifier, ChenBlindingClassifier, train_adv_surrogate
from shadow.attacks import project_Lp_ball

D = 42
W = 10
GAMMA_RECALIBRATED = 2.3061

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
    train_X = torch.tensor(np.load(f"{DATA_DIR}/train_features.npy"), dtype=torch.float32)
    test_X = torch.tensor(np.load(f"{DATA_DIR}/test_features.npy"), dtype=torch.float32)
    test_y = torch.tensor(np.load(f"{DATA_DIR}/test_labels.npy"), dtype=torch.float32)
    return train_X, test_X, test_y


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
        y_nat = (classifier(ds[i]) > 0.5).float().item()
        labs.append(y_nat)
    return ds, trajs, torch.tensor(labs)


# ---------- PGD attack on a generic baseline forward (single defended pred) --
def pgd_baseline(x0, y, forward, eps, alpha, steps, naive):
    x_t = x0.clone().detach() + torch.empty_like(x0).uniform_(-eps, eps)
    x_t = project_Lp_ball(x_t, x0, eps)
    window = None  # baselines are memoryless; no rotation window
    for _ in range(steps):
        x_t = x_t.clone().detach().requires_grad_(True)
        pred = forward(x_t, naive=naive).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(pred, y)
        loss.backward()
        with torch.no_grad():
            x_t = x_t + alpha * torch.sign(x_t.grad)
            x_t = project_Lp_ball(x_t, x0, eps)
    return x_t.detach()


# ---------- C&W-L2 attack on a generic baseline forward ----------------------
def cw_baseline(x0, y, forward, iters, c_steps, lr, initial_c, naive):
    w = (torch.atanh(2.0 * x0.clamp(0.05, 0.95) - 1.0)).clone().detach().requires_grad_(True)
    best = None
    best_dist = None
    for cstep in range(c_steps):
        c = initial_c * (4.0 ** cstep)
        w.data = (torch.atanh(2.0 * x0.clamp(0.05, 0.95) - 1.0)).clone()
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            x_t = 0.5 * (torch.tanh(w) + 1.0)
            pred = forward(x_t, naive=naive).clamp(1e-7, 1 - 1e-7)
            l2 = torch.sum((x_t - x0) ** 2)
            loss = l2 + c * F.binary_cross_entropy(pred, y)
            loss.backward()
            opt.step()
        x_adv = (0.5 * (torch.tanh(w) + 1.0)).detach()
        with torch.no_grad():
            pred = forward(x_adv, naive=False)
            succ = (pred > 0.5).item() != y.item() if not _rejected(pred) else False
        dist = torch.norm(x_adv - x0).item()
        if succ and (best_dist is None or dist < best_dist):
            best = x_adv
            best_dist = dist
    if best is None:
        best = (0.5 * (torch.tanh(w) + 1.0)).detach()
        best_dist = torch.norm(best - x0).item()
    with torch.no_grad():
        pred = forward(best, naive=False)
    succ = False
    if not _rejected(pred):
        succ = (pred > 0.5).item() != y.item()
    # For an undisputed attack-on-input success we also accept a flip-to-reject:
    # if rejection itself was triggered that IS the defense holding.
    return succ, best, best_dist


def _rejected(pred):
    p = float(pred)
    conf = p if p > 0.5 else 1.0 - p
    return conf < 0.75  # matches ChenBlindingClassifier.tau=0.75


def make_forward(baseline, classifier, c_base, basis, config):
    """Return forward(x, naive) -> defended prediction per baseline."""
    def midas_forward(x, na, coupled_basis):
        from shadow.defense import midas_defense_forward, rotate_manifold_fixed_basis
        from shadow.scratch.debug_gradients import midas_defense_forward_vulnerable
        window = [torch.zeros(D)]
        if coupled_basis:
            xd, _ = midas_defense_forward_vulnerable(x, window, c_base, config, naive=na)
        else:
            xd, _ = midas_defense_forward(x, window, c_base, basis, config, naive=na)
        return classifier(xd)
    if baseline == "midas":
        return midas_forward
    if baseline == "input_smoothing":
        return lambda x, na: smoothing(x, na)
    if baseline == "chen_query_blinding":
        return lambda x, na: blinding(x, na) if not na else _blinding_naive(x)
    if baseline == "adv_training":
        return lambda x, na: adv_model(x)
    raise ValueError(baseline)


def main():
    BL = os.environ.get("BL_BASELINE", "input_smoothing")
    ATTACK = os.environ.get("BL_ATTACK", "pgd")
    N = int(os.environ.get("BL_N", "100"))
    DEVICE_ORACLE = os.environ.get("BL_ORACLE", "surrogate")  # "surrogate" or "tflite_native"

    print("="*60)
    print(f"#8 baseline: {BL}  attack={ATTACK}  n={N}")
    print("="*60)

    classifier = load_real_surrogate()
    train_X, test_X, test_y = load_real_data()
    ds, trajs, labs = build_dataset(test_X, test_y, N, W, classifier)
    print(f"  dataset: {len(ds)} benign samples")

    # pre-build baseline objects
    smoothing = None
    blinding = None
    adv_model = None
    if BL == "input_smoothing":
        smoothing = SmoothedClassifier(classifier, sigma=0.1, K=32)
        def smoothing_fwd(x, naive=None):
            if naive:
                # naive attacker ignores smoothing -> attacks base classifier
                return classifier(x).clamp(1e-7, 1 - 1e-7)
            # adaptive attacker differentiates through the MC expectation
            return smoothing(x)
        print("  input_smoothing(sigma=0.1, K=32) ready")
        forward = smoothing_fwd
    elif BL == "chen_query_blinding":
        blinding = ChenBlindingClassifier(classifier, tau=0.75)
        def blinding_fwd(x, naive=None):
            return blinding(x, naive)
        forward = blinding_fwd
    elif BL == "adv_training":
        save = os.environ.get("BL_ADV_SAVE", "shadow/trained_weights/trained_adv_real.pt")
        if os.path.exists(save):
            m = TrainedSurrogateModel(D, hidden=16)
            m.load_state_dict(torch.load(save, weights_only=True))
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
            adv_model = m
        else:
            adv_model, _ = train_adv_surrogate(
                d=D, n_epochs=int(os.environ.get("BL_ADV_EPOCHS", "200")),
                T=int(os.environ.get("BL_ADV_T", "10")))
            torch.save(adv_model.state_dict(), save)
        forward = lambda x, naive=None: adv_model(x)
        print("  adv_training ready")
    else:
        raise ValueError(BL)

    # evaluate
    c_base = torch.tensor(np.load(f"{DATA_DIR}/c_base_real.npy"), dtype=torch.float32)
    basis = torch.tensor(np.load(f"{DATA_DIR}/basis_real.npy"), dtype=torch.float32) if os.path.exists(f"{DATA_DIR}/basis_real.npy") else None

    t0 = time.time()
    flips_naive = 0
    flips_ada = 0
    rejected_naive = 0
    rejected_ada = 0
    if ATTACK == "pgd":
        eps = float(os.environ.get("BL_EPS", "0.1"))
        alpha = float(os.environ.get("BL_ALPHA", "0.01"))
        T = int(os.environ.get("BL_T", "50"))
        for i in range(N):
            y_nat = labs[i]
            x_adv_naive = pgd_baseline(ds[i], y_nat, forward, eps, alpha, T, naive=True)
            x_adv_ada = pgd_baseline(ds[i], y_nat, forward, eps, alpha, T, naive=False)
            with torch.no_grad():
                pn = forward(x_adv_naive, False)
                pa = forward(x_adv_ada, False)
            if BL == "chen_query_blinding":
                if _rejected(pn):
                    rejected_naive += 1
                elif (pn > 0.5).item() != y_nat.item():
                    flips_naive += 1
                if _rejected(pa):
                    rejected_ada += 1
                elif (pa > 0.5).item() != y_nat.item():
                    flips_ada += 1
            else:
                if (pn > 0.5).item() != y_nat.item():
                    flips_naive += 1
                if (pa > 0.5).item() != y_nat.item():
                    flips_ada += 1
    elif ATTACK == "cw":
        iters = int(os.environ.get("BL_ITERS", "100"))
        c_steps = int(os.environ.get("BL_C_STEPS", "3"))
        lr = float(os.environ.get("BL_LR", "0.01"))
        initial_c = float(os.environ.get("BL_INITIAL_C", "1.0"))
        for i in range(N):
            y_nat = labs[i]
            _, xa_naive, _ = cw_baseline(ds[i], y_nat, forward, iters, c_steps, lr, initial_c, naive=True)
            _, xa_ada, _ = cw_baseline(ds[i], y_nat, forward, iters, c_steps, lr, initial_c, naive=False)
            with torch.no_grad():
                pn = forward(xa_naive, False)
                pa = forward(xa_ada, False)
            is_blind = (BL == "chen_query_blinding")
            for p, fname, rname in ((pn, "n", "nr"), (pa, "a", "ar")):
                if is_blind and _rejected(p):
                    if rname == "nr":
                        rejected_naive += 1
                    else:
                        rejected_ada += 1
                elif (p > 0.5).item() != y_nat.item():
                    if fname == "n":
                        flips_naive += 1
                    else:
                        flips_ada += 1
    else:
        raise ValueError(ATTACK)
    dt = time.time() - t0

    out = {
        "baseline": BL, "attack": ATTACK, "gamma": GAMMA_RECALIBRATED,
        "n": N,
        "surrogate_asr_naive": flips_naive / N,
        "surrogate_asr_adaptive": flips_ada / N,
        "surrogate_defense_success_naive": 1 - flips_naive / N,
        "surrogate_defense_success_adaptive": 1 - flips_ada / N,
        "rejected_naive": rejected_naive,
        "rejected_adaptive": rejected_ada,
        "naive_eq_adaptive_note": ("Baselines with no rotational manifold parameter "
                                   "(chen_query_blinding, adv_training) have naive==adaptive; "
                                   "only input_smoothing has a naive-as-detached-expectation split."),
        "elapsed_seconds": dt,
    }
    if ATTACK == "pgd":
        out["eps"] = eps
        out["alpha"] = alpha
        out["T"] = T
    else:
        out["iters"] = iters
        out["c_steps"] = c_steps
        out["lr"] = lr
        out["initial_c"] = initial_c
    out_name = os.environ.get("BL_OUT", f"results/pi/baseline_{BL}_{ATTACK}.json")
    os.makedirs(os.path.dirname(out_name), exist_ok=True)
    with open(out_name, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"  saved {out_name} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
