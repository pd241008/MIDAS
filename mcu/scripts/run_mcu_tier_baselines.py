"""
MCU-tier baseline battery: measure defense-success (= 1 - ASR) of four
defenses against the DEPLOYED per-source INT8 specialists (NSL / UNSW), under
the same PGD + C&W-L2 suites used on the Edge tier.

Baselines (port of shadow/baselines.py; evaluated at d=12 on the int8 replica):
  - undefended           : Int8SpecialistReplica alone (deployed TFLM boundary)
  - dacm                 : the MIDAS rotation defense (windowed penetration eps
                           -> Givens rotation angle in the deployed fixed PCA-2
                           basis). Stateful: the window grows along the attack
                           trajectory exactly as mcu/scripts/adaptive_attacker_gate.py.
                           naive = theta/basis detached; adaptive = full flow.
  - input_smoothing      : SmoothedClassifier(sigma=0.1, K=32) around the replica
  - chen_query_blinding  : ChenBlindingClassifier(tau=0.75) around the replica
  - adv_training         : adversarially-trained d=12 MLP (12->16->1 tanh/sigmoid,
                           the specialist architecture), PGD L-inf eps=0.1 T=10
                           augmentation (Madry-style), cached per source.

Protocol mirrors shadow/run_baselines.py + adaptive_attacker_gate.py:
  - dataset: benign rows with natural confidence <= 0.95, real sliding
    W=10 trajectory windows from mcu/features unified files.
  - PGD (eps=0.1, alpha=0.01, T=50); C&W-L2 (iters=100, c_steps=3, lr=0.01,
    initial_c=1.0), both against the DEFENDED forward.
  - naive vs adaptive split: only where the defense is differentiable (dacm,
    input_smoothing). chen/adv_training/undefended -> single value.
  - Chen rejection = withheld -> defense SUCCESS (not a flip), matching the
    Edge ASR convention; flips counted on the final defended prediction.

Emits:
  - mcu/features/mcu_tier_baselines.json   (stable, table-friendly summary)
  - mcu/results/mcu_tier_baselines_<ts>.json  (full provenance via save_results)
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_attacker_gate import (  # noqa: E402
    Int8SpecialistReplica, defense_forward, project_Lp,
)
from save_results import save_results  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")
RESULTS_DIR = os.path.join(MCU_DIR, "results")
ADV_CACHE = os.path.join(FEATURES_DIR, "adv_trained_specialists")

D = 12
W = 10
DELTA_THETA_MAX_DEG = 45.0
PGD = {"eps": 0.1, "alpha": 0.01, "T": 50}
CW = {"iters": 100, "c_steps": 3, "lr": 0.01, "initial_c": 1.0}


def load_gamma():
    with open(os.path.join(FEATURES_DIR, "gamma_params.json")) as f:
        return json.load(f)["gamma_per_source"]


def load_manifold():
    with open(os.path.join(FEATURES_DIR, "manifold_basis.json")) as f:
        data = json.load(f)
    return (torch.tensor(data["basis"], dtype=torch.float32),
            torch.tensor(data["c_base"], dtype=torch.float32))


def load_source(prov):
    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy")).astype(np.float32)
    y = np.load(os.path.join(FEATURES_DIR, "unified_labels.npy")).astype(np.float32)
    p = np.load(os.path.join(FEATURES_DIR, "unified_provenance.npy"))
    mask = p == prov
    return torch.tensor(X[mask]), torch.tensor(y[mask])


def build_dataset(Xs, ys, classifier, n_samples, W):
    benign_idx = torch.where(ys == 0)[0]
    with torch.no_grad():
        probs = classifier(Xs[benign_idx])
    nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
    keep = nat_conf.squeeze() <= 0.95
    benign_idx = benign_idx[keep]
    dataset, trajectories = [], []
    for k in range(len(benign_idx)):
        i = int(benign_idx[k].item())
        if i < W - 1:
            continue
        dataset.append(Xs[i])
        trajectories.append([Xs[j] for j in range(i - (W - 1), i)])
        if len(dataset) >= n_samples:
            break
    return dataset, trajectories


# ---------------------------------------------------------------------------
# Baseline wrappers
# ---------------------------------------------------------------------------
class SmoothedClassifier(nn.Module):
    def __init__(self, base, sigma=0.1, K=32, seed=42):
        super().__init__()
        self.base, self.sigma, self.K = base, sigma, K
        self._g = torch.Generator().manual_seed(seed)
        self.base.eval()

    def forward(self, x):
        x_shape = x.shape
        xb = x.expand(self.K, *x_shape)
        noise = torch.randn(self.K, *x_shape, generator=self._g, dtype=x.dtype)
        preds = self.base((xb + self.sigma * noise).reshape(-1, x_shape[-1]))
        return preds.reshape(self.K, *x_shape[:-1]).mean(dim=0).clamp(1e-7, 1 - 1e-7)


class ChenBlindingClassifier(nn.Module):
    def __init__(self, base, tau=0.75):
        super().__init__()
        self.base, self.tau = base, tau
        self.base.eval()

    def forward(self, x):
        p = self.base(x).squeeze(-1)
        conf = torch.where(p > 0.5, p, 1.0 - p)
        reject = (conf < self.tau).float()
        return (1.0 - reject) * p + reject * torch.full_like(p, 0.5)


def _rejected(p):
    conf = p if p > 0.5 else 1.0 - p
    return conf < 0.75


# ---------------------------------------------------------------------------
# Adv-trained specialist (12->16->1, same architecture as deployed)
# ---------------------------------------------------------------------------
class _AdvMLP(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(D, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x)))).squeeze(-1)


def _pgd_augment(model, x, y, eps, alpha, steps):
    x0 = x.clone().detach()
    xa = (x0 + torch.empty_like(x0).uniform_(-eps, eps)).clamp(0.0, 1.0)
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        pred = model(xa).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(pred, y.float())
        loss.backward()
        with torch.no_grad():
            xa = (xa + alpha * torch.sign(xa.grad)).clamp(0.0, 1.0)
    return xa.detach()


def train_adv_specialist(X, y, n_epochs=25, lr=0.01, batch=512, eps=0.1,
                         alpha=0.01, T=10, adv_epoch=3, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    Xt, yt = X[perm[:n_train]], y[perm[:n_train]]
    Xv, yv = X[perm[n_train:]], y[perm[n_train:]]
    model = _AdvMLP()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    best_acc, best_sd = 0.0, None
    n_batches = (len(Xt) + batch - 1) // batch
    for epoch in range(n_epochs):
        model.train()
        order = torch.randperm(len(Xt))
        for b in range(n_batches):
            idx = order[b * batch:(b + 1) * batch]
            xb, yb = Xt[idx], yt[idx]
            if (epoch + 1) % adv_epoch == 0:
                xb = _pgd_augment(model, xb, yb, eps, alpha, T)
            pred = model(xb).clamp(1e-7, 1 - 1e-7)
            loss = loss_fn(pred, yb.float())
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = float(((model(Xv) > 0.5).float() == yv.float()).float().mean())
        if acc > best_acc:
            best_acc, best_sd = acc, {k: v.clone() for k, v in model.state_dict().items()}
    if best_sd is not None:
        model.load_state_dict(best_sd)
    model.eval()
    return model


def get_adv_specialist(tag, Xs, ys):
    torch.manual_seed(42)
    os.makedirs(ADV_CACHE, exist_ok=True)
    path = os.path.join(ADV_CACHE, f"adv_{tag}.pt")
    if os.path.exists(path):
        m = _AdvMLP()
        m.load_state_dict(torch.load(path, weights_only=True))
        m.eval()
        return m
    print(f"  [training adv specialist {tag}]")
    m = train_adv_specialist(Xs, ys)
    torch.save(m.state_dict(), path)
    return m


# ---------------------------------------------------------------------------
# Defense objects: each exposes reset(traj) and forward(x, naive)->scalar
# ---------------------------------------------------------------------------
class DACMDefense(nn.Module):
    """Stateful MIDAS rotation defense (deployed fixed PCA-2 basis).

    Window grows along the attack trajectory (W-1 prior rows + each candidate),
    matching adaptive_attacker_gate.pgd_attack's sliding convention.
    """
    def __init__(self, classifier, c_base, basis, cfg):
        super().__init__()
        self.classifier, self.c_base, self.basis = classifier, c_base, basis
        self.cfg = cfg
        self.win = []
        self.naive_eq_adaptive = False
        self.reject_based = False

    def reset(self, traj):
        self.win = [w.clone().detach() for w in traj]

    def forward(self, x, naive=False):
        # ws = W-1 prior rows (trajectory + previous perturbed candidates);
        # defense_forward appends the current x itself. Then slide x in.
        ws = self.win[-(self.cfg["W"] - 1):]
        xd, _ = defense_forward(x, ws, self.c_base, self.basis, self.cfg,
                                naive=naive, vulnerable=False)
        self.win.append(x.detach().clone())
        return self.classifier(xd)


class WrapperDefense(nn.Module):
    """Memoryless defense (smoothing / blinding / undefended / adv-trained)."""
    def __init__(self, fwd, naive_eq_adaptive=False, reject_based=False):
        super().__init__()
        self.fwd = fwd
        self.naive_eq_adaptive = naive_eq_adaptive
        self.reject_based = reject_based

    def reset(self, traj):
        pass

    def forward(self, x, naive=False):
        return self.fwd(x, naive)


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------
def pgd_attack(x0, y, defense, eps, alpha, steps, naive):
    x = (x0 + torch.empty_like(x0).uniform_(-eps, eps))
    x = project_Lp(x, x0, eps)
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        pred = defense.forward(x, naive).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(pred, y)
        loss.backward()
        with torch.no_grad():
            x = x + alpha * torch.sign(x.grad)
            x = project_Lp(x, x0, eps)
    return x.detach()


def cw_attack(x0, y, defense, iters, c_steps, lr, initial_c, naive):
    w = (torch.atanh(2.0 * x0.clamp(0.05, 0.95) - 1.0)).clone().detach().requires_grad_(True)
    best = best_dist = None
    for cstep in range(c_steps):
        c = initial_c * (4.0 ** cstep)
        w.data = (torch.atanh(2.0 * x0.clamp(0.05, 0.95) - 1.0)).clone()
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            x_t = 0.5 * (torch.tanh(w) + 1.0)
            pred = defense.forward(x_t, naive).clamp(1e-7, 1 - 1e-7)
            l2 = torch.sum((x_t - x0) ** 2)
            loss = l2 + c * F.binary_cross_entropy(pred, y)
            loss.backward()
            opt.step()
        x_a = (0.5 * (torch.tanh(w) + 1.0)).detach()
        with torch.no_grad():
            p = float(defense.forward(x_a, False))
        succ = (not _rejected(p)) and ((p > 0.5) != (y.item() > 0.5))
        d = torch.norm(x_a - x0).item()
        if succ and (best_dist is None or d < best_dist):
            best, best_dist = x_a, d
    if best is None:
        best = x_a
    return best, float(defense.forward(best, False))


def flip_success(p, y, reject_based):
    if reject_based and _rejected(p):
        return False  # withheld by query-blinding -> defense holds
    return (p > 0.5) != (y.item() > 0.5)


def run_battery(defense, dataset, trajectories, classifier, attack, n):
    asrs = {"naive": 0, "adaptive": 0}
    for x0, traj in zip(dataset, trajectories):
        x0 = x0.detach()
        with torch.no_grad():
            y = (classifier(x0) > 0.5).float()
        for key, naive in (("adaptive", False), ("naive", True)):
            defense.reset(traj)
            if attack == "pgd":
                xa = pgd_attack(x0, y, defense, PGD["eps"], PGD["alpha"], PGD["T"], naive)
                with torch.no_grad():
                    p = float(defense.forward(xa, False))
            else:
                xa, p = cw_attack(x0, y, defense, CW["iters"], CW["c_steps"],
                                  CW["lr"], CW["initial_c"], naive)
            if flip_success(p, y, defense.reject_based):
                asrs[key] += 1
    return {k: v / n for k, v in asrs.items()}


def main():
    torch.manual_seed(42)
    gammas = load_gamma()
    basis, c_base = load_manifold()
    cfg = {"W": W, "gamma": 0.0, "lambda": 1.0, "k": 2.0,
           "delta_theta_max": DELTA_THETA_MAX_DEG * (3.141592653589793 / 180.0)}
    n_samples = int(os.environ.get("BL_N", "100"))
    attacks = [a for a in ("pgd", "cw") if a in os.environ.get("BL_ATTACKS", "pgd,cw").split(",")]

    report = {"experiment": "MCU-tier baselines vs deployed per-source INT8 specialists",
              "defense_success": "1 - ASR (flip of defended prediction on the final defended forward)",
              "d": D, "W": W, "n": n_samples, "pgd": PGD, "cw": CW,
              "note": "naive==adaptive for defenses with no differentiable manifold "
                      "parameter (undefended, chen_query_blinding, adv_training); dacm and "
                      "input_smoothing report the split. Chen rejection = defense success."}

    for tag, prov in [("nsl", 0), ("unsw", 1)]:
        gamma = gammas[tag]
        cfg["gamma"] = gamma
        print(f"\n===== {tag.upper()} (gamma={gamma:.4f}) =====")
        classifier = Int8SpecialistReplica(tag, gamma)
        classifier.eval()
        Xs, ys = load_source(prov)
        dataset, trajectories = build_dataset(Xs, ys, classifier, n_samples, W)
        n = len(dataset)
        print(f"  benign samples: {n}")

        dacm = DACMDefense(classifier, c_base, basis, cfg)
        smoothing = WrapperDefense(lambda x, na: (classifier(x) if na else
                                                  SmoothedClassifier(classifier)(x)), naive_eq_adaptive=False)
        blinding = WrapperDefense(lambda x, na: ChenBlindingClassifier(classifier)(x),
                                  naive_eq_adaptive=True, reject_based=True)
        adv_model = get_adv_specialist(tag, Xs, ys)
        adv = WrapperDefense(lambda x, na: adv_model(x), naive_eq_adaptive=True)
        undef = WrapperDefense(lambda x, na: classifier(x), naive_eq_adaptive=True)

        for attack in attacks:
            src = {}
            t0 = time.time()
            print(f"  --- {attack.upper()} (eps={PGD['eps']}/T={PGD['T']} or iters={CW['iters']}) ---")
            for name, defn in [("undefended", undef), ("dacm", dacm),
                               ("input_smoothing", smoothing),
                               ("chen_query_blinding", blinding),
                               ("adv_training", adv)]:
                asrs = run_battery(defn, dataset, trajectories, classifier, attack, n)
                if defn.naive_eq_adaptive:
                    ds = {"naive": 1 - asrs["adaptive"], "adaptive": 1 - asrs["adaptive"]}
                else:
                    ds = {"naive": 1 - asrs["naive"], "adaptive": 1 - asrs["adaptive"]}
                line = ", ".join(f"{k}={v:.3f}" for k, v in ds.items())
                print(f"    {name:<22s} defense_success: {line}")
                src[name] = {"asr": asrs, "defense_success": ds}
            report.setdefault(tag, {})["attack_" + attack] = src
            report[tag]["attack_" + attack + "_elapsed_s"] = round(time.time() - t0, 1)

    table = {}
    for tag in ("nsl", "unsw"):
        for atk in attacks:
            pref = "PGD" if atk == "pgd" else "C&W"
            r = report[tag]["attack_" + atk]
            table[f"{pref} / {tag}"] = {
                "undefended": r["undefended"]["defense_success"]["adaptive"],
                "dacm": r["dacm"]["defense_success"]["adaptive"],
                "dacm_naive": r["dacm"]["defense_success"]["naive"],
                "input_smoothing": r["input_smoothing"]["defense_success"]["adaptive"],
                "chen_query_blinding": r["chen_query_blinding"]["defense_success"]["adaptive"],
                "adv_training": r["adv_training"]["defense_success"]["adaptive"],
            }

    with open(os.path.join(FEATURES_DIR, "mcu_tier_baselines.json"), "w") as f:
        json.dump({"summary": table, "full": report}, f, indent=2)
    print(f"\nsaved {os.path.join(FEATURES_DIR, 'mcu_tier_baselines.json')}")
    print(json.dumps(table, indent=2))
    save_results("run_mcu_tier_baselines.py",
                 config={"pgd": PGD, "cw": CW, "W": W, "d": D, "n": n_samples},
                 results=report)


if __name__ == "__main__":
    main()