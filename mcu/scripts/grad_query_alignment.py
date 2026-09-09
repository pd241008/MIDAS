"""
Gradient-to-query alignment for the coupled-basis (vulnerable) rotation defense.

For each benign sample x0 (same N=200 set and eps0.1_50 config as the gate),
run the adaptive PGD trajectory and, at each step, project the loss gradient
grad_x J onto the attacker's live coupled subspace span{v_t, v_t_perp} where
v_t = (x - c_base)/||x - c_base|| (the vulnerable basis). Report:

  - mean |cos(grad, v_t)|                       (query-direction alignment)
  - mean projected-energy fraction onto span    (|p0|^2 + |p1|^2) / |g|^2

USW vs UNSW over the same attack batches. If UNSW's gradient energy is mostly
orthogonal to the query direction, that geometrically explains why the
coupled-basis adaptive advantage registers on NSL but not UNSW.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adaptive_attacker_gate import (  # noqa: E402
    Int8SpecialistReplica, load_source, load_manifold, load_gamma,
    defense_forward, project_Lp, DELTA_THETA_MAX_DEG,
)
from save_results import save_results  # noqa: E402

D = 12
W_MAX = 10
N_SAMPLES = int(os.environ.get("GA_N", "200"))
EPS = float(os.environ.get("GA_EPS", "0.1"))
ALPHA = 0.01
STEPS = int(os.environ.get("GA_STEPS", "50"))
RECORD_EVERY = max(1, STEPS // 10)
FEATURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features")


def build_dataset(Xs, ys, classifier, n_samples, w):
    benign_idx = torch.where(ys == 0)[0]
    with torch.no_grad():
        probs = classifier(Xs[benign_idx])
    nat_conf = torch.where(probs > 0.5, probs, 1.0 - probs)
    keep = nat_conf.squeeze() <= 0.95
    benign_idx = benign_idx[keep]
    dataset, traj = [], []
    for k in range(len(benign_idx)):
        i = int(benign_idx[k].item())
        if i < w - 1:
            continue
        dataset.append(Xs[i])
        traj.append([Xs[j] for j in range(i - (w - 1), i)])
        if len(dataset) >= n_samples:
            break
    return dataset, traj


def gradient_alignment_sweep(classifier, dataset, traj_wmax, c_base, cfg, w):
    """Run adaptive PGD; record grad-to-query alignment at sampled steps."""
    cos_v = []      # |cos(grad, v_t)|
    energy = []     # projected-energy fraction onto span{v_t, v_t_perp}
    n_steps = 0
    for x0, traj0 in zip(dataset, traj_wmax):
        traj = traj0[-(w - 1):] if w - 1 <= len(traj0) else traj0[:]
        x = x0.clone().detach()
        x = x + torch.empty_like(x).uniform_(-EPS, EPS)
        x = project_Lp(x, x0, EPS)
        win = [z.clone().detach() for z in traj]
        for _ in range(STEPS):
            x = x.detach().requires_grad_(True)
            ws = win[-(cfg["W"] - 1):]
            # adaptive: gradient flows through the coupled basis (vulnerable=True)
            g = torch.zeros_like(x)
            xd, _ = defense_forward(x, ws, c_base, None, cfg,
                                    naive=False, vulnerable=True)
            pred = classifier(xd).clamp(1e-7, 1 - 1e-7)
            y = (classifier(x0) > 0.5).float()
            loss = F.binary_cross_entropy(pred, y.expand_as(pred))
            loss.backward()
            g = x.grad.detach()
            v = vulnerable_basis(x, c_base)
            b0 = v[0]                      # query direction
            b1 = v[1]
            g0 = torch.sum(g * b0)
            g1 = torch.sum(g * b1)
            gnorm = torch.norm(g)
            if gnorm > 1e-9:
                cos_v.append(abs(g0 / gnorm).item())
                energy.append(((g0 ** 2 + g1 ** 2) / (gnorm ** 2)).item())
            n_steps += 1
            with torch.no_grad():
                x = x.detach() + ALPHA * torch.sign(x.grad)
                x = project_Lp(x, x0, EPS)
                win.append(x.clone().detach())
    return {"mean_abs_cos_grad_v": float(np.mean(cos_v)),
            "mean_projected_energy_fraction": float(np.mean(energy)),
            "n_samples": len(dataset), "n_steps": n_steps}

def vulnerable_basis(x, c_base):
    dim = x.shape[0]
    b0 = x - c_base
    n0 = torch.norm(b0, p=2)
    b0 = b0 / n0 if n0 > 1e-7 else torch.zeros_like(x)
    b1 = torch.zeros_like(x); b1[1] = 1.0 if dim > 1 else b1[0] + 1.0
    d = torch.sum(b1 * b0); b1 = b1 - d * b0
    n1 = torch.norm(b1, p=2)
    b1 = b1 / n1 if n1 > 1e-7 else torch.zeros_like(x)
    return torch.stack([b0, b1])


def main():
    torch.manual_seed(42)
    gammas = load_gamma()
    _, c_base = load_manifold()
    cfg = {"W": W_MAX, "gamma": 0.0, "lambda": 1.0, "k": 2.0,
           "delta_theta_max": DELTA_THETA_MAX_DEG * (3.141592653589793 / 180.0)}
    report = {"experiment": "Gradient-to-query alignment (coupled basis) per specialist",
              "note": "mean |cos(grad_x J, v_t)| and projected-energy fraction onto "
                      "span{v_t, v_perp} over the adaptive PGD trajectory (eps=%g,T=%d) "
                      "on the same N=%d benign set as the gate." % (EPS, STEPS, N_SAMPLES),
              "config": {"eps": EPS, "alpha": ALPHA, "T": STEPS, "N": N_SAMPLES}}
    for tag, prov in [("nsl", 0), ("unsw", 1)]:
        gamma = gammas[tag]
        cfg["gamma"] = gamma
        classifier = Int8SpecialistReplica(tag, gamma)
        classifier.eval()
        Xs, ys = load_source(prov)
        dataset, traj = build_dataset(Xs, ys, classifier, N_SAMPLES, W_MAX)
        r = gradient_alignment_sweep(classifier, dataset, traj, c_base, cfg, W_MAX)
        print(f"\n===== {tag.upper()} (gamma={gamma:.4f}) =====")
        print(f"  mean |cos(grad, v_t)|       = {r['mean_abs_cos_grad_v']:.4f}")
        print(f"  mean proj-energy into span  = {r['mean_projected_energy_fraction']:.4f} "
              f"({r['n_samples']} samples x {r['n_steps']} steps)")
        report[tag] = dict(r, gamma=gamma)

    out = os.path.join(FEATURES_DIR, "grad_query_alignment.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {out}")
    save_results("grad_query_alignment.py", config=report["config"], results=report)


import json  # noqa: E402


if __name__ == "__main__":
    main()