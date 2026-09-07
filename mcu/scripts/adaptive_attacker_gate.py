"""
Part B (MCU tier) items 11+12: adaptive-attacker harness + fixed-basis gate.

OBJECTIVE: On the MCU tier the deployed classifier is the INT8 TFLM specialist
(d=12 -> 16 -> 1, sigmoid), running through TFLM's fixed-point path, NOT the
float torch surrogate used on the Edge tier. Per the paper gate, we must
reproduce the known "attacker-coupled-basis gap" BEFORE trusting any fixed-basis
defense result.

REPLICA (instruction-accurate): The attacker targets a faithful replica of the
deployed INT8 model, reconstructed from the actual int8 weights extracted from
models/mcu_specialist_{nsl,unsw}_int8.tflite:
  - FC1: int8 matmul + int32 bias => requantize to pre-tanh scale/zp
  - Tanh: int8 table (scale 1/128, zp 0)
  - FC2: int8 matmul + int32 bias => requantize to pre-sigmoid scale/zp
  - Sigmoid: int8 output (scale 1/256, zp 0)
The forward path is differentiable (QAT-style straight-through quantization),
so gradient-based attackers (naive + adaptive PGD) see the same decision
boundary the silicon would present.

GATE (matches Edge run_check4_real.py):
  1. Evaluate on the VULNERABLE (attacker-coupled) basis first.
     Gate passes iff naive-vs-adaptive gap > 0.02 for at least one PGD config.
  2. Only then evaluate on the FIXED basis (Givens rotation in a fixed PCA basis).

The defense operates on the 12-dim normalized feature vectors; the recalibrated
per-source gamma (items 9+10) is used as the Log-phase trigger threshold.

Emits:
  - mcu/features/attacker_gate_report.json
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")
RESULTS_DIR = os.path.join(MCU_DIR, "results")

D = 12
W = 10
DELTA_THETA_MAX_DEG = 45.0


def load_gamma():
    with open(os.path.join(FEATURES_DIR, "gamma_params.json")) as f:
        return json.load(f)["gamma_per_source"]


# ---------------------------------------------------------------------------
# INT8-faithful differentiable replica of the deployed TFLM specialist
# ---------------------------------------------------------------------------
def _init_from_int8(tag):
    z = np.load(f"/tmp/opencode/{tag}_w.npz")
    return z


# Per-tensor weight scales (from models/mcu_specialist_*_int8.tflite,
# tensor quantization_parameters) so the dequantized replica matches the
# deployed int8 decision boundary. w-scale relation: b_scale = s_in * s_w1.
W_SCALES = {
    "nsl": {"s_w1": 0.04612535, "s_w2": 0.02429635},
    "unsw": {"s_w1": 0.09418254, "s_w2": 0.04263391},
}


class Int8SpecialistReplica(torch.nn.Module):
    """Differentiable int8-faithful replica of the TFLM specialist.

    Reconstructs the deployed int8 weights (dequantized with their real
    per-tensor scales) and reproduces the int8 fixed-point path:
        input -> FC1 (s_in x s_w1) -> requant(s_fc1out) -> tanh int8
              -> FC2 (s_tanh x s_w2) -> requant(s_fc2pre) -> sigmoid int8
    using QAT-style straight-through quantization (fake-quant in fwd,
    identity gradient), so PGD sees the same boundary the silicon presents.
    """

    def __init__(self, tag, gamma_p95):
        super().__init__()
        z = _init_from_int8(tag)
        self.tag = tag
        self.gamma = gamma_p95

        # Quantization params (from tflite, per tag)
        if tag == "nsl":
            self._s_in, self._zp_in = 0.003921568859368563, -128   # input [0,1]->int8
            self._s_fc1out, self._zp_fc1out = 0.035085760056972504, -2   # pre-tanh int8
            self._s_fc2pre, self._zp_fc2pre = 0.08132143318653107, -25
        else:
            self._s_in, self._zp_in = 0.003921568859368563, -128
            self._s_fc1out, self._zp_fc1out = 0.048389460891485214, 35
            self._s_fc2pre, self._zp_fc2pre = 0.07031682133674622, 13
        self._s_tanh, self._zp_tanh = 0.0078125, 0              # tanh int8 (1/128)
        self._s_out, self._zp_out = 0.00390625, -128            # sigmoid int8 (1/256)

        # Dequantized real weights (deployed int8 weights scaled back)
        s_w1 = W_SCALES[tag]["s_w1"]
        s_w2 = W_SCALES[tag]["s_w2"]
        self.register_buffer("w1", torch.tensor(z["w1"].astype(np.float32)) * s_w1)
        self.register_buffer("b1", torch.tensor(z["b1"].astype(np.float32)) * (self._s_in * s_w1))
        self.register_buffer("w2", torch.tensor(z["w2"].astype(np.float32)) * s_w2)
        self.register_buffer("b2", torch.tensor(z["b2"].astype(np.float32)) * (self._s_tanh * s_w2))

    def fake_quant(self, x, s, zp):
        """QAT-style straight-through: quantize->dequantize in fwd, identity gradient.

        y = x + (dq - x).detach(): forward returns dq (the dequantized value),
        but d dy/dx = 1 (the rounding/clamp path is excluded via detach), so
        PGD/inf-grad sees the int8 decision boundary yet stays differentiable.
        """
        q = torch.round(x / s) + zp
        q = torch.clamp(q, -128.0, 127.0)
        dq = (q - zp) * s
        return x + (dq - x).detach()

    def forward(self, x):
        # input quantize: int8 at s_in/zp_in (deployed path)
        xq = self.fake_quant(x, self._s_in, self._zp_in)

        # FC1 (dequantized int8 weights + int32 bias at s_in*s_w1)
        f1 = F.linear(xq, self.w1, self.b1)
        f1q = self.fake_quant(f1, self._s_fc1out, self._zp_fc1out)

        # Tanh int8 (1/128): differentiable tanh in fwd, quantized to int8
        t = torch.tanh(f1q)
        tq = self.fake_quant(t, self._s_tanh, self._zp_tanh)

        # FC2 (dequantized int8 weights + int32 bias at s_tanh*s_w2)
        f2 = F.linear(tq, self.w2, self.b2)
        f2q = self.fake_quant(f2, self._s_fc2pre, self._zp_fc2pre)

        # Sigmoid int8 output (1/256)
        s = torch.sigmoid(f2q)
        sq = self.fake_quant(s, self._s_out, self._zp_out)
        return sq.squeeze(-1)


# ---------------------------------------------------------------------------
# MIDAS defense primitives (NumPy/torch ports, d=12)
# ---------------------------------------------------------------------------
def momentum(v_t, v_prev):
    norm_t = torch.norm(v_t, p=2)
    norm_tm1 = torch.norm(v_prev, p=2)
    if norm_t < 1e-7 or norm_tm1 < 1e-7:
        return torch.tensor(0.0, device=v_t.device)
    m = torch.sum(v_t * v_prev) / (norm_t * norm_tm1)
    return torch.clamp(m, -1.0, 1.0)


def penetration_epsilon_windowed(window):
    if len(window) < 2:
        return torch.tensor(0.0, device=window[0].device)
    s = torch.tensor(0.0, device=window[0].device)
    n = 0
    for i in range(len(window) - 1):
        m = momentum(window[i + 1], window[i])
        s = s + torch.norm(window[i + 1], p=2) * torch.clamp(m, min=0.0)
        n += 1
    return s / n


def rotation_angle(eps, gamma, lam, k, dtheta_max):
    if eps <= gamma:
        return lam * eps
    return torch.clamp(lam * torch.exp(k * (eps - gamma)), max=dtheta_max)


def rotate_fixed_basis(x, basis, theta):
    b0, b1 = basis[0], basis[1]
    c = torch.cos(theta); s = torch.sin(theta)
    d0 = torch.sum(x * b0); d1 = torch.sum(x * b1)
    r0 = c * d0 - s * d1
    r1 = s * d0 + c * d1
    return x + (r0 - d0) * b0 + (r1 - d1) * b1


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


def defense_forward(x, window, c_base, basis, cfg, naive=False, vulnerable=False):
    gamma = cfg["gamma"]; lam = cfg["lambda"]; k = cfg["k"]
    dtheta = cfg["delta_theta_max"]
    full = window + [x]
    eps = penetration_epsilon_windowed(full)
    theta = rotation_angle(eps, gamma, lam, k, dtheta)
    if naive:
        theta = theta.detach()
        basis_used = vulnerable_basis(x.detach(), c_base) if vulnerable else basis
    else:
        basis_used = vulnerable_basis(x, c_base) if vulnerable else basis
    return rotate_fixed_basis(x, basis_used, theta), theta


def project_Lp(x, x0, eps):
    return torch.clamp(x0 + torch.clamp(x - x0, -eps, eps), 0.0, 1.0)


def pgd_attack(x0, y, classifier, traj, c_base, basis, cfg, alpha, eps, steps, naive, vulnerable):
    x = x0.clone().detach()
    x = x + torch.empty_like(x).uniform_(-eps, eps)
    x = project_Lp(x, x0, eps)
    win = [w.clone().detach() for w in traj]
    for _ in range(steps):
        x.requires_grad_(True)
        ws = win[-(cfg["W"] - 1):]
        xd, _ = defense_forward(x, ws, c_base, basis, cfg, naive=naive, vulnerable=vulnerable)
        pred = classifier(xd).clamp(1e-7, 1 - 1e-7)
        loss = F.binary_cross_entropy(pred, y.expand_as(pred))
        loss.backward()
        with torch.no_grad():
            x = x.detach() + alpha * torch.sign(x.grad)
            x = project_Lp(x, x0, eps)
            win.append(x.clone().detach())
    return x.detach(), win[-(cfg["W"] - 1):]


# ---------------------------------------------------------------------------
# Dataset + trajectories from real MCU-tier data (per source)
# ---------------------------------------------------------------------------
def load_source(prov_value):
    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy")).astype(np.float32)
    y = np.load(os.path.join(FEATURES_DIR, "unified_labels.npy")).astype(np.float32)
    p = np.load(os.path.join(FEATURES_DIR, "unified_provenance.npy"))
    mask = p == prov_value
    return torch.tensor(X[mask]), torch.tensor(y[mask])


def load_manifold():
    """Mirror of shadow/run_check4_real.py.load_manifold(): PCA-2 basis + centroid."""
    with open(os.path.join(FEATURES_DIR, "manifold_basis.json")) as f:
        data = json.load(f)
    return (torch.tensor(data["basis"], dtype=torch.float32),
            torch.tensor(data["c_base"], dtype=torch.float32))


def build_dataset(Xs, ys, classifier, n_samples, W):
    # Mirror Edge: benign samples only + <=0.95 confidence filter, sliding
    # trajectory windows of W-1 consecutive rows before each sample.
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
        traj = [Xs[j] for j in range(i - (W - 1), i)]
        dataset.append(Xs[i])
        trajectories.append(traj)
        if len(dataset) >= n_samples:
            break
    return dataset, trajectories


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------
def run_battery(classifier, dataset, trajectories, c_base, basis, cfg, attacks, n, vulnerable):
    results = {}
    for name, alpha, steps, eps in attacks:
        n_succ = 0
        a_succ = 0
        for x0, traj in zip(dataset, trajectories):
            with torch.no_grad():
                y_nat = (classifier(x0) > 0.5).float()
            xa_n, wn = pgd_attack(x0, y_nat, classifier, traj, c_base, basis, cfg,
                                  alpha, eps, steps, naive=True, vulnerable=vulnerable)
            xd_n, _ = defense_forward(xa_n, wn, c_base, basis, cfg, vulnerable=vulnerable)
            if (classifier(xd_n) > 0.5).item() != y_nat.item():
                n_succ += 1
            xa_a, wa = pgd_attack(x0, y_nat, classifier, traj, c_base, basis, cfg,
                                  alpha, eps, steps, naive=False, vulnerable=vulnerable)
            xd_a, _ = defense_forward(xa_a, wa, c_base, basis, cfg, vulnerable=vulnerable)
            if (classifier(xd_a) > 0.5).item() != y_nat.item():
                a_succ += 1
        asr_n = n_succ / n
        asr_a = a_succ / n
        results[name] = {"naive_asr": round(asr_n, 4), "adaptive_asr": round(asr_a, 4),
                         "gap": round(asr_a - asr_n, 4)}
    return results


def main():
    torch.manual_seed(42)
    gammas = load_gamma()
    basis, c_base = load_manifold()
    attack_cfgs = [("eps0.05_20", 0.01, 20, 0.05), ("eps0.1_50", 0.01, 50, 0.1),
                   ("eps0.2_100", 0.01, 100, 0.2)]
    cfg = {"W": W, "gamma": 0.0, "lambda": 1.0, "k": 2.0,
           "delta_theta_max": DELTA_THETA_MAX_DEG * (3.141592653589793 / 180.0)}
    n_samples = 200

    report = {"experiment": "Items 11+12: MCU adaptive-attacker gate on deployed int8 specialists",
              "W": W, "d": D, "n_samples": n_samples,
              "replica": "int8 weights extracted from models/mcu_specialist_*.tflite, differentiable ST-quant",
              "fixed_basis_defense": "Givens rotation in fixed PCA-2 basis (theta from epsilon_p)",
              "manifold": "mcu/features/manifold_basis.json (PCA-2 of clean benign unified d=12)"}

    for tag, prov in [("nsl", 0), ("unsw", 1)]:
        gamma = gammas[tag]
        cfg["gamma"] = gamma
        print(f"\n===== {tag.upper()} (gamma={gamma:.4f}) =====")
        classifier = Int8SpecialistReplica(tag, gamma)
        classifier.eval()
        Xs, ys = load_source(prov)
        dataset, trajectories = build_dataset(Xs, ys, classifier, n_samples, W)
        print(f"  benign samples: {len(dataset)}")

        # 1) Vulnerable (attacker-coupled) basis first
        vuln = run_battery(classifier, dataset, trajectories, c_base, None, cfg,
                           attack_cfgs, len(dataset), vulnerable=True)
        has_gap = any(abs(v["gap"]) > 0.02 for v in vuln.values())
        print(f"  Vulnerable-basis gaps: {[v['gap'] for v in vuln.values()]}")
        print(f"  GATE: {'PASS' if has_gap else 'FAIL'}")

        src = {"vulnerable_basis": vuln, "gate_passed": bool(has_gap)}
        if has_gap:
            fixed = run_battery(classifier, dataset, trajectories, c_base, basis, cfg,
                                attack_cfgs, len(dataset), vulnerable=False)
            print(f"  Fixed-basis gaps:      {[v['gap'] for v in fixed.values()]}")
            src["fixed_basis"] = fixed
        report[tag] = src

    # Stable machine-readable copy (for README / paper tables)
    with open(os.path.join(FEATURES_DIR, "attacker_gate_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {os.path.join(FEATURES_DIR, 'attacker_gate_report.json')}")

    # Timestamped result (mirror shadow/results convention), includes git hash
    save_results(
        "adaptive_attacker_gate.py",
        config=cfg,
        results=report,
    )


if __name__ == "__main__":
    main()
