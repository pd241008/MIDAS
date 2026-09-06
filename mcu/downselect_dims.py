"""
Part B (MCU tier) step 4: dimensionality down-select under MCU constraints.

Decision input for the fused MCU feature set. Trains the Edge-compatible
surrogate on candidate feature subsets of the joint normalized corpus and
reports per-schema accuracy, so the final d is chosen by BOTH fit ("5 EXACT +
{state,is_sm_ips_ports}" vs the full 12-anchor) and SRAM budget.

Candidates:
  - CORE_7:  dur, proto, service, state, sbytes, dbytes, is_sm_ips_ports
            (5 EXACT + 2 statically-sound analog)  -- MCU-minimal
  - ANCHOR_12: full step-1 overlap set  -- maximum overlap recall

Emits:
  - mcu/features/downselect_report.json   accuracy (joint + per-provenance)
  - saved fused surrogate weights per candidate for step 5/6 QAT
"""
import json, os

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(HERE, "features")
WEIGHTS_DIR = os.path.join(HERE, "trained_weights")

CORE_7_IDX = [0, 1, 2, 3, 4, 5, 6]       # dur, proto, service, state, sbytes, dbytes, is_sm_ips_ports
ANCHOR_12_IDX = list(range(12))
N_NSL = 148516

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


class Surrogate(torch.nn.Module):
    def __init__(self, d, hidden=16):
        super().__init__()
        self.fc1 = torch.nn.Linear(d, hidden)
        self.fc2 = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.tanh(self.fc1(x)))).squeeze(-1)


def load_joint():
    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy"))
    y = np.load(os.path.join(FEATURES_DIR, "unified_labels.npy")) if os.path.exists(
        os.path.join(FEATURES_DIR, "unified_labels.npy")
    ) else None
    if y is None:
        raise FileNotFoundError("mcu/features/unified_labels.npy missing — run step 3 predates it; rebuild")
    return X, y


def train_eval(name, idx, X, y, epochs=60, lr=0.01, bs=4096):
    Xs = X[:, idx].astype(np.float32)
    d = len(idx)
    model = Surrogate(d, hidden=16)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(Xs, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # provenance split: NSL (0..N_NSL) / UNSW (N_NSL..)
    nsl_mask = torch.arange(len(X_t)) < N_NSL
    unsw_mask = ~nsl_mask

    for ep in range(epochs):
        perm = torch.randperm(len(X_t))
        model.train()
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            out = model(X_t[b])
            loss = F.binary_cross_entropy(out, y_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        prob = model(X_t)
        pred = (prob > 0.5).float()
        acc_joint = (pred == y_t).float().mean().item()
        acc_nsl = (pred[nsl_mask] == y_t[nsl_mask]).float().mean().item()
        acc_unsw = (pred[unsw_mask] == y_t[unsw_mask]).float().mean().item()
        auroc = torch.tensor(0.0)  # placeholder (not needed for down-select)

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, f"mcu_surrogate_{name}.pt"))
    return {"d": d, "acc_joint": acc_joint, "acc_nsl": acc_nsl,
            "acc_unsw": acc_unsw, "features": idx}


def main():
    X, y = load_joint()

    # Persist normalized labels once (step-3 did not write them).
    np.save(os.path.join(FEATURES_DIR, "unified_labels.npy"), y)
    print(f"joint corpus: {X.shape}, positives {int(y.sum())}")

    W = 10
    budget = 192 * 1024
    report = {"candidates": [], "sram_budget_bytes": budget, "w": W}
    for name, idx, note in [("core7", CORE_7_IDX, "5 EXACT + state + is_sm_ips_ports"),
                            ("anchor12", ANCHOR_12_IDX, "full 12-anchor overlap")]:
        r = train_eval(name, idx, X, y)
        r["ring_buffer_bytes"] = W * 4 * r["d"]
        r["note"] = note
        report["candidates"].append(r)
        print(f"{name}: d={r['d']} acc_joint={r['acc_joint']:.4f} "
              f"nsl={r['acc_nsl']:.4f} unsw={r['acc_unsw']:.4f} "
              f"ring={r['ring_buffer_bytes']} B")

    # pick candidate with best joint accuracy (tie -> smaller d)
    best = min(report["candidates"], key=lambda c: (-c["acc_joint"], c["d"]))
    report["selected"] = best["d"]
    report["decision"] = (f"selected d={best['d']} "
                          f"({best['note']}) acc_joint={best['acc_joint']:.4f} "
                          f"- best fit under 192 KB (ring {best['ring_buffer_bytes']} B)")

    out = os.path.join(FEATURES_DIR, "downselect_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\ndecision: {report['decision']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()