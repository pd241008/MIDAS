"""
Generate FFI validation vectors: a deterministic sample of UNSW-NB15 test rows
plus PyTorch surrogate reference probabilities.

Output: results/ffi_vectors.json (consumed by `cargo run --bin tflite_validate`).

Usage:
    .venv/bin/python shadow/gen_ffit_vectors.py [--n 2000]
"""
import argparse
import json
import os

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "ffi_vectors.json"))
    args = ap.parse_args()

    W = torch.load(
        os.path.join(REPO, "shadow/trained_weights/trained_surrogate_real.pt"),
        map_location="cpu",
    )
    X = np.load(os.path.join(REPO, "data/test_features.npy")).astype(np.float32)

    idx = np.random.default_rng(0).choice(len(X), size=min(args.n, len(X)), replace=False)
    Xs = X[idx]

    w1 = W["fc1.weight"].numpy()
    b1 = W["fc1.bias"].numpy()
    w2 = W["fc2.weight"].numpy()
    b2 = W["fc2.bias"].numpy()
    h = np.tanh(Xs @ w1.T + b1)
    z = (h @ w2.T + b2).flatten()
    ref = 1.0 / (1.0 + np.exp(-z))

    payload = {
        "input_dim": int(X.shape[1]),
        "n": int(len(Xs)),
        "source_model": "shadow/trained_weights/trained_surrogate_real.pt",
        "inputs": [[float(v) for v in row] for row in Xs],
        "reference": [float(p) for p in ref],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)
    print(f"wrote {args.out}: n={len(Xs)} dim={payload['input_dim']} "
          f"ref_acc_vs_labels=n/a (labels not needed here)")


if __name__ == "__main__":
    main()
