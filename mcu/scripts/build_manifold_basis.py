"""
Part B (MCU tier) manifold basis builder.

Mirror of the Edge-tier `results/basis.json` producer: computes the PCA-2
basis + centroid (c_base) of the real MCU-tier feature manifold (clean benign
rows across both sources, unified 12-dim normalized space) that the fixed-basis
Givens rotation is anchored to.

Emits:
  - mcu/features/manifold_basis.json   (schema matches results/basis.json)
  - mcu/results/build_manifold_basis_<timestamp>.json   (git-hashed)
"""
import json
import os

import numpy as np

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")


def main():
    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy")).astype(np.float32)
    y = np.load(os.path.join(FEATURES_DIR, "unified_labels.npy"))

    benign = X[y == 0]
    center = benign.mean(axis=0)
    Xc = benign - center
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    basis = Vt[:2]

    # Sign consistency: force largest |component| of b0 to be positive.
    for i in range(2):
        k = int(np.argmax(np.abs(basis[i])))
        if basis[i][k] < 0:
            basis[i] = -basis[i]

    record = {
        "basis": basis.tolist(),
        "c_base": center.tolist(),
        "note": "PCA-2 basis + centroid of clean benign MCU unified features (d=12, both sources)",
        "n_benign": int(len(benign)),
        "explained_variance": [round(float(s) ** 2 / (benign.size / len(benign) - 1), 6)
                               for s in np.linalg.svd(Xc, full_matrices=False)[1][:2]],
    }
    with open(os.path.join(FEATURES_DIR, "manifold_basis.json"), "w") as f:
        json.dump(record, f, indent=2)
    print("saved mcu/features/manifold_basis.json")

    save_results("build_manifold_basis.py",
                 config={"d": 12, "note": record["note"]},
                 results=record)


if __name__ == "__main__":
    main()