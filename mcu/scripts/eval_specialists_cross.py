"""
Section VII-A classifier specifics: per-source held-out accuracy on the
deployed INT8 specialists, INCLUDING cross-source rows.

Evaluates both deployed TFLM INT8 specialists (models/mcu_specialist_*_int8.tflite)
on NSL-sourced AND UNSW-sourced held-out rows (deterministic 20% split, seed 42,
matching qat_export_specialists.py). The routing mechanism that separates the
sources at inference time is measured in train_routing_gate.py (the routing
gate); this script quantifies whether accuracy ALSO differs by source, i.e.
whether a single merged classifier would work (Section VII-A / Table hw_results).
"""
import json
import os

import numpy as np

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")
MODELS_DIR = os.path.join(MCU_DIR, "..", "models")

SEED = 42
N_PER_CELL = 20000


def tflite_eval(path, X, y, n=N_PER_CELL):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    i_q, i_z = ind["quantization"][0], ind["quantization"][1]
    o_q, o_z = outd["quantization"][0], outd["quantization"][1]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    preds = np.empty(len(idx))
    for j, row in enumerate(X[idx]):
        x = np.round(np.clip(row, 0, 1) / i_q + i_z) if i_q else row.reshape(1, -1)
        interp.set_tensor(ind["index"], x.astype(ind["dtype"]).reshape(1, -1))
        interp.invoke()
        o = interp.get_tensor(outd["index"]).flatten()[0]
        preds[j] = float(o) if o_q == 0 else (float(o) - o_z) * o_q
    lab = y[idx]
    acc = float(((preds > 0.5) == lab).mean())
    pos, neg = preds[lab == 1], preds[lab == 0]
    auc = float(np.mean(pos[:, None] > neg[None, :])) if len(pos) and len(neg) else float("nan")
    return {"acc": round(acc, 4), "auroc_rough": round(auc, 4), "n": int(len(idx))}


def main():
    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy")).astype(np.float32)
    y = np.load(os.path.join(FEATURES_DIR, "unified_labels.npy")).astype(np.float32)
    p = np.load(os.path.join(FEATURES_DIR, "unified_provenance.npy"))
    rng = np.random.default_rng(SEED)

    splits = {}
    for src, prov in [("nsl", 0), ("unsw", 1)]:
        m = p == prov
        prm = rng.permutation(np.flatnonzero(m))
        k = int(0.8 * len(prm))
        splits[src] = {"Xte": X[prm[k:]], "yte": y[prm[k:]]}

    specialists = {
        "nsl": os.path.join(MODELS_DIR, "mcu_specialist_nsl_int8.tflite"),
        "unsw": os.path.join(MODELS_DIR, "mcu_specialist_unsw_int8.tflite"),
    }

    cells = {}
    for sp, path in specialists.items():
        cells.setdefault(sp, {})
        for src in ["nsl", "unsw"]:
            cells[sp][src] = tflite_eval(path, splits[src]["Xte"], splits[src]["yte"])

    report = {"experiment": "Section VII-A: per-source held-out accuracy of deployed INT8 specialists",
              "d": 12, "hidden": 16, "test_split": "80/20 per source, seed 42",
              "n_per_cell": N_PER_CELL,
              "rows_test_N": {s: {"n": int(len(sp["yte"]))} for s, sp in splits.items()},
              "accuracy": cells,
              "same_source": {s: cells[s][s] for s in ["nsl", "unsw"]},
              "cross_delta": {
                  "nsl_specialist_on_unsw_vs_nsl": round(
                      cells["nsl"]["unsw"]["acc"] - cells["nsl"]["nsl"]["acc"], 4),
                  "unsw_specialist_on_nsl_vs_unsw": round(
                      cells["unsw"]["nsl"]["acc"] - cells["unsw"]["unsw"]["acc"], 4)},
              }
    # Provenance-routed deployment: NSL spec on NSL rows, UNSW spec on UNSW rows.
    n_nsl = splits["nsl"]["yte"].size
    n_unsw = splits["unsw"]["yte"].size
    routed = (cells["nsl"]["nsl"]["acc"] * n_nsl + cells["unsw"]["unsw"]["acc"] * n_unsw) / (n_nsl + n_unsw)
    report["routed_deployment_accuracy"] = round(routed, 4)
    report["single_classifier_on_merged"] = {
        "nsl_spec_merged": round((cells["nsl"]["nsl"]["acc"] * n_nsl
                                  + cells["nsl"]["unsw"]["acc"] * n_unsw) / (n_nsl + n_unsw), 4),
        "unsw_spec_merged": round((cells["unsw"]["nsl"]["acc"] * n_nsl
                                   + cells["unsw"]["unsw"]["acc"] * n_unsw) / (n_nsl + n_unsw), 4)}
    report["rows_test_N"] = {s: {"n": int(len(sp["yte"])),
                                 "pos_rate": round(float(sp["yte"].mean()), 4)}
                             for s, sp in splits.items()}

    print("deployed INT8 specialist accuracy (held-out, rows=dest source):")
    print(f"{'specialist':<10s} {'NSL rows':>20s} {'UNSW rows':>20s}")
    for sp in ["nsl", "unsw"]:
        a, b = cells[sp]["nsl"], cells[sp]["unsw"]
        print(f"{sp:<10s} acc={a['acc']:.4f} n={a['n']:<5d} acc={b['acc']:.4f} n={b['n']}")
    print(f"\nsame-source: NSL {report['same_source']['nsl']['acc']:.4f}  "
          f"UNSW {report['same_source']['unsw']['acc']:.4f}")
    print(f"cross deltas: NSL-spec on UNSW vs NSL = {report['cross_delta']['nsl_specialist_on_unsw_vs_nsl']:+.4f}; "
          f"UNSW-spec on NSL vs UNSW = {report['cross_delta']['unsw_specialist_on_nsl_vs_unsw']:+.4f}")
    print(f"routed deployment accuracy (NSL spec on NSL rows + UNSW spec on UNSW rows): "
          f"{report['routed_deployment_accuracy']:.4f}")
    print(f"single-classifier-on-merged: NSL-spec={report['single_classifier_on_merged']['nsl_spec_merged']:.4f} "
          f"UNSW-spec={report['single_classifier_on_merged']['unsw_spec_merged']:.4f}")

    out = os.path.join(FEATURES_DIR, "specialists_cross_eval.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {out}")
    save_results("eval_specialists_cross.py",
                 config={"d": 12, "hidden": 16, "seed": SEED, "test_split": "80/20 per source"},
                 results={"accuracy": cells}, extra={"note": report["experiment"]})


if __name__ == "__main__":
    main()