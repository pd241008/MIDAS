"""
Part B (MCU tier) replica-fidelity check: Int8SpecialistReplica vs deployed
TFLite int8 specialist.

Mirror of shadow/check_trained_surrogate.py: verifies (on real MCU-tier data)
that the differentiable attacker replica reproduces the decision boundary of
the deployed int8 TFLite/TFLM model.

Emits:
  - mcu/results/check_replica_fidelity_<timestamp>.json   (git-hashed)
"""
import importlib.util
import os

import numpy as np
import torch

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")


def _load_script_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import tensorflow as tf

    aag = _load_script_module("aag", os.path.join(HERE, "adaptive_attacker_gate.py"))
    N = 600

    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy")).astype(np.float32)
    p = np.load(os.path.join(FEATURES_DIR, "unified_provenance.npy"))

    report = {"experiment": "MCU replica fidelity: int8 replica vs deployed TFLite",
              "n_per_source": N}
    for tag, prov in [("nsl", 0), ("unsw", 1)]:
        interp = tf.lite.Interpreter(
            model_path=os.path.join(MCU_DIR, "..", "models", f"mcu_specialist_{tag}_int8.tflite"),
            experimental_delegates=[])
        interp.allocate_tensors()
        in_d = interp.get_input_details()[0]
        out_d = interp.get_output_details()[0]
        s_in, zp_in = in_d["quantization_parameters"]["scales"][0], in_d["quantization_parameters"]["zero_points"][0]
        s_out, zp_out = out_d["quantization_parameters"]["scales"][0], out_d["quantization_parameters"]["zero_points"][0]

        rep = aag.Int8SpecialistReplica(tag, 0.0).eval()
        Xs = X[p == prov][:N]

        matches, label_flips = 0, 0
        abs_diffs = []
        for x in Xs:
            q = np.clip(np.round(x / s_in) + zp_in, -128, 127).astype(np.int8).reshape(1, 12)
            interp.set_tensor(in_d["index"], q)
            interp.invoke()
            pred_tfl = ((interp.get_tensor(out_d["index"]).astype(np.float32)[0, 0] - zp_out) * s_out)
            pred_rep = rep(torch.tensor(x)).item()
            abs_diffs.append(abs(pred_tfl - pred_rep))
            if (pred_tfl > 0.5) == (pred_rep > 0.5):
                matches += 1
            else:
                label_flips += 1

        abs_diffs = np.array(abs_diffs)
        report[tag] = {
            "label_agreement": round(matches / len(Xs), 4),
            "label_flips": int(label_flips),
            "max_abs_diff": round(float(abs_diffs.max()), 6),
            "p95_abs_diff": round(float(np.percentile(abs_diffs, 95)), 6),
            "mean_abs_diff": round(float(abs_diffs.mean()), 6),
            "note": "residual = int8 weight-dequant/requant rounding noise near decision boundary",
        }
        print(f"{tag}: label agreement {matches}/{len(Xs)} ({report[tag]['label_agreement']:.4f}), "
              f"max|d|={report[tag]['max_abs_diff']:.6f}")

    save_results("check_replica_fidelity.py",
                 config={"n_per_source": N, "replica": "Int8SpecialistReplica (ST-quant, dequantized int8 weights)"},
                 results=report)


if __name__ == "__main__":
    main()