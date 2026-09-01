"""
Export the trained PyTorch surrogate MLP (d=42 -> hidden=16 -> tanh -> sigmoid)
to TFLite: INT8-quantized (deployed) + float32 (reference).

Usage:
    python3 shadow/export_surrogate_to_tflite.py \
        --weights shadow/trained_weights/trained_surrogate_real.pt \
        --meta   shadow/trained_weights/trained_surrogate_real_meta.json \
        --out-dir models

Requires: torch, numpy, tensorflow-cpu
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "data")


def build_keras_model(input_dim, hidden_dim):
    # NOTE: hidden activation MUST match TrainedSurrogateModel (tanh, not relu).
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(hidden_dim, activation="tanh", name="fc1"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="fc2"),
    ])


def copy_weights(keras_model, state_dict):
    keras_model.get_layer("fc1").set_weights([
        state_dict["fc1.weight"].numpy().T,
        state_dict["fc1.bias"].numpy(),
    ])
    keras_model.get_layer("fc2").set_weights([
        state_dict["fc2.weight"].numpy().T,
        state_dict["fc2.bias"].numpy(),
    ])


def representative_dataset(calib_X):
    for row in calib_X:
        yield [row.reshape(1, -1).astype(np.float32)]


def convert(keras_model, calib_X, out_path, mode):
    """mode: 'fp32' | 'fp16' | 'dynamic' | 'int8'"""
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    if mode == "fp16":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.target_spec.supported_types = [tf.float16]
    elif mode == "dynamic":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    elif mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: representative_dataset(calib_X)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    buf = conv.convert()
    with open(out_path, "wb") as f:
        f.write(buf)
    return len(buf)


def verify(out_path, X, y_ref, mode, n=5000):
    """Run n samples through both PyTorch and TFLite; report drift."""
    idx = np.random.default_rng(0).choice(len(X), size=min(n, len(X)), replace=False)
    Xn = X[idx].astype(np.float32)
    y = y_ref[idx]

    with torch.no_grad():
        h = torch.tanh(torch.nn.functional.linear(torch.from_numpy(Xn), W["fc1.weight"], W["fc1.bias"]))
        pt = torch.sigmoid(torch.nn.functional.linear(h, W["fc2.weight"], W["fc2.bias"])).squeeze(-1).numpy()

    interp = tf.lite.Interpreter(model_path=out_path)
    interp.allocate_tensors()
    ind, outd = interp.get_input_details()[0], interp.get_output_details()[0]
    out = np.empty(len(Xn), dtype=np.float32)
    for i, row in enumerate(Xn):
        x = row.reshape(1, -1)
        s_in = ind.get("quantization", (0, 0))[0]
        if mode == "int8":
            x = np.round(x / ind["quantization"][0] + ind["quantization"][1]).astype(ind["dtype"])
        interp.set_tensor(ind["index"], x)
        interp.invoke()
        o = interp.get_tensor(outd["index"]).flatten()[0].astype(np.float32)
        if mode == "int8":
            o = (float(o) - outd["quantization"][1]) * outd["quantization"][0]
        out[i] = o

    diff = np.abs(pt - out)
    agree_pt = ((pt > 0.5) == y).mean()
    agree_tl = ((out > 0.5) == y).mean()
    disagree = ((pt > 0.5) != (out > 0.5)).mean()
    print(f"  [{os.path.basename(out_path)}] n={len(Xn)}")
    print(f"    max|Δp|={diff.max():.6f}  mean|Δp|={diff.mean():.6f}")
    print(f"    label disagreement PT vs TFLite: {disagree:.4%}")
    print(f"    acc: pytorch={agree_pt:.4%}  tflite={agree_tl:.4%}")
    return {"max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
            "label_disagreement": float(disagree),
            "acc_pytorch": float(agree_pt), "acc_tflite": float(agree_tl)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(REPO, "shadow/trained_weights/trained_surrogate_real.pt"))
    ap.add_argument("--meta", default=os.path.join(REPO, "shadow/trained_weights/trained_surrogate_real_meta.json"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "models"))
    args = ap.parse_args()

    input_dim, hidden_dim = 42, 16
    if os.path.exists(args.meta):
        m = json.load(open(args.meta))
        input_dim = m.get("d", input_dim)          # meta uses "d"/"hidden"
        hidden_dim = m.get("hidden", hidden_dim)

    global W
    W = torch.load(args.weights, map_location="cpu")

    train_X = np.load(os.path.join(DATA_DIR, "train_features.npy")).astype(np.float32)
    test_X = np.load(os.path.join(DATA_DIR, "test_features.npy")).astype(np.float32)
    test_y = np.load(os.path.join(DATA_DIR, "test_labels.npy"))

    km = build_keras_model(input_dim, hidden_dim)
    copy_weights(km, W)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for name, mode in [("classifier_float32.tflite", "fp32"),
                       ("classifier_fp16.tflite", "fp16"),
                       ("classifier_dr_int8.tflite", "dynamic"),
                       ("classifier_int8.tflite", "int8")]:
        p = os.path.join(args.out_dir, name)
        nbytes = convert(km, train_X[:2048], p, mode=mode)
        results[name] = {"bytes": nbytes, "kb": round(nbytes / 1024, 2), "mode": mode}
        results[name]["verification"] = verify(p, test_X, test_y, mode=mode)

    with open(os.path.join(args.out_dir, "export_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nTable VI figures:")
    for k, v in results.items():
        print(f"  {k}: {v['bytes']} bytes ({v['kb']} KB)")


if __name__ == "__main__":
    main()
