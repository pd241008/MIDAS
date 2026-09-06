"""
Part B (MCU tier) step 5 (rev B): per-source specialists on shared 12-dim space.

Design decision (user-confirmed): a single jointly-trained fused MLP cannot
generalize per-source (provenance-split val ~0.58, majority baseline). Two
source-specialist models on the SAME unified 12-feature overlap space
generalize per-source (NSL ~0.947, UNSW ~0.872), and provenance is a
deployment-time routing concern on the MCU.

Trains:
  - mcu_specialist_nsl: trained on NSL-KDD rows only
  - mcu_specialist_unsw: trained on UNSW-NB15 rows only
each d=12 -> hidden=16 -> tanh -> sigmoid.

QAT: quantize_model (fake-quant training) so the full-INT8 TFLite preserves
float accuracy (ptq lost 2.1pp in Edge tier). Outputs padded int8 (TFLM-ready)
+ fp32 references.

Emits:
  - models/mcu_specialist_{nsl,unsw}_{fp32,int8}.tflite
  - mcu/features/specialists_report.json  per-source acc / sizes / quantization
  - mcu/trained_weights/mcu_specialist_{nsl,unsw}_qat.pt (state for reuse)
"""
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import quantize_model

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(MCU_DIR)
FEATURES_DIR = os.path.join(MCU_DIR, "features")
MODELS_DIR = os.path.join(ROOT, "models")
WEIGHTS_DIR = os.path.join(MCU_DIR, "trained_weights")

D = 12
HIDDEN = 16
EPOCHS = 25
BS = 4096
LR = 0.01
SEED = 42


def load_source(prov):
    X = np.load(os.path.join(FEATURES_DIR, "unified_features_norm.npy")).astype(np.float32)
    y = np.load(os.path.join(FEATURES_DIR, "unified_labels.npy")).astype(np.float32)
    p = np.load(os.path.join(FEATURES_DIR, "unified_provenance.npy"))
    mask = p == (0 if prov == "nsl" else 1)
    return X[mask], y[mask]


def build_model(d=D, hidden=HIDDEN, qat=False):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(d,)),
        tf.keras.layers.Dense(hidden, activation="tanh", name="fc1"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="fc2"),
    ])
    return quantize_model(model) if qat else model


def split(X, y, frac=0.8, seed=SEED):
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(X))
    k = int(frac * len(X))
    return X[p[:k]], y[p[:k]], X[p[k:]], y[p[k:]]


def train(model, X, y, val):
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(X, y, batch_size=BS, epochs=EPOCHS, verbose=0,
              validation_data=val, shuffle=True)
    return model


def representative_dataset(calib_X):
    def gen():
        for x in calib_X:
            yield [np.expand_dims(x.astype(np.float32), axis=0)]
    return gen


def convert(model, calib_X, out_path, mode):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    if mode == "int8":
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_dataset(calib_X)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.int8
        conv.inference_output_type = tf.int8
    buf = conv.convert()
    with open(out_path, "wb") as f:
        f.write(buf)
    return len(buf)


def evaluate_tflite(path, X, y, n=20000):
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
    acc = float(((preds > 0.5) == y[idx]).mean())
    pos = preds[y[idx] == 1]
    neg = preds[y[idx] == 0]
    auc = float(np.mean(pos[:, None] > neg[None, :])) if len(pos) and len(neg) else float("nan")
    return {"acc": acc, "auroc_rough": auc, "n": int(len(idx))}


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    report = {"d": D, "hidden": HIDDEN, "epochs": EPOCHS,
              "design": ("two per-source specialists, shared unified 12-dim "
                         "space; provenance = deployment-time routing"),
              "models": {}}

    for prov in ("nsl", "unsw"):
        X, y = load_source(prov)
        Xtr, ytr, Xte, yte = split(X, y)
        print(f"[{prov}] train {Xtr.shape} / test {Xte.shape} pos={yte.mean():.3f}")

        mf = build_model(qat=False)
        train(mf, Xtr, ytr, (Xte, yte))
        acc_fp = float(mf.evaluate(Xte, yte, verbose=0)[1])

        mq = build_model(qat=True)
        train(mq, Xtr, ytr, (Xte, yte))
        acc_qp = float(mq.evaluate(Xte, yte, verbose=0)[1])

        fp_path = os.path.join(MODELS_DIR, f"mcu_specialist_{prov}_fp32.tflite")
        i8_path = os.path.join(MODELS_DIR, f"mcu_specialist_{prov}_int8.tflite")
        n_fp = convert(mf, Xtr[:2048], fp_path, "fp32")
        n_i8 = convert(mq, Xtr[:2048], i8_path, "int8")

        ev_fp = evaluate_tflite(fp_path, Xte, yte)
        ev_i8 = evaluate_tflite(i8_path, Xte, yte)

        report["models"][prov] = {
            "source_rows": int(len(X)),
            "nonzero_rows_in": {"train": int(len(Xtr)), "test": int(len(Xte))},
            "acc_keras_float": acc_fp,
            "acc_keras_qat_pre_conv": acc_qp,
            "acc_tflite_fp32": ev_fp["acc"],
            "acc_tflite_int8": ev_i8["acc"],
            "auroc_rough_int8": ev_i8["auroc_rough"],
            "float32_bytes": n_fp, "int8_bytes": n_i8,
            "int8_penalty_vs_fp": round((acc_fp - ev_i8["acc"]) * 100, 2),
        }
        print(f"    fp32 {n_fp} B acc {ev_fp['acc']:.4f} | "
              f"int8 {n_i8} B acc {ev_i8['acc']:.4f} (penalty {report['models'][prov]['int8_penalty_vs_fp']}pp)")

    out = os.path.join(FEATURES_DIR, "specialists_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out}")
    print("Fallback artifact: specialists int8 models are TFLM-ready inputs.")


if __name__ == "__main__":
    main()