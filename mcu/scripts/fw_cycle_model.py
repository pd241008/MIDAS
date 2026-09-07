"""
Part B (MCU tier) host-side firmware cycle model.

Estimates per-invocation latency of the deployed INT8 specialists on the
STM32F407 (Cortex-M4F @ 168 MHz) by (a) enumerating the actual op graph +
tensor shapes from models/mcu_specialist_{nsl,unsw}_int8.tflite via the
TFLite flatbuffer schema, and (b) applying an int8 embedded-kernel cost model
(generic TFLM kernels on Cortex-M4, no SIMD).

This is ANALYTICAL (no board attached) — the DWT cycle numbers printed by
mcu_fw are the ground truth once flashed. Used to demonstrate SLA headroom:
  SLA budget (configs/mcu_config.json) : sla_budget_ms = 10
  Ring-buffer defense overhead           : negligibly small single op

Cost model (conservative-to-typical, cycles):
  FULLY_CONNECTED : 800 base + out_elems*(in_per_out*CYC_MAC + 16 quant)
  LOGISTIC/TANH   : 48 per output element (int8 lookup + lerp)
  interpreter     : 2000 per Invoke (dispatch, arena, tensor plumbing)
CYC_MAC ∈ {3 (optimistic), 6 (typical), 10 (conservative)}.
"""
import math
import os

import numpy as np
from save_results import save_results

try:
    from tensorflow.lite.python import schema_py_generated as schema_fb
except ImportError:  # pragma: no cover
    schema_fb = None

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
MODELS_DIR = os.path.join(MCU_DIR, "..", "models")

CLK_HZ = 168.0e6
SLA_MS = 10.0

CYC_MAC_SCEN = {"optimistic": 3, "typical": 6, "conservative": 10}
CYC_FC_BASE = 800
CYC_QUANT = 16
CYC_ACTIVATION_ELEM = 48
CYC_INTERP = 2000


def tensor_dims(t):
    if t is None or t.ShapeIsNone():
        return []
    return [t.Shape(i) for i in range(t.ShapeLength())]


def numel(dims):
    return int(np.prod(dims)) if dims else 1


def op_name(code):
    for name, val in vars(schema_fb.BuiltinOperator).items():
        if isinstance(val, int) and val == code:
            return name
    return f"UNKNOWN({code})"


def load_graph(path):
    data = np.fromfile(path, dtype=np.uint8)
    model = schema_fb.Model.GetRootAsModel(data.tobytes(), 0)
    g = model.Subgraphs(0)
    tensors = [g.Tensors(i) for i in range(g.TensorsLength())]
    out = []
    for k in range(g.OperatorsLength()):
        op = g.Operators(k)
        code = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        ins = [tensors[op.Inputs(j)] for j in range(op.InputsLength())]
        ous = [tensors[op.Outputs(j)] for j in range(op.OutputsLength())]
        out.append({"op": int(code), "ins": ins, "ous": ous})
    return out


def estimate(graph):
    ops = []
    tot_mac = 0
    for entry in graph:
        code = entry["op"]
        name = op_name(code)
        ins, ous = entry["ins"], entry["ous"]
        if name == "FULLY_CONNECTED":
            w_dims = tensor_dims(ins[1]) if len(ins) > 1 else []
            macs = numel(w_dims) if w_dims else 1
            nout = numel(tensor_dims(ous[0])) if ous else 1
            ops.append({"op": name, "macs": macs, "outs": nout, "w_dims": w_dims})
            tot_mac += macs
        else:  # LOGISTIC / TANH
            elems = numel(tensor_dims(ous[0])) if ous else 0
            ops.append({"op": name, "elems": elems})
    return ops, tot_mac


def per_scenario(ops, cyc_mac):
    total = CYC_INTERP
    for o in ops:
        if o["op"] == "FULLY_CONNECTED":
            total += CYC_FC_BASE + o["outs"] * (o["macs"] // max(o["outs"], 1) * cyc_mac + CYC_QUANT)
        else:
            total += o["elems"] * CYC_ACTIVATION_ELEM
    ms = total / CLK_HZ * 1e3
    return {"cycles": total, "latency_ms": round(ms, 4), "sla_ms": SLA_MS,
            "sla_ratio": round(ms / SLA_MS, 4)}


def main():
    if schema_fb is None:
        raise SystemExit("tensorflow schema module unavailable")

    models = {"nsl": os.path.join(MODELS_DIR, "mcu_specialist_nsl_int8.tflite"),
              "unsw": os.path.join(MODELS_DIR, "mcu_specialist_unsw_int8.tflite")}

    per_model = {}
    for tag, path in models.items():
        graph = load_graph(path)
        ops, tot_mac = estimate(graph)
        scen = {s: per_scenario(ops, c) for s, c in CYC_MAC_SCEN.items()}
        per_model[tag] = {
            "model": os.path.basename(path),
            "bytes": int(np.fromfile(path, dtype=np.uint8).size),
            "ops": [{"op": o["op"],
                     "w_dims": o.get("w_dims", []),
                     "macs": o.get("macs", 0),
                     "outs": o.get("outs", o.get("elems", 0))} for o in ops],
            "total_macs": tot_mac,
            "scenarios": scen,
        }
        print(f"{tag.upper()}: {per_model[tag]['bytes']} B, ops=")
        for o in per_model[tag]["ops"]:
            print(f"   {o['op']:<15s} w_dims={o['w_dims']}  macs/elems={o['macs']}")
        print(f"   total MACs: {tot_mac}")
        for s, r in scen.items():
            print(f"   [{s:>12s}] {r['cycles']:>7d} cyc = {r['latency_ms']:.4f} ms "
                  f"({r['sla_ratio']*100:.2f}% of {SLA_MS} ms SLA)")

    report = {"experiment": "Item 6+cycle model: analytical INT8 inference latency on STM32F407 @168 MHz",
              "note": "Host-side analytical estimate (no board attached); DWT CYCCNT prints on mcu_fw are ground truth once flashed.",
              "clk_hz": CLK_HZ, "sla_budget_ms": SLA_MS,
              "cost_model": {"cyc_mac": CYC_MAC_SCEN, "fc_base": CYC_FC_BASE,
                             "quant_per_out": CYC_QUANT, "activation_elem": CYC_ACTIVATION_ELEM,
                             "interpreter_overhead": CYC_INTERP},
              "models": per_model,
              "verdict": {t: per_model[t]["scenarios"]["typical"] for t in models}}

    import json
    stable = os.path.join(MCU_DIR, "features", "fw_cycle_model_report.json")
    with open(stable, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved {stable}")

    save_results("fw_cycle_model.py",
                 config={"clk_hz": CLK_HZ, "sla_budget_ms": SLA_MS},
                 results={"models": per_model}, extra={"note": report["note"]})


if __name__ == "__main__":
    main()