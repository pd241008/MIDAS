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

On-device ground truth now exists: the routing gate, delta-theta defense and
fixed-basis rotation are implemented in mcu/fw/src/main.cc and every pipeline
stage is DWT-measured over 50 frames (see MEASURED below).
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

# Per-stage cycle cost (all ANALYTICAL - no board, DWT unmeasured):
#   T_sense  : DMA moves bytes (0 CPU); USART2 IDLE handler + fixed-point
#              ASCII parse. One frame ≈ 84 chars ("0.5000,...,0.5000\n").
#              ~20-80 cycles/char incl. dispatch + float accumulation.
#   T_theta  : epsilon_p over W=10 trajectory -> gamma comparison -> angle.
#   T_rotate : Givens rotation of one d-vector in the [u,v] plane
#              (2 dot over d + 2 saxpy) ~ 4d flops + angle scaling.
#   T_gate   : routing gate (ridge-logistic provenance classifier on the
#              12-dim frame). One 12x1 FC + logistic LUT, run at the
#              fusion point in parallel with the W-window rotation, before
#              the specialist Invoke. Weights: 13 x int8 + scale (21 B
#              SRAM). Deployed weights: mcu/trained_weights/routing_gate.npz.
SENSE_CHARS_PER_FRAME = 84
CYC_PER_CHAR = {"optimistic": 20, "typical": 45, "conservative": 80}
CYC_THETA = 200
CYC_ROTATE = 4 * 12 + 120   # 4d + angle bookkeeping for d=12
CYC_GATE_LOGISTIC_LUT = 60  # sigmoid from a small LUT (arguably swappable
                            # for quantized exp, same order) once per frame
GATE_DIMS = 12

# On-device ground truth (DISCO-F407VG, DWT, 50 frames @ 168.56 MHz): every
# defense stage LIVE in firmware main.cc. Mean cycles:
#   T_sense  : DMA IDLE handler + ASCII parse + window push + quantize (~4705)
#   T_theta  : windowed penetration + rotation_angle                      (284)
#   T_gate   : folded 12x1 FC + logistic LUT + route                     (117)
#   T_rotate : fixed-basis Givens (cos/sin via newlib; model's 168 cyc
#              underestimates trig)                                      (839)
#   T_inf    : routed INT8 specialist Invoke (rotated input; actual
#              per-feature cost depends on activation sparsity)         (~16,320)
# Full path = 22,266 cyc mean = 0.1321 ms = 1.32% of the 10 ms SLA.
MEASURED = {
    "stages_cycles": {"T_sense": 4705.0, "T_theta": 284.3, "T_gate": 117.0,
                      "T_rotate": 839.1, "T_inf": 16320.3},
    "full_path_cycles": 22265.7, "full_path_ms": 0.1321,
    "sla_ratio_pct": 1.321, "n_frames": 50,
    "route_split": {"NSL_specialist": 25, "UNSW_specialist": 25}}


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


def per_scenario(ops, cyc_mac, cyc_char):
    t_inf = CYC_INTERP
    for o in ops:
        if o["op"] == "FULLY_CONNECTED":
            t_inf += CYC_FC_BASE + o["outs"] * (o["macs"] // max(o["outs"], 1) * cyc_mac + CYC_QUANT)
        else:
            t_inf += o["elems"] * CYC_ACTIVATION_ELEM
    t_sense = SENSE_CHARS_PER_FRAME * cyc_char
    t_theta = CYC_THETA
    t_rot = CYC_ROTATE
    t_gate = GATE_DIMS * cyc_mac + CYC_GATE_LOGISTIC_LUT
    total = t_sense + t_theta + t_rot + t_gate + t_inf
    ms = lambda c: c / CLK_HZ * 1e3
    return {"cycles": total, "latency_ms": round(ms(total), 4), "sla_ms": SLA_MS,
            "sla_ratio": round(ms(total) / SLA_MS, 4),
            "stages_cycles": {"T_sense": t_sense, "T_theta": t_theta,
                              "T_gate": t_gate, "T_rotate": t_rot, "T_inf": t_inf},
            "stages_ms": {k: round(ms(v), 4) for k, v in
                          (("T_sense", t_sense), ("T_theta", t_theta),
                           ("T_gate", t_gate), ("T_rotate", t_rot), ("T_inf", t_inf))},
            "t_inf_only_ms": round(ms(t_inf), 4)}


def main():
    if schema_fb is None:
        raise SystemExit("tensorflow schema module unavailable")

    models = {"nsl": os.path.join(MODELS_DIR, "mcu_specialist_nsl_int8.tflite"),
              "unsw": os.path.join(MODELS_DIR, "mcu_specialist_unsw_int8.tflite")}

    per_model = {}
    for tag, path in models.items():
        graph = load_graph(path)
        ops, tot_mac = estimate(graph)
        scen = {s: per_scenario(ops, c_mac, c_chr)
                for (s, c_mac), c_chr in
                zip(CYC_MAC_SCEN.items(), (CYC_PER_CHAR["optimistic"],
                                           CYC_PER_CHAR["typical"],
                                           CYC_PER_CHAR["conservative"]))}
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
            st = r["stages_ms"]
            print(f"   [{s:>12s}] {r['cycles']:>6d} cyc = {r['latency_ms']:.4f} ms "
                  f"({r['sla_ratio']*100:.2f}% of {SLA_MS} ms SLA)   "
                  f"[sense {st['T_sense']:.4f} | theta {st['T_theta']:.4f} | "
                  f"gate {st['T_gate']:.4f} | rot {st['T_rotate']:.4f} | T_inf {st['T_inf']:.4f}]")

    report = {"experiment": "Item 6+cycle model: analytical INT8 inference latency on STM32F407 @168 MHz",
              "note": "Analytical estimates reconciled against FULL-PIPELINE on-device DWT ground truth (DISCO-F407VG over SWD). The routing gate, delta-theta defense and fixed-basis rotation are now LIVE in firmware (mcu/fw/src/main.cc) and each stage is DWT-measured over 50 frames with 4 real feature seeds (25 routed to each specialist): T_sense 4,705 cyc (DMA IDLE handler + ASCII parse + window + quantize), T_theta 284 cyc (windowed penetration + rotation_angle), T_gate 117 cyc (folded 12x1 FC + logistic LUT + route; standardization folded into weights offline - bit-match verified), T_rotate 839 cyc (fixed-basis Givens; newlib cosf/sinf dominate - the analytical 168 cyc underestimates trig 5x), T_inf mean 16,320 cyc (routed specialist on the rotated input; input-dependent, per-route means 16,319 NSL / 16,322 UNSW). Full path 22,266 cyc = 0.1321 ms = 1.32% of the 10 ms SLA. The prior dummy-input T_inf measurement (15,719/15,718 cyc) is retained below for reference. Counter certified earlier: DWT vs SysTick within 3 cyc over 8.43e6; rate 168.56 MHz. GPIO-toggle-vs-DWT scope leg still needs an external logic analyzer.",
              "measured_on_device": {
                  "board": "DISCO-F407VG via ST-LINK/V2 (usbipd) + OpenOCD SWD",
                  "method": "firmware DWT_CYCCNT per frame around each defense stage into SRAM (g_stage_cyc[5][50] + g_route[50]); 50 frames, 4 real feature seeds cycled; read back over SWD.",
                  "stages_cycles_mean": MEASURED["stages_cycles"],
                  "full_path_cycles_mean": MEASURED["full_path_cycles"],
                  "full_path_ms": MEASURED["full_path_ms"],
                  "sla_ratio_pct": MEASURED["sla_ratio_pct"],
                  "route_split": MEASURED["route_split"],
                  "T_inf_by_route": {"NSL_specialist_mean": 16318.6, "UNSW_specialist_mean": 16322.0},
                  "invoke_only_dummy_input_mean": {"nsl": 15719.0, "unsw": 15718.0},
                  "overhead_ratio_vs_analytical_typical_full_path": 2.18,
                  "overhead_ratio_vs_analytical_typical_invoke_only": 2.648},
              "clk_hz": CLK_HZ, "sla_budget_ms": SLA_MS,
              "cost_model": {"cyc_mac": CYC_MAC_SCEN, "fc_base": CYC_FC_BASE,
                             "quant_per_out": CYC_QUANT, "activation_elem": CYC_ACTIVATION_ELEM,
                             "interpreter_overhead": CYC_INTERP,
                             "sense_chars_frame": SENSE_CHARS_PER_FRAME,
                             "cyc_per_char": CYC_PER_CHAR,
                             "cyc_theta": CYC_THETA, "cyc_rotate": CYC_ROTATE,
                             "cyc_gate": {"dims": GATE_DIMS, "cycles": GATE_DIMS * 6 + CYC_GATE_LOGISTIC_LUT,
                                           "logistic_lut": CYC_GATE_LOGISTIC_LUT,
                                           "sram_bytes": 21, "fp": "mcu/trained_weights/routing_gate.npz"}},
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