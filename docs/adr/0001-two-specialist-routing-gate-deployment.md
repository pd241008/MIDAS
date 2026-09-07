# ADR 0001 — Two-specialist routing-gate deployment on the MCU

- Status: Accepted
- Date: 2026-09-07
- Deciders: MIDAS-Edge MCU tier (firmware + cycle-model + measurement)

## Context

MIDAS trains two per-source INT8 specialists on the merged 12-dim overlap
features (`mcu_models/mcu_specialist_{nsl,unsw}_int8.tflite`, 2,528 B each).
At runtime a single 12-dim feature vector arrives per frame and must hit one
specialist. The paper section VI-A/VII-A mechanism requires a **routing gate**
that classifies each query provenance before the specialists see it.

## Decision

Ship a **three-component deployment frame** on the μC: `prob = sigmoid(x@W + b)`
where `(W, b)` come from a host-trained ridge-logistic on labeled provenance
(`mcu/trained_weights/routing_gate.npz`; route-mechanism accuracy 0.8766).
Route **UNSW specialist iff `prob > 0.5`**, else NSL. The gate runs in the
defense stage (T_gate), in parallel with the delta-theta rotation, so neither
specialist runs on the wrong provenance and only the routed one is invoked.

## Alternatives considered

- Single merged model — rejected: per-source QAT accuracy loss, and loses the
  explicit provenance mechanism the paper needs.
- Per-feature source labels (row-wise `source` column) — rejected in commit
  `7b78476`: the gate, not the label, provides the 0.8766 mechanism-backed
  claim.

## Consequences

- Per frame only one Invoke (the routed specialist) → T_inf = 16,320 cyc on
  rotated real features, 25/25 split across the four seeds in our 50-frame run.
- Gate cost is 117 cyc (12 FMAs + logistic LUT) — negligible.
- Confirmed by on-device measurement (`mcu/results/fw_ondevice_defense_pipeline_20260907T030335Z.json`).

## Follow-ups

- Per-source gamma (`configs/mcu_config.json`: gamma 0.970044 nsl / 0.612702
  unsw) is **not yet** applied on-device; firmware uses a single scalar
  `gamma=0.292`. The routed specialist should select its gamma (see ADR 0003).