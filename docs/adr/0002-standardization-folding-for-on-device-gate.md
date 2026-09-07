# ADR 0002 — Fold gate standardization into the weights for on-device gate

- Status: Accepted
- Date: 2026-09-07

## Context

The host-trained routing gate standardizes features before the logistic layer:
`z = (x − mu) / sd`, `logit = z@W + b`. Porting that literally onto the μC for
the 12-dim vector needs `mu`, `sd`, `W`, `b` (42 floats) plus 12 divisions per
frame — wasteful and a source of float divergence vs the host result that the
paper's latency-vs-accuracy link depends on.

## Decision

Fold standardization into the weights **offline, at image-build time**:

```
An  =  W / sd                          (12 floats)
bn  =  b − dot(mu, W / sd)             (1 float)
logit = x@An + bn                       (route iff > 0  ⇔  sigmoid > 0.5)
```

Constants live in flash as `g_An[12]`, `g_bn` (no per-frame divisions, no risk
of a mis-typed scale/shift). Verified before embedding:
**sign-agreement 1.0 and max |Δlogit| ≤ 3.81e-6** vs the reference path over
the full 12-dim feature set.

## Alternatives considered

- Runtime `(x−mu)/sd` — rejected: 12 extra divs, 42 floats in RAM, divergence risk.
- Full-INT8 folded weights (the paper's 21-B compressed variant) — deferred:
  the measurement tier keeps the folded **float32** gate (0.7 µs) so on-device
  routing is bit-identical to the host's float path; the int8 variant is a
  documented compression follow-up, not what T_gate measures.

## Consequences

- T_gate = 117 cyc ≈ 0.7 µs, stdev 0 (data-independent).
- SRAM-neutral (gate consts are flash-resident); only the feature snapshot
  `g_x[12]` (48 B) is added.