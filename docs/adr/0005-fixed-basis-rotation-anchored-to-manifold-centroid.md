# ADR 0005 — Fixed rotation basis anchored to the manifold centroid

- Status: Accepted
- Date: 2026-09-07

## Context

The delta-theta defense rotates each query into the rotation-invariant manifold
plane before inference. The rotation must be a **fixed, offline-computable**
basis — per-query on-device PCA is out of budget and non-deterministic across
trained models. `core/src/rotation.rs` exposes `compute_fixed_basis(centroid)`
but its PCA-2 candidate is a documented TODO.

## Decision

Compute the basis **offline at image-build time** and embed as flash constants:

```
b0 = c_base / ||c_base||          (c_base = mean of unified_features_norm.npy)
b1 = Gram-Schmidt(e1, b0), normalized
```

`g_b0[12]` / `g_b1[12]` are 96 B of `.rodata` (flash; SRAM-neutral). The
firmware Givens-rotates the quantized float window snapshot in the `(b0, b1)`
plane with `rotation_angle(eps, delta_theta_max=45°)` flowing from the windowed
penetration epsilon (see `core/src/trajectory.rs` semantics, ported 1:1),
then feeds the rotated vector to the routed specialist.

## Consequences

- T_rotate = 839 cyc ≈ 5.0 µs, dominated by newlib `cosf`/`sinf` (the naive
  168-cyc model was ~5× low).
- Bit-held on-device basis keeps host/firmware rotation consistent (both derive
  from the same `c_base`).

## Follow-ups

- Replace the mean-centroid anchor with the host's real PCA-2 once computed
  (`compute_fixed_basis` TODO); then regenerate `g_b0`/`g_b1` — the firmware
  path is unchanged.