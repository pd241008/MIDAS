# EDGE

**E**fficient **D**efense via **G**ivens rotatio**n**s — a three-thread, budget-gated
adversarial defense for TinyML classifiers on ARM Cortex-A76.

[![Rust](https://img.shields.io/badge/rust-1.85%2B-blue)](https://www.rust-lang.org)
[![IEEE ICCD 2026](https://img.shields.io/badge/paper-ICCD%202026-purple)](#)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-red)](#)

---

## Overview

EDGE is the reference implementation of the **MIDAS-Edge** pipeline described in an
IEEE ICCD 2026 paper draft. It protects a TinyML image/tabular classifier against
adversarial examples by continuously rotating the decision manifold in reaction to
observed feature trajectories.

The core insight is a **dual-phase rotation schedule**:

| Phase | Condition | Rotation angle |
|-------|-----------|----------------|
| **Archimedean** | `epsilon_p <= gamma` | `delta_theta = lambda * epsilon_p` |
| **Logarithmic** | `epsilon_p > gamma` | `delta_theta = min(lambda * exp(k * (epsilon_p - gamma)), delta_theta_max)` |

A **budget-gated** fallback ensures real-time guarantees: if the Givens rotation
exceeds the SLA budget, the sample is projected onto the unrotated base manifold
instead of being dropped.

## Architecture

```
┌──────────┐   bounded    ┌──────────┐   bounded    ┌──────────┐
│  Sensor  │───channel───▶│ Defense  │───channel───▶│ Rotation │
│  Thread  │              │  Thread  │              │  Thread  │
│  Core 0  │              │  Core 1  │              │  Core 2  │
└──────────┘              └──────────┘              └──────────┘
     │                         │                         │
     │ feature vectors         │ M_t, epsilon_p,         │ Givens rotation +
     │ (trait SensorSource)    │ delta_theta             │ inference
                               │                         │
                          ┌────┴────┐               ┌────┴────┐
                          │  Ring   │               │  Model  │
                          │  Buffer │               │ (trait) │
                          └─────────┘               └─────────┘
```

- **Core 0** — receives feature vectors (TCP/UART mock behind a trait, swappable for
  real hardware).
- **Core 1** — owns the ring buffer, computes cosine-similarity momentum
  `M_t = cos_sim(v_t, v_{t-1})` and penetration epsilon
  `epsilon_p = ||v_t - v_{t-1}||_2 * max(0, M_t)`, decides the rotation angle.
- **Core 2** — applies a budget-gated Givens rotation, runs inference through the
  `InferenceModel` trait, records latency.
- **Core 3** — reserved (unpinned).

## Quick start

```bash
# Run the three-thread pipeline with MockModel (200 synthetic samples)
EDGE_CONFIG=configs/edge_config.json cargo run --bin edge

# Generate LaTeX-ready report tables (CSV + JSON)
cargo run --bin edge -- harness

# Both sequentially
cargo run --bin edge -- all
```

## Crate layout

| Crate | Type | Responsibility |
|-------|------|----------------|
| [`edge-core`](./edge-core) | lib | Ring buffer, trajectory math (`compute_momentum`, `penetration_epsilon`), rotation math (`rotate_manifold_givens`, `project_to_manifold`, `budget_gated_rotation`), config deserialization |
| [`edge`](./edge) | bin | Three-thread runtime, `InferenceModel` trait, `MockModel`/`TfliteModel`, attack harness (PGD, FGSM, C&W), CSV dataset loader, LaTeX-ready report writers |

## Configuration

All hyperparameters from the paper are in [`configs/edge_config.json`](./configs/edge_config.json):

| Field | Default | Description |
|-------|---------|-------------|
| `W` | `4` | Ring buffer window size |
| `D` | `10` | Feature vector dimension |
| `gamma` | `0.5` | Phase transition threshold |
| `lambda` | `1.0` | Rotation scaling factor |
| `k` | `2.0` | Logarithmic growth rate |
| `tau` | `0.3` | Similarity threshold |
| `delta_theta_max_deg` | `45.0` | Maximum rotation angle |
| `sla_budget_ms` | `10` | Budget-gating deadline |
| `channel_capacity` | `4` | Bounded channel size |

Each field is validated against recommended ranges at startup.

## Tests

```bash
cargo test
```

18 unit tests covering momentum computation, penetration epsilon (orthogonal,
anti-aligned, zero-norm edge cases), Givens rotation, phase boundary, budget-gated
fallback, ring buffer eviction, and CSV dataset loading.

## TODO

- [ ] Real TFLite C++ FFI (`libtensorflowlite_c.so`) behind `--features tflite`
- [ ] Load manifold centroid from `config.manifold_path`
- [ ] UNSW-NB15 dataset integration in harness
- [ ] Full C&W L2 attack implementation
- [ ] Experimental results (currently TBD)

## License

MIT
