# EDGE

Stateful adversarial defense for TinyML classifiers, targeting Raspberry Pi 5 / ARM
Cortex-A76. Implementation of the three-thread pipeline described in the IEEE ICCD 2026
paper.

## Crate layout

- **edge-core** — pure math: ring buffer, trajectory (momentum, penetration epsilon),
  Givens rotation (Archimedean/Logarithmic dual-phase), budget-gated fallback.
- **edge** — single binary: three-thread runtime (sensor → defense → rotation) with
  `InferenceModel` trait, `MockModel`, and `TfliteModel` stub. Also includes attack
  harness (PGD, FGSM, C&W stubs, CSV dataset loader, report writers).

## Running

```bash
# Pipeline mode (default): run sensor->defense->rotation with MockModel
EDGE_CONFIG=configs/edge_config.json cargo run --bin edge

# Harness mode: generate report tables matching the paper
cargo run --bin edge -- harness

# Both
cargo run --bin edge -- all
```

## Configuration

Edit `configs/edge_config.json`. Fields match the paper's hyperparameter table.
Validation at startup; fail-fast on out-of-range values.

## Thread topology

- Core 0: Sensor thread (feature vector receiver)
- Core 1: Defense thread (momentum / epsilon / rotation-angle computation)
- Core 2: Rotation thread (budget-gated Givens rotation + inference)
- Core 3: Reserved (unpinned)

Core pinning via the `core_affinity` crate; no-op on non-Linux/non-4-core hosts.

## TODO

- [ ] Replace `MockModel` with real TFLite C++ FFI (`libtensorflowlite_c.so`) behind `--features tflite`
- [ ] Load real manifold centroid from `config.manifold_path`
- [ ] Wire UNSW-NB15 dataset into harness attack loop
- [ ] Full C&W L2 attack implementation
- [ ] Collect experimental results (currently TBD in paper)
