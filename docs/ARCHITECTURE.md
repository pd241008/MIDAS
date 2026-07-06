# Architecture

## Pipeline

EDGE runs three pinned threads communicating over bounded crossbeam channels:

```
                     bounded(4)                    bounded(4)
  Sensor Thread ──────────────────▶ Defense Thread ──────────────────▶ Rotation Thread
  (Core 0)                          (Core 1)                           (Core 2)
      │                                │                                   │
      │  read feature vector           │  compute momentum & epsilon       │  Givens rotation
      │  via SensorSource trait        │  decide rotation angle            │  inference via
      │  (blocking send if full)       │  send DefenseCommand              │  InferenceModel trait
      ▼                                ▼                                   ▼
  MockSensor /                   RingBuffer                          MockModel /
  real TCP/UART                  (Arc<RwLock<>>)                     TfliteModel (FFI)
```

### Backpressure

All channels use **blocking `send`** (not `try_send`). When the bounded channel
(`channel_capacity`, default 4) is full, the producer blocks. This propagates
backpressure all the way to the sensor source, matching the paper's description and
preventing data loss.

## Math

### Momentum

```
M_t = cos_sim(v_t, v_{t-1}) = dot(v_t, v_{t-1}) / (||v_t|| * ||v_{t-1}||)
```

Guarded against zero-norm vectors: if either norm is below `f32::EPSILON`, returns
`0.0`.

### Penetration epsilon

```
epsilon_p = ||v_t - v_{t-1}||_2 * max(0, M_t)
```

The `max(0, ·)` gate ensures that anti-aligned trajectories (`M_t < 0`) produce
zero penetration, avoiding unnecessary rotation.

### Rotation angle

**Archimedean phase** (`epsilon_p <= gamma`):

```
delta_theta = lambda * epsilon_p
```

**Logarithmic phase** (`epsilon_p > gamma`):

```
delta_theta = min(lambda * exp(k * (epsilon_p - gamma)), delta_theta_max)
```

The logarithmic phase provides aggressive rotation scaling when the adversarial
trajectory penetrates deeper, while `delta_theta_max` caps the angle to prevent
instability.

### Givens rotation

A 2D rotation in the `(0, 1)` plane:

```
x[0]' = cos(theta) * x[0] - sin(theta) * x[1]
x[1]' = sin(theta) * x[0] + cos(theta) * x[1]
```

### Budget-gated fallback

```rust
let deadline = Instant::now() + sla_budget;
let rotated = rotate_manifold_givens(x_adv, theta);
if Instant::now() > deadline {
    project_to_manifold(x_adv, c_base)  // fallback to unrotated
} else {
    rotated
}
```

## Harness

The harness generates attack configs matching the paper's hyperparameter table:

| Attack | Parameters |
|--------|------------|
| PGD | `alpha=0.01`, `T in {20, 50, 100}`, `epsilon in {0.05, 0.1, 0.2}` |
| FGSM | `epsilon in {0.05, 0.1, 0.2}` |
| C&W L2 | `kappa=0`, 9 binary search steps |

Baselines implemented as strategy stubs:
- Undefended
- Adversarial training (accepts pre-trained weights)
- Input smoothing (Gaussian sigma=0.1)
- Chen et al. query blinding (similarity threshold reject)
- **MIDAS-Edge** (calls the real pipeline over TCP)

Report writers emit CSV/JSON shaped exactly like the paper's three result tables:
- Defense Success Rate vs. Baselines
- Latency Distribution (p50/p95/p99/max)
- Defense Rate vs. PGD Iteration Count

## Configuration

See [`configs/edge_config.json`](../configs/edge_config.json). All 11 hyperparameters
are validated at startup with fail-fast errors on out-of-range values. Set via the
`EDGE_CONFIG` environment variable.
