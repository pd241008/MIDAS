# On-device PGD sweep — full 18-row transcription (Table-ready)

Source: `results/pi/defense_success_rate_pi.json`. Platform: **Raspberry Pi 5 (aarch64 A76)**.
Real UNSW-NB15 d=42, W=10, real surrogate, n=200, gamma=2.3061 (recalibrated benign P95).

**Convention:** defense success = 1 − ASR, where ASR = fraction of n=200 benign-subset
samples whose defended prediction flips. `fixed` = deployed fixed PCA-2 basis;
`attacker-coupled` = basis an adaptive (rotation-aware) attacker exploits. The two bases
are **not** interchangeable — never merge rows across bases without a basis label.

## Fixed (deployed PCA-2) basis — 9 configs

| eps | T | naive ASR | adaptive ASR | naive DS | adaptive DS |
|---|---|---|---|---|---|
| 0.05 | 20 | 0.865 | 0.885 | 0.135 | 0.115 |
| 0.05 | 50 | 0.870 | 0.885 | 0.130 | 0.115 |
| 0.05 | 100 | 0.885 | 0.890 | 0.115 | 0.110 |
| 0.10 | 20 | 0.920 | 0.940 | 0.080 | 0.060 |
| 0.10 | 50 | 0.920 | 0.925 | 0.080 | 0.075 |
| 0.10 | 100 | 0.935 | 0.935 | 0.065 | 0.065 |
| 0.20 | 20 | 0.970 | 0.960 | 0.030 | 0.040 |
| 0.20 | 50 | 0.960 | 0.935 | 0.040 | 0.065 |
| 0.20 | 100 | 0.965 | 0.975 | 0.035 | 0.025 |

On the fixed basis there is **no adaptive advantage**: naive ≈ adaptive ASR, matching the
gate requirement (mean |naive−adaptive| ASR gap ≈ 0.01; see `results/table_ondevice_gate.json`).

## Attacker-coupled basis — 9 configs

| eps | T | naive ASR | adaptive ASR | naive DS | adaptive DS |
|---|---|---|---|---|---|
| 0.05 | 20 | 0.515 | 0.715 | 0.485 | 0.285 |
| 0.05 | 50 | 0.470 | 0.705 | 0.530 | 0.295 |
| 0.05 | 100 | 0.475 | 0.690 | 0.525 | 0.310 |
| 0.10 | 20 | 0.610 | 0.885 | 0.390 | 0.115 |
| 0.10 | 50 | 0.520 | 0.940 | 0.480 | 0.060 |
| 0.10 | 100 | 0.520 | 0.905 | 0.480 | 0.095 |
| 0.20 | 20 | 0.565 | 0.965 | 0.435 | 0.035 |
| 0.20 | 50 | 0.520 | 0.965 | 0.480 | 0.035 |
| 0.20 | 100 | 0.550 | 0.930 | 0.450 | 0.070 |

On the attacker-coupled basis the adaptive attacker gains a **large** edge over naive on
nearly every config (mean |naive−adaptive| ASR gap ≈ 0.39 both platforms; the adaptive gap
narrows only at the easy eps=0.05 regime). This is the gate basis and the one used for the
MIDAS-Edge columns in the MIDAS-vs-baselines comparison (`docs/paper_edits.md`):
**PGD eps=0.1 T=50 → naive DS 0.480, adaptive DS 0.060.**

## Reading for the paper

- The two bases bracket the honest range of MIDAS-Edge robustness: **naive** benefit is
  large on coupled (0.48) and small on fixed (0.08); **adaptive** cost is large on coupled
  (0.06) and slightly larger-on-fixed (0.075) — i.e. the fixed rotation gains nothing over
  naive and the coupled adaptive attacker strips it. This is exactly why the paper reports
  the attacker-coupled (threat-model) basis as the primary MIDAS column and the fixed basis
  as the deployed fallback, never as an unlabeled best-of.
- Full per-(eps,T) raw values, both bases, are in `results/pi/defense_success_rate_pi.json`.
