# ADR 0003 — Port defense pipeline, gate, and rotation into firmware; DWT-measure per stage

- Status: Accepted
- Date: 2026-09-07

## Context

The paper's on-μC latency claim had to move from analytical modeling to
measured ground truth. The firmware previously only benchmarked `Invoke()`
(dummy input, 15,719/15,718 cyc, commit `2198424`). The routing gate, the
delta-theta defense (windowed penetration epsilon + `rotation_angle`) and the
fixed-basis rotation existed only in Rust (`core/src/trajectory.rs`,
`core/src/rotation.rs`) and Python.

## Decision

Port the whole frame path into `mcu/fw/src/main.cc` as one live production
pipeline — DMA-IDLE ingest + ASCII parse + window push + quantize (T_sense),
windowed penetration + `rotation_angle` (T_theta), folded gate + route
(T_gate), fixed-basis Givens rotation (T_rotate), quantize + routed Invoke
(T_inf) — and record the DWT cycle counter around each stage into SRAM arrays
`g_stage_cyc[5][50]` + `g_route[50]`, read back by a single-halt OpenOCD dump
(`mcu/openocd/mcu_dump_defense.cfg`, parser `mcu/scripts/parse_defense_capture.py`).

Measured (50 frames, 4 real feature seeds, 25 routed to each specialist,
168.56 MHz DWT counter):

| stage | mean cyc | ms | how measured |
|---|---|---|---|
| T_sense   | 4,705  | 0.0279 | DMA IDLE handler + parse + window + quantize |
| T_theta   | 284    | 0.0017 | penetration + rotation_angle (297 steady-state) |
| T_gate    | 117    | 0.0007 | folded FC + logistic LUT + route |
| T_rotate  | 839    | 0.0050 | fixed-basis Givens (newlib cosf/sinf; 844 steady) |
| T_inf     | 16,320 | 0.0968 | routed specialist on rotated input |
| **full path** | **22,266** | **0.1321 = 1.32% of 10 ms SLA** | |

Sanity: T_inf NSL 16,318.6 / UNSW 16,322.0; DWT↔SysTick ±3 cyc over 8.43e6;
DWT rate ≈168.56 MHz vs host clock.

## Key constraints the measurement depends on

- **Call-boundary DWT windows**: stage helpers are `__attribute__((noinline))`
  so GCC cannot sink pure inline math past the trailing volatile counter read
  (see postmortem `dwt-window-code-sinking`).
- **kNumIters=50 warm-up**: first W=10 frames vaccuum the window, so T_theta
  min is 28 cyc; steady-state means are the published ones.

## Alternatives / known deviations (honest framing)

- **Gamma is a single scalar on-device** (`gamma=0.292`) while the deployment
  config is per-source (0.970044 nsl / 0.612702 unsw, see ADR 0001). The
  measured numbers describe the firmware as built; the per-source gamma
  selection is a documented follow-up that does not change the stage costs.
- **Electrical layer not exercised**: no USB-UART dongle / wire on PA3
  (UM1472 keeps the ST-Link VCP off USART2), so T_sense is measured on the
  simulated-IDLE ingest path, not on physical RS-232 bytes. The ingest *compute*
  is identical; the trigger source is simulated.

## Consequences

- Paper claim now grounded: full path 0.132 ms = 1.32% of the 10 ms SLA,
  analytics over-predict the budget 2.18× headroom at worst.
- Stage costs revealed rotate's real cost (newlib trig, ~5× the naive model).