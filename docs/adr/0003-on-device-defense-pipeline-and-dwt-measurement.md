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
- **Electrical layer**: T_sense numbers above were measured on the simulated-IDLE
  ingest path. Two hardware validations now cover the real electrical path:
  1. **HDSEL single-wire loopback** (RM0433, USART HD single-wire mode, see
     ADR 0004). With PA2 internally pulled up (`GPIOA->PUPDR`
     bit 2, required by half-duplex), real bytes echoed from TX→RX flowed
     through DMA into the ring and were parsed: `g_idle_cnt=0x0a` (IDLE IRQ
     fired 10× on real wire traffic), `SR=0xc0` (TXE|TC, no framing error;
     earlier runs without the pull-up showed `SR=0x1c2` FE), `g_win_cnt`
     50→55 (5 new frames parsed from real echoed bytes).
  2. **Saleae Logic 8 capture of PA2 (USART2_TX)**: 60 s @ 200 kHz on the
     finder build that drives PA2 at a calibrated ~5.4 Hz square wave gave
     646 transitions, duty 50.0/50.0 %, 645/645 half-periods = 0.0929 s =
     exactly 5.38 Hz (matches the loop cycle math at 168 MHz), saved in
     `results/hardware/la_pa2_toggle_60s.csv`. The PA2 line electrically
     toggles at the wire level as the firmware drives it.
  Physical RS-232 bytes from an external host are not yet exercised (the
  board's ST-Link VCP is off USART2), but the ingest *compute* is identical and
  the electrical trigger source (real wire toggling on PA2/PA3, real echoed
  bytes through the DMA+IDLE path) has been validated both directions.

## Consequences

- Paper claim now grounded: full path 0.132 ms = 1.32% of the 10 ms SLA,
  analytics over-predict the budget 2.18× headroom at worst.
- Stage costs revealed rotate's real cost (newlib trig, ~5× the naive model).