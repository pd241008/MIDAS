# Post-mortem: OpenOCD breakpoint-pair sampling stalled at exactly 10 (twice)

- Date: 2026-09-07 (found during the T_inf measurement, prior commit `2198424`)
- Severity: workflow (measurement stalled, needed a workaround)

## Symptom

A fine-grained firmware-upload loop — OpenOCD setting a breakpoint at the
Invoke entry/exit and pausing to read the DWT counter via `polling` — stalled
at exactly **10 samples** on two separate runs, on different firmware builds.
Not a firmware crash: the board kept running after manual continues; the
host-side poll link had wedged on the ST-Link/V2.

## Root cause

Not fully diagnosed (ST-Link firmware behavior, not ours). The two runs
stalling at the same count (10) suggests the ST-Link's poll/breakpointing
state got out of sync rather than a data issue — OOCD kept asserting "target
running" states that never delivered the breakpoint event.

## Detection

- 10 is suspiciously round (kNumIters=10 at the time) — but the *board* kept
  producing an array that read back correctly when dumped wholesale.
- A full-SRAM `dump_image` + parsing gave complete, correct results, proving
  the firmware side had finished far beyond sample 10.

## Workaround (now the standard procedure)

Stop polling per-sample. The firmware runs the whole benchmark unhindered into
SRAM arrays (`g_stage_cyc[5][50]`, `g_route[50]`, `g_bench_done`), then the
host does a **single halt + poll `g_bench_done` + two `dump_image`s** and
shuts down — `mcu/openocd/mcu_dump_defense.cfg`. Total readback time is
independent of sample count.

## Prevent

- Design firmware so a single halt captures everything (arrays, not per-sample
  breakpoints) — it is also faster (the DWT measures the unperturbed run).
- If you must poll, add a timeout and re-trigger; do not leave a half-read
  state (the wedged ST-Link needs the port re-opened).