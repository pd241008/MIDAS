# Post-mortem: DWT base address bug — the counter never ran

- Date: 2026-09-07 (fixed in commit `2198424`)
- Severity: measurement-blocker (silent; every cycle metric was wrong by 8×)

## Symptom

Firmware timestamps printed `cyc 0` for every Invoke; the "distance from
routine" readout was frozen at its init value (12). Worse, the numbers were so
consistently wrong (stdev exactly 0, all identical) they looked like a clean
"data-independent" measurement.

## Root cause

`mcu/fw/src/stm32f4xx.h` defined

```
DWT_CTRL  = SCS_BASE + 0x100 + 0x000   // 0xE000E100
DWT_CYCCNT = SCS_BASE + 0x100 + 0x004  // 0xE000E104   ← WRONG
```

`0xE000E100` is the **SCS** region (MPU/DBG), not DWT — the real DWT is
`0xE0001000` (`DWT_CTRL=+0`, `DWT_CYCCNT=+0x4`), and the counter also needs
`DEMCR.TRCENA` (0x01000000 at 0xE000EDFC) + `DWT_CTRL.CYCCNTENA` (0x1).
Neither was set, so reads returned 0.

## Detection

The pattern was impossible, so we probed the register itself over SWD instead
of trusting the printed output: `DWT_CYCCNT == 0` while SysTick ticked — a
counter that never ran. After the fix, DWT was cross-validated two ways
(DWT↔SysTick ±3 cyc over 8.43e6; DWT rate ≈168.56 MHz vs a host-timed window)
before any latency claim was accepted.

## Prevent

- Always read back the counter, not the derived metric; a metric with `stdev=0`
  across 50 runs of a data-*dependent* stage is the tell.
- Validate with a second timebase (SysTick) before trusting a counter.
- Centralize register defines (bare-metal headers are the usual villain) and
  confirm against the RM reference memory map, not by adjoining offsets.