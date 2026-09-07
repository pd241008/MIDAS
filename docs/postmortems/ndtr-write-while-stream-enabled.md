# Post-mortem: NDTR write ignored while the stream is enabled

- Date: 2026-09-07 (fixed in commit `0bf2e08`)
- Severity: medium (sim-only; the fake-ingest helper couldn't inject lines)

## Symptom

`simulate_rx_line()` wrote NDTR to prime the circular ring, but `avail` stayed
0 and no line was consumed — the DMA never "saw" the fake byte even though
USART2_RX had apparently used a transfer.

## Root cause

On the STM32F4, `NDTR` is the number of items **remaining**; the peripheral
reloads/banks it on completion and **software writes to NDTR while
`CR.EN=1` are ignored** (RM0090: program NDTR only when the stream is
disabled). The helper wrote NDTR with the stream live, so nothing happened.
(RMT that NDTR is not a byte count for a byte-sized peripheral here — item
size = Byte, but the write-before-disable rule is the real one.)

## Detection

Stepped the fake-ingest path under OpenOCD and probed NDTR/EN directly;
EN=1 && NDTR untouched → write dropped.

## Fix

Sequence for reseeding the simulated buffer: clear EN, wait until EN reads 0,
write NDTR, set EN, (buffer gets filled by the next line send). The firmware
uses this in the trace-mode line injector.

## Prevent

- Treat NDTR as read-only while EN=1; always EN=0 → wait EN==0 → write → EN=1.
- Note the interplay: this is the same class of "silently dropped writes" as
  the DMA clock-gating bug — on the F4, if a DMA register write doesn't stick,
  check clock first, EN-state second.