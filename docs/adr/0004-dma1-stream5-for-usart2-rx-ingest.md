# ADR 0004 — USART2_RX ingest on DMA1 Stream5 Ch4 (circular + IDLE framing)

- Status: Accepted
- Date: 2026-09-07

## Context

Item 6 requires host feature vectors to arrive over USART2 without occupying
the CPU (the MCU spends ~0.13 ms/frame computing; the rest is idle). The
initial firmware shipped with `stm32f4xx.h` macros that addressed the wrong
peripheral and wrong stream, and `system_init.c` gated the wrong AHB clock;
upstream the whole data path appeared dead (see postmortem
`dma-stream-and-clock-bug`).

## Decision

According to RM0090 + AN4031, USART2_RX is **DMA1 Stream5 Channel4**
(alternate: DMA1 Stream7 Ch4); USART2_TX is DMA1 Stream6 Ch4. The firmware:

- Points `DMA1_Stream5 = DMA1_BASE + 0x88` (the real S5 offset; the old macro
  used 0xA0 = Stream6) with `DMA1_BASE = 0x40026000`, `DMA2_BASE = 0x40026400`.
- Clocks **DMA1EN** (RCC_AHB1ENR bit 21) in `system_init.c` — before, DMA2EN
  (bit 22) was set while all code wrote to the unclocked DMA1, so every NDTR/CR
  write was silently dropped.
- Uses a **circular** buffer (256 B) + **IDLE-line** framing; IDLE interrupt →
  `ingest_idle_frame()` parses `len,class,12-floats` ASCII lines and pushes the
  W=10 window used by the defense.

## Consequences

- Ingest compute path live + measured: T_sense = 4,705 cyc incl. IDLE handler,
  ASCII parse, window push, quantize.
- The DMA register fixes were prerequisites: without them `avail` was always 0
  (`tail = 256 − NDTR = 256`), the parser never ran, and `g_win_cnt` stayed 0.

## Follow-ups

- Electrical validation on real PA3 bytes still needs a USB-UART dongle/wire
  (see ADR 0003).