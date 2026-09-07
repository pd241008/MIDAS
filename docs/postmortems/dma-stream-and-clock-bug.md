# Post-mortem: DMA stream/base/clock bug — the ingest path was never running

- Date: 2026-09-07 (fixed in commit `0bf2e08`)
- Severity: correctness (data path dead end-to-end; masked until the first
  live-ingest trace)

## Symptom

With the defense pipeline live we turned on ingest tracing and measured:
`g_win_cnt == 0`, `g_dma_head == 0`, T_sense = 109 cyc (i.e. immediate early-out
— no line was ever parsed), and **every frame routed to UNSW** (fake "25/25"
look-alike: actually all-UNSW). The DWT counter itself was fine this time
(0x19e4248 after boot), which isolated the fault to DMA configuration.

## Root cause (three compounding errors in one table)

1. **Wrong peripheral address.** `stm32f4xx.h` put `DMA2_BASE = AHB1_BASE + 0x6000
   = 0x40026000` — that is **DMA1's** base (AHB1 0x40020000; DMA1 at +0x6000,
   DMA2 at +0x6400). The macros named "DMA2 / DMA2_Stream5" were really DMA1.
2. **Wrong stream offset.** `DMA2_Stream5` was `base + 0xA0`; offset 0xA0 is
   **Stream6**. The real Stream5 offset is **0x88**. USART2_RX (RM0090/AN4031)
   is DMA1 Stream5 Channel4; 0xA0 = Stream6 = USART2_TX.
3. **Wrong clock gate / unclocked writes.** `system_init.c` enabled `DMA2EN`
   (bit 22) while every stream write went to DMA1. On the STM32F4, writes to an
   **unclocked** AHB peripheral are silently dropped → NDTR stayed 0 →
   `tail = 256 − 0 = 256` → `avail = (head − tail) & mask = 0`. Hence the parser
   never ran and the window never filled.

## Detection

After the Windows-vs-Linux data-path fixes, we read `g_win_cnt`/`g_dma_head`
over SWD (both 0) and compared the register map against AN4031's USART2 table.
The "all-UNSW routes" looked statistically clean but was physically impossible
(firmware draws 2 NSL seeds); a sanity check on *route distribution* caught it.

## Fix

- Header: `DMA1_BASE=0x40026000`, `DMA2_BASE=0x40026400`, both
  `{DMA1,DMA2}_Stream5 = BASE + 0x88`, `RCC_AHB1ENR_DMA1EN = (1 << 21)`.
- `system_init.c`: enable DMA1EN.
- `main.cc`: use `DMA1_Stream5` for USART2_RX.

## Prevent

- Bare-metal headers: cross-check offsets and bases against the RM memory map
  (`0x4002 6410` ↔ DMA2_LISR made the mis-address obvious in hindsight).
- When registers read back 0 but the clock seems configured, check the **AHB
  clock enable for the peripheral actually being written**.
- Trace data-path state (window count, ring head), not just final metrics —
  `win_cnt==0` was visible long before any latency number looked off.