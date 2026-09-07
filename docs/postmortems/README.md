# Post-mortems — Midas MCU tier measurement failures

Every bug below was found by looking at **measured patterns that were too
regular or too impossible to be true** (stdev 0, cyc=0, or an always-same route
split), not by reading the code first. That is the recurring lesson.

| Date | Incident | Symptom | Root cause | Fix commit |
|------|----------|---------|------------|------------|
| 2026-09-07 | [dwt-base-address](dwt-base-address-bug.md) | all DWT reads = 0, "dist" stuck at 12 | `DWT_BASE` pointed at SCS (0xE000E100) instead of DWT (0xE0001000); CYCENA never set | `2198424` |
| 2026-09-07 | [dma-stream-and-clock](dma-stream-and-clock-bug.md) | ingest dead: `win_cnt=0`, all routes UNSW, sense=109 cyc | macros addressed DMA1 "Stream6" (offset 0xA0) while clocking unclocked DMA1; real USART2_RX is DMA1 S5 Ch4 | `0bf2e08` |
| 2026-09-07 | [ndtr-en](ndtr-write-while-stream-enabled.md) | simulated ingest never consumed lines | `NDTR` writes ignored while Stream EN=1 | `0bf2e08` |
| 2026-09-07 | [dwt-window-code-sinking](dwt-window-code-sinking.md) | T_gate = 2 cycles, stdev 0 | GCC sank pure inline math past the trailing volatile DWT read | `0bf2e08` |
| 2026-09-07 | [openocd-breakpoint-sampling-stall](openocd-breakpoint-sampling-stall.md) | fine-grained polling stalled at exactly 10 samples (twice) | ST-Link breakpoint poll hiccup; firmware fine | workaround only |

Full text in each file, including detection signal, timeline, and the
prevention rule.