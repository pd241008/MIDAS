# Architecture Decision Records — Midas MCU tier

Decisions recorded for the STM32F407 (DISCO-F407VG) MCU tier of MIDAS-Edge.
Each record captures the decision, the alternative, and the evidence/reasoning,
so paper claims and firmware behavior trace back to a written rationale.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-two-specialist-routing-gate-deployment.md) | Route to a per-source INT8 specialist via an on-μC logistic gate (three-component deployment) | Accepted |
| [0002](0002-standardization-folding-for-on-device-gate.md) | Fold gate standardization into the weights (`x·An+bn`) instead of computing `(x−mu)/sd` at runtime | Accepted |
| [0003](0003-on-device-defense-pipeline-and-dwt-measurement.md) | Ported delta-theta defense + fixed-basis rotation into firmware; stage DWT counters in SRAM, read back over SWD | Accepted |
| [0004](0004-dma1-stream5-for-usart2-rx-ingest.md) | USART2_RX ingest on **DMA1 Stream5 Ch4** circular ring + IDLE framing | Accepted |
| [0005](0005-fixed-basis-rotation-anchored-to-manifold-centroid.md) | Offline-computed fixed rotation basis (centroid-`b0`, Gram-Schmidt `b1`) embedded in flash | Accepted |

All decisions are dated 2026-09-07 unless otherwise noted.