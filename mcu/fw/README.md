# STM32F407VG TFLM Firmware

Bare-metal TFLM inference firmware for the STM32F407VGT6 Discovery board
(DISCO-F407VG, Cortex-M4F, 192 KB SRAM, 1 MB flash).

Runs two per-source QAT INT8 specialists (NSL / UNSW) on shared 12-dim
overlap features. Outputs cycle counts + quantized probabilities via
USART2 @ 115200 8N1.

## Prerequisites

- `arm-none-eabi-gcc` 14+ (installed via `pacman -S arm-none-eabi-gcc`)
- TFLM v2.21.0 full checkout at `/tmp/opencode/tflm/tflite-micro/`
  with `libtensorflow-microlite.a` already built for `cortex-m4+fp`

## Build

```bash
cd mcu/fw
make
```

Produces:
- `build/mcu_fw.elf` — ELF with debug symbols
- `build/mcu_fw.bin` — raw binary (for `st-flash`)
- `build/mcu_fw.hex` — Intel HEX (for OpenOCD / STM32CubeProgrammer)
- `build/mcu_fw.map` — linker map

## Flash

Requires ST-LINK attached via `usbipd` (WSL) or Windows-side tools.

```bash
# Option A: st-flash (WSL, requires usbipd attach)
make flash

# Option B: STM32CubeProgrammer (Windows)
# Use build/mcu_fw.bin @ 0x08000000
```

## Hardware

| Pin  | Function           | Notes                        |
|------|--------------------|------------------------------|
| PA2  | USART2 TX          | Serial output @ 115200 baud  |
| PA3  | USART2 RX          | Ingest input (DMA); HDSEL loopback validated |
| PD12 | LED Green          | Boot/done indicator          |
| PD13 | LED Orange         | Iteration marker             |
| PD14 | LED Red            | Error indicator              |
| PD15 | LED Blue           | Iteration marker             |

## Memory Budget (measured, from `sram_flash_reconciliation.py`)

| Resource | Used    | Budget  | Utilization |
|----------|---------|---------|-------------|
| Flash    | 60.9 KB | 1024 KB | 5.9%        |
| SRAM     | 18.7 KB | 192 KB  | 9.7%        |

SRAM breakdown (19,166 B):
- Tensor arena: 16,384 B (reused sequentially; only the routed specialist runs/frame)
- Benchmark DWT capture (non-production): 1,050 B (`g_stage_cyc[5][50]` + `g_route[50]`)
- State/pointers/other: 624 B
- Middleware (MicroInterpreter + resolver): 344 B
- newlib/stdio (FILE, reent): 316 B
- Live ingest DMA byte ring: 256 B
- Live ingest parser + input: 100 B

The routing-gate weights (`g_An`/`g_bn`, 52 B), the fixed rotation basis
(`g_b0`/`g_b1`, 96 B) and the 64-entry sigmoid LUT (256 B) all reside in
`.rodata` (flash) — they add no linker-allocated SRAM beyond the `g_x[12]`
snapshot (48 B). The W=10 float window (480 B) is READ by
`penetration_epsilon_windowed` (live defense, not a dead store).

Flash: .text 62,316 B + .data 92 B = 62,408 B. Both INT8 specialists are
2,528 B each. The deployed MIDAS tier over the TFLM baseline (ring + C matrices
at d=12 + state + folded gate) is ~1.70 KB incremental (see reconciliation).

## TFLM Library

The firmware links against `libtensorflow-microlite.a` (1,577 KB,
~480 object files) built from TFLM v2.21.0 for cortex-m4+fp:

```bash
# Build the library (one-time)
cd /tmp/opencode/tflm/tflite-micro
make -j8 -f tensorflow/lite/micro/tools/make/Makefile microlite \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp \
  TARGET_TOOLCHAIN_ROOT=/usr/bin/
```

## Inference Flow

1. Boot → `SystemInit` → PLL 168 MHz (HSE 8 MHz) → USART2 + GPIO init
2. Load NSL specialist (2,528 B INT8 TFLite) → `AllocateTensors`
3. Load UNSW specialist (2,528 B INT8 TFLite) → `AllocateTensors`
4. Trace-mode self-check: cycles 4 real feature seeds (2 NSL / 2 UNSW) through
   the full defense pipeline, recording per-stage DWT counts to
   `g_stage_cyc[5][50]` + `g_route[50]` (read back by OpenOCD;
   `mcu/scripts/parse_defense_capture.py` prints the published stage table)
5. Production ingest: USART2 DMA RX ring + IDLE-line framing; each parsed
   host vector → window push → gate routing → delta-theta rotation → routed
   specialist Invoke → prints probabilities + per-stage cycle counts
6. Idle (WFI)

## DMA RX Ring Buffer (Item 6)

Host streams one feature vector per line over USART2 RX at 115200 8N1:

```
0.5000,0.3000,0.1000,0.7000,0.2000,0.4000,0.6000,0.8000,0.3000,0.9000,0.1000,0.5000
```

- **DMA1 Stream5 Channel4** (USART2_RX — the correct mapping per RM0090/AN4031;
  an earlier header bug pointed the macros at DMA1 "Stream6" offset 0xA0 while
  clocking DMA2EN, silently dropping every register write) runs in circular
  mode into a 256 B byte ring — the CPU never touches the RX path.
- The **USART2 IDLE interrupt** fires on the inter-frame gap; it computes the
  DMA write position from `NDTR`, drains the new bytes, and (HAL-style)
  pauses `DMAR` while clearing IDLE to avoid an RXNE/DMA race.
- A compact fixed-point ASCII parser (no `strtof` dependency) converts each
  frame into a 12-float vector, shifts it into a **W=10-deep float window**
  for the defense, and quantizes the latest vector into the INT8 input tensor.
- Per frame the **defense pipeline** runs: windowed penetration epsilon →
  `rotation_angle` → folded routing gate (logistic LUT) → fixed-basis Givens
  rotation → quantize → routed specialist Invoke. Each stage is DWT-timestamped
  (call-boundary `noinline` stages; GCC never sinks them out of the window).

Field formats `f1,...,f12\n` (`\r` accepted). Non-finite or malformed lines
are skipped by the parser.

## Production Notes

- In production, `g_input[12]` is filled by the DMA RX parser (Item 6), and
  the W=10 × 4 × d=12 float window holds the trailing trajectory for the defense.
- The on-device routing gate (folded 12×1 FC + logistic LUT, standardization
  folded into the weights) selects the specialist per frame; only the routed
  model runs (measured T_gate = 117 cyc).
- Both models share the 16 KB arena (only one runs at a time).
- Ops registered: FULLY_CONNECTED, LOGISTIC, TANH (3 ops only).
- Quantization: input scale=1/255, zp=-128; output scale=1/256, zp=0.
- Per-stage DWT ground truth and full-path numbers: `docs/mcu-measurement-procedure.md`.

## Hardware validation

### HDSEL single-wire loopback (RX ingest, real electrical bytes)

RM0433 USART half-duplex (`HDSEL`) ties TX and RX onto PA2/PA3 as a single
wire. With PA2 internally pulled up (`PUPDR` bit 2 — required in half-duplex,
without it the line floats and every byte frames as a framing error, `SR` with
`FE` bit), the firmware transmission is echoed back onto its own RX, flows
through DMA1-Stream5 into the ring, and the IDLE handler parses it:

- `g_idle_cnt = 0x0a` — IDLE IRQ fired 10× on real wire traffic
- `SR = 0xc0` (TXE|TC, no `FE`); earlier runs without the pull-up: `SR = 0x1c2`
- `g_win_cnt` 50 → 55 — 5 new frames parsed from the real echoed bytes
- ring contents are clean ASCII `HDSEL loopback done, frames=50\n`

### LA capture of PA2 (wire-level toggle proof)

A Saleae Logic 8 (VID 0925:3881, sigrok `fx2lafw`, 8 ch) captured PA2 directly.
Test build drove PA2 as GPIO at a calibrated toggle (loop cycle math @ 168 MHz);
60 s @ 200 kHz on the probe gave:

- 646 transitions / 60 s = exactly **5.38 Hz**
- duty **50.0 % / 50.0 %**, 645/645 half-periods = 0.0929 s (perfect square wave)
- artifact: `results/hardware/la_pa2_toggle_60s.csv` (+ summary `.txt`)

Note: the probe wire used for the sweep landed on Saleae logical channel 2
(driver maps physical CH1..CH8 → sigrok D0..D7 non-trivially); always re-scan
`--channels D0..D7` and confirm which reads the target.

### GPIO-toggle-vs-DWT cross-check (two independent timebases)

The `DWT_GPIO_CROSS` build toggles PA2 high around each 50-frame defense
pipeline rep while capturing `DWT->CYCCNT` at toggle-high and toggle-low; the
Saleae (500 kHz) measures the same pulse's wall width. 5 identical pulses were
captured on D2:

| rep | LA width (ms) | DWT (cyc)  | implied DWT rate |
|-----|---------------|------------|------------------|
| 0   | 7.5160        | 1,261,688  | 167.87 MHz       |
| 1–4 | 7.5800        | 1,273,545  | 168.01 MHz       |

Inter-pulse spacing agrees too: DWT span 49,261,741 → 49,273,598 cyc vs LA span
293.23 ms → 168.00 MHz. The DWT counter rate therefore reads **168.0 MHz**, within
0.03 % of the 168 MHz PLL and well within the LA's and the 8 MHz HSE crystal's
tolerances — the counter and the logic analyzer corroborate each other.
Artifacts: `results/hardware/dwt_gpio_crosscheck_summary.txt`,
`results/hardware/la_dwt_cross_reps.sr` (+ `.csv` D2 runs).

### Caveats

- External host bytes over a USB-UART dongle are still not exercised; the
  Discovery's ST-Link VCP is not routed to USART2 (UM1472 §6.1.3). The loopback
  closes TX→RX electrically; the missing leg is an external sender/CLK domain.
