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
| PA3  | USART2 RX          | (unused, reserved)           |
| PD12 | LED Green          | Boot/done indicator          |
| PD13 | LED Orange         | Iteration marker             |
| PD14 | LED Red            | Error indicator              |
| PD15 | LED Blue           | Iteration marker             |

## Memory Budget (measured)

| Resource | Used    | Budget  | Utilization |
|----------|---------|---------|-------------|
| Flash    | 53.4 KB | 1024 KB | 5.2%        |
| SRAM     | 17.4 KB | 192 KB  | 9.1%        |

SRAM breakdown:
- Tensor arena: 16,384 B (reused sequentially for both models)
- MicroInterpreter: 208 B
- MicroMutableOpResolver<3>: 120 B
- Input buffer: 12 B
- DMA RX byte ring: 256 B (USART2 RX, DMA2 Stream5 Ch4, circular)
- Float window (W=10 × 4 × d=12): 480 B
- Vector parser state: ~100 B
- Stack + newlib: ~320 B

Ring-buffer ingest (DMA RX byte ring + W-deep float window = +~840 B SRAM)
keeps the board light: 53.4 KB flash / 17.4 KB SRAM, well within budget.

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
4. 5 iterations: quantize dummy input → `Invoke()` both models → print
   cycle counts (DWT CYCCNT) + dequantized probabilities
5. Production ingest: USART2 DMA RX ring + IDLE-line framing; on each parsed
   host vector run both specialists, print cycle counts + probabilities
6. Idle (WFI)

## DMA RX Ring Buffer (Item 6)

Host streams one feature vector per line over USART2 RX at 115200 8N1:

```
0.5000,0.3000,0.1000,0.7000,0.2000,0.4000,0.6000,0.8000,0.3000,0.9000,0.1000,0.5000
```

- **DMA2 Stream5 Channel4** (USART2_RX) runs in circular mode into a 256 B
  byte ring — the CPU never touches the RX path.
- The **USART2 IDLE interrupt** fires on the inter-frame gap; it computes the
  DMA write position from `NDTR`, drains the new bytes, and (HAL-style)
  pauses `DMAR` while clearing IDLE to avoid an RXNE/DMA race.
- A compact fixed-point ASCII parser (no `strtof` dependency) converts each
  frame into a 12-float vector, shifts it into a **W=10-deep float window**
  for the defense, and quantizes the latest vector into the INT8 input tensor.
- Inference runs only once a full window is present (`g_new_frame`), so the
  rotational defense has the trailing 10-vector trajectory on-die.

Field formats `f1,...,f12\n` (`\r` accepted). Non-finite or malformed lines
are skipped by the parser.

## Production Notes

- In production, `g_input[12]` is filled by the DMA RX parser (Item 6), and
  the W=10 × 4 × d=12 float window holds the trailing trajectory for the defense
- Deployment-time provenance routing selects which specialist runs
- Both models share the 16 KB arena (only one runs at a time)
- Ops registered: FULLY_CONNECTED, LOGISTIC, TANH (3 ops only)
- Quantization: input scale=1/255, zp=-128; output scale=1/256, zp=0
