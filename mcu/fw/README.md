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
| Flash    | 52.2 KB | 1024 KB | 5.1%        |
| SRAM     | 16.8 KB | 192 KB  | 8.8%        |

SRAM breakdown:
- Tensor arena: 16,384 B (reused sequentially for both models)
- MicroInterpreter: 208 B
- MicroMutableOpResolver<3>: 120 B
- Input buffer: 12 B
- Stack + newlib: ~320 B

Ring buffer headroom at d=12: 480 B (W=10 × 4 × 12) — well within
remaining 175 KB SRAM.

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
5. Idle (WFI)

## Production Notes

- In production, `g_input[12]` is filled from the ring buffer
  (W × 4 × d = 480 B at W=10, d=12), not dummy values
- Deployment-time provenance routing selects which specialist runs
- Both models share the 16 KB arena (only one runs at a time)
- Ops registered: FULLY_CONNECTED, LOGISTIC, TANH (3 ops only)
- Quantization: input scale=1/255, zp=-128; output scale=1/256, zp=0
