# Measuring on-device defense-pipeline latency (DISCO-F407VG, SWD)

Reproducible recipe for the firmware DWT measurements behind ADR 0003. Uses
only the board's ST-Link (WSL2 `usbipd`, OpenOCD 0.12.0, `st-flash`).

## 0. Prerequisites

- Build host: Arch WSL2 root; the Python capture pipeline needs
  `.venv/bin/python` (numpy).
- Test board: DISCO-F407VG, factory ST-Link/V2 exposed as WSL device `1-1`.
- Console ≠ UART: UM1472 keeps the ST-Link VCP off USART2, so **all** capture
  is via SWD OpenOCD, single-halt.

## 1. Firmware build + flash

```
cd mcu/fw
touch src/system_init.c      # Makefile does not track header deps — do this
                             # after ANY stm32f4xx.h edit
make                         # → build/mcu_fw.{elf,bin,hex,map}
st-flash write build/mcu_fw.bin 0x08000000
```

Quick smoke test: after flash the live PROD loop prints routed Invoke lines;
with the firmware booting, `openocd -f mcu/openocd/mcu_dump_defense.cfg`
reports `benchmark done flag set` and writes the two .bin files.

## 2. Run the 50-frame defense-pipeline trace

1. Flash, let it boot (trace mode is a compile-time flag; the default PROD build
   runs one continuous live loop).
2. Read back:
   ```
   # from mcu/openocd/mcu_dump_defense.cfg (output into /tmp/opencode/):
   #   g_bench_done (0x200001e0) polled via halt/resume
   #   dump_image stage_cyc.bin 0x20000218 0x3E8   (g_stage_cyc[5][50])
   #   dump_image route.bin     0x200001e4 0x32    (g_route[50])
   openocd -f mcu/openocd/mcu_dump_defense.cfg
   ```
3. Parse + sanity:
   ```
   .venv/bin/python mcu/scripts/parse_defense_capture.py \
       /tmp/opencode/stage_cyc.bin /tmp/opencode/route.bin
   ```

## 3. Cross-checks that must pass before trusting a number

1. **Counter actually runs** (`DWT_CYCCNT != 0` after boot) — see postmortem
   `dwt-base-address-bug`. Verbatim check: DWT↔SysTick agree ±3 cyc over a
   50 ms host-timed window; derived rate ≈ 168.56 MHz.
2. **Route split plausible**: firmware cycles 4 seeded vectors (2 NSL / 2 UNSW),
   so 50 frames must split ~25/25. All-UNSW ⇒ DMA dead (postmortem
   `dma-stream-and-clock-bug`). When the electrical UART is unverified, this
   bias is the only route-sanitizer you have.
3. **Per-stage minima sane**: T_gate ≥ ~12 cyc (12 FMAs); a sub-10-cycle
   "stable" stage means code-sinking (postmortem `dwt-window-code-sinking`),
   not speed.
4. **Warm-up accounted**: first W=10 frames run with a partial window (T_theta
   min 28, mean 284.3); published steady-state = frames 10–49 (theta 297,
   rotate 844, full 22,296.5).

## 4. Published result (2026-09-07, 50 frames, capture
   `mcu/results/fw_ondevice_defense_pipeline_20260907T030335Z.json`)

| stage | mean cyc | ms |
|---|---|---|
| T_sense   | 4,705  | 0.0279 |
| T_theta   | 284    | 0.0017 |
| T_gate    | 117    | 0.0007 |
| T_rotate  | 839    | 0.0050 |
| T_inf     | 16,320 | 0.0968 |
| full path | 22,266 | 0.1321 = 1.32% of the 10 ms SLA |

## 5. Known limits of this measurement

- T_sense measures the ingest **compute** on a simulated-IDLE path, not
  physical RS-232 bytes on PA3 (needs a USB-UART dongle/wire; the ST-Link VCP
  is not wired to USART2 per UM1472). The 115200-baud link budget (~7 ms for a
  76-byte line) is a separate number and not in T_sense.
- FPGA/scope leg of the counter cross-check (GPIO-toggle vs DWT) is documented
  but pending an external logic analyzer.
- Per-source gamma selection (see ADR 0001/0003) is not yet in firmware; the
  measured stage costs do not depend on it.