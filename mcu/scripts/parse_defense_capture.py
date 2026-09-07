"""
Per-stage DWT cycle stats from an OpenOCD SRAM dump of the defense pipeline.

Inputs: two raw little-endian uint32 dumps produced by
mcu/openocd/mcu_dump_defense.cfg (or `dump_image` for the same addresses):

  stage_cyc.bin : g_stage_cyc[5][kNumIters] at 0x20000218  (1000 B)
                  rows: 0=T_sense 1=T_theta 2=T_gate 3=T_rotate 4=T_inf
  route.bin     : g_route[kNumIters]    at 0x200001e4       (50 B, 0/1)

Prints mean/min/max/stdev per stage (marked as warm if kNumIters=50 includes
the W=10 window-fill phase), the NSL/UNSW route split, and the full-path
total + ms + %-of-10 ms SLA (design-rate 168.56 MHz DWT counter).

Usage:
  .venv/bin/python mcu/scripts/parse_defense_capture.py \
      /path/to/stage_cyc.bin /path/to/route.bin [iters]
"""
import struct
import sys

STAGE_NAMES = ["T_sense", "T_theta", "T_gate", "T_rotate", "T_inf"]
RATE_MHZ = 168.56
SLA_MS = 10.0


def load_u32(path, n):
    with open(path, "rb") as f:
        raw = f.read(4 * n)
    if len(raw) < 4 * n:
        raise SystemExit(f"{path}: expected {4 * n} B, got {len(raw)}")
    return struct.unpack("<" + "I" * n, raw[: 4 * n])


def load_u8(path, n):
    with open(path, "rb") as f:
        raw = f.read(n)
    if len(raw) < n:
        raise SystemExit(f"{path}: expected {n} B, got {len(raw)}")
    return struct.unpack("<" + "B" * n, raw[:n])


def stats(v):
    return {
        "mean": round(float(v.mean()), 1),
        "min": int(v.min()),
        "max": int(v.max()),
        "stdev": round(float(v.std()), 2),
    }


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cyc_path, route_path = sys.argv[1], sys.argv[2]
    iters = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    import numpy as np

    cyc = np.array(load_u32(cyc_path, 5 * iters), dtype=np.uint32)
    cyc = cyc.reshape(5, iters).astype(np.float64)
    route = np.array(load_u8(route_path, iters), dtype=np.uint8)

    full = cyc.sum(axis=0)
    print(f"{'stage':9s} {'mean':>8s} {'min':>7s} {'max':>7s} {'sd':>7s}   ms")
    for r, name in enumerate(STAGE_NAMES):
        s = stats(cyc[r])
        print(f"{name:9s} {s['mean']:8.1f} {s['min']:7d} {s['max']:7d} "
              f"{s['stdev']:7.2f}   {s['mean'] / RATE_MHZ / 1e3:.4f}")
    f = stats(full)
    ms = f["mean"] / RATE_MHZ / 1e3
    print(f"{'full path':9s} {f['mean']:8.1f} {f['min']:7d} {f['max']:7d} "
          f"{f['stdev']:7.2f}   {ms:.4f}  ({ms / SLA_MS * 100:.2f}% of {SLA_MS:.0f} ms SLA)")

    n_route = int(route.sum())
    print(f"route        ns={iters - n_route}  unsw={n_route} "
          f"(split {route.tolist().count(0)}/{route.tolist().count(1)})")
    if n_route not in (0, iters):
        for label, mask in (("NSL", route == 0), ("UNSW", route == 1)):
            print(f"T_inf {label:4s} spec  mean={stats(cyc[4][mask])['mean']:.1f}")
    else:
        print("T_inf per-route means: all frames went to a single specialist")


if __name__ == "__main__":
    main()