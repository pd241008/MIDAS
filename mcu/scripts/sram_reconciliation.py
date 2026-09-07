"""
Table sram + flash reconciliation for the MCU tier.

Reads the ACTUAL linker map/ELF (mcu/fw/build/mcu_fw.elf) and attributes every
SRAM byte and flash byte to a line item. Distinguishes:

  (a) ON-DEVICE footprint  = what is really linked today (17.1 KB SRAM).
  (b) MIDAS ALGORITHMIC tier = the paper's "~14 KB / 7.3 %" abstract number,
      which at the reference d=42 is C_base + C_rotated = 2*d^2*4 = 14,112 B
      (7.35% of 192 KB). At the deployed d=12 this collides to 2*576 = 1,152 B.

The two are DIFFERENT quantities and must not be conflated in the paper.
"""
import json
import os
import subprocess

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FW_DIR = os.path.join(MCU_DIR, "fw")
ELF = os.path.join(FW_DIR, "build", "mcu_fw.elf")

D = 12
WB = 4  # float32 bytes
ARENA = 16 * 1024


def nm():
    out = subprocess.check_output(["arm-none-eabi-nm", "-S", "-n", ELF],
                                  stderr=subprocess.DEVNULL).decode()
    ret = {"bss": {}, "data": {}}
    for line in out.splitlines():
        p = line.split()
        if len(p) < 3:
            continue
        try:
            sz, typ, name = int(p[1], 16), p[2], p[3]
        except (ValueError, IndexError):
            sz, typ, name = int("0", 16), p[1], p[2]
        if typ in ("b", "B"):
            ret["bss"][name] = ret["bss"].get(name, 0) + sz
        elif typ in ("d", "D"):
            ret["data"][name] = sz
    return ret


def objdump_flash():
    out = subprocess.check_output(["arm-none-eabi-objdump", "-h", ELF],
                                  stderr=subprocess.DEVNULL).decode()
    total = 0
    per = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) > 6 and p[1] in (".text", ".rodata", ".data"):
            try:
                s = int(p[2], 16)
            except ValueError:
                continue
            per[p[1]] = per.get(p[1], 0) + s
            total += s
    return per, total


def categorize(syms):
    cat = {}
    for name, sz in syms["bss"].items():
        if name.startswith("_ZL7g_arena"):
            cat.setdefault("scratch (TFLM tensor arena)", 0)
            cat["scratch (TFLM tensor arena)"] += sz
        elif "interp" in name or "resolver" in name:
            cat.setdefault("middleware (MicroInterpreter + resolver)", 0)
            cat["middleware (MicroInterpreter + resolver)"] += sz
        elif "std" in name or "reent" in name or "_reent" in name or name == "__sf":
            cat.setdefault("newlib/stdio (FILE, reent)", 0)
            cat["newlib/stdio (FILE, reent)"] += sz
        elif "g_dma_rx" in name:
            cat.setdefault("live ingest: DMA byte ring", 0)
            cat["live ingest: DMA byte ring"] += sz
        elif any(k in name for k in ("g_vec", "g_num", "g_input")):
            cat.setdefault("live ingest: parser + input", 0)
            cat["live ingest: parser + input"] += sz
        else:
            cat.setdefault("state/pointers/other", 0)
            cat["state/pointers/other"] += sz
    return cat


def main():
    assert os.path.exists(ELF), f"{ELF} missing - build mcu/fw first"
    syms = nm()
    cats = categorize(syms)
    data_total = sum(syms["data"].values())
    bss_total = sum(syms["bss"].values())
    sram_total = bss_total + data_total
    flash_sections, flash_total = objdump_flash()

    # MIDAS algorithmic tier at d=12 (paper Table sram categories; NOT yet all linked)
    midas_tier = {
        "ring_buf (W=10 x d=12 x f32)": D * 10 * WB,
        "C_base (d^2 x f32)": D * D * WB,
        "C_rotated (d^2 x f32)": D * D * WB,
        "state (window cnt, gamma, lambda, k)": 48,
    }
    midas_total = sum(midas_tier.values())

    # Reference: paper's ~14KB / 7.3 % number as 2 x d^2 x f32 at d=42
    ref42 = {"C_base+C_rotated @d=42": 2 * 42 * 42 * 4,
             "7.3 % of 192 KB": round(192 * 1024 * 0.073)}
    report = {
        "elf": ELF,
        "on_device_SRAM": {k: v for k, v in sorted(cats.items(), key=lambda x: -x[1])},
        "on_device_SRAM_total_bytes": sram_total,
        "flash_sections": flash_sections,
        "flash_total_bytes": flash_total,
        "midas_algorithmic_tier_d12": midas_tier,
        "midas_algorithmic_tier_d12_total": midas_total,
        "paper_ref_14112B_reconstructed": ref42,
        "interpretation":
            "paper's ~14KB/7.3% == 2 x d^2 x f32 rotation-matrix pair at reference d=42 "
            "(14,112 B = 7.35%). Device SRAM (17.1 KB incl 16 KB TFLM arena) is a DIFFERENT "
            "quantity. At d=12 the matrix pair collapses to 1,152 B; planned MIDAS tier "
            "(ring+c matrices+state) = ~1.7 KB incremental over the TFLM baseline.",
        "note_window":
            "g_win (480 B W-window) is current-py optimized out as a dead store (only "
            "written, never read until the on-device defense lands); it reappears when the "
            "defense consumes it.",
    }
    out = os.path.join(MCU_DIR, "features", "sram_flash_reconciliation.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("== ON-DEVICE SRAM (linked today) ==")
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {v:6d} B   {k}")
    print(f"  {data_total:6d} B   .data (initialized)")
    print(f"  {'-'*50}\n  {sram_total:6d} B  TOTAL  = {sram_total/1024:.1f} KB = {sram_total/(192*1024)*100:.1f}% of 192KB")
    print("\n== FLASH (linked today) ==")
    for k, v in flash_sections.items():
        print(f"  {v:6d} B   {k}")
    print(f"  {flash_total:6d} B  TOTAL = {flash_total/1024:.1f} KB = {flash_total/(1024*1024)*100:.1f}% of 1MB")
    print("\n== MIDAS algorithmic tier at d=12 (paper Table sram categories) ==")
    for k, v in midas_tier.items():
        print(f"  {v:6d} B   {k}")
    print(f"  {midas_total:6d} B  TOTAL = {midas_total/1024:.1f} KB")
    print(f"\n== paper '~14KB / 7.3%' reconstruction ==")
    for k, v in ref42.items():
        print(f"  {v:6d} B   {k}")
    print(f"\nsaved {out}")
    save_results("sram_reconciliation.py",
                 config={"d": D},
                 results={"on_device_sram": report["on_device_SRAM"],
                          "on_device_sram_total": sram_total,
                          "flash_total": flash_total,
                          "midas_tier_d12": midas_tier,
                          "paper_ref_d42": ref42},
                 extra={"interpretation": report["interpretation"]})


if __name__ == "__main__":
    main()