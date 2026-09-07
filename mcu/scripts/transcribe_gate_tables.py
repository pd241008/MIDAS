"""
Transcribe the MCU adaptive-attacker gate into Edge's PGD-sweep table format
(mirror of shadow/results/check_trained_surrogate_*.json row treatment).

Source rows:
  - N=200 naive/adaptive ASR + gap: mcu/features/attacker_gate_report.json
  - N=800 naive/adaptive ASR + gap + gap/SE: mcu/features/gate_recheck_report.json

Emits mcu/features/pgd_iters_tables.json (git-hashed via save_results) and prints
the LaTeX-ready per-source table (pgd_iters_naive / pgd_iters_coupled analog).
"""
import json
import os

from save_results import save_results

HERE = os.path.dirname(os.path.abspath(__file__))
MCU_DIR = os.path.dirname(HERE)
FEATURES_DIR = os.path.join(MCU_DIR, "features")

GATE = json.load(open(os.path.join(FEATURES_DIR, "attacker_gate_report.json")))
RECHECK = json.load(open(os.path.join(FEATURES_DIR, "gate_recheck_report.json")))

CONFIGS = ["eps0.05_20", "eps0.1_50", "eps0.2_100"]
EPS = {"eps0.05_20": 0.05, "eps0.1_50": 0.1, "eps0.2_100": 0.2}
T = {"eps0.05_20": 20, "eps0.1_50": 50, "eps0.2_100": 100}


def build_matrix():
    rows = []
    for source in ["nsl", "unsw"]:
        for basis in ["vulnerable_basis", "fixed_basis"]:
            for cfg in CONFIGS:
                g200 = GATE[source][basis][cfg]
                r800 = RECHECK["configs"][source]
                r800 = (r800["coupled"] if basis == "vulnerable_basis"
                        else r800["fixed"])[cfg]
                rows.append({
                    "source": source,
                    "basis": "coupled" if basis == "vulnerable_basis" else "fixed",
                    "vulnerable": basis == "vulnerable_basis",
                    "config": cfg,
                    "epsilon": EPS[cfg],
                    "T": T[cfg],
                    "alpha": 0.01,
                    "n200": {"naive_asr": g200["naive_asr"],
                             "adaptive_asr": g200["adaptive_asr"],
                             "gap": g200["gap"]},
"n800": {"naive_asr": r800["naive_asr"],
                              "adaptive_asr": r800["adaptive_asr"],
                              "gap": r800["gap"],
                              "gap_se": r800["gap_se"]},
                })
    return rows


def main():
    matrix = build_matrix()
    report = {
        "experiment": "MCU gate transcription — Edge pgd_iters table treatment",
        "sources": ["mcu/features/attacker_gate_report.json (N=200)",
                    "mcu/features/gate_recheck_report.json (N=800)"],
        "n800_seed": RECHECK["seed"],
        "gate_rule": RECHECK["gate_rule"],
        "rows": matrix,
    }
    save_results("transcribe_gate_tables.py", config={"n200": 200, "n800": 800},
                 results={"rows": matrix},
                 extra={"gate_rule": report["gate_rule"]})

    # LaTeX-ready per-source tables (pgd_iters_naive / pgd_iters_coupled analog).
    # Per-row verdict labels are deliberately NOT included (matches Edge's own
    # tables: rows carry only gap/gap_se; PASS/FAIL verdicts live in the
    # aggregate recheck summary). A per-row "PASS/FAIL" is direction-agnostic:
    # on the coupled basis finding a gap is the expected result, on the fixed
    # basis it would be the threat outcome -- so only the aggregate verdicts
    # below use the language.
    for src in ["nsl", "unsw"]:
        print(f"\n===== MCU gate: {src.upper()} =====")
        print(f"{'basis':<10s} {'eps':>4s} {'T':>4s} | {'naive':>7s} {'adapt':>7s} {'gap':>7s} | {'naive*':>7s} {'adapt*':>7s} {'gap*':>7s} {'gap/SE*':>8s}")
        print("-" * 96)
        for r in [x for x in matrix if x["source"] == src]:
            print(f"{r['basis']:<10s} {r['epsilon']:>4.2f} {r['T']:>4d} | "
                  f"{r['n200']['naive_asr']:>7.3f} {r['n200']['adaptive_asr']:>7.3f} {r['n200']['gap']:>+7.3f} | "
                  f"{r['n800']['naive_asr']:>7.3f} {r['n800']['adaptive_asr']:>7.3f} {r['n800']['gap']:>+7.3f} {r['n800']['gap_se']:>8.2f}")
    print("\n   * = independent high-N recheck pool, N=800, seed 42")
    print("   Rule: |gap| > 0.02 AND gap/SE > 1.0 (Edge recheck_n200 bar)")
    print("\n   AGGREGATE VERDICTS (reserved for the gate summary, not per-row):")
    print("   NSL:  harness confirmed — PASS (with residual ~1.3 sigma edge on fixed eps0.2_100)")
    print("   UNSW: signal did not survive high-N recheck — NOT confirmed (reportable negative)")


if __name__ == "__main__":
    main()