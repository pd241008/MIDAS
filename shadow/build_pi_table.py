"""
Build the on-device PGD sweep table (defense_success_rate_pi.json) from the raw
check4 battery output (check4_real_data_pi.json).

Mirrors the paper-table format: 9 PGD configs x (fixed, attacker-coupled) bases,
reporting naive/adaptive ASR and defense-success (=1-ASR). Regenerated after the
gamma recalibration (gamma=2.3061) with each raw ASR preserved verbatim.
"""
import sys, os, json

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "results", "pi", "check4_real_data_pi.json")
OUT = os.path.join(REPO, "results", "pi", "defense_success_rate_pi.json")


def main():
    d = json.load(open(SRC))
    gamma = d["gamma"]
    n = d["n_samples"]
    vuln = d["vuln"]       # attacker-coupled basis (gate basis)
    fixed = d["fixed"]     # deployed fixed PCA-2 basis

    def row(cfg, basis, r):
        return {
            "config": cfg,
            "basis": basis,
            "naive_ASR": r["naive"],
            "adaptive_ASR": r["adaptive"],
            "naive_defense_success=1-ASR": round(1 - r["naive"], 4),
            "adaptive_defense_success=1-ASR": round(1 - r["adaptive"], 4),
            "n": n,
        }

    rows = []
    for cfg in fixed:
        rows.append(row(cfg, "fixed", fixed[cfg]))
    for cfg in vuln:
        rows.append(row(cfg, "attacker-coupled", vuln[cfg]))

    table = {
        "table": "On-device adaptive-robustness - PGD sweep (Raspberry Pi 5, aarch64 A76)",
        "note": (f"Attack-success = prediction flipped by the defended classifier. Presented as both raw ASR "
                 f"and defense success = 1 - ASR. 'fixed' = deployed fixed PCA-2 basis; 'attacker-coupled' = "
                 f"basis an adaptive attacker who knows the rotation could exploit (gate basis). Real UNSW-NB15 "
                 f"d=42 W=10, real surrogate, n={n}, gamma={gamma} (recalibrated benign P95; was 1.984/P75, "
                 f"corrected to 2.3061/P95)."),
        "convention_warning": ("These on-device numbers use the real d=42/W=10 surrogate + recalibrated "
                               f"gamma={gamma}. They do NOT match the legacy results/defense_success_rate.json "
                               "columns (synthetic d=10/W=4 SmoothMockModel at gamma=0.292). Do not mix the two "
                               "generations into one table without a consistent convention/model."),
        "columns": ["config", "basis", "naive_ASR", "adaptive_ASR",
                    "naive_defense_success=1-ASR", "adaptive_defense_success=1-ASR", "n"],
        "rows": rows,
        "sources": ["results/pi/check4_real_data_pi.json"],
        "gamma_recalibration_note": ("Previous gamma=1.984 (benign P75) deprecated to "
                                     "defense_success_rate_pi_gamma1984_deprecated.json; this file uses the "
                                     "corrected gamma=2.3061 = benign P95 per the paper rule."),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(table, f, indent=2)
    print(f"wrote {OUT} with {len(rows)} rows, gamma={gamma}")


if __name__ == "__main__":
    main()
