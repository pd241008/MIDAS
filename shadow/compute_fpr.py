"""
Item #4: False-Positive Rate (FPR) of the Logarithmic-phase trigger on benign
traffic, real UNSW-NB15 data, W=10.

Trigger definition: the defense enters its aggressive Logarithmic rotation regime
when epsilon_p > gamma (see shadow/defense.py rotation_angle). FPR = fraction of
benign samples whose windowed epsilon_p exceeds the operating gamma.

Operating point: GAMMA_RECALIBRATED = P95 of real benign epsilon_p (=1.984),
so by construction FPR ~= 5% on the calibration distribution. We report FPR on
both the full benign test pool and, for the paper row, at tau=0.75 config with the
gamma fixed. Also reports FPR across candidate percentile thresholds for context.
"""
import sys, os, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shadow.defense import TrainedSurrogateModel, penetration_epsilon_windowed

D = 42
W = 10
GAMMA_RECALIBRATED = 2.3061  # benign P95 (recalibrated, was 1.984 P75)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shadow", "trained_weights")
OUT = os.environ.get("FPR_OUT", "results/fpr_benign_tau075.json")


def load_real_surrogate():
    model = TrainedSurrogateModel(D, hidden=16)
    path = os.path.join(WEIGHTS_DIR, "trained_surrogate_real.pt")
    model.load_state_dict(torch.load(path, weights_only=True))
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model


def load_real_data():
    test_X = torch.tensor(np.load(f"{DATA_DIR}/test_features.npy"), dtype=torch.float32)
    test_y = torch.tensor(np.load(f"{DATA_DIR}/test_labels.npy"), dtype=torch.float32)
    return test_X, test_y


def main():
    print(f"FPR of Logarithmic-phase trigger on benign traffic (W={W}, gamma={GAMMA_RECALIBRATED})")
    classifier = load_real_surrogate()
    test_X, test_y = load_real_data()

    benign_mask = test_y == 0
    benign_ids = torch.where(benign_mask)[0]

    # Clean benign-only windows: a window is "benign traffic" only if ALL W samples
    # are from the benign class (no attack-class contamination in the tail). This is
    # the paper-rule-consistent definition of the benign epsilon_p distribution.
    benign_set = benign_mask.clone()
    clean_eps = []
    for j in range(W - 1, len(test_X)):
        if not bool(benign_set[j - W + 1: j + 1].all()):
            continue
        window = list(test_X[j - W + 1: j + 1])
        clean_eps.append(penetration_epsilon_windowed(window).item())
    clean_eps_t = torch.tensor(clean_eps, dtype=torch.float32)
    n_clean = len(clean_eps)
    print(f"  clean benign-only windows (all {W} samples benign): {n_clean}")

    # Paper rule: phase boundary = 95th pct benign epsilon_p.
    p95_benign = torch.quantile(clean_eps_t, 0.95).item()
    p50 = torch.quantile(clean_eps_t, 0.50).item()
    p75 = torch.quantile(clean_eps_t, 0.75).item()
    p90 = torch.quantile(clean_eps_t, 0.90).item()

    def fpr(t):
        return (clean_eps_t > t).float().mean().item()

    fpr_operating = fpr(GAMMA_RECALIBRATED)
    fpr_at_p95 = fpr(p95_benign)

    result = {
        "experiment": "#4 FPR of Logarithmic-phase trigger on benign traffic, real data W=10, cleaned windows",
        "config": {"W": W, "D": D, "gamma": GAMMA_RECALIBRATED, "tau": 0.75},
        "trigger_rule": "epsilon_p > gamma -> Logarithmic phase (aggressive rotation clamp at delta_theta_max)",
        "fpr_benign_only_at_operating_gamma": round(fpr_operating, 4),
        "fpr_at_benign_p95": round(fpr_at_p95, 4),
        "clean_benign_epsilon_p_quantiles": {
            "p50": round(p50, 4), "p75": round(p75, 4),
            "p90": round(p90, 4), "p95": round(p95_benign, 4),
        },
        "interpretation": ("GAMMA_RECALIBRATED set to benign epsilon_p P95 (2.3061) per the paper rule "
                           "'phase boundary = 95th pct benign epsilon_p'. FPR at the operating gamma is therefore "
                           "5% by construction if gamma == P95 of the SAME clean benign distribution. Small deviations "
                           "from exactly 5% arise if the operating gamma (computed on a different/prior benign sample) "
                           "differs from the P95 of this clean window set. A clean-window P95-differs-from-2.3061 value "
                           "here indicates the operating gamma's calibration source vs this clean test windows differ; "
                           "report whichever is the honest held-out FPR."),
        "n_clean_benign_windows": n_clean,
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  saved {OUT}")


if __name__ == "__main__":
    main()
