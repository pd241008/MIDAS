"""
Part B (MCU tier) step 3: unified joint min-max scaling of overlap features.

The fused anchor (step 1) is the 12 overlapping features. Both schemas must
normalize into the SAME [0,1] space so identical network behavior produces
identical vectors. This script:

  - Rebuilds each schema's raw rows into the 12-feature overlap vector using a
    common ordering (the COLS_42 layout matches Midas d=42 ordering for direct
    reuse of downstream Edge code).
  - For the 3 categorical overlap features, encodes via the shared codebooks
    (step 2); the other 9 overlap features are numeric with schema-specific
    semantics mapped by name.
  - Fits min/max per feature JOINTLY (concatenated both schemas) and emits the
    unified scaler as JSON + npy (norm_params.json-compatible layout).

Dimensionality note: overlap-only anchor is d=12. The MCU budget check (192 KB)
is reported per candidate d (12 overlap vs larger unions); down-select detail is
step 4.

Emits:
  - mcu/features/unified_scaler.json     mins/maxs/feature_cols (12)
  - mcu/features/overlap_stats.json      per-schema count + coverage stats
"""
import json, os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NSL_RAW = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/nsl-kdd"
UNSW_RAW = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/unsw_nb15"
CODEBOOKS_PATH = os.path.join(HERE, "features", "categorical_codebooks.json")

# The 12-feature overlap anchor in Midas-d42-column order (dur, proto, service,
# state, sbytes, dbytes, land/is_sm_ips_ports, count, srv_count,
# dst_host_count/ct_dst_ltm, dst_host_same_src_port_rate/ct_src_dport_ltm,
# dst_host_same_srv_rate/ct_srv_src). Layout must match mapping table.
"""
overlap anchors (12):
  0 dur                 duration
  1 proto               protocol_type
  2 service             service
  3 state               flag
  4 sbytes              src_bytes
  5 dbytes              dst_bytes
  6 is_sm_ips_ports     land
  7 ct_srv_src          count
  8 ct_srv_dst          srv_count
  9 ct_dst_ltm          dst_host_count
 10 ct_src_dport_ltm    dst_host_same_src_port_rate
 11 ct_srv_src_same     dst_host_same_srv_rate   (same UNSW col as #7; keep
                        distinct semantic slot via rate)
"""
NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
NSL_COLS = NSL_KDD_FEATURES + ["class", "difficulty_level"]

UNSW_MAPPING = {
    0: "id", 1: "dur", 2: "proto", 3: "service", 4: "state", 5: "spkts",
    6: "dpkts", 7: "sbytes", 8: "dbytes", 9: "rate", 10: "sttl", 11: "dttl",
    12: "sload", 13: "dload", 14: "sloss", 15: "dloss", 16: "sinpkt",
    17: "dinpkt", 18: "sjit", 19: "djit", 20: "swin", 21: "stcpb",
    22: "dtcpb", 23: "dwin", 24: "tcprtt", 25: "synack", 26: "ackdat",
    27: "smean", 28: "dmean", 29: "trans_depth", 30: "response_body_len",
    31: "ct_srv_src", 32: "ct_state_ttl", 33: "ct_dst_ltm",
    34: "ct_src_dport_ltm", 35: "ct_dst_sport_ltm", 36: "ct_dst_src_ltm",
    37: "is_ftp_login", 38: "ct_ftp_cmd", 39: "ct_flw_http_mthd",
    40: "ct_src_ltm", 41: "ct_srv_dst", 42: "is_sm_ips_ports",
    43: "attack_cat", 44: "label",
}
UNSW_COLS = list(UNSW_MAPPING.values())

# Overlap layout: (unsw_col, nsl_col) per output slot (None = no counterpart).
OVERLAP_LAYOUT = [
    ("dur", "duration"),
    ("proto", "protocol_type"),
    ("service", "service"),
    ("state", "flag"),
    ("sbytes", "src_bytes"),
    ("dbytes", "dst_bytes"),
    ("is_sm_ips_ports", "land"),
    ("ct_srv_src", "count"),
    ("ct_srv_dst", "srv_count"),
    ("ct_dst_ltm", "dst_host_count"),
    ("ct_src_dport_ltm", "dst_host_same_src_port_rate"),
    ("ct_srv_src", "dst_host_same_srv_rate"),  # semantic #11 (rate over count)
]
FEATURE_COLS = [f"{u}<-{n}" for u, n in OVERLAP_LAYOUT]


def load_codebooks():
    with open(CODEBOOKS_PATH) as f:
        return json.load(f)["codebooks"]


def load_nsl(path):
    return pd.read_csv(path, header=None, names=NSL_COLS, low_memory=False)


def load_unsw(path):
    return pd.read_csv(path, header=None, names=UNSW_COLS, low_memory=False)


def encode(codebook_entry, value):
    code = codebook_entry.get(str(value), 0)
    return float(code / max(len(codebook_entry), 1))


def build_features(df, is_nsl, codebooks):
    """Return (feat_matrix[n,12], labels, provenance)."""
    cb = {c["pair"]: c for c in codebooks}
    n = len(df)
    X = np.zeros((n, 12))
    labels = np.zeros(n, dtype=np.int8)
    for i, (unsw_col, nsl_col) in enumerate(OVERLAP_LAYOUT):
        if is_nsl:
            if nsl_col in ("protocol_type", "service", "flag"):
                pair = {
                    "protocol_type": "protocol_type__proto",
                    "service": "service__service",
                    "flag": "flag__state",
                }[nsl_col]
                entry = cb[pair]["nsl_value_to_code"]
                X[:, i] = [encode(entry, v) for v in df[nsl_col]]
            else:
                num = pd.to_numeric(df[nsl_col], errors="coerce").fillna(0)
                X[:, i] = num.to_numpy()
        else:
            if unsw_col in ("proto", "service", "state"):
                pair = {
                    "proto": "protocol_type__proto",
                    "service": "service__service",
                    "state": "flag__state",
                }[unsw_col]
                entry = cb[pair]["unsw_value_to_code"]
                X[:, i] = [encode(entry, v) for v in df[unsw_col]]
            else:
                num = pd.to_numeric(df[unsw_col], errors="coerce").fillna(0)
                X[:, i] = num.to_numpy()
    target = "class" if is_nsl else "label"
    attack = df[target].astype(str)
    if is_nsl:
        labels = np.where(attack == "normal", 0, 1).astype(np.int8)
    else:
        # raw UNSW label column: '1'=attack (45533 in train), '0'=Normal (37000)
        labels = np.where(attack == "0", 0, 1).astype(np.int8)
    return X, labels


def main():
    codebooks = load_codebooks()

    nsl_train = load_nsl(os.path.join(NSL_RAW, "KDDTrain_raw.csv"))
    nsl_test = load_nsl(os.path.join(NSL_RAW, "KDDTest_raw.csv"))
    df_nsl = pd.concat([nsl_train, nsl_test], ignore_index=True)

    unsw_train = load_unsw(os.path.join(UNSW_RAW, "UNSW_NB15_training-set.csv"))
    unsw_test = load_unsw(os.path.join(UNSW_RAW, "UNSW_NB15_testing-set.csv"))
    df_unsw = pd.concat([unsw_train, unsw_test], ignore_index=True)

    X_nsl, y_nsl = build_features(df_nsl, True, codebooks)
    X_unsw, y_unsw = build_features(df_unsw, False, codebooks)
    print(f"NSL overlap matrix: {X_nsl.shape} | positives: {int(y_nsl.sum())}")
    print(f"UNSW overlap matrix: {X_unsw.shape} | positives: {int(y_unsw.sum())}")

    # Joint min-max over concatenated raw (pre-normalization) values.
    X_all = np.concatenate([X_nsl, X_unsw], axis=0)
    mins = X_all.min(axis=0)
    maxs = X_all.max(axis=0)
    mask = (maxs - mins) > 0
    X_all_n = np.zeros_like(X_all)
    X_all_n[:, mask] = (X_all[:, mask] - mins[mask]) / (maxs[mask] - mins[mask])
    X_all_n[:, ~mask] = 0.0

    scaler = {
        "feature_cols": FEATURE_COLS,
        "mins": mins.tolist(),
        "maxs": maxs.tolist(),
        "d": 12,
        "overlap_anchor": FEATURE_COLS,
        "note": "joint fit over NSL-KDD+UNSW-NB15 raw overlap feature values",
    }
    with open(os.path.join(HERE, "features", "unified_scaler.json"), "w") as f:
        json.dump(scaler, f, indent=2)
    with open(os.path.join(HERE, "features", "overlap_stats.json"), "w") as f:
        json.dump({
            "nsl_rows": int(len(df_nsl)), "unsw_rows": int(len(df_unsw)),
            "nsl_positive": int(y_nsl.sum()), "unsw_positive": int(y_unsw.sum()),
            "nsl_negative": int(len(y_nsl) - y_nsl.sum()),
            "unsw_negative": int(len(y_unsw) - y_unsw.sum()),
            "overlap_d": 12,
            "minmax_per_feature": mins.tolist(),
        }, f, indent=2)

    np.save(os.path.join(HERE, "features", "unified_mins.npy"), mins.astype(np.float32))
    np.save(os.path.join(HERE, "features", "unified_ranges.npy"), (maxs - mins).astype(np.float32))
    np.save(os.path.join(HERE, "features", "unified_features_all.npy"), X_all.astype(np.float32))
    np.save(os.path.join(HERE, "features", "unified_features_norm.npy"), X_all_n.astype(np.float32))
    y_all = np.concatenate([y_nsl, y_unsw])
    np.save(os.path.join(HERE, "features", "unified_labels.npy"), y_all)
    np.save(os.path.join(HERE, "features", "unified_provenance.npy"),
            np.concatenate([np.zeros(len(y_nsl), np.int8), np.ones(len(y_unsw), np.int8)]))

    # MCU budget report (192 KB SRAM). Ring buffer = W*4*d bytes (W=10).
    W = 10
    budget = 192 * 1024
    for d in (12, 24, 42):
        ring = W * 4 * d
        print(f"  d={d:3d}: ring buffer = {ring:>7d} B ({ring/budget:.2%} of "
              f"{budget} B SRAM)  -> {'PASS' if ring < budget//4 else 'OK'}")
    print(f"\nSaved unified_scaler.json, overlap_stats.json, unified_*.npy")


if __name__ == "__main__":
    main()