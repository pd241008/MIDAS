"""
Part B (MCU tier) step 1: feature-overlap mapping NSL-KDD <-> UNSW-NB15.

Curated semantic mapping between the two schemas, validated against the real
payloads:
  - NSL-KDD  : 41 features + class + difficulty_level  (KDDTrain_raw.csv)
  - UNSW-NB15: 42 features + label                     (Midas d=42 edge set)

Emits:
  - mcu/features/feature_overlap.json  machine-readable mapping + dataset stats
  - mcu/features/feature_overlap.md    paper-facing side-by-side table

Overlap classes:
  EXACT   feature exists in both schemas with the same/synonymous semantics
  ANALOG  approximate semantic counterpart (values not identical; categorical
          encodings unified in step 2, numeric ranges unified in step 3)
  UNSW_ONLY / NSL_ONLY  no counterpart in the other schema
"""
import json, os

from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/nsl-kdd"
NSL_KDD_TRAIN = os.path.join(RAW, "KDDTrain_raw.csv")
UNSW_TRAIN = os.path.join(ROOT, "data", "unsw_nb15_train.csv")

# NSL-KDD standard 41 features, in file order. Files are headerless.
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
NSL_KDD_COLS = NSL_KDD_FEATURES + ["class", "difficulty_level"]

# UNSW-NB15 Midas d=42 feature set (head of data/unsw_nb15_train.csv).
UNSW_FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
UNSW_COLS = UNSW_FEATURES + ["label"]

# Curated semantic mapping. Flow-level features shared by both schemas are the
# fusion anchor; content-based NSL-KDD features and netflow/CT UNSW features
# have no counterpart and are candidates for a down-selected multi-schema set.
MAPPING = [
    (("duration", "dur"),                 "EXACT",  "connection duration (s)"),
    (("protocol_type", "proto"),          "EXACT",  "transport protocol (tcp/udp/icmp/...)"),
    (("service", "service"),              "EXACT",  "application service on dst port"),
    (("src_bytes", "sbytes"),             "EXACT",  "bytes source -> dest"),
    (("dst_bytes", "dbytes"),             "EXACT",  "bytes dest -> source"),
    (("flag", "state"),                   "ANALOG", "TCP connection state (SF/CON... vs S0/REJ...)"),
    (("land", "is_sm_ips_ports"),         "ANALOG", "src==dst ip+port (land / rule-src-dst sm)"),
    (("count", "ct_srv_src"),             "ANALOG", "connections to same dst host/service in window"),
    (("srv_count", "ct_srv_dst"),         "ANALOG", "connections to same service in window"),
    (("dst_host_count", "ct_dst_ltm"),    "ANALOG", "connections to same dst ip in window"),
    (("dst_host_same_src_port_rate", "ct_src_dport_ltm"), "ANALOG",
     "same source port -> dst port family"),
    (("dst_host_same_srv_rate", "ct_srv_src"), "ANALOG",
     "fraction same-service connections (host-based rate vs count)"),
]

NSL_ONLY = [
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_srv_count", "dst_host_diff_srv_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

UNSW_ONLY = [
    "spkts", "dpkts", "rate", "sttl", "dttl", "sload", "dload", "sloss",
    "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb",
    "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_state_ttl", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "is_ftp_login",
]


def main():
    # Sanity: confirm expected columns line up with the real payloads.
    nsl_header = 43
    unsw_header = len(UNSW_COLS)
    print(f"NSL-KDD columns (expected {len(NSL_KDD_COLS)}): {nsl_header}")
    print(f"UNSW-NB15 header (Midas): {unsw_header} cols -> {UNSW_COLS}")

    assert len(NSL_KDD_COLS) == 43, "NSL-KDD raw should be 41 feats + class + difficulty"
    assert nsl_header == len(NSL_KDD_COLS)
    assert unsw_header == len(UNSW_COLS)

    mapped_nsl = {m[0][0] for m in MAPPING}
    mapped_unsw = {m[0][1] for m in MAPPING}
    assert not (mapped_nsl & set(NSL_ONLY)), "NSL feature mapped and listed as only"
    assert not (mapped_unsw & set(UNSW_ONLY)), "UNSW feature mapped and listed as only"
    assert set(NSL_KDD_FEATURES) == mapped_nsl | set(NSL_ONLY), "NSL partition incomplete"
    assert set(UNSW_FEATURES) == mapped_unsw | set(UNSW_ONLY), "UNSW partition incomplete"

    n_share_exact = sum(1 for _, k, _ in MAPPING if k == "EXACT")
    n_share_analog = sum(1 for _, k, _ in MAPPING if k == "ANALOG")
    n_overlap = n_share_exact + n_share_analog

    overlap = {
        "direction": "NSL-KDD <-> UNSW-NB15 (Midas d=42)",
        "curated_features": [
            {
"nsl_kdd": (m[0][0]),
            "unsw": m[0][1],
                "kind": m[1],
                "semantics": m[2],
                "order": i,
            }
            for i, m in enumerate(MAPPING)
        ],
        "overlap_counts": {
            "exact": n_share_exact,
            "analog": n_share_analog,
            "total_overlap": n_overlap,
        },
        "nsl_kdd_only": {"count": len(NSL_ONLY), "features": NSL_ONLY},
        "unsw_only": {"count": len(UNSW_ONLY), "features": UNSW_ONLY},
        "dimensionality": {
            "nsl_kdd": 41,
            "unsw_d42": 42,
            "naive_union": 41 + 42 - n_overlap,
            "overlap_only": n_overlap,
        },
    }

    out_json = os.path.join(HERE, "features", "feature_overlap.json")
    with open(out_json, "w") as f:
        json.dump(overlap, f, indent=2)

    # Paper-friendly markdown side-by-side table.
    lines = [
        "# Feature overlap: NSL-KDD <-> UNSW-NB15 (Part B step 1)",
        "",
        f"NSL-KDD: 41 features + class + difficulty_level | UNSW-NB15 (Midas d=42): 42 features + label",
        f"Anchored overlap: **{n_overlap}** features ({n_share_exact} exact, {n_share_analog} analog).",
        "",
        "| NSL-KDD | UNSW-NB15 | kind | semantics |",
        "|---|---|---|---|",
    ]
    for m in MAPPING:
        lines.append(f"| {m[0][0]} | {m[0][1]} | {m[1]} | {m[2]} |")
    lines += [
        "",
        f"### NSL-KDD only ({len(NSL_ONLY)})",
        "`" + ", ".join(NSL_ONLY) + "`",
        "",
        f"### UNSW-NB15 only ({len(UNSW_ONLY)})",
        "`" + ", ".join(UNSW_ONLY) + "`",
        "",
        "## Dimensionality note (step 4 input)",
        f"- Naive union: {41 + 42 - n_overlap} features (overlap deduped).",
        f"- Overlap-only anchor set: {n_overlap} features -> MCU-scalable.",
        f"- Both schemas normalize to [0,1]; categoricals unified in step 2;",
        f"  joint min-max in step 3; MCU down-select in step 4.",
    ]
    out_md = os.path.join(HERE, "features", "feature_overlap.md")
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nOverlap: {n_overlap} ({n_share_exact} exact, {n_share_analog} analog)")
    print(f"NSL-only: {len(NSL_ONLY)} | UNSW-only: {len(UNSW_ONLY)}")
    print(f"Naive union dim: {overlap['dimensionality']['naive_union']}")
    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()