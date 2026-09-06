"""
Part B (MCU tier) step 2: unified categorical codebooks NSL-KDD <-> UNSW-NB15.

The Edge-tier UNSW pipeline label-encoded protocol/service/state to fixed [0,1]
ordinals (data/norm_params.json). For a defensible fusion, both schemas must
land in ONE shared low-cardinality vocabulary per analog categorical pair, so a
sample from either schema embeds identically.

This is a curated mapping (paper-defensible), derived against the raw value sets
of both schemas. Per pair we define a shared semantic codebook plus a
per-schema value->code mapping table; unmapped/rare values fold to 0 ("other").
Codes are 1..N ordinals; encoder returns code / N in [0,1] for MCU input.

Pairs:
  - protocol_type (NSL)  <-> proto  (UNSW): {tcp,udp,icmp,other}
  - service      (NSL)  <-> service (UNSW): shared well-known + other
  - flag         (NSL)  <-> state   (UNSW): connection-state semantic groups

Emits:
  - mcu/features/categorical_codebooks.json
"""
import json, os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NSL_RAW = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/nsl-kdd"
UNSW_RAW = "/root/workspace/workspace/03-Code/Projects/Legacy/centralized_datasets/raw/unsw_nb15"

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

# ----- Shared semantic codebooks + per-schema value tables (curated) -----
# protocol_type <-> proto. NSL: {tcp,udp,icmp}. UNSW: 134 IANA names folded into
# the four protocol families (non-tcp/udp/icmp families -> other).
PROTO_CODEBOOK = {"other": 0, "tcp": 1, "udp": 2, "icmp": 3}
PROTO_NSL_MAP = {"tcp": "tcp", "udp": "udp", "icmp": "icmp"}
PROTO_UNSW_FAMILY = {
    "tcp": "tcp", "udp": "udp", "icmp": "icmp",
    # representative non-TCP/IPv4-UDP/ICMP IANA names
    "arp": "other", "igmp": "other", "ipv6": "other", "gre": "other",
    "ospf": "other", "esp": "other", "ah": "other", "ipip": "other",
    "sctp": "other", "dccp": "other", "any": "other", "none": "other",
}
# common UNSW protos (all else -> other by default)
PROTO_UNSW_DEFAULT = "other"

# service <-> service. Shared well-known vocabulary (NSL has 70; UNSW has 13).
SERVICE_CODEBOOK = {
    "other": 0, "http": 1, "http_443": 2, "ftp": 3, "ftp_data": 4,
    "smtp": 5, "ssh": 6, "dns": 7, "pop3": 8, "imap4": 9, "telnet": 10,
    "snmp": 11, "ntp": 12, "domain": 13, "https": 14, "irc": 15, "ssl": 16,
    "dhcp": 17, "radius": 18, "mysql": 19, "redis": 20,
}
# NSL service names -> shared code (unmapped -> other)
SERVICE_NSL_MAP = {
    "http": "http", "http_443": "http_443", "http_8001": "http_443",
    "http_2784": "http", "ftp": "ftp", "ftp_data": "ftp_data",
    "smtp": "smtp", "ssh": "ssh", "domain": "domain", "domain_u": "domain",
    "pop_3": "pop3", "pop_2": "pop3", "imap4": "imap4", "telnet": "telnet",
    "snmp": "snmp", "ntp_u": "ntp", "nntp": "ntp", "irc": "irc",
    "https": "https",
}
# UNSW service names -> shared code
SERVICE_UNSW_MAP = {
    "-": "other", "http": "http", "ftp": "ftp", "ftp-data": "ftp_data",
    "smtp": "smtp", "ssh": "ssh", "dns": "dns", "pop3": "pop3",
    "snmp": "snmp", "ssl": "ssl", "dhcp": "dhcp", "radius": "radius",
    "irc": "irc",
}

# flag <-> state. Connection-state semantic groups (curated). NSL flag
# (S0/S1/S2/S3/SF/SH/REJ/RSTO/RSTR/RSTOS0/OTH) and UNSW state
# (ACC/CLO/CON/ECO/FIN/INT/PAR/REQ/RST/URN/no) both describe TCP lifecycle;
# group into: other, SYN-open(handshake), normal(CON/FIN/ACC/CLO), reset,
# inactivity(ECO/URN/REQ/PAR).
STATE_CODEBOOK = {
    "other": 0,
    "syn_open": 1,
    "normal": 2,
    "reset": 3,
    "inactive": 4,
}
STATE_NSL_MAP = {
    "S0": "syn_open", "S1": "syn_open", "S2": "syn_open", "S3": "syn_open",
    "SF": "normal", "SH": "normal",
    "REJ": "reset", "RSTO": "reset", "RSTR": "reset", "RSTOS0": "reset",
    "OTH": "inactive",
}
STATE_UNSW_MAP = {
    "no": "other", "PAR": "syn_open", "REQ": "syn_open",
    "CON": "normal", "FIN": "normal", "ACC": "normal", "CLO": "normal",
    "RST": "reset", "INT": "reset",
    "ECO": "inactive", "URN": "inactive",
}

PAIRS = [
    {
        "name": "protocol_type__proto",
        "nsl_col": "protocol_type", "unsw_col": "proto",
        "codebook": PROTO_CODEBOOK, "nsl_map": PROTO_NSL_MAP, "unsw_map": PROTO_UNSW_FAMILY,
        "unsw_default": PROTO_UNSW_DEFAULT,
    },
    {
        "name": "service__service",
        "nsl_col": "service", "unsw_col": "service",
        "codebook": SERVICE_CODEBOOK, "nsl_map": SERVICE_NSL_MAP,
        "unsw_map": SERVICE_UNSW_MAP,
        "unsw_default": "other",
    },
    {
        "name": "flag__state",
        "nsl_col": "flag", "unsw_col": "state",
        "codebook": STATE_CODEBOOK, "nsl_map": STATE_NSL_MAP,
        "unsw_map": STATE_UNSW_MAP, "unsw_default": "other",
    },
]


def load_nsl(path):
    return pd.read_csv(path, header=None, names=NSL_COLS, low_memory=False)


def load_unsw(path):
    return pd.read_csv(path, header=None, names=UNSW_COLS, low_memory=False)


def show_stats(df, col, label, mapping):
    vals = sorted(df[col].astype(str).unique())
    mapped = {v: mapping.get(v, "other") for v in vals}
    unmapped = [v for v in vals if mapping.get(v, "other") == "other"]
    print(f"    {label}: {len(vals)} values | mapped->other {len(unmapped)}: "
          f"{sorted(unmapped)[:12]}")
    return vals


def main():
    nsl_train = load_nsl(os.path.join(NSL_RAW, "KDDTrain_raw.csv"))
    nsl_test = load_nsl(os.path.join(NSL_RAW, "KDDTest_raw.csv"))
    df_nsl = pd.concat([nsl_train, nsl_test], ignore_index=True)

    unsw_train = load_unsw(os.path.join(UNSW_RAW, "UNSW_NB15_training-set.csv"))
    unsw_test = load_unsw(os.path.join(UNSW_RAW, "UNSW_NB15_testing-set.csv"))
    df_unsw = pd.concat([unsw_train, unsw_test], ignore_index=True)

    print(f"NSL-KDD rows: {len(df_nsl)} | UNSW-NB15 rows: {len(df_unsw)}")

    out = []
    for p in PAIRS:
        print(f"\n[{p['name']}]")
        nsl_vals = show_stats(df_nsl, p["nsl_col"], "NSL", p["nsl_map"])
        unsw_vals = show_stats(df_unsw, p["unsw_col"], "UNSW", p["unsw_map"])
        shared = sorted(set(nsl_vals) & set(unsw_vals))
        if shared:
            print(f"    literal shared values: {shared}")
        print(f"    -> codebook ({len(p['codebook'])} codes): {p['codebook']}")
        out.append({
            "pair": p["name"],
            "nsl_column": p["nsl_col"],
            "unsw_column": p["unsw_col"],
            "codebook": p["codebook"],
            "nsl_value_to_code": {k: int(p["codebook"][p["nsl_map"].get(k, "other")])
                                  for k in nsl_vals},
            "unsw_value_to_code": {k: int(p["codebook"][p["unsw_map"].get(k, "other")])
                                   for k in unsw_vals},
            "nsl_value_count": len(nsl_vals),
            "unsw_value_count": len(unsw_vals),
            "note": ("ordinal code / len(codebook) -> [0,1]; "
                     "unmapped values fold to 'other' (code 0)"),
        })

    out_path = os.path.join(HERE, "features", "categorical_codebooks.json")
    with open(out_path, "w") as f:
        json.dump({"codebooks": out,
                   "policy": "shared low-cardinality semantic vocabulary; "
                             "per-schema value->code tables; other=0"},
                  f, indent=2)
    print(f"\nSaved: {out_path}")

    # Cross-check: every NSL/UNSW raw value is accounted for in a codebook.
    for p in PAIRS:
        cb = p["codebook"]
        for col, mapping, df in ((p["nsl_col"], p["nsl_map"], df_nsl),
                                 (p["unsw_col"], p["unsw_map"], df_unsw)):
            for v in df[col].astype(str).unique():
                code = cb[mapping.get(v, "other")]
                assert 0 <= code < len(cb), f"out-of-range code for {v}"
    print("Cross-check: all raw values encode within codebook range.")


if __name__ == "__main__":
    main()