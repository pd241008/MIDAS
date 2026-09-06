# Feature overlap: NSL-KDD <-> UNSW-NB15 (Part B step 1)

NSL-KDD: 41 features + class + difficulty_level | UNSW-NB15 (Midas d=42): 42 features + label
Anchored overlap: **12** features (5 exact, 7 analog).

| NSL-KDD | UNSW-NB15 | kind | semantics |
|---|---|---|---|
| duration | dur | EXACT | connection duration (s) |
| protocol_type | proto | EXACT | transport protocol (tcp/udp/icmp/...) |
| service | service | EXACT | application service on dst port |
| src_bytes | sbytes | EXACT | bytes source -> dest |
| dst_bytes | dbytes | EXACT | bytes dest -> source |
| flag | state | ANALOG | TCP connection state (SF/CON... vs S0/REJ...) |
| land | is_sm_ips_ports | ANALOG | src==dst ip+port (land / rule-src-dst sm) |
| count | ct_srv_src | ANALOG | connections to same dst host/service in window |
| srv_count | ct_srv_dst | ANALOG | connections to same service in window |
| dst_host_count | ct_dst_ltm | ANALOG | connections to same dst ip in window |
| dst_host_same_src_port_rate | ct_src_dport_ltm | ANALOG | same source port -> dst port family |
| dst_host_same_srv_rate | ct_srv_src | ANALOG | fraction same-service connections (host-based rate vs count) |

### NSL-KDD only (29)
`wrong_fragment, urgent, hot, num_failed_logins, logged_in, num_compromised, root_shell, su_attempted, num_root, num_file_creations, num_shells, num_access_files, num_outbound_cmds, is_host_login, is_guest_login, serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_srv_count, dst_host_diff_srv_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate`

### UNSW-NB15 only (31)
`spkts, dpkts, rate, sttl, dttl, sload, dload, sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len, ct_state_ttl, ct_dst_sport_ltm, ct_dst_src_ltm, ct_ftp_cmd, ct_flw_http_mthd, ct_src_ltm, is_ftp_login`

## Dimensionality note (step 4 input)
- Naive union: 71 features (overlap deduped).
- Overlap-only anchor set: 12 features -> MCU-scalable.
- Both schemas normalize to [0,1]; categoricals unified in step 2;
  joint min-max in step 3; MCU down-select in step 4.
