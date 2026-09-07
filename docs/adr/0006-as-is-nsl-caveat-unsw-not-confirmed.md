# ADR 0006 — Keep the NSL gate caveat and the UNSW "not confirmed" finding as-is

- Status: Accepted
- Date: 2026-09-07

## Context

The adaptive-attacker robustness gate (N=200, then N=800 recheck) produced two
inconvenient, per-source results:

1. **NSL** — the fixed-basis residual at the harshest budget
   (eps=0.2/T=100) is a small positive edge with **Gap/SE ≈ 1.33** (not perfectly
   flat). The verdict is PASS-with-caveat.
2. **UNSW** — the coupled-basis known-vulnerable control collapses to noise at
   high N (max +0.019, Gap/SE ≤ 0.75); the attacker-coupled-basis advantage is
   absent on UNSW data. The verdict is NOT confirmed.

Both are documented in `mcu/features/attacker_gate_report.json`,
`mcu/results/recheck_gate_n800_*.json`, and the README gate table. The question
is whether to change anything (re-tune, re-run, re-explain) before committing
the narrative — this ADR records the decision NOT to.

## Decision

Leave both findings as-is:

- **NSL 1.33σ fixed-basis residual**: keep the PASS-with-caveat verdict. The
  1.33 edge is a single configuration at the harshest budget
  (eps=0.2/T=100), within typical run-to-run noise band observed elsewhere
  (e.g., the Pi reproduction landed +0.035 → +0.00/−0.015 across runs). No
  gate-rule change, no per-config re-tuning, and no re-weighting of the
  conclusion.
- **UNSW gate "NOT confirmed"**: keep as a reported per-source behavioral
  difference, NOT an absence claim for the harness. The NSL control proves the
  harness sensitivity (Gap/SE 9–15 on the coupled basis); UNSW simply does not
  exhibit an attacker-coupled-basis advantage in these data. No alternate
  dataset, no re-labeling, and no softening of the wording.

## Consequences

- Commit the honest framing: NSL is validated, UNSW shows no coupled-basis
  vulnerability in-repo-data. The deployed routing gate + two-specialist
  accuracy (0.8766) stands on the mechanism-backed gate, not the UNSW ADR.
- The per-source asymmetry becomes a reported result — a defensible finding
  for the paper (harness-sensitivity validated by the NSL control), rather
  than a loose end to hide.

## Follow-ups

- If the UNSW "not confirmed" status ever needs revisiting, re-run the gate on
  a held-out UNSW-NB15 split before concluding anything about the harness.
- Nothing else changes for 1.33σ: re-measure only if the harshest-budget
  residual is ever cited as a primary (top-line) number.