# Paper Edits — MIDAS-Edge honest framing, gamma reconciliation, and leftovers

> Source of truth for prose changes to the IEEE ICCD 2026 paper draft. All numeric
> values below are grounded in repo artifacts (`results/`). Paste into the paper
> source; do not silently renumber — each section labels what changed and why.
>
> Date: 2026-08-31. Calibration basis: `gamma = 2.3061` (benign epsilon_p P95 on the
> real d=42 UNSW-NB15 surrogate, W=10). n=200 for all attack/defense rows.

---
so we 
## Item 1 — Honest-framing pass (abstract gesture + Limitations + Conclusion)

**Why:** Table 7 shows adversarial training dominates MIDAS-Edge on PGD
(defense-success 0.705 vs MIDAS adaptive 0.060), and input-smoothing dominates on
C&W (1.000 vs 0.520). MIDAS-Edge is *not* the strongest defense on either attack in
isolation; its defensible value is narrower. The prose currently frames the
fixed-basis result as the core success story without this qualification, which a
reviewer's re-run would contradict. Fix: state the actual value proposition
explicitly rather than leaving it to be discovered.

### Abstract (add one sentence, near the robustness claim)
> Whereas conventional defenses require retraining or a fixed threat basis, MIDAS-Edge
> delivers a retraining-free, basis-adaptive defense whose measured advantage is
> specific to rotation-aware adaptive adversaries on the attacker-coupled basis; it
> is not the strongest defense against a plain PGD attack (adversarial training still
> wins on PGD), but it is the only on-device-feasible method that both avoids
> retraining and best preserves robustness when transferred from the surrogate to the
> production classifier.

### Limitations (replace any claim that MIDAS is broadly strongest; add this clause)
> It is important to be precise about what MIDAS-Edge does **not** achieve. Measured
> against the real d=42 surrogate on the same attack suites, MIDAS-Edge does **not**
> dominate all baselines on defense success: adversarial training attains the highest
> defense success on PGD (0.705 vs. 0.060 for MIDAS-Edge adaptive), and input
> smoothing the highest on C&W (1.000 vs. 0.520). The distinct, defensible
> contributions of MIDAS-Edge are therefore (a) **no retraining and no retrained
> weights** — the defense is a deterministic, budget-gated geometric transform
> applied to a stock classifier; (b) a genuine **adaptive advantage on the
> attacker-coupled basis for C&W** (0.520 adaptive vs. 0.260 naive), i.e. it resists
> a rotation-aware adversary better than a naive one; and (c) the **best transfer
> fidelity**: its surrogate->production-FFI adversarial transfer rate (0.270
> adaptive) is the lowest among the methods evaluated, whereas the baselines'
> surrogate-side robustness partially collapses when the same attack batches are
> replayed through the deployed classifier (input-smoothing 1.000, Chen 0.685,
> adv-training 0.380). We report these limits rather than claim blanket superiority.

### Conclusion (align the closing claim)
> MIDAS-Edge is not a universal strongest defense: on a plain PGD attack, adversarial
> training remains superior, and on C&W, input smoothing. Its value is specific and
> measurable: a retraining-free, cheap (sub-millisecond, 50 ms SLA non-binding)
> rotation defense that uniquely couples (i) a naive-vs-adaptive gap on the
> attacker-coupled basis and (ii) the most robust transfer from the surrogate to the
> deployed classifier. We state this narrow value proposition explicitly so that the
> reported results and the prose agree.

---

## Item 2 — Gamma-table reconciliation (definitional, not a bug)

**Why:** Table hyperparams lists a recommended gamma range of **0.3–0.7**, but the
deployed value is **2.3061**. These are **not comparable**: they are different
quantities because the epsilon_p definition changed.

**The reconciliation (use this wording in / near Table hyperparams):**
> The gamma operating range is **definitional** to the penetration statistic. The
> earlier recommended range 0.3–0.7 corresponds to the single-step displacement
> norm, `epsilon_p = ||v_t − v_{t−1}||_2 · max(0, M_t)`, and to the synthetic d=10
> scaling (gamma=0.292). For the **deployed windowed form**,
> `epsilon_p = (1/|W|)·Σ_{i∈W} ||v_i||_2 · max(0, M_i)`, on real d=42 features the
> benign-epsilon_p distribution spans roughly [1.4, 3.2] (p50=1.528, p75=1.983,
> p90=2.270, p95=2.306), and the operating value is set at the benign 95th
> percentile, **gamma_op = 2.3061**. The paper's stated design rule — "phase boundary
> = 95th pct benign epsilon_p" — is what fixes gamma_op; the tabulated range must be
> re-expressed for the current formula as ~[1.4, 3.2], **not** 0.3–0.7. (Runtime
> validation is `gamma ∈ [0,3]`.)

---

## Item 3a — Memory table + config snippet: D = 30 → d = 42

**Why:** The deployed classifier and pipeline run at **d = 42** (README config table,
`configs/edge_config.json`, TFLite input 42). Two leftover spots still say D=30 with
stale byte sizing.

- **Memory table row** — change feature-vector footprint from 1,200 B (d=30) to
  **1,680 B (d=42)**. (Batch/driver memory, ring buffer, and per-window allocations
  scale with d=42 as well.)
- **Config snippet** — change the dimension line from `D: 30` to
  `"D": 42` in the printed hyperparameters block.
- Cross-check everywhere: `D`, `d`, input_width must read 42; hidden=16, W=10,
  gamma=2.3061, sla_budget_ms=50.

---

## Item 3b — Attack-library bracket "[Foolbox / ART — TBD]": fill in truthfully

**Why:** Foolbox/ART were **not** used. Filling this with a name would be false. Use
the truthful replacement describing the actual generators.

> **Attack implementations.** Adversarial examples were generated by the repository's
> own attacker in `shadow/`, implementing standard formulations rather than Foolbox or
> ART. PGD (L∞, many-step sign-gradient, T=50, eps=0.1, alpha=0.01) is applied on the
> attacker-coupled basis (naive vs. adaptive variants); the Carlini–Wagner L2 attack
> (Adam, lr=0.01, iterations=100, 3 increasing-c steps starting at c=1) is applied on
> both attacker-coupled and fixed bases. Model gradients for the adaptive variants are
> taken **through** the rotation defense (`midas_defense_forward_vulnerable`), i.e.
> they are white-box and defense-aware. All batches are exported to JSON and replayed
> through the production TFLite FFI classifier for transfer verification, matching the
> protocol in #7.

---

## Item 3c — Section VII-C: Pi hardware-gate reproduction paragraph (add)

**Why:** Previously reported only in chat; not yet in the .tex. Backed by
`results/table_ondevice_gate.json` and `results/pi/check4_real_data_pi.json`.

> **On-device reproduction of the robustness gate.** We reproduced the dev-box gate
> (Check #4, real UNSW-NB15, d=42, W=10, n=200) on the Raspberry Pi 5 Cortex-A76 via
> the on-device PGD harness. The mean `|naive − adaptive|` ASR gap on the
> **attacker-coupled** basis is 0.388 on x86_64 vs. 0.396 on the Pi — statistically
> indistinguishable. On the **fixed PCA-2** basis the gap collapses to 0.012 (x86_64)
> and 0.007 (Pi), i.e. no adaptive advantage, as required by the gate rule. The gate
> (`|gap| > 0.02` on the coupled basis, `< 0.02` on fixed) passes on both platforms,
> confirming the central adaptive-robustness claim transfers intact from the dev box
> to real A76 silicon. The two x86 fixed-basis configs that marginally exceeded the
> 0.02 threshold (eps=0.1/T=50 +0.035; eps=0.2/T=100 +0.045) land at +0.00 and −0.015
> on the Pi, indicating run-to-run noise rather than signal.

---

## Key numbers for the prose (all from results/)

| item | value | source |
|---|---|---|
| **MIDAS basis convention** | MIDAS-Edge rows below use the **attacker-coupled basis** (user decision; matches gate basis): the adaptive attacker couples to the rotation basis MIDAS is running, the worst/threat-model case. The deployed fixed PCA-2 basis gives different values (e.g. PGD eps0.1 T50 adaptive DS = 0.075 fixed vs 0.060 coupled). Never mix coupled and fixed into one MIDAS column without labeling. | `results/pi/defense_success_rate_pi.json`, `results/pi/cw_l2_pi_final.json` |
| gamma_op | 2.3061 (benign P95) | `results/fpr_benign_tau075.json` |
| benign eps_p quantiles (d=42, W=10) | p50 1.528, p75 1.983, p90 2.270, p95 2.306 | same |
| FPR at gamma_op | 5.01% | same |
| defense-success PGD eps0.1 T50 (attacker-coupled) | undef 0.145 / adv-train 0.705 / smooth 0.000 / Chen 0.485 / MIDAS adaptive 0.060 | `results/pi/defense_success_vs_baselines.json` |
| (fixed-basis PGD eps0.1 T50 for the deployed fallback) | MIDAS adaptive 0.075 (fixed PCA-2) vs 0.060 (coupled) — label the basis when citing | `results/pi/defense_success_rate_pi.json` |
| **full on-device PGD sweep** | 18 rows, both bases, per-(eps,T) ASR+DS transcribed | `results/pi/pgd_sweep_transcription.md` |
| defense-success C&W (attacker-coupled) | undef 0.945 / adv-train 0.840 / smooth 1.000 / Chen 0.945 / MIDAS adaptive 0.520 | same |
| FFI transfer (adaptive ASR) | MIDAS 0.270 / smooth 1.000 / Chen 0.685 / adv-train 0.380 | `results/pi/baseline_transfer_ffi_*.json`, `results/pi/transfer_ffi.json` |
| gate gap coupled | x86 0.388 / Pi 0.396 | `results/table_ondevice_gate.json` |
| on-device SLA #5 | 50k queries, 0 violations, p99 0.0021 ms, CPU peak 99.8% | `results/pi/sla_load_test.json` |
| production-FFI deployment (consolidated) | latency, throughput, FPR against the real TFLite FFI classifier — see file | `results/pi/prod_ffi_deployment.json` |
| phase-split latency (real TFLite FFI model) | x86 dev: Arch p50 0.0020 / p99 0.0124 ms; Log p50 0.0021 / p99 0.0070 ms. **Pi 5 aarch64 (on-device, confirmed): Arch & Log p50/p95/p99 both 0.0014 ms** (max 0.054 / 0.015). #6 prototype's `0.1029 ms` was `MockModel` w/ artificial sleep — superseded. Well under the 50 ms SLA | `results/pi/phase_split_latency_real_ffi_pi.json` + x86 run `target/release/phase_latency` + `EDGE_MODEL_PATH=models/classifier_float32.tflite` |

---

## Item — MCU-tier baseline battery vs deployed INT8 specialists (add)

**Why:** The Edge-tier baselines (above) were run against the float d=42 surrogate;
this is the on-device/deployed-column companion: the same baseline set re-run against
the production per-source full-INT8 specialists (NSL-NetFlow gamma 0.9700 / UNSW-NB15
gamma 0.6127, fake-quant ST, fixed PCA-2 basis). Script
`mcu/scripts/run_mcu_tier_baselines.py`, data `mcu/features/mcu_tier_baselines.json`.

> **MCU-tier baselines (per-source INT8 specialists, n=100, defense-success = 1−ASR).**
> On **PGD (eps=0.1, T=50)** the deployed int8 specialists order the on-device-feasible
> methods as: Chen query-blinding strongest (NSL 0.44 / UNSW 0.30, counting its own
> rejection as hold), then undefended (0.08 / 0.13), MIDAS rotation / DACM adaptive
> (0.06 / 0.04; DACM naive UNSW 0.10), adversarial training ≈ 0, input-smoothing
> adaptive = 0. The MIDAS adaptive value mirrors the Edge coupled-basis 0.060 — the
> rotation defense is cheaply defeated by an adaptive PGD attacker on-device as well.
> On **C&W-L2 (iters=100)** every row returns defense-success 1.000 (0 flips) — the
> deployed int8 specialist is invariant to C&W-L2, matching the Edge float undefended
> 0.945 (its C&W is already nearly impotent). The MCU C&W row is a degenerate
> equal-1.0 insensitivity of the quantized specialist, not a discriminative baseline
> comparison; report PGD as the on-device discriminator and note C&W as insensitive
> on the deployed model.


---

## Item — Ring-buffer / window-size W ablation of the gate (add)

**Why:** The gate verdicts are claimed on W=10 only; this shows they hold across the
ring-buffer size (MCU SRAM knob). Script `mcu/scripts/w_ablation.py`, data
`mcu/features/w_ablation_report.json` + `mcu/results/w_ablation_*.json` (N=200/source,
same sample set for every W, both basis configurations).

> **Window-size invariance.** We swept the defense ring-buffer size
> W ∈ {2, 3, 4, 6, 8, 10} (W−1 prior rows + current, fixed 200-sample benign set)
> and re-ran the adaptive-attacker gate on the deployed INT8 specialists. On NSL the
> attacker-coupled-basis (known-vulnerable control) naive–adaptive gap is +0.13..+0.41
> at every W — down to the minimal W=2 buffer — and the deployed fixed-basis gap stays
> flat; the gate is window-size invariant, so the adaptive-edge finding does not
> depend on the on-device trajectory memory. On UNSW the verdict oscillates ±0.05
> around the 0.02 rule at N=200 (pass W=2,3,4,10; fail W=6,8), which equals the
> sampling noise already established by the N=800 collapse — the NOT-confirmed
> per-source finding is likewise not a W artifact. Ring-buffer SRAM cost scales as
> (W−1)·d·4 B (at W=10: 9·12·4 = 432 B), so W=2..4 is the memory-cheap operating
> point with no measurable gate consequence.

