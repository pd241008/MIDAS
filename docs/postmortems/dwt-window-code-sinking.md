# Post-mortem: DWT window collapsed to 2 cycles — GCC sank the work out of the window

- Date: 2026-09-07 (fixed in commit `0bf2e08`)
- Severity: measurement integrity (one stage silently unmeasured)

## Symptom

Stage timing table showed a perfect row — then a physically impossible one:
T_gate = **2 cycles, stdev 0** across 50 frames, while T_sense (109), T_theta,
T_rotate, and T_inf all looked plausible. 12 fused multiply-adds + a LUT lookup
cannot run in 2 cycles.

## Root cause

GCC's `-O2` hoisted/sank the gate computation. The gate was implemented as a
**small, static, pure, single-call-site function** that got inlined, and its
result was hoisted past the *trailing* `volatile` DWT counter read (the value
is only consumed after `a1 = DWT->CYCCNT`), so the measured window contained
only the two reads of the counter. Nothing about the code is "wrong" — the
counter simply bracketed the wrong span of instructions.

## Detection

The `stdev=0 + absurdly small` pattern was the tell (a DWT window that small is
a *fence-around-constant*, not a computation). We confirmed by disassembling
the window span: the `.fma`/`.ldr` gate body was already executed before the
window's first counter read.

## Fix

Mark every stage body `__attribute__((noinline))` (`compute_momentum`,
`penetration_epsilon_windowed`, `rotation_angle`, `gate_logit`, `sigmoid_lut`,
`rotate_givens`): a function **call** is a scheduling barrier GCC will not
speculate across, giving each DWT window a true entry/exit. After the fix,
T_gate measured 117 cyc, stable (the plausible value).

## Prevent

- Never DWT a window that is *pure inline computation* — force a call boundary
  or an explicit compiler barrier (`asm volatile("" ::: "memory")`) inside it.
- Sanity-bound every stage: assert a minimum cycle count against the op budget
  (12 FMAs ≥ ~12 cyc) — seconds to add, catches exactly this class.
- Trust the `stdev=0` pattern as a warning, not a sign of determinism.