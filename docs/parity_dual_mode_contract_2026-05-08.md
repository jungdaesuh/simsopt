# Parity Dual-Mode Contract — Plan (revised 3b semantics)

- Date: 2026-05-08 (revised same day after Codex adversarial review)
- Branch: `gpu-purity-stage2-20260405`
- Status: planning artifact (not started)
- Companion docs:
  - `docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`
    (Phase 4 — strict byte-identity work; this doc adds **diagnostic
    granularity** on top of Phase 4, it does NOT loosen Phase 4's
    production tolerance contract)
  - `benchmarks/validation_ladder_contract.py` (existing
    `PARITY_LADDER_TOLERANCES` SSOT; this plan adds a separate reporting
    context SSOT rather than overloading tolerance lanes)
  - `src/simsopt/backend/runtime.py` (existing `VALID_BACKEND_MODES` and
    `_MODE_POLICY_DEFAULTS` SSOT)

## 0. Revision Note

The original v1 of this plan (drafted earlier on 2026-05-08) proposed
moving the strict gate at `benchmarks/single_stage_init_parity.py:1905`
from "release blocker" to "diagnostic acceptance test" and adding a
production tolerance gate at `rtol=1e-12`. Codex's adversarial review
flagged three real bugs in v1:

1. The proposed `rtol=1e-12` tolerance gate did not pass the cited
   failing artifact (`pre_newton_state.max_abs_diff = 4.52e-9` ≫
   `2e-12 = rtol·|reference|`). The doc's §4 Slice B test plan and §7
   risk register contradicted each other on this.
2. The "new" tolerance lane duplicated existing predicate
   `_IOTA_DECOMPOSITION_DIAGNOSTIC_RTOL = 1e-12` /
   `_IOTA_DECOMPOSITION_DIAGNOSTIC_ATOL = 1e-13` at
   `benchmarks/single_stage_init_parity.py:136-137` and `:1706-1715`.
   Renaming the predicate would not have meaningfully relaxed
   anything.
3. Cross-doc contract reversal: Phase 4 plan §2 "Hard Constraints"
   says "Do not loosen tolerances," and v1 of this plan loosened them
   without an explicit amendment.

This v2 revision resolves all three bugs by **walking back the
production-loosening claim**:

- The strict gate at `single_stage_init_parity.py:1905` **stays as a
  release blocker**.
- Phase 4 stays a **release-blocker workstream** — production CI
  cannot turn green until the strict `pre_newton_*` gate goes green.
- Per-layer empirically-measured thresholds (the user's preference 2c)
  drive **reporting and severity classification**, not gating. They
  augment the divergent-layer census messages with "this drift is N×
  the empirical baseline" context; they do not admit failures.
- The new modes (`jax_cpu_fast`, additional speed lanes) are
  **speed-opt-out runtime modes** for experimental work that
  explicitly accepts non-byte-identical output. They do not introduce
  a production tolerance contract.
- Phase 4 plan §2's "Do not loosen tolerances" rule is **untouched**.

The dual mode is therefore **mode-of-execution dual** (production runs
matmul/jacfwd hot paths in `jax_cpu_fast`; verification runs cpu_ordered
twins in `jax_cpu_parity`) but **single contract** at the gate (strict
byte identity, both lanes).

## 1. Why This Plan Still Exists (Despite the Walk-Back)

Even without changing the production gate, there is real value in:

1. **Finer-grained divergence reporting.** Today the gate emits
   "boozer_solve.pre_newton_state diverged at max_abs_diff=4.52e-9" —
   that's the failure. It does not say *what* the empirical baseline
   drift is on well-behaved (passing) artifacts, so engineers can't
   immediately tell whether 4.52e-9 is "5× baseline drift, marginal
   regression" or "10⁶× baseline, hard physics break." Adding per-layer
   empirical thresholds to the report (not the gate) gives that
   context.

2. **Boundary-input byte tests** (P4.5 / P4.5b). Pinning the inputs
   that go INTO the residual kernel and asserting byte identity at
   that boundary is a finer-grained diagnostic than the end-to-end
   strict gate. It isolates "is the residual assembly drifting?" from
   "are the upstream surface/BS twins drifting?" Useful for
   triangulating future regressions, even though it doesn't change
   what production CI gates on.

3. **Mode scaffolding for speed-opt-out experiments.** A researcher
   running coil-design sweeps may want to pay the speed cost of
   `jax_cpu_parity` (cpu_ordered twins, ~5-20× slower) only on
   regression-sensitive runs, and use a faster `jax_cpu_fast` lane
   (matmul/jacfwd hot path) for non-publication-grade exploration.
   The fast lane is explicitly labeled as opt-out from byte-identity
   to C++ — it's not a tolerance-mode promise; it's a runtime-speed
   promise without a parity claim.

4. **Cleaner internal contracts.** Today the relationship between
   `parity_policy="cpu_ordered"`, `is_parity_mode()`, and the gate
   semantics is implicit. Documenting it as the dual-mode contract
   (production ≠ diagnostic at the **runtime** layer; production ==
   diagnostic at the **gate** layer) makes future changes easier to
   reason about.

What this plan does **not** claim:

- It does **not** unblock production research before Phase 4 closes.
  If you need that, the work is Phase 4, not this plan.
- It does **not** loosen any production tolerance.
- It does **not** introduce a "production gate" separate from the
  current strict gate.
- It does **not** invalidate Phase 4 plan §2 "Hard Constraints."

## 2. Target Architecture

### 2.1 Mode matrix

```
                    ┌────────────────┬────────────────┬─────────────────┐
                    │  native_cpu    │  *_fast        │  *_parity       │
                    │  (C++ ref)     │  (speed-opt-   │  (verification) │
                    │                │  out)          │                 │
├───────────────────┼────────────────┼────────────────┼─────────────────┤
│ JAX involved?     │ no             │ yes            │ yes             │
│ Hot-path kernels  │ C++ direct     │ matmul/jacfwd/ │ cpu_ordered     │
│                   │                │ einsum         │ twins           │
│ Speed             │ reference      │ fast           │ slow (5-20×)    │
│ Byte-identical    │ self-           │ explicitly NO  │ yes (Phase 4   │
│ to C++?           │ consistent      │ (different     │ closes the FMA  │
│                   │ oracle          │ reduction      │ residual)       │
│                   │                │ topology)      │                 │
│ Subject to        │ N/A (ref)      │ yes            │ yes             │
│ strict gate?      │                │                │                 │
│ Use for           │ oracle, paper  │ speed-          │ release CI,     │
│                   │ baselines      │ experiments    │ verification    │
│                   │                │ that accept    │ that must clear │
│                   │                │ no parity      │ strict gate     │
└───────────────────┴────────────────┴────────────────┴─────────────────┘
```

Key point: **all three modes share a single gate contract** (strict
byte identity at `pre_newton_*`). The `*_fast` lane will simply *fail*
the gate by construction — that's the point: it's an opt-out for
exploration runs that don't need to clear release CI.

### 2.2 Concrete `VALID_BACKEND_MODES` extension

Current modes (verified at `runtime.py:100-106`):

```python
VALID_BACKEND_MODES = (
    "native_cpu",
    "jax_cpu_parity",
    "jax_gpu_parity",
    "jax_gpu_fast",
    "jax_metal_smoke",
)
```

Add **`jax_cpu_fast`** to close the symmetry with `jax_gpu_fast`:

```python
VALID_BACKEND_MODES = (
    "native_cpu",
    "jax_cpu_fast",       # NEW — speed-opt-out CPU
    "jax_cpu_parity",     # verification CPU, byte-identity goal
    "jax_gpu_fast",       # speed-opt-out GPU (existing)
    "jax_gpu_parity",     # verification GPU, byte-identity within build
    "jax_metal_smoke",
)
```

`_MODE_TO_RUNTIME["jax_cpu_fast"] = ("jax", "cpu")`.

`_MODE_POLICY_DEFAULTS["jax_cpu_fast"]`:

```python
{
    "parity_mode": False,
    "requires_x64": True,
    "chunk_policy": "performance_tuned",
    "tolerance_tier": "fast",
    "compilation_cache_policy": "optional_persistent",
    "provenance_label": "jax_cpu_fast",
    **_NO_CI_REPRODUCIBILITY_DEFAULTS,
}
```

`tolerance_tier="fast"` is descriptive metadata for provenance; it does
**not** trigger a different gate. The strict gate is mode-blind.

The remaining mode-keyed dicts also need explicit entries (otherwise
`set_backend("jax_cpu_fast")` throws `KeyError` through chunk-tuning or
transfer-guard wiring):

```python
_FIELD_KERNEL_DEFAULTS["jax_cpu_fast"] = {
    "coil_chunk_size": 64,
    "quadrature_block_size": 64,
}
_MODE_SHARDING_DEFAULTS["jax_cpu_fast"] = "none"
_DEFAULT_TRANSFER_GUARD_BY_MODE["jax_cpu_fast"] = "log"
```

The chunk sizes mirror `jax_gpu_fast`'s `performance_tuned` tuning. The
sharding default is `"none"` because CPU sharding does not currently
benefit `jax_cpu_*` execution; researchers needing hybrid CPU sharding
should open a separate slice. The transfer-guard `"log"` matches every
other non-`native_cpu` mode.

### 2.3 Empirical reporting context (not a tolerance lane)

Add **one** new reporting context to a separate SSOT,
`PARITY_LADDER_REPORTING_CONTEXT`, to carry the per-layer empirical
thresholds derived in DM-B from passing artifacts:

```python
PARITY_LADDER_REPORTING_CONTEXT = {
    "pre_newton_state_empirical": {
        "threshold_kind": "empirical_per_layer",
        "purpose": "report_severity",  # NOT "gate"
        "source_artifacts": [
            # Set in DM-B from .artifacts/parity/<passing-artifact>/result.json
        ],
        # Per-layer thresholds populated empirically — see §11
        "per_layer": {
            "boozer_solve.pre_newton_state": {
                "baseline_max": ...,
                "safety_factor": ...,
            },
            "boozer_solve.pre_newton_objective_gradient": {...},
            "iota_penalty.adjoint": {...},
            "boozer_solve.linear_solve_factors": {...},
            "boozer_solve.final_solved_state": {...},
            "boozer_solve.final_hessian": {...},
        },
        "requires_byte_identity": False,  # severity is relative; gate stays strict
    },
}
```

This context is **not** part of `PARITY_LADDER_TOLERANCES`, whose current
type is `dict[str, dict[str, float | bool | None]]`. It is referenced by
the *report* `_pre_newton_census_gate_failures` emits when a strict
divergent layer is found — the report says "drift is N× baseline,"
where N is computed against this context's `per_layer` thresholds.

### 2.4 Gate function: refactor for context, NOT for policy split

`_pre_newton_census_gate_failures` at
`benchmarks/single_stage_init_parity.py:1905` is **not split**. It
keeps strict-gate semantics. Its failure messages are augmented:

```python
def _pre_newton_census_gate_failures(parity_bug_census, *, severity_context=None):
    """Hard-gate: any boozer_solve.pre_newton_* divergent layer fails.

    Augments the failure message with empirical-baseline context drawn
    from `severity_context`
    (PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"])
    so engineers see how anomalous the drift is. The empirical context
    drives REPORTING, not gating. The gate's pass/fail decision is
    unchanged from the prior strict-only behavior.
    """
    if not parity_bug_census:
        return []
    failures = []
    for entry in parity_bug_census.get("divergent_layers", []):
        family = entry.get("family")
        layer = str(entry.get("layer", ""))
        if family != "boozer_solve" or not layer.startswith("pre_newton"):
            continue
        max_abs = float(entry.get("max_abs_diff", 0.0))
        ref_abs = float(entry.get("reference_abs", 1.0))
        ctx = _empirical_severity_context(
            f"{family}.{layer}", max_abs, severity_context,
        )
        # ctx is e.g. " (drift is 100× empirical baseline of 4.5e-11)"
        failures.append(
            f"Parity bug census reported divergent {family}.{layer}: "
            f"max_abs_diff={max_abs} at pair {entry.get('pair_index')} "
            f"(line-search eval {entry.get('line_search_evaluation')}){ctx}."
        )
    return failures
```

The `severity_context` argument is optional; without it, behavior is
identical to today's strict gate. With it, messages carry context.

### 2.5 What stays the same

- The cpu_ordered twins
  (`src/simsopt/geo/surface_fourier_jax_cpu_ordered.py`,
  `src/simsopt/jax_core/biotsavart_cpu_ordered.py`) are **unchanged**.
- `parity_policy="production" | "cpu_ordered"` plumbing — **unchanged**.
- `is_parity_mode()` — **unchanged**.
- `native_cpu` — **unchanged** as the reference oracle.
- The strict gate at `single_stage_init_parity.py:1905` — keeps
  release-blocker semantics.
- Phase 4 plan §2 "Hard Constraints" — **unchanged**.
- All Phase 0/1/2/3/6 code that landed in `e61370cdf` and the docs
  revisions in `f5a6c3042` — **unchanged**.

### 2.6 Carve-out vs. Phase 4 plan §15 "Untouched (do not modify)"

The bit-identity plan §15 (`docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md:772-781`)
declares the following files **untouched within the bit-identity slice
itself**:

- `benchmarks/single_stage_init_parity.py:1905` (gate function)
- `tests/test_benchmark_helpers.py:1415` (gate regression test)
- `benchmarks/validation_ladder_contract.py`

This dual-mode plan **explicitly carves out** the right to modify
those three files **for empirical-reporting scaffolding only** (DM-A's
reporting-context addition and DM-B's `severity_context` argument
refactor). The
carve-out is narrowly scoped:

- **Permitted by this plan**: adding
  `PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]`;
  adding an optional `severity_context` argument to
  `_pre_newton_census_gate_failures`;
  updating the matching test to verify the augmented message format.
- **NOT permitted by this plan**: changing the gate's pass/fail
  decision logic, the strict-byte-identity contract, or any
  PARITY_LADDER_TOLERANCES entry that affects gate semantics.

Phase 4 plan §15's "untouched" promise applies to Phase 4's *own*
implementation work; DM-A/B's mode/reporting-context scaffolding is tracked here
as an **adjacent** workstream. The bit-identity plan should be
amended (DM-D) to add the cross-pointer; Phase 4 itself remains
allowed to assume those files are stable when its own checkboxes
fire. If DM-A/B ships before Phase 4's gate work, the file edits
land first and Phase 4 inherits the new shape. If Phase 4 ships
first, DM-A/B layers on top.

## 3. Phase 4 Disposition (revised 3b)

| Question | Answer |
|---|---|
| Is Phase 4 a release blocker? | **Yes.** Production CI cannot turn green until the strict `pre_newton_*` gate clears. No reframe to diagnostic. |
| What does this plan add to Phase 4? | (a) Per-layer empirical reporting in the gate's failure messages so triage is faster. (b) Boundary-input byte tests (P4.5/P4.5b in the bit-identity plan §10) get cleaner naming and explicit canonical-input assignment. (c) The `jax_cpu_fast` mode lets researchers run faster speed experiments that opt out of byte identity, but those runs **cannot be cited as production-grade** until the strict gate passes for the same physics. |
| Does Phase 4 plan §2 "Hard Constraints" change? | **No.** "Do not loosen tolerances" still applies. |
| Does Phase 4 acceptance criteria change? | **No.** P4.5/P4.5b boundary byte tests pass + P4.6 strict gate passes. |
| What if Phase 4 closes the LS layers but `iota_penalty.adjoint` (2.32e-10) still drifts? | The strict gate still fails. That's a separate slice (DM-E #1, IFT adjoint parity). Production CI stays red until that slice also lands. The dual-mode plan does not provide a relaxation path. |

This is harsher than v1 promised — it does NOT unblock production.
It buys engineering hygiene (faster triage, cleaner mode taxonomy)
without changing what gates production.

## 4. Migration Path

Four slices (DM-C from v1 is killed). DM-A adds one public opt-in runtime
mode, but existing modes keep their behavior. The other slices are
docs/test/refactor work.

### Slice DM-A — Mode + empirical-reporting scaffolding

- [ ] Add `jax_cpu_fast` to **all six** mode-keyed dicts in
      `src/simsopt/backend/runtime.py` (any one missing fires a
      `KeyError` at `set_backend("jax_cpu_fast")`):
      - `VALID_BACKEND_MODES` (line 100)
      - `_MODE_TO_RUNTIME` (line 108) — `("jax", "cpu")`
      - `_MODE_POLICY_DEFAULTS` (line 124) — see §2.2 body
      - `_FIELD_KERNEL_DEFAULTS` (line 176) —
        `{"coil_chunk_size": 64, "quadrature_block_size": 64}`
        (mirrors `jax_gpu_fast`'s performance_tuned tuning)
      - `_MODE_SHARDING_DEFAULTS` (line 189) — `"none"`
        (CPU sharding does not typically apply; if a researcher needs
        hybrid CPU sharding later, add a separate slice)
      - `_DEFAULT_TRANSFER_GUARD_BY_MODE` (line 297) — `"log"`
        (matches every other non-`native_cpu` mode)
- [ ] Add
      `PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]`
      (single reporting context, per-layer thresholds populated by
      DM-B) to `benchmarks/validation_ladder_contract.py` or a
      dedicated dependency-light reporting-contract module imported by
      the benchmark helper.
- [ ] Add a unit test under `tests/backend/` asserting wiring:
      `set_backend("jax_cpu_fast")` round-trips successfully; resolves
      runtime target `("jax", "cpu")`; reports `is_parity_mode() is
      False`; returns policy/provenance fields
      `chunk_policy="performance_tuned"`, `tolerance_tier="fast"`,
      `provenance_label="jax_cpu_fast"`; returns static chunk defaults
      `coil_chunk_size=64`, `quadrature_block_size=64`; returns sharding
      default `"none"`; and returns transfer guard `"log"`. This proves
      all mode-keyed SSOT entries are present.

**Behavior change:** no behavior change for existing modes. `jax_cpu_fast`
is a new public opt-in mode; the reporting context is inert until DM-B
references it.

### Slice DM-B — Empirical threshold derivation + gate report refactor

> **Important: corpus is empty until Phase 4 lands the first passing
> artifact.** As of 2026-05-08, every `.artifacts/parity/*/result.json`
> reports `passed: false`. The empirical corpus is therefore inert —
> not a deliverable shape error, an upstream gating relationship.

- [ ] **Gate refactor (lands without the corpus).** Refactor
      `_pre_newton_census_gate_failures` at
      `benchmarks/single_stage_init_parity.py:1905` to accept an
      optional `severity_context` argument and append empirical-baseline
      context to its failure messages **when** the context is populated.
      When `severity_context` is `None` or the context's `per_layer` dict is
      empty/`INSUFFICIENT_SAMPLES`, behavior is identical to today's
      strict gate. **Keep strict-gate semantics intact** — the gate
      still fails any divergent `pre_newton_*` layer regardless of
      context state.
- [ ] **Test refactor.** Update
      `tests/test_benchmark_helpers.py:1415` to verify the augmented
      failure message format **conditional on context content**. Cover
      both shapes: (a) `severity_context=None` → unchanged message, (b)
      `severity_context` populated with a stub layer → message includes
      drift/baseline ratio. Do NOT add a tolerance gate test.
- [ ] **Corpus build (gated on Phase 4).** Once Phase 4 produces
      `.artifacts/parity/<DATE>-derivative-bit-identity-zeroing-pass/result.json`
      with `passed: true`, walk every directory under
      `.artifacts/parity/` whose `result.json` reports `passed: true`
      AND has `same_candidate_replay.parity_bug_census`. List the
      candidates and freeze the corpus.
- [ ] **Threshold derivation (gated on corpus ≥ 1).** For each layer
      in the `parity_bug_census.max_layer_diffs` of every corpus
      artifact, compute the spread (min, max, median, p95) of observed
      `max_abs_diff` values. Per-layer `baseline_max = p95` (or `max`
      if sample size ≥ 5 and spread < 1 OOM); `safety_factor = 5×`
      default. Final per-layer reporting threshold = `safety_factor ×
      baseline_max`. Layers with sample size < 3 are marked
      `INSUFFICIENT_SAMPLES` and omitted (not reported, not gated).
- [ ] **Reporting-context population.** Populate
      `PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]["per_layer"]`
      with the measured numbers. Cite the corpus artifact list and
      measurement command in the context's docstring.

**Behavior change:** failure messages get richer once the corpus is
populated; until then, the gate refactor is a no-op extension. Gate
pass/fail decisions never change.

**DM-B is two sub-slices in dependency order**: gate/test refactor
ships first (immediately, no corpus needed); corpus + thresholds ship
**after** Phase 4's first passing artifact exists. Do not block DM-A
or DM-D on DM-B's corpus availability.

### Slice DM-C — KILLED in v2

(Was: production CI on tolerance gate. Killed because production gate
stays strict — there is no tolerance gate to switch to.)

### Slice DM-D — Bit-identity plan cross-reference update

- [ ] Add a "see `parity_dual_mode_contract_2026-05-08.md` (3b
      semantics)" pointer at the top of
      `docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`
      §10. Body: this plan ADDS diagnostic granularity to Phase 4,
      it does NOT loosen the production contract or move the strict
      gate to "diagnostic only."
- [ ] §16 "Today's First Checkbox" still points at P4.1 as the
      critical-path action; no change to that pointer.

**Behavior change:** docs cross-link only.

### Slice DM-E — Downstream parity surfaces (post-Phase 4)

After Phase 4 closes the LS-side byte parity, address the remaining
strict-gate blockers in their own slices:

- [ ] IFT adjoint parity slice for
      `iota_penalty.adjoint = 2.32e-10` — separate planning doc
      mirroring this one's structure.
- [ ] LU factorization decision for
      `linear_solve_factors = 8.65e-12`. Either align
      `jax.scipy.linalg.lu_factor` to LAPACK byte-for-byte (likely
      impossible without patching one or the other) or accept a
      permanent **strict-gate** exemption with explicit justification.
- [ ] Cross-host pinning if multi-host byte identity is required
      (`-march=skylake-avx512`,
      `--xla_cpu_enable_platform_dependent_math=false`). If not,
      document explicitly that strict-mode bytes are valid only on
      the local build host.
- [ ] (Optional) Noise-tolerant BFGS — adopt as a SciPy-callable
      optimizer option per Shi/Xie/Byrd/Nocedal 2020 lengthening
      (<https://github.com/hjmshi/noise-tolerant-bfgs>) IF the project
      decides to relax convergence sensitivity at the algorithm
      level. Independent of the dual-mode contract.

## 5. Test / CI Lane Map (revised)

```
                            production CI       diagnostic CI       paper artifact
                            (every commit)      (cron / on-demand)  (pre-publication)
─────────────────────────   ─────────────────   ─────────────────   ─────────────────
backend mode                 native_cpu          jax_cpu_parity      jax_cpu_parity
                             jax_cpu_parity      jax_gpu_parity      + cross-host pin
                             jax_gpu_parity                          (DM-E)
gate                         strict              strict              strict
empirical reporting?         yes (DM-B)          yes (DM-B)          yes (DM-B)
runs Phase 4 fixes?          yes (must close)    yes (must close)    yes (must close)
runs IFT adjoint check?      yes (must close,    yes (must close)    yes
                             DM-E)
acceptance criterion         max|jax−cpu| == 0   max|jax−cpu| == 0   max|jax−cpu| == 0
                             at byte level on    at byte level on    at byte level on
                             pre_newton_*        all gated layers    all gated layers
                                                                     + cross-host
                                                                     verification
```

`jax_cpu_fast` and `jax_gpu_fast` are NOT in the CI lane map. They are
**researcher-opt-in** modes for speed experiments. Such runs must be
explicitly labeled as non-production-grade in any artifact they
produce; pre-publication artifacts MUST come from `*_parity` lanes.

## 6. Documentation Updates

- [ ] `CLAUDE.md` — add a brief "Parity modes" subsection (≤ 20 lines)
      summarizing: production runs the strict gate; speed-opt-out
      modes exist for research exploration but are not citable as
      production-grade; verification runs both production CI and
      diagnostic CI use the same strict gate. Link to this doc.
- [ ] `docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`
      §10 — DM-D pointer (see above).
- [ ] `benchmarks/validation_ladder_contract.py` — docstring at the
      top of
      `PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]`
      pointing here and citing the corpus artifact list.
- [ ] `benchmarks/single_stage_init_parity.py` — argparse / docstring
      update for the new `severity_context` argument on the gate
      function.

## 7. Risk Register (revised)

| Risk | Mitigation |
|---|---|
| Researchers use `jax_cpu_fast` for paper-grade artifacts and miss the byte-identity warning. | Provenance metadata records `backend_mode` per artifact; pre-publication review checks it. Add a CI lint that fails if a paper artifact's provenance shows a `*_fast` mode. |
| The empirical baselines drift over time as the corpus grows. | Re-run DM-B's measurement pass periodically (e.g. on every JAX/XLA upgrade); re-derive thresholds. Document the corpus + measurement script so it's reproducible. |
| Adding empirical reporting to gate messages changes test fixtures that match exact failure strings. | DM-B's gate refactor takes the reporting context as an OPTIONAL argument with default `None`. Existing tests calling the gate without a context see unchanged messages. |
| The dual-mode plan adds a mode and reporting context that nobody uses. | DM-A/B/D are the bulk of the work. If `jax_cpu_fast` sees no adoption, the cost is low (a few SSOT entries). DM-E slices are gated by Phase 4 closing and IFT adjoint slice progress. |
| The IFT adjoint and LU factor parity issues are intractable. | DM-E #2 explicitly allows a documented strict-gate exemption for `linear_solve_factors`. The exemption requires justification and a separate decision; this plan does not preempt it. |
| Phase 4 takes longer than expected; production CI stays red for an extended period. | This plan does not relieve that pressure. If production-research velocity becomes a critical concern, escalate as a separate decision: either accelerate Phase 4 staffing, adopt noise-tolerant BFGS to make production tolerance-of-noise an algorithmic property (not a gate property — see DM-E #4), or revisit whether the byte-identity contract was right. None of those decisions live in this plan. |

(The original v1 risk register row about `rtol=1e-12` being "too tight
or too loose" is **deleted** because there is no `rtol=1e-12` lane in
v2. The `pre_newton_state_empirical` reporting context uses per-layer measured
thresholds, not a global rtol.)

## 8. Files Touched (Summary, by slice)

### Slice DM-A
- `src/simsopt/backend/runtime.py` — add `jax_cpu_fast`.
- `benchmarks/validation_ladder_contract.py` — add
  `PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]`
  (per-layer skeleton; populated by DM-B).
- `tests/backend/test_runtime.py` (or extend) — assert SSOT wiring.

### Slice DM-B
- `benchmarks/single_stage_init_parity.py:1905` — accept
  `severity_context` argument; emit augmented failure messages.
- `benchmarks/validation_ladder_contract.py` — populate
  `PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]["per_layer"]`
  with measured numbers.
- `tests/test_benchmark_helpers.py:1415` — verify augmented format.
- New: a measurement script (e.g.,
  `benchmarks/parity/derive_pre_newton_baselines.py`) that walks
  the corpus and emits the per-layer numbers; commit script + corpus
  manifest.

### Slice DM-D
- `docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md` —
  add cross-pointer to this doc at top of §10.
- `CLAUDE.md` — add "Parity modes" subsection.

### Slice DM-E
- (Each downstream parity slice is its own planning doc.)

## 9. Acceptance / Done Criteria

This plan is "done" when:

1. **DM-A**: `jax_cpu_fast` mode and `pre_newton_state_empirical`
   reporting context ship; no regression in existing tests; the context has the
   per-layer skeleton in place (numbers populated by DM-B).
2. **DM-B**: per-layer baselines measured against the frozen corpus;
   gate failure messages include the empirical-severity context;
   strict-gate behavior unchanged on the failing artifact (still
   fails); strict-gate behavior unchanged on passing artifacts (still
   passes).
3. **DM-D**: bit-identity plan §10 has the cross-pointer; CLAUDE.md
   has the parity-modes section; researchers reading either doc reach
   a consistent picture.
4. **DM-E**: gated on Phase 4 closing; this plan does not commit to
   when DM-E lands.

After this plan completes, Phase 4 (the bit-identity plan §10) ships
under unchanged acceptance criteria. The strict gate at
`single_stage_init_parity.py:1905` continues to be the release-blocker
contract for production CI.

## 10. Open Decisions

These need project owner sign-off but do not block DM-A/B/D from
shipping; they shape DM-E.

1. **Default mode.** Keep `native_cpu` as the unset default
   (recommended), or shift the default to `jax_cpu_fast` once
   researchers have validated `jax_cpu_fast` produces equivalent
   physics on a subset of well-conditioned problems? Conservative
   recommendation: keep `native_cpu`. Researchers opt into
   `jax_cpu_fast` explicitly per run.

2. **Noise-tolerant BFGS adoption.** Adopt as a research-time SciPy
   optimizer option (Shi/Xie/Byrd/Nocedal 2020 lengthening), defer,
   or skip? **Independent** of this plan's contract; purely an
   optimizer-quality decision. Adopting could let production runs
   tolerate larger gradient noise without changing the gate, which
   may incidentally accelerate Phase 4's path to closure.

(The original v1 had three more open decisions:
production-tolerance-floor, diagnostic-CI-cadence, and
`jax_gpu_parity`-strict-lane-semantics. All three are eliminated by
3b: there is no production tolerance floor, no separate diagnostic CI
lane, and `jax_gpu_parity` follows the same strict-gate rule as
`jax_cpu_parity` — byte identity within build, cross-host parity
deferred to DM-E #3.)

## 11. Per-Layer Empirical Threshold Derivation (DM-B Spec)

This section pins the methodology DM-B uses to populate
`PARITY_LADDER_REPORTING_CONTEXT["pre_newton_state_empirical"]["per_layer"]`.
The thresholds are reporting context only — they augment failure messages,
they do not gate.

### 11.1 Corpus selection

The "passing artifacts" corpus is every directory under
`.artifacts/parity/` whose `result.json` reports `"passed": true` AND
whose `same_candidate_replay.parity_bug_census` is recorded. Freeze
this list in DM-B; cite it in the reporting context's docstring.

**Current corpus state (2026-05-08): empty.** Every existing
`.artifacts/parity/*/result.json` reports `passed: false` (verified by
JSON inspection on 2026-05-08). The two `report.json` files
(`2026-05-02-release-grade-cpp-jax-gate/`,
`20260428-cpu-cpp-jax-cpu-trace-e2e/`) use a different schema and do
NOT count as corpus members.

The corpus only becomes populated when Phase 4 produces the first
passing strict-gate artifact (per the bit-identity plan, this is
`.artifacts/parity/<DATE>-derivative-bit-identity-zeroing-pass/`).
Until then:

- DM-B's gate-refactor sub-slice ships with `per_layer = {}`.
- The empirical reporting context reports `INSUFFICIENT_SAMPLES` for every layer.
- The gate's augmented messages fall back to the strict-only format.

When Phase 4's passing artifact exists, the corpus initially has
**one** member. That is enough to ship per-layer baselines under
explicit `sample_size = 1` provenance, but most layers will be marked
`INSUFFICIENT_SAMPLES` (the spec requires ≥ 3 for a real baseline).
Re-derive the baselines as the corpus grows (typically one new
passing artifact per release cycle) until ≥ 3 samples per layer give
the empirical reporting context its first usable thresholds.

Always exclude `20260507-bfgs-prenewton-cpuordered-vg-m1/` — that's
the failing baseline.

### 11.2 Layers to characterize

Layers from `parity_bug_census.max_layer_diffs` (verified against the
failing artifact's report.json):

- `boozer_solve.pre_newton_state`
- `boozer_solve.pre_newton_objective_gradient`
- `boozer_solve.final_solved_state`
- `boozer_solve.final_objective`
- `boozer_solve.final_residual`
- `boozer_solve.final_gradient`
- `boozer_solve.final_hessian`
- `boozer_solve.linear_solve_factors`
- `iota_penalty.solved_state`
- `iota_penalty.linear_solve_factors`
- `iota_penalty.dJ_ds`
- `iota_penalty.adjoint`
- `iota_penalty.optimizer_projection_gradient`
- `iota_penalty.penalty_scale`
- `iota_penalty.penalty_optimizer_gradient`
- `iota_penalty.weighted_penalty_optimizer_gradient`

### 11.3 Statistic

For each layer, across the corpus, collect every `max_abs_diff` value.
Compute:

- `min`, `median`, `p95`, `max` across the corpus.
- Sample size (number of corpus artifacts that recorded this layer).

If sample size < 3 for a layer, mark it `INSUFFICIENT_SAMPLES` and
omit from the empirical reporting context. The corpus must grow before that layer
gets a baseline.

### 11.4 Threshold

Per-layer `baseline_max = p95` (or `max` if sample size ≥ 5 and
spread is < 1 OOM); `safety_factor = 5×` by default (override for
known-noisy layers). Final reporting threshold = `safety_factor ×
baseline_max`.

### 11.5 Severity classification

In the gate's augmented failure message, classify drift relative to
the empirical threshold:

- `drift / threshold ≤ 1.0`: marginal — within typical baseline noise
  amplified by the safety factor. Probably benign; investigate if
  it's growing.
- `1.0 < drift / threshold ≤ 10.0`: moderate — noticeably above
  baseline. Investigate.
- `drift / threshold > 10.0`: severe — orders of magnitude above
  baseline. Likely a real regression.

Example augmented message:

> Parity bug census reported divergent boozer_solve.pre_newton_state:
> max_abs_diff=4.519706948979962e-09 at pair 4 (line-search eval 4)
> [SEVERE: drift is 100× empirical baseline of 4.52e-11 (5× safety
> factor over corpus p95=9.04e-12 across 4 passing artifacts)].

The gate decision (fail) is unchanged; the severity tag and baseline
help triage.

### 11.6 Maintenance

- Re-run the measurement when (a) JAX or XLA upgrades, (b) the
  corpus grows by ≥ 2 artifacts, (c) any kernel touching the surface,
  BS, or residual code changes. Bump the context's `source_artifacts`
  field accordingly.
- Keep the measurement script under version control with explicit
  inputs (corpus list, statistic choice, safety factor) so the
  derivation is reproducible.
