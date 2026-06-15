# Single-Stage 11-vs-51 Matrix on Host-Driven SciPy (drop ondevice)

> Created 2026-06-13. Updated 2026-06-14 after external review (corrected a false fullgraph
> "feasibility proven" claim). Supersedes the ondevice rows of
> `docs/single_stage_11_51_matrix_2026-06-13.{json,md}`.

> **Execution status (2026-06-14):** Phases 1–3 IMPLEMENTED + locally validated — generator emits the
> 8 host-driven cells; launchers default `scipy-jax` and pass `--boozer-optimizer-backend`; manifest
> regenerated; new `tests/integration/test_single_stage_matrix_manifest.py` (8 tests, green). The
> Phase 0 GATE smoke RAN on RunPod A100 at `b5f97fdf9`: the `free_x` blocker is **RETIRED**
> (fullgraph-51 runs past DOF extraction; GPU lane did 5 outer iterations, 887 MiB peak). Two findings:
> (1) the fullgraph **CPU** lane crashed at `boozer_surface.py:5659` because its inner Boozer defaulted
> to `scipy` under `jax_cpu_parity` — **FIXED** by forcing `--boozer-optimizer-backend ondevice` on the
> fullgraph cells (the GPU lane already got it via the cuda auto-default); a confirmatory CPU smoke with
> the fix is in progress. (2) the **GPU** lane is blocked by the pre-existing scipy-jax GPU compile
> pathology (`docs/jax_scipy_jax_gpu_compile_diagnostic_next.md`) — separate from this migration. Still
> pending: confirmatory CPU pass, donor resubmit + mpol10, GPU-pathology decision.

## Purpose

Rebuild the single-stage production parity/performance matrix so **both** the 11-dim
and 51-dim formulations run on the **host-driven SciPy outer loop**, and **remove the
`ondevice` (monolithic-`jit`) lane entirely**. This is the research-endorsed fix for the
422 GiB compile-time OOM (host-drive the outer loop; compile only the per-step value/grad)
applied to *both* dimensions, using backends that already exist in code.

**Caveat (do not treat as done):** the 11-dim `scipy-jax` lane is run-proven, but the **only
harvested 51-dim `scipy-jax-fullgraph` run to date FAILED before optimization** (pre-fix `free_x`
bug, since fixed). fullgraph-51 feasibility must be re-confirmed by a passing smoke at the current
SHA before any production cell depends on it (see Current Context + the GATE below).

## Goals

- Matrix = **8 core cells**: `{scipy-jax (11-dim), scipy-jax-fullgraph (51-dim)} × {cpu, gpu} × {mpol2, mpol10}`.
- No `ondevice` cells anywhere in the matrix, and `ondevice` is no longer the production launcher default.
- Both lanes are OOM-safe **by construction** (single-eval compile, host outer loop) — no monolithic `jit(run)`.
- **GATE:** a passing `scipy-jax-fullgraph` smoke/init artifact (`rc=0`, `passed=true`) at the current SHA
  is required before any production (mpol=10) cell depends on the 51-dim lane — the only fullgraph artifact
  so far FAILED.
- 4 `mpol2` cells runnable immediately; 4 `mpol10` cells runnable once the donor lands **and** the
  fullgraph smoke passes.
- Change set passes the full Crucible reviewer loop before commit.

## Non-Goals

- Re-engineering or rescuing the `ondevice` lane (de-fusing the monolith). Out of scope; deferred.
- Inner Boozer least-squares variants `LM` / `lm-minpack`. Dropped (quasi-newton inner only); their
  effect is unvalidated on host-SciPy lanes and the inner backend differs between the two lanes.
- `optax-lbfgs` / `optimistix-lbfgs` lanes. Excluded (as before).
- Changing the native cpp/CPU reference (still driven at 51 full-space DOFs by the harness).

## Current Context (verified against code)

- **Both target backends exist and are host-driven (SSOT):**
  `TARGET_SCIPY_CONTROL_OPTIMIZER_BACKENDS = {"scipy-jax", "scipy-jax-fullgraph"}`
  (`src/simsopt_jax/geo/_optimizer_backend_choices.py:12-14`). Both dispatch to
  `optimizer_jax_reference.target_scipy_minimize_value_and_grad` (host SciPy L-BFGS-B) via
  `_TARGET_SCIPY_CONTROL_METHODS` (`src/simsopt_jax/geo/optimizers/optimizer.py:5121-5152`);
  method map `lbfgs-scipy-jax-fullgraph → scipy_lbfgsb` (`optimizer.py:388`).
- **`scipy-jax` (11-dim):** reduced coil-only, surface solved each iteration by the inner Boozer
  solve; outer L-BFGS-B on the host. Run-proven on RunPod (~5 GB host RSS, diagnostics doc).
- **`scipy-jax-fullgraph` (51-dim) — feasibility NOT yet proven (BLOCKER):** host SciPy L-BFGS-B
  over the full CPU-order `JF.x` vector with JAX-evaluated wrapper value/grad
  (`single_stage_banana_example.py:4806-4815`; contract
  `single_stage_optimizer_contract_uses_full_graph_jax_scipy`, `:8790-8796`). Inner Boozer backend
  defaults to `scipy`/native for this lane (`--boozer-optimizer-backend` help, `:4835`).
  **The only harvested fullgraph artifact**
  (`.artifacts/clean_reconciliation_benchmarks/cpu_330925564_single_stage_fullgraph_x64_surfacefix_20260612T000148Z/single_stage_cpu_fullgraph.json`)
  is `status: "case-execution-failed"`, `passed: false` (JSON), exit code **1** (the
  `single_stage_cpu_fullgraph.rc` sidecar = `1`, and the JSON `error.message` reports "Subprocess
  failed with exit code 1"; there is no structured `rc` field in the JSON) — it died **before optimization** with
  `AttributeError: 'jaxlib._jax.ArrayImpl' object has no attribute 'free_x'` at the fullgraph
  initial-DOF extraction (`single_stage_banana_example.py:9728`, the `JF.x` path). That exact crash —
  the `DeferredSurfaceXYZTensorFourier` proxy storing its runtime dof array in `_dofs`, shadowing
  `opt._dofs.free_x` — was fixed in `521fa05f1` (`_dofs`→`_runtime_dofs`; verified ancestor of HEAD,
  commit message quotes the `free_x` error), **but no passing fullgraph artifact exists at/after the
  fix.** The artifact records `provenance.optimizer_backend: "scipy-jax-fullgraph"` /
  `provenance.lane: "target-scipy-fullgraph-control"`. The ~11 min GPU figure in `project_perlmutter_run_timings`
  is a memory recall, **not** a passing-artifact citation.
- **Production harness already recognizes fullgraph:** `single_stage_init_parity.py:109,142`
  (`SCIPY_JAX_FULLGRAPH_OPTIMIZER_BACKEND → "lbfgs-scipy-jax-fullgraph"`).
- **Parity replay:** `scipy-jax-fullgraph` is replay-capable (exact same-candidate replay);
  `scipy-jax` (11) is **not**. Verified: `_EXACT_SAME_CANDIDATE_REPLAY_BACKENDS`
  (`single_stage_init_parity.py:146-150`) = {ondevice, scipy-jax-fullgraph, optax-lbfgs}; plain
  `scipy-jax` is absent. So the 51 lane *would* carry the bit-parity story **if it runs**; the 11 lane
  is performance/feasibility (dim-mismatched vs the 51 cpp reference).
- **Matrix generator/manifest SSOT:** `benchmarks/perlmutter/build_single_stage_matrix.py`
  (`FORMULATIONS`, `INNER_LS`, `PLATFORMS`, `TIERS`, `build_cells`) →
  `docs/single_stage_11_51_matrix_2026-06-13.{json,md}`; consumed by
  `benchmarks/perlmutter/submit_single_stage_matrix.py`.
- **Production launchers default to ondevice:** `single_stage_production_{cpu,gpu}.slurm`
  set `PROD_OPTIMIZER_BACKEND="${PROD_OPTIMIZER_BACKEND:-ondevice}"` (`cpu:26`, `gpu:28`). Cells pass
  it per-cell, but the default must change.
- **mpol=10 needs a warm-start donor** (cold high-res contract block
  `_require_supported_single_stage_seed_contract`, `single_stage_init_parity.py:1465`,
  resolution-based). The donor launcher `single_stage_continuation_donor.slurm` was fixed this session
  (`SIMSOPT_BACKEND_MODE=native_cpu`, Crucible strict PASS) but is **not yet committed/resubmitted**.
- **Working-tree state (uncommitted; `git status --short`):** modified
  `benchmarks/perlmutter/build_single_stage_matrix.py`,
  `benchmarks/perlmutter/single_stage_continuation_donor.slurm`,
  `docs/single_stage_11_51_matrix_2026-06-13.{json,md}`; untracked `HANDOFF.md`,
  `HANDOFF-ss-11-51-matrix.md`, this plan doc, and
  `tests/integration/test_continuation_donor_backend_contract.py`. The donor fix + new test are KEEP;
  the doc-drift edits to `build_single_stage_matrix.py`/manifest (which described the *ondevice* 51
  lane) are **superseded** by this rewrite, since ondevice is removed.

## Rationale

The `ondevice` lane compiles the entire outer loop + inner solve + adjoint + polish into one XLA
graph → 422 GiB compile-time section memory, infeasible on any node. Host-driving the outer loop
(SciPy) compiles only one value/grad eval per step; with hundreds of expensive iterations the
per-step host↔device transfer is negligible and nothing differentiates *through* the loop, so
on-device looping buys nothing here. `scipy-jax` (11) and `scipy-jax-fullgraph` (51) are exactly
those host-driven lanes (OOM-safe by construction). The 11 lane is run-proven; **fullgraph-51's only
run to date failed pre-fix and its feasibility is pending a passing smoke.** Together — once
fullgraph-51 is confirmed — they give a real 11-vs-51 comparison without the OOM. Quasi-newton inner
only keeps the matrix clean and symmetric (LM/lm-minpack unvalidated on host-SciPy lanes; the two
lanes use different inner backends).

## Assumptions

- `scipy-jax-fullgraph` runs to a `passed=true` result **at all** at the current SHA. Its only
  harvested run FAILED pre-fix with `free_x` (now fixed in `521fa05f1`); this assumption is UNVERIFIED
  and is the gating risk — confirm with a fresh smoke before relying on it.
- The cold-high-res warm-start contract is backend-agnostic, so **both** `mpol10` lanes need the
  donor warm-start (VERIFY — see Validation).
- `scipy-jax-fullgraph` accepts the production launcher's target-lane flag set
  (`--target-lane-boozer-*`, `--maxiter`, resolution flags) without error, or those flags are
  no-ops for its native inner solve (VERIFY).
- `scipy-jax-fullgraph` per-eval value/grad compiles within node memory at mpol=10 (it is one eval,
  not the whole loop) — UNVERIFIED (no passing fullgraph artifact yet).
- The native_cpu donor (already fixed/validated) produces a contract-valid warm-start consumable by
  both host-SciPy lanes (warm-start is backend-independent geometry).

## Fair-comparison protocol (mpol10 — the defensible deliverable)

The existing scipy-jax numbers are confounded: different host CPU slice (cpp reference 44s→2299s
between otherwise-identical runs), thread oversubscription, 219-vs-34 iterations to *different* optima,
mpol2 too small for GPU to amortize, and the scipy-jax GPU recompile storm. A fair comparison holds
everything constant except the device, using harness primitives that already exist.

**Controls:** (1) same node — cpp + JAX-CPU + JAX-GPU on one GPU node; (2) threads capped to allocated
cores (DONE in the launchers); (3) production resolution mpol10 + donor warm-start; (4) isolate the
optimizer-path divergence via **replay**; (5) separate compile from steady-state.

**Run 1 — Parity (port correctness, host-independent).** fullgraph mpol10, warm-started. The harness
auto-runs the same-candidate replay (`scipy-jax-fullgraph ∈ _EXACT_SAME_CANDIDATE_REPLAY_BACKENDS`;
`single_stage_init_parity.py:1517` → `:1657`, `_compare_same_candidate_*` `:1966-2222`) → per-candidate
value/grad/hardware **bit-parity vs cpp, convergence-independent**.
This is THE correctness number and does not depend on the convergence stall.

**Run 2 — Perf (cpu vs gpu, same node).** New `single_stage_fair_compare_gpu.slurm` on a GPU node runs
fullgraph mpol10 warm-started under `--platform cpu` then `--platform cuda`, thread-capped (cpu lane hides
the GPU; cuda lane samples GPU mem). Each invocation co-produces the cpp reference (parity anchor) on the
identical host. Report **per-iteration throughput**, MaxRSS, GPU mem — NOT raw total wall (iteration counts
diverge across lanes). Compile-vs-steady-state separation is recorded directly via
`--record-jax-compile-diagnostics` (now wired; default-on in the launcher, `FAIR_RECORD_COMPILE_DIAGNOSTICS=0`
to disable), which also counts GPU XLA recompiles — the key signal for the "GPU slower" question.

**Reuse, do not re-run:** scipy-jax 11-dim perf comes from the existing `06b7f1a8f` runs (perf-only /
dim-mismatched vs the 51 reference). The genuinely-new work is the fullgraph-51 lane.

**Implementation tasks:**
- [x] Thread caps (OMP/OPENBLAS/MKL/NUMEXPR = `SLURM_CPUS_PER_TASK`) in both production launchers.
- [x] `single_stage_fair_compare_gpu.slurm` — same-node cpu+cuda fullgraph mpol10 (one venv/build/clean-check,
      then a `run_lane` function runs the harness twice; cpu lane sets `CUDA_VISIBLE_DEVICES=""`, cuda lane
      samples GPU mem; job fails red if either lane fails).
- [x] Compile-vs-steady-state separation WIRED: `single_stage_init_parity.py` now exposes
      `--record-jax-compile-diagnostics` (default-off), threaded to the JAX target-lane child gated to
      `backend == "jax"` (the relay/resolver `_append_optional_single_stage_flags` /
      `resolve_target_lane_compile_diagnostics` already existed — only the CLI flag + one arg-sourced wire
      were missing). `single_stage_fair_compare_gpu.slurm` passes it on both lanes by default
      (`FAIR_RECORD_COMPILE_DIAGNOSTICS=1`; `=0` for a pristine throughput run). It is observational (toggles
      JAX compile logging only — compiled code/numerics unchanged), so the comparison stays fair, and the
      child writes a compile/cache-miss summary (incl. recompile counts) into `results.json`. Tests:
      `tests/integration/test_single_stage_init_parity_compile_diagnostics.py` (full chain, incl. the
      backend gate) + `test_fair_compare_launcher_contract.py`.

**Achievable now:** parity (replay) + CPU perf, once the donor lands. **GPU perf gated** on the recompile
fix — the compile diagnostics quantify it.

## Implementation Plan

0. **GATE — confirm fullgraph-51 actually runs (do this first; blocks Phases 4/production)**
   - [x] DONE (b5f97fdf9 gate smoke, RunPod A100, 2026-06-14): `free_x` RETIRED — fullgraph-51 runs
         past DOF extraction (GPU lane: 5 outer iterations, objective decreasing, 887 MiB peak). Neither
         lane reached `passed=true` yet: GPU is compile-bound (separate pre-existing issue), CPU crashed
         at `boozer_surface.py:5659`. A confirmatory CPU smoke with the fix below is in progress to reach
         `rc=0`/`passed=true`.
   - [x] Root cause (NOT the `free_x`/`JF.x` path, which is fixed): the fullgraph inner Boozer defaults to
         `scipy` and the harness only auto-supplies `--boozer-optimizer-backend ondevice` on cuda, not cpu
         (`single_stage_init_parity.py` `_resolve_target_boozer_optimizer_backend`). FIX: fullgraph cells
         force `--boozer-optimizer-backend ondevice` (generator `inner_boozer_optimizer_backend` →
         `PROD_BOOZER_OPTIMIZER_BACKEND`; launchers pass it). The GPU compile pathology is tracked
         separately (`docs/jax_scipy_jax_gpu_compile_diagnostic_next.md`).
1. **Rewrite the matrix generator** (`benchmarks/perlmutter/build_single_stage_matrix.py`)
   - [ ] `FORMULATIONS`: keep `11 → optimizer_backend="scipy-jax"`; change `51` from `ondevice`
         to `optimizer_backend="scipy-jax-fullgraph"`, `outer_optimizer="host-scipy"`, and a correct
         description (host SciPy over full `JF.x`; native inner Boozer solve). Remove the ondevice entry.
   - [ ] Collapse the inner-LS axis to quasi-newton only: reduce `INNER_LS` to `{"quasinewton": "quasi-newton"}`
         (or drop the `INNER_LS` loop in `build_cells`) so each `(dim, platform, tier)` yields exactly one cell.
   - [x] DONE — dropped the inner-LS axis entirely (no `inner_ls_applies`/`inner_ls_name`/`inner_ls_value` fields); the inner solve is recorded once as manifest-level `inner_boozer_least_squares: "quasi-newton"` (the child default, so `PROD_BOOZER_LS_ALGORITHM` stays empty and the launcher omits the override).
   - [ ] Update `formulation_backend_coupling` and the manifest `notes[]` (remove ondevice/penalty
         language; state both lanes are host-driven SciPy; 51=fullgraph full `JF.x`).
   - [ ] Confirm `build_cells` yields exactly 8 cells: `{scipy-jax, scipy-jax-fullgraph} × {cpu, gpu} × {mpol2, mpol10}`.
   - [ ] Keep `dim_matched_reference`: `False` for 11, `True` for 51 (cpp reference is 51).
2. **Update the production launchers** (`single_stage_production_{cpu,gpu}.slurm`)
   - [ ] Change `PROD_OPTIMIZER_BACKEND` default `ondevice → scipy-jax`.
   - [ ] Verify the launcher passes `--optimizer-backend "${PROD_OPTIMIZER_BACKEND}"` and that
         `scipy-jax-fullgraph` flows through unchanged (it is a valid `--optimizer-backend` choice).
   - [ ] Audit the target-lane flag set (`--target-lane-boozer-bfgs-maxiter`,
         `--target-lane-boozer-newton-*`, polish policy) against both host-SciPy lanes; drop or guard
         any flag that errors / is ondevice-only. Set per-cell `PROD_OPTIMIZER_BACKEND` for
         `scipy-jax-fullgraph` cells.
   - [ ] Remove ondevice-only mitigation comments/notes (422 GiB host-RAM doubling, polish-skip OOM notes).
3. **Regenerate the manifest**
   - [ ] `python benchmarks/perlmutter/build_single_stage_matrix.py --source-sha <clean SHA>` →
         `docs/single_stage_11_51_matrix_2026-06-13.{json,md}` (8 cells). Use the post-commit clean SHA.
4. **Donor → mpol=10 enablement** (depends on the already-fixed `native_cpu` donor AND Phase 0 GATE)
   - [ ] **GATE:** do not submit any mpol=10 `scipy-jax-fullgraph` cell until the Phase 0 fullgraph
         smoke is `rc=0`/`passed=true` at the current SHA.
   - [ ] Commit the donor `native_cpu` fix + its regression test (already Crucible-PASS this session).
   - [ ] Restage a fresh donor checkout at the committed SHA; resubmit `single_stage_continuation_donor.slurm`.
   - [ ] When the donor COMPLETES and yields a contract-valid warm-start, submit the mpol=10 cells via
         `submit_single_stage_matrix.py --tier mpol10 --warm-start-run-dir <donor>` (11-dim cells
         unconditionally; 51-dim cells only after the GATE passes).
5. **Submit the immediately-runnable smoke tier**
   - [ ] Submit the 4 `mpol2` cells (`--tier mpol2`): `{scipy-jax, scipy-jax-fullgraph} × {cpu, gpu}`,
         one submodule-initialized checkout per job. The two `scipy-jax-fullgraph` mpol2 cells ARE the
         Phase 0 GATE smoke at production staging — inspect their result JSON for `passed=true`.
6. **Docs / memory**
   - [ ] Mark `ondevice` smoke-only/deprecated for production in the matrix docs; retire the ondevice
         OOM-bisect plan (research settled it). Update the relevant memory entries.

## Validation Plan

- [ ] **fullgraph-51 GATE:** a `scipy-jax-fullgraph` run JSON shows `status != "case-execution-failed"`,
      `passed: true`, rc=0 at the current SHA (the prior artifact was the opposite).
- [ ] `build_single_stage_matrix.py` differential: regenerated manifest has exactly 8 cells with the
      right `optimizer_backend`/`formulation_dim`/`platform`/`tier`; zero `ondevice` cells.
- [ ] `python -m pytest tests/integration/test_continuation_donor_backend_contract.py tests/integration/test_continuation_ladder.py -q` passes.
- [ ] `ruff check` + `py_compile` on changed Python; `bash -n` on changed launchers.
- [ ] **VERIFY (contract):** confirm `mpol10` cold-high-res block applies to both lanes (read
      `single_stage_init_parity.py:1465` seed-contract path); confirm whether `scipy-jax-fullgraph` needs
      the same warm-start as `scipy-jax`.
- [ ] **VERIFY (flags):** dry-run `submit_single_stage_matrix.py` (prints sbatch) and parse-validate the
      launcher argv for a `scipy-jax-fullgraph` cell against the wrapper parser before remote submit.
- [ ] Perlmutter smoke: the 4 `mpol2` cells COMPLETE (both lanes, both platforms); confirm
      `scipy-jax-fullgraph`-51 compiles + runs + `passed=true` at smoke res (no OOM, sane iters/iota).
- [ ] Full Crucible reviewer loop (correctness/contract, matrix accuracy/SSOT, launcher) → strict PASS.

## Risks and Mitigations

- Risk (PRIMARY): `scipy-jax-fullgraph`-51 may not run at all — its only harvested artifact FAILED
  before optimization with `free_x` (`single_stage_cpu_fullgraph.json`, pre-`521fa05f1`). The fix is now
  in HEAD, but feasibility is unproven and a fresh run may surface a different failure (failing path =
  the fullgraph `JF.x` DOF extraction, `single_stage_banana_example.py:9728`).
  Mitigation: Phase 0 GATE — a passing mpol=2 fullgraph smoke (`rc=0`/`passed=true`) at the current SHA
  before any mpol=10 fullgraph cell. If it still fails, fullgraph-51 becomes a debug item and the 51-dim
  lane is blocked; 11-dim `scipy-jax` is unaffected and proceeds.
- Risk: `scipy-jax-fullgraph` per-eval value/grad compile is slow/large at mpol=10 (bigger graph than 11).
  Mitigation: it is a single eval (not the whole loop); watch the first mpol=10 compile; if slow,
  `XLA_FLAGS=--xla_..._compilation_parallelism` (stopgap), not a redesign.
- Risk: production launcher target-lane flags are ondevice-shaped and error on the host-SciPy lanes.
  Mitigation: the flag-audit task (Phase 2) + parse-validate dry-run before submit.
- Risk: `mpol10` warm-start contract differs between the two lanes (one rejects the donor).
  Mitigation: the contract-verify task; if fullgraph needs a different seed field, synthesize/extend.
- Risk: 11-dim `scipy-jax` is dim-mismatched vs the 51 cpp reference and not replay-capable.
  Mitigation: report it as performance/feasibility; bit-parity rides on the 51 lane **iff it runs**.

## Completion Criteria

- [ ] **fullgraph-51 GATE passed:** `scipy-jax-fullgraph` mpol=2 smoke `rc=0`/`passed=true` at the current SHA.
- [ ] Generator + launchers + regenerated manifest contain only the 8 host-driven cells; no ondevice.
- [ ] All Validation items checked; Crucible strict PASS.
- [ ] Donor fix committed; donor resubmitted; 4 `mpol2` cells submitted (and `mpol10` queued post-donor + post-GATE).
- [ ] Matrix docs/memory updated (ondevice deprecated for production).

## Open Questions

- **(Likely BLOCKING)** Does `scipy-jax-fullgraph`-51 optimize the surface as genuinely free variables
  (true full-space penalty) or solve it via the native inner backend? The only fullgraph run failed at
  the `JF.x` DOF-extraction path (`single_stage_banana_example.py:9728`), so the surface/DOF handling is
  implicated in whether the lane runs at all — not just in the lane's description. Resolve as part of the
  Phase 0 GATE.
- Confirm both `mpol10` lanes require the donor (vs only 11) — gates whether 51 can run cold at mpol=10.
- Keep an 11-only LM/lm-minpack *extended* probe later, or leave the inner-LS axis fully dropped?
