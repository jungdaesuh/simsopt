# JAX-Native Performance TODOs — Round 3 (2026-05-18)

Lineage:

- Round-1 audit & port: `docs/jax_parity_manifest.md`,
  `docs/jax_port_*` (M1-M6 layout).
- Round-2 perf TODOs: `docs/jax_native_round2_performance_todos_2026-05-18.md`
  added in commit `2592b3d0b` and closed (N1-N20) in commit `f5411c412`
  the same day. Round-2 closeout table at lines 159-187 of that doc is
  authoritative; the unchecked `## TODO details` sections below it are
  retained as the original audit breakdown and are **not** active.

Provenance: this document was opened after a 5-agent parallel Opus
audit (host-device transfers / JAX primitive utilization / GPU & sharding /
compilation & memory pressure / backend ergonomics) followed by three
external cross-validation passes against HEAD `6b5867e04`. Items that
round-2 already closed are explicitly excluded from this round. The
third cross-validation tightened wording on N26 (VJP fan-out counts),
	N30 (CUDA toolchain citation), N31 (allocator env vars and proposed
API surface), and N34 (no transparent in-process platform retargeting).

## Purpose

Track the performance and ergonomics residuals that remain after the
round-2 closeout. Three user concerns drive the prioritization:

1. **Maximize GPU acceleration** — minimize host↔device transfers in
   production hot paths.
2. **Leverage JAX/CUDA strengths** — `jit`, `vmap`, donation, sharding,
   precision contracts, deterministic enforcement.
3. **Provide CPU-fallback configs that relieve GPU memory pressure**
   from full compilation — runtime-level JAX/XLA memory env vars, per-platform
   dense-Jacobian budgets, selective host residency, documented
   recovery workflows.

The boundary between this round and round-2 is sharp: round-2 closed
the in-kernel and sharding-scaffolding items. Round-3 picks up adapter-
layer residuals, host-sync hot spots, runtime memory plumbing, and the
multi-GPU GPU-speedup proof that round-2 explicitly did not claim.

## Goal

By the end of this round:

- M5 IFT adjoint and direct-coil objective paths emit zero host
  transfers under `jax.transfer_guard("disallow")` over the inner solve
  and value/grad regions.
- Adapter-level Python `for`-over-jitted-kernel sites are vectorized
  or jit-stable.
- `apply_jax_runtime_config()` is the single source of truth for GPU
  memory-pressure mitigation env vars (preallocate, mem-fraction,
  allocator).
- `max_dense_jacobian_bytes` defaults are platform-aware and env-overridable.
- A selective host-residency path exists for large warm-start factors.
- N11 and N12 multi-GPU GPU-speedup is measured on real hardware (gated
  on hardware access).
- A documented checkpoint-and-restart OOM-recovery workflow exists for
  the `jax_gpu_*` → `jax_cpu_*` transition.

## Scope

In scope:

- Items **N21-N34** below.
- Tightening the matmul-precision contract under the parity ladder.
- CPU-fallback ergonomics **within** the constraint that JAX's
  platform/backend selection is process-import-locked once
  `jax.devices()` has been queried.

Out of scope:

- Re-litigating round-2 closed items (N1-N20). If a round-2 item is
  found regressed, open a separate revert/repair item, not a round-3
  entry.
- Pallas/Triton custom kernels — round-2 N18 closed this as a
  feasibility-decision-only item; the same gating applies here.
- Transparent per-call GPU-OOM-to-CPU retry of an already-compiled
  JIT. JAX does not support retargeting a compiled JIT across
  platforms; this round's CPU-fallback work is checkpoint/restart
  plus selective host residency (see N34).
- Multi-host (`jax.distributed.initialize`) production validation
  beyond the single-host multi-GPU proof in N30. Multi-host is the
  next round.
- VMEC / wireframe / QFM out-of-scope kernels — same boundary as
  `docs/jax_parity_manifest.md`.

## Global guardrails

These apply to every item below unless the item explicitly relaxes
them.

1. **Parity ladder is sacred.** No change may regress
   `benchmarks/single_stage_init_parity.py::_pre_newton_census_gate_failures`
   on any `*_parity` mode. Validate against
   `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`.
2. **Donation policy.** Per the updated round-2 N3 closeout at
   current HEAD, optimizer-runner donation is not retained: the
   audited L-BFGS-B state donation emitted JAX "donated buffers were
   not usable" warnings. New donation is allowed only when a local
   wrapper owns a fresh, semantically-dead buffer and the output can
   actually reuse it without warnings. The current canonical positive
   pattern is the caller-copy boundary at
   `src/simsopt/geo/surfaceobjectives_jax.py:4810-4845`, which copies
   caller input before invoking the donating JIT. JAX buffer-donation
   docs are the upstream reference.
3. **Profiling before refactoring.** Per round-2 N9 and N18, no
   speculative remat or kernel rewrite without a recorded HLO or
   profile baseline. Use `benchmarks/grouped_adjoint_memory_probe.py`
   and `benchmarks/surface_rz_geometry_hlo_probe.py` as the existing
   measurement scaffolding.
4. **Transfer-guard discipline.** Wave-2 items (N21, N22, N23) must
   demonstrate zero host transfers under
   `jax.transfer_guard("disallow")` on the inner region. Use the
   pattern already proven at
   `src/simsopt/geo/optimizer_jax.py:1468` (Wave-1 N13 closeout).
5. **CPU vs GPU defaults split.** Where a knob differs in safe value
   between platforms, plumb the split through `BackendPolicy`, not
   through caller `options=`. Caller `options=` is the override
   layer.
6. **No new public Optimizable API breakage.** New configuration
   surfaces are additive. The existing `set_backend()` /
   `BackendPolicy` / `BoozerSurfaceJAX(options=...)` boundaries are
   the extension points. When a new keyword needs to flow through
   `set_backend()`, add it explicitly to the signature in the same
   patch; do not assume a `policy=` parameter exists today (the
   current signature at `runtime.py:1740-1748` is
   `(mode, *, strict, debug_nans, transfer_guard,
   compilation_cache_dir, configure_runtime)`).
7. **No platform retargeting at runtime.** Per JAX upstream
   constraints, a JIT compiled against GPU cannot be retargeted to
   CPU. CPU-fallback solutions in this round are checkpoint/restart,
   per-instance device pinning, and selective host residency — not
   transparent retry.

## Cross-validation provenance

This round-3 doc is the synthesis of:

- 5-agent parallel audit (host-device / primitives / GPU+sharding /
  memory pressure / ergonomics) producing 15-row table; see
  conversation transcript 2026-05-18.
- First cross-validation (external) that flagged the round-2 closeout
  and corrected six rows.
- Second cross-validation (external) that confirmed the first and
  tightened wording on N25 (grouped field) and surfaced the
  matmul-precision pin gap as standalone N27 (separate from round-2
  N7's fused-contraction closeout).
- Third cross-validation (external) that corrected:
  - N26 VJP counts (file has 44 total `jax.vjp` calls; 26 in the
    lines 84-338 cluster) and softened the prescription from
    "`jax.linearize` only" to allow composed multi-arg `jax.vjp`.
  - N30 CUDA toolchain citation: the canonical wording lives in
    `docs/source/jax_gpu_setup.rst`. Project memory was internal, not
    repo-canonical.
  - N31 allocator semantics: `XLA_PYTHON_CLIENT_ALLOCATOR` accepts
    `platform` / `vmm` (BFC when unset); `vmm` uses
    `XLA_CLIENT_MEM_FRACTION`; `cudaMallocAsync` is selected via the
    separate env var `TF_GPU_ALLOCATOR=cuda_malloc_async`. Also:
    `set_backend()` does NOT currently accept a `policy=` keyword, and
    N31 uses explicit keyword extensions rather than a broad policy
    override.
  - N34: dropped transparent in-process GPU-OOM-to-CPU retry of
    compiled JITs; the deliverable is now checkpoint/restart plus
    selective residency, with narrow `jax.default_device(cpu)` use
    only where a fresh compile is permissible.
- Fourth cross-validation (this pass) incorporated current HEAD
  `54b084d29` ("fix: close unsafe lbfgs donation"), correcting the
  round-3 donation guardrail and N22 donation wording to match
  round-2 N3's final "no optimizer-runner donation retained" closeout.

External references used:

- JAX buffer donation: `https://docs.jax.dev/en/latest/buffer_donation.html`
- JAX GPU memory allocation env vars: `https://docs.jax.dev/en/latest/gpu_memory_allocation.html`
- JAX memories and host offloading: `https://docs.jax.dev/en/latest/notebooks/host-offloading.html`
- JAX `jax.default_device`: `https://docs.jax.dev/en/latest/_autosummary/jax.default_device.html`
- JAX configuration options: `https://docs.jax.dev/en/latest/config_options.html`

## Status summary

| Item | Concern | Status | One-line note |
| --- | --- | --- | --- |
| N21 | 1 | Confirmed | Replace `_with_host_status` adjoint sync with on-device NaN sentinel |
| N22 | 1 | Confirmed | Port manual-method LS loop to `lax.while_loop` |
| N23 | 1 | Confirmed | Stop materializing direct-coil scalar before public boundary |
| N24 | 2 | Confirmed | Vectorize `_per_coil_unit_field` over coil axis |
| N25 | 2 | Confirmed opportunity | Investigate `_grouped_field` Python loop; pad-and-stack where homogeneous |
| N26 | 2 | Confirmed | Amortize framed-curve VJP fan-out (26 of 44 calls in lines 84-338) via shared linearization or composed multi-arg VJP |
| N27 | 2 | Confirmed | Pin `jax_default_matmul_precision` for `*_parity` modes |
| N28 | 2 | Confirmed | Make CUDA-determinism enforcement unconditional under CUDA |
| N29 | 2 | Confirmed (doc-only) | Document `jax_gpu_parity` single-device default; reconsider after N30 |
| N30 | 2 | Complete for pre-sharded steady state | Earn real-GPU speedup proof on N11 surface sharding + N12 seed batching |
| N31 | 3 | Confirmed | Plumb JAX GPU memory env vars through `apply_jax_runtime_config`; add explicit `set_backend()` override keywords |
| N32 | 3 | Confirmed | Split `max_dense_jacobian_bytes` into CPU and GPU defaults |
| N33 | 3 | Confirmed | Add selective CPU-residency option for large warm-start factors |
| N34 | 3 | Confirmed (narrow) | Document checkpoint/restart workflow; offer per-instance device pin; no transparent compiled-JIT retargeting |

Concern mapping legend:

- **1**: maximize GPU acceleration (host-device trips).
- **2**: JAX/CUDA strengths leveraged.
- **3**: CPU-fallback configs for GPU memory pressure.

## Recommended sequencing

Five waves. Items within a wave can be parallelized; items across
waves should be sequenced because later waves consume contract
decisions established by earlier waves.

### Wave 1: runtime-policy and contract fixes (low-risk, mechanical)

- N27: matmul-precision pin for parity modes.
- N28: CUDA-determinism unconditional under CUDA.
- N29: document `jax_gpu_parity` single-device default.
- N31: JAX GPU memory env vars plumbed (and `set_backend()` signature
  extension).
- N32: dense-Jacobian budget split.

Rationale: these are runtime / `BackendPolicy` decisions that everything
downstream uses. Getting them right first avoids re-touching code in
Waves 2-4.

### Wave 2: host-sync elimination on the M5 hot path

- N21: `_with_host_status` adjoint sync.
- N22: manual-LS host pulls.
- N23: direct-coil scalar pulls.

Rationale: same code surface (`boozersurface_jax.py` and
`surfaceobjectives_jax.py` adapter glue), same review burden,
landed together for review cost amortization.

### Wave 3: adapter-layer Python-for elimination

- N24: `_per_coil_unit_field` vectorize.
- N25: `_grouped_field` group loop audit.
- N26: framed-curve VJP fan-out collapse.

Rationale: each is independent in code surface but each requires the
same kind of measurement scaffold (HLO probe + steady-state timing).
Parallelize after Wave-2 calibration.

### Wave 4: CPU-fallback ergonomics

- N33: selective CPU residency for warm-start factors.
- N34: checkpoint/restart workflow and per-instance device pin.

Rationale: Wave 1 (N31, N32) ships the runtime knobs; Wave 4 ships
the user-visible API on top, within the JAX-upstream constraint that
compiled JITs cannot be retargeted across platforms.

### Wave 5: multi-GPU GPU-speedup proof

- N30: closed on Perlmutter via
  `docs/jax_multi_gpu_proof_2026-05-19.md`.

## TODO details

The detailed checklists below mirror the round-2 doc idiom: each item
gets Context, Rationale, Implementation, and Acceptance criteria, with
a top-line status checkbox. Items move from `- [ ]` to `- [x]` only
when the acceptance criteria are met and a closeout note is added.

## N21: replace `_with_host_status` adjoint sync while preserving public failure errors

- [ ] Status: confirmed.

### Context

`src/simsopt/geo/boozersurface_jax.py:3522` defines `pack_callbacks`
that wraps `solve_forward_with_status` and `solve_transpose_with_status`
to return `(solution, _host_bool(success))`. Every M5 IFT adjoint
solve in `BoozerResidualJAX.dJ`, `IotasJAX.dJ`, and
`NonQuasiSymmetricRatioJAX.dJ` therefore blocks on at least one scalar
device→host transfer per term per outer iteration. The chain leads
through `_checked_boozer_linear_solve` at
`src/simsopt/geo/surfaceobjectives_jax.py:1887-1901` to a Python
`if`-on-host.

### Rationale

The host bool exists solely to choose between returning the solution
and surfacing a failure. The runtime callback contract can keep the
success indicator as a JAX scalar instead of converting it immediately
inside `BoozerSurfaceJAX`. The public `J()` / `dJ()` boundary is the
place where a failed adjoint solve must become a host-visible error,
matching the CPU Boozer objective contract. A review pass rejected
caching NaN gradients at that public boundary because it can silently
persist non-physical gradients in user-facing objects.

### Implementation

1. Keep runtime solve callbacks returning `(masked_solution,
   success_jax_scalar)` without calling `_host_bool` inside
   `BoozerSurfaceJAX`; the masked value is for device consumers and is
   not a public-gradient fallback.
2. In `pack_callbacks` (line 3522), stop calling `_host_bool`. Return
   `(masked_solution, success_jax_scalar)`. Downstream consumers that
   need the device bool keep it as a JAX scalar; consumers that need a
   Python bool materialize it once at the outermost boundary.
3. In `_checked_boozer_linear_solve` at `surfaceobjectives_jax.py:1887`,
   materialize the status at the public boundary and raise
   `RuntimeError` on failed status instead of caching NaN gradients.
4. Update `BoozerResidualJAX.dJ`, `IotasJAX.dJ`,
   `MajorRadiusJAX.dJ`, and `NonQuasiSymmetricRatioJAX.dJ` to consume
   the checked public-boundary helper.
5. Audit `_solver_diagnostics_payload` at `boozersurface_jax.py:3897`
   for downstream impact: it currently consumes `success` as a host
   bool to format failure metadata; gate behind a host-cached flag.

### Acceptance criteria

- New test in `tests/geo/test_boozersurface_jax.py` runs
  `BoozerResidualJAX.dJ`, `IotasJAX.dJ`, `NonQuasiSymmetricRatioJAX.dJ`
  under `with jax.transfer_guard("disallow"):` on a real fixture; the
  adjoint inner region triggers zero host transfers.
- Failure-injection test: forced failed solve status raises at the
  public Boozer objective gradient boundary, with `failure_category`
  reporting still correct at the public boundary.
- `benchmarks/grouped_adjoint_memory_probe.py` shows steady-state
  cache stability unchanged.
- Single-stage smoke fixture parity holds on `jax_cpu_parity`.

## N22: port manual-method LS loop to `lax.while_loop`

- [ ] Status: confirmed.

### Context

`src/simsopt/geo/boozersurface_jax.py:4475-4506`
(`_run_manual_penalty_least_squares`) executes a host-driven LS loop:
per iteration, three `float(_host_scalar(...))` materializations
(norm, cost, candidate_cost) plus four `_host_all_finite(...)` checks
on `candidate_x`, `candidate_residual`, `candidate_gradient`,
`candidate_normal_matrix`. That is at least seven host stalls per LS
iteration before the next iteration is scheduled.

### Rationale

The traceable LM and private L-BFGS-B paths already model equivalent
state transitions with `lax.while_loop`, `lax.cond`, and in-graph
finiteness checks
(`src/simsopt/geo/optimizer_jax.py:1280,1380,2365,3747`,
`src/simsopt/geo/optimizer_jax_private/_lbfgs.py:247-258`). The
manual method is the laggard. Closing the gap brings the manual
fallback up to the traceable path's GPU efficiency, which matters
because users who hit numerical edge cases sometimes opt into
`method='manual'` for diagnostic transparency, and they should not pay
a 7x host-stall tax.

### Implementation

1. Audit the manual-method LS state pytree. Likely shape: `{x, cost,
   grad, normal_matrix, step, accepted, finite_flags, iter,
   converged}`.
2. Convert the Python loop body to a `lax.while_loop` `body_fn`.
   - Replace `_host_all_finite(...)` with
     `jnp.all(jnp.isfinite(...))` accumulated into `finite_flags` in
     the state.
   - Replace per-iter scalar materialization with state-carried JAX
     scalars.
3. Convert convergence and accept/reject decisions to `cond_fn`
   logic using `lax.cond`.
4. Wrap the loop in `jax.jit` with the appropriate `static_argnames`.
   Do not add optimizer-runner donation in this pass; round-2 N3
   closed with no optimizer-runner donation retained after warning
   audit.
5. Materialize host scalars only at the outermost boundary for
   logger or callback use, guarded by a verbosity flag.

### Acceptance criteria

- New test runs `_run_manual_penalty_least_squares` under
  `jax.transfer_guard("disallow")` on a real BoozerSurface fixture;
  inner region triggers zero host transfers.
- Manual-method regression test: same final iterate within
  `rtol=1e-12` of the pre-port iterate sequence on a deterministic
  fixture.
- Steady-state runtime on `jax_cpu_fast` for manual method shows
  monotonic improvement vs pre-port; gather on
  `benchmarks/tier5_performance_characterization.py`.

## N23: stop materializing direct-coil scalar before public boundary

- [ ] Status: confirmed.

### Context

`src/simsopt/geo/surfaceobjectives_jax.py:2131` and `:2155` apply
`_host_scalar(objective_value)` inside
`_evaluate_direct_coil_objective_value` and
`_value_and_direct_coil_gradient`. `_boozer_solve_observability_payload`
at `:2173-2183` reads JAX scalars on every `_ensure_solved_value_state`
call (fired via `_log_boozer_solve_state` at `:2217` from every
objective `J()`/`dJ()` invocation).

### Rationale

The scalar is only needed at the public `J()` / `dJ()` boundary. The
helpers can return JAX scalars; the wrapper materializes once at the
end. The observability payload can be gated on whether a logger
handler is actually attached, eliminating the routine cost.

### Implementation

1. Change `_evaluate_direct_coil_objective_value` return type from
   `(host_scalar, ...)` to `(jax_scalar, ...)`. Update callers at
   `BoozerResidualJAX._value_and_dJ_by_dcoil_dofs` (`:2502-2520`)
   and the parallel `IotasJAX`, `MajorRadiusJAX`, and
   `NonQuasiSymmetricRatioJAX` paths at `:2625`, `:2686`, and
   `:2795`.
2. Materialize the public scalar exactly once at the
   `_BoozerObjectiveBase.J()` / `compute()` boundary (`:2416-2442`).
3. Wrap `_boozer_solve_observability_payload` in
   `if _booz_solve_observer_active(): ...`; the gate checks
   `logging.getLogger(...).isEnabledFor(...)` or a CLAUDE.md-style
   `SIMSOPT_BOOZER_OBSERVABILITY` env flag.
4. When the gate is closed, skip the `_host_inf_norm` calls at
   `:2175` and `:2177`.

### Acceptance criteria

- New unit test: `_evaluate_direct_coil_objective_value` returns
  `jnp.ndarray` (not Python `float`) on a JAX-typed input.
- `J()` / `dJ()` on M5 wrappers materializes exactly one host scalar
  at the public boundary, verified via
  `jax.transfer_guard("disallow")` over the helper region.
- Observability-disabled run logs zero host transfers in the
  observability payload (assert via counter).

## N24: vectorize `_per_coil_unit_field` over coil axis

- [ ] Status: confirmed.

### Context

`src/simsopt/field/biotsavart_jax_backend.py:198-208`
(`_per_coil_unit_field`) runs `for group: for position:` Python loops
over jitted kernel calls. With `ncoils` coils, this produces `ncoils`
separate kernel launches; compile cache still hits, but launch
overhead grows with `ncoils`.

### Rationale

This adapter glue sits between the device-clean BiotSavartJAX core
and the SIMSOPT Optimizable wrapper. The core already supports a
batched coil axis (`biot_savart_B` is vmap'd over points). Lifting
the coil axis to a `jax.vmap` is the natural completion. Mixed
quadrature is supported by grouping coils by quadrature count then
vmapping within each group.

### Implementation

1. Inspect current call shapes: what does the kernel receive per
   coil, and what does it return?
2. For each quadrature group, stack inputs along a new coil-batch
   axis with `jnp.stack` (already done by `_extract_coil_data_grouped`
   at `:1655-1664`).
3. Replace inner `for position:` loop with `jax.vmap(kernel,
   in_axes=...)` over the coil-batch axis.
4. Sum across groups (the existing group-loop pattern remains;
   tackled separately in N25).
5. Verify the donation-policy boundary: vmap inputs must not be
   caller-owned arrays unless caller-copy is established.

### Acceptance criteria

- `tests/field/test_biotsavart_jax.py` `ncoils` sweep test passes
  with the vmap path.
- New benchmark probe `benchmarks/per_coil_unit_field_vmap_probe.py`
  shows steady-state runtime scales sub-linearly with `ncoils` after
  the change (vs linear before).
- HLO size for the new kernel does not exceed `(coil_axis_size)x` of
  the pre-change single-coil kernel; verified via
  `benchmarks/surface_rz_geometry_hlo_probe.py`-style instrumentation.

## N25: audit `_grouped_field` Python loop; pad-and-stack where homogeneous

- [x] Status: implemented for the heterogeneous production fixture regime.

### Context

`src/simsopt/jax_core/biotsavart.py:734-739` is `_grouped_field`:

```python
def _grouped_field(field_fn, points, coil_arrays):
    g0, gd0, c0 = coil_arrays[0]
    result = field_fn(points, g0, gd0, c0)
    for gammas, gammadashs, currents in coil_arrays[1:]:
        result = result + field_fn(points, gammas, gammadashs, currents)
    return result
```

The Python `for` over groups invokes `field_fn` once per group inside
a single device program. Note: this function is **distinct** from the
sharded sibling in `src/simsopt/jax_core/field.py:204-225`, which
uses `shard_map` + `lax.psum`. The cross-validation pass made the
distinction explicit.

### Rationale

Groups exist because coils have different quadrature counts; equal-
shape coils share a group. Two regimes:

- **Heterogeneous groups (common case)**: shapes differ; vectorization
  across groups requires padding to the largest quadrature count,
  which inflates the kernel and may waste work on small-quad coils.
  Decision depends on the typical distribution.
- **Homogeneous group (all coils share quadrature count)**: the
  Python loop runs once; no benefit to refactoring.

The action is an audit first, not a blanket refactor.

### Implementation

1. Probe production coil sets (Stage 2 banana, single-stage) for
   group-count and shape-uniformity distribution.
2. If a single group covers >80% of coils on the typical fixture:
   make the common path a `vmap` and the rare-group path a Python
   loop. Use `lru_cache` on `(group_count, group_shape_tuple)` to
   keep compile cache stable.
3. If groups are routinely diverse: leave the Python loop as is,
   but mark `_grouped_field` with `@partial(jax.jit, static_argnames=
   ("group_count",))` so that compile cache stays stable across
   coil-set re-configurations with same group structure.
4. Document the chosen regime in CLAUDE.md under "Mixed quadrature
   support" with the measured distribution.

### Acceptance criteria

- Probe report committed under `docs/grouped_field_distribution_probe_*.md`.
- For the chosen refactor regime, steady-state runtime on
  `jax_cpu_fast` shows no regression on heterogeneous fixtures.
- For the common-case homogeneous path, runtime improves by the
  ratio measured in the probe (target: 1.3x or better).

## N26: amortize framed-curve VJP fan-out

- [ ] Status: confirmed.

### Context

`src/simsopt/geo/framedcurve_jax.py` has **44 total `jax.vjp` calls**;
**26 of those cluster in the lines 84-338 block** across five
helpers:

- `_frame_twist_vjps` (lines 84-93): 4 calls.
- `_centroid_torsion_vjps` (lines 181-197): 5 calls.
- `_centroid_binormal_curvature_vjps` (lines 214-238): 5 calls.
- `_frenet_torsion_vjps` (lines 258-288): 6 calls.
- `_frenet_binormal_curvature_vjps` (lines 308-338): 6 calls.

The remaining 18 calls live in adjacent helpers, including 4 in the
block at lines 589-601 and others in subsequent VJP families. Each
`jax.vjp(fn, arg)[1](cotangent)` invocation re-traces the forward
pass plus computes its own pullback. Where the forward computations
share structure (which they do, since they're all derivatives of a
few shared geometric quantities), the forward trace is duplicated.

### Rationale

Two primitives can amortize the shared forward work:

- `jax.linearize(fn, *primals)` runs the forward pass once and
  returns a linearized callable that accepts cotangents. Best when
  the forward `fn` is identical across a group and only the
  cotangents differ.
- Composed multi-arg `jax.vjp(combined_fn, primal1, primal2, ...)`
  packs multiple primal arguments into one trace. Best when the
  helper functions differ slightly per call but share intermediate
  computations that can be hoisted into a wrapping `combined_fn`.

Either primitive reduces traces from 26 down to a handful per
helper; the right choice depends on which primal-argument tuples
share computation in each helper.

### Implementation

1. Inventory the 26 VJP sites in lines 84-338 and the 4 at
   lines 589-601; tag each with its primal-argument tuple and the
   forward expression it differentiates.
2. Group sites whose primal-argument tuples are identical or
   differ only in a single varying argument. Per helper:
   - If the same forward `fn` is called with different cotangents,
     replace with one `jax.linearize` and a per-cotangent loop or
     `jax.vmap` over a stacked cotangent batch.
   - If the forward differs slightly but shares intermediates,
     refactor into a wrapping `combined_fn` and use a single
     multi-arg `jax.vjp`.
3. Validate per helper before moving to the next: each helper's
   output gradients must match within `rtol=1e-12` of the pre-port
   reference on the existing framed-curve unit fixtures.
4. After all five helpers are converted, audit the remaining 14
   VJP calls outside the 84-338 + 589-601 blocks; apply the same
   triage.

### Acceptance criteria

- New benchmark `benchmarks/framedcurve_vjp_scaling.py` reports
  forward-trace count before and after; total trace count reduced
  by >50% across the file (from 44 down to <22).
- Steady-state framed-curve `dJ` wall-time improves by >25% on
  `jax_cpu_fast`.
- No parity regression on framed-curve tests in
  `tests/geo/test_curvexyzfourier*.py` family.

## N27: pin `jax_default_matmul_precision` for `*_parity` modes

- [ ] Status: confirmed.

### Context

`Precision.HIGHEST` is opted into in three sites only:
`src/simsopt/jax_core/biotsavart.py:418`,
`src/simsopt/geo/surfaceobjectives_jax.py:1657-1658`,
`src/simsopt/geo/optimizer_jax_private/_common.py:27-28`. The global
`jax.config.update("jax_default_matmul_precision", "highest")` is
never called in `apply_jax_runtime_config`. Round-2 N7 closed a
fused-contraction path at
`src/simsopt/jax_core/regular_grid_interp.py:659`; that is unrelated
to the global precision pin and does not satisfy this concern.

### Rationale

On CUDA hardware with TF32 default, an unpinned matmul can run at TF32
precision and silently violate parity. The strict byte-identity gate
depends on every implicit matmul matching across CPU and GPU. The
*_parity* modes need a hard pin; the *_fast* modes can keep the JAX
default to allow TF32 speedups where parity is not claimed.

### Implementation

1. In `src/simsopt/backend/runtime.py`, add field
   `matmul_precision: str = "highest"` on `BackendPolicy`.
2. `_MODE_POLICY_DEFAULTS` for `jax_cpu_parity`, `jax_gpu_parity` →
   `"highest"`. For `jax_cpu_fast`, `jax_gpu_fast` → `"default"`
   (JAX's `jax_default_matmul_precision`).
3. In `apply_jax_runtime_config`, call
   `jax.config.update("jax_default_matmul_precision",
   policy.matmul_precision)`.
4. Update `docs/parity_dual_mode_contract_2026-05-08.md` to document
   the precision contract per mode.
5. Audit the existing three explicit-pin sites; keep them for safety
   but document them as defense-in-depth.

### Acceptance criteria

- Subprocess smoke test: `jax_gpu_parity` startup leaves
  `jax.config.read("jax_default_matmul_precision") == "highest"`.
- Strict byte-identity gate passes on
  `benchmarks/single_stage_init_parity.py` from a fresh shell with
  no manual `XLA_FLAGS`-of-precision override.
- `jax_gpu_fast` benchmark shows expected TF32 speedup over
  `jax_gpu_parity` on contraction-heavy kernels (target: 1.5x or
  better on GEMM-dominated paths).

## N28: make CUDA-determinism enforcement unconditional under CUDA

- [ ] Status: confirmed.

### Context

`_validate_cuda_parity_determinism_env` at
`src/simsopt/backend/runtime.py:1700-1713` raises only when
`config.mode == "jax_gpu_parity"` or `strict=True`, and only fires
from `apply_jax_runtime_config()` which only runs eagerly when
`should_eagerly_configure_jax()` returns True
(`runtime.py:1658-1663`). A user with `JAX_PLATFORMS=cuda` but no
`SIMSOPT_BACKEND_MODE*` env gets no validation at all. Worse, if
JAX is imported before `apply_jax_runtime_config` runs, the XLA
flags are no-ops.

### Rationale

CUDA non-determinism is a silent parity-killer. The current
posture protects users who follow the mode-selector contract;
users who route through `JAX_PLATFORMS=cuda` directly are silently
unprotected. The check is cheap and should fire whenever CUDA is the
active platform.

### Implementation

1. Add a pre-import hook in `src/simsopt/__init__.py:34-35` (the
   existing `apply_jax_runtime_config` callsite) that checks for
   CUDA in the environment regardless of mode:
   - `JAX_PLATFORMS` contains `cuda`, OR
   - JAX has already imported and `jax.devices()` reports CUDA.
2. If CUDA detected and `_xla_flags_enable_gpu_determinism` returns
   False:
   - Under `*_parity` mode or `strict=True`: raise `RuntimeError`.
   - Otherwise: emit `RuntimeWarning` advising the flag.
3. If JAX has already imported with non-deterministic flags, raise
   an explanatory error (the flags can no longer take effect).
4. Add a regression test in `tests/subprocess/jax_runtime_cases.py`
   covering the four matrix cells:
   `{cuda, no-cuda} × {flag-set, no-flag-set}`.

### Acceptance criteria

- Subprocess test: `JAX_PLATFORMS=cuda` without
  `SIMSOPT_BACKEND_MODE` emits the warning; with `--strict` raises.
- Subprocess test: `import jax` before `import simsopt` under CUDA
  raises if non-deterministic flags are active.
- `docs/source/jax_gpu_setup.rst` updated with the new enforcement
  posture.

## N29: document `jax_gpu_parity` single-device default

- [ ] Status: confirmed (doc-only until N30 closes).

### Context

`src/simsopt/backend/runtime.py:207-214` (`_MODE_SHARDING_DEFAULTS`):

```
jax_cpu_parity   -> "none"
jax_cpu_fast     -> "none"
jax_gpu_parity   -> "none"   # ← parity GPU is single-device by default
jax_gpu_fast     -> "hybrid"
jax_mps_smoke    -> "none"
```

This is intentional: the strict byte-identity gate cannot tolerate
sharding-introduced reduction-order variation today. Users on a
multi-GPU host who set `SIMSOPT_BACKEND_MODE=jax_gpu_parity`
silently get single-device execution.

### Rationale

The defaults are correct given the parity-ladder constraint. What is
missing is the user-visible signal. Setup docs should explain the
default, the override path, and the parity caveat. After N30 lands
real-GPU speedup for surface sharding (N11 closure) and seed-batch
sharding (N12 closure), the default may move; but only after a real
multi-GPU parity proof. Until then, document only.

### Implementation

1. Add a "Sharding defaults" section to `docs/source/jax_gpu_setup.rst`
   that lists the table above with rationale per mode.
2. Add a note in CLAUDE.md under the "Parity modes" section.
3. At runtime, when the user explicitly requests `jax_gpu_parity`
   on a multi-GPU host, emit an informational log line: "parity GPU
   defaults to single-device; set SIMSOPT_JAX_SHARDING=hybrid to
   opt-in (no parity proof yet)."

### Acceptance criteria

- `docs/source/jax_gpu_setup.rst` and CLAUDE.md updated.
- Single subprocess test covers the log emission on a multi-GPU
  forced-CPU-multi-device proxy fixture
  (`tests/subprocess/jax_runtime_cases.py` style).

## N30: earn real-GPU speedup proof on N11 and N12 sharding

- [x] Status: complete for the pre-sharded steady-state contract.

### Context

Round-2 N11 closed the surface-axis sharding of `integral_BdotN`
with CPU-forced multi-device equivalence and HLO collective proof.
Round-2 N12 closed seed-batch sharding the same way. Both closeouts
explicitly do **not** claim real-GPU speedup.

The current CUDA runtime compatibility block is documented in
`docs/source/jax_gpu_setup.rst`. That section is the authoritative repo
source for CUDA userspace/toolchain mismatches; resolve from there.

### Rationale

A CPU-multi-device proof is a logical-equivalence check, not a
performance check. The collective patterns may have HBM-bandwidth or
PCIe-bottleneck issues that only surface on real GPUs. Until the
real-GPU lift is measured and recorded, the multi-GPU promise is
unfulfilled.

### Implementation

1. Resolve CUDA toolchain mismatches documented in
   `docs/source/jax_gpu_setup.rst`: rebuild `jaxlib` against the host
   CUDA, or pin a known-good jaxlib for the target H100 configuration.
2. Run a 1-vs-2-vs-4-GPU sweep for `integral_BdotN_surface_sharded`
   (`src/simsopt/jax_core/integral_bdotn.py:240`). Measure
   wall-time, HBM peak, HLO collective bytes.
3. Run the same sweep for seed-batch scoring
   (`src/simsopt/geo/surfaceobjectives_jax.py:5088`) at
   production-relevant seed counts.
4. Write `docs/jax_multi_gpu_proof_2026-05-19.md` recording the
   results.
5. If acceptable speedups are demonstrated, reconsider the
   `jax_gpu_parity` sharding default (N29 caveat); otherwise document
   the bottleneck and open a follow-up round-4 entry.

### Acceptance criteria

- Both probes show >1.5x speedup at 2 GPUs and >2.5x at 4 GPUs
  versus single-GPU baseline on the same fixture.
- HBM peak per device reported and compared against 1x baseline.
- Parity preserved (no regression on
  `benchmarks/single_stage_init_parity.py` under the proven sharding
  config).

### Closeout

Perlmutter jobs `53168131` (`debug`) and `53168132` (`regular`) passed the
1 / 2 / 4-GPU pre-sharded steady-state proof. The regular job measured
`integral_BdotN_surface_sharded` at 2.03x on 2 GPUs and 3.87x on 4 GPUs, with
`NamedSharding`, mesh-axis `d`, and one all-reduce in the lowered HLO for the
multi-GPU rows. Seed-batch scoring measured 1.93x on 2 GPUs and 3.78x on 4
GPUs. Peak GPU memory per visible device and baseline-subtracted deltas are
recorded in `docs/jax_multi_gpu_proof_2026-05-19.md`.

Follow-up job `53170493` (`debug`, four A100 GPUs) passed
`benchmarks/single_stage_init_parity.py` with active point sharding after the
private optimizer and Boozer penalty geometry active-replicated-placement fixes.
It wrote `single_stage_cuda_init.json` with `"passed": true`, reported
`|iota diff|=0.00e+00`, volume relative difference `0.00e+00`, field-error
relative difference `2.51e-16`, and completed in 7:34 Slurm elapsed
(`7:27.85` by `/usr/bin/time`) with Slurm batch MaxRSS `7905296K`.

## N31: plumb JAX GPU memory env vars through `apply_jax_runtime_config`

- [ ] Status: confirmed.

### Context

The following env vars are not set or asserted by
`src/simsopt/backend/runtime.py:1716-1737`:

- `XLA_PYTHON_CLIENT_PREALLOCATE` (bool): controls eager 75%
  preallocation.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` (float in `(0, 1]`): bounds total
  preallocated fraction for the default BFC allocator when
  preallocation is on.
- `XLA_PYTHON_CLIENT_ALLOCATOR`: per official JAX GPU
  memory-allocation docs, accepts `platform` or `vmm`; default
  (unset) selects BFC.
- `XLA_CLIENT_MEM_FRACTION`: per official JAX GPU memory-allocation
  docs, controls the fraction for the experimental `vmm` allocator.
- `TF_GPU_ALLOCATOR=cuda_malloc_async`: this is a **separate** env
  var that selects CUDA's asynchronous allocator. It is NOT a value
  of `XLA_PYTHON_CLIENT_ALLOCATOR`.

Current production practice: CUDA entrypoints hand-export
`XLA_PYTHON_CLIENT_PREALLOCATE=false` in `.github/workflows/jax_smoke.yml`
and `repo_bootstrap.py`. Users who skip this hand-setting inherit JAX's
default 75% preallocation.

### Rationale

GPU OOM is the primary memory-pressure pain. Users should not need
to know to hand-set these env vars; the runtime should set them based
on the active mode and policy. Per the JAX GPU-memory-allocation
docs, these env vars MUST be set before `import jax` for the XLA
client to honor them.

### Implementation

1. Add to `BackendPolicy` in `src/simsopt/backend/runtime.py`:
   - `xla_gpu_preallocate: bool | None = None`
   - `xla_gpu_mem_fraction: float | None = None`
     (semantic fraction knob; apply to
     `XLA_PYTHON_CLIENT_MEM_FRACTION` for default/BFC allocation and
     to `XLA_CLIENT_MEM_FRACTION` when `xla_gpu_allocator == "vmm"`).
   - `xla_gpu_allocator: Literal["platform", "vmm"] | None = None`
     (None means do-not-set, falling back to BFC).
   - `tf_gpu_allocator: Literal["cuda_malloc_async"] | None = None`
     (separate seam for the TF allocator env var; None means
     do-not-set).
2. `_MODE_POLICY_DEFAULTS` for `jax_gpu_*` modes:
   - `xla_gpu_preallocate=False`,
   - `xla_gpu_allocator=None` (leave BFC; users who want `platform`
     or `vmm` opt in via env override),
   - `xla_gpu_mem_fraction=None` (None means do-not-set, falling
     back to JAX's allocator defaults),
   - `tf_gpu_allocator=None`.
3. In `apply_jax_runtime_config`, set the env vars via
   `os.environ[...]` before the local `import jax` and before the
   `jax.config.update("jax_platforms", ...)` call at
   `runtime.py:1725-1728`.
4. If JAX is already imported, raise an explanatory error rather
   than silently accept no-op env writes.
5. Env-var overrides:
   - `SIMSOPT_JAX_GPU_PREALLOCATE` (`true`/`false`).
   - `SIMSOPT_JAX_GPU_MEM_FRACTION` (float in `(0, 1]`).
   - `SIMSOPT_JAX_GPU_ALLOCATOR` (`platform`/`vmm`).
   - `SIMSOPT_TF_GPU_ALLOCATOR` (`cuda_malloc_async`).
6. **`set_backend()` signature extension**. The current signature
   at `runtime.py:1740-1748` is
   `(mode, *, strict, debug_nans, transfer_guard,
   compilation_cache_dir, configure_runtime)` — no `policy=`
   parameter. Do not add a broad `policy=` override. This item adds
   explicit typed keywords for the four new knobs:
   `xla_gpu_preallocate`, `xla_gpu_mem_fraction`,
   `xla_gpu_allocator`, and `tf_gpu_allocator`. Plumb them through
   `_config_from_mode`, and document them in the docstring.
7. Resolution priority (after the extension):
   explicit `set_backend()` keyword > `SIMSOPT_*` env > mode-derived
   `_MODE_POLICY_DEFAULTS`.

### Acceptance criteria

- Subprocess test under `jax_gpu_fast`: confirms
  `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"` after
  `apply_jax_runtime_config()`.
- Subprocess test under `SIMSOPT_JAX_GPU_MEM_FRACTION=0.5`:
  confirms env propagation; downstream `jax.devices()[0].memory_stats()`
  reports halved bytes_in_use_limit (or equivalent JAX-version-
  specific metric).
- Subprocess test under `SIMSOPT_JAX_GPU_ALLOCATOR=vmm` and
  `SIMSOPT_JAX_GPU_MEM_FRACTION=0.5`: confirms
  `XLA_PYTHON_CLIENT_ALLOCATOR=vmm` and
  `XLA_CLIENT_MEM_FRACTION=0.5` are set before JAX initialization.
- Subprocess test under `SIMSOPT_TF_GPU_ALLOCATOR=cuda_malloc_async`:
  confirms env propagation; XLA log line on JAX import names the
  cudaMallocAsync allocator.
- `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`
  on GPU works from a fresh shell without manual
  `XLA_PYTHON_CLIENT_PREALLOCATE` export.
- New explicit `set_backend(...)` keyword arguments accept all four
  fields without breaking existing callers.

## N32: split `max_dense_jacobian_bytes` into CPU and GPU defaults

- [ ] Status: confirmed.

### Context

`src/simsopt/geo/boozersurface_jax.py:3073`:
`_DEFAULT_MAX_DENSE_JACOBIAN_BYTES = 512 * 1024 * 1024`. This single
ceiling drives both LS-lane `max_dense_linearization_bytes`
(`:3087`) and exact-lane `max_dense_jacobian_bytes` (`:3097`)
defaults. There is no env override and no per-platform split. The
ceiling triggers `failure_category="scaling_limit"` at
`src/simsopt/geo/optimizer_jax.py:2652,2786,2876`.

### Rationale

CPU systems typically have 64-256 GB of system RAM; GPU systems
(H100, A100) have 24-80 GB HBM. The 512 MB ceiling means GPU users
hit `scaling_limit` at N ≈ 8192 (`N² × 8 = 512 MB`), and CPU users
hit the same point despite having 100x more memory available. Single
ceiling is wrong for both: too generous on GPU at smaller HBM cards,
too restrictive on CPU.

### Implementation

1. Replace `_DEFAULT_MAX_DENSE_JACOBIAN_BYTES` with platform-aware
   defaults:
   - `_DEFAULT_MAX_DENSE_JACOBIAN_BYTES_CPU = 4 * 1024 * 1024 * 1024`
     (~4 GB, N ≈ 23000).
   - `_DEFAULT_MAX_DENSE_JACOBIAN_BYTES_GPU = 256 * 1024 * 1024`
     (~256 MB, N ≈ 5800).
2. Add `BackendPolicy.max_dense_jacobian_bytes: int | None` resolved
   from the active platform at construction time.
3. Env overrides:
   - `SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_CPU`
   - `SIMSOPT_MAX_DENSE_JACOBIAN_BYTES_GPU`
4. Caller `options={"max_dense_jacobian_bytes": ...}` continues to
   override everything (per CLAUDE.md exact-scaling contract).
5. Update CLAUDE.md "Exact Boozer scaling-limit contract" to
   reflect the platform split.

### Acceptance criteria

- New test: GPU mode without override sees `scaling_limit` at
  `N ≈ 5800`; CPU mode at `N ≈ 23000`.
- Backward compatibility test: explicit `options={"max_dense_jacobian_bytes": 512*1024*1024}`
  still produces the pre-split behavior.
- Documentation updated in CLAUDE.md.

## N33: selective CPU-residency for warm-start linearization factors

- [ ] Status: confirmed.

### Context

All 16+ `jax.device_put` call sites in `src/simsopt/` stage TO the
active device. None target `jax.devices('cpu')[0]` for host
residency. The LS-lane `(P, L, U)` factors stored at
`boozersurface_jax.py:3609-3669` and consumed downstream at
`surfaceobjectives_jax.py:4660-4666, 6075-6081` live on the active
device. For N = 8192, dense `P`, `L`, and `U` arrays occupy
approximately `3 × N² × 8 = 1.5 GiB` per LS solver instance when all
three are materialized.

### Rationale

Per JAX host-offloading docs: explicit `jax.device_put(..., device=
jax.devices('cpu')[0])` is supported and preserves the array's
logical identity. For workflows where the factors are reused
infrequently relative to the per-call GPU compute, host residency
frees up to `3 × N² × 8` HBM bytes per solver, depending on whether
the runtime carries dense `P`, `L`, and `U` or a packed solve
representation. The cost is a host→device transfer when the factors
are consumed. Whether this is a win depends on the call ratio;
offering it as an option is independently useful.

### Implementation

1. Add `BoozerSurfaceJAX(linearization_residency:
   Literal["device", "host"] = "device")` option.
2. When `"host"`, in the LS-lane factor materialization at
   `boozersurface_jax.py:3609-3669`, append
   `jax.device_put(factor, device=jax.devices('cpu')[0])` after the
   factorization.
3. At each consumption site (`surfaceobjectives_jax.py:4660-4666`
   and `:6075-6081`), re-stage on demand via `jax.device_put(...,
   device=jax.devices()[0])` (active device).
4. Record per-instance residency in
   `BoozerSurfaceJAX.get_adjoint_runtime_state()` for observability.
5. Measure: `benchmarks/grouped_adjoint_memory_probe.py` extension
   that compares device vs host residency on the same fixture.

### Acceptance criteria

- New test instantiates two `BoozerSurfaceJAX` solvers in the same
  process with `linearization_residency="device"` and `"host"`;
  both produce equal adjoint gradients within `rtol=1e-12`.
- Memory probe reports expected bytes from the actual factor
  representation (`P,L,U` dense triples up to `3 × N² × 8`; packed
  forms lower) and shows host-residency frees that amount from
  device memory.
- Per-solve transfer cost recorded in the probe report.

## N34: checkpoint/restart workflow and per-instance device pin

- [ ] Status: confirmed (narrow).

### Context

JAX's backend selection is process-import-locked: once
`jax.devices()` is queried, the platform set is fixed. A JIT
compiled against GPU cannot be retargeted to CPU. There is no
in-process `with backend("cpu"):` context manager that retargets
an existing compiled program. The CPU fallback path today is: set
env var → restart Python. There is no per-object or per-call CPU
routing for compiled work.

### Rationale

Full mode-switch is constrained by JAX upstream. What IS possible:

- **Data-residency switch** via explicit `jax.device_put` to a
  specific device (CPU or GPU). This is what N33 uses for
  warm-start factors and what JAX host-offloading docs describe.
- **Per-instance device pin** for new objects constructed and first
  evaluated under a `with jax.default_device(jax.devices('cpu')[0]):`
  block. Arrays and computations created inside that block default to
  CPU; existing GPU-compiled JITs are unaffected. This admits a
  "build a CPU-resident BoozerSurfaceJAX alongside a GPU-resident one"
  pattern, but pays a compile-cache miss on first call.
- **Documented checkpoint-and-restart workflow**: save solver state
  to disk, restart with a `jax_cpu_*` mode env var, load state,
  resume. This is the realistic OOM-recovery path for users
  already mid-iteration on GPU.

This item delivers all three and explicitly rules out transparent
per-call retargeting of compiled JITs.

### Implementation

1. Document the checkpoint-and-restart workflow in
   `docs/source/jax_gpu_setup.rst` under a new "OOM Recovery"
   section:
   - Save solver state through the existing SIMSOPT artifact path
     (`Optimizable.save()` / `load()`; for Stage 2, the
     `biot_savart_opt.json` and adjacent run artifacts used by
     `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py`).
   - Restart with `SIMSOPT_BACKEND_MODE=jax_cpu_fast` or
     `jax_cpu_parity`.
   - Load state and resume.
2. Add a small helper
   `simsopt.backend.runtime.with_cpu_device_for_construction()` that
   returns a context manager wrapping
   `jax.default_device(jax.devices('cpu')[0])`. Document that arrays
   and computations created inside the block default to CPU and that
   GPU-compiled siblings remain GPU-resident.
3. Build on N33: when a user constructs a `BoozerSurfaceJAX` inside
   the helper context, the `linearization_residency` option may be
   set to `"device"` or `"host"`; the active device is now CPU, so
   `"device"` and `"host"` coincide.
4. Do NOT implement transparent OOM-to-CPU retry of existing
   compiled JITs. Such a retry would require recompilation against
   a different platform, which JAX does not support without
   destructing the previously compiled cache; the user-visible
   behavior would surprise (silent recompile, large latency, cache
   thrash on retry).

### Acceptance criteria

- Documentation includes a worked checkpoint-and-restart example
  in `docs/source/jax_gpu_setup.rst`.
- New test instantiates a CPU-pinned and a GPU-routed
  `BoozerSurfaceJAX` in the same process; both work; compile cache
  records two separate entries.
- No public API regression: existing callers without the new
  helper or option are unaffected.
- Explicit doc paragraph stating that transparent OOM-to-CPU retry
  is out of scope and why.

## Execution update (2026-05-18 local implementation pass)

| Item | Implementation state | Evidence boundary |
| --- | --- | --- |
| N21 | Source implemented: runtime solve callbacks keep success as a JAX scalar; public Boozer objective gradients raise on failed solve status rather than caching NaN gradients. | Local unit proof: `tests/geo/test_surface_objectives_jax.py::test_checked_boozer_linear_solve_uses_public_status_boundary`, `::test_checked_boozer_linear_solve_raises_on_failed_status`. Real-fixture strict-transfer proof: `tests/integration/test_single_stage_jax_cpu_reference.py::TestCompositeObjective::test_public_wrapper_dj_boundaries_allow_strict_transfer_guard_real_fixture`. |
| N22 | Source implemented: manual LS compatibility loop now uses `jax.lax.while_loop` with device-resident cost, norm, finite, damping, and accept/reject state. | Local proof: `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_manual_ls_api_increases_damping_after_worsening_trial`, `::test_public_manual_ls_loop_runs_under_strict_transfer_guard`. Benchmark improvement not recorded. |
| N23 | Source implemented: direct-coil value helpers return JAX scalars; public wrapper boundaries materialize scalar values once; cached-solve observability skips norm host reads unless debug/env/failure logging is active. | Local proof: `tests/geo/test_surface_objectives_jax.py::test_direct_coil_value_helpers_keep_value_as_jax_scalar`. Transfer-count counter and real wrapper sweep still not recorded. |
| N24 | Source implemented: `_per_coil_unit_field` now vmaps over coils inside each quadrature group and only loops over groups/order reconstruction. | Local proof: `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppCoilCurrentParity::test_per_coil_unit_field_vectorizes_within_quadrature_group`. Scaling benchmark/HLO report not recorded. |
| N25 | Source/docs implemented: the production fixture probe found two heterogeneous groups, `(20, 15, 3)` and `(10, 128, 3)`, so `_grouped_field` keeps per-group accumulation and now uses a JIT boundary keyed by static field function and group count instead of padding groups. | Local proof: `docs/grouped_field_distribution_probe_2026-05-18.md` is intent-to-add tracked in this slice and records the fixture distribution plus local CPU timing. `tests/field/test_biotsavart_jax.py::TestBiotSavartJaxAnalytical::test_grouped_biot_savart_B_jit_handles_mixed_quadrature_groups` covers the mixed-quadrature JIT path. Probe measured 1.259x manual-loop time over JIT-keyed helper time on local CPU. |
| N26 | Source implemented: `src/simsopt/geo/framedcurve_jax.py` now uses composed multi-arg VJPs; `rg "jax\\.vjp" src/simsopt/geo/framedcurve_jax.py \| wc -l` reports 9 sites, below the <22 target. | Local proof: `tests/geo/test_framedcurve_jax_wrappers_item18.py -q`, `tests/geo/test_curvexyzfouriersymmetries_spec_jax.py -q`. Dedicated wall-time benchmark not recorded. |
| N27 | Source implemented: `BackendPolicy.matmul_precision` pins `highest` for parity modes and `apply_jax_runtime_config()` applies it. | Local proof: `tests/test_backend.py -q` and an interactive `jax_cpu_parity` config probe. CUDA TF32 speedup comparison not recorded. |
| N28 | Source implemented: CUDA determinism validation now runs for direct CUDA environment selection and warns/raises by strictness. | Local proof: `tests/test_backend.py -q`. CUDA pre-import subprocess matrix partially covered by import-smoke; real CUDA run not recorded. |
| N29 | Source/docs implemented: setup docs and `CLAUDE.md` document single-device `jax_gpu_parity`; runtime emits an info log for multi-device parity default. | Local proof: `tests/test_backend.py -q`. Real multi-GPU proof is now recorded under N30. |
| N30 | Closed for the pre-sharded steady-state contract. | Perlmutter jobs `53168131` and `53168132`; `docs/jax_multi_gpu_proof_2026-05-19.md` records 1 / 2 / 4-GPU wall time, peak GPU memory, active `NamedSharding`, and HLO collective evidence for `integral_BdotN_surface_sharded` plus seed-batch scoring. |
| N31 | Source/docs implemented: runtime owns JAX/XLA GPU memory env policy, `SIMSOPT_*` overrides, and explicit `set_backend()` kwargs. | Local proof: `tests/test_backend.py -q`; `tests/test_jax_import_smoke.py::test_import_package_root_without_generated_version_file -q` after updating the raw-source stub. Real allocator log/memory-limit proof not recorded. |
| N32 | Source/docs implemented: dense-Jacobian defaults are resolved through `BackendPolicy` with CPU/GPU env overrides and constructor-level explicit option precedence. | Local proof: `tests/test_backend.py -q`, `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_ls_surface_exact_newton_has_default_dense_jacobian_ceiling`. Large-N scaling-limit fixture not recorded. |
| N33 | Partially implemented: `linearization_residency={"device","host"}` is accepted and dense factors can be stored on CPU and restaged for runtime solve callbacks. | Local proof: `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_linearization_residency_host_places_dense_factors_on_cpu`; dual-instance solve/VJP parity: `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_linearization_residency_dual_instance_gradient_path_matches`. Production memory probe is still missing. |
| N34 | Partially implemented: `with_cpu_device_for_construction()` helper exported; GPU OOM docs prescribe checkpoint/restart and explicitly reject transparent compiled-JIT retargeting. | Local proof: `tests/test_backend.py::test_with_cpu_device_for_construction_uses_real_jax_cpu_default_device`; dual-instance cache observation: `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_linearization_residency_dual_instance_gradient_path_matches`. Worked command-level restart example and real GPU hardware proof remain open. |

Validation commands run in this pass:

- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/test_backend.py -q` -> 106 passed, 2 expected CUDA-determinism warnings.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/test_jax_import_smoke.py -q` -> 110 passed, 11 skipped on this CPU-only host.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/geo/test_surface_objectives_jax.py::test_checked_boozer_linear_solve_uses_public_status_boundary tests/geo/test_surface_objectives_jax.py::test_checked_boozer_linear_solve_raises_on_failed_status tests/geo/test_surface_objectives_jax.py::test_checked_boozer_linear_solve_rejects_statusless_solver tests/geo/test_surface_objectives_jax.py::test_direct_coil_value_helpers_keep_value_as_jax_scalar -q` -> 4 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_manual_ls_api_increases_damping_after_worsening_trial tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_public_manual_ls_loop_runs_under_strict_transfer_guard tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_linearization_residency_host_places_dense_factors_on_cpu tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_ls_surface_exact_newton_has_default_dense_jacobian_ceiling -q` -> 4 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_result_dict_keys tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_result_materializes_dense_plu_when_not_verbose tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_linearization_residency_host_places_dense_factors_on_cpu -q` -> 3 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/field/test_biotsavart_jax.py::TestBiotSavartJaxCppCoilCurrentParity::test_per_coil_unit_field_vectorizes_within_quadrature_group -q` -> 1 passed.

Additional 2026-05-19 review validation:

- `python -m pytest tests/integration/test_single_stage_jax_cpu_reference.py::TestCompositeObjective::test_public_wrapper_dj_boundaries_allow_strict_transfer_guard_real_fixture tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_linearization_residency_dual_instance_gradient_path_matches tests/test_backend.py::test_with_cpu_device_for_construction_uses_real_jax_cpu_default_device -q` -> 3 passed.
- `python -m pytest tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_get_adjoint_runtime_state_status_stays_jax_scalar_until_public_boundary tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXExactPath::test_exact_linearization_residency_dual_instance_gradient_path_matches tests/test_backend.py::test_with_cpu_device_for_construction_uses_real_jax_cpu_default_device -q` -> 3 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/field/test_biotsavart_jax.py::TestBiotSavartJaxAnalytical::test_grouped_biot_savart_A_host_helper_matches_dense_kernel tests/field/test_biotsavart_jax.py::TestBiotSavartJaxAnalytical::test_grouped_biot_savart_B_jit_handles_mixed_quadrature_groups -q` -> 2 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/field/test_biotsavart_jax_parity.py::TestGroupedBiotSavartGradient::test_mixed_quad_gradient_fd -q` -> 1 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/integration/test_single_stage_physics_parity.py::test_single_stage_subprocess_env_preserves_existing_xla_flags tests/test_benchmark_helpers.py::test_repo_pythonpath_env_bundled_cuda_clears_local_toolchain_overrides tests/test_benchmark_helpers.py::test_repo_pythonpath_env_replaces_stale_cuda_determinism_flag tests/test_benchmark_helpers.py::test_build_provenance_includes_compilation_cache_metadata tests/test_benchmark_helpers.py::test_single_stage_init_case_threads_phase1_diagnostic_flags_and_env tests/test_benchmark_helpers.py::test_gpu_parity_workflow_enforces_strict_transfer_guard_contract tests/test_benchmark_helpers.py::test_gpu_parity_workflow_adds_full_suite_disallow_lane tests/test_benchmark_helpers.py::test_smoke_workflow_adds_cuda_e2e_target_lane_gate tests/test_benchmark_helpers.py::test_smoke_workflow_adds_cuda_strict_transfer_guard_pytest_lane -q` -> 9 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/geo/test_framedcurve_jax_wrappers_item18.py -q` -> 5 passed.
- `PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python -m pytest tests/geo/test_curvexyzfouriersymmetries_spec_jax.py -q` -> 23 passed.
- `.conda/jax/bin/python -m py_compile src/simsopt/backend/runtime.py src/simsopt/geo/boozersurface_jax.py src/simsopt/geo/surfaceobjectives_jax.py src/simsopt/jax_core/biotsavart.py benchmarks/validation_ladder_common.py tests/conftest.py tests/test_backend.py tests/test_benchmark_helpers.py tests/geo/test_boozersurface_jax.py tests/geo/test_surface_objectives_jax.py tests/field/test_biotsavart_jax.py` -> passed.
- `git diff --check -- <review-touched runtime/Boozer/Biot-Savart/docs files>` -> passed.

## Reporting and closeout

When an item lands, append a one-line closeout note under
"Execution update" (added to this doc at closeout time, mirroring
round-2's closeout table at lines 159-187 of
`docs/jax_native_round2_performance_todos_2026-05-18.md`). The note
must record:

- Evidence boundary (e.g., "CPU forced multi-device proof; real GPU
  speedup not claimed") if the closure is partial.
- Acceptance criteria met checkboxes (link to test or benchmark).
- Any policy caveats discovered during implementation.

A round-3 closeout commit lands the doc update plus the
implementation diff under the same commit, mirroring `f5411c412`.
