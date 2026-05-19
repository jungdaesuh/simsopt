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
N30 (Runpod toolchain citation), N31 (allocator env vars and proposed
API surface), and N34 (no transparent in-process platform retargeting).

## Purpose

Track the performance and ergonomics residuals that remain after the
round-2 closeout. Three user concerns drive the prioritization:

1. **Maximize GPU acceleration** — minimize host↔device transfers in
   production hot paths.
2. **Leverage JAX/CUDA strengths** — `jit`, `vmap`, donation, sharding,
   precision contracts, deterministic enforcement.
3. **Provide CPU-fallback configs that relieve GPU memory pressure**
   from full compilation — runtime-level XLA memory flags, per-platform
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
2. **Donation policy.** Per round-2 N3 closeout, donation is allowed
   only on internal buffers. The canonical pattern is the caller-copy
   helpers at `src/simsopt/geo/surfaceobjectives_jax.py:4810-4845`:
   the host or boundary wrapper invokes `.copy()` on the caller input
   before passing to a `donate_argnums`-jitted callable. Any new
   donation site must follow this pattern. JAX buffer-donation docs
   are the upstream reference.
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
  - N30 Runpod toolchain citation: the canonical wording lives at
    `docs/source/jax_gpu_setup.rst:421-466` ("Exact JAX 0.9.2 on the
    stock CUDA 12.4 Runpod image needed CUDA toolkit 12.9", "CUDA
    userspace/toolchain mismatches"). Project memory was internal,
    not repo-canonical.
  - N31 allocator semantics: `XLA_PYTHON_CLIENT_ALLOCATOR` accepts
    `platform` / `vmm` (BFC when unset); `cudaMallocAsync` is
    selected via the separate env var `TF_GPU_ALLOCATOR=cuda_malloc_async`.
    Also: `set_backend()` does NOT currently accept a `policy=`
    keyword; an extension is proposed as part of N31.
  - N34: dropped transparent in-process GPU-OOM-to-CPU retry of
    compiled JITs; the deliverable is now checkpoint/restart plus
    selective residency, with narrow `jax.default_device(cpu)` use
    only where a fresh compile is permissible.

External references used:

- JAX buffer donation: `https://github.com/google/jax/blob/main/docs/buffer_donation.md`
- JAX GPU memory allocation env vars: `https://github.com/google/jax/blob/main/docs/gpu_memory_allocation.rst`
- JAX host offloading / explicit `device_put` sharding: `https://github.com/google/jax/blob/main/docs/notebooks/host-offloading.md`

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
| N30 | 2 | Hardware-gated | Earn real-GPU speedup proof on N11 surface sharding + N12 seed batching |
| N31 | 3 | Confirmed | Plumb `XLA_PYTHON_CLIENT_*` and `TF_GPU_ALLOCATOR` through `apply_jax_runtime_config`; extend `set_backend()` signature |
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
- N31: XLA memory flags plumbed (and `set_backend()` signature
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

- N30: hardware-gated. Drops in once the Runpod CUDA toolchain
  mismatch documented at `docs/source/jax_gpu_setup.rst:421-466` is
  resolved.

## TODO details

The detailed checklists below mirror the round-2 doc idiom: each item
gets Context, Rationale, Implementation, and Acceptance criteria, with
a top-line status checkbox. Items move from `- [ ]` to `- [x]` only
when the acceptance criteria are met and a closeout note is added.

## N21: replace `_with_host_status` adjoint sync with on-device NaN sentinel

- [ ] Status: confirmed.

### Context

`src/simsopt/geo/boozersurface_jax.py:3522` defines `pack_callbacks`
that wraps `solve_forward_with_status` and `solve_transpose_with_status`
to return `(solution, _host_bool(success))`. Every M5 IFT adjoint
solve in `BoozerResidualJAX.dJ`, `IotasJAX.dJ`, and
`NonQuasiSymmetricRatioJAX.dJ` therefore blocks on at least one scalar
device→host transfer per term per outer iteration. The chain leads
through `_checked_boozer_linear_solve` at
`src/simsopt/geo/surfaceobjectives_jax.py:1881-1894` to a Python
`if`-on-host.

### Rationale

The host bool exists solely to choose between returning the solution
and surfacing a failure. JAX supports the same dispatch via
`jax.lax.cond` on a device-resident success indicator, or by
returning NaN-filled output on failure and detecting with
`jnp.isfinite(...).all()` at the next boundary. The downstream
consumer that actually needs the host scalar is the public
`J()` / `dJ()` boundary or the failure-reporting path, not the
inner adjoint loop. Per CLAUDE.md "Adjoint / warm-start operator
solves" rule: "A successful traceable forward solve with a failed
adjoint solve must surface a non-finite gradient, not a finite
direct-gradient or failure-penalty fallback." The NaN-sentinel
approach is consistent with that rule.

### Implementation

1. Introduce `_solve_with_nan_on_failure(solver, *args)` that calls
   `solver` and uses `jax.lax.cond(success, lambda s: s, lambda s:
   jnp.where(False, s, jnp.nan), solution)` to mask outputs on
   failure. Place near other private helpers in
   `boozersurface_jax.py`.
2. In `pack_callbacks` (line 3522), stop calling `_host_bool`. Return
   `(masked_solution, success_jax_scalar)`. Downstream consumers that
   need the device bool keep it as a JAX scalar; consumers that need
   a Python bool materialize it once at the outermost boundary.
3. In `_checked_boozer_linear_solve` at `surfaceobjectives_jax.py:1881`,
   replace `if not _host_bool(success): ...` with a `jnp.isfinite`
   check carried in the IFT adjoint state.
4. Update `BoozerResidualJAX.dJ`, `IotasJAX.dJ`,
   `NonQuasiSymmetricRatioJAX.dJ` to materialize success only at the
   public boundary, where it feeds into failure reporting.
5. Audit `_solver_diagnostics_payload` at `boozersurface_jax.py:3897`
   for downstream impact: it currently consumes `success` as a host
   bool to format failure metadata; gate behind a host-cached flag.

### Acceptance criteria

- New test in `tests/geo/test_boozersurface_jax.py` runs
  `BoozerResidualJAX.dJ`, `IotasJAX.dJ`, `NonQuasiSymmetricRatioJAX.dJ`
  under `with jax.transfer_guard("disallow"):` on a real fixture; the
  adjoint inner region triggers zero host transfers.
- Failure-injection test: forced singular LS at a known iterate
  surfaces non-finite gradient (the CLAUDE.md rule), with
  `failure_category` reporting still correct at the public boundary.
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

The LM / L-BFGS traceable paths
(`src/simsopt/geo/optimizer_jax.py:1436`,
`src/simsopt/geo/optimizer_jax_private/_lbfgs.py:172`) already model
the equivalent logic with `lax.while_loop`, `lax.cond`, and in-graph
finiteness checks. The manual method is the laggard. Closing the gap
brings the manual fallback up to the traceable path's GPU efficiency,
which matters because users who hit numerical edge cases sometimes
opt into `method='manual'` for diagnostic transparency, and they
should not pay a 7x host-stall tax.

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
4. Wrap the loop in `jax.jit` with the appropriate
   `static_argnames`. Donation may apply to the state pytree if
   caller-copy invariant is established; defer to Wave 4 audit.
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
  `tier5_performance_characterization.py`.

## N23: stop materializing direct-coil scalar before public boundary

- [ ] Status: confirmed.

### Context

`src/simsopt/geo/surfaceobjectives_jax.py:2131` and `:2155` apply
`_host_scalar(objective_value)` inside
`_evaluate_direct_coil_objective_value` and
`_value_and_direct_coil_gradient`. `_boozer_solve_observability_payload`
at `:2174` reads JAX scalars on every `_ensure_solved_value_state`
call (fired via `_log_boozer_solve_state` at `:2211` from every
objective `J()`/`dJ()` invocation).

### Rationale

The scalar is only needed at the public `J()` / `dJ()` boundary. The
helpers can return JAX scalars; the wrapper materializes once at the
end. The observability payload can be gated on whether a logger
handler is actually attached, eliminating the routine cost.

### Implementation

1. Change `_evaluate_direct_coil_objective_value` return type from
   `(host_scalar, ...)` to `(jax_scalar, ...)`. Update callers at
   `BoozerResidualJAX._value_and_dJ_by_dcoil_dofs` (line 2510) and
   the parallel `IotasJAX` / `NonQuasiSymmetricRatioJAX` calls at
   `:2620, :2682, :2798`.
2. Materialize the public scalar exactly once at the
   `J()` / `dJ()` return statement (lines 765, 773).
3. Wrap `_boozer_solve_observability_payload` in
   `if _booz_solve_observer_active(): ...`; the gate checks
   `logging.getLogger(...).isEnabledFor(...)` or a CLAUDE.md-style
   `SIMSOPT_BOOZER_OBSERVABILITY` env flag.
4. When the gate is closed, skip the `_host_inf_norm` calls at
   `:2169-2171`.

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

- [ ] Status: confirmed opportunity.

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
`src/simsopt/geo/surfaceobjectives_jax.py:1651-1652`,
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
jax_metal_smoke  -> "none"
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

- [ ] Status: hardware-gated.

### Context

Round-2 N11 closed the surface-axis sharding of `integral_BdotN`
with CPU-forced multi-device equivalence and HLO collective proof.
Round-2 N12 closed seed-batch sharding the same way. Both closeouts
explicitly do **not** claim real-GPU speedup.

The current Runpod block is documented in
`docs/source/jax_gpu_setup.rst:421-466` ("Runpod Operational
Notes"). Canonical wording: "Exact JAX 0.9.2 on the stock CUDA 12.4
Runpod image needed CUDA toolkit 12.9 ... CUDA userspace/toolchain
mismatches." That section is the authoritative repo source for the
blocker; resolve from there.

### Rationale

A CPU-multi-device proof is a logical-equivalence check, not a
performance check. The collective patterns may have HBM-bandwidth or
PCIe-bottleneck issues that only surface on real GPUs. Until the
real-GPU lift is measured and recorded, the multi-GPU promise is
unfulfilled.

### Implementation

1. Resolve the Runpod CUDA toolchain mismatch documented at
   `docs/source/jax_gpu_setup.rst:421-466`: rebuild `jaxlib` against
   the host CUDA, or pin a known-good jaxlib for the target H100
   configuration. Land any in-flight launcher patches.
2. Run a 1-vs-2-vs-4-GPU sweep for `integral_BdotN_surface_sharded`
   (`src/simsopt/jax_core/integral_bdotn.py:240`). Measure
   wall-time, HBM peak, HLO collective bytes.
3. Run the same sweep for seed-batch scoring
   (`src/simsopt/geo/surfaceobjectives_jax.py:5088`) at
   production-relevant seed counts.
4. Write `docs/jax_multi_gpu_proof_2026-XX-XX.md` recording the
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

## N31: plumb XLA memory env vars through `apply_jax_runtime_config`

- [ ] Status: confirmed.

### Context

The following env vars are not set or asserted by
`src/simsopt/backend/runtime.py:1716-1737`:

- `XLA_PYTHON_CLIENT_PREALLOCATE` (bool): controls eager 75%
  preallocation.
- `XLA_PYTHON_CLIENT_MEM_FRACTION` (float in `(0, 1]`): bounds total
  preallocated fraction when preallocation is on.
- `XLA_PYTHON_CLIENT_ALLOCATOR`: per official JAX GPU
  memory-allocation docs, accepts `platform` or `vmm`; default
  (unset) selects BFC.
- `TF_GPU_ALLOCATOR=cuda_malloc_async`: this is a **separate** env
  var that selects CUDA's asynchronous allocator. It is NOT a value
  of `XLA_PYTHON_CLIENT_ALLOCATOR`.

Current production practice: scripts hand-export
`XLA_PYTHON_CLIENT_PREALLOCATE=false`
(`benchmarks/hf_jobs/run_production_gpu_proof.sh`,
`.github/workflows/jax_smoke.yml:346,411`,
`scripts/runpod_single_stage_continuation.py`, and
`repo_bootstrap.py:287` for CUDA entrypoints only). Users who skip
this hand-setting inherit JAX's default 75% preallocation.

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
     back to JAX's 0.75 default),
   - `tf_gpu_allocator=None`.
3. In `apply_jax_runtime_config`, set the env vars via
   `os.environ[...]` BEFORE the `jax.config.update("jax_platforms",
   ...)` call at `runtime.py:1725-1728`.
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
   parameter. This item adds an explicit `policy: BackendPolicy |
   None = None` keyword. When `policy` is provided, it overrides
   the mode-derived defaults field-by-field for the four new fields
   above. Add an entry to the signature, plumb through
   `_config_from_mode`, and document in the docstring.
7. Resolution priority (after the extension):
   `set_backend(..., policy=...)` > `SIMSOPT_*` env > mode-derived
   `_MODE_POLICY_DEFAULTS`.

### Acceptance criteria

- Subprocess test under `jax_gpu_fast`: confirms
  `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"` after
  `apply_jax_runtime_config()`.
- Subprocess test under `SIMSOPT_JAX_GPU_MEM_FRACTION=0.5`:
  confirms env propagation; downstream `jax.devices()[0].memory_stats()`
  reports halved bytes_in_use_limit (or equivalent JAX-version-
  specific metric).
- Subprocess test under `SIMSOPT_TF_GPU_ALLOCATOR=cuda_malloc_async`:
  confirms env propagation; XLA log line on JAX import names the
  cudaMallocAsync allocator.
- `banana_coil_solver.py` on GPU works from a fresh shell without
  manual `XLA_PYTHON_CLIENT_PREALLOCATE` export.
- New `set_backend(mode, policy=BackendPolicy(...))` call accepts
  the policy keyword without breaking existing callers.

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
device. For N = 8192 the live device footprint is approximately
`2 × N² × 8 = 1 GB` per LS solver instance.

### Rationale

Per JAX host-offloading docs: explicit `jax.device_put(..., device=
jax.devices('cpu')[0])` is supported and preserves the array's
logical identity. For workflows where the factors are reused
infrequently relative to the per-call GPU compute, host residency
frees ~`N² × 8 × 2` HBM bytes per solver at the cost of a
host→device transfer per adjoint solve. Whether this is a win
depends on the call ratio; offering it as an option is independently
useful.

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
- Memory probe shows host-residency variant frees the
  expected `N² × 8 × 2` device bytes per solver.
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
- **Per-instance device pin** for new objects constructed under a
  `with jax.default_device(jax.devices('cpu')[0]):` block. Any
  NEW JIT compiled inside that block targets CPU; existing
  compiled JITs are unaffected. This admits a "build a CPU-resident
  BoozerSurfaceJAX alongside a GPU-resident one" pattern, but pays
  a compile-cache miss on first call.
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
   - Save solver state to disk via existing pickle path.
   - Restart with `SIMSOPT_BACKEND_MODE=jax_cpu_fast` or
     `jax_cpu_parity`.
   - Load state and resume.
2. Add a small helper
   `simsopt.backend.runtime.with_cpu_device_for_construction()` that
   returns a context manager wrapping
   `jax.default_device(jax.devices('cpu')[0])`. Document that
   anything compiled inside the block is CPU-resident and that
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
