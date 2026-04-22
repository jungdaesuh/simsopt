# simsopt-jax single-stage on-device hot-path diagnosis

This note complements `docs/single_stage_banana_jax_gpu_dependency_trace_2026-04-13.md`.
It records the callback and persistent-compilation-cache behavior of the current
single-stage target lane, with the main correction that "exactly one
persistent-cache candidate per solve" only applies to the outer optimization
artifact, not to every public runtime-bundle entrypoint.

## 2026-04-23 status update

This diagnosis is still directionally correct, but part of the optimizer hot
path changed after 2026-04-19. The main compile-cost drivers remain structural,
yet the seeded target-lane path now avoids some previously flagged overhead.

### Landed since this note was written

- [x] The optimizer-facing single-stage seeded helper now lowers through a
  general-only value-and-grad path, so the baseline-aware `same_coils`
  `lax.cond` no longer sits on the seeded outer-optimizer hot path.
- [x] Private on-device L-BFGS can reuse a seeded initial value/gradient when
  `state.k == 0`, which avoids an otherwise redundant objective/gradient
  reevaluation in that seeded early-exit case.
- [x] L-BFGS history allocation is capped to the reachable correction budget,
  which reduces traced solver state and memory pressure for small-dimensional or
  low-iteration solves.
- [x] The stale runtime-array tracer split described below is still fixed at the
  root and remains closed.

### Still true after the landing work

- [ ] Nested `lax.while_loop` / `lax.cond` structure remains a primary
  compile-cost driver.
- [ ] The implicit-gradient path still uses forward-over-reverse composition for
  the stationarity term.
- [ ] Final dense Jacobian / dense linearization materialization still
  contributes meaningful compile and runtime cost when those paths are active.
- [ ] The new dense least-squares fallback is a robustness path, not a
  performance path; when it triggers it can increase memory and runtime.

## Verified from the current tree

| Claim | Location | Status |
| --- | --- | --- |
| `_emit_iteration_callbacks` fires on accepted steps | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:595-597` | verified |
| `failure_callback` fires on rejected steps with 13 forwarded payload fields | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:542-574` | verified |
| `_emit_host_callback` deliberately routes through `jax.debug.callback(..., ordered=False)` so strict transfer-guard lanes do not trip on the JAX 0.9.2 host token | `src/simsopt/geo/optimizer_jax_private/_common.py:67-75` | verified |
| `can_cache_solver` bypasses the private in-process solver cache when any callback is present | `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:722-728` | verified |
| Inner Newton and line-search bodies stay on device via `lax.while_loop`, `lax.cond`, and `lax.map` / `fori_loop` style primitives rather than host control flow | `src/simsopt/geo/optimizer_jax.py:1459-1462`, `src/simsopt/geo/optimizer_jax.py:1944-2031` | verified |

## Additional host-callback vectors

1. `SIMSOPT_LBFGS_DEBUG` is a denser callback source than `--diagnostic-callbacks`.
   `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:33` and `:70-123` gate
   `_emit_lbfgs_runtime_debug(...)`, while
   `src/simsopt/geo/optimizer_jax_private/_line_search.py:26` and `:35-68`
   gate `_emit_line_search_runtime_debug(...)`. When enabled, both the L-BFGS
   body and each zoom/Wolfe trial emit `jax.debug.callback(...)`.

2. `--diagnostic-callbacks` does not force CPU primary.
   `repo_bootstrap.py:224-231` only appends `cpu` to a comma-separated
   `JAX_PLATFORMS` value such as `cuda,cpu`, which preserves CUDA as primary
   while keeping a callback lane available.

## Runtime-array root fix

A previously reported hot-path concern was that traced coil DOFs would fall into
`np.asarray(...)` inside `src/simsopt/jax_core/_math_utils.py` and force a host
round-trip each iteration. That concern was a false alarm in the old code and
has now been removed at the source.

Before the fix, the NumPy fallback depended on a stale predicate:

- `is_tracer(...)` was defined as
  `hasattr(value, "aval") and not isinstance(value, jax.Array)`.
- `as_runtime_array(...)` only took the NumPy branch when
  `is_tracer(reference)` was true and the value tree contained no JAX leaves.

An isolated local reproduction against that old implementation under JAX 0.9.2
produced:

```text
jax_version 0.9.2
is_tracer False
_contains_jax_leaves True
result_type DynamicJaxprTracer
```

So for real traced inputs, the NumPy branch was effectively unreachable and the
result stayed on the traced/device path.

The root fix now removes that dead split entirely:

- `src/simsopt/jax_core/_math_utils.py` routes `as_runtime_array(...)` straight
  through `as_jax_array(...)` while preserving the public `reference=` API.
- `src/simsopt/geo/curve.py` and `src/simsopt/geo/curvexyzfourier.py` no longer
  carry their own copy-pasted tracer predicates; both delegate to the shared
  `_math_utils.as_runtime_float64(...)` helper instead.

That leaves one SSOT for runtime float conversion on JAX-enabled paths and
eliminates the stale tracer heuristic from the active tree.

## Compile-cost driver structure

The main compile-cost contributors are structural, not evidence that the JAX
port leaks a per-iteration host transfer:

1. HLO depth.
   The optimization stack nests multiple `lax.while_loop` regions: outer
   L-BFGS, line-search/zoom, and the exact Newton solve, with `lax.cond`
   branches inside those regions. That lowers to deep HLO even when each body is
   traced only once.

2. Forward-over-reverse composition on the coil-side geometry pipeline.
   `_traceable_objective_gradient_parts(...)` at
   `src/simsopt/geo/surfaceobjectives_jax.py:2558` splits the gradient into two
   distinct sweeps:
   - `direct_grad` via a single reverse-mode pullback —
     `_strict_scalar_grad(_evaluate_objective_of_coils, coil_dofs)` at `:2696`.
   - `implicit_grad` via forward-over-reverse —
     `jax.vjp(stationarity_of_coils, coil_dofs)` at `:2711-2715`, where
     `stationarity_of_coils` internally calls
     `_traceable_inner_stationarity_grad(...)` at `:2221-2241`. That inner grad
     is deliberately `vmap(jvp)` (forward mode) over an identity basis; the
     adjacent comment at `:2232-2235` records the rationale — a reverse-over-
     reverse trace would trip strict transfer guard on the JAX 0.9.2
     implicit-adjoint path.
   Net: one reverse sweep + one forward-over-reverse pair inline into the same
   HLO per outer step.

3. Both `lax.cond` branches trace on the baseline-aware public forward path.
   `_traceable_forward_result(...)` traces both the baseline and general cases
   at `src/simsopt/geo/surfaceobjectives_jax.py:2379-2500`, even though only
   one executes for a given input. That statement is still true for the public
   baseline-aware runtime bundle. It is no longer the whole optimizer story,
   because the seeded optimizer-facing helper now uses a general-only compiled
   value-and-grad path for the outer target-lane hot path.

4. Final dense Jacobian materialization.
   `_materialize_dense_linear_operator(...)` uses `jax.vmap(...)` over an
   identity basis at `src/simsopt/geo/optimizer_jax.py:1526-1529`. That is
   compile-cost O(1) in trace structure but execution-cost O(n) in JVPs, and it
   still contributes to the late-stage exact-Newton footprint.

Related non-fixes:

- Shrinking `maxcor` only changes ring-buffer shapes; it does not change the
  trace structure of `_two_loop_recursion(...)`.
- Replacing `lax.while_loop` with `lax.scan` is not a semantics-preserving
  switch for L-BFGS because convergence is dynamic. A bounded masked-scan design
  is possible, but it is a different solver contract rather than a simple
  compile-time optimization.

## Persistent compilation cache

The decisive cacheability boundary for the main optimization path is still the
outer L-BFGS `run_solver(...)` at
`src/simsopt/geo/optimizer_jax_private/_lbfgs.py:646-729`. The inner fused
objective pieces inline into that solver, so the outer artifact determines
whether the primary optimization loop can persist compiled executables.

That said, the runtime-bundle API also exposes stable public boundaries backed
by separate compiled callables:

- `runtime_entry["objective"]` is created by
  `_make_traceable_objective_from_compiled_bundle(...)` and wrapped in
  `jax.jit(...)` at `src/simsopt/geo/surfaceobjectives_jax.py:2709-2747`.
- `runtime_entry["public_value_and_grad"]` is created by
  `_make_traceable_value_and_grad_boundary(...)` at
  `src/simsopt/geo/surfaceobjectives_jax.py:2786-2800`.
- `runtime_entry["batched_value_and_grad"]` is created by the separately jitted
  `_make_traceable_batched_value_and_grad_pipeline(...)` at
  `src/simsopt/geo/surfaceobjectives_jax.py:2984-2992` and then exposed through
  `public_batched_value_and_grad` at `:2995-3001`.
- `runtime_entry["public_reporting_metrics"]` lazily selects between the jitted
  reporting-metrics pipelines built by `_make_traceable_reporting_metrics(...)`
  at `src/simsopt/geo/surfaceobjectives_jax.py:2819-2910`.
- `_ensure_traceable_runtime_public_boundaries(...)` holds those stable public
  wrappers on the cached runtime entry at
  `src/simsopt/geo/surfaceobjectives_jax.py:2648-2667`.

Direct example callsites already hit these runtime-bundle boundaries outside the
outer solver:

- accepted-step snapshot refresh:
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:4851`
- phase-1 host wrapper path:
  `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:5653-5654`

So the corrected scope is:

- "exactly one persistent-cache candidate per solve" is true for the main outer
  optimization artifact.
- It is not true for the entire repo call graph, because direct runtime-bundle
  entrypoints can also compile and persist independently when they are called
  outside that solver.

## Callback-bearing sites that poison cacheability

For full persistent-cache reuse on the main solve path, the emitted HLO must be
callback-free. The live callback-bearing sites are:

- accepted-step callback and `progress_callback`:
  `src/simsopt/geo/optimizer_jax_private/_common.py:208-231`
- rejected-step `failure_callback`:
  `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:542-574`
- debug tracing controlled by `SIMSOPT_LBFGS_DEBUG`:
  `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:70-123` and
  `src/simsopt/geo/optimizer_jax_private/_line_search.py:35-68`

These are all Python-level conditionals that collapse before lowering. Removing
the callbacks removes the host-callback ops from the emitted HLO rather than
just silencing them at runtime.

The private Python cache gate at `src/simsopt/geo/optimizer_jax_private/_lbfgs.py:722-728`
is separate and lighter-weight. It only affects repeated traces inside one
Python process.

## Actionable production recommendations

1. Treat callback-free execution as a precondition for persistent-cache reuse on
   the outer solver.
   Do not pass `callback`, `progress_callback`, or `failure_callback`, do not
   use `--diagnostic-callbacks`, and keep `SIMSOPT_LBFGS_DEBUG` unset.

2. If direct runtime-bundle callsites matter for repeated runs, keep those
   callsites on the callback-free public boundaries rather than rebuilding host
   wrappers around mutable graph objects.

3. Put `JAX_COMPILATION_CACHE_DIR` on a persistent volume before the first cold
   compile. The first compile at higher `mpol` remains structurally expensive
   even when the hot path is otherwise clean.

4. Do not expect a debug run to warm the production cache.
   Callback-bearing and callback-free paths lower to different HLO.

5. The stale `_math_utils.is_tracer(...)` cleanup item is closed.
   The shared runtime-float helpers now go through one callback-free JAX array
   conversion path, and backend tests lock in the traced-reference behavior.

## 2026-04-22 update: vmap audit outcome

A focused audit of every `jax.vmap` site in the JAX-port-specific code
identified two suspicious uses of `vmap(jvp) @ eye` for scalar-output
gradients. Resolution:

- `_traceable_objective_gradient_parts.direct_grad` —
  `src/simsopt/geo/surfaceobjectives_jax.py:2695-2696` — **switched** from
  `jax.vmap(lambda t: jax.jvp(objective_of_coils, (coil_dofs,), (t,))[1])(coil_basis)`
  to `_strict_scalar_grad(_evaluate_objective_of_coils, coil_dofs)`. For a
  scalar objective the two are mathematically identical; the reverse-mode
  variant drops compute from O(n_coil) JVPs to a single VJP and removes the
  vmap memory-replication peak on the coil-DOF axis.

- `_traceable_inner_stationarity_grad` —
  `src/simsopt/geo/surfaceobjectives_jax.py:2221-2241` — **kept in forward
  mode by design**, now with an inline comment. Because the outer gradient
  path differentiates this map again via `jax.vjp(stationarity_of_coils, ...)`
  at `:2711-2715`, an all-reverse choice would be reverse-over-reverse and
  trip JAX 0.9.2's strict-transfer-guard host-scalar materialization on null
  tangent paths. Forward-over-reverse is the correct compose for the mixed
  IFT term `(∂²g/∂x_inner ∂coils)^T · adjoint`.

The other five vmap sites were verified clean on the same pass:
`_materialize_dense_linear_operator` (dense operator basis, size-gated by
`max_dense_jacobian_bytes`), `jax_core/biotsavart.py:506` (3×3 tangent set,
forward = reverse cost), `jax_core/biotsavart.py:514` (point-chunk batch),
`jax_core/curve_geometry.py:627` (pairwise distance inside `lax.scan`), and
`objectives/stage2_target_objective_jax.py:265` (banana-symmetry
replication with explicit `in_axes=(0, 0)`).

No nested `vmap`, no `pmap`, and no `vmap` crossing the `@jax.custom_vjp`
boundary was found in port-specific code.
