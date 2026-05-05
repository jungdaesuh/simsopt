# banana coil optimization — JAX-native port: remaining TODOs

Status: post-review plan as of 2026-05-05. Supersedes the prior in-chat port analysis after corrections from the staged-state review.

Companion docs:
- `docs/banana_cpp_cpu_dependency_manifest_2026-05-05.md` — CPU/C++ dep manifest (file inventory)
- `docs/single_stage_banana_jax_gpu_dependency_trace_2026-04-13.md` — JAX/GPU lane trace and host seams checklist
- `/Users/suhjungdae/code/columbia/analysis/jax_gpu_port_dependency_graph_2026-04-17.md` — JAX-port module graph (lives in the sibling `columbia/analysis` repo, not under `simsopt-jax/`)

## Scope and current-state baseline

Stage 2 banana and single-stage banana both run on the JAX target lane today. The core compute path (`BiotSavartJAX`, `SquaredFluxJAX`, `BoozerSurfaceJAX`, `surfaceobjectives_jax.py`, `optimizer_jax.py::target_minimize`) is JAX-native. Forward AND VJP for `CurveCWSFourierCPP` are already supported via the `curve.surf` + `surface_spec` native branch in `_supports_native_curve_geometry` (`src/simsopt/field/biotsavart_jax_backend.py:629`) and `curve_spec_from_curve` (`src/simsopt/jax_core/curve_geometry.py:99`). The Stage 2 target bundle bypasses the C++ candidate culler entirely via fixed-shape JAX scans (`src/simsopt/objectives/stage2_target_objective_jax.py:296, 370`). Single-stage surface self-intersection has a JAX implementation for `SurfaceXYZTensorFourier` and `SurfaceRZFourier` (`single_stage_banana_example.py:5631, 5808`).

What remains is split into three categories:

1. **Target-lane purity proof** — codify that the optimizer never re-enters the legacy graph (highest leverage; not a port, a guarded test)
2. **Legacy public-API JAX-native paths** — for users who consume the public objective classes outside the target bundle
3. **Startup / finalization seam reduction** — generic spec hydration and spec-backed result emission

Items explicitly **NOT** in this plan, with rationale:

- `CurveCWSFourierCPP.to_spec` shim — not needed; existing `surface_spec` branch handles CWS RZ curves
- Surface self-intersection JAX port for non-`XYZTensorFourier` / non-`RZFourier` surfaces — banana doesn't use those surface types
- Warm-start `_fit_surface_xyz_tensor_dofs_to_gamma` JAX rewrite — explicit host pin in code comments at `single_stage_banana_example.py:3220-3225` due to documented Hopper-only cuSolver/runtime failures. Reopening requires validated parity fixtures and a plan addressing the real failure mode.
- VTK / matplotlib export — host-only by nature; already opt-in (`--full-artifacts` for single-stage; `--skip-postprocess` for Stage 2 VTK only)

## TODO 1 — Stage 2 target-lane purity proof (highest leverage)

**Status:** implemented 2026-05-05
**Owner:** TBD
**Estimated effort:** ~2-3 days (revised; was 2 days before scope correction)
**Risk:** low-medium

### Scope (REVISED 2026-05-05 after staged-state review)

Add a runtime guard plus a parity test that prove the Stage 2 target-lane optimizer's **`target_objective_bundle.value_and_grad`** path stays inside the target bundle and never re-enters the legacy `JF.J()` / `JF.dJ()` graph or the legacy `CurveCurveDistance.compute_candidates()` C++ culler.

**Important scope correction:** the strict guard CANNOT cover a full `banana_coil_solver.py --backend jax` run end-to-end as originally written. Stage 2 currently re-enters the legacy graph at known non-gradient-path call sites:

- `accepted_callback(...)` at `banana_coil_solver.py:3248` sets `JF.x = ...` and calls `capture_stage2_feasible_partial_candidate(JF, Jls, Jccdist, ...)`, which evaluates `Jccdist.J()` / `Jccdist.shortest_distance()` and therefore the C++ culler.
- `capture_stage2_trajectory_snapshot(trajectory, JF, new_bs, new_surf, Jf, Jls, Jccdist, Jc, ...)` at `banana_coil_solver.py:3289+` and at the post-optimizer call site does the same (`banana_coil_solver.py:1114-1128` shows the snapshot evaluator pulling `Jccdist.J()` host floats).
- Many `Jccdist.shortest_distance()` callers exist throughout: lines 290, 292, 962, 1128, 1221, 1323, 1326, 1458, 1685.

A naïve full-run strict mode would fail on these legitimate (non-gradient) snapshot/callback re-entries. The proof must scope strictly to the value-and-grad path.

### Motivation

The corrected port analysis concludes items 1 and 2 (distance culling, CWS VJP) are not live blockers for the target lane. That conclusion needs a guard so it doesn't regress silently. Today, nothing prevents a future refactor from inadvertently routing the optimizer's gradient evaluation through the legacy graph and quietly reintroducing C++ residue on the gradient hot path.

### Plan

1. **Identify legacy entry points on the gradient hot path only.** The strict guard must distinguish "hot path" from "snapshot/callback":
   - `simsopt.geo.curveobjectives.CurveCurveDistance.compute_candidates` (`curveobjectives.py:702`) and `.J()` / `.dJ()` (line 736)
   - `simsopt.geo.curveobjectives.CurveSurfaceDistance.compute_candidates` (`curveobjectives.py:927`) and `.J()` / `.dJ()`
   - `Optimizable.J()` / `Optimizable.dJ()` on the legacy `JF` graph in Stage 2 (the composite assembled at `banana_coil_solver.py:2837-2844`)

2. **Add a stack-scoped strict guard.** `SIMSOPT_TARGET_LANE_STRICT=1` should be effective only when a context-manager flag is also active:
   ```python
   with strict_target_lane_purity():
       value, grad = target_objective_bundle.value_and_grad(dofs)
   ```
   Inside that context, the listed entry points raise `RuntimeError("target-lane bypass: <entry>")`. Outside the context (snapshots, callbacks, finalization), the guard is a no-op. Implementation: thread-local flag in `src/simsopt/backend/runtime.py`; the entry points consult the flag.

3. **Wire the context into the optimizer.** In `optimizer_jax.py::target_minimize` and the value-and-grad caller in `banana_coil_solver.py`, wrap each `value_and_grad` invocation in `strict_target_lane_purity()` when `is_jax_backend() and os.environ.get("SIMSOPT_TARGET_LANE_STRICT") == "1"`.

4. **Trace the optimizer call stack.** Run `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py --backend jax --num-iterations 2` with `SIMSOPT_TARGET_LANE_STRICT=1` and existing JAX trace logging. Confirm no `RuntimeError` from inside the context and no transfer-guard hits.

5. **Write the regression tests.** New `tests/integration/test_stage2_target_lane_purity.py`:
   - Guard scoping: `SIMSOPT_TARGET_LANE_STRICT=1` only raises inside `strict_target_lane_purity()`, so snapshot/callback code outside the value/grad context remains unchanged.
   - Negative guard tests: legacy `OptimizableSum.J()` and `CurveCurveDistance.compute_candidates()` raise inside the strict context.
   - Optimizer wiring: `optimizer_jax.target_minimize(...)` and `banana_coil_solver.py::run_stage2_optimizer(...)` wrap explicit value-and-grad calls before dispatch.
   - Target-bundle coverage: `tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::test_strict_mode_allows_target_scalar_objective_evaluation` evaluates the scalar objective and value/grad under strict mode with no `pure_callback` in the traced value/grad path.

### Optional follow-up (out of scope for TODO 1; tracked here)

If the goal becomes "no legacy graph anywhere on the JAX lane" (not just on the gradient hot path), the prerequisite is rerouting `capture_stage2_trajectory_snapshot` and `capture_stage2_feasible_partial_candidate` through target-bundle reporting metrics. That is a separate, larger refactor — explicitly NOT covered by this TODO.

### Validation

- `tests/integration/test_stage2_target_lane_purity.py` passes and proves the strict guard is env-and-stack scoped.
- `tests/integration/test_stage2_jax.py` passes with strict target-bundle evaluation and the existing CPU/JAX value, gradient, and short-run parity tests.
- Manual local sanity: reduced Stage 2 JAX target run and restart run from the emitted `biot_savart_opt.json` both complete on the local CPU JAX runtime. CUDA/GPU trace parity remains a hardware validation gate, not a code fallback.

### Dependencies

None. This work is independent and unblocks confidence on items 2 and 4.

---

## TODO 2 — JAX-native distance objectives for the legacy public API

**Status:** implemented 2026-05-05
**Owner:** TBD
**Estimated effort:** ~1-1.5 days
**Risk:** low

### Scope

Replace the C++ `sopp.get_pointclouds_closer_than_threshold_*` candidate culler call inside `CurveCurveDistance.compute_candidates()` and `CurveSurfaceDistance.compute_candidates()` with a JAX-native dense-pairwise mask path when `is_jax_backend()` is active. Public-API only — the target lane already bypasses this via `stage2_target_objective_jax.py`.

### Motivation

For users who consume `CurveCurveDistance` / `CurveSurfaceDistance` outside the target bundle (older notebooks, ad-hoc scripts, parity comparisons against the legacy CPU lane), the C++ culler is currently the only path. Closing it gives full lane purity for the public API and removes the last cross-lane C++ residue on the public objective surface.

### Plan (REVISED 2026-05-05 after staged-state review)

The C++ kernel emits **strictly lower-triangle pairs** (`j < i`) and gates on `j < num_base_curves` (`src/simsoptpp/python_distance.cpp:75-80`). It also applies a final exact `two_points_too_close_exist` threshold check before returning (`python_distance.cpp:93-97`). `CurveCurveDistance.J()` sums `cc_distance_pure(...)` over `for i, j in self.candidates` (`src/simsopt/geo/curveobjectives.py:736`), so emitting both `(i, j)` and `(j, i)` would double-count the penalty. The original plan's `(i != j)` + `jnp.argwhere` would have been semantically wrong; the fix below preserves C++ semantics.

1. **Design the JAX culler.** For `C` curves with `Q` quadpoints each:
   - Stack curve gammas to shape `(C, Q, 3)`
   - Pairwise via `jax.vmap`: `dists[i, j] = min over (k, l) of ||gamma_i[k] - gamma_j[l]||`
   - Lower-triangle + base-curve mask: `valid_mask[i, j] = (j < i) & (j < num_base_curves)` — matches C++ pair enumeration at `python_distance.cpp:75-80`
   - Threshold: `close_mask[i, j] = (dists[i, j] < threshold)`
   - Final candidate mask: `candidates_mask[i, j] = valid_mask[i, j] & close_mask[i, j]` — already exact (C++ does the same exact final check at `python_distance.cpp:93-97`, so the spatial-hash culling is purely an optimization, not a different semantic)
   - Output static-shape `(C, C)` boolean array; downstream consumers iterate over `jnp.argwhere(candidates_mask)` (host-side once, since candidate selection is non-differentiable)

2. **Add a JAX-mode branch in `compute_candidates`.** At `src/simsopt/geo/curveobjectives.py:702` and `:927`:
   ```python
   if is_jax_backend():
       candidates = _jax_get_close_candidates(point_clouds, threshold, num_base_curves)
   else:
       candidates = sopp.get_pointclouds_closer_than_threshold_within_collection(...)
   ```
   New helper `_jax_get_close_candidates` in `src/simsopt/geo/_distance_jax.py` (new file). For `CurveSurfaceDistance.compute_candidates()`, mirror the same approach against the two-collection variant (`get_pointclouds_closer_than_threshold_between_two_collections`); that kernel does NOT have the lower-triangle constraint (cross-collection pairs are inherently directional), so the mask is rectangular `(|A|, |B|)` with no `j < i` reduction.

3. **Optimize for banana scale.** With ~21 curves at most and 128 banana / 15 TF quadpoints, `(C, C, Q_max, Q_max)` ≈ 21² × 128² ≈ 7M pairs — trivial on GPU, ~30ms cold compile. No spatial hashing needed at this scale.

4. **Preserve gradient semantics.** Candidate selection is non-differentiable on both lanes; gradient flows through the existing JAX pure kernels (`cc_distance_pure`, `cs_distance_pure`) that consume the candidate list. No gradient-path change needed.

### Validation

- `tests/geo/test_distance_jax.py` asserts exact candidate set equality against the C++ culler for within-collection and between-collection point clouds.
- The same test module verifies the JITted masks have static `(C, C)` / `(C, S)` shapes and that public `CurveCurveDistance` / `CurveSurfaceDistance` JAX-mode `compute_candidates()` calls do not call the `simsoptpp` cullers.

### Dependencies

- TODO 1 (strict guard) makes the regression test trivial to write

---

## TODO 3 — Direct Stage 2 JSON → runtime-spec converter

**Status:** implemented 2026-05-05
**Owner:** TBD
**Estimated effort:** ~3-4 days (revised; was 2-3 days before schema correction)
**Risk:** medium (SIMSOPT GSON graph schema is more involved than originally scoped)

### Scope (REVISED 2026-05-05 after staged-state review)

Add a sibling reader to `_core/json.py::load` that emits immutable JAX specs (`CoilSetSpec`, `CurveCWSFourierRZSpec`, `SurfaceRZFourierSpec`, `SurfaceXYZTensorFourierSpec`, …) directly from a checkpoint JSON, without instantiating any `simsoptpp` Python objects.

Implementation note (2026-05-05): `src/simsopt/_core/json.py::load_specs` now reads legacy SIMSON Stage 2 JSON into immutable JAX specs and also reads SIMSON-wrapped spec JSON written by the new spec emitters. The return value includes `graph` plus typed aliases such as `biot_savart_spec`, `coil_set_spec`, and `surface_spec`; JAX restart artifacts carry a `BiotSavartSpec` with the coil DOF extraction contract and flat Stage 2 DOF vector.

**Important reframing:** the original draft framed this as "the JAX-lane startup boundary." That was wrong — `single_stage_banana_example.py:10674-10688` shows the JAX branch already builds `bs = SingleStageRuntimeSpecBiotSavartJAX(warm_start_runtime_spec_state["runtime_spec"])`; `load(stage2_bs_path)` runs only on the CPU else-branch. The current single-stage JAX startup does not call the legacy `load()` for the field. So this TODO is really: *"give Stage 2 (and any other consumer of the saved JSON) a direct path from disk JSON to a runtime spec, without round-tripping through `Optimizable` objects"* — useful both for Stage 2 internal restarts and for hydrating runtime specs from offline JSON without the live graph.

### Motivation

Stage 2's internal restart and any tool that wants to hydrate a runtime spec from a saved JSON checkpoint currently has to round-trip through `Optimizable` objects (`load()`), which means instantiating `simsoptpp` Python objects with C++ state restoration. A direct-to-spec reader removes that round-trip and gives the JAX lane an alternative spec-source for warm-start that doesn't depend on the host-side `Optimizable` graph being constructible.

### Plan

1. **Inventory the actual JSON schema.** SIMSOPT uses GSON (`src/simsopt/_core/json.py:125`+), not a flat `class_name` dispatcher. The real schema markers are:
   - `@class`, `@module`, `@name`, `@version` — type discriminators
   - Outer `SIMSON` wrapper with `graph` (a `$type: "ref"` pointer) and `simsopt_objs` (a dict of objects keyed by ID string)
   - `DOFs` nodes (`@class: DOFs`, `@module: simsopt._core.optimizable`) — referenced from many other nodes
   - `$type: "ref"` cross-references between nodes

   Inspect representative bundles:
   - `STAGE_2/outputs-wout_nfp22ginsburg_*/.../biot_savart_opt.json`
   - `STAGE_2/outputs-wout_nfp22ginsburg_*/.../surf_opt.json`

   Catalogue every `@class` value that appears. Banana bundles are known to include at minimum:
   - `SIMSON` (outer wrapper)
   - `BiotSavart`
   - `Coil`
   - `Current` and `ScaledCurrent` (symmetry-replicated currents)
   - `CurveXYZFourier`, `CurveCWSFourierCPP`
   - `RotatedCurve` (symmetry-replicated curves with `flip` and `phi` rotation parameters)
   - `SurfaceXYZTensorFourier`, `SurfaceRZFourier`
   - `DOFs`

   The original plan's per-class reader list **omitted `RotatedCurve` and `ScaledCurrent`**. Without those, banana coil sets — which always include symmetry-replicated copies — would be reconstructed with wrong curve orientations and current scales.

2. **Add the dispatcher.** New `src/simsopt/_core/json.py::load_specs(path: str) -> dict[str, Spec]`:
   ```python
   def load_specs(path):
       with open(path) as f:
           data = json.load(f)
       assert data["@class"] == "SIMSON", "expected GSON SIMSON wrapper"
       return _build_spec_graph(data)
   ```
   `_build_spec_graph` walks `simsopt_objs`, resolves `$type: "ref"` pointers exactly like `_core/json.py::SIMSON.from_dict` does, and emits specs instead of `Optimizable` objects.

3. **Per-class spec readers.** Dispatch on `(@module, @class)` pairs (not bare `class_name`). Required minimum for banana:
   - `_read_dofs(node) -> np.ndarray` — extracts the `numpy/array` payload
   - `_read_curve_xyz_fourier_spec(node, dofs_resolver) -> CurveXYZFourierSpec`
   - `_read_curve_cws_fourier_rz_spec(node, dofs_resolver, surface_resolver) -> CurveCWSFourierRZSpec`
   - `_read_rotated_curve_spec(base_curve_spec, flip, phi) -> RotatedCurveSpec` — wraps a base curve with stellsym flip + rotation; banana symmetry-replicated coils use this
   - `_read_surface_rz_fourier_spec(node, dofs_resolver) -> SurfaceRZFourierSpec`
   - `_read_surface_xyz_tensor_fourier_spec(node, dofs_resolver) -> SurfaceXYZTensorFourierSpec`
   - `_read_current_spec(node, dofs_resolver) -> CurrentSpec`
   - `_read_scaled_current_spec(base_current_spec, scale) -> ScaledCurrentSpec` — required to preserve symmetry-replicated current scales
   - `_read_coil_spec(node, curve_resolver, current_resolver) -> CoilSpec`
   - `_read_biot_savart_spec(node, coil_resolver) -> CoilSetSpec`

   If any reader hits an `@class` not in the dispatcher, raise `NotImplementedError(f"{module}.{cls}")`.

4. **Hook a use site for Stage 2.** Stage 2 internal restart logic (or any `--from-json` tool) can call `load_specs(path)` and feed the resulting `CoilSetSpec` directly into `SingleStageRuntimeSpecBiotSavartJAX`. Leave the legacy `bs = load(...)` path untouched on the CPU lane and on Stage 2 startup paths that haven't been refactored.

5. **Schema-evolution test.** Add `tests/core/test_load_specs.py` that:
   - Loads every checked-in `biot_savart_opt.json` / `surf_opt.json` under `STAGE_2/outputs-*/`
   - Asserts the spec graph is structurally consistent
   - Compares DOF arrays bit-for-bit against `host_load(path)` traversal of the corresponding `Optimizable` graph
   - Exercises both `RotatedCurve` and `ScaledCurrent` (banana bundles always have symmetry-replicated coils, so this is automatic)

### Validation

- `tests/core/test_load_specs.py` serializes real SIMSOPT JSON and verifies `load_specs(path)` preserves surface DOFs and Biot-Savart field values against `load(path)` at `rtol=1e-12`.
- The legacy Biot-Savart fixture uses `nfp=2, stellsym=True`, so it exercises both `RotatedCurve` and `ScaledCurrent` graph nodes.
- The restart-spec test writes a `BiotSavartSpec`, reads it with `load_specs(path)`, hydrates `SpecBackedBiotSavartJAX`, and matches the legacy field at `rtol=1e-12`.
- Negative tests reject unsupported SIMSON wrappers, unsupported GSON value types, and unsupported `@class` values.

### Dependencies

- None for the loader itself
- TODO 4 (final results emitter) becomes much cleaner once specs are first-class on the read side too

---

## TODO 4 — Stage 2 final-results emitter: consume specs instead of live `Optimizable`

**Status:** implemented 2026-05-05
**Owner:** TBD
**Estimated effort:** ~2 days (revised to include the spec-materializer prerequisite; was 1-2 days)
**Risk:** low

### Scope (REVISED 2026-05-05 after staged-state review)

Decouple **Stage 2's** final-results JSON emission from live `Optimizable` graphs. Stage 2 currently calls `new_bs.save(stage2_bs_output_path)` and `new_surf.save(stage2_surface_output_path)` at `banana_coil_solver.py:3499-3500` — unconditionally, before the `--skip-postprocess` gate (which gates only VTK at `:3502+`).

Implementation note (2026-05-05): JAX Stage 2 finalization now routes through `Stage2TargetObjectiveBundle.final_specs_from_dofs` and writes SIMSON-wrapped immutable spec payloads. `load_specs(path)` returns the typed spec aliases, normal `load(path)` reconstructs `BiotSavartSpec` / surface spec dataclasses rather than live `Optimizable` objects on this JAX artifact path, and the JAX `--stage2-bs-path` restart loader hydrates `SpecBackedBiotSavartJAX` directly from the saved `BiotSavartSpec`.

**Important scope correction:** the original draft included single-stage final-results in this TODO. That was wrong. Single-stage's target lane already enforces the spec-only contract: `_require_cached_target_lane_reporting_metrics` at `single_stage_banana_example.py:5212` is the final-metrics resolver, and it raises if the cached accepted-step summary is missing. The dispatch at `single_stage_banana_example.py:5298-5302` routes target-lane runs through that cache-only path:

```python
if use_target_lane:
    return _require_cached_target_lane_reporting_metrics(
        run_dict,
        coil_dofs,
        benchmark_mode=benchmark_mode,
    )
```

Single-stage is already done on the JAX lane. This TODO is Stage 2 artifact emission only, plus an explicit confirmation that single-stage cache enforcement remains the contract.

### Motivation

Stage 2's `new_bs.save(...)` and `new_surf.save(...)` at finalization both call into live `Optimizable` graphs to serialize state. On the JAX lane, the runtime bundle already holds the equivalent data as immutable specs; the emitter just needs to read from those specs and emit GSON-compatible JSON without round-tripping through `Optimizable.save()`.

### Plan

1. **Inventory live-graph reads in Stage 2 final results.**
   - From `banana_coil_solver.py:3490` (results section start) to end of file
   - Specifically: lines 3497-3500 (the unconditional `bs.save()` / `surf.save()`), plus any `.J()` / `.dJ()` / `host_float(.J())` calls during the post-optimizer reporting (lines 3286+ already call `capture_stage2_trajectory_snapshot` — those are tracked as the snapshot/callback rerouting follow-up in the out-of-scope list, not here)
   - Categorize: (a) reads that already come from the runtime bundle, (b) reads that re-enter the legacy graph during artifact emission.

2. **PREREQUISITE — expose final specs from optimizer DOFs on the bundle.** `Stage2TargetObjectiveBundle` (`src/simsopt/objectives/stage2_target_objective_jax.py:104`) is a `NamedTuple` whose current fields are:
   - `objective`, `expected_dof_count`, `value_and_grad`, `terms`, `raw_terms`, `least_squares_residual`, `alm_value_and_grad_builder`, `field_sharding_summary`, `pairwise_penalty_sharding_summary`

   It does **not** expose `accepted_step_summary`, `coil_set_spec`, or `surface_spec`. The original implementation sketch referenced fields that don't exist; before the spec writers (subtask 3) can be wired, the bundle must be extended:
   - Add a new field `final_specs_from_dofs: Callable[[jnp.ndarray], FinalSpecBundle] | None = None` to `Stage2TargetObjectiveBundle`
   - Implement a `FinalSpecBundle` `NamedTuple` (or dataclass) holding `coil_set_spec: GroupedCoilSetSpec` and `surface_spec: SurfaceRZFourierSpec | SurfaceXYZTensorFourierSpec`
   - In the bundle factory (the function that constructs `Stage2TargetObjectiveBundle`), close over the runtime `coil_set_spec` and `surface_spec` and return a `final_specs_from_dofs(final_dofs)` callable that materializes the final specs by applying `final_dofs` to the captured spec's DOF map (no re-entry into `Optimizable`; pure spec arithmetic mirroring `grouped_coil_set_spec_from_lists` / `grouped_coil_set_spec_from_coil_specs` at `stage2_target_objective_jax.py:25-26`)
   - Add a unit test that constructs a bundle, calls `final_specs_from_dofs(initial_dofs)`, and asserts the resulting specs equal the input specs bit-for-bit
   - Estimated subtask cost: ~½ day

3. **Implement spec-based JSON writers.** New helpers in `src/simsopt/_core/json.py` (or a sibling `_spec_writers.py`):
   - `save_biot_savart_spec(path: str, coil_set_spec) -> None` — emits a SIMSON-wrapped JSON that round-trips through `load(path)` / `load_specs(path)` (TODO 3) to an equivalent state
   - `save_surface_rz_fourier_spec(path: str, surface_spec) -> None`
   - `save_surface_xyz_tensor_fourier_spec(path: str, surface_spec) -> None`

4. **Wire Stage 2's emitter to the spec writers.** At `banana_coil_solver.py:3499-3500`, branch on `is_jax_backend()`:
   ```python
   if is_jax_backend() and target_objective_bundle.final_specs_from_dofs is not None:
       final_specs = target_objective_bundle.final_specs_from_dofs(final_dofs)
       save_biot_savart_spec(stage2_bs_output_path, final_specs.coil_set_spec)
       save_surface_xyz_tensor_fourier_spec(stage2_surface_output_path, final_specs.surface_spec)
   else:
       new_bs.save(stage2_bs_output_path)
       new_surf.save(stage2_surface_output_path)
   ```
   (`final_dofs` = the DOFs returned by the optimizer at termination; already in scope at the call site.)

5. **Add an artifact-contract assertion.** The JAX emitter intentionally writes strict immutable spec payloads rather than byte/canonical-key-identical legacy `Optimizable` graphs. The regression assertion is: `load_specs(path)` exposes `biot_savart_spec`, `coil_set_spec`, and `surface_spec`; `load(path)` returns the corresponding spec dataclass on JAX-emitted artifacts; and Stage 2 can restart from the emitted `biot_savart_opt.json` through the spec-backed field adapter.

6. **Keep the CPU lane untouched.** CPU lane still uses `Optimizable.save(...)`.

### Validation

- `tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::test_stage2_script_skip_postprocess_still_writes_restart_artifacts` asserts the reduced JAX Stage 2 run writes spec artifacts, both `load_specs(path)` and `load(path)` read them as specs, and a second Stage 2 run restarts from the emitted `biot_savart_opt.json`.
- `tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::test_target_scalar_objective_exposes_final_specs_from_dofs` verifies final specs materialized from optimizer DOFs match the live input field and surface state.
- Single-stage target-lane cache enforcement remains out of this Stage 2 emitter change; no new host wrapper path was added.

### Dependencies

- TODO 1 strict guard makes regressions visible (but the strict guard does not cover the artifact-emission code path, since that's outside `value_and_grad`)
- TODO 3 spec loader is the symmetric read side. Best landed together so the writers and readers share schema work.

---

## TODO 5 — Documentation correction

**Status:** implemented 2026-05-05
**Owner:** TBD
**Estimated effort:** <1 hour
**Risk:** none

### Scope

Update the caveat section of `docs/banana_cpp_cpu_dependency_manifest_2026-05-05.md` to reflect the verified state of `_supports_native_curve_geometry`. Same correction applies to any analysis-graph note that still mentions a `Coil.vjp()` C++ fallback for non-`CurveXYZFourier` curves.

### Motivation

The manifest currently parrots an analysis-graph note about CWS VJP fallback. Finding 1 of the staged-state review invalidated that note. Leaving it in place misleads anyone reading the manifest into thinking there's a port to do.

### Plan

1. Edit the relevant caveat in `docs/banana_cpp_cpu_dependency_manifest_2026-05-05.md` to note: CWS forward AND VJP are both JAX-native via the `curve.surf` + `surface_spec` native branch (`biotsavart_jax_backend.py:629`, `jax_core/curve_geometry.py:99`); no `.to_spec()` shim is needed for `CurveCWSFourierCPP`.
2. Cross-check the JAX-port dependency graph at `/Users/suhjungdae/code/columbia/analysis/jax_gpu_port_dependency_graph_2026-04-17.md` (lives in the sibling `columbia/analysis` repo, **not** under `simsopt-jax/`) — §2.2 and §3.6. If those sections still say `Coil.vjp()` falls back to `sopp.biot_savart_vjp_graph` for non-`CurveXYZFourier` curves, append an "as of 2026-05-05" correction noting the second branch of `_supports_native_curve_geometry`.

### Validation

- Doc grep: no remaining "fallback to Coil.vjp" or ".to_spec shim" mentions in `docs/*` or `analysis/*` except as historical context

### Dependencies

None.

---

## Sequencing recommendation

1. **TODO 1 first** (target-lane purity proof). Lowest risk, highest leverage. Codifies the invariant the rest of the plan rests on, and produces the regression net for the other TODOs.
2. **TODO 5** in parallel with TODO 1 (doc correction is independent and trivial).
3. **TODO 2** next (legacy public-API distance JAX). Small, contained, validated by TODO 1's guard.
4. **TODOs 3 and 4 together** (spec loader + spec-based emitter). They share schema work and benefit from being implemented as a paired read/write contract.

After all five land, banana single-stage and Stage 2 should both run with **zero C++ in the gradient hot path** on the JAX lane, with the warm-start `lstsq` host pin remaining as the single documented exception.

**The stronger guarantee — "zero re-entry into the legacy `Optimizable` graph anywhere on the JAX lane" — is NOT delivered by this plan.** TODO 1's strict guard explicitly excludes Stage 2 snapshot/callback paths. Live re-entries remain at:

- `banana_coil_solver.py:1114` (`_build_stage2_explicit_term_payload`) — calls `host_float(context.Jf.J())`, `host_float(context.Jls.J())`, `host_float(context.Jccdist.J())`, `host_float(context.Jc.J())` for snapshot diagnostics
- `banana_coil_solver.py:2477` (`capture_stage2_trajectory_snapshot`) — accepts the legacy `JF, Jf, Jls, Jccdist, Jc` and routes them to the explicit-term payload above
- `banana_coil_solver.py:3248` (`accepted_callback`) — sets `JF.x = ...` and calls `capture_stage2_feasible_partial_candidate(JF, Jls, Jccdist, ...)` per accepted optimizer step

Removing these requires a Stage 2 reporting refactor that reads the same diagnostics from the target-bundle's accepted-step summary instead. That refactor is recorded as an out-of-scope follow-up below; folding it into this plan would significantly expand TODO 4's scope.

## Out-of-scope follow-ups

These are deliberately not included; record here so they don't get lost:

- Warm-start `_fit_surface_xyz_tensor_dofs_to_gamma` JAX rewrite — unblock requires a separate plan that addresses the documented Hopper cuSolver instability
- VTK / matplotlib export decoupling — opt-in already; further refactor is cosmetic
- Self-intersection JAX coverage for surface types beyond `XYZTensorFourier` / `RZFourier` — banana doesn't use those types
- ALM outer-loop trace / device residency — `alm_utils.py` is shared; the inner step is already JAX-native, so outer-loop arithmetic on host is acceptable
- **Stage 2 snapshot / callback / accepted-step diagnostics rerouting** — required to deliver "zero re-entry into the legacy `Optimizable` graph anywhere on the JAX lane". Touch sites: `banana_coil_solver.py:1114, 2477, 3248`. Replace `host_float(Jf.J())` / `host_float(Jccdist.J())` / `Jccdist.shortest_distance()` reads with equivalent accessors on `target_objective_bundle.terms` and on a new accepted-step summary surface (see TODO 4 subtask 2 for the spec materializer that this would also depend on). Larger-than-TODO-4 refactor; deferred until the lighter-weight TODOs land and surface any architectural constraints.
