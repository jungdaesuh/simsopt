Here is the best combined plan:

Update as of 2026-04-01:

* the public mode-based backend API now exists
* strict compatibility-rejection mode now exists
* a first `jax_core/` subtree now exists
* first-wave immutable specs now exist for grouped coils, fixed-surface flux,
  `SurfaceRZFourier`, `CurveXYZFourier`, `CurveRZFourier`,
  `CurvePlanarFourier`, `CurveHelical`, `CurvePerturbed`,
  `CurveFilament`, current values, coils, and field-evaluation point sets
* hot-path legacy objects now expose `to_spec()` where the adapter seam is
  stable (`CurveXYZFourier`, `CurveRZFourier`, `CurvePlanarFourier`,
  `CurveHelical`, `CurvePerturbed`, `CurveFilament`, `SurfaceRZFourier`,
  `CurrentBase`, `Coil`)
* the `SurfaceRZFourier` JAX path now covers the broader geometry/derivative
  contract beyond the original fixed-surface hot path
* grouped forward-field compatibility paths now route through `jax_core`
* the grouped forward field path now has a first chunked point-axis reduction
  implementation in `jax_core.field`
* `BiotSavartJAX.coil_set_spec()` now requires explicit immutable grouped-coil
  state; the legacy live-graph geometry seam is only available through the
  explicit compatibility wrapper
* `BoozerSurfaceJAX` now rejects hidden grouped-coil compatibility
  reconstruction in `_refresh_coil_data()` instead of silently using
  `_extract_coil_data_grouped()` or raw `_coils` list extraction
* a first broader non-hot-path objective-family migration is landed:
  `NonQuasiSymmetricRatioJAX` and `BoozerResidualJAX` now route their direct
  objective-side field setup through immutable coil specs reconstructed from
  explicit DOFs
* the next immutable-spec broadening slice is also landed:
  `CurvePlanarFourier`, `CurveHelical`, `CurvePerturbed`, and
  `CurveFilament` now round-trip through immutable curve specs, and
  `BiotSavartJAX` rejects unsupported curve families instead of using grouped
  live-graph reconstruction
* `docs/using_jax_backend.md` now includes copy-paste Stage 2 and single-stage
  examples plus honest performance guidance for compile time, warm timing,
  parity lanes, fast lanes, and memory tradeoffs
* benchmark productization is now more concrete:
  * a stable-hardware weekly manifest exists under `benchmarks/manifests/`
  * a standardized markdown report template exists under
    `benchmarks/reports/`
  * a renderer exists for JSON -> markdown benchmark reports
  * scheduled benchmark reporting now exists as a workflow contract for the
    stable-hardware lane
* `native_cpu` remains the default rollout lane; JAX stays opt-in

Read this document as architecture rationale and module-level guidance for the
remaining work, not as a claim that Phase 0 and Phase 1 are still entirely
untouched.

Use **Plan A’s backend UX and parity CI**, **Plan B’s functional/PyTree mindset**, and **my layered design** where Simsopt keeps its public API but the numerical core moves to a **pure JAX backend**. The key change is: do **not** try to JIT the current mutable `simsoptpp` object graph directly. Build a JAX kernel layer underneath it, then adapt the existing classes to call that layer. That is the safest way to get a reliable GPU port and the only path that cleanly unlocks `jit`, `vmap`, autodiff, sharding, and multi-GPU scaling. JAX’s own docs are explicit that transformations like `jit`, `vmap`, and `grad` assume pure functions, not hidden mutable state. ([JAX Documentation][1])

Why this is the right synthesis is visible in the current code. Simsopt already has a JAX foothold: `geo/__init__.py` globally enables `jax_enable_x64`, `curve.py` uses `vjp`, `jacfwd`, and `jvp`, and there are existing `JaxCurveXYZFourier` and `JaxCurvePlanarFourier` classes. But the current JAX geometry path is still import-pinned to CPU in `geo/jit.py`, while the heavy field path remains stateful and C++-backed: `MagneticField` requires C-contiguous host arrays and caches results behind `set_points`, `BiotSavart` subclasses `sopp.BiotSavart`, `InterpolatedField` mutates the wrapped field by saving/restoring points, and `SquaredFlux` sets field points from the surface during initialization. That is exactly the combination that makes a direct “swap NumPy for JAX” port fragile.         

## 1) Target architecture

### A. Pure JAX kernel layer

This layer owns all GPU-capable numerics. Inputs are immutable pytrees or dataclasses of arrays; outputs are arrays or pytrees. No caches, no hidden mutation, no Python callbacks inside compiled regions. This is where `jit`, `vmap`, `grad`, `custom_vjp`, explicit sharding, and multi-device execution apply naturally. Use custom pytree registration for new immutable spec objects, not for the existing mutable wrappers themselves. ([JAX Documentation][1])

### B. Simsopt compatibility façade

Keep `Curve`, `Surface`, `BiotSavart`, `SquaredFlux`, `LeastSquaresProblem`, wireframe, and PM APIs intact, but make them thin adaptors over either:

* `native_cpu` backend, which calls today’s NumPy/C++ path, or
* `jax_cpu` / `jax_gpu` backends, which call the new pure JAX kernels.

This preserves user code while avoiding a big-bang rewrite. The façade can still cache results for legacy behavior, but compiled numerics must live below it in the functional layer.   

### C. CPU boundary layer

Keep VMEC, SPEC, Boozer-heavy external equilibrium workflows, tracing helpers, and interpolated-field workflows as CPU boundaries in wave 1. They are either external-code boundaries or deeply stateful wrappers and are not the right first GPU target. Multi-host scaling can later use `jax.distributed.initialize()` when the core kernels are already correct.   ([JAX Documentation][2])

## 2) Backend modes

Implement four explicit modes:

* `native_cpu`
* `jax_cpu_parity`
* `jax_gpu_parity`
* `jax_gpu_fast`

`jax_cpu_parity` is the first oracle. `jax_gpu_parity` must run the same algorithms, dtypes, chunking, and reduction order as `jax_cpu_parity`. `jax_gpu_fast` is where lower precision, bigger kernels, or more aggressive sharding can be enabled later. This matters because JAX defaults to x64 **off** unless enabled, while current Simsopt already enables x64 in `geo/__init__.py`. Also, JAX explicitly says exact numerics are not guaranteed stable across backends or even within/without `jit`, so parity must be a defined contract, not a bitwise-equality promise.  ([JAX Documentation][3])

## 3) Repo refactor map

Add a backend subtree like:

```text
src/simsopt/backend/
  config.py
  adapters/
    curve.py
    surface.py
    field.py
    objectives.py
  jax_core/
    specs.py
    geometry_curves.py
    geometry_surfaces.py
    biotsavart.py
    wireframe_field.py
    pm_matrix.py
    objectives_flux.py
    objectives_curve.py
    solvers_continuous.py
    solvers_greedy.py
    sharding.py
    testing.py
```

Use immutable spec objects such as `CurveSpec`, `SurfaceSpec`, `CoilSpec`, `FieldSpec`, and `ObjectiveSpec`. Existing `Optimizable` objects get `to_spec()` / `from_spec()` or `to_backend()` adaptors. That gives you JAX-friendly data without trying to make current mutable cached objects themselves the compiled state. ([JAX Documentation][4])

## 4) Detailed implementation phases

### Phase 0: unpin JAX from CPU and formalize runtime selection

Remove the import-time `jax.config.update('jax_platform_name', 'cpu')` from `geo/jit.py`. Keep x64 as the default for parity mode. Add a backend selector and mode selector at runtime instead of hiding platform policy in module import side effects. This is the first blocker to remove because right now existing JAX geometry cannot become a real GPU path.   ([JAX Documentation][3])

**Done when:** the current JAX curve classes run on CPU and GPU without changing user code except backend selection.

### Phase 1: build immutable JAX specs and adapters

Create spec dataclasses for curves, surfaces, currents, coils, and field evaluation requests. Adapters extract array state from legacy `Optimizable` trees and feed pure JAX kernels. Do not register the full legacy objects as pytrees if they still own caches or mutable dependencies. The façade can remain stateful; the compiled layer cannot.   ([JAX Documentation][1])

**Done when:** `Curve`, `Surface`, and `BiotSavart` can be converted to/from specs without losing semantic state.

### Phase 2: finish the JAX geometry backend

Start from what already exists:

* `curve.py` already has JAX pure helpers and AD usage.
* `JaxCurveXYZFourier` and `JaxCurvePlanarFourier` already exist and are explicitly described as autodiff-compatible JAX versions.

Now extend this pattern to:

* `SurfaceRZFourier.gamma`
* tangents and normals
* `unitnormal`, area, volume
* first- and second-order geometry derivatives used by objectives

`SurfaceRZFourier` is a prime target because it is still a C++/OpenMP Fourier kernel today and it sits on the hot path for stage-2 optimization.     

**Done when:** JAX CPU reproduces native `SurfaceRZFourier` geometry outputs and area/volume derivatives in parity mode.

### Phase 3: port Biot–Savart as a real JAX kernel

This is the centerpiece. Implement pure JAX kernels for:

* `B(points, coils)`
* `A(points, coils)`
* `dB/dX`, `dA/dX`
* optionally second derivatives if profiling shows they belong on-device early

Use a **chunked reduction** design:

* shape inputs as `points[P,3]`, `gamma[C,Q,3]`, `dgamma[C,Q,3]`, `I[C]`
* `vmap` over point blocks and coil blocks
* `scan` or static-trip-count `fori_loop` over quadrature chunks to avoid materializing a giant `P×C×Q×3` tensor
* keep chunk counts static in parity mode so reverse-mode differentiation remains available through `fori_loop`/`scan` lowering

This is much safer than a naive broadcast implementation, and JAX documents that `fori_loop` supports reverse-mode when the trip count is static because it lowers through `scan`. Do **not** use `jax.pure_callback` as the port: JAX’s docs say the callback runs on local CPU arrays, which defeats the purpose of a GPU backend.    ([JAX Documentation][5])

**Done when:** `BiotSavart` on JAX CPU matches the native CPU implementation for `B`, `A`, and first derivatives on randomized coil/point test sets.

### Phase 4: compile the full objective chain

Next port the objectives that matter most operationally:

* `SquaredFlux` in all three definitions
* curve length / curvature / distance penalties used in stage-2 optimization
* least-squares residual assembly where it can be made pure

The key is not just to JIT the field kernel; it is to JIT the entire chain:
`surface -> normals -> field -> B·n / penalties -> objective -> gradient`.

That is where JAX can beat the native code by fusing work and eliminating Python↔C++↔host traffic. The current test suite already locks down exact `SquaredFlux` definitions and runs Taylor checks on gradients, so you already have the right contract to validate against.   

**Done when:** `SquaredFlux.J()` and gradients agree across native CPU, JAX CPU, and JAX GPU within parity tolerances.

### Phase 5: continuous solvers first

Port the smooth / mostly-smooth solvers before the greedy ones:

* least-squares objective and Jacobian assembly
* relax-and-split style updates
* RCLS-like continuous subproblems
* PM matrix assembly (`A`, `A^T b`, normal-field contractions) as batched JAX contractions

For this repo, that should be split into two explicit subplans rather than one
generic “later PM/wireframe” placeholder:

* **PM continuous-subproblem plan**
  * freeze `PermanentMagnetGrid` geometry, fixed plasma surface data, and
    target `Bn` as immutable adapter inputs
  * first port the continuous PM matrix/objective path in
    `src/simsopt/solve/permanent_magnet_optimization.py`
  * keep `GPMO` and other discrete sparsification logic explicitly out of
    scope until the continuous parity gate is green
* **Wireframe continuous-subproblem plan**
  * freeze `ToroidalWireframe` geometry, segment metadata, and fixed-surface
    field-eval inputs as immutable payloads
  * first port the continuous `rcls` matrix/objective path in
    `src/simsopt/solve/wireframe_optimization.py` and
    `src/simsopt/field/wireframefield.py`
  * keep `gsco` current-addition and loop-validity logic explicitly out of
    scope until the continuous wireframe lane is green

Keep the outer optimizer API compatible. In the first pass, it is acceptable for the façade to keep a CPU optimizer call if the objective/gradient stay fully on JAX CPU/GPU; then move the full loop on-device once the kernels are stable. This is lower-risk than rewriting every solver at once.   

**Done when:** stage-two coil optimization plus one continuous PM solve slice
and one continuous wireframe `rcls` slice run end-to-end through the JAX
backend.

### Phase 6: greedy/discrete solvers later, with determinism rules

Only after the continuous path is correct should you port:

* GSCO
* GPMO variants
* MwPGP-related greedy or projected updates

These kernels are less “just dense linear algebra” than they look; the current C++ wireframe and PM code includes branching, loop-validity logic, complement updates, and discrete choices. Port them as masked batched scans with **stable tie-breaking**:

* fixed candidate ordering
* deterministic epsilon policy
* “smallest index wins” on ties
* fixed reduction chunking in parity mode

That is how you keep CPU/GPU trajectories aligned enough for reproducibility.  

**Done when:** greedy solver outputs match native solutions up to defined objective/final-state tolerances and stable tie-breaking rules.

### Phase 7: multi-GPU and multi-host scaling

Once single-device JAX is correct, scale over the large axes:

* field evaluation points
* perturbation samples
* design batches
* large candidate sets in PM/wireframe problems

Use explicit sharding and `shard_map`/`jax.shard_map` rather than treating multi-GPU as an afterthought; JAX’s docs position `shard_map` as a strong manual-parallelism route and explicit sharding as the modern model for data placement. For clusters, use `jax.distributed.initialize()` only after the single-host kernels are solid. ([JAX Documentation][6])

**Done when:** large point-cloud or batched-design runs scale across multiple GPUs without changing the public Simsopt API.

## 5) CPU/GPU parity and reliability contract

Do not promise universal bitwise equality. JAX explicitly says exact numerics are not guaranteed stable across releases, across accelerator platforms, or even within/without `jit`. So define parity in three layers instead: ([JAX Documentation][7])

1. **Algorithmic parity:** native CPU vs JAX CPU, same formulas, same quadrature, same chunking, same dtype.
2. **Device parity:** JAX CPU vs JAX GPU in x64 parity mode.
3. **Physics parity:** invariants and final objective quality, especially for greedy/discrete solvers.

Treat these as separate reporting buckets in CI, benchmarks, and proof runs.
Do not collapse them into one generic "parity" label.

Build CI around the tests you already have:

* JAX vs non-JAX curve families
* exact `SquaredFlux` definition checks
* Taylor tests for objective gradients
* magnetic-field invariants like divergence-free gradients, symmetry conditions, and `curl(A)=B` checks

The current test suite already contains those ingredients, which is a huge advantage.   

Add these runtime safeguards:

* `checkify` assertions for bad geometry, invalid indices, near-singular denominators, and NaN-producing branches
* `jax_debug_nans` in debug CI
* transfer guard to catch accidental host/device copies
* persistent compilation cache for reproducible compile behavior
* profiling and device-memory profiling as part of benchmark runs
* buffer donation on large solver-state arrays once kernels stabilize ([JAX Documentation][8])

Status in the current Columbia JAX tree:

* parity modes now keep x64 as a hard runtime requirement
* parity modes default transfer guard to `log`
* a dedicated CPU `jax_debug_nans` guardrail lane exists in `jax_smoke.yml`
* subprocess smoke coverage now checks a practical transfer-guard rejection case
  at a jitted NumPy-to-JAX boundary
* a user-facing backend guide now exists at
  `docs/using_jax_backend.md`

## 6) How the JAX backend surpasses native Simsopt

It will beat the native path only if you exploit what JAX does better than “Python calling C++”:

* **Fuse the full pipeline**, not just kernels. Native Simsopt crosses Python/C++ boundaries repeatedly; the JAX backend can compile geometry, field evaluation, objective assembly, and gradients into one executable graph. ([JAX Documentation][1])
* **Batch more aggressively** with `vmap`: evaluation points, perturbation ensembles, parameter sweeps, and multi-configuration design studies. Existing JAX geometry already points in this direction.  ([JAX Documentation][1])
* **Shard across large axes**, not tiny parameter vectors. Points, samples, and batched designs are the right scaling axes. ([JAX Documentation][6])
* **Use AD first, `custom_vjp` second.** Let JAX own most derivative plumbing; only add `custom_vjp` where profiling shows backward memory or runtime is the bottleneck. ([JAX Documentation][9])
* **Only drop lower**—FFI or custom low-level kernels—after profiling proves a hotspot remains. JAX has official FFI support, but that should be the last optimization step, not the first. ([JAX Documentation][10])

## 7) What not to do

Do **not**:

* try to make the current mutable cached `MagneticField` hierarchy itself the compiled JAX state,
* treat `jax.pure_callback` as a GPU port,
* start with VMEC/SPEC/Boozer/tracing/interpolated-field as the first-wave GPU target,
* or write a naive fully-broadcasted Biot–Savart that materializes `points × coils × quadrature` intermediates.   ([JAX Documentation][11])

## 8) The first milestone to build

The best first end-to-end milestone is:

**JAX `SurfaceRZFourier` + existing JAX curve families + pure-JAX `BiotSavart` + JAX `SquaredFlux` + stage-two continuous penalties + native CPU / JAX CPU / JAX GPU parity suite**

That milestone is narrow enough to finish cleanly, broad enough to unlock real GPU value, and directly aligned with how Simsopt is already used in stage-two optimization examples. It also creates the reusable kernel base for later wireframe and permanent-magnet work.    

That is the combined plan I would trust: **keep the public Simsopt surface, replace the numerical engine with a pure JAX core, validate native CPU → JAX CPU → JAX GPU, and only then expand to greedy solvers and multi-GPU scaling.**

[1]: https://docs.jax.dev/en/latest/stateful-computations.html?utm_source=chatgpt.com "Stateful computations"
[2]: https://docs.jax.dev/en/latest/multi_process.html?utm_source=chatgpt.com "Introduction to multi-controller JAX (aka multi-process ..."
[3]: https://docs.jax.dev/en/latest/default_dtypes.html?utm_source=chatgpt.com "Default dtypes and the X64 flag"
[4]: https://docs.jax.dev/en/latest/custom_pytrees.html?utm_source=chatgpt.com "Custom pytree nodes - JAX documentation"
[5]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html?utm_source=chatgpt.com "jax.lax.fori_loop"
[6]: https://docs.jax.dev/en/latest/notebooks/shard_map.html?utm_source=chatgpt.com "Manual parallelism with shard_map"
[7]: https://docs.jax.dev/en/latest/api_compatibility.html?utm_source=chatgpt.com "API compatibility"
[8]: https://docs.jax.dev/en/latest/debugging/checkify_guide.html?utm_source=chatgpt.com "The checkify transformation"
[9]: https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html?utm_source=chatgpt.com "jax.custom_vjp"
[10]: https://docs.jax.dev/en/latest/ffi.html?utm_source=chatgpt.com "Foreign function interface (FFI)"
[11]: https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html?utm_source=chatgpt.com "jax.pure_callback"

I reviewed the current port-in-progress, and overall my reaction is positive: this is **well past “prototype” stage**. The strongest parts are the **validation/parity ladder**, the **traceable objective path**, the explicit **reference → transitional → target** optimizer lanes, and the now-real immutable `jax_core` spec/pytree layer. The main weakness is no longer “missing a strict spec layer”; it is that architectural cleanup still matters because parts of the public implementation lean on mutable adapter state and flat-vector orchestration around that spec boundary, and the Biot–Savart kernel shape still needs a more GPU-native memory strategy.

## Overall verdict

This is **closer to the plan I recommended than I expected**.

The clearest evidence is that you already have:

* a distinct backend selector (`cpu` vs `jax`, plus JAX platform selection), 
* a `BoozerSurfaceJAX` class that explicitly defines `scipy` as the trusted reference lane, `hybrid` as the transitional lane, and `ondevice` as the target full-GPU lane, 
* a substantial parity and validation ladder covering Stage 2 value/gradient parity, single-stage init parity, run-code parity, and adjoint/FD validation,    
* and, most importantly, a **traceable target path** around `run_code_traceable()` / `make_traceable_objective()` that is explicitly intended to avoid the stateful path and route through JAX control flow, with `run_code_functional()` reduced to a thin alias over the same runtime-native schema.

That still leaves the public acceptance posture intentionally conservative:
the trusted public gates remain `scipy`-centric, while `hybrid` / `ondevice`
stay on a separate validation track until their own acceptance criteria close.

That aligns very well with the right JAX direction, because JAX transformations fundamentally want **pure functions**, not hidden mutable state. ([JAX][1])

## What is already strong

The best part of the current work is the **discipline of the migration strategy**. `optimizer_jax.py` documents a clear split between oracle/reference methods (`bfgs`, `lbfgs`), a transitional hybrid path, and target on-device methods (`bfgs-ondevice`, `lbfgs-ondevice`). That is exactly the right way to migrate a scientific codebase without losing trust in the results. 

The second major strength is the **testing philosophy**. The repo is not just testing “does it run?” It is testing:

* `SquaredFluxJAX` value parity and gradient parity against CPU,  
* isolated CPU vs JAX `run_code()` LS parity from the same initial guess, 
* Biot–Savart parity, `B_vjp` parity, and Maxwell-style invariants such as `∇·B = 0`, plus finite-difference checks on `dB/dX`,   
* and a much stronger-than-usual adjoint validation ladder with fixed-surface FD, full re-solve FD, recomposed gradient checks, and grouped-adjoint memory probes.   

That is exactly the kind of parity contract I recommended: not vague “GPU should match CPU,” but concrete staged checks.

The third major strength is the **traceable objective contract**. The goal is to make the single-stage objective fully JAX-traceable so the outer optimizer can route through JAX control flow rather than the old host-callback bridge, with the actual trace-safe route built around `run_code_traceable()` plus the traceable objective builders. That is the right long-term architecture, because it matches JAX’s pure-function model. ([JAX][1])

You are also testing purity rather than just assuming it. The traceable-path tests snapshot `bs_jax.x`, surface DOFs, `booz_jax.res`, dirty flags, and caches, then verify the traceable objective does **not** mutate them and does **not** accumulate child graph state.   That is very good engineering.

## Where the current port still diverges from the best architecture

The main issue is that the project currently has **two architectural stories at once**:

1. the **correct future story**: pure JAX array-backed kernels, functional inner solves, custom VJPs, traceable objective, no stateful `run_code()` dependency, 

2. the **legacy-compatible present story**: `Optimizable` wrappers, `need_to_run_code`, cached `res`, mutable point state in `BiotSavartJAX`, and adapter methods that still behave like stateful Simsopt objects.  

That is fine temporarily, but the remaining work is to keep that layered design
explicit outside the hot path as well, rather than letting the mixed
mutable-wrapper / pure-kernel style harden into permanent complexity.

In particular, the backend abstraction is now materially better than it was
when this note was first written. The public runtime modes now exist, but the
broader façade/package split and full mode-owned numerical policy are still not
completely finished. The target distinction remains:

* `native_cpu`
* `jax_cpu_parity`
* `jax_gpu_parity`
* `jax_gpu_fast`

What is still incomplete is making every important numerical policy live under
that runtime contract, rather than leaving some of it distributed across tests
and benchmark scripts.

## Biggest technical risk: the Biot–Savart kernel shape

This is the one place I would change first.

Your current JAX Biot–Savart path is already far enough along to support parity testing and VJP testing, which is a major milestone. The tests show `BiotSavartJAX.B()` and `B_vjp` are being compared directly against CPU, and the pure kernel path is already exposed and validated.   

But against the plan I recommended, the likely next bottleneck is **memory scaling**. The right GPU design for Biot–Savart is usually a **chunked reduction** over quadrature or coil blocks, using `scan` or static-trip-count `fori_loop`, not a naive full broadcast over all coils/quadrature for every point. JAX explicitly notes that `scan` is the right primitive for static iteration and that `fori_loop` lowers through `scan` when the trip count is static, which is exactly what you want for differentiable chunked reductions. ([JAX][2])

So my feedback here is:

**What you have is already scientifically useful. What you need next is the memory-safe GPU version of the same kernel.**

That means:

* fixed-shape chunk sizes in parity mode,
* reduction over quadrature/coils with `scan`/static `fori_loop`,
* no giant point×coil×quad intermediates in the hot path,
* and only later any lower-level kernel work if profiling proves it is needed.

## Strong sign you are on the right path: no `pure_callback` fake port

One thing I specifically wanted to avoid was a “GPU port” that still quietly runs the heavy numerics on CPU through callbacks. Your current direction appears better than that: the traceable objective tests explicitly emphasize the pure JAX route, and your design is centered on JAX-native kernels and on-device optimizers rather than callback-wrapping legacy C++.  JAX’s own docs are clear that `pure_callback` hands arrays to a local CPU callback, so avoiding that as the main port path is absolutely right. ([JAX][3])

## The parity story is excellent, but GPU CI is still too soft

This is the biggest process gap I see.

You have a very impressive validation ladder, including:

* tiered tolerances in `validation_ladder_contract.py`,
* stage-2 parity probes, single-stage init parity, production-grid Boozer diagnostics, adjoint FD validation, and grouped-adjoint memory probes,
* and external GPU-proof tooling.

But the routine GitHub JAX workflow shown in `jax_smoke.yml` is still **CPU-only**. It installs `jax==0.9.2` / `jaxlib==0.9.2`, runs public-lane CPU tests, integration tests, and lint, but not a real GPU parity lane. 

That means the codebase already has **GPU proof infrastructure**, but not yet **GPU regression infrastructure**.

My recommendation is to add at least one automated GPU lane that runs a minimal but high-value subset:

* `tests/field/test_biotsavart_jax.py`
* Stage 2 value/gradient parity
* one single-stage init parity probe
* one grouped-adjoint smoke/memory probe

Even a small GPU lane is much better than relying entirely on ad hoc external proof runs.

## The import/package story still needs cleanup

This is a smaller issue, but it matters if you want the JAX backend to become a clean first-class part of Simsopt.

You already added proper import smoke tests through the real `simsopt` package entrypoints, specifically to catch import-chain regressions in a no-`simsoptpp` environment. That is good. 

At the same time, a number of lower-level JAX tests still import files directly via `importlib.util`, create stub `simsopt`, `simsopt.geo`, and `simsopt.field` packages in `sys.modules`, load modules manually, then clean up those stubs.  

That is acceptable during migration, but it is also a sign that the internal package boundaries are not fully clean yet. I would treat this as a signal to continue moving toward:

* package-stable pure modules,
* fewer relative-import hacks,
* and a clearer separation between public adapters and internal pure kernels.

## The optimizer strategy is good, but version-locking is a real debt

`optimizer_jax.py` explicitly says the private optimizer methods mirror JAX 0.9.2 semantics so line-search and iteration behavior stay stable. I think that is a reasonable short-term move, because scientific reproducibility matters. 

But it is also technical debt.

So my recommendation is:

* keep the pinned-private optimizer approach for now,
* but isolate it hard,
* and make the long-term contract be **objective/gradient parity and solver outcome quality**, not exact mirroring of one JAX version’s optimizer behavior.

That matters because JAX itself does not guarantee exact numerics across versions, devices, or even in/out of `jit`. ([JAX][4])

## My prioritized feedback

### 1. Broaden the architecture you already started to formalize

Treat the current code as two layers:

* **pure JAX kernel/spec layer**
* **legacy-compatible adapter layer**

Do not keep adding ad hoc traceable escape hatches. Instead, make the pure layer the official implementation target.

The strongest clue that this is the right next step is your own `run_code_traceable()` / traceable-objective contract.

### 2. Rework Biot–Savart around chunked reductions

This is the single highest-value technical change for GPU reliability and scalability.

Use:

* static chunk counts,
* `scan` or static `fori_loop`,
* parity mode with fixed chunking,
* then fast mode later with tuned chunk sizes.

That aligns with JAX’s control-flow guidance and will help both memory behavior and differentiation. ([JAX][2])

### 3. Finish mode-owned numerical policy

The repo now has first-class backend modes. The remaining work is to tie them
fully to:

* x64 enablement,
* chunking policy,
* tolerance policy,
* guardrail policy,
* and CI expectations.

### 4. Add a real automated GPU parity lane

Keep production GPU proof, but do not let it be the only GPU correctness signal. `jax_smoke.yml` is already a good scaffold; it just needs one true GPU lane.

### 5. Keep SciPy as the trusted solver oracle longer than you may want

Your current `scipy` / `hybrid` / `ondevice` split is good. Resist the temptation to over-expand on-device solver coverage before the kernel layer is fully stable.  

### 6. Add JAX-native runtime safeguards

I did not see `checkify` wired into the main path here. I would add:

* `checkify` for jit-able numerical guards,
* `jax_debug_nans` in debug lanes,
* transfer guard in test/probe runs,
* persistent compilation cache for parity/proof environments.

The `jax_debug_nans` lane and transfer-guard/x64 guardrails are now in place; the
remaining runtime-safeguard gap here is mostly `checkify`.

## My summary judgment

This port is **substantially on the right track**.

What I would say bluntly is:

**The science-validation side is ahead of the architecture side.**
That is not a bad place to be. It means the team is being careful about trust. But now is the moment to prevent the current mixed style from hardening into permanent complexity.

The best path forward is:

* keep the excellent validation ladder,
* keep the trusted `scipy` reference lane,
* keep the functional traceable objective work,
* and now convert that into a formally layered architecture with a chunked pure-JAX Biot–Savart core and an explicit parity/fast runtime model.

If you want, I can turn this into a **line-by-line implementation review checklist** organized by file/module (`backend.py`, `biotsavart_jax.py`, `biotsavart_jax_backend.py`, `boozersurface_jax.py`, `optimizer_jax.py`, tests/CI).

[1]: https://docs.jax.dev/en/latest/stateful-computations.html?utm_source=chatgpt.com "Stateful computations"
[2]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html?utm_source=chatgpt.com "jax.lax.fori_loop"
[3]: https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html?utm_source=chatgpt.com "jax.pure_callback"
[4]: https://docs.jax.dev/en/latest/api_compatibility.html?utm_source=chatgpt.com "API compatibility"

According to a document from March 31, 2026, here is the module-by-module implementation review checklist for the current Simsopt JAX/GPU port, merged with the architecture I recommended earlier. The short version is: **the port is on the right path**. The strongest pieces are already in place: a backend switch, a real pure-JAX Biot–Savart kernel, pure-JAX surface geometry and label constraints, a lane-aware `BoozerSurfaceJAX`, a callback-free traceable objective, and a serious parity/validation ladder. The main thing to tighten now is the separation between the **pure JAX kernel layer** and the **legacy mutable adapter layer**, because JAX transformations work best on pure functions with explicit state, not hidden object mutation.       ([JAX Documentation][1])

## Top priorities

1. **Add a minimal automated GPU CI lane.**
   The highest-value next product step is now a small always-on GPU lane that
   exercises the chunked grouped-field path, one Stage 2 mixed-quadrature
   parity probe, and one Boozer grouped-spec smoke.

2. **Keep the low-level `biotsavart_jax.py` chunked rewrite narrow and proven.**
   The current kernel is correct and already useful, and the low-level
   chunking slice now covers the first coil-axis reduction plus exact two-chunk
   coil/quadrature hot spots. The next kernel work is only to broaden chunking
   where profiling still shows remaining quadrature/block materialization
   pressure, with the existing memory-scaling benchmarks as the gate.

3. **Broaden the pure JAX layer beyond the current hot-path coverage.**
   You already have it concretely via `make_traceable_objective()`, `run_code_traceable()`, the `jax_core` subtree, and the current immutable spec pytrees. The next step is to widen that coverage instead of letting pure kernels and mutable wrapper/orchestration code keep mixing.

   The hot-path spec layer is already broader than a toy first slice: concrete
   curve/current/coil/field-eval specs and `to_spec()` adapters already exist
   for the main Fourier/coil objects plus the current full-graph wrapper
   families. The remaining work is to extend that coverage through any
   residual legacy wrapper families and thread it through the remaining
   non-hot-path objective families.

   `BiotSavartJAX` is also thinner than the earlier draft state: the old CPU
   curve-geometry / coil-pullback paths are gone, explicit DOF
   reconstruction now uses immutable per-coil specs, and legacy curve families
   without immutable-spec support are rejected until a native spec is added.
   In strict mode, hidden grouped-array/live-graph spec reconstruction is
   rejected explicitly instead of being taken silently.

   The current objective cleanup slice is now materially better than the
   earlier draft state:
   * `SquaredFluxJAX` and the Stage 2 scalar target bundle share immutable
     fixed-surface setup through `FieldEvalSpec` + `FixedSurfaceFluxSpec`
   * the Stage 2 target lane now covers all three `SquaredFlux` definitions
   * target-objective gradients now have centered-FD plus first-order Taylor
     checks
   * the traceable single-stage path keeps the explicit `surface_kind`
     contract instead of relying on hidden geometry reconstruction
   * the stateful single-stage wrappers now require streaming grouped adjoint
     callbacks (`res["vjp_groups"]`) instead of silently carrying the legacy
     full-pytree adjoint path
   * the single-stage `ondevice` outer lane now consumes the scalar
     traceable objective directly instead of the older explicit
     `(value, grad)` adapter contract
   * Stage 2 and single-stage outer continuous solver routing now share one
     explicit `ContinuousOptimizerContract`; the supported
     `scipy` / transitional `hybrid` / target `ondevice` split is defined in
     one place instead of being re-decided per entrypoint
   * the current Stage 2 route helpers are internally consistent:
     `resolve_stage2_optimizer_contract(...)` returns the full contract and
     only `resolve_stage2_optimizer_method(...)` strips that down to the
     method string; both Stage 2 and single-stage now share
     `resolve_outer_loop_optimizer_contract()` in `optimizer_jax.py` which
     bakes in the common `limited_memory=True, allow_hybrid=False` policy
   * unsupported single-stage outer `hybrid` routing is now rejected
     explicitly instead of silently degrading to plain `lbfgs`
   * the next broader non-hot-path objective slice is now landed:
     `NonQuasiSymmetricRatioJAX` and `BoozerResidualJAX` route their direct
     coil/objective terms through immutable coil-spec reconstruction
   * the traceable single-stage scalar objective now also keeps its iota-target
     penalty on the same pure immutable-spec forward path instead of dropping
     back to the inner Boozer objective value
   * active non-strict JAX mode now warns once per component/detail when code
     crosses explicitly supported compatibility seams instead of silently
     mixing legacy CPU paths into JAX execution; strict mode rejects those same
     seams
   * stale warning-covered CPU geometry/pullback paths have been replaced
     by immutable-spec contracts in `BiotSavartJAX` and `SquaredFluxJAX`;
     unsupported inputs are rejected instead of routed through CPU objective
     or coil-pullback paths

4. **Finish mode-owned numerical policy.**
   Backend modes are now real, and chunk tuning is materially more centralized
   than before, but tolerance, provenance, and the last numerical policy seams
   still need to be fully unified under them.

5. **Tighten packaging/version contracts.**
   Workflows and env files pin JAX 0.9.2, but package metadata still claims support for much older JAX/JAXLIB.  

6. **Keep the SciPy/reference lane longer than you may want.**
   Your current `scipy` / `hybrid` / `ondevice` split is a strength. Preserve it while the pure kernel layer matures.  

---

# File-by-file checklist

## `src/simsopt/backend.py`

### Keep

* The file already gives one source of truth for code-path backend (`cpu` vs `jax`) and JAX platform (`cpu` vs `cuda`), with legacy env fallback. That is a good starting abstraction. 

### Change now

* Replace the two-axis public contract with explicit runtime modes:

  * `native_cpu`
  * `jax_cpu_parity`
  * `jax_gpu_parity`
  * `jax_gpu_fast`

* Make each mode define:

  * x64 policy
  * chunk-size policy
  * tolerance policy
  * compilation-cache policy
  * provenance label

Right now that policy lives mostly in tests and benchmark scripts instead of in the runtime contract.  

### Later

* Add helper functions like `is_parity_mode()`, `requires_x64()`, `default_chunk_policy()`, `default_tolerance_tier()`.

### Done when

* A benchmark or test can say “run `jax_gpu_parity`” and get all numerical policy from one place, not from ad hoc flags.

---

## `src/simsopt/field/biotsavart_jax.py`

### Keep

* This is a real pure JAX kernel, not a callback wrapper.
* The math is clear and traceable.
* `biot_savart_B` is already `@jax.jit` and vectorized over points with `vmap`.
* You already expose `B`, `A`, `dA/dX`, `dB/dX`, `B_vjp`, and grouped helpers.  

### Change now

This is the most important technical change.

The current structure is:

* `_biot_savart_one_point(x, gammas, gammadashs, currents)`
* full broadcast over all coils and quadrature
* `vmap` over points

That is good for correctness, but dangerous for GPU scaling. The next version should use **chunked reductions**:

* chunk over quadrature and/or coil groups
* keep chunk counts static in parity mode
* use `lax.scan` or static-trip-count `lax.fori_loop`
* reduce partial sums incrementally rather than materializing all intermediates at once

JAX’s docs explicitly note that `fori_loop` uses `scan` when the trip count is static, and reverse-mode is supported in that case. That is exactly the pattern you want for a differentiable, memory-safe Biot–Savart reduction.  ([JAX Documentation][2])

A concrete target split:

* `biot_savart_B_chunk(points_block, gamma_block, gammadash_block, current_block)`
* `biot_savart_B_chunked(points, grouped_coils, chunk_spec)`
* parity mode uses fixed chunk sizes
* fast mode may autotune chunk sizes per backend

### Also change

* Make quadrature normalization explicit and consistent across grouped and ungrouped paths.
* Add optional `precision` control on reductions only if later profiling shows a need.

### Later

* If profiling still shows Biot–Savart dominating after chunking, only then consider lower-level work such as JAX FFI or Pallas. Do not start there. ([JAX Documentation][3])

### Done when

* Large point clouds and larger coil sets run on GPU without OOM risk.
* JAX CPU and JAX GPU still satisfy current parity tests in x64 mode.

---

## `src/simsopt/field/biotsavart_jax_backend.py`

### Keep

* The module is already architecturally correct in one key way: `BiotSavartJAX` is an adapter and does **not** inherit from `sopp.BiotSavart` or `sopp.MagneticField`. That is exactly the right direction. 
* The one-coil VJP helper is a good instinct for memory control. Its own comment says it avoids materializing grouped multi-coil intermediates. 

### Change now

* Narrow the responsibility of this class to:

  * extracting immutable coil/spec arrays from the `Optimizable` tree
  * storing user-facing point state
  * converting pure JAX gradients back into Simsopt `Derivative` structures

Do **not** let this adapter become the place where numerical policy, chunking, or special-case math lives.

* Refactor grouped extraction into a stable pure helper that returns a pytree-like spec:

  * grouped `gammas`
  * grouped `gammadashs`
  * grouped `currents`
  * metadata for chunking and parity

* Treat `_points_jax`, `_points_version`, and other mutable fields as adapter-only concerns. Keep them outside compiled kernels.

### Watch item

* The current native fast path only supports coils that can be reconstructed as JAX-native geometry, which is fine for Stage 2. Keep that limitation explicit in docs and errors, rather than letting it look like universal GPU support. 

### Later

* Add a `to_spec()` export so this object can feed multi-GPU sharding code directly.

### Done when

* Every call from `BiotSavartJAX.B()` to the real numerical kernel goes through a pure array/spec function.

---

## `src/simsopt/geo/surface_fourier_jax.py`

### Keep

* This is one of the cleanest parts of the port.
* It is already a pure JAX replacement for the C++ `SurfaceXYZTensorFourier` geometry path.
* It exposes the right primitives: basis builders, `gamma`, `gammadash1`, `gammadash2`, `normal`, area, volume, coefficient derivatives, and scatter helpers. 

### Change now

* Precompute and cache basis matrices and scatter topology at the adapter/spec level, not deep inside repeatedly called kernels.
* Make the stellarator-symmetry assumptions explicit in function contracts. The module already says forbidden entries must be zeroed by the caller. Keep that strict. 

### Add tests

* random-DOF parity tests for:

  * `gamma`
  * `normal`
  * `surface_area`
  * `surface_volume`
  * first derivatives with Taylor checks

You already have label-constraint FD tests that lean on these functions, which is good. 

### Later

* Consider adding `SurfaceRZFourier`-specific pure kernels if you want the public geometry story to align more closely with classic Simsopt users.

### Done when

* Geometry kernels are fully spec-driven and reused by Boozer, Stage 2, and label constraints without wrapper-specific duplication.

---

## `src/simsopt/geo/label_constraints_jax.py`

### Keep

* This file is exactly the kind of thing that should be JAX-native.
* It already states the right goal: replace CPU `Volume.J()` / `ToroidalFlux.J()` inside the penalty objective so the inner solve stays on-device.
* It is already pure and traceable. 

### Change now

* Add `checkify`-style guards around invalid geometry or obviously bad inputs once you wire reliability checks into the JAX lanes. JAX’s `checkify` is designed for jit-compatible runtime checks. ([JAX Documentation][4])

### Small cleanup

* `compute_G_from_currents()` currently uses `sum(abs(currents))`. Keep that, but document clearly that this is a model convention and not a generic “differentiate everywhere smoothly” quantity. It may matter for edge cases if current signs are allowed to cross zero. 

### Done when

* These functions are the only path used by on-device Boozer penalty objectives.

---

## `src/simsopt/objectives/fluxobjective_jax.py`

### Keep

* `SquaredFluxJAX` is thoughtfully designed.
* The docstring is honest: it is a drop-in replacement, uses the native JAX path when the field adapter exposes immutable specs, and rejects unsupported fields instead of routing through a mixed CPU/JAX path.
* It also correctly states that surface geometry is frozen at construction, which is valid for Stage 2 because the plasma surface is fixed. 

### Change now

* Make that Stage-2-only assumption first-class in the class contract. Right now it is documented, but not strongly enforced. If someone later uses it with a mutable surface, they could get silently stale geometry. 
* Add an explicit invalidation or reconstruction path if future use cases need movable surfaces.

### Architectural note

* This module is already very close to the right design: pure kernel underneath, adapter behavior above.
* Keep the native JAX path separate from explicit compatibility/reporting boundaries; do not advertise host compatibility paths as “fully GPU.”

### Add tests

* A test that changing surface DOFs after construction is either rejected or triggers rebuild, rather than silently reusing stale cached geometry.

### Done when

* `SquaredFluxJAX` is explicitly scoped as fixed-surface or explicitly handles surface invalidation.

---

## `src/simsopt/geo/boozer_residual_jax.py`

### Keep

* The composed-derivative architecture is good, and `BoozerSurfaceJAX` explicitly builds on `_surface_geometry_from_dofs`, `boozer_residual_scalar`, `boozer_residual_vector`, and coil VJPs from this layer. 

### Change now

* Make this the canonical **pure residual layer**.
* Ensure every function here accepts only:

  * surface spec / surface DOFs
  * coil spec / grouped coil arrays
  * scalar parameters (`iota`, `G`, weights, labels)
* No reads from `self`, no reliance on `Optimizable`, no cached mutable objects.

### Done when

* `BoozerSurfaceJAX` becomes a façade over this residual layer rather than mixing residual math and object semantics.

---

## `src/simsopt/geo/boozersurface_jax.py`

### Keep

* The lane-aware design is excellent:

  * `scipy` = trusted reference
  * `hybrid` = transitional
  * `ondevice` = target GPU lane
    This is the right migration strategy for a scientific codebase. 
* Keeping the public CPU `BoozerSurface`-like API is also a good choice. 

### Change now

This is the file where the mixed architecture is most visible.

The code already admits that:

* `run_code()` is still stateful
* `run_code_traceable()` keeps the inner solve on explicit array state
* full traceability is one layer up in `make_traceable_objective()` 

That is the right diagnosis. So the implementation decision should be:

**Do not spend too much effort trying to make a legacy host wrapper the final JIT/grad path.**
Make `make_traceable_objective()` the canonical compiled path.

### Specific refactors

* Pull solver state into immutable data records:

  * warm-start state
  * baseline PLU state
  * solver options
  * label/constraint spec
* Keep `need_to_run_code`, `self.res`, and surface mutation only in the façade.
* Move anything numerically meaningful into pure helper functions.

### Watch item

* If label behavior is currently routed by string names or class-name matching, replace that with a small explicit protocol or enum. That will be less brittle for tracing and testing.

### Done when

* `BoozerSurfaceJAX` is mostly orchestration and state-compatibility, while the math lives below it.

---

## `src/simsopt/geo/surfaceobjectives_jax.py`

### Keep

* This is the best architectural piece in the port.

* `make_traceable_objective()` already states the exact contract I wanted:

  * pure function `f(coil_dofs) -> scalar`
  * no object mutation
  * no `jax.pure_callback`
  * backward path expressed with pure JAX arrays
  * requires the `ondevice` optimizer backend for LS cases 

* The tests around it are also excellent:

  * `make_jaxpr` must succeed
  * the jaxpr must not contain `pure_callback`
  * it must not depend on `run_code()`
  * it must not mutate state
  * traceable and explicit paths must agree for a short on-device L-BFGS run  

### Change now

* Promote this module to the official outer-loop implementation target.
* Package the warm-start predictor state and objective kwargs into immutable spec/state objects instead of large ad hoc closures.

### Add reliability hooks

* Put optional `checkify` guards around:

  * singular or near-singular linear solves
  * invalid warm-start predictors
  * NaN residuals
  * bad shapes/indexing in symmetry scatter operations
    JAX’s `checkify` is a good fit here because it works under JIT. ([JAX Documentation][4])

### Done when

* The single-stage outer loop uses this path by default for the true on-device lane.

---

## `src/simsopt/geo/optimizer_jax.py`

### Keep

* This file is in good shape.
* The lazy private-package import is the right compatibility move.
* The public module staying free of `jax._src` imports is the correct policy, and you already test it.  
* The `scipy` / `hybrid` / `ondevice` backend contract is clean. 

### Change now

* Make runtime lane/provenance more explicit in every result object and benchmark payload:

  * optimizer lane
  * JAX/JAXLIB version
  * x64 enabled
  * compile behavior
  * platform request
  * chunk policy

You already print a lot of provenance in benchmark scripts. Fold the critical subset into reusable helper functions.  

### Keep for now

* The private optimizer package mirroring JAX 0.9.2 semantics.
  That is technical debt, but it is acceptable debt for now because it stabilizes migration. 

### Later

* Loosen the solver contract away from “must mirror one JAX version” and toward “must meet outcome and parity tolerances.”

### Done when

* Solver semantics are isolated and reproducible, but the rest of the codebase does not depend on optimizer-version quirks.

---

## `tests/field/test_biotsavart_jax.py` and `tests/field/test_biotsavart_jax_parity.py`

### Keep

These are excellent.
You are already testing:

* analytic circular-loop reference
* `∇·B = 0`
* C++ parity when available
* quadrature convergence
* `B = curl(A)`
* derivative Taylor tests
* VJP Taylor tests
* grouped-coil self-consistency  

### Change now

* Add stress tests sized to exercise chunking once the kernel is reworked:

  * many points, moderate coils
  * moderate points, large quadrature
  * mixed grouped quadrature
* Split correctness tests from performance-sensitive tests so CI stays fast.

### Done when

* These tests pass unchanged after the chunked kernel rewrite.

---

## `tests/geo/test_label_constraints_jax.py`

### Keep

* Very strong contract for area/volume/flux gradient correctness, and importantly it avoids `simsoptpp`. 

### Change now

* Reduce the amount of manual import bootstrapping over time.
  The current stub-package and direct-file loading is acceptable during migration, but it is also a packaging smell. 

### Done when

* The test imports through real package paths without stub package injection.

---

## `tests/integration/test_jax_native_path.py`

### Keep

* This is the right “pure JAX objective path” test.
* It validates the native Fourier-basis forward/gradient path and shared-DOF gradient accumulation without depending on `simsoptpp`. 

### Change now

* Keep this test as the minimum “no callback / no simsoptpp / pure objective” contract. It should remain green before any on-device solver changes are considered finished.

### Done when

* This is the canonical smoke test for the pure Stage 2 JAX path.

---

## `tests/integration/test_stage2_jax.py`

### Keep

* The test contract is good:

  * objective value parity
  * gradient parity
  * short optimization quality comparison 

### Change now

* Keep the explicit distinction between:

  * matched-state gradient parity
  * final-solution quality parity
  * barrier-edge portability allowances
    That nuance is scientifically important, especially around curvature barriers. 

### Done when

* The same suite runs in both CPU and GPU parity lanes.

---

## `tests/integration/test_single_stage_jax.py` and `tests/geo/test_boozersurface_jax.py`

### Keep

* These are unusually strong integration tests.
* They validate adjoint consistency, finite direct gradients, backend selection, and the traceable objective contract.   

### Change now

* Continue to treat these as the gate for “ondevice lane is real.”
* Add a GPU-marked subset that is small but always run in automated GPU CI.

### Done when

* One minimal on-device path is automatically exercised on real GPU hardware in CI.

---

## `benchmarks/*` and validation ladder

### Keep

* The benchmark/probe framework is one of the strongest parts of the port.
* It already captures provenance, tolerances, lane identity, compile behavior, and parity comparisons.
* Tiered tolerances for Stage 2 and adjoint validation are clearly in use.  

### Change now

* Promote a small subset of these probes into automated CI, not just manual or external benchmarking.
* Keep the rest as heavier nightly or benchmark runs.
* Standardize recurring benchmark reporting around:
  * a checked-in manifest describing stable hardware, command contract, and
    artifact paths
  * one shared markdown report template
  * one lightweight renderer from JSON payload to markdown report
  * a scheduled stable-hardware workflow instead of ad hoc local runs

### Done when

* Every merge gets at least one real GPU parity signal, not just CPU smoke.

---

## `.github/workflows/jax_smoke.yml` and CI

### Keep

* The workflow already targets the right files and runs public/private lanes separately.
* Ruff checks over the JAX modules are good hygiene.   

### Change now

Add one real GPU job with a minimal set:

* `tests/field/test_biotsavart_jax.py`
* `tests/field/test_biotsavart_jax_parity.py`
* one Stage 2 value/gradient parity probe
* one single-stage traceable ondevice smoke
* optionally one grouped-adjoint smoke

That is the biggest process gap right now.

### Done when

* A regression in the actual GPU lane is caught automatically.

---

## Packaging and environment files

### Keep

* The dedicated env file pinning `jax==0.9.2` / `jaxlib==0.9.2` is good for reproducibility. 

### Change now

* Update package metadata and build recipe to reflect reality.
  Right now:
* env/workflows pin JAX 0.9.2
* but `conda.recipe/meta.yaml` still allows `jax >=0.2.5` and `jaxlib >=0.1.56`  

That mismatch is too wide.

### Recommendation

* Either pin tightly for the JAX lane, or split extras:

  * base `simsopt`
  * `simsopt[jax]`
  * `simsopt[jax-dev]`

### Done when

* Users cannot accidentally install a nominally “supported” but actually untested JAX stack.

---

## Transitional import machinery and smoke tests

### Keep

* `tests/test_jax_import_smoke.py` is good and important.
  It verifies:
* package-root import without `simsoptpp`
* public JAX entrypoints import cleanly
* lazy optimizer import behavior
* public methods work without the private package
* private methods fail clearly when blocked
* no `jax._src` imports leak into public or private optimizer modules  

### Change now

* Reduce dependence on meta-path patching and manual `importlib` file loading over time.
  The current editable-finder patch in `tests/integration/conftest.py` is useful, but it should remain temporary. 

### Done when

* Real package imports work without test-only import surgery.

---

# What I would not change right now

* I would **not** replace the SciPy reference lane yet.
* I would **not** rewrite all solver logic before finishing the Biot–Savart chunked kernel.
* I would **not** drop to FFI/Pallas unless profiling still shows a true hotspot after the kernel rewrite.
* I would **not** try to make every legacy `Optimizable` object itself JAX-transformable. Keep the façade; make the numerical core pure.   ([JAX Documentation][1])

# The best next implementation sequence

1. **Minimal GPU CI lane**

   * grouped Biot-Savart parity on the chunked grouped path
   * one Stage 2 mixed-quadrature parity probe
   * one Boozer grouped-spec smoke

2. **Biot–Savart kernel rewrite**

   * preserve the landed low-level chunking rewrite in `biotsavart_jax.py`
   * only broaden beyond the landed two-chunk hot spots if profiling still
     justifies it
   * preserve current parity tests as the oracle  ([JAX Documentation][2])

3. **Adapter and spec cleanup**

   * narrow `BiotSavartJAX` and `BoozerSurfaceJAX` to façade roles

4. **SurfaceRZFourier consumer migration**

   * keep downstream callers on the landed JAX geometry/derivative contract
   * only broaden new consumers where parity coverage exists

5. **Traceable path promotion**

   * treat `make_traceable_objective()` as the real on-device path 

6. **Packaging/version cleanup**

   * align envs, workflows, and package metadata

# Bottom line

The current port is **good**. It is already doing several hard things right:

* real pure-JAX kernels,
* a callback-free traceable objective,
* a careful reference/transitional/target optimizer split,
* and a parity ladder that is much stronger than most GPU ports.     

The most important correction is not conceptual, it is structural:

**finish the separation between pure JAX kernels and legacy mutable adapters, and make Biot–Savart memory-safe on GPU.**

If you want, I’ll turn this into a **PR-ready action matrix** with columns: `file`, `issue`, `priority`, `exact change`, `risk`, `test to add/update`, and `acceptance criterion`.

[1]: https://docs.jax.dev/en/latest/stateful-computations.html?utm_source=chatgpt.com "Stateful computations"
[2]: https://docs.jax.dev/en/latest/_autosummary/jax.lax.fori_loop.html?utm_source=chatgpt.com "jax.lax.fori_loop"
[3]: https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html?utm_source=chatgpt.com "jax.pure_callback"
[4]: https://docs.jax.dev/en/latest/debugging/checkify_guide.html?utm_source=chatgpt.com "The checkify transformation"
