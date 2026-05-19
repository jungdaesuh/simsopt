# Runtime Dtype Policy Implementation Plan

Date: 2026-05-19

## Purpose

Make dtype selection a backend-policy-owned runtime contract instead of a
collection of hard-coded `float64` assumptions spread across import hooks and
JAX helper modules.

The immediate problem is that the `jax_metal_smoke` backend declares
`requires_x64=False`, but the current kernel-helper boundary still rejects
actual `float32` inputs through hard-coded `float64` guards. The broader
problem is ownership: `simsopt.__init__`, `BackendPolicy`,
`_math_utils.py`, optimizer defaults, and helper modules each encode part of
runtime behavior. This plan restores one source of truth.

## Goals

- [ ] `BackendPolicy` owns runtime dtype and host dtype.
- [ ] `jax_metal_smoke` uses `float32` policy dtype without importing JAX from
  `simsopt.backend.runtime` at module scope.
- [ ] `jax_cpu_*`, `jax_gpu_*`, and `native_cpu` keep `float64` policy dtype.
- [ ] `_math_utils.py` accepts policy dtype when converting or validating runtime
  arrays.
- [ ] The Metal float32 path becomes reachable by construction.
- [ ] Import-time x64 behavior has one explicit owner and one documented
  contract.
- [ ] Cross-platform CI can test the policy mapping without requiring
  `jax-metal`.
- [ ] A separate opt-in Metal smoke can prove the real backend path on a host
  with `jax-metal` installed.

## Non-Goals

- [ ] Do not prove full Metal numerical parity in the isolated PR.
- [ ] Do not collapse all remaining `_as_jax_float64` duplicates in the same PR.
- [ ] Do not add dynamic imports.
- [ ] Do not add fallback behavior that silently reroutes Metal or target lanes
  to CPU/reference paths.
- [ ] Do not import `jax` or `jax.numpy` at module scope in
  `src/simsopt/backend/runtime.py`.
- [ ] Do not loosen tolerances to hide dtype drift.

## Current Tree Facts

- `src/simsopt/geo/surface_fourier_jax.py` is already a 9-line shim into
  `simsopt.jax_core.surface_fourier_kernels`.
- `src/simsopt/__init__.py` still sets `jax_enable_x64=True` when `jax` is
  already imported, and otherwise sets `JAX_ENABLE_X64=True`.
- `BackendPolicy` currently has `requires_x64` but does not have
  `runtime_dtype`, `host_dtype`, `default_residency`,
  `default_optimizer_backend`, or `disable_jit`.
- `src/simsopt/geo/optimizer_jax.py` still has
  `VALID_OPTIMIZER_BACKENDS = frozenset({"scipy", "ondevice"})`; there is no
  `"auto"` optimizer routing.
- There is no `src/simsopt/backend/dtypes.py`.
- There is no `use_runtime(...)` wrapper.
- There is no `runtime_device_put`.
- `_math_utils.py` hard-codes `_FLOAT64_DTYPE = np.dtype(np.float64)`.
- `_math_utils.require_float64_dtype(...)` compares inputs directly against that
  hard-coded dtype.
- `_math_utils.as_runtime_float64(...)` calls `require_float64_dtype(...)` for
  both `reference` and `value`.
- `boozer_residual_jax.py` imports `require_float64_dtype` and uses it for
  `B`, `xphi`, and `xtheta`.
- `set_backend(..., configure_runtime=False)` exists and can exercise policy
  resolution without touching `jax.config`.
- `SIMSOPT_BACKEND_MODE=jax_metal_smoke` import is not a cross-platform test:
  runtime config writes `jax_platforms=METAL` and validation fails on hosts
  without the Metal backend.

## Design Principles

- [ ] Keep `src/simsopt/backend/runtime.py` as the policy SSOT.
- [ ] Keep JAX imports out of `runtime.py` module scope.
- [ ] Store dtype policy in `runtime.py` as strings, not `jnp.dtype` objects.
- [ ] Resolve string dtype names to `jnp`/`np` dtypes inside modules that already
  import JAX or NumPy.
- [ ] Keep strict contracts: unsupported dtype or backend states should fail
  clearly.
- [ ] Separate reachability from real hardware proof. A policy unit test can
  prove that the float32 lane is selectable; only a Metal smoke can prove that
  `jax-metal` executes the path.

## Open Contract Decision

Step 1 must choose and document one of these contracts before code is changed.
Do not treat "delete the x64 setters" as a complete decision by itself.

- [ ] Option A: explicit-selector-only runtime ownership. Delete the import-time
  x64 setters in `simsopt.__init__`. `apply_jax_runtime_config()` owns dtype only
  when an explicit backend selector is present or when callers invoke
  `set_backend(..., configure_runtime=True)`.
- [ ] Option B1: always configure a JAX default policy on `import simsopt`.
  Delete the x64 setters, but when no explicit selector is present and JAX is
  available, resolve the implicit JAX-loaded default to a JAX mode such as
  `jax_cpu_parity` or a new `jax_cpu_default`. This makes
  `apply_jax_runtime_config()` run and preserves historical implicit x64
  behavior through the runtime-policy path.
- [ ] Option B2: keep a narrow import-owned fallback for the
  JAX-loaded-but-unconfigured state. Delete broad eager setters, but retain a
  deliberately scoped import contract that sets x64 only when JAX is already
  loaded and no backend selector has been provided. In this option, runtime
  policy owns mode-driven dtype, while `simsopt.__init__` owns only the legacy
  implicit JAX-loaded default.

Recommendation: choose Option B1 if the project still expects plain
`import simsopt` to preserve historical float64 behavior and wants runtime
policy to be the single owner. Choose Option A only if all supported launchers
are expected to configure the backend explicitly before JAX arrays are created.
Choose Option B2 only if the project explicitly accepts a small import-hook
compatibility contract for the legacy default state.

Do not implement "delete the setters" without making this contract explicit;
that would stop implicit imports from configuring JAX x64 at all.

## Immediate Isolated PR Scope

This is the smallest PR that makes the Metal float32 lane reachable without
taking on the full runtime-policy cleanup.

### 1. Import-Time x64 Ownership

File: `src/simsopt/__init__.py`

- [ ] Remove the direct `jax.config.update("jax_enable_x64", True)` branch.
- [ ] Remove the direct `JAX_ENABLE_X64=True` environment default.
- [ ] Preserve CUDA determinism validation.
- [ ] Preserve eager runtime config for explicit selectors.
- [ ] Implement the chosen contract from "Open Contract Decision".
- [ ] Update the header comment so it no longer claims this file owns global x64.

Acceptance:

- [ ] No direct `jax_enable_x64=True` update remains in `simsopt.__init__`.
- [ ] No `JAX_ENABLE_X64=True` default remains in `simsopt.__init__`.
- [ ] Existing import smoke expectations are updated in
  `tests/subprocess/import_smoke_cases.py` to the chosen contract.
- [ ] Import smoke cases that currently assert native/default x64 behavior are
  either preserved by Option B1/B2 or deliberately changed under Option A.

### 2. BackendPolicy dtype fields

File: `src/simsopt/backend/runtime.py`

- [ ] Add `runtime_dtype: str` to `BackendPolicy`.
- [ ] Add `host_dtype: str` to `BackendPolicy`.
- [ ] Populate `_MODE_POLICY_DEFAULTS`:
  - [ ] `native_cpu`: `runtime_dtype="float64"`, `host_dtype="float64"`.
  - [ ] `jax_cpu_fast`: `runtime_dtype="float64"`, `host_dtype="float64"`.
  - [ ] `jax_cpu_parity`: `runtime_dtype="float64"`, `host_dtype="float64"`.
  - [ ] `jax_gpu_fast`: `runtime_dtype="float64"`, `host_dtype="float64"`.
  - [ ] `jax_gpu_parity`: `runtime_dtype="float64"`, `host_dtype="float64"`.
  - [ ] `jax_metal_smoke`: `runtime_dtype="float32"`, `host_dtype="float32"`.
- [ ] Thread the two fields through `_policy_from_config(...)`.
- [ ] Add validation for allowed dtype strings if there is an existing policy
  validation point.
- [ ] Do not import `jax` or `jax.numpy` at module scope.
- [ ] Decide whether `host_dtype` should be `float32` for Metal immediately or
  kept `float64` for host-side oracle construction. If kept `float64`, document
  why. The default recommendation for the isolated reachability PR is
  `float32` for both runtime and host dtype in `jax_metal_smoke`.

Acceptance:

- [ ] `get_backend_policy("jax_metal_smoke").runtime_dtype == "float32"`.
- [ ] `get_backend_policy("jax_metal_smoke").host_dtype == "float32"`.
- [ ] All non-Metal modes report `float64` for both fields.
- [ ] `rg -n "^import jax|^from jax" src/simsopt/backend/runtime.py` stays empty.

### 3. Runtime dtype helpers

File: `src/simsopt/jax_core/_math_utils.py`

- [ ] Replace `_FLOAT64_DTYPE` with a dtype-name mapping.
- [ ] Add `_DTYPE_BY_NAME = {"float64": jnp.float64, "float32": jnp.float32}`.
- [ ] Add `_HOST_DTYPE_BY_NAME = {"float64": np.dtype(np.float64),
  "float32": np.dtype(np.float32)}` if host dtype resolution is needed here.
- [ ] Add `runtime_jnp_dtype()`:
  - [ ] import `get_backend_policy` inside the function;
  - [ ] return the `jnp` dtype selected by `get_backend_policy().runtime_dtype`;
  - [ ] fail with a clear error for an unsupported dtype string.
- [ ] Add `runtime_np_dtype()` or `runtime_host_dtype()` if host conversion needs
  an explicit NumPy dtype resolver.
- [ ] Rename `require_float64_dtype(...)` to
  `require_runtime_dtype(name, value, *, dtype=None)`.
- [ ] Make `dtype=None` default to `runtime_jnp_dtype()` or its equivalent
  normalized NumPy dtype.
- [ ] Keep explicit dtype override support so parity-only code can still require
  `float64` intentionally.
- [ ] Introduce `as_runtime_value(value, *, reference, dtype=None)` or equivalent
  policy-dtype conversion helper.
- [ ] Decide whether `as_runtime_float64(...)` remains:
  - [ ] if retained, make it an explicit-fp64 alias only for lanes that truly
    require hard fp64;
  - [ ] otherwise, update call sites to the policy-aware helper.

Acceptance:

- [ ] `require_runtime_dtype("x", jnp.asarray(..., dtype=jnp.float32))` passes
  under a `jax_metal_smoke` policy.
- [ ] The same check fails under a float64 policy unless `dtype` is overridden.
- [ ] Existing float64 CPU/GPU parity callers still get float64 arrays.
- [ ] There is no hard-coded `_FLOAT64_DTYPE` gate in the policy-aware helper.

### 4. Boozer residual call sites

File: `src/simsopt/geo/boozer_residual_jax.py`

- [ ] Update the import from `require_float64_dtype` to
  `require_runtime_dtype`.
- [ ] Rename `_require_boozer_float64_inputs(...)` to
  `_require_boozer_runtime_inputs(...)` if the function becomes policy-aware.
- [ ] Update the three direct guard calls for `B`, `xphi`, and `xtheta`.
- [ ] Decide whether each direct Boozer guard wants:
  - [ ] policy dtype for runtime reachability; or
  - [ ] explicit `dtype="float64"` for CPU-ordered parity-only paths.
- [ ] Audit every `_as_runtime_float64(...)` call in this file, not only the
  three direct dtype guards. The current file uses `_as_runtime_float64(...)`
  for `G`, `iota`, `num_res`, scalar literals, zero surface dofs, and derivative
  factors. Under the current helper implementation these calls also hard-gate
  on `require_float64_dtype("reference", reference)` and
  `require_float64_dtype("value", value)`.
- [ ] Pick one file-level strategy:
  - [ ] migrate Boozer runtime conversions to
    `as_runtime_value(value, *, reference, dtype=None)` so the general path uses
    policy dtype; or
  - [ ] redefine `as_runtime_float64(...)` as an explicit cast helper that no
    longer validates `reference` and `value` as float64, then reserve
    `require_runtime_dtype(..., dtype="float64")` for true parity-only gates.
- [ ] If using the first strategy, explicitly update the `_as_runtime_float64`
  calls in the scalar objective region around `G`, `iota`, `num_res`, and the
  `0.5` literal before claiming Boozer-path fp32 reachability.
- [ ] Keep `cpu_ordered` reduction semantics separate from general runtime dtype.

Acceptance:

- [ ] Boozer residual kernels no longer hard-gate all inputs to float64 by name.
- [ ] Boozer residual kernels no longer hard-gate fp32 inputs indirectly through
  `_as_runtime_float64(...)` on the general runtime path.
- [ ] Parity-mode tests still enforce float64 where required.
- [ ] Metal policy tests can route through the helper boundary without dtype
  rejection.

### 5. Cross-platform policy tests

File: `tests/test_runtime_dtype_policy.py`

- [ ] Add `test_jax_metal_smoke_policy_runtime_dtype`.
- [ ] Use `set_backend("jax_metal_smoke", configure_runtime=False)`.
- [ ] Assert `get_backend_policy().runtime_dtype == "float32"`.
- [ ] Assert `get_backend_policy().host_dtype == "float32"` if that is the
  selected contract.
- [ ] Restore the prior backend config and relevant env state. Do not restore to
  a hard-coded `jax_cpu_parity` baseline unless the test module is explicitly
  isolated from all other backend tests.
- [ ] Prefer the existing autouse backend-runtime guard in `tests/conftest.py`
  when the test lives under the normal pytest tree; if writing a local fixture,
  snapshot and restore the same `SIMSOPT_*`, `JAX_PLATFORMS`, XLA, CUDA, and
  allocator environment keys, then call `invalidate_backend_cache()`.
- [ ] Add non-Metal mode assertions for `float64`.
- [ ] Add `_math_utils` round-trip tests that do not require real Metal hardware.

Example shape:

```python
def test_jax_metal_smoke_policy_runtime_dtype():
    from simsopt.backend import get_backend_config, get_backend_policy, set_backend

    previous = get_backend_config()

    set_backend("jax_metal_smoke", configure_runtime=False)
    try:
        policy = get_backend_policy()
        assert policy.runtime_dtype == "float32"
        assert policy.host_dtype == "float32"
    finally:
        set_backend(
            previous.mode,
            strict=previous.strict,
            debug_nans=previous.debug_nans,
            transfer_guard=previous.transfer_guard,
            compilation_cache_dir=previous.compilation_cache_dir,
            xla_gpu_preallocate=previous.xla_gpu_preallocate,
            xla_gpu_mem_fraction=previous.xla_gpu_mem_fraction,
            xla_gpu_allocator=previous.xla_gpu_allocator,
            tf_gpu_allocator=previous.tf_gpu_allocator,
            configure_runtime=False,
        )
```

Acceptance:

- [ ] The test passes on CPU-only hosts.
- [ ] The test does not set `SIMSOPT_BACKEND_MODE=jax_metal_smoke` before
  importing `simsopt`.
- [ ] The test does not require `jax-metal`.

### 6. Optional real Metal smoke

File: `tests/test_metal_smoke_dtype.py`

- [ ] Add only if there is already an accepted marker pattern for hardware tests.
- [ ] Mark with `@pytest.mark.metal` or the repo's equivalent opt-in marker.
- [ ] Skip unless `jax-metal` / Metal backend is available.
- [ ] Run a minimal import/config/one-array round trip under
  `SIMSOPT_BACKEND_MODE=jax_metal_smoke`.
- [ ] Assert backend policy dtype and resulting array dtype are float32.

Acceptance:

- [ ] The test is not part of default CPU/Linux CI.
- [ ] The test proves actual Metal execution only on hosts that advertise Metal.

## Follow-Up Policy TODOs

These remain outside the isolated PR unless the immediate patch exposes a direct
dependency.

### 7. Default residency policy

File: `src/simsopt/backend/runtime.py`

- [ ] Add `default_residency: str` to `BackendPolicy`.
- [ ] Define allowed values, for example `"device"` and `"host"`.
- [ ] Set the default per mode.
- [ ] Use the field only at runtime boundary helpers, not ad hoc in kernels.

### 8. Default optimizer backend policy

Files:

- `src/simsopt/backend/runtime.py`
- `src/simsopt/geo/optimizer_jax.py`
- `src/simsopt/geo/boozersurface_jax.py`
- Stage 2 and single-stage entrypoints that currently infer optimizer backend.

TODOs:

- [ ] Add `default_optimizer_backend: str` to `BackendPolicy`.
- [ ] Decide per-mode default:
  - [ ] CPU/reference modes: `"scipy"`.
  - [ ] JAX target modes: `"ondevice"` unless a mode-specific exception exists.
- [ ] Add `"auto"` to the public optimizer-backend validation layer.
- [ ] Resolve `"auto"` through `get_backend_policy().default_optimizer_backend`.
- [ ] Preserve explicit caller overrides.
- [ ] Keep SciPy/reference lanes separate from target lanes.

### 9. Debug overlay and disable_jit

Files:

- `src/simsopt/backend/runtime.py`
- `src/simsopt/backend/__init__.py` if a new public helper is exported.

TODOs:

- [ ] Add `disable_jit: bool` to `BackendPolicy` or `BackendConfig`.
- [ ] Add `use_runtime(mode=..., debug=...)` if this becomes public API.
- [ ] Define `SIMSOPT_DEBUG=1` overlay semantics:
  - [ ] `debug_nans=True`;
  - [ ] `transfer_guard="disallow"`;
  - [ ] `disable_jit=True`;
  - [ ] `strict=True`.
- [ ] Apply the overlay through the same runtime config path.
- [ ] Do not add try/except recovery around failed runtime configuration.

### 10. Central dtype module

File: `src/simsopt/backend/dtypes.py`

TODOs:

- [ ] Create only after the immediate `_math_utils.py` policy helper is stable.
- [ ] Expose `runtime_dtype()`.
- [ ] Expose `host_dtype()`.
- [ ] Expose `as_runtime_array(...)`.
- [ ] Expose `runtime_zeros(...)`.
- [ ] Expose `runtime_eye(...)`.
- [ ] Keep the implementation as the SSOT for dtype helper functions.
- [ ] Re-export from existing helper modules instead of duplicating conversion
  logic.

### 11. Collapse remaining local `_as_jax_float64` duplicates

Current duplicate definitions to collapse:

- [ ] `src/simsopt/geo/curve.py`
- [ ] `src/simsopt/geo/_pairwise_reductions.py`
- [ ] `src/simsopt/geo/framedcurve.py`
- [ ] `src/simsopt/geo/curveobjectives.py`
- [ ] `src/simsopt/geo/curvecwsfourier.py`
- [ ] `src/simsopt/jax_core/surface_rzfourier.py`
- [ ] `src/simsopt/objectives/stage2_target_objective_jax.py`

TODOs:

- [ ] Replace local definitions with imports/re-exports from the central helper.
- [ ] Preserve explicit fp64 helpers only where a lane really requires fp64.
- [ ] Add focused tests before broad replacement if a file feeds parity-critical
  math.

### 12. Repo-wide `as_runtime_float64` migration

Files:

- `src/simsopt/jax_core/*`
- `src/simsopt/geo/*`
- `src/simsopt/objectives/*`

TODOs:

- [ ] Inventory all `as_runtime_float64` and `_as_runtime_float64` call sites.
- [ ] Split call sites into true explicit-fp64 parity contracts and general
  runtime-dtype conversions.
- [ ] Move general runtime conversions to the policy-aware helper.
- [ ] Preserve explicit fp64 gates with
  `require_runtime_dtype(..., dtype="float64")` or an explicit-fp64 conversion
  helper.
- [ ] Keep this migration out of the isolated PR unless a call site is required
  for the Boozer-path reachability target.

### 13. Runtime device placement helper

Files:

- `src/simsopt/backend/dtypes.py` or `src/simsopt/backend/runtime.py`
- hot path modules after policy helpers are stable.

TODOs:

- [ ] Add `runtime_device_put(...)` with explicit policy-owned dtype and
  residency semantics.
- [ ] Use it only in hot paths that currently duplicate device placement.
- [ ] Avoid broad mechanical churn until dtype policy tests are stable.

## Detailed Implementation Order

### Phase 0: Pin contract and tests

- [ ] Decide Option A, Option B1, or Option B2 for import-time config ownership.
- [ ] Add policy unit tests for dtype mapping.
- [ ] Add helper unit tests that show current hard float64 behavior fails for
  Metal-policy float32 inputs.
- [ ] Run the new tests and confirm they fail for the expected reason.

### Phase 1: Policy fields

- [ ] Add `runtime_dtype` and `host_dtype` defaults.
- [ ] Thread fields through `BackendPolicy`.
- [ ] Add any necessary policy-string validation.
- [ ] Re-run policy tests.

### Phase 2: Helper conversion

- [ ] Add dtype-name resolvers in `_math_utils.py`.
- [ ] Implement `require_runtime_dtype(...)`.
- [ ] Implement policy-aware runtime conversion.
- [ ] Keep or adapt `as_runtime_float64(...)` according to explicit-fp64 needs.
- [ ] Re-run helper tests.

### Phase 3: Boozer guard update

- [ ] Update imports and call sites.
- [ ] Update or intentionally preserve every `_as_runtime_float64(...)` use that
  can block the Boozer-path fp32 reachability target.
- [ ] Preserve explicit fp64 for CPU-ordered parity paths only if needed.
- [ ] Run focused Boozer residual tests.

### Phase 4: Import x64 ownership

- [ ] Delete direct x64 setters in `simsopt.__init__`.
- [ ] Implement the chosen import-time ownership contract.
- [ ] Update `tests/subprocess/import_smoke_cases.py`.
- [ ] Re-run subprocess import smoke tests that mention x64 policy.

### Phase 5: Validation

- [ ] Run `tests/test_backend.py` focused backend-policy tests.
- [ ] Run the new `tests/test_runtime_dtype_policy.py`.
- [ ] Run focused Boozer residual tests.
- [ ] Run import smoke tests that cover `jax_enable_x64` behavior.
- [ ] If Metal hardware exists, run the opt-in Metal smoke.

## Suggested Commands

Use the repo-local environment when available:

```bash
./.venv-local/bin/python -m pytest tests/test_backend.py -q
./.venv-local/bin/python -m pytest tests/test_runtime_dtype_policy.py -q
./.venv-local/bin/python -m pytest tests/geo/test_boozersurface_jax.py -q
./.venv-local/bin/python -m pytest tests/subprocess/import_smoke_cases.py -q
```

If using plain `python`, force this checkout first to avoid importing the
sibling `simsopt-surrogate` checkout:

```bash
PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src python -m pytest tests/test_runtime_dtype_policy.py -q
```

## Acceptance Checklist

- [ ] `rg -n "jax_enable_x64.*True|JAX_ENABLE_X64.*True" src/simsopt/__init__.py`
  finds no direct setter.
- [ ] `get_backend_policy("jax_metal_smoke").runtime_dtype == "float32"`.
- [ ] `get_backend_policy("jax_metal_smoke").host_dtype == "float32"` or a
  documented alternative.
- [ ] Non-Metal modes keep `float64` policy dtype.
- [ ] `_math_utils.require_runtime_dtype(...)` accepts `float32` arrays under
  Metal policy.
- [ ] `_math_utils.require_runtime_dtype(...)` rejects `float32` arrays under
  float64 policy.
- [ ] `boozer_residual_jax.py` no longer imports `require_float64_dtype`.
- [ ] Boozer residual runtime conversions no longer block fp32 through
  `_as_runtime_float64(...)` on the general runtime path.
- [ ] Cross-platform dtype policy tests do not require `jax-metal`.
- [ ] No module-scope JAX import is added to `src/simsopt/backend/runtime.py`.
- [ ] Existing CPU/GPU parity tests remain strict about float64 where required.

## PR Description Boundary

Use this boundary in the isolated PR description:

```text
This PR makes the float32 Metal policy reachable through the Boozer residual
path and establishes the runtime dtype policy surface. Other JAX kernels still
contain explicit or helper-mediated float64 conversions and will be migrated in
follow-up PRs. The cross-platform test proves policy reachability, not real
Metal correctness.
```

Do not describe the isolated PR as repo-wide fp32 support or as Metal hardware
signoff.

## Risks And Mitigations

- Risk: deleting import-time x64 setters silently changes implicit
  `import simsopt` behavior.
  - Mitigation: choose Option A or B explicitly and update import smoke tests.
- Risk: `as_runtime_float64(...)` is used broadly and may encode true parity
  assumptions at some call sites.
  - Mitigation: keep explicit dtype override support and migrate call sites in
    focused slices.
- Risk: reviewers assume this PR makes every JAX kernel fp32-capable.
  - Mitigation: state the Boozer-path-only reachability boundary in the PR
    description and keep repo-wide `as_runtime_float64` migration as a follow-up
    TODO.
- Risk: Metal policy tests accidentally initialize the real Metal backend.
  - Mitigation: use `set_backend(..., configure_runtime=False)` for
    cross-platform tests.
- Risk: dtype strings drift from helper mappings.
  - Mitigation: centralize allowed dtype names and add policy validation.
- Risk: float32 reachability is mistaken for Metal correctness.
  - Mitigation: keep a separate hardware-marked Metal smoke and do not call the
    isolated PR a real Metal signoff.

## Done Definition

The isolated PR is done when:

- [ ] runtime dtype policy is represented in `BackendPolicy`;
- [ ] `_math_utils.py` uses that policy for runtime dtype validation and
  conversion;
- [ ] Boozer residual guards no longer hard-code float64 on the general runtime
  path;
- [ ] Boozer residual helper conversions no longer indirectly hard-code float64
  on the general runtime path;
- [ ] `simsopt.__init__` no longer owns direct x64 mutation;
- [ ] the import-time runtime ownership contract is documented and tested;
- [ ] cross-platform tests prove Metal policy selects float32 without requiring
  Metal hardware;
- [ ] no module-scope JAX import is introduced in `runtime.py`;
- [ ] focused tests pass.
