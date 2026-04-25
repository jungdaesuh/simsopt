# Plan-claim verifications (2026-04-25)

Branch: `gpu-purity-stage2-20260405` · cwd: `/Users/suhjungdae/code/columbia/simsopt-jax`.
Conda env (true 0.9.2 lane): `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python`
with `PYTHONNOUSERSITE=1` (without that flag, the env's `python` resolves
`/Users/suhjungdae/.local/lib/python3.11/site-packages/jax` at version 0.10.0
because user-site is on `sys.path`).

---

## Claim 1 — JAX 0.9.2 buffer-donation API (`is_deleted()` exists, deleted-array reads raise)

**Verdict: CONFIRMED.**

Probe (clean 0.9.2):

```
PYTHONNOUSERSITE=1 .conda/jax-0.9.2/bin/python -c "
import jax, jaxlib, jax.numpy as jnp
print('jax:', jax.__version__)             # 0.9.2
print('jaxlib:', jaxlib.__version__)       # 0.9.2
arr = jnp.arange(5.0)
print('has is_deleted:', hasattr(arr, 'is_deleted'))   # True
print('is_deleted before:', arr.is_deleted())          # False
g = jax.jit(lambda x: x*2.0, donate_argnums=(0,))
out = g(arr); out.block_until_ready()
print('is_deleted after donate:', arr.is_deleted())    # True
try:
    jnp.asarray(arr)
except Exception as e:
    print('asarray raised:', type(e).__name__, str(e)[:200])
    # RuntimeError Array has been deleted with shape=float32[5].
"
```

Findings:
- `jax.Array.is_deleted()` is a real bound method on jax 0.9.2. After
  `donate_argnums=(0,)` and a forced read of the output, the donated input
  reports `is_deleted() is True`.
- `jnp.asarray(deleted)` raises `RuntimeError("Array has been deleted with
  shape=float32[5].")` — not garbage, not silent. Same RuntimeError on
  `float(deleted.sum())`. So a donation-invariant test can use either of:

  ```python
  with pytest.raises(RuntimeError, match="has been deleted"):
      jnp.asarray(donated_input)
  # or
  assert donated_input.is_deleted()
  ```

  Both are stable and don't require driver-level inspection.

**Implication for plan**: P0 fix #4 (`tests/test_biotsavart_donation_probe.py:75-95`)
should retain a single donated `points` array across the call, then assert
`points.is_deleted() is True` AND `pytest.raises(RuntimeError, match="has
been deleted")` against `jnp.asarray(points)`. Both probes work on the lane
the test actually targets (jax-0.9.2). No version-guard needed.

---

## Claim 2 — Stage 2 platform guard parity with single-stage

**Verdict: CONFIRMED.** Stage 2 is missing the guard; single-stage has it.

`benchmarks/stage2_e2e_comparison.py:20-59` imports
`apply_requested_platform`, `require_x64_runtime`, etc. from
`benchmarks.validation_ladder_common` but *does not* import
`require_requested_platform_runtime`. The runtime preamble ends at line 59:

```python
46:  REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
47:  apply_requested_platform(REQUESTED_PLATFORM)
48:  apply_benchmark_compilation_cache_policy(
49:      "stage2_e2e_comparison",
50:      requested_platform=REQUESTED_PLATFORM,
51:  )
52:  bootstrap_local_simsopt()
53:
54:  import jax
55:  import jaxlib
56:
57:  maybe_initialize_distributed_runtime()
58:  jax.config.update("jax_enable_x64", True)
59:  require_x64_runtime(jax, context="Stage 2 end-to-end comparison")
```

`grep -n require_requested_platform_runtime benchmarks/stage2_e2e_comparison.py`
returns *no* matches.

`benchmarks/single_stage_init_parity.py` *does* call it, exactly where the
audit pointed (line 73-77):

```python
73:  require_requested_platform_runtime(
74:      jax,
75:      requested_platform=REQUESTED_PLATFORM,
76:      context=_RUNTIME_CONTEXT,
77:  )
```

The function lives at `benchmarks/validation_ladder_common.py:288-305`. It
*does* fail loud:

```python
def require_requested_platform_runtime(jax_module, *, requested_platform, context):
    if requested_platform == "auto":
        return
    actual_backend = str(jax_module.default_backend()).lower()
    expected_backends = _REQUESTED_PLATFORM_RUNTIME_BACKENDS[requested_platform]
    if actual_backend in expected_backends:
        return
    devices = [str(device) for device in jax_module.devices()]
    raise RuntimeError(
        f"{context} requested JAX platform '{requested_platform}' but initialized "
        f"backend '{actual_backend}' on devices {devices}."
    )
```

So a CUDA-less host running `--platform cuda` against `single_stage_init_parity`
hard-fails, but the same condition against `stage2_e2e_comparison` silently
falls back to CPU (because `apply_requested_platform` only sets envs and does
not assert the result).

Caveat: when the caller passes `--platform auto`, the guard is a no-op by
design — that branch is not a parity gap, just intentional behaviour.

**Implication for plan**: P0 fix #1 / #14 are correct as written. Insert
`require_requested_platform_runtime(jax, requested_platform=REQUESTED_PLATFORM,
context="Stage 2 end-to-end comparison")` right after the existing
`require_x64_runtime(...)` call at line 59. Update the
`benchmarks.validation_ladder_common` import at line 33 to include the symbol.

---

## Claim 3 — GPU-proof aggregator drops provenance fields

**Verdict: CONFIRMED.** The aggregator at
`benchmarks/hf_jobs/run_production_gpu_proof.sh:249-287` reduces every probe
payload to exactly five fields. The richer schema is produced upstream by
`build_provenance` and *is* present in each per-probe payload, then thrown
away by the aggregator.

Aggregator (verbatim, lines 249-287):

```bash
python - "${RESULTS_DIR}" "${EXPECTED_PROBES[@]}" <<'PY'
import json
import sys
from pathlib import Path

results_dir = Path(sys.argv[1])
expected = sys.argv[2:]
summary = {}
for probe_name in expected:
    path = results_dir / f"{probe_name}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            summary[path.name] = {
                "passed": False,
                "elapsed_s": None,
                "failures": [f"corrupt payload: {exc}"],
                "missing_payload": False,
                "corrupt_payload": True,
            }
            continue
        summary[path.name] = {
            "passed": payload.get("passed"),
            "elapsed_s": payload.get("elapsed_s"),
            "failures": payload.get("failures"),
            "missing_payload": False,
            "corrupt_payload": False,
        }
        continue
    summary[path.name] = {
        "passed": False,
        "elapsed_s": None,
        "failures": ["missing payload"],
        "missing_payload": True,
        "corrupt_payload": False,
    }
print(json.dumps(summary, indent=2))
PY
```

What probes actually emit (per
`benchmarks.validation_ladder_common.build_provenance` at lines 468-492):

```python
provenance = {
    "title": title,
    "repo_sha": get_git_sha(),
    "jax": jax_module.__version__,
    "jaxlib": jaxlib_module.__version__,
    "backend": jax_module.default_backend(),       # "gpu" / "cpu"
    "devices": [str(device) for device in jax_module.devices()],
    "x64_enabled": _x64_enabled(jax_module),
    "peak_rss_mb": peak_rss_mb(),
    **backend_guardrails,                          # backend_mode, backend_strict, transfer_guard
    **compilation_cache,                           # compilation_cache_enabled, _dir, _policy
    **_current_sharding_metadata(),
    # plus optional gpu_memory_mb when nvidia-smi is available
    # plus per-probe extras: lane, fixture, platform_request, optimizer_backend, ...
}
```

Single-stage payload (`benchmarks/single_stage_init_parity.py:817-848,
930-940`) wraps that as `payload = {"provenance": provenance, "cpu_results":
..., "jax_results": ..., "comparison": ..., "timings": ...}`. Stage 2
(`benchmarks/stage2_e2e_comparison.py:919-948`) does the same plus
`provenance["cpu_endpoint_lane"] = {...}`. Both emit `passed`, `elapsed_s`,
`failures` at the top level — those are the only three the aggregator keeps.

Concretely lost when running on a CUDA-less host:
- `provenance.backend == "cpu"` (the smoking gun for the 2026-04-20 Runpod
  cubin incident)
- `provenance.devices`
- `provenance.gpu_memory_mb`
- `comparison.field_error_rel_diff`, `final_iota_abs_diff`,
  `final_volume_rel_diff`, `max_surface_pointwise_rel`
- `cpu_results` / `jax_results` value/gradient floors
- `provenance.transfer_guard`, `backend_strict`, `backend_mode`

There is no `xla_flags` / `jaxlib.cuda_versions` field in `build_provenance`
today; those are aspirational additions in the audit's P0 fix #1, not present
fields the aggregator drops.

**Implication for plan**: P0 fix #1 is correct. The aggregator must (a)
preserve `payload["provenance"]` verbatim (or at minimum
`{backend, devices, gpu_memory_mb, transfer_guard, backend_mode, backend_strict}`),
(b) preserve `payload["comparison"]` rtol/abs diffs so a parity drift on a
GPU-equipped host is visible, and (c) extend `build_provenance` itself to
emit `xla_flags = sorted(k for k in os.environ if k.startswith("XLA_") or k ==
"JAX_PLATFORMS")` and `jaxlib_cuda_versions = jaxlib.cuda_versions if
hasattr(jaxlib, "cuda_versions") else None` before the aggregator can keep
them.

---

## Claim 4 — FD escape-hatch line numbers in `test_single_stage_jax_cpu_reference.py`

**Verdict: CONFIRMED for line 4213; PARTIALLY-CONFIRMED for line 5792 (the
`or abs < 1e-8` condition is at 5792 but the *majority gate* assertion is at
5804); CONFIRMED for line 1979 (this is the `_REAL_RESOLVE_FD_MIN_STABLE_EPS`
gate, not the majority gate over directions).**

Constants at lines 1438-1444:

```
1438:  _REAL_RESOLVE_FD_ABS_TOL = 1e-8
1439:  _REAL_RESOLVE_FD_TAYLOR_RATE = 0.55
1440:  _REAL_RESOLVE_FD_EPSILONS = (4.0e-4, 2.0e-4, 1.0e-4)
1441:  _REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3
1442:  _REAL_RESOLVE_FD_MIN_STABLE_EPS = 2
1443:  _REAL_RESOLVE_FD_AXIS_DIRECTION_FRACTIONS = (0.0, 0.5, 1.0)  # 3 directions
```

So the resolve-FD scheme is: 3 directions × 3 epsilons; require ≥2 epsilons
stable per direction (line 1979 below) AND ≥3 of 3 directions stable (line
1990). That is *not* a 2-of-3 majority over directions — it is a hard
all-of-3 gate over directions, but combined with a 2-of-3 *epsilon-ladder*
majority per direction.

Line 1965-1985 (the per-eps Taylor check, where the `0.55` rate applies):

```
1965:  if err_old is not None:
1966:      threshold = max(
1967:          _REAL_RESOLVE_FD_ABS_TOL,
1968:          _REAL_RESOLVE_FD_TAYLOR_RATE * err_old,    # 0.55 * err_old
1969:      )
1970:      if abs_err >= threshold:
1971:          mismatch_reasons.append(...)
1972:          direction_ok = False
1973:          break
1979:  if direction_ok and stable_eps_count >= _REAL_RESOLVE_FD_MIN_STABLE_EPS:
1980:      stable_samples += 1
1990:  if stable_samples < _REAL_RESOLVE_FD_MIN_STABLE_SAMPLES:
        ...
2004:      pytest.fail(...)
```

So a direction is "stable" once 2 of 3 epsilons survive the `0.55*prev`
Taylor gate; the test only fails if fewer than 3 of 3 directions are stable.
A wrong-sign IFT term that produces noise on 1 of 3 epsilons in 1 of 3
directions still passes.

Line 4213 (`TestEndToEndCompositeFD` style, fixed-surface FD on
`BoozerResidualJAX`) — verbatim:

```
4198:  for i in range(3):
4199:      d = rng.randn(len(x0))
4200:      d /= np.linalg.norm(d)
4202:      dd_composed = float(np.dot(g_composed, d))
4203:      dd_fd = (J_at_fixed_surface(x0 + eps * d) - J_at_fixed_surface(x0 - eps * d)) / (2 * eps)
4207:      abs_err = abs(dd_composed - dd_fd)
4208:      rel_err = abs_err / (abs(dd_fd) + 1e-30)
4213:      assert rel_err < 1e-3 or abs_err < 1e-8, ...
```

Line 5792 (IotasJAX controlled LS resolve-FD; same pattern but with
explicit `validated_directions >= 2` majority gate at 5804):

```
5775:  for eps in eps_candidates:
5776:      plus = iota_at(x0 + eps * direction)
5777:      minus = iota_at(x0 - eps * direction)
5784:      directional_fd = float((plus - minus) / (2.0 * eps))
5785:      abs_err = abs(directional_adjoint - directional_fd)
5786:      rel_err = abs_err / (abs(directional_fd) + 1e-30)
5792:      if rel_err < 1e-3 or abs_err < 1e-8:
5793:          validated_directions += 1
5794:          direction_validated = True
5795:          break
5804:  assert validated_directions >= 2, (
5805:      "IotasJAX controlled LS re-solve FD found too few local directions: "
5806:      f"{validated_directions}/3 validated. " + ...
5807:  )
```

Three out of three claims confirmed at the cited lines; one cited line
(1979) is the *eps-ladder majority gate* not the *direction majority gate*.

Other identical patterns in the same file (`grep -n "or abs_err < 1e-8\|or
abs < 1e-8\|stable_samples\|>= 2\|_REAL_RESOLVE_FD_TAYLOR_RATE"`):

```
510:   assert rel_err < rel_tol or abs_err < abs_tol, ...   # parameterized helper
1438:  _REAL_RESOLVE_FD_ABS_TOL = 1e-8
1439:  _REAL_RESOLVE_FD_TAYLOR_RATE = 0.55
1441:  _REAL_RESOLVE_FD_MIN_STABLE_SAMPLES = 3
1919:  stable_samples = 0
1968:  _REAL_RESOLVE_FD_TAYLOR_RATE * err_old,
1980:  stable_samples += 1
1990:  if stable_samples < _REAL_RESOLVE_FD_MIN_STABLE_SAMPLES:
1998:  f"stable directions={stable_samples}/{num_directions} "
3744:  assert rel_err < 1e-3 or abs_err < 1e-8, ...        # **fourth occurrence** (audit missed)
4213:  assert rel_err < 1e-3 or abs_err < 1e-8, ...
5153:  assert rel_err < 1e-3 or abs_err < 1e-8, ...        # **fifth occurrence** (audit missed)
5792:  if rel_err < 1e-3 or abs_err < 1e-8:
5804:  assert validated_directions >= 2, ...
```

So the `rel < 1e-3 OR abs < 1e-8` escape pattern occurs at lines **510, 3744,
4213, 5153, 5792** — five sites, not three. The audit's three named lines are
correct but the fix list should include 3744 and 5153 too.

**Implication for plan**: P0 fix #3 is structurally correct. Expand the line
list from {1979, 4213, 5792} to {3744, 4213, 5153, 5792} for the OR-escape
pattern, plus the {1979, 1990} pair for the eps-ladder + direction-stability
gate. The `0.55` Taylor rate constant lives at line 1439 (single source); the
audit's recommendation to drop it to 0.4 is a one-line constant edit. Line 510
is a tunable helper-level acceptance, not a hardcoded constant — leave it for
P1 unless callers turn out to feed below-1e-8 oracles.

---

## Claim 5 — `test_force_objectives_taylor_test` collapse + 3 deleted curve tests + lost seed

**Verdict: CONFIRMED on all three sub-claims, with one naming caveat.**

### 5a. Force-objective Taylor sweep collapse

The actual function name is `test_Taylor` (not `test_force_objectives_taylor_test`),
inside `class TestForceObjectives` (or equivalent). The audit-named string
does not exist in either upstream or HEAD. Both
`tests/field/test_selffieldforces.py` versions only have `test_Taylor` for
this purpose.

Upstream (`upstream_hss/master:tests/field/test_selffieldforces.py:1017`)
sweeps:

```python
ncoils_list = [2]                                      # 1
nfp_list = [1, 3]                                      # 2
stellsym_list = [True]                                 # 1
p_list = [2.5]                                         # 1
threshold_list = [0.0, 1e-3]                           # 2
regularization_types = [("circular", ...), ("rectangular", ...)]   # 2
downsample_list = [1, 2]                               # 2
jax_flag_list = [False, True]                          # 2
numquadpoints_list = [10]                              # 1
# config product: 1*2*1*1*2*2*2*2*1 = 32 configs
# 10 objectives per config: NetFluxes (sum), B2Energy, LpCurveTorque,
#   sum(LpCurveTorque), sum(SquaredMeanTorque), SquaredMeanTorque,
#   sum(LpCurveForce), LpCurveForce, sum(SquaredMeanForce), SquaredMeanForce
# total sub-cases: 32 * 10 = 320  + a 3x retry loop on failure
```

HEAD (`tests/field/test_selffieldforces.py:1720-1862`) replaced the cartesian
product with two hand-picked cfg dicts and a 6-objective list:

```python
test_configs = [
    {"ncoils":2,"nfp":1,"stellsym":True,"p":2.5,"threshold":0.0,
     "regularization":regularization_circ(a),"downsample":1,
     "use_jax_curve":False,"numquadpoints":10,"seed":7},
    {"ncoils":2,"nfp":3,"stellsym":True,"p":2.5,"threshold":1e-3,
     "regularization":regularization_rect(a,b),"downsample":2,
     "use_jax_curve":True,"numquadpoints":10,"seed":17},
]
# 2 configs x 6 objectives = 12 sub-cases (was 320)
```

A docstring at line 1722-1727 explicitly acknowledges the cut: "The
historical version of this test swept a large Cartesian product of
configurations [...] This version keeps the same objective families under
both NumPy and JAX curve paths, but limits coverage to representative cases
suitable for default CI."

The collapse is `1 - 12/320 = 96.25%` reduction. The same `0.5 * prev_error`
Taylor rate is preserved (HEAD line 1778, upstream line 1071). The retry
loop (`max_retries = 3`) was also dropped — failures now fast-fail on the
first attempt.

Coverage axes specifically lost (the JAX-port-rewritten paths the audit
flags):
- `nfp ∈ {1, 3}` collapsed to one of each across the two configs (1 in cfg0,
  3 in cfg1) — preserved.
- `use_jax_curve ∈ {False, True}` likewise split across configs (False in
  cfg0, True in cfg1) — preserved as 1×1 instead of 2×2.
- `downsample ∈ {1, 2}` likewise split — preserved as 1×1 instead of 2×2.
- `threshold ∈ {0.0, 1e-3}` likewise split — preserved as 1×1 instead of 2×2.
- `regularization_types` (circular vs rectangular) — preserved as 1×1.
- The `sum([...])` per-coil objective variants (5 of the 10 upstream
  objectives) are entirely deleted in HEAD; HEAD only keeps the collective
  `LpCurveTorque`/`SquaredMeanTorque`/`LpCurveForce`/`SquaredMeanForce` plus
  `B2Energy` and `NetFluxes`. So the per-coil-summation paths
  (`sum(LpCurveTorque(...) for i)`) — which exercise reduce-add Optimizable
  composition — are uncovered.

### 5b. Three deleted upstream tests in `test_curve_objectives.py`

`grep -nE "^def test_|^    def test_" upstream_hss/master:tests/geo/test_curve_objectives.py`
yields these among others:

```
241:    def test_arclength_variation_circle_planar(self):
406:    def test_linking_number_planar(self):
440:    def test_curve_curve_distance_empty_candidates(self):
```

`grep -n "test_arclength_variation_circle_planar\|test_linking_number_planar
\|test_curve_curve_distance_empty_candidates"
tests/geo/test_curve_objectives.py` returns **zero matches** in HEAD —
confirming all three were deleted.

The HEAD file count is 21 test functions (vs upstream's 15 in this same
file's class — net new tests in HEAD include `*_reuses_shared_jit_kernels`
and `*_pairwise_penalty_*` accessibility / sharding tests). The three
deletions are pure regressions: `_planar` variants exercise
`CurvePlanarFourier` × the analytic invariant, and
`_empty_candidates` covers the pairwise-distance kernel boundary at
zero-candidate input.

### 5c. `test_curve_minimum_distance_taylor_test` lost seed and downsample loop

Upstream `subtest_curve_minimum_distance_taylor_test` opens with:

```python
def subtest_curve_minimum_distance_taylor_test(self, curve):
    np.random.seed(0)                                           # <-- removed in HEAD
    ncurves = 3
    ...
    for downsample in [1, 2, 3]:                                # <-- removed in HEAD
        J = CurveCurveDistance(curves, 0.4, downsample=downsample)
        ...
        for k in range(ncurves):
            ...
            for i in range(5, 12):
                eps = 0.5**i
                ...
                self.assertLess(err_new, 0.6 * err, ...)
```

HEAD `subtest_curve_minimum_distance_taylor_test` opens with:

```python
def subtest_curve_minimum_distance_taylor_test(self, curve):
    ncurves = 3                                                 # NO seed
    ...
    distance_threshold = 0.4 if curve_t == "CurveHelical" else 0.2
    J = CurveCurveDistance(curves, distance_threshold)          # NO downsample loop
    ...
    for k in range(ncurves):
        ...
        for i in range(5, 12):
            eps = 0.5**i
            ...
            assert err_new < 0.6 * err
```

So both the `np.random.seed(0)` line and the `for downsample in [1,2,3]:`
loop were removed. The 0.6 Taylor rate is preserved. The downsample
parameter on `CurveCurveDistance` is no longer exercised at all by this
test path.

The audit-cited line numbers (`tests/geo/test_curve_objectives.py:639-644,
906-928`) point to the surviving outer dispatcher + the `test_linking_number`
function — both correct.

**Implication for plan**: P0 fix #11 (restore force-objective sweep) and #12
(restore curve-objective coverage + un-delete the 3 tests + restore seed) are
correct and well-scoped. Two additions to the fix:
- For the force test, also restore the `sum([Lp/SquaredMean*(coils[i],
  coils2) for i in range(len(coils))])` per-coil-sum objective variants — HEAD
  drops 4 of the 10 objective entries even on the surviving configs.
- For `test_curve_minimum_distance_taylor_test`, the
  `distance_threshold = 0.4 if "CurveHelical" else 0.2` change in HEAD is
  incompatible with restoring `for downsample in [1,2,3]:`; the upstream code
  unconditionally used `0.4`. Pick one and stick with it (recommend keeping
  the per-curvetype `0.4 / 0.2` split since `CurveHelical` has `deriv = 0`
  for `0.2`, but verify by running the upstream `0.4` branch under HEAD's
  helical path).

---

## DONE — verifications at /Users/suhjungdae/code/columbia/simsopt-jax/.artifacts/jax-test-audit-2026-04-25/VERIFY_plan_claims.md
