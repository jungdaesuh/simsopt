# `/goal` prompt - close active-scope JAX port gaps

Date: 2026-05-12
Branch context: `gpu-purity-stage2-20260405`
Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
Original prompt base commit: `fa3f877af`
Original source audit base: `8b471e8e3` (parent of the original prompt base)
Current repo reconciliation base commit:
`e0e6f21d71d0234ad4cdefeed63329c5648cbfb0`

## Review status

This revision fixes issues found in the original prompt after checking the
current tree, upstream SIMSOPT, the repo-local JAX runtime, and official JAX /
SIMSOPT / CUDA documentation:

- The default scope is now P0-P2 only. P3-P5 are future-scope inventory in this
  document and must be initialized as skipped unless the human launching the
  goal explicitly expands `active_scope`.
- Local docs are inputs, not ground truth. Current source, upstream SIMSOPT,
  official docs, and executed validation evidence win when they disagree.
- Validation uses the repo-local interpreter
  `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python`,
  which currently imports `jax==0.10.0` and `jaxlib==0.10.0` on CPU. The global
  conda env name `jax-0.9.2` is not available on this host.
- Real CUDA proof is not claimed from CPU or StableHLO proxies. Do not launch
  GPU jobs unless the user explicitly approves them.
- No new dependency (`diffrax`, `lineax`, `optimistix`, `optax`, etc.) may be
  introduced by this prompt. If an item needs a new dependency, block the item
  and propose the dependency for human review.
- `.artifacts/` is not ignored in this repo, so state updates are useful for
  resumption but must not create noisy in-progress commits.
- Context7 resolution in this review returned `/google/jax`,
  `/hiddensymmetries/simsopt`, and `/websites/nvidia_cuda`. Upstream SIMSOPT
  local HEAD and `hiddenSymmetries/simsopt` remote HEAD both resolved to
  `1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.
- Parity-coverage matrix is now a required plan deliverable. Every active
  item builds a CPU/C++ → JAX coverage matrix at
  `.artifacts/jax_port_goal/plans/<id>-coverage.md` (section 4a) and the
  completion gate in section 4c refuses to mark COMPLETE while any
  applicable oracle test row is unmapped or unclassified.
- State schema source of truth is section 1's `manifest_version: 4` and
  migration note. A v2 or v3 state.json is not silently upgraded; in-progress
  items pause to backfill the required artifacts before resuming.
- Red-step evidence: cited new parity test must be confirmed FAILING against
  the parent commit before implementation; capture saved to
  `.artifacts/jax_port_goal/red/<id>.txt`. Empty-oracle N/A only applies to
  Tier P1 new-kernel ports where no pre-impl tree exists.

## How to use

Paste the contents of the `## Prompt` section below into a `/goal`-style
autonomous orchestrator. The prompt is self-contained: it boots from current
source and official docs, maintains a resumable state file, walks a
dependency-ordered active manifest, and refuses to declare victory without
validation evidence. Edit only `active_scope` and the work manifest before
launching if you want to narrow or broaden scope.

---

## Prompt

~~~markdown
# Goal: close active-scope JAX port gaps in simsopt-jax

You will iterate until every item in `active_scope` is either COMPLETE (with
full validation evidence), BLOCKED (with a documented, non-self-resolvable
blocker), or SKIPPED (human-declared out of scope). You do not stop on the
first failure. You do not declare victory without the validation gate. You do
not invent shortcuts.

Default active scope: P0-P2. P3-P5 are future-scope inventory in this document
and must be initialized as skipped unless `active_scope` explicitly includes
them.

Default scope profile: `port_closure`. This means the run closes JAX-native
port gaps with CPU/C++ oracle parity, transfer-guard evidence, production-scale
CPU fixtures, and explicit CUDA deferral when no approved GPU run is available.
Do not report this as CUDA performance release readiness.

Optional scope profile: `cuda_perf_release`. This profile is active only when
`active_scope_profile` is explicitly set to `cuda_perf_release` and the user has
approved the required GPU runs. It adds real CUDA parity/performance artifacts,
warm/cold compile-cache timings, GPU memory high-water marks, and production
multi-device/sharding proof. CPU, HLO, and multi-device CPU evidence remain
proxies under this profile until a current-SHA CUDA artifact exists.

## Required closure checklists

Do not mark an item COMPLETE until each applicable checkbox is checked in that
item's plan. For BLOCKED or SKIPPED items, the blocker or skip note must list
the unchecked checkboxes and the evidence that prevents closure.

Host/device flow checklist:

- [ ] All host-to-device staging is explicit and outside jitted hot paths.
- [ ] No implicit host transfers occur under `SIMSOPT_JAX_TRANSFER_GUARD=disallow`.
- [ ] Boundary scalar casts (`int`, `bool`, host result dict values) happen only
  after the compiled path returns.
- [ ] Multi-device CPU proxy covers any sharding or collective path touched by
  the item.
- [ ] CPU/HLO proxy evidence is not reported as real CUDA proof.

JAX-native implementation checklist:

- [ ] The compiled path consumes immutable specs and explicit DOF arrays.
- [ ] The compiled path does not read mutable `Optimizable` wrapper state.
- [ ] Internal gradients use native cotangents; `Derivative` projection remains
  only at public compatibility boundaries.
- [ ] No silent CPU fallback, broad exception handler, dynamic import, `Any`
  cast, or new runtime dependency was added.
- [ ] Existing CPU/C++ oracle paths remain intact.

Gap and downstream checklist:

- [ ] Current source audit identifies existing JAX coverage and any remaining
  CPU or host-transfer edge.
- [ ] Upstream SIMSOPT oracle behavior is cited for the item.
- [ ] Downstream wrappers, examples, docs, tests, and benchmarks that consume
  the item are listed.
- [ ] Existing local docs that still say CPU-only / not ported are either still
  correct or updated with validation evidence.
- [ ] If the item is not fully JAX-native, the residual gap is recorded as
  BLOCKED or SKIPPED rather than reported complete.

Parity-coverage checklist (every active item, gated by section 4a + 4c):

- [ ] Coverage matrix `.artifacts/jax_port_goal/plans/<id>-coverage.md`
  exists and enumerates every repo and upstream test reachable via
  `git grep` over the item's kernel name, public class, and `simsoptpp`
  symbol. `upstream_audit_sha` is recorded.
- [ ] Every matrix row is classified `covered_by_unit_parity`,
  `covered_by_integration_parity`, `oracle_only`, `wrapper_only`,
  `not_applicable`, or `blocked`. No row is `unclassified`.
- [ ] Every JAX parity test cited in the matrix exists on disk at the
  commit's tree, is collected by pytest, is not skip/xfail, and asserts
  against tolerances imported from `PARITY_LADDER_TOLERANCES[<lane>]`.
- [ ] Empty-oracle items add ≥ 1 new parity test against a hand-derived
  reference and cite the oracle source in the test docstring.
- [ ] Red-step evidence: the new parity test was confirmed failing
  against the parent commit; capture saved to
  `.artifacts/jax_port_goal/red/<id>.txt`. The one allowed N/A is a
  Tier P1 new-kernel port with no pre-impl tree.

JAX transform and memory strategy checklist:

- [ ] The item's plan lists the compiled boundary and the static arguments /
  static spec metadata used to keep shapes stable.
- [ ] The plan names every JAX transform used by the hot path (`jit`, `vmap`,
  `scan`, `fori_loop`, `checkpoint`/`remat`, `shard_map`, `pmap`/collectives)
  or states `N/A: <reason>` for each class of transform that is not applicable.
- [ ] The plan explains why the chosen transform structure matches the SIMSOPT
  math contract and does not change the scalar objective, derivative shape, or
  solve residual being compared.
- [ ] The plan states the dense materialization budget, expected largest array
  shape/dtype, whether buffer donation is used, and where the HLO or benchmark
  evidence will be saved.
- [ ] Any touched sharding or collective path has a CPU proxy plus a required
  follow-up CUDA artifact path; CPU proxy evidence alone cannot close a CUDA
  performance claim.

Math/physics invariant checklist:

- [ ] The item records units/scales, current sign convention, orientation,
  `stellsym=True` and `stellsym=False` coverage, derivative shape, and excluded
  singular or near-coil regimes.
- [ ] For field/objective items, the plan states whether parity is a
  fixed-state scalar, fixed-state gradient/VJP, final optimizer envelope, or
  step-by-step optimizer trajectory contract. Optimizer traces are diagnostics,
  not the CPU/C++ oracle unless trajectory parity is explicitly scoped.
- [ ] For Boozer or linear-solve work, residual evidence is reported in the
  original physical basis after any preconditioning, normalization, or basis
  transform. A transformed residual alone is not sufficient.
- [ ] For new or changed parity tests, either red-step evidence or an explicit
  negative control / tolerance-ratchet check proves the test catches a wrong
  sign, wrong scale, or wrong state dependency.

Serialization and restart compatibility checklist:

- [ ] Any item touching specs, field objects, objectives, surfaces, Boozer
  state, or runtime metadata proves that public SIMSOPT-facing artifact paths
  still load consistently (`as_dict`/`from_dict`, restart JSON, runtime spec, or
  the documented consumer for that item).
- [ ] Compatibility/reporting metadata is not described as a production compute
  lane unless the compiled runtime path actually consumes it.

## 0. Boot - read these every iteration

Ground truth order:

1. Current repository source at the current HEAD.
2. Upstream SIMSOPT source at
   `/Users/suhjungdae/code/opensource/simsopt/src/simsopt/`, verified against
   the upstream `hiddenSymmetries/simsopt` HEAD when network is available.
3. `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES` for
   all parity tolerances.
4. Official documentation checked in the same run.
5. Local planning docs as hypotheses and historical context only.

Read these local files before selecting work:

- `/Users/suhjungdae/code/columbia/simsopt-jax/CLAUDE.md` - repo conventions,
  module layout, parity-ladder pointer, and known runtime commands. Verify
  command freshness before executing them. Known stale form: CLAUDE.md still
  prescribes `conda activate jax-0.9.2` / `conda run -n jax-0.9.2 python …`,
  but the global conda env of that name is missing
  (`EnvironmentLocationNotFound`). The working interpreter is the repo-local
  `.conda/jax-0.9.2/bin/python` (now provides jax / jaxlib 0.10.0 on CPU); use
  that and rewrite the CLAUDE.md command form if you touch it.
- `/Users/suhjungdae/code/columbia/simsopt-jax/jax_native_remaining_impl_plan_2026-04-24.md`
  - architecture decisions. Keep upstream `Optimizable`, `Derivative`, and CPU
  `BiotSavart` as parity oracles; do not rewrite them.
- `/Users/suhjungdae/code/columbia/simsopt-jax/jax_gpu_port_todos_2026-04-08.md`
  - closed issues and PARTIAL items. Do not re-open DONE work or claim open GPU
  signoff without current artifacts.
- `src/simsopt/jax_core/specs.py`, `src/simsopt/jax_core/__init__.py` - the
  immutable spec / pure-function pattern every new JAX kernel must follow.
- `docs/using_jax_backend.md` and `docs/source/jax_migration.rst` - current
  user-visible support boundaries. If these disagree with source, fix the stale
  doc or mark the prompt item blocked.

Official docs to consult before introducing or changing JAX, SIMSOPT, or CUDA
behavior:

- Use Context7 under the local three-command budget. This review already
  resolved the relevant IDs, so use these IDs directly for docs unless a new
  library is introduced:
  - JAX: `npx ctx7@latest docs /google/jax "<full question>"`
  - SIMSOPT:
    `npx ctx7@latest docs /hiddensymmetries/simsopt "<full question>"`
  - CUDA: `npx ctx7@latest docs /websites/nvidia_cuda "<full question>"`
  If a new library is introduced, first run
  `npx ctx7@latest library <name> "<full question>"`, then fetch docs for the
  chosen `/org/project` ID.
- Use official JAX pages when the issue involves these topics:
  - `https://docs.jax.dev/en/latest/jit-compilation.html`
  - `https://docs.jax.dev/en/latest/transfer_guard.html`
  - `https://docs.jax.dev/en/latest/gpu_memory_allocation.html`
  - `https://docs.jax.dev/en/latest/persistent_compilation_cache.html`
  - `https://docs.jax.dev/en/latest/default_dtypes.html`
  - `https://docs.jax.dev/en/latest/async_dispatch.html`
- Use official SIMSOPT docs for public API parity:
  - `https://simsopt.readthedocs.io/latest/simsopt_user.field.html`
  - `https://simsopt.readthedocs.io/latest/simsopt_user.geo.html`
  - `https://simsopt.readthedocs.io/latest/simsopt_user.objectives.html`
- Use official CUDA / NVIDIA docs when the issue involves GPU runtime,
  driver/toolkit compatibility, streams, memory, or proof artifact claims:
  - `https://docs.jax.dev/en/latest/installation.html#nvidia-gpu`
  - `https://docs.nvidia.com/deploy/cuda-compatibility/`
  - `https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html`

Runtime preflight:

```bash
PY="${SIMSOPT_JAX_PYTHON:-/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python}"
"$PY" - <<'PY'
import jax, jaxlib, sys
print(sys.executable)
print("jax", jax.__version__, "jaxlib", jaxlib.__version__)
print("devices", [d.platform for d in jax.devices()])
PY
```

If the repo-local interpreter is missing or imports fail, mark environment
validation BLOCKED. Do not switch to another Python silently.

Local GPU preflight:

```bash
command -v nvidia-smi
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
```

If `nvidia-smi` is absent, exits nonzero, or reports no CUDA-capable device,
mark local GPU validation BLOCKED and keep all GPU evidence fields
`not_claimed` or `deferred`. Do not infer CUDA success from the CPU backend.

## 1. State file (resumption protocol)

Maintain `.artifacts/jax_port_goal/state.json` for resumption. Because
`.artifacts/` is not ignored, do not make in-progress state-only commits.
Commit state only as part of a completed or blocked item commit.

```json
{
  "manifest_version": 4,
  "active_scope": ["P0", "P1", "P2"],
  "active_scope_profile": "port_closure|cuda_perf_release",
  "items": [
    {
      "id": "<stable id>",
      "tier": "P0|P1|P2|P3|P4|P5",
      "files": ["src/...", "src/..."],
      "depends_on": ["<other id>"],
      "status": "pending|in_progress|complete|blocked|skipped",
      "closure_level": "open|cpu_oracle_complete|cuda_verified|blocked_architecture|blocked_dependency|blocked_env_gpu|blocked_math_parity|blocked_transfer_guard|skipped_future_scope",
      "evidence": {
        "source_audit": "src/...:line-line",
        "upstream_oracle": "/Users/suhjungdae/code/opensource/simsopt/src/simsopt/...",
        "upstream_audit_sha": "1b0cc3a96063197cdbdd01559e04c25456fbe6ff",
        "kernel_module": "src/simsopt/jax_core/...",
        "adapter_module": "src/simsopt/.../*_jax*.py",
        "jax_transform_plan": ".artifacts/jax_port_goal/plans/<id>-jax-transform.md",
        "math_physics_invariants": ".artifacts/jax_port_goal/plans/<id>-invariants.md",
        "oracle_contract": "fixed_scalar|fixed_gradient_vjp|optimizer_final_envelope|optimizer_trajectory",
        "serialization_restart_check": ".artifacts/jax_port_goal/restart/<id>.md",
        "coverage_matrix": ".artifacts/jax_port_goal/plans/<id>-coverage.md",
        "red_evidence": ".artifacts/jax_port_goal/red/<id>.txt",
        "parity_test": "tests/...::Test...",
        "transfer_guard_test": "tests/...",
        "multi_device_test": "tests/test_jax_import_smoke.py::...",
        "bench_artifact": ".artifacts/jax_port_goal/bench/<id>.json",
        "parity_lane": "direct-kernel|derivative-heavy|adjoint|...",
        "cuda_smoke": "not_claimed|deferred|verified",
        "cuda_proof": {
          "status": "not_claimed|deferred|verified",
          "artifact": ".artifacts/jax_port_goal/cuda/<id>-<sha>.json|CI URL|N/A",
          "current_sha": "<sha>",
          "worktree_dirty": true,
          "gpu_device": "H100|H200|A100|RTX...|N/A",
          "driver_version": "<driver>|N/A",
          "cuda_runtime": "<runtime>|N/A",
          "jax_version": "<jax>",
          "jaxlib_version": "<jaxlib>",
          "backend": "cuda|cpu|N/A",
          "xla_flags": "<XLA_FLAGS>",
          "cuda_visible_devices": "<CUDA_VISIBLE_DEVICES>",
          "peak_gpu_mem_mb": 0,
          "target_arrays_on_cuda": true,
          "compiled_executable_backend": "cuda|cpu|N/A"
        },
        "commit_sha": "<sha>"
      },
      "blocker": null
    }
  ],
  "current_iter": 0,
  "last_done_sha": "<sha>"
}
```

Schema migration note: v2 → v3 adds `coverage_matrix`, `upstream_audit_sha`,
`red_evidence`, and `bench_artifact`. v3 → v4 adds `active_scope_profile`,
`closure_level`, `jax_transform_plan`, `math_physics_invariants`,
`oracle_contract`, `serialization_restart_check`, and `cuda_proof`. A v2 or v3
state.json is forward-compatible only after each in-progress item is paused and
the missing artifacts are backfilled per section 4a and the required closure
checklists. Do not silently upgrade `manifest_version` without producing the new
artifacts.

For blocked items, replace `"blocker": null` with one of the section-5
refusal categories:

```json
{
  "category": "architecture|missing_dependency|parity_unreachable|transfer_guard_unreachable|tolerance_policy|gpu_run_needed",
  "detail": "free-text plus pointer to .artifacts/jax_port_goal/blockers/<id>-debug.md",
  "debug_artifact": ".artifacts/jax_port_goal/blockers/<id>-debug.md",
  "needs_user": true
}
```

The `cuda_smoke` field has a per-run lifetime:

- `not_claimed` is the default for any item validated only on CPU. Persists
  across runs.
- `deferred` requires a `deferred_reason` and an `expected_artifact` path
  in the item's plan. A `deferred` value DOES NOT carry forward
  automatically. The next run that touches this item must either promote
  it to `verified` with the cited artifact in hand, or re-park it as
  `deferred` with a fresh dated justification appended to the plan. The
  final REPORT must list every `deferred` entry under "Deferred CUDA
  verification" with the artifact path the next run needs to produce.
  `deferred` is not a `closure_level`; for `active_scope_profile=port_closure`,
  a CPU-oracle-complete item with deferred CUDA proof remains `status=complete`
  and `closure_level=cpu_oracle_complete`.
- `verified` requires a current-SHA CUDA artifact path in evidence (e.g.
  `.artifacts/jax_port_goal/cuda/<id>-<sha>.json` or a CI run URL). The
  artifact must be produced by an approved GPU run, not inferred from
  CPU + HLO proxies.

`cuda_proof.status` must match `cuda_smoke`. For `verified`, every field in the
`cuda_proof` object must be populated from the real CUDA run artifact; `N/A` is
valid only for `not_claimed` or `deferred`. The artifact must prove the compiled
executable backend is CUDA and that target arrays were resident on CUDA devices
for the measured path. CPU proxy, HLO-only, or dry-run output may be cited in
the plan, but cannot populate `cuda_proof.status: verified`.

At each iteration:

1. Read `state.json` if present; otherwise create it from this manifest.
2. Pick the lowest-numbered `pending` item in `active_scope` whose
   `depends_on` items are all `complete`.
3. Set the item `in_progress` in the working tree.
4. Run section 4 end-to-end.
5. On success: set `complete`, fill evidence, set `closure_level` to
   `cpu_oracle_complete` or `cuda_verified`, and make one scoped commit with
   code, tests, docs, plans, and state for that item.
6. On unrecoverable blocker: set `blocked`, fill blocker evidence, and make one
   scoped blocker commit only if the blocker note/state file is useful to keep.
   Set `closure_level` from the section-5 blocker-to-closure mapping.
7. If no eligible pending active item remains: stop and write the final report.

## 2. Architecture invariants - never violate

- Upstream `Optimizable`, `Derivative`, and CPU `BiotSavart` stay as-is. They
  are the parity oracle and Python-facing boundary. JAX-native work uses
  immutable specs, pytrees, and spec-driven pure functions, not new Optimizable
  subclasses.
- Compiled JAX lanes consume specs and explicit DOF arrays. They never read
  mutable wrapper state.
- `Derivative` projection lives only at compatibility boundaries; internal
  gradient paths use native cotangents.
- Operator-backed solves stay on production adjoint paths. Dense PLU /
  Jacobian factors are public metadata in exact lanes and load-bearing only in
  LS lanes under the existing `boozersurface_jax.py` and
  `surfaceobjectives_jax.py` contracts.
- No silent CPU fallback in any JAX target mode.
- No new tolerances outside `PARITY_LADDER_TOLERANCES`.
- No new runtime dependency without explicit human approval.
- No dynamic imports. No `Any` casts. No broad `except Exception`. No
  `try/except ImportError` around runtime configuration.
- Do not launch GPU jobs unless the user explicitly approves them.

## 3. WORK MANIFEST (dependency-ordered)

### Tier P0 - active: current JAX coverage audit and closeouts

These files already contain JAX paths in the current tree. Do not rewrite them
as if they were CPU-only. For each item, audit current source first, identify
the remaining CPU/host-transfer edge if one exists, and either close that edge
or mark the item complete with evidence.

1. [ ] `field/coilobjective.py` (`CurrentPenalty`) and remaining
   `geo/curveobjectives.py` wrappers around `geo/_distance_jax.py`
2. [ ] `field/selffield.py` - regularized self-field JAX coverage and tests
3. [ ] `geo/curveobjectives.py` non-distance objectives - `CurveLength`,
   `LpCurveCurvature(Barrier)`, `LpCurveTorsion`, `ArclengthVariation`,
   `MeanSquaredCurvature`, `LinkingNumber`, `FramedCurveTwist`. Distance
   classes (`CurveCurveDistance(Barrier)`, `CurveSurfaceDistance`,
   `MinimumDistance`, `MinCurveCurveDistance`) are owned by item 1.
4. [ ] `geo/strain_optimization.py` - strain accumulators / postprocessing
5. [ ] `field/force.py` - finite-build force kernel pre-compute layers

### Tier P1 - active: independent C++ kernels

Port only after proving the upstream C++ API shape and downstream consumer
contract. If the algorithm requires a dependency not already in this repo,
block the item instead of adding the dependency.

6. [ ] `simsoptpp/dommaschk.cpp` + `simsoptpp/reiman.cpp` to
   `jax_core/analytic_fields.py` + specs
7. [ ] `simsoptpp/regular_grid_interpolant_3d*` to
   `jax_core/regular_grid_interp.py`
8. [ ] `simsoptpp/tracing.cpp` to `jax_core/tracing.py` using an in-repo JAX RK
   implementation only

### Tier P2 - active: Python wrappers that depend on Tier P1

9. [ ] `field/magneticfieldclasses.py` (`Dommaschk`, `Reiman`,
   `InterpolatedField`) - depends on items 6 and 7
10. [ ] `field/tracing.py` - depends on item 8

### Tier P3 - future-scope; skipped unless `active_scope` includes P3

11. [ ] `geo/framedcurve.py` + `geo/orientedcurve.py` - ODE/framing replacement
12. [ ] `geo/qfmsurface.py` - private on-device optimizer contract
13. [ ] `geo/finitebuild.py` - finite-build geometry kernel

### Tier P4 - future-scope; skipped unless `active_scope` includes P4

14. [ ] `simsoptpp/dipole_field.cpp` to `jax_core/dipole_field.py`
15. [ ] `simsoptpp/permanent_magnet_optimization.cpp` to
    `jax_core/pm_optimization.py`
16. [ ] `field/magneticfieldclasses.py` (`DipoleField`) - depends on 14
17. [ ] `geo/permanent_magnet_grid.py` - depends on 14 and 15
18. [ ] `solve/permanent_magnet_optimization.py` - depends on 14 and 15
19. [ ] `simsoptpp/wireframe_optimization.cpp`,
    `magneticfield_wireframe.cpp`, and `wireframe_field_impl.h` to
    `jax_core/wireframe.py`
20. [ ] `field/wireframefield.py` - depends on 19
21. [ ] `solve/wireframe_optimization.py` - depends on 19

### Tier P5 - future-scope; skipped unless `active_scope` includes P5

22. [ ] `simsoptpp/boozerradialinterpolant.cpp` and
    `boozermagneticfield*.h` to `jax_core/boozer_radial_interp.py`
23. [ ] `field/boozermagneticfield.py` - depends on 22

Skip list (do not port; if you touch these, escalate):

- `mhd/*` (external Fortran / C++ binary wrappers)
- `_core/optimizable.py`, `_core/derivative.py`, `_core/json.py`
- `field/biotsavart.py` (parity oracle)
- `objectives/{least_squares,constrained,functions,utilities}.py`
- `solve/serial.py`, `solve/mpi.py` (orchestration)
- `util/`, `geo/plotting.py`, `field/mgrid.py`, `field/coilset.py`
- `geo/{surfacegarabedian,surfacehenneberg,accessibility,hull,ports,wireframe_toroidal}.py`
  (auxiliary / historic / CAD; port only on explicit user request)

If a non-skip-list active item cannot be ported without violating these
invariants, mark BLOCKED with `needs_user: true` and move on. Do not silently
rewrite around the invariant.

## 4. Per-item protocol

For each active manifest item:

### 4a. Plan (write to `.artifacts/jax_port_goal/plans/<id>.md`)

- Current-state source audit: what JAX path already exists, what remains CPU,
  and exact file:line evidence.
- Upstream oracle audit: exact upstream C++/Python API and numerical contract.
  Record the upstream SHA audited in the item's `upstream_audit_sha` evidence
  field; if upstream advances during the port, re-audit before claiming
  complete.
- Downstream consumer graph: wrappers, examples, docs, tests, benchmarks, and
  integration paths that consume this item.
- The C++ / Python kernel(s) being replaced or extended. Cite file:line.
- The new or existing spec name and pure-function names in `jax_core/`.
- The adapter class or module and where it slots into existing code.
- Parity-ladder lane (direct-kernel / derivative-heavy / adjoint / etc.).
- Test files to create or extend: fixed-state parity, VJP/gradient parity,
  transfer guard, multi-device CPU, and downstream integration where relevant.
- Existing-test coverage matrix (mandatory artifact at
  `.artifacts/jax_port_goal/plans/<id>-coverage.md`). The matrix is the
  primary parity-coverage gate; without it the item cannot reach COMPLETE.
  Build it as follows:
  - Enumerate every existing test exercising the same kernel / numerical
    contract, in BOTH locations:
    - this repo: `git grep -nE "<KernelName>|<simsoptpp_symbol>|<public_class>" tests/`
    - upstream SIMSOPT:
      `git -C /Users/suhjungdae/code/opensource/simsopt grep -nE "..." src/simsopt/tests/`
  - For each test, record `{repo_or_upstream, test_path, node_id, brief_intent}`
    plus a classification:
    - `covered_by_unit_parity` — equivalent JAX parity test exists; cite path
    - `covered_by_integration_parity` — pipeline coverage via a named JAX
      integration test; cite path
    - `oracle_only` — exists solely to validate CPU/C++ behavior; not portable
    - `wrapper_only` — exercises `Optimizable` mutability or simsoptpp
      plumbing; not portable
    - `not_applicable` — one-line reason (external binary, file I/O, plotting)
    - `blocked` — refusal category from section 5; cite blocker note
  - Empty-oracle case: if both grep searches return zero applicable tests,
    state that explicitly in the matrix AND add at least one new parity test
    against a hand-derived reference (closed-form, NumPy oracle, or
    documented limiting case). Empty oracle is not a license to skip
    parity work.
  - Stale-upstream guard: every matrix row that cites an upstream test
    records the upstream SHA at audit time. If `upstream_audit_sha` advances
    before the item is closed, re-run the upstream grep and append any new
    tests to the matrix before marking complete.
- Risk register: what could break, what existing test would catch it, and what
  new test closes the gap.

### 4b. Implement

- Spec in `jax_core/specs.py` when a new immutable spec is actually needed.
- Kernel function in `jax_core/<area>.py`: `*_from_spec(spec, ...)` and
  `*_from_dofs(spec, dofs, ...)` variants. Pure, no host transfers, no Python
  control flow over traced values.
- Adapter in the Python-facing module: spec extraction at construction,
  JAX-jitted hot path, and `Derivative` projection only at the public VJP.
- Stage host scalars with `_device_scalars` helpers or explicit
  `jnp.asarray(..., dtype=jnp.float64)` outside jitted hot paths.
- No bare `jnp.asarray(numpy_array)` in hot paths; it violates the
  transfer-guard contract.
- Cast JAX scalar boundary returns with `int()` / `bool()` before storing them
  in result dicts consumed by SciPy / NumPy.
- Preserve existing public API behavior unless the item is explicitly blocked
  for a required user-facing API decision.
- Do not add dependencies, fallback modes, compatibility shims, broad
  try/except blocks, or duplicate tolerance constants.

### 4c. Validate (gate - all applicable checks pass before status = complete)

Run from repo root. Use this interpreter unless the user explicitly changes it:

```bash
PY="${SIMSOPT_JAX_PYTHON:-/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python}"
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
```

Static checks:

```bash
ruff check <changed-files>
ruff format --check <changed-files>
```

Core public pure-JAX regression set:

```bash
"$PY" -m pytest \
  tests/test_jax_import_smoke.py \
  tests/field/test_biotsavart_jax.py \
  tests/geo/test_surface_fourier_jax.py \
  tests/geo/test_boozer_residual_jax.py \
  tests/objectives/test_integral_bdotn_jax.py \
  tests/geo/test_boozer_derivatives_jax.py \
  tests/geo/test_boozersurface_jax.py \
  tests/integration/test_jax_native_path.py \
  -m "not private_optimizer_runtime" -v
```

Private optimizer and integration set:

```bash
"$PY" -m pytest \
  tests/geo/test_boozersurface_jax.py \
  tests/integration/test_single_stage_jax.py \
  -m "private_optimizer_runtime" -v
```

Cross-env integration when simsoptpp is required. The historical
`candidate-fixed` interpreter is currently broken (numpy / jax string-dtype
mismatch on `import simsopt`); guard the invocation so the lane reports BLOCKED
instead of silently failing, and do not switch interpreters without user
approval:

```bash
PY_CROSSENV="${SIMSOPT_CROSSENV_PYTHON:-/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python}"
if [ -x "$PY_CROSSENV" ] && "$PY_CROSSENV" -c "import simsopt" >/dev/null 2>&1; then
  "$PY_CROSSENV" -m pytest tests/integration/ -v
else
  echo "BLOCKED: cross-env interpreter unusable at $PY_CROSSENV; set SIMSOPT_CROSSENV_PYTHON or escalate" >&2
  exit 1
fi
```

Multi-device CPU subprocess proxy. Note: `tests/subprocess/` contains the
target scripts driven by outer tests via `_assert_python_script_passes`;
running `pytest tests/subprocess/` directly collects zero tests. The real
4-device entries live in `tests/test_jax_import_smoke.py` (e.g.
`test_grouped_biot_savart_coil_collective_parity_and_lowering`,
`test_grouped_biot_savart_accepts_explicit_point_sharding`,
`test_pairwise_penalty_accepts_explicit_row_sharding`). The outer test
already propagates the multi-device env (`XLA_FLAGS=--xla_force_host_platform_device_count=4`,
`SIMSOPT_JAX_SHARDING=…`) into its own subprocess via
`_build_clean_subprocess_env` / `extra_env`, and the parent process performs
no multi-device assertion; therefore do NOT prepend `XLA_FLAGS` at the parent
level. On slower hosts the extra parent-side 4-device JAX initialization can
push the subprocess past its hard-coded 60 s timeout. Run:

```bash
"$PY" -m pytest tests/test_jax_import_smoke.py \
  -k "collective or sharding or subprocess" -v
```

If you change which outer test names drive the subprocess cases, update the
`-k` filter accordingly.

Transfer-guard hardening for affected tests:

```bash
SIMSOPT_JAX_TRANSFER_GUARD=disallow \
  "$PY" -m pytest <affected test paths> -v
```

Add and run item-specific tests for both `stellsym=False` and `stellsym=True`
where applicable.

Required parity evidence:

- [ ] Forward kernel parity vs upstream C++/SciPy oracle within the lane's
  `PARITY_LADDER_TOLERANCES` entry.
- [ ] VJP / gradient parity using finite differences or the upstream VJP
  contract within the same lane.
- [ ] Downstream API parity for every wrapper this item feeds.
- [ ] For collective paths: StableHLO or optimized HLO contains the expected
  collective, checked with `jax.jit(...).lower(...).as_text()` or
  `jax.jit(...).lower(...).compile().as_text()` as appropriate. Mark N/A
  only if `git grep` over the item's diff shows zero `shard_map`, `psum`,
  `all_reduce`, or `pjit` introductions; otherwise this checkbox is
  required.

Cited-test integrity (no fictional or de-fanged tests):

- [ ] Every `parity_test` / `transfer_guard_test` / `multi_device_test` path
  in this item's state-evidence entry resolves on disk at the commit's tree
  (`git show <sha>:<path>` succeeds) and the named pytest node is collected
  by `pytest --collect-only -q <path>::<node>`.
- [ ] No cited test is decorated with `@pytest.mark.skip`,
  `@pytest.mark.skipif`, `@pytest.mark.xfail`, or wrapped in a
  `pytest.skip(...)` early-return at module / class / function scope.
- [ ] Every parity assertion in the new test imports its tolerance from
  `benchmarks.validation_ladder_contract.PARITY_LADDER_TOLERANCES[<lane>]`
  or a helper that reads from it. No `atol=` / `rtol=` numeric literals
  inlined inside the test body. Run
  `git diff <base>..HEAD -- <new test paths> | grep -E "(atol|rtol)\s*=\s*[0-9eE.+-]+"`
  and confirm zero hits.

Production-scale floor (the lesson from closed TODO #39 — toy fixtures
silently pass while real workloads diverge):

- [ ] At least one parity fixture for this item runs at production scale.
  Floor for Biot-Savart / flux / Boozer / surface kernels: `nphi ≥ 16`,
  `ntheta ≥ 8`, `ncoils ≥ 4`. For curve / coil-objective kernels:
  `ncoils ≥ 4`, `nquadpoints ≥ 64`. For tracing / fieldline integrators:
  trajectory length ≥ the smallest production lab-notebook setting in
  `examples/`. For new kernel families with no precedent: cite the closest
  existing production-scale test in the repo and match or exceed its grid.
- [ ] If the production-scale fixture cannot run inside the validation
  budget on this host, the item is BLOCKED, not COMPLETE. Do not mark N/A
  on this checkbox.

Coverage completeness (the existing-test parity gate; see 4a coverage matrix):

- [ ] The coverage matrix at `.artifacts/jax_port_goal/plans/<id>-coverage.md`
  exists, is referenced from the item's `coverage_matrix` evidence field,
  and lists every test surfaced by the repo and upstream `git grep` per 4a.
- [ ] Every matrix row is classified — no row is `unclassified` or missing
  a status. `covered_by_unit_parity` / `covered_by_integration_parity` rows
  cite an existing JAX test path that resolves on disk and is collected by
  `pytest --collect-only`.
- [ ] For every cited JAX parity test in the matrix: it executes against the
  item's commit and exits zero at the lane tolerance from
  `PARITY_LADDER_TOLERANCES`. The exact pytest command + result line is
  recorded in the item's plan or in the commit message.
- [ ] `oracle_only` / `wrapper_only` / `not_applicable` rows each carry a
  one-line reason. Generic classifications without a reason are not valid.
- [ ] `blocked` rows cite a refusal category from section 5 and link the
  `<id>-debug.md` artifact.
- [ ] If `upstream_audit_sha` advanced between plan time and validate time,
  the upstream grep has been re-run and any new tests appended and
  classified before this checkbox is checked.
- [ ] Empty-oracle items (zero rows from both greps) include at least one
  new parity test built against a hand-derived / closed-form / NumPy
  oracle, with the oracle source cited in the test docstring.

Red-step evidence (lightweight TDD discipline — catches tests written
after the implementation that ride on the already-green lane):

- [ ] The new parity test cited in this item's `parity_test` evidence was
  confirmed FAILING against the parent commit (pre-implementation tree)
  before the implementation began. Capture the failing pytest stderr /
  stdout (or a `< 100`-line excerpt with file:line pointers) to
  `.artifacts/jax_port_goal/red/<id>.txt` and reference it from
  `red_evidence` in state.json and from the commit message.
- [ ] For partial-coverage closeouts where the existing JAX path already
  passes most tests, the new parity test targets the specific NumPy edge,
  host-transfer leak, or scale-floor behavior being closed — not a
  rerun of the already-green path. Show this by citing the asserted
  invariant (e.g., `transfer_guard=disallow` clean, production-scale
  grid, finite-difference VJP at the lane tolerance) that the test
  exercises and that the pre-impl tree did not satisfy.
- [ ] If the item's only new test is the empty-oracle hand-derived
  reference and there is no pre-impl tree at which it could fail (e.g.,
  the kernel did not exist), state that explicitly in `red_evidence`
  with the reason; this is the one acceptable N/A and only applies to
  Tier P1 new-kernel ports.

Required performance / memory evidence. "Proportional to item size" is not
a license to skip; it bounds the scope of the bench, not whether one exists.
Every hot-path item must have at least one of (a) a real micro-bench or (b)
a written "no perf change expected because …" justification cited against
concrete kernel changes — never both omitted.

- [ ] For hot-path kernels: a `benchmarks/` micro-bench file is committed
  alongside the implementation. It compares median + p95 warm-run time
  against the upstream baseline (or the closest existing JAX kernel if no
  upstream baseline exists) over **≥ 100 timed calls** after a
  `.block_until_ready()`-gated warmup of **≥ 5 calls**. Timings come from
  `time.perf_counter()` (not wall-clock or `%timeit`).
- [ ] Bench output (median, p95, std, sample count) is appended to the
  item's plan or saved under `.artifacts/jax_port_goal/bench/<id>.json`
  and referenced from the commit message. Single-shot timings are not
  acceptable.
- [ ] For any new dense materialization: report bytes, compare against
  `max_dense_jacobian_bytes`, and keep the operator-backed alternative path.
- [ ] If buffer donation is used, prove it at a real outer-jit boundary
  with HLO showing the donated buffer. Internal `fori_loop` carries do not
  count as donation evidence.
- [ ] CPU-only validation must be stated explicitly in the commit message
  and in the item's `cuda_smoke` field (`not_claimed` or `deferred`). This
  is a reporting requirement, not a substitute for the production-scale
  parity gate above.
- [ ] The item's `jax_transform_plan` artifact names the transform stack,
  static-shape strategy, dense materialization budget, donation decision, and
  expected largest array shape/dtype. The benchmark or HLO artifact must link
  back to this plan.

GPU reality:

- [ ] Do not claim real CUDA success without a current-SHA CUDA artifact from
  an approved GPU run.
- [ ] Multi-device CPU subprocesses and HLO inspection are useful regression
  proxies, but they are not GPU proof.
- [ ] Do not launch GPU jobs unless the user explicitly approves them.
- [ ] Tag each CPU-only item's state evidence with `cuda_smoke: not_claimed`
  or `cuda_smoke: deferred`, never `verified`.
- [ ] `active_scope_profile=port_closure` may finish with
  `closure_level=cpu_oracle_complete` plus `cuda_smoke=deferred` entries in the
  final REPORT; it may not claim CUDA release readiness.
- [ ] `active_scope_profile=cuda_perf_release` requires
  `closure_level=cuda_verified` for every CUDA-claimed item and a populated
  `cuda_proof` object from a real GPU artifact.

### 4d. Commit

One scoped commit per completed active item, only after validation or a
documented blocker. Preserve unrelated dirty and untracked files.

Message format:

```text
jax-port: <short title> [item <id>]

- spec: <spec name or existing>
- kernel: jax_core/<file or existing>
- adapter: <file>
- status: complete | blocked | skipped
- closure level: cpu_oracle_complete | cuda_verified | blocked_architecture | blocked_dependency | blocked_env_gpu | blocked_math_parity | blocked_transfer_guard | skipped_future_scope
- parity lane: <lane>
- oracle contract: fixed_scalar | fixed_gradient_vjp | optimizer_final_envelope | optimizer_trajectory
- JAX transform plan: .artifacts/jax_port_goal/plans/<id>-jax-transform.md
- math/physics invariants: .artifacts/jax_port_goal/plans/<id>-invariants.md
- serialization/restart check: .artifacts/jax_port_goal/restart/<id>.md
- tests: <new or affected test paths>
- coverage matrix: .artifacts/jax_port_goal/plans/<id>-coverage.md
- red evidence: .artifacts/jax_port_goal/red/<id>.txt (or "N/A: <reason>"
  for Tier P1 new-kernel ports with no pre-impl tree)
- bench artifact: .artifacts/jax_port_goal/bench/<id>.json (or "N/A: no
  hot-path change because <one-line justification>")
- upstream audit sha: <upstream SIMSOPT sha audited for this item>
- cuda smoke: not_claimed | deferred | verified
- cuda proof: .artifacts/jax_port_goal/cuda/<id>-<sha>.json | CI URL | N/A
- docs checked: <official docs URLs or Context7 IDs>
```

Every field is mandatory. Use the literal `N/A: <reason>` form when a
field genuinely does not apply per the section 4c carve-outs; an empty
field or a missing line is treated as omitted evidence and the commit
must not be made.

Do not add co-author footers unless the user explicitly asks for them.

## 5. Refusal triggers (BLOCK the item, write the diagnostic, move on)

A refusal trigger fires only after the diagnostic budget below is exhausted
and the artifact in `.artifacts/jax_port_goal/blockers/<id>-debug.md` has
been written. A BLOCKED status without that artifact is not a valid stop —
treat it as in_progress and finish the budget.

Exception: `gpu_run_needed` caused only by absent local CUDA hardware, absent
`nvidia-smi`, missing remote GPU quota/auth, or lack of explicit user approval
does not require two debug timeboxes. In that case, run the preflight command,
write the blocker artifact immediately with the exact approved-run command and
expected CUDA artifact path, set `status=blocked` and
`closure_level=blocked_env_gpu`, and move on. Do not launch the GPU job. If the
item has already satisfied the CPU/C++ oracle gates under
`active_scope_profile=port_closure` and only optional CUDA proof is deferred,
do not use this blocker path; set `status=complete`,
`closure_level=cpu_oracle_complete`, `cuda_smoke=deferred`, and
`cuda_proof.status=deferred`.

Blocker-to-closure mapping is exact:

| blocker `category` | required `closure_level` |
| --- | --- |
| `architecture` | `blocked_architecture` |
| `missing_dependency` | `blocked_dependency` |
| `parity_unreachable` | `blocked_math_parity` |
| `tolerance_policy` | `blocked_math_parity` |
| `transfer_guard_unreachable` | `blocked_transfer_guard` |
| `gpu_run_needed` | `blocked_env_gpu` |

Diagnostic budget (mandatory before BLOCKED is set):

- Two timeboxes of **≤ 2 hours each** of active debugging.
- After each timebox, append a dated section to
  `.artifacts/jax_port_goal/blockers/<id>-debug.md` containing: the exact
  failing command, the captured stderr/stdout (or a `< 100`-line excerpt
  with file:line pointers), the current hypothesis, the next experiment.
- If the second timebox does not close the issue, mark BLOCKED with the
  category below and link the debug artifact from the blocker note.

Refusal categories:

- `architecture` — the port would require subclassing or replacing
  `Optimizable`, `Derivative`, or CPU `BiotSavart`.
- `missing_dependency` — the algorithm requires a runtime dependency not
  already in `pyproject.toml`. Propose the dependency in the blocker note;
  do not add it.
- `parity_unreachable` — parity vs the upstream C++/SciPy oracle fails at
  the lane tolerance after the two-timebox budget. The blocker note must
  include the worst-case residual, the lane, the fixture parameters, and a
  ruled-out hypothesis list.
- `transfer_guard_unreachable` — closing the item would require weakening
  the `transfer_guard=disallow` contract. Cite the offending op and the
  JAX upstream issue / docs link if one exists.
- `tolerance_policy` — closing the item would require modifying
  `validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`. Tolerances
  are user-owned policy; never edit them to make a test pass.
- `gpu_run_needed` — the work requires a GPU run and the user has not
  approved launching one. Record what artifact the run would produce.

## 6. Anti-patterns - refuse unconditionally

Implementation hygiene:

- Silent CPU fallback inside a JAX target lane.
- `except Exception:` to hide a real error.
- Dynamic imports or `Any` casts.
- `jnp.asarray(host_array)` in a hot path.
- Dense materialization on production adjoint paths.
- Removing existing CPU / C++ paths. The parity oracle stays.

Reporting and gate hygiene:

- Declaring a CUDA result from CPU-only or HLO-only evidence.
- Marking `cuda_smoke: verified` when `cuda_proof.status` is not `verified`
  with a current-SHA CUDA artifact.
- "It is fine, the test passes locally" without the applicable validation
  set.
- Reporting CPU-only closure as full end-to-end closure.
- Reporting `closure_level=cpu_oracle_complete` as CUDA performance release
  readiness.

Test integrity (do not game the parity gates):

- Inlining `atol` / `rtol` literals inside a new test instead of importing
  from `benchmarks.validation_ladder_contract.PARITY_LADDER_TOLERANCES`.
- Citing a `parity_test` / `transfer_guard_test` / `multi_device_test`
  path that does not exist on disk at the commit's tree or is decorated
  with `@pytest.mark.skip` / `skipif` / `xfail` or wrapped in a module-
  or class-level `pytest.skip(...)`.
- Marking the production-scale parity fixture as N/A. The floor in
  section 4c is mandatory unless the item is BLOCKED.
- Skipping a `stellsym=True` case in a parity test.
- Submitting a one-shot timing as a benchmark; section 4c requires ≥ 100
  timed calls with median + p95.

Self-termination hygiene:

- BLOCKING an item without writing `.artifacts/jax_port_goal/blockers/<id>-debug.md`
  and exhausting the two-timebox diagnostic budget from section 5, except for
  the explicit `gpu_run_needed` environment/approval exception in section 5.
- BLOCKING more than one active item per run with `architecture`,
  `parity_unreachable`, or `transfer_guard_unreachable` self-issued.
  Hitting that quota forces ESCALATE before any further BLOCK.
- Marking an item `skipped` without an explicit human directive recorded
  in the state file.

## 7. Stop condition

Stop only when ALL of the following hold; the OR shortcuts that previously
allowed self-termination via mass-BLOCKED have been removed:

- Every item in `active_scope` is `complete`, `blocked`, or `skipped`. AND
- Every completed item has `closure_level=cpu_oracle_complete` or
  `closure_level=cuda_verified`; every blocked/skipped item has the matching
  closure level from section 5 or `skipped_future_scope`. AND
- The applicable targeted, regression, transfer-guard, and multi-device CPU
  test runs from section 4c are green at the current HEAD. AND
- The final report `.artifacts/jax_port_goal/REPORT.md` exists and lists
  every active item's status, closure level, oracle contract, and one-line
  evidence pointer. AND
- The BLOCK / ESCALATE quota was respected: at most **one** active item per
  run may be BLOCKED with `architecture` / `parity_unreachable` /
  `transfer_guard_unreachable` self-issued by the agent without prior
  ESCALATE. Hitting that quota forces an ESCALATE entry that pauses the
  run for user decision; the agent does not silently roll forward by
  BLOCKING the remaining items in the same run.
- For broad source changes (item touches > 5 source files or any file in
  `src/simsopt/jax_core/`), the full test suite was run with the repo-local
  interpreter and is green. AND
- For every `cuda_smoke: deferred` evidence entry, the REPORT lists it
  under "Deferred CUDA verification" with the artifact path the next run
  needs to produce. `deferred` does not carry across runs implicitly; the
  next run must explicitly promote it to `verified` or re-park it with a
  fresh justification.
- If `active_scope_profile=cuda_perf_release`, every CUDA-claimed item has
  `closure_level=cuda_verified` and a populated `cuda_proof` object from an
  approved real GPU run. Otherwise the run is not a CUDA performance release
  closure.

If full-suite, cross-env, or GPU validation is blocked by environment or
hardware, state that explicitly in the REPORT and do not claim full
end-to-end closure. Reporting CPU-only closure as full closure is an
anti-pattern (section 6).

Write `.artifacts/jax_port_goal/REPORT.md` with:

- Completed active items and validation commands.
- Blocked/skipped active items and exact blockers.
- Closure level taxonomy for every active item:
  `cpu_oracle_complete`, `cuda_verified`, `blocked_architecture`,
  `blocked_dependency`, `blocked_env_gpu`, `blocked_math_parity`,
  `blocked_transfer_guard`, or `skipped_future_scope`.
- Oracle contract for every active item:
  `fixed_scalar`, `fixed_gradient_vjp`, `optimizer_final_envelope`, or
  `optimizer_trajectory`.
- JAX transform/memory strategy artifacts and math/physics invariant artifacts.
- Serialization/restart compatibility artifacts or explicit `N/A: <reason>`.
- CPU, multi-device CPU, transfer-guard, cross-env, and GPU evidence separated.
- CUDA proof table with `cuda_smoke`, `cuda_proof.status`, artifact path,
  device/backend, and whether target arrays were resident on CUDA.
- Any stale rows discovered in `jax_gpu_port_todos_2026-04-08.md` or local docs.
- Any upstream/downstream API, math, physics, or computation discrepancy found.

## 8. Escalate immediately (do not loop in the dark)

ESCALATE writes a user-facing `.artifacts/jax_port_goal/blockers/<id>.md`
note AND a `needs_user: true` entry on the next-iteration state.json, then
moves to the next eligible item. ESCALATE and BLOCKED share the same
artifact contract — see section 5's diagnostic budget. The only difference
is recipient: ESCALATE entries are surfaced at the top of the final REPORT
under "User decisions required" and the run pauses on them by default.

Escalate triggers:

- A blocker would require a user-facing API change to upstream `Optimizable`,
  `Derivative`, public method signatures, env-var contracts, or
  `pyproject.toml` dependencies.
- A skip-list item is actually critical to an active item.
- An open PARTIAL item from `jax_gpu_port_todos_2026-04-08.md` (#24, #26,
  #27, #29) becomes load-bearing.
- The two-timebox parity budget from section 5 is exhausted and the failure
  is not categorically refusable (i.e., none of the refusal categories fit
  cleanly). Write the diagnostic to
  `.artifacts/jax_port_goal/blockers/<id>.md` with the same content the
  BLOCKED protocol requires, plus a "Proposed user decision" section.

Begin at iteration 0: read `state.json` if it exists. Otherwise create it from
the manifest above with P0-P2 pending, P3-P5 skipped unless `active_scope`
explicitly includes them, and `current_iter=0`.
~~~

---

## Notes for the human launching this

- P0-P2 is the intended default scope. P3-P5 include algorithmic research lanes
  such as PM, wireframe, Boozer radial interpolation, and frame ODE solvers;
  expand scope only after a fresh review.
- The state file is for resumability, not a reason to create state-only
  commits. Commit only scoped useful artifacts.
- CPU multi-device and HLO checks are regression proxies. They do not prove real
  CUDA execution.
- The skip list is load-bearing: many entries are CPU-only-by-design or parity
  oracles, not neglected port work.
