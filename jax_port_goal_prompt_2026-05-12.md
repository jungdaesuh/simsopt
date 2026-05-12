# `/goal` prompt - close active-scope JAX port gaps

Date: 2026-05-12
Branch context: `gpu-purity-stage2-20260405`
Repo: `/Users/suhjungdae/code/columbia/simsopt-jax`
Reviewed against current HEAD: `8b471e8e3`

## Review status

This revision fixes issues found in the original prompt after checking the
current tree, upstream SIMSOPT, the repo-local JAX runtime, and official JAX /
SIMSOPT documentation:

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

Official docs to consult before introducing or changing JAX behavior:

- Resolve JAX docs with Context7 first:
  `npx ctx7@latest library JAX "<full question>"`
  then:
  `npx ctx7@latest docs /google/jax "<full question>"`.
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

## 1. State file (resumption protocol)

Maintain `.artifacts/jax_port_goal/state.json` for resumption. Because
`.artifacts/` is not ignored, do not make in-progress state-only commits.
Commit state only as part of a completed or blocked item commit.

```json
{
  "manifest_version": 2,
  "active_scope": ["P0", "P1", "P2"],
  "items": [
    {
      "id": "<stable id>",
      "tier": "P0|P1|P2|P3|P4|P5",
      "files": ["src/...", "src/..."],
      "depends_on": ["<other id>"],
      "status": "pending|in_progress|complete|blocked|skipped",
      "evidence": {
        "source_audit": "src/...:line-line",
        "upstream_oracle": "/Users/suhjungdae/code/opensource/simsopt/src/simsopt/...",
        "kernel_module": "src/simsopt/jax_core/...",
        "adapter_module": "src/simsopt/.../*_jax*.py",
        "parity_test": "tests/...::Test...",
        "transfer_guard_test": "tests/...",
        "multi_device_test": "tests/subprocess/...",
        "parity_lane": "direct-kernel|derivative-heavy|adjoint|...",
        "cuda_smoke": "not_claimed|deferred|verified",
        "commit_sha": "<sha>"
      },
      "blocker": null
    }
  ],
  "current_iter": 0,
  "last_done_sha": "<sha>"
}
```

For blocked items, replace `"blocker": null` with:

```json
{ "category": "...", "detail": "...", "needs_user": true }
```

At each iteration:

1. Read `state.json` if present; otherwise create it from this manifest.
2. Pick the lowest-numbered `pending` item in `active_scope` whose
   `depends_on` items are all `complete`.
3. Set the item `in_progress` in the working tree.
4. Run section 4 end-to-end.
5. On success: set `complete`, fill evidence, and make one scoped commit with
   code, tests, docs, plans, and state for that item.
6. On unrecoverable blocker: set `blocked`, fill blocker evidence, and make one
   scoped blocker commit only if the blocker note/state file is useful to keep.
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

1. [ ] `field/coilobjective.py` and remaining wrappers around `geo/_distance_jax.py`
2. [ ] `field/selffield.py` - regularized self-field JAX coverage and tests
3. [ ] `geo/curveobjectives.py` - Lp curvature / length / centerline-offset
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

9. [ ] `field/magneticfieldclasses.py` (`Dommaschk`, `Reiman`, `DipoleField`,
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
16. [ ] `geo/permanent_magnet_grid.py` - depends on 14 and 15
17. [ ] `solve/permanent_magnet_optimization.py` - depends on 14 and 15
18. [ ] `simsoptpp/wireframe_optimization.cpp`,
    `magneticfield_wireframe.cpp`, and `wireframe_field_impl.h` to
    `jax_core/wireframe.py`
19. [ ] `field/wireframefield.py` - depends on 18
20. [ ] `solve/wireframe_optimization.py` - depends on 18

### Tier P5 - future-scope; skipped unless `active_scope` includes P5

21. [ ] `simsoptpp/boozerradialinterpolant.cpp` and
    `boozermagneticfield*.h` to `jax_core/boozer_radial_interp.py`
22. [ ] `field/boozermagneticfield.py` - depends on 21

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
- Downstream consumer graph: wrappers, examples, docs, tests, benchmarks, and
  integration paths that consume this item.
- The C++ / Python kernel(s) being replaced or extended. Cite file:line.
- The new or existing spec name and pure-function names in `jax_core/`.
- The adapter class or module and where it slots into existing code.
- Parity-ladder lane (direct-kernel / derivative-heavy / adjoint / etc.).
- Test files to create or extend: fixed-state parity, VJP/gradient parity,
  transfer guard, multi-device CPU, and downstream integration where relevant.
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
`test_pairwise_penalty_accepts_explicit_row_sharding`). The outer test sets
its own subprocess env (`XLA_FLAGS=--xla_force_host_platform_device_count=4`,
`SIMSOPT_JAX_SHARDING=…`); re-running the smoke file with the same flag at
the parent level just exposes the multi-device proxy assertion in the parent
process too:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
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
  `jax.jit(...).lower(...).compile().as_text()` as appropriate. Mark N/A if
  the item does not introduce a collective.

Required performance / memory evidence, proportional to item size:

- [ ] For hot-path kernels: a small `benchmarks/` micro-bench comparing
  warm-run time against the upstream baseline and the closest existing JAX
  kernel. Call `.block_until_ready()` before reading timings.
- [ ] For any new dense materialization: report bytes, compare against
  `max_dense_jacobian_bytes`, and keep the operator-backed alternative path.
- [ ] If buffer donation is used, prove it at a real outer-jit boundary. Do
  not claim donation benefit from internal `fori_loop` carries.
- [ ] If only CPU validation ran, report it as CPU validation only.

GPU reality:

- [ ] Do not claim real CUDA success without a current-SHA CUDA artifact from
  an approved GPU run.
- [ ] Multi-device CPU subprocesses and HLO inspection are useful regression
  proxies, but they are not GPU proof.
- [ ] Do not launch GPU jobs unless the user explicitly approves them.
- [ ] Tag each CPU-only item's state evidence with `cuda_smoke: not_claimed`
  or `cuda_smoke: deferred`, never `verified`.

### 4d. Commit

One scoped commit per completed active item, only after validation or a
documented blocker. Preserve unrelated dirty and untracked files.

Message format:

```text
jax-port: <short title> [item <id>]

- spec: <spec name or existing>
- kernel: jax_core/<file or existing>
- adapter: <file>
- parity lane: <lane>
- tests: <new or affected test paths>
- docs checked: <official docs URLs or Context7 IDs>
```

Do not add co-author footers unless the user explicitly asks for them.

## 5. Refusal triggers (stop the item, mark BLOCKED, move on)

- The port would require subclassing or replacing `Optimizable` or `Derivative`.
- The algorithm requires a dependency not already present in this repo.
- Parity vs the upstream C++/SciPy oracle fails at the lane tolerance after
  honest debugging.
- A new `transfer_guard=disallow` violation would require weakening the runtime
  contract.
- The change would require modifying `validation_ladder_contract.py`
  tolerances. Tolerances are user-owned policy, not agent-owned.
- The work requires a GPU run and the user has not approved launching one.

## 6. Anti-patterns - refuse unconditionally

- Silent CPU fallback inside a JAX target lane.
- `except Exception:` to hide a real error.
- Dynamic imports or `Any` casts.
- `jnp.asarray(host_array)` in a hot path.
- Dense materialization on production adjoint paths.
- Inlining tolerances.
- Declaring a CUDA result from CPU-only or HLO-only evidence.
- "It is fine, the test passes locally" without the applicable validation set.
- Skipping a `stellsym=True` case in a parity test.
- Removing existing CPU / C++ paths. The parity oracle stays.

## 7. Stop condition

Stop when either:

- Every item in `active_scope` is `complete`, `blocked`, or `skipped`, and
  applicable targeted, regression, transfer-guard, multi-device CPU, and
  downstream integration tests are green, and the final report exists; or
- No eligible active item remains because dependencies or user decisions are
  blocked.

For broad source changes, run the full test suite with the repo-local
interpreter before declaring full closure. If full-suite, cross-env, or GPU
validation is blocked by environment or hardware, state that explicitly and do
not claim full end-to-end closure.

Write `.artifacts/jax_port_goal/REPORT.md` with:

- Completed active items and validation commands.
- Blocked/skipped active items and exact blockers.
- CPU, multi-device CPU, transfer-guard, cross-env, and GPU evidence separated.
- Any stale rows discovered in `jax_gpu_port_todos_2026-04-08.md` or local docs.
- Any upstream/downstream API, math, physics, or computation discrepancy found.

## 8. Escalate immediately (do not loop in the dark)

- A blocker would require a user-facing API change.
- A skip-list item is actually critical to an active item.
- An open PARTIAL item from `jax_gpu_port_todos_2026-04-08.md` (#24, #26, #27,
  #29) becomes load-bearing.
- Repeated parity failure at the lane tolerance after two honest fix attempts.
  Write the diagnostic to `.artifacts/jax_port_goal/blockers/<id>.md` and move
  on.

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
