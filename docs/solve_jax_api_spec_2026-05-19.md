# `simsopt.solve.jax` API specification

**Status:** Draft v0.3. Bindings are not final; resolved decisions and corrected review findings are listed in §14. Sign off on each before any code lands against this spec.

**Owner:** simsopt-jax port maintainers.

**Cross-references:** `docs/parity_dual_mode_contract_2026-05-08.md`, `docs/source/jax_acceptance.rst`, `CLAUDE.md` (M1–M6 sections), `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md`.

**External contracts checked:** JAX `jax.Array` / `jax.typing.ArrayLike` and async-dispatch timing docs; JAX Python-version policy; SciPy 1.17 `minimize(method="L-BFGS-B")`, `minimize(method="BFGS")`, and `least_squares(method="lm")` docs; Optax `lbfgs` / `adam` docs; Optimistix `least_squares` / `LevenbergMarquardt` docs; Lineax solver docs; upstream SIMSOPT `simsopt.solve` package docs and current upstream source.

---

## 1. Purpose & scope

This document is the **public-API contract** for the JAX-lane optimizer surface, replacing today's `simsopt.geo.optimizer_jax.{jax_minimize, jax_least_squares}` entry points and their multi-string dispatch (`method=`, `optimizer_backend=`, `least_squares_algorithm=`).

**In scope:**
- The new public package `simsopt.solve.jax`.
- The `Driver` enum (SSOT for solver selection).
- The `minimize` and `least_squares` function signatures.
- Per-driver typed options dataclasses.
- The `OptimizerResult` schema.
- The value/grad callable contract.
- The old-string-API → new-API mapping and compat shim.
- Deprecation timeline and rollback plan.
- The additive import boundary under upstream `simsopt.solve`.

**Out of scope:**
- Validation results for the supersedability claims that motivate this design (`OPTAX_LBFGS` vs `SIMSOPT_LBFGSB`, `OPTAX_ADAM` vs `SIMSOPT_ADAM*`, `OPTIMISTIX_LM` vs `SIMSOPT_LM_*`). Those live in a separate validation artifact and gate the **removal** of bridge drivers, not the introduction of new ones.
- The caller-inventory grep artifact (§8 references it; published as `docs/solve_jax_api_caller_inventory_2026-05-19.md`).
- Per-PR sequencing of the migration. This spec is the *contract*; the PR plan is downstream.
- Constraint handling (bounds, equality, inequality). The contract is unconstrained.
- Raising the project-wide SIMSOPT Python floor. The new `simsopt.solve.jax` subpackage may require Python 3.11+, but `simsopt.solve.__init__` must not import it while upstream SIMSOPT remains `requires-python >=3.8`.

---

## 2. Current state summary

The current tree exposes 13 `jax_minimize` / `jax_least_squares` method strings, organized by three coupled string axes (`optimizer_backend × method × least_squares_algorithm`), plus one example-local Optax L-BFGS diagnostic lane. This produces the SOFTWARE_DESIGN.md red flags catalogued below; the spec exists to fix them.

| Today's identifier | Driver impl | Inner linalg | SOFTWARE_DESIGN red flag |
|---|---|---|---|
| `lbfgs` (backend=scipy) | SciPy Fortran `setulb` | host LAPACK | conjoined with `lbfgs-scipy-jax-*` |
| `lbfgs-trace` | in-tree numpy port | none | duplicates `scipy.optimize._linesearch` (~60% of 1611 lines) |
| `lbfgs-scipy-jax` (backend=scipy-jax) | SciPy Fortran + JAX v/g | host LAPACK | special-general mixture w/ fullgraph |
| `lbfgs-scipy-jax-fullgraph` | SciPy Fortran + fullgraph JAX v/g | host LAPACK | same driver as above, different callable |
| `lbfgs-ondevice` (backend=ondevice) | JAX port of L-BFGS-B | LAPACK CPU / cuSOLVER CUDA | conjoined with `lbfgs-trace` (shared status enums) |
| `bfgs` | SciPy dense BFGS | host LAPACK | active small-n path; kept as production driver per D9 |
| `bfgs-ondevice` | in-tree dense BFGS | LAPACK CPU / cuSOLVER CUDA | active small-n path; kept as production driver per D9 |
| `adam` | in-tree Adam | host-driven JAX loop | active old method; not losslessly replaced by Optax until validated |
| `adam-ondevice` | in-tree Adam | none | reimplements `optax.adam` |
| `lm` | in-tree host/reference JAX LM | matrix-free GMRES | **not SciPy MINPACK**; lossless shim must preserve this algorithm |
| `lm-ondevice` | in-tree trace-safe JAX LM | matrix-free GMRES | potentially superseded by Optimistix+LSMR (validation pending) |
| `lm-minpack-ondevice` | in-tree dense QR LM | LAPACK CPU / cuSOLVER CUDA | MINPACK-conditioned, but not MINPACK byte identity; removal needs QR/rank audit |
| `optimistix-lm-ondevice` | Optimistix LM + Lineax LSMR | matrix-free LSMR | optional target lane today; no callback support in current implementation |
| Optax `lbfgs` + zoom (examples/) | Optax | none | example-local diagnostic today; production promotion requires new wrapper tests |

The cumulative effect: ~6775 lines across `optimizer_jax.py` (4628) + `optimizer_host_lbfgs.py` (1611) + `optimizer_jax_reference.py` (536), with a public API that requires the caller to know which `(backend, method, least_squares_algorithm)` tuples are mutually compatible. Information leakage across files; internally-owned behavior config exposed as a registry.

`SCIPY_LM` in this spec is a new SciPy/MINPACK driver. It is **not** the lossless translation for today's `jax_least_squares(..., method="lm")`, which routes through the in-tree host/reference JAX LM loop.

---

## 3. Driver enum (SSOT)

`Driver` is a `StrEnum` (Python 3.11+) declared in `simsopt.solve.jax.contracts`. **It is the only solver-selection axis exposed to callers.**

```python
from enum import StrEnum

class Driver(StrEnum):
    # Production drivers (target end state)
    SCIPY_LBFGSB         = "scipy_lbfgsb"
    SCIPY_LM             = "scipy_lm"
    SCIPY_BFGS           = "scipy_bfgs"
    OPTAX_LBFGS          = "optax_lbfgs"
    OPTAX_ADAM           = "optax_adam"
    OPTIMISTIX_LM        = "optimistix_lm"
    SIMSOPT_LBFGSB       = "simsopt_lbfgsb"
    SIMSOPT_BFGS         = "simsopt_bfgs"
    SIMSOPT_TRACE_LBFGS  = "simsopt_trace_lbfgs"

    # Bridge values, present during migration; gated for removal pending validation.
    SIMSOPT_ADAM_HOST    = "simsopt_adam_host"     # lossless old method="adam"
    SIMSOPT_ADAM         = "simsopt_adam"          # lossless old method="adam-ondevice"
    SIMSOPT_LM_GMRES_HOST = "simsopt_lm_gmres_host" # lossless old method="lm"
    SIMSOPT_LM_GMRES     = "simsopt_lm_gmres"      # lossless old method="lm-ondevice"
    SIMSOPT_LM_QR        = "simsopt_lm_qr"         # lossless old method="lm-minpack-ondevice"
```

### Production drivers (target end state)

| Driver | Algorithm | Where it runs | Library | Best for | Trade-off |
|---|---|---|---|---|---|
| `SCIPY_LBFGSB` | L-BFGS-B (compact form, More-Thuente) | host CPU | SciPy (Fortran `setulb`) | reference oracle; any scalar minimize with cheap v/g | host↔device transfer per iter if v/g is JAX |
| `SCIPY_LM` | Levenberg-Marquardt (MINPACK `lmder`) | host CPU | SciPy | new SciPy/MINPACK LS oracle | host-bound; no GPU residency; not the old `method="lm"` shim target |
| `SCIPY_BFGS` | dense BFGS (Hessian approximation maintained in full) | host CPU | SciPy | small problems (n ≲ 200) where dense Hessian fits and converges faster than L-BFGS | O(n²) memory; impractical above ~1000 DOFs |
| `OPTAX_LBFGS` | plain L-BFGS + zoom line search | JAX device | Optax | on-device scalar minimize where compact-form not required | different convergence path than L-BFGS-B; no bounds |
| `OPTAX_ADAM` | Adam / AdamW first-order momentum | JAX device | Optax | stochastic / noisy objectives, high-DOF first-order regimes | no convergence proof; hyperparameter sensitive; old Adam methods stay on bridge drivers until parity is validated |
| `OPTIMISTIX_LM` | Levenberg-Marquardt (caller-chosen linear solver) | JAX device | Optimistix + Lineax | on-device LS; matrix-free or dense per `linear_solver` option | no callback in the initial API; linear-solver choice is externally-owned config (§5) |
| `SIMSOPT_LBFGSB` | L-BFGS-B (compact form, More-Thuente) | JAX device | in-tree | large DOF on GPU; byte-comparable to `SCIPY_LBFGSB` oracle | in-tree maintenance burden; no library equivalent for L-BFGS-B |
| `SIMSOPT_BFGS` | dense BFGS | JAX device | in-tree | small problems on GPU (e.g., Boozer inner solve, n ≈ 50–200) | O(n²) memory; only justified at small n |
| `SIMSOPT_TRACE_LBFGS` | plain L-BFGS w/ instrumentation | host CPU (numpy) | in-tree | **debug only** — rejected line-search samples, invalid-step events, per-iteration state trace | ~60% duplicate of `scipy.optimize._linesearch` |

### Bridge drivers (removal pending validation)

| Driver | Removal gate |
|---|---|
| `SIMSOPT_ADAM_HOST` | `OPTAX_ADAM` replacement is validated against old `method="adam"` on deterministic host-loop fixtures, including default hyperparameters and status behavior |
| `SIMSOPT_ADAM` | `OPTAX_ADAM` replacement is validated against old `method="adam-ondevice"` on target-lane fixtures |
| `SIMSOPT_LM_GMRES_HOST` | No automatic removal. This preserves today's host/reference JAX LM. `SCIPY_LM` is a different MINPACK algorithm, so replacement requires an explicit algorithm-change signoff and compatibility-test update |
| `SIMSOPT_LM_GMRES` | `OPTIMISTIX_LM` + `LinearSolver.LSMR` matches at `rtol=1e-10` on `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` and the production Boozer LS fixture |
| `SIMSOPT_LM_QR` | `OPTIMISTIX_LM` + `LinearSolver.QR` or `SVD` matches and a separate audit proves no caller depends on MINPACK-style QR info-code details; QR-only validation must include rank-deficient and ill-conditioned fixtures because Lineax QR is not rank-revealing |

---

## 4. Public API surface

Two top-level functions in `simsopt.solve.jax`:

```python
def minimize(
    value_and_grad_fn: ValueAndGradFn,
    x0: ArrayLike,
    *,
    driver: Driver,
    options: OptionsBase | None = None,
    callback: Callable[[OptimizerCallbackEvent], None] | None = None,
) -> OptimizerResult: ...

def least_squares(
    residual_fn: ResidualFn,
    x0: ArrayLike,
    *,
    driver: Driver,
    options: OptionsBase | None = None,
    callback: Callable[[OptimizerCallbackEvent], None] | None = None,
) -> OptimizerResult: ...
```

**Rules:**
- `driver` is **required**, keyword-only. **No silent default** (D4 default applied).
- `options` defaults to `None` → driver applies its built-in defaults (documented in §5).
- `options`'s concrete type must match the chosen `driver` (e.g., `Driver.OPTIMISTIX_LM` requires `OptimistixLMOptions | None`). Mismatch raises `TypeError` at the dispatch boundary.
- `callback` receives a typed `OptimizerCallbackEvent` (defined in `contracts`) at each accepted iteration for callback-capable drivers. Drivers without a real iteration-callback contract raise `ValueError` if `callback` is passed. Initial no-callback drivers: `SCIPY_LM` (SciPy documents least-squares callbacks only for `trf` and `dogbox`, not `lm`) and `OPTIMISTIX_LM` (current wrapper rejects callbacks).
- Both functions raise the chosen driver's library `ImportError` only if the required dependency isn't installed. Importing `simsopt.solve.jax` and `simsopt.solve.jax.contracts` must not import Optax, Optimistix, Lineax, or CUDA-specific packages.
- `least_squares` rejects scalar drivers (`SCIPY_LBFGSB`, `OPTAX_LBFGS`, etc.) and vice versa with `ValueError` at dispatch.
- JAX-backed drivers call `jax.block_until_ready()` on returned JAX leaves before measuring `wallclock_s` or converting final outputs to host arrays. Otherwise `wallclock_s` would only measure asynchronous dispatch on GPU/TPU.

---

## 5. Options dataclasses

> **D5 default applied:** per-library location. Each library subdir owns its options.

> **D2 default applied:** `LinearSolver` exposes `LSMR` + `QR` only initially. Add others when there's a research need.

`LinearSolver.QR` is a direct dense solver choice, not a MINPACK-pivoted-QR promise. Lineax documents QR as non-rank-revealing and full-rank only; rank-deficient or ill-conditioned replacement claims must use the §3 validation gate.

> **D3 default applied:** `OptaxLineSearch` exposes `ZOOM` only initially.

All options classes are frozen dataclasses (`@dataclass(frozen=True)`) inheriting from `OptionsBase` (sentinel base for type-narrowing dispatch).

### SciPy

```python
# simsopt/solve/jax/scipy/contracts.py

@dataclass(frozen=True)
class ScipyLBFGSBOptions(OptionsBase):
    maxiter: int = 15000
    maxfun: int = 15000
    gtol: float = 1e-10
    ftol: float = 1e-10
    maxcor: int = 200
    maxls: int = 20

@dataclass(frozen=True)
class ScipyLMOptions(OptionsBase):
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8

@dataclass(frozen=True)
class ScipyBFGSOptions(OptionsBase):
    maxiter: int = 1500
    gtol: float = 1e-10
    xrtol: float = 0.0          # SciPy BFGS step-size tolerance; 0 disables
    norm: float = float("inf")  # gradient norm order; matches SciPy default
```

### Optax

```python
# simsopt/solve/jax/optax/contracts.py

class OptaxLineSearch(StrEnum):
    ZOOM = "zoom"

@dataclass(frozen=True)
class OptaxLBFGSOptions(OptionsBase):
    maxiter: int = 15000
    gtol: float = 1e-10
    memory_size: int = 200
    line_search: OptaxLineSearch = OptaxLineSearch.ZOOM
    scale_init_precond: bool = True
    max_linesearch_steps: int = 30

@dataclass(frozen=True)
class OptaxAdamOptions(OptionsBase):
    maxiter: int = 10000
    learning_rate: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    gtol: float | None = None        # None ⇒ ignore (Adam is not gradient-tolerance based)
    weight_decay: float = 0.0         # > 0 selects AdamW
```

### Optimistix

```python
# simsopt/solve/jax/optimistix/contracts.py

class LinearSolver(StrEnum):
    LSMR = "lsmr"
    QR   = "qr"

@dataclass(frozen=True)
class OptimistixLMOptions(OptionsBase):
    maxiter: int = 1500
    tol: float = 1e-10                # used as rtol=atol for both LM and inner linear solve
    linear_solver: LinearSolver = LinearSolver.LSMR
    materialize_dense_linearization: bool = False
    max_dense_linearization_bytes: int | None = None    # None ⇒ BackendPolicy default
```

### simsopt (in-tree)

```python
# simsopt/solve/jax/simsopt/contracts.py

@dataclass(frozen=True)
class SimsoptLBFGSBOptions(OptionsBase):
    # mirrors ScipyLBFGSBOptions semantics; algorithm is the L-BFGS-B compact-form port
    maxiter: int = 15000
    maxfun: int = 15000
    gtol: float = 1e-10
    ftol: float = 1e-10
    maxcor: int = 200
    maxls: int = 20

@dataclass(frozen=True)
class SimsoptBFGSOptions(OptionsBase):
    # JAX on-device dense BFGS; only justified at small n (≲ 200)
    maxiter: int = 1500
    gtol: float = 1e-10
    xrtol: float = 0.0

@dataclass(frozen=True)
class SimsoptAdamHostOptions(OptionsBase):  # bridge
    maxiter: int = 10000
    learning_rate: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    gtol: float | None = None

@dataclass(frozen=True)
class SimsoptAdamOptions(SimsoptAdamHostOptions):  # bridge
    pass

@dataclass(frozen=True)
class SimsoptTraceLBFGSOptions(OptionsBase):
    maxiter: int = 15000
    gtol: float = 1e-10
    maxcor: int = 200
    ftol: float = 1e-10
    maxls: int = 20
    initial_step_size: float | None = None
    record_optimizer_state_trace: bool = True
    max_optimizer_state_trace_bytes: int | None = None   # None ⇒ 64 MiB default
    invalid_step_log_capacity: int = 256

@dataclass(frozen=True)
class SimsoptLMGMRESHostOptions(OptionsBase):   # bridge
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float | None = None

@dataclass(frozen=True)
class SimsoptLMGMRESOptions(SimsoptLMGMRESHostOptions):   # bridge
    materialize_dense_linearization: bool = True
    max_dense_linearization_bytes: int | None = None

@dataclass(frozen=True)
class SimsoptLMQROptions(OptionsBase):      # bridge
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float | None = None
    max_dense_linearization_bytes: int | None = None
```

**Why per-library:** §SOFTWARE_DESIGN "decompose by knowledge owned". Each library subdir owns one library's knob vocabulary; the options class lives next to the wrapper that consumes it.

---

## 6. `OptimizerResult` shape

> **D1 default applied:** define our own typed dataclass; document the SciPy-compatible field subset for callers that consume `scipy.optimize.OptimizeResult`.

```python
# simsopt/solve/jax/contracts.py

@dataclass(frozen=True)
class OptimizerResult:
    # --- SciPy-compatible subset (always present) ---
    x: np.ndarray                     # final iterate, host array
    fun: float                        # objective at x
    jac: np.ndarray | None            # gradient at x (None for derivative-free)
    nit: int                          # iterations
    nfev: int                         # function evaluations
    njev: int                         # gradient evaluations
    status: int                       # driver-native terminal code; see STATUS_CODES
    success: bool
    message: str

    # --- simsopt.solve.jax extensions (always present) ---
    driver: Driver                    # the driver used
    options_used: OptionsBase         # the resolved options dataclass (post-defaults)
    wallclock_s: float                # end-to-end runtime

    # --- Optional fields (driver-specific; None when not produced) ---
    residual: np.ndarray | None = None              # LS only
    residual_jacobian: np.ndarray | None = None     # LS only
    hessian: np.ndarray | None = None               # LS / dense BFGS

    # --- Trace driver only ---
    invalid_step_log: list[InvalidStepEvent] | None = None
    optimizer_state_trace: list[OptimizerStateTraceEntry] | None = None

    # --- Optimistix driver only ---
    optimistix_result: str | None = None            # raw RESULTS enum value as string
    optimistix_result_message: str | None = None
```

**Status code contract:**
- `success` is the only cross-driver success predicate.
- `status` is driver-native after a narrow translation layer. Do not infer success from the sign of `status`.
- The full mapping per driver lives in `contracts.STATUS_CODES` (a dict keyed by `Driver`). Examples that must be represented exactly: SciPy L-BFGS-B maxiter/maxfun termination has `success=False`; SciPy `least_squares` treats `status > 0` as convergence and `status == 0` as max-evaluation failure.

`OptimizerResult` is **not hashable**: it contains `np.ndarray` and list fields. Parity tests use an explicit helper instead:

```python
@dataclass(frozen=True, slots=True)
class OptimizerResultFingerprint:
    driver: Driver
    status: int
    success: bool
    x_shape: tuple[int, ...]
    x_dtype: str
    x_digest_blake2b: str

def fingerprint_optimizer_result(result: OptimizerResult) -> OptimizerResultFingerprint: ...
```

The helper includes shape and dtype as first-class fields so two different arrays cannot collide before byte hashing. It never stores `x.tobytes()` in the key object.

---

## 7. Callable contracts

Three typed callables flow through the public API: the value/grad evaluator (for `minimize`), the residual evaluator (for `least_squares`), and the optional per-iteration callback. All three are statically typed; the callback uses a discriminated dataclass union so each callback-capable driver's iteration metadata is reachable without `Any` or opaque dict escape hatches.

### 7.1 Value/grad evaluator (for `minimize`)

```python
ScalarResult = float | np.floating | jax.Array
ArrayResult = np.ndarray | jax.Array
OptimizerInput = np.ndarray | jax.Array
ValueAndGradFn = Callable[[OptimizerInput], tuple[ScalarResult, ArrayResult]]
```

The callable receives a host or device array (driver's choice) and returns `(scalar value, gradient array)`.

**Host/device casting rules:**
- Drivers in `solve/jax/scipy/` always pass a host `np.ndarray` and expect a host `np.ndarray` return.
- Drivers in `solve/jax/optax/`, `solve/jax/optimistix/`, and `solve/jax/simsopt/` (except `_trace_lbfgs`) pass a `jax.Array` and accept either `jax.Array` or `np.ndarray` returns; conversion to JAX-resident is performed inside the driver wrapper.
- The caller's callable does not need to handle casting itself if it routes through `jax.value_and_grad`; the driver wrapper does the boundary cast.

### 7.2 Residual evaluator (for `least_squares`)

```python
ResidualFn = Callable[[OptimizerInput], ArrayResult]
```

Returns a 1-D residual vector. The Jacobian is computed by the driver:
- `SCIPY_LM` — by finite difference (or by passing `jac=...` separately; outside this spec's contract).
- `OPTIMISTIX_LM`, `SIMSOPT_LM_*` — by `jax.jacrev` / `jax.jacfwd` per driver-internal policy.

### 7.3 Optimizer callback event (typed discriminated union)

> **D10 default applied:** discriminated dataclass union with `Literal[Driver.X]` tagging. No `extras: dict[str, Any]` escape hatch — each callback-capable driver's iteration metadata is a typed field on its event subclass.

The callback signature:

```python
Callback = Callable[[OptimizerCallbackEvent], None]
```

`OptimizerCallbackEvent` is a tagged union of per-driver event subclasses for **callback-capable** drivers. The base class enforces the cross-driver minimum field set; each subclass adds its own driver-specific typed fields. Static type checkers (pyright, mypy) narrow on the `driver: Literal[Driver.X]` discriminant.

The union intentionally omits `SCIPY_LM` and `OPTIMISTIX_LM`: their initial contracts reject callbacks rather than synthesizing misleading events after the fact.

```python
# simsopt/solve/jax/contracts.py

@dataclass(frozen=True, kw_only=True, slots=True)
class _OptimizerCallbackEventBase:
    iteration: int
    x: np.ndarray
    fun: float
    grad_norm_inf: float
    wallclock_s: float

@dataclass(frozen=True, kw_only=True, slots=True)
class ScipyLBFGSBCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SCIPY_LBFGSB] = Driver.SCIPY_LBFGSB
    # SciPy callback exposes only the base fields

@dataclass(frozen=True, kw_only=True, slots=True)
class ScipyBFGSCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SCIPY_BFGS] = Driver.SCIPY_BFGS

@dataclass(frozen=True, kw_only=True, slots=True)
class OptaxLBFGSCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.OPTAX_LBFGS] = Driver.OPTAX_LBFGS
    learning_rate: float
    num_linesearch_steps: int
    decrease_error: float
    curvature_error: float

@dataclass(frozen=True, kw_only=True, slots=True)
class OptaxAdamCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.OPTAX_ADAM] = Driver.OPTAX_ADAM
    learning_rate: float

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLBFGSBCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_LBFGSB] = Driver.SIMSOPT_LBFGSB
    accepted_alpha: float
    num_linesearch_steps: int

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptBFGSCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_BFGS] = Driver.SIMSOPT_BFGS
    accepted_alpha: float
    num_linesearch_steps: int

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptTraceLBFGSCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_TRACE_LBFGS] = Driver.SIMSOPT_TRACE_LBFGS
    accepted_alpha: float
    rejected_alphas: tuple[float, ...]
    line_search_status: LineSearchStatus            # see _shared/_status.py
    invalid_step_reason: InvalidStepReason | None   # see _shared/_status.py

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptAdamHostCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_ADAM_HOST] = Driver.SIMSOPT_ADAM_HOST
    learning_rate: float

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptAdamCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_ADAM] = Driver.SIMSOPT_ADAM
    learning_rate: float

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLMGMRESHostCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_LM_GMRES_HOST] = Driver.SIMSOPT_LM_GMRES_HOST
    residual_norm: float
    damping: float
    gmres_iterations: int

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLMGMRESCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_LM_GMRES] = Driver.SIMSOPT_LM_GMRES
    residual_norm: float
    damping: float
    gmres_iterations: int

@dataclass(frozen=True, kw_only=True, slots=True)
class SimsoptLMQRCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SIMSOPT_LM_QR] = Driver.SIMSOPT_LM_QR
    residual_norm: float
    damping: float
    minpack_info: int | None                        # MINPACK info code if computed

OptimizerCallbackEvent: TypeAlias = (
    ScipyLBFGSBCallbackEvent
    | ScipyBFGSCallbackEvent
    | OptaxLBFGSCallbackEvent
    | OptaxAdamCallbackEvent
    | SimsoptLBFGSBCallbackEvent
    | SimsoptBFGSCallbackEvent
    | SimsoptTraceLBFGSCallbackEvent
    | SimsoptAdamHostCallbackEvent
    | SimsoptAdamCallbackEvent
    | SimsoptLMGMRESHostCallbackEvent
    | SimsoptLMGMRESCallbackEvent
    | SimsoptLMQRCallbackEvent
)
```

**Caller pattern (with static narrowing):**

```python
def my_callback(event: OptimizerCallbackEvent) -> None:
    log(f"iter {event.iteration}: f={event.fun:.4e} |g|={event.grad_norm_inf:.2e}")
    if event.driver == Driver.SIMSOPT_TRACE_LBFGS:
        # pyright/mypy narrows: event is SimsoptTraceLBFGSCallbackEvent
        for alpha in event.rejected_alphas:
            record_rejected_alpha(alpha)
    elif event.driver == Driver.SIMSOPT_LM_GMRES:
        # narrows: event is SimsoptLMGMRESCallbackEvent
        record_damping(event.damping)
```

**Rules:**
- Each callback-capable driver emits exactly one concrete event subclass; the subclass is fixed per driver.
- Drivers that cannot surface extra mid-iteration data (`SCIPY_BFGS`) emit the base-field-only subclass; the subclass still exists (no `None` discriminant).
- Adding callback support for a new driver requires defining its concrete event subclass and adding it to the `OptimizerCallbackEvent` union. This is a Tier 3 API change (per the existing API evolution gate in §11).
- `LineSearchStatus` and `InvalidStepReason` are enums defined in `solve/jax/_shared/_status.py`; they are part of the public type surface.

---

## 8. Old API → new API mapping

Built against the live inventory in `src/simsopt/geo/optimizer_jax.py` (`_SUPPORTED_METHODS`, `_SUPPORTED_LEAST_SQUARES_METHODS`, and optimizer-backend sets). The full caller inventory — every external `file:line` for every old method string, plus hot-spot ranking and recommended PR sequencing — lives in [`docs/solve_jax_api_caller_inventory_2026-05-19.md`](./solve_jax_api_caller_inventory_2026-05-19.md). This table is the migration cookbook; the inventory is the migration target list.

Rows that mention `optimizer_backend` describe wrapper-resolved selector tuples from `resolve_optimizer_backend_method(...)`, not direct `jax_minimize(...)` kwargs. Current direct `jax_minimize(...)` and `jax_least_squares(...)` call sites select lanes with `method=` only.

| Old selector | New call | Migration risk |
|---|---|---|
| `method="lbfgs"` (`optimizer_backend="scipy"`, `limited_memory=True`) | `minimize(fn, x0, driver=Driver.SCIPY_LBFGSB)` | trivial |
| `method="lbfgs-trace"` | `minimize(fn, x0, driver=Driver.SIMSOPT_TRACE_LBFGS, options=SimsoptTraceLBFGSOptions(record_optimizer_state_trace=True))` | low |
| `method="lbfgs-scipy-jax"` (`optimizer_backend="scipy-jax"`, `limited_memory=True`) | `minimize(per_term_jax_value_and_grad, x0, driver=Driver.SCIPY_LBFGSB)` | low; caller already has a JAX v/g |
| `method="lbfgs-scipy-jax-fullgraph"` (`optimizer_backend="scipy-jax-fullgraph"`, `limited_memory=True`) | `minimize(fullgraph_jax_value_and_grad, x0, driver=Driver.SCIPY_LBFGSB)` | low; fullgraph builder is the caller's responsibility |
| `method="lbfgs-ondevice"` (`optimizer_backend="ondevice"`, `limited_memory=True`) | `minimize(fn, x0, driver=Driver.SIMSOPT_LBFGSB)` | trivial |
| `method="bfgs"` (`optimizer_backend="scipy"`, `limited_memory=False`) | `minimize(fn, x0, driver=Driver.SCIPY_BFGS)` | trivial; no algorithm change |
| `method="bfgs-ondevice"` (`optimizer_backend="ondevice"`, `limited_memory=False`) | `minimize(fn, x0, driver=Driver.SIMSOPT_BFGS)` | trivial; no algorithm change |
| `method="adam"` | bridge: `minimize(fn, x0, driver=Driver.SIMSOPT_ADAM_HOST)`; target: `Driver.OPTAX_ADAM` | medium; Optax replacement parity validation required |
| `method="adam-ondevice"` | bridge: `minimize(fn, x0, driver=Driver.SIMSOPT_ADAM)`; target: `Driver.OPTAX_ADAM` | medium; Optax replacement parity validation required |
| `method="lm"` | `least_squares(r, x0, driver=Driver.SIMSOPT_LM_GMRES_HOST)` | trivial; preserves current host/reference JAX LM |
| `method="lm-ondevice"` | bridge: `least_squares(r, x0, driver=Driver.SIMSOPT_LM_GMRES)`; target: `Driver.OPTIMISTIX_LM` + `LinearSolver.LSMR` | medium; algorithm change pending validation |
| `method="lm-minpack-ondevice"` | bridge: `Driver.SIMSOPT_LM_QR`; target: `Driver.OPTIMISTIX_LM` + `LinearSolver.QR` | medium; MINPACK info-code audit required |
| `method="optimistix-lm-ondevice"` | `least_squares(r, x0, driver=Driver.OPTIMISTIX_LM, options=OptimistixLMOptions(linear_solver=LinearSolver.LSMR))` | low (already opt-in today) |

`Driver.SCIPY_LM` is an additional new API driver for SciPy/MINPACK `least_squares(method="lm")`. It is intentionally absent from the old-API compatibility table because it is not a lossless translation of today's `method="lm"`.

**`BoozerSurfaceJAX` two-axis case** (outer driver + inner solver):
- Today: `BoozerSurfaceJAX` LS options accept `optimizer_backend in {"auto", "scipy", "ondevice"}` and `least_squares_algorithm in {"quasi-newton", "lm", "lm-minpack", "optimistix-lm"}`. The `scipy-jax-fullgraph` string is an outer single-stage optimizer backend, not a valid `BoozerSurfaceJAX` constructor option.
- New: the outer single-stage solve selects `outer_driver=Driver.SCIPY_LBFGSB` with a caller-owned fullgraph value/grad factory; the inner Boozer LS solve selects `inner_driver=Driver.SIMSOPT_LM_GMRES_HOST`, `Driver.SIMSOPT_LM_GMRES`, `Driver.SIMSOPT_LM_QR`, or `Driver.OPTIMISTIX_LM` according to the previous `optimizer_backend` / `least_squares_algorithm` pair.
- The fullgraph factory is invoked once at outer solve time; the resulting callable is handed to `minimize`. Inner Boozer LS construction does not own that factory.

---

## 9. Compat shim contract

> **D6 default applied:** both `DeprecationWarning` and structured log.

```python
# simsopt/geo/optimizer_jax.py (existing module, kept as the shim)

def jax_minimize(fn, x0, *, method="bfgs", **kwargs):
    warnings.warn(
        "simsopt.geo.optimizer_jax.jax_minimize is deprecated; "
        "use simsopt.solve.jax.minimize(fn, x0, driver=...) instead. "
        f"Translation: method={method!r} → driver={driver}.",
        DeprecationWarning,
        stacklevel=2,
    )
    _DEPRECATION_LOGGER.info(
        "deprecated_solve_jax_call",
        extra={
            "old_method": method,
            "translated_driver": driver,
            "stack": _shim_caller_stack(stacklevel=2),
        },
    )
    return simsopt.solve.jax.minimize(fn, x0, driver=driver, options=options, **rest)
```

**Shim guarantees:**
- One `DeprecationWarning` per (process, call-site stack frame). Suppressed thereafter using a module-global `set[DeprecationCallSite]` protected by a lock; stack hashes are values, not weak-reference targets.
- Structured log entry per call (no suppression). Logger name: `simsopt.solve.jax.deprecation`. Goes to autoresearch run logs.
- Translation is **lossless** for combinations explicitly listed in §8. Combinations that do not map cleanly raise `ValueError` at the shim boundary. The shim does not dispatch to a closest replacement; algorithm changes belong in explicit migration PRs and validation artifacts.

---

## 10. Migration path

Categorized by caller class. Concrete file/line lists, hot-spot ranking, and the 15-PR sequence live in [`docs/solve_jax_api_caller_inventory_2026-05-19.md`](./solve_jax_api_caller_inventory_2026-05-19.md) (§§2, 8, 10). Headline scale from that audit: **~129 external call sites** across 13 old method strings; **fewer than 10** in production `src/`; the rest are tests, benchmarks, and one large example.

### Category A: Direct callers of `jax_minimize` / `jax_least_squares`

- **Production tests** (`tests/geo/test_boozersurface_jax.py`, `tests/integration/test_single_stage_jax*.py`): mechanical rewrite per the §8 table. One PR per test file.
- **`benchmarks/`**: same.
- **`examples/`**: same, but `single_stage_banana_example.py` has the largest cluster of method-string call sites. Single dedicated PR.

### Category B: solver-owning wrapper constructor args

- `BoozerSurfaceJAX` accepts solver policy through its LS options: `optimizer_backend=` and `least_squares_algorithm=`.
- `QfmSurfaceJAX` and `BiotSavartJAX` do not currently expose these constructor kwargs; the [caller inventory](./solve_jax_api_caller_inventory_2026-05-19.md) §6 confirmed no solver-string plumbing in their call paths. They are not part of this migration.
- Migration: add `outer_driver` and `inner_driver` kwargs to `BoozerSurfaceJAX`'s solver-option surface; deprecate the old string kwargs.
- Tier 3 (changes `BoozerSurfaceJAX` public API).

### Category C: autoresearch run configs and external scripts

- Out of repo. Migration via release notes + the structured log telemetry pointing affected runs at the new API.

---

## 11. Compatibility tests

These are the **pinned tests** that must keep passing through every step of the migration. They are the §SOFTWARE_DESIGN §API evolution gate "compatibility tests" artifact for this change.

> **D8 default applied:** new tests under `tests/solve/jax/`.

**New tests** (`tests/solve/jax/`):
- `test_driver_dispatch.py` — every `Driver` value → executes the documented dispatch path; trivial smoke per driver.
- `test_options_typing.py` — `options` type mismatch → `TypeError` at dispatch.
- `test_value_grad_contract.py` — exercises the host/device casting rules in §7.
- `test_optimizer_result_schema.py` — checks all fields present per-driver; checks `STATUS_CODES` completeness.
- `test_compat_shim_translation.py` — every lossless bridge row in §8 → old call produces same `(x, fun, status, success)` as new call within `rtol=0`.
- `test_algorithm_change_gates.py` — target replacements that are not lossless today (`OPTAX_ADAM`, `OPTIMISTIX_LM`, `SCIPY_LM` as a replacement for old `method="lm"`) stay behind explicit validation gates and are not used by the shim.
- `test_deprecation_warnings.py` — old API raises one `DeprecationWarning` per stack frame; structured log entry per call.

**Inherited tests** (kept; must not regress):
- `tests/geo/test_boozersurface_jax.py` — all `method=`/`optimizer_backend=` paths must work via the shim.
- `tests/integration/test_single_stage_jax*.py` — same.
- `tests/test_run_code_benchmark_common.py` — same.
- The strict-CPU/JAX byte-parity gate in `benchmarks/single_stage_init_parity.py::_pre_newton_census_gate_failures` — unchanged.

---

## 12. Deprecation timeline

> **D7 default applied:** two minor releases of shim, removal in the third.

Let `N` denote the release that lands this spec's API and the compat shim.

| Release | Status of old string API | Status of new `solve.jax` API |
|---|---|---|
| `N` | shim active; `DeprecationWarning` + structured log on every call | available, fully tested; documented as preferred |
| `N+1` | shim still active; same warning behavior; release notes call out coming removal | preferred; migration of category-A callers complete in this release |
| `N+2` | **removed**; old call paths raise `ImportError` with a clear message pointing at the new API | only API |

If validation of the supersedability claims (§3 bridge drivers) is not complete by `N+2`, the affected bridge drivers stay (`SIMSOPT_ADAM_HOST`, `SIMSOPT_ADAM`, `SIMSOPT_LM_GMRES_HOST`, `SIMSOPT_LM_GMRES`, `SIMSOPT_LM_QR`); the bridge gate is independent of the string-API removal timeline.

---

## 13. Rollback plan

If the shim or new API ships and breaks a known caller class:

1. **Within `N`**: revert is one PR (the new package is purely additive at this stage; the shim is a single module). Existing `optimizer_jax.py` functionality unchanged.
2. **In `N+1`**: the migration PRs (category A above) become individually reversible — each PR is a `Driver.*` substitution that re-targets the same algorithm via the shim. Revert any one without breaking the others.
3. **In `N+2`**: removal of the old string API is the only irreversible step. Mitigation: do not land removal until **zero** structured-log entries for the deprecated call paths have been observed across the active autoresearch run set for at least one release cycle. The structured log is the kill-switch.

---

## 14. Resolved decisions

| # | Decision | Default applied (subject to override) | Status |
|---|---|---|---|
| D1 | `OptimizerResult` shape | Own typed dataclass with SciPy-compatible field subset | applied |
| D2 | Lineax solvers exposed in `LinearSolver` enum | `LSMR` + `QR` only initially | applied |
| D3 | Optax line searches exposed in `OptaxLineSearch` enum | `ZOOM` only initially | applied |
| D4 | Default driver when caller omits the kwarg | Raise `ValueError`; no silent default | applied |
| D5 | Options dataclasses location | Per-library subdir | applied |
| D6 | Compat shim signalling | Both `DeprecationWarning` and structured log | applied |
| D7 | Deprecation length | Two minor releases of shim, removal in the third | applied |
| D8 | Test location | `tests/solve/jax/` mirroring source layout | applied |
| D9 | Dense BFGS (`method="bfgs"`, `method="bfgs-ondevice"`) | **Keep** as `SCIPY_BFGS` and `SIMSOPT_BFGS` production drivers. Caller audit confirmed active use at small n (Boozer inner solve: `boozersurface_jax.py:5791, 5847`; 6 explicit `method="bfgs"` sites in `tests/geo/test_boozersurface_jax.py`; benchmark expectations in `tests/test_benchmark_helpers.py:5372-5421`). | applied |
| D10 | `OptimizerCallbackEvent` schema | Typed discriminated dataclass union with `Literal[Driver.X]` tagging. One concrete event subclass per callback-capable driver; cross-driver minimum fields on `_OptimizerCallbackEventBase` (`iteration`, `x`, `fun`, `grad_norm_inf`, `wallclock_s`). No `extras: dict[str, Any]` escape hatch. See §7.3. | applied |
| D11 | Where the fullgraph value/grad builders live | `simsopt.objectives.jax.*` (separate from `solve.jax`) | applied |
| D12 | Python floor for `solve.jax` | The new subpackage is Python 3.11+ at runtime, but this spec does not raise upstream SIMSOPT's project-wide `requires-python >=3.8`. `simsopt.solve.__init__` must not import `simsopt.solve.jax`; importing the subpackage directly on Python <3.11 is unsupported. Native `StrEnum` stays inside the subpackage boundary. | applied |
| D13 | Lossless old `method="lm"` mapping | Preserve the current in-tree host/reference JAX LM as `SIMSOPT_LM_GMRES_HOST`. Do not map it to `SCIPY_LM`; SciPy MINPACK LM is a new driver with different algorithm and callback/status behavior. | applied |
| D14 | Lossless old Adam mapping | Preserve old `method="adam"` / `"adam-ondevice"` through `SIMSOPT_ADAM_HOST` / `SIMSOPT_ADAM` bridge drivers until Optax parity is validated. | applied |
| D15 | Result fingerprinting | `OptimizerResult` is not hashable because it owns ndarray/list fields. Parity tests use `fingerprint_optimizer_result()` returning a small typed digest object. | applied |

---

## 15. Non-goals

This document does **not** promise:
- That `OPTAX_LBFGS` converges identically to `SIMSOPT_LBFGSB`. The algorithms differ (plain L-BFGS vs L-BFGS-B); parity is a validation question.
- That `OPTIMISTIX_LM` + `LSMR` matches the existing in-tree matrix-free GMRES LM. Both are matrix-free Krylov but the iteration shape and termination differ.
- That `SCIPY_LM` is a replacement for old `method="lm"`. Today's old `method="lm"` is the in-tree JAX LM loop; SciPy/MINPACK LM is a separate new oracle driver.
- That `OPTAX_ADAM` byte-matches the in-tree Adam. Default hyperparameters can subtly differ.
- That bridge drivers will be removed. `SIMSOPT_ADAM_HOST`, `SIMSOPT_ADAM`, `SIMSOPT_LM_GMRES_HOST`, `SIMSOPT_LM_GMRES`, and `SIMSOPT_LM_QR` remain until the validation gates in §3 are satisfied.
- That `lbfgs-trace` callers can be silently migrated to SciPy + a callback. The rejected-line-search-sample data SciPy can't surface is the entire justification for the lane (see prior conversation thread for the detailed argument).
- A parity contract between the old and new APIs beyond the §11 compatibility-test set. Anything outside that set is best-effort.

---

## Appendix A: file layout this spec assumes

```
src/simsopt/solve/
├── serial.py                       # upstream
├── mpi.py                          # upstream
├── permanent_magnet_optimization.py
├── wireframe_optimization.py
├── __init__.py                     # must not import .jax while upstream supports Python 3.8
└── jax/
    ├── __init__.py                 # exports: minimize, least_squares, Driver, OptimizerResult
    ├── contracts.py                # Driver, OptimizerResult, OptionsBase, OptimizerCallbackEvent, STATUS_CODES
    ├── _dispatch.py                # Driver → impl table (SSOT for routing)
    │
    ├── scipy/
    │   ├── __init__.py
    │   ├── contracts.py            # ScipyLBFGSBOptions, ScipyLMOptions, ScipyBFGSOptions
    │   ├── _lbfgsb.py              # SCIPY_LBFGSB impl
    │   ├── _lm.py                  # SCIPY_LM impl
    │   └── _bfgs.py                # SCIPY_BFGS impl
    │
    ├── optax/
    │   ├── __init__.py
    │   ├── contracts.py            # OptaxLBFGSOptions, OptaxAdamOptions, OptaxLineSearch
    │   ├── _lbfgs.py               # OPTAX_LBFGS impl
    │   └── _adam.py                # OPTAX_ADAM impl
    │
    ├── optimistix/
    │   ├── __init__.py
    │   ├── contracts.py            # OptimistixLMOptions, LinearSolver
    │   └── _lm.py                  # OPTIMISTIX_LM impl
    │
    ├── simsopt/
    │   ├── __init__.py
    │   ├── contracts.py            # SimsoptLBFGSBOptions, SimsoptBFGSOptions, SimsoptTraceLBFGSOptions, bridge options
    │   ├── _lbfgsb.py              # SIMSOPT_LBFGSB impl (from optimizer_jax_private/_lbfgsb_scipy.py)
    │   ├── _bfgs.py                # SIMSOPT_BFGS impl (from optimizer_jax_private/_bfgs.py)
    │   ├── _trace_lbfgs.py         # SIMSOPT_TRACE_LBFGS impl (from optimizer_host_lbfgs.py)
    │   ├── _adam_host.py           # SIMSOPT_ADAM_HOST bridge
    │   ├── _adam.py                # SIMSOPT_ADAM bridge
    │   ├── _lm_gmres_host.py       # SIMSOPT_LM_GMRES_HOST bridge
    │   ├── _lm_gmres.py            # SIMSOPT_LM_GMRES bridge
    │   └── _lm_qr.py               # SIMSOPT_LM_QR bridge
    │
    └── _shared/
        ├── _adapters.py            # value/grad ↔ scipy callback casting
        ├── _status.py              # cross-driver status code utilities
        └── _state_trace.py         # InvalidStepEvent, OptimizerStateTraceEntry, serialization
```

Old file (`src/simsopt/geo/optimizer_jax.py`) retained as the compat shim entry point until the §12 removal.

---

## Appendix B: changelog

- 2026-05-19 v0 — initial draft.
- 2026-05-19 v0.1 — D9, D10, D12 resolved.
  - **D9**: caller audit confirmed active production use of dense BFGS; promoted `SCIPY_BFGS` and `SIMSOPT_BFGS` from deprecation candidates to first-class drivers. Production set: 9 drivers (was 7).
  - **D10**: callback contract changed from informal "minimal + extras dict" to typed discriminated dataclass union (`OptimizerCallbackEvent`) with `Literal[Driver.X]` tagging. One concrete event subclass per driver. See §7.3.
  - **D12**: Python floor pinned at 3.11+ for the whole `solve.jax` package. Superseded by the v0.2 upstream-packaging correction below.
- 2026-05-19 v0.2 — review fixes after live-tree, upstream, and official-doc validation.
  - Corrected old `method="lm"` mapping: it is the in-tree host/reference JAX LM, not SciPy MINPACK.
  - Added bridge drivers for old Adam and in-tree LM lanes so the compat shim remains lossless.
  - Fixed callback support: SciPy LM and Optimistix LM reject callbacks in the initial API.
  - Fixed `OptimizerResult` status/fingerprint contracts and required JAX result blocking before timing.
  - Corrected Python-floor wording so upstream SIMSOPT's project-wide `>=3.8` install contract is not silently broken.
- 2026-05-19 v0.3 — cross-referenced the published caller inventory at `docs/solve_jax_api_caller_inventory_2026-05-19.md` from §1, §8, §10, and §10's wrapper note. Confirmed QfmSurfaceJAX/BiotSavartJAX exclusion from the inventory.
