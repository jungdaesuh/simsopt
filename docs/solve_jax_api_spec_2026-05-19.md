# `simsopt.solve.jax` API specification

**Status:** Draft v0. Bindings are not final; open decisions are listed in §14. Sign off on each before any code lands against this spec.

**Owner:** simsopt-jax port maintainers.

**Cross-references:** `docs/parity_dual_mode_contract_2026-05-08.md`, `docs/source/jax_acceptance.rst`, `CLAUDE.md` (M1–M6 sections), `/Users/suhjungdae/.agent-docs/SOFTWARE_DESIGN.md`.

---

## 1. Purpose & scope

This document is the **public-API contract** for the JAX-lane optimizer surface, replacing the today's `simsopt.geo.optimizer_jax.{jax_minimize, jax_least_squares}` entry points and their multi-string dispatch (`method=`, `optimizer_backend=`, `least_squares_algorithm=`).

**In scope:**
- The new public package `simsopt.solve.jax`.
- The `Driver` enum (SSOT for solver selection).
- The `minimize` and `least_squares` function signatures.
- Per-driver typed options dataclasses.
- The `OptimizerResult` schema.
- The value/grad callable contract.
- The old-string-API → new-API mapping and compat shim.
- Deprecation timeline and rollback plan.

**Out of scope:**
- Validation results for the supersedability claims that motivate this design (`OPTAX_LBFGS` vs `SIMSOPT_LBFGSB`, `OPTIMISTIX_LM` vs `SIMSOPT_LM_*`). Those live in a separate validation artifact and gate the **removal** of bridge drivers, not the introduction of new ones.
- The caller-inventory grep artifact (§8 references it).
- Per-PR sequencing of the migration. This spec is the *contract*; the PR plan is downstream.
- Constraint handling (bounds, equality, inequality). The contract is unconstrained.

---

## 2. Current state summary

13 active solver lanes today, organized by three coupled string axes (`optimizer_backend × method × least_squares_algorithm`). This produces the SOFTWARE_DESIGN.md red flags catalogued below; the spec exists to fix them.

| Today's identifier | Driver impl | Inner linalg | SOFTWARE_DESIGN red flag |
|---|---|---|---|
| `lbfgs` (backend=scipy) | SciPy Fortran `setulb` | host LAPACK | conjoined with `lbfgs-scipy-jax-*` |
| `lbfgs-trace` | in-tree numpy port | none | duplicates `scipy.optimize._linesearch` (~60% of 1611 lines) |
| `lbfgs-scipy-jax` (backend=scipy-jax) | SciPy Fortran + JAX v/g | host LAPACK | special-general mixture w/ fullgraph |
| `lbfgs-scipy-jax-fullgraph` | SciPy Fortran + fullgraph JAX v/g | host LAPACK | same driver as above, different callable |
| `lbfgs-ondevice` (backend=ondevice) | JAX port of L-BFGS-B | LAPACK CPU / cuSOLVER CUDA | conjoined with `lbfgs-trace` (shared status enums) |
| `bfgs` | SciPy dense BFGS | host LAPACK | small caller base, deprecation candidate |
| `bfgs-ondevice` | in-tree dense BFGS | LAPACK CPU / cuSOLVER CUDA | same |
| `adam-ondevice` | in-tree Adam | none | reimplements `optax.adam` |
| `lm` | scipy MINPACK | host LAPACK | reference oracle, keeps |
| `lm-ondevice` | in-tree matrix-free GMRES LM | matrix-free Krylov | potentially superseded by Optimistix+LSMR (validation pending) |
| `lm-minpack-ondevice` | in-tree dense pivoted-QR LM | LAPACK CPU / cuSOLVER CUDA | potentially superseded by Optimistix+QR (validation pending) |
| `optimistix-lm-ondevice` | Optimistix LM + Lineax LSMR | matrix-free LSMR | doubly-opt-in experimental today; promoted to production by this spec |
| Optax `lbfgs` + zoom (examples/) | Optax | none | diagnostic-only today; promoted to production by this spec |

The cumulative effect: ~6775 lines across `optimizer_jax.py` (4628) + `optimizer_host_lbfgs.py` (1611) + `optimizer_jax_reference.py` (536), with a public API that requires the caller to know which `(backend, method, least_squares_algorithm)` tuples are mutually compatible. Information leakage across files; internally-owned behavior config exposed as a registry.

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
    SIMSOPT_LM_GMRES     = "simsopt_lm_gmres"      # removal pending OPTIMISTIX_LM+LSMR parity
    SIMSOPT_LM_QR        = "simsopt_lm_qr"         # removal pending OPTIMISTIX_LM+QR parity + MINPACK info-code audit
```

### Production drivers (target end state)

| Driver | Algorithm | Where it runs | Library | Best for | Trade-off |
|---|---|---|---|---|---|
| `SCIPY_LBFGSB` | L-BFGS-B (compact form, More-Thuente) | host CPU | SciPy (Fortran `setulb`) | reference oracle; any scalar minimize with cheap v/g | host↔device transfer per iter if v/g is JAX |
| `SCIPY_LM` | Levenberg-Marquardt (MINPACK `lmder`) | host CPU | SciPy | reference oracle for LS; MINPACK info codes 1–8 | host-bound; no GPU residency |
| `SCIPY_BFGS` | dense BFGS (Hessian approximation maintained in full) | host CPU | SciPy | small problems (n ≲ 200) where dense Hessian fits and converges faster than L-BFGS | O(n²) memory; impractical above ~1000 DOFs |
| `OPTAX_LBFGS` | plain L-BFGS + zoom line search | JAX device | Optax | on-device scalar minimize where compact-form not required | different convergence path than L-BFGS-B; no bounds |
| `OPTAX_ADAM` | Adam / AdamW first-order momentum | JAX device | Optax | stochastic / noisy objectives, high-DOF first-order regimes | no convergence proof; hyperparameter sensitive |
| `OPTIMISTIX_LM` | Levenberg-Marquardt (caller-chosen linear solver) | JAX device | Optimistix + Lineax | on-device LS; matrix-free or dense per `linear_solver` option | linear-solver choice is externally-owned config (§5) |
| `SIMSOPT_LBFGSB` | L-BFGS-B (compact form, More-Thuente) | JAX device | in-tree | large DOF on GPU; byte-comparable to `SCIPY_LBFGSB` oracle | in-tree maintenance burden; no library equivalent for L-BFGS-B |
| `SIMSOPT_BFGS` | dense BFGS | JAX device | in-tree | small problems on GPU (e.g., Boozer inner solve, n ≈ 50–200) | O(n²) memory; only justified at small n |
| `SIMSOPT_TRACE_LBFGS` | plain L-BFGS w/ instrumentation | host CPU (numpy) | in-tree | **debug only** — rejected line-search samples, invalid-step events, per-iteration state trace | ~60% duplicate of `scipy.optimize._linesearch` |

### Bridge drivers (removal pending validation)

| Driver | Removal gate |
|---|---|
| `SIMSOPT_LM_GMRES` | `OPTIMISTIX_LM` + `LinearSolver.LSMR` matches at `rtol=1e-10` on `build_ls_parity_problem(ncoils=4, nphi=16, ntheta=8)` |
| `SIMSOPT_LM_QR` | `OPTIMISTIX_LM` + `LinearSolver.QR` matches and (separate audit) no caller depends on MINPACK info codes 4/8 |

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
- `callback` receives a typed `OptimizerCallbackEvent` (defined in `contracts`) at each accepted iteration. Drivers that can't surface a callback raise `ValueError` if one is passed.
- Both functions raise the chosen driver's library `ImportError` only if the required dependency isn't installed — never at module import time of `simsopt.solve.jax`.
- `least_squares` rejects scalar drivers (`SCIPY_LBFGSB`, `OPTAX_LBFGS`, etc.) and vice versa with `ValueError` at dispatch.

---

## 5. Options dataclasses

> **D5 default applied:** per-library location. Each library subdir owns its options.

> **D2 default applied:** `LinearSolver` exposes `LSMR` + `QR` only initially. Add others when there's a research need.

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
    gtol: float | None = None

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
    materialize_dense_linearization: bool = True
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
class SimsoptLMGMRESOptions(OptionsBase):   # bridge
    maxiter: int = 1500
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float | None = None

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
    status: int                       # 0=success; negative=internal; positive=library-specific
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
- `0` — success per the driver's convergence criteria.
- `> 0` — library-specific success-but-noteworthy (e.g., max iterations reached, line search exhausted).
- `< 0` — internal failure (NaN gradient, line search failed, etc.).
- The full mapping per driver lives in `contracts.STATUS_CODES` (a dict keyed by `Driver`).

`OptimizerResult` is hashable on `x.tobytes() + driver + status` for use as a parity-test fingerprint key.

---

## 7. Callable contracts

Three typed callables flow through the public API: the value/grad evaluator (for `minimize`), the residual evaluator (for `least_squares`), and the optional per-iteration callback. All three are statically typed; the callback uses a discriminated dataclass union so each driver's iteration metadata is reachable without `Any` or opaque dict escape hatches.

### 7.1 Value/grad evaluator (for `minimize`)

```python
ValueAndGradFn = Callable[[ArrayLike], tuple[float, np.ndarray]]
```

The callable receives a host or device array (driver's choice) and returns `(scalar value, gradient array)`.

**Host/device casting rules:**
- Drivers in `solve/jax/scipy/` always pass a host `np.ndarray` and expect a host `np.ndarray` return.
- Drivers in `solve/jax/optax/`, `solve/jax/optimistix/`, and `solve/jax/simsopt/` (except `_trace_lbfgs`) pass a `jax.Array` and accept either `jax.Array` or `np.ndarray` returns; conversion to JAX-resident is performed inside the driver wrapper.
- The caller's callable does not need to handle casting itself if it routes through `jax.value_and_grad`; the driver wrapper does the boundary cast.

### 7.2 Residual evaluator (for `least_squares`)

```python
ResidualFn = Callable[[ArrayLike], np.ndarray]
```

Returns a 1-D residual vector. The Jacobian is computed by the driver:
- `SCIPY_LM` — by finite difference (or by passing `jac=...` separately; outside this spec's contract).
- `OPTIMISTIX_LM`, `SIMSOPT_LM_*` — by `jax.jacrev` / `jax.jacfwd` per driver-internal policy.

### 7.3 Optimizer callback event (typed discriminated union)

> **D10 default applied:** discriminated dataclass union with `Literal[Driver.X]` tagging. No `extras: dict[str, Any]` escape hatch — each driver's iteration metadata is a typed field on its event subclass.

The callback signature:

```python
Callback = Callable[[OptimizerCallbackEvent], None]
```

`OptimizerCallbackEvent` is a tagged union of per-driver event subclasses. The base class enforces the cross-driver minimum field set; each subclass adds its own driver-specific typed fields. Static type checkers (pyright, mypy) narrow on the `driver: Literal[Driver.X]` discriminant.

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
class ScipyLMCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.SCIPY_LM] = Driver.SCIPY_LM
    residual_norm: float

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
class OptimistixLMCallbackEvent(_OptimizerCallbackEventBase):
    driver: Literal[Driver.OPTIMISTIX_LM] = Driver.OPTIMISTIX_LM
    damping: float
    inner_lineax_iterations: int
    optimistix_result_intermediate: str | None      # raw RESULTS enum string mid-iteration

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
    | ScipyLMCallbackEvent
    | ScipyBFGSCallbackEvent
    | OptaxLBFGSCallbackEvent
    | OptaxAdamCallbackEvent
    | OptimistixLMCallbackEvent
    | SimsoptLBFGSBCallbackEvent
    | SimsoptBFGSCallbackEvent
    | SimsoptTraceLBFGSCallbackEvent
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
    elif event.driver == Driver.OPTIMISTIX_LM:
        # narrows: event is OptimistixLMCallbackEvent
        record_damping(event.damping)
```

**Rules:**
- Each driver emits exactly one concrete event subclass; the subclass is fixed per driver.
- Drivers that cannot surface mid-iteration data (`SCIPY_BFGS`) emit the base-field-only subclass; the subclass still exists (no `None` discriminant).
- Adding a new driver requires defining its concrete event subclass and adding it to the `OptimizerCallbackEvent` union. This is a Tier 3 API change (per the existing API evolution gate in §11).
- `LineSearchStatus` and `InvalidStepReason` are enums defined in `solve/jax/_shared/_status.py`; they are part of the public type surface.

---

## 8. Old API → new API mapping

Built against the inventory at `src/simsopt/geo/optimizer_jax.py:202-208` (old method set) and `:171` (old backend set). The full caller inventory artifact is referenced separately; this table is the migration cookbook.

| Old call | New call | Migration risk |
|---|---|---|
| `jax_minimize(fn, x0, method="lbfgs", optimizer_backend="scipy")` | `minimize(fn, x0, driver=Driver.SCIPY_LBFGSB)` | trivial |
| `jax_minimize(fn, x0, method="lbfgs-trace")` | `minimize(fn, x0, driver=Driver.SIMSOPT_TRACE_LBFGS, options=SimsoptTraceLBFGSOptions(record_optimizer_state_trace=True))` | low |
| `jax_minimize(fn, x0, method="lbfgs-scipy-jax", optimizer_backend="scipy-jax")` | `minimize(per_term_jax_value_and_grad, x0, driver=Driver.SCIPY_LBFGSB)` | low; caller already has a JAX v/g |
| `jax_minimize(fn, x0, method="lbfgs-scipy-jax-fullgraph", optimizer_backend="scipy-jax-fullgraph")` | `minimize(fullgraph_jax_value_and_grad, x0, driver=Driver.SCIPY_LBFGSB)` | low; fullgraph builder is the caller's responsibility |
| `jax_minimize(fn, x0, method="lbfgs-ondevice", optimizer_backend="ondevice")` | `minimize(fn, x0, driver=Driver.SIMSOPT_LBFGSB)` | trivial |
| `jax_minimize(fn, x0, method="bfgs", optimizer_backend="scipy")` | `minimize(fn, x0, driver=Driver.SCIPY_BFGS)` | trivial; no algorithm change |
| `jax_minimize(fn, x0, method="bfgs-ondevice")` | `minimize(fn, x0, driver=Driver.SIMSOPT_BFGS)` | trivial; no algorithm change |
| `jax_minimize(fn, x0, method="adam-ondevice")` | `minimize(fn, x0, driver=Driver.OPTAX_ADAM, options=OptaxAdamOptions(...))` | low; parity validation pending |
| `jax_least_squares(r, x0, method="lm")` | `least_squares(r, x0, driver=Driver.SCIPY_LM)` | trivial |
| `jax_least_squares(r, x0, method="lm-ondevice")` | bridge: `least_squares(r, x0, driver=Driver.SIMSOPT_LM_GMRES)`; target: `Driver.OPTIMISTIX_LM` + `LinearSolver.LSMR` | medium; algorithm change pending validation |
| `jax_least_squares(r, x0, method="lm-minpack-ondevice")` | bridge: `Driver.SIMSOPT_LM_QR`; target: `Driver.OPTIMISTIX_LM` + `LinearSolver.QR` | medium; MINPACK info-code audit required |
| `jax_least_squares(r, x0, method="optimistix-lm-ondevice")` | `least_squares(r, x0, driver=Driver.OPTIMISTIX_LM, options=OptimistixLMOptions(linear_solver=LinearSolver.LSMR))` | low (already opt-in today) |

**`BoozerSurfaceJAX` two-axis case** (outer driver + inner solver):
- Today: `BoozerSurfaceJAX(optimizer_backend="scipy-jax-fullgraph", least_squares_algorithm="lm")`.
- New: `BoozerSurfaceJAX(outer_driver=Driver.SCIPY_LBFGSB, inner_driver=Driver.SCIPY_LM, fullgraph_value_and_grad_factory=build_single_stage_fullgraph_value_and_grad)`.
- The fullgraph factory is invoked once at solve time; the resulting callable is handed to `minimize`.

---

## 9. Compat shim contract

> **D6 default applied:** both `DeprecationWarning` and structured log.

```python
# simsopt/geo/optimizer_jax.py (existing module, kept as the shim)

def jax_minimize(fn, x0, *, method=None, optimizer_backend=None, **kwargs):
    warnings.warn(
        "simsopt.geo.optimizer_jax.jax_minimize is deprecated; "
        "use simsopt.solve.jax.minimize(fn, x0, driver=...) instead. "
        f"Translation: method={method!r}, optimizer_backend={optimizer_backend!r} → driver={driver}.",
        DeprecationWarning,
        stacklevel=2,
    )
    _DEPRECATION_LOGGER.info(
        "deprecated_solve_jax_call",
        extra={
            "old_method": method,
            "old_optimizer_backend": optimizer_backend,
            "translated_driver": driver,
            "stack": _shim_caller_stack(stacklevel=2),
        },
    )
    return simsopt.solve.jax.minimize(fn, x0, driver=driver, options=options, **rest)
```

**Shim guarantees:**
- One `DeprecationWarning` per (process, call-site stack frame). Suppressed thereafter using a `WeakSet` keyed by stack hash.
- Structured log entry per call (no suppression). Logger name: `simsopt.solve.jax.deprecation`. Goes to autoresearch run logs.
- Translation is **lossless** for combinations explicitly listed in §8. Combinations that don't map cleanly (e.g., `method="bfgs"`) raise `DeprecationWarning("...")` and dispatch to the closest replacement with a second warning naming the algorithm change.

---

## 10. Migration path

Categorized by caller class (caller inventory artifact gives the actual file/line list).

### Category A: Direct callers of `jax_minimize` / `jax_least_squares`

- **Production tests** (`tests/geo/test_boozersurface_jax.py`, `tests/integration/test_single_stage_jax*.py`): mechanical rewrite per the §8 table. One PR per test file.
- **`benchmarks/`**: same.
- **`examples/`**: same, but `single_stage_banana_example.py` has the largest cluster of method-string call sites. Single dedicated PR.

### Category B: `BoozerSurfaceJAX` / `QFMSurfaceJAX` / `BiotSavartJAX` constructor args

- These accept `optimizer_backend=` and `least_squares_algorithm=` as constructor kwargs.
- Migration: add `outer_driver` and `inner_driver` kwargs; deprecate the old string kwargs.
- One PR per class. Tier 3 each (changes `BoozerSurfaceJAX` public API).

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
- `test_compat_shim_translation.py` — every row in §8 → old call produces same `(x, fun, status, success)` as new call within `rtol=0` (lossless translation).
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

If validation of the supersedability claims (§3 bridge drivers) is not complete by `N+2`, `SIMSOPT_LM_GMRES` and `SIMSOPT_LM_QR` stay; the bridge gate is independent of the string-API removal timeline.

---

## 13. Rollback plan

If the shim or new API ships and breaks a known caller class:

1. **Within `N`**: revert is one PR (the new package is purely additive at this stage; the shim is a single module). Existing `optimizer_jax.py` functionality unchanged.
2. **In `N+1`**: the migration PRs (category A above) become individually reversible — each PR is a `Driver.*` substitution that re-targets the same algorithm via the shim. Revert any one without breaking the others.
3. **In `N+2`**: removal of the old string API is the only irreversible step. Mitigation: do not land removal until **zero** structured-log entries for the deprecated call paths have been observed across the active autoresearch run set for at least one release cycle. The structured log is the kill-switch.

---

## 14. Open decisions

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
| D10 | `OptimizerCallbackEvent` schema | Typed discriminated dataclass union with `Literal[Driver.X]` tagging. One concrete event subclass per driver; cross-driver minimum fields on `_OptimizerCallbackEventBase` (`iteration`, `x`, `fun`, `grad_norm_inf`, `wallclock_s`). No `extras: dict[str, Any]` escape hatch. See §7.3. | applied |
| D11 | Where the fullgraph value/grad builders live | `simsopt.objectives.jax.*` (separate from `solve.jax`) | applied |
| D12 | Python floor for `solve.jax` | `requires-python = ">=3.11"` across the package. Matches the existing `JAX_OPTIMISTIX` extra (`pyproject.toml:89-91`), the JAX 0.10.0 runtime, NumPy main, and SPEC 0 direction. Native `StrEnum`; no `class Driver(str, Enum)` mixin. Local-convention tie-breaker (SOFTWARE_DESIGN.md §Consistency rule 3) favors the closest local convention (this repo's JAX work) over upstream simsopt's legacy 3.8+ floor. | applied |

---

## 15. Non-goals

This document does **not** promise:
- That `OPTAX_LBFGS` converges identically to `SIMSOPT_LBFGSB`. The algorithms differ (plain L-BFGS vs L-BFGS-B); parity is a validation question.
- That `OPTIMISTIX_LM` + `LSMR` matches the existing in-tree matrix-free GMRES LM. Both are matrix-free Krylov but the iteration shape and termination differ.
- That `OPTAX_ADAM` byte-matches the in-tree Adam. Default hyperparameters can subtly differ.
- That `SIMSOPT_LM_GMRES` and `SIMSOPT_LM_QR` will be removed. They are bridge drivers; removal is gated on the validation artifacts referenced in §3.
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
    │   ├── contracts.py            # SimsoptLBFGSBOptions, SimsoptBFGSOptions, SimsoptTraceLBFGSOptions, SimsoptLM*Options
    │   ├── _lbfgsb.py              # SIMSOPT_LBFGSB impl (from optimizer_jax_private/_lbfgsb_scipy.py)
    │   ├── _bfgs.py                # SIMSOPT_BFGS impl (from optimizer_jax_private/_bfgs.py)
    │   ├── _trace_lbfgs.py         # SIMSOPT_TRACE_LBFGS impl (from optimizer_host_lbfgs.py)
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
  - **D12**: Python floor pinned at 3.11+ for the whole `solve.jax` package. Stratified design (3.8+ core / 3.11+ extras) dropped.
