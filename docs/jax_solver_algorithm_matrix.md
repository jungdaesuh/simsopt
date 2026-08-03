# SIMSOPT solver and algorithm matrix

Code snapshot audited: working tree, 2026-08-01 solver-runtime refactor

Status: **target architecture, not current behavior**. “Current” means implemented
at the audited snapshot. “Target” is a migration decision and remains gated by
the contracts below.

## Problem-to-solver map

| Problem | Native / CPU current | JAX current | Target generic-kernel provider | Target action |
|---|---|---|---|---|
| Smooth deterministic scalar, unconstrained | SciPy `BFGS` or `L-BFGS-B`; physics often in C++ | SIMSOPT custom BFGS/L-BFGS with eager fixed-step and traced whole-solve routes; Optax explicit comparator | SIMSOPT custom; Optax comparator | Qualify trajectory/performance; retain SciPy reference |
| Smooth scalar, box constrained | SciPy `L-BFGS-B` | Bound-aware private machinery, but the live route fixes `bounds=None`; no public bound-bearing path | SciPy CPU | Keep supported CPU lane; reject unsupported JAX/GPU bounds explicitly |
| Stochastic/noisy scalar or fixed-step schedule | No native C++ Adam | SIMSOPT host/traceable Adam; Optax adapter | Optax Adam/AdamW | Migrate call sites; delete only duplicated update equations after certification |
| Dense/modest nonlinear least squares | SciPy `least_squares`; manual Boozer Newton paths | SIMSOPT LM with JAX pivoted QR; Optimistix LM with selectable Lineax LSMR/QR (LSMR default) | Optimistix LM + Lineax QR | Change default only after differential and failure-mode certification |
| Large matrix-free least squares | SciPy `least_squares` reference | SIMSOPT LM around JAX GMRES; Optimistix LM + Lineax LSMR | Optimistix LM + certified Lineax iterative solve | Qualify by conditioning regime; retain QR/FP64 or CPU fallback |
| Nonlinear roots / Newton polish | Python/SciPy Newton and dense NumPy solves | SIMSOPT Newton/damping/refinement policy around JAX/Lineax primitives | Optimistix Newton + Lineax | Delegate step kernels; retain SIMSOPT acceptance and certificate |
| SPD linear system | NumPy/Eigen/direct solve as applicable | SIMSOPT policy around Lineax CG | Lineax CG | Consolidate adapter and certificate |
| General square matrix-free system | NumPy/Eigen/direct solve as applicable | SIMSOPT policy around `jax.scipy.sparse.linalg.gmres` | Lineax GMRES | Qualify before replacing JAX GMRES |
| Rectangular least-squares system | NumPy/Eigen QR | JAX pivoted QR in SIMSOPT LM; Lineax LSMR/QR in Optimistix adapter | Lineax QR/SVD/LSMR | Select from declared shape, rank, conditioning, and memory budget |
| General nonlinear constraints | SciPy SLSQP, `trust-constr`, COBYLA | Explicitly unsupported generic JAX route | SciPy CPU | Keep; no silent fallback |
| Magnetic-axis ODE | SciPy RK45 | SIMSOPT JAX Dormand-Prince 5(4) | Diffrax candidate | Separate ODE parity audit |
| Field-line ODE and events | C++ Boost Odeint Dormand-Prince 5(4) and Boost TOMS748 event localization | SIMSOPT JAX Dormand-Prince 5(4) and Illinois event localization | Diffrax + Optimistix event root candidate | Separate trajectory/event audit |
| QFM optimization | SciPy L-BFGS-B/SLSQP | SIMSOPT JAX BFGS, penalty, and augmented Lagrangian | Commodity inner kernels only | Keep QFM outer algorithm in SIMSOPT |
| Permanent magnets | C++ MwPGP and GPMO variants | SIMSOPT JAX MwPGP/GPMO variants | Commodity inner kernels only | Keep domain algorithms in SIMSOPT |
| Wireframe coils | C++ GSCO | SIMSOPT JAX GSCO | Commodity inner kernels only | Keep domain algorithm in SIMSOPT |

## Current JAX inventory and target disposition

| Current implementation | Actual ownership | Candidate | Target disposition |
|---|---|---|---|
| SIMSOPT BFGS and L-BFGS-B | Algorithm, line search, stopping, callbacks, SciPy-shaped result; eager fixed-step runtime plus traced whole-solve compatibility | Optax L-BFGS | Keep as production custom provider; compare Optax explicitly |
| Private L-BFGS-B machinery; live route specializes to `bounds=None` | Bound-aware internals, projected-gradient logic, and More-Thuente search; no supported bound-bearing route | None selected for supported JAX | Keep private during qualification; delete if JAX bounds remain unsupported |
| SIMSOPT host/traceable Adam | Update equations, stopping, callbacks, result policy | Optax Adam/AdamW | Replace equations; retain normalized policy |
| SIMSOPT LM + JAX GMRES/pivoted QR | Nonlinear state machine, damping, acceptance, linear-solve policy | Optimistix + Lineax | Replace generic state/step machinery after certification |
| SIMSOPT Newton/exact Newton | Damping, acceptance, continuation, refinement, certificate | Optimistix + Lineax | Retain SIMSOPT policy; evaluate kernels per call site |
| Lineax CG/LSMR/QR and JAX GMRES/LU/QR wrappers | Selection, refinement, condition estimates, forward-error checks | Lineax consolidation | Replace wrappers only where result and diagnostic contracts match |
| Mixed FP32/FP64 refinement and FP64 fallback | Precision and scientific-acceptance policy | No drop-in replacement | Keep |
| Fixed-iteration roots with implicit VJP | Differentiable nested-solve contract | Optimistix implicit differentiation | Evaluate per call site |
| Dormand-Prince 5(4) and Illinois false position | Domain integration/event semantics | Diffrax + Optimistix | Separate migration audit |
| QFM ALM; MwPGP/GPMO; GSCO | Domain algorithms and state transitions | No generic replacement | Keep |
| Backend-specific result/status/counters | Public compatibility is not normalized today | SIMSOPT semantic adapter | Implement one shared layer before backend deletion |

## External library capability

| Library | Adam | Unconstrained quasi-Newton | Box-constrained scalar | Nonlinear least squares | Roots | Linear solves | General constraints | Selected role |
|---|---|---|---|---|---|---|---|---|
| Optax | Adam/AdamW | L-BFGS | Projection utilities, not L-BFGS-B | No | No | No | No general constrained optimizer | Explicit comparator; no silent custom-provider replacement |
| Optimistix | Via Optax adapter; not native | BFGS/L-BFGS | No L-BFGS-B | LM | Newton/root solvers | Via Lineax | No general constrained optimizer | Nonlinear least squares and roots |
| Lineax | No | No | No | Linear subproblems only | No | LU, Cholesky, QR, SVD, CG, GMRES, LSMR | No | Linear kernels |
| SciPy | No | BFGS/L-BFGS-B | L-BFGS-B | Yes; bounds via TRF/dogbox, not LM | Yes | Yes | SLSQP, `trust-constr`, COBYLA | CPU bounds, constraints, and reference |
| Diffrax | No | No | No | No | Event localization via Optimistix | Internal use of Lineax | No | ODE/event candidate only |
| JAX core | No optimizer | Public BFGS only; hidden unsupported route named `l-bfgs-experimental-do-not-rely-on-this` | No | No general LM | No general root solver | CG, GMRES, BiCGSTAB | No | Low-level primitives |
| JAXopt | Optax wrapper | BFGS/L-BFGS | L-BFGS-B and projection-based methods | LM | Bisection/Broyden/Anderson and root wrappers | LU, Cholesky, inverse, QR, CG, normal CG, GMRES, BiCGSTAB; iterative refinement | Box/QP/equality-QP/OSQP methods | Prior art only; no longer maintained |

## Target ownership

All external providers remain behind a SIMSOPT adapter.

| Owner | Target responsibility |
|---|---|
| Optax | Adam/AdamW and explicit unconstrained L-BFGS comparator; no default replacement decision |
| Optimistix | LM and Newton/root generic state and step kernels |
| Lineax | Linear factorization and iterative-solve kernels |
| SciPy | CPU L-BFGS-B; bounded least squares via TRF/dogbox; general constraints; reference lane |
| Diffrax | Candidate ODE/event kernels only after a separate audit |
| SIMSOPT | Custom BFGS/L-BFGS algorithms; problem classification; solver selection; domain algorithms; acceptance/certification; damping/continuation policy; budgets/counters; status normalization; placement; precision escalation; fallback |

## Evidence status

| Evidence | Scope | Result | Permitted use |
|---|---|---|---|
| Generic-solver replacement authority | No tracked clean-revision receipts at the audited snapshot | Uncertified | Blocks promotion and predecessor deletion; requires a new tracked differential/failure campaign |
| Tracked native/JAX campaign summary | Local-only bounded run `20260729T005942Z-5ade9aee` at `11340c829690fdc0652e47588f5da549829c056a`: 26 external-solver-free mirrors; native CPU, JAX CPU, and strict RTX 5090 GPU; native-default scale not run; raw receipts are host-local and unshipped | Summary records 26/26 cases, 78/78 lane receipts, and 1,248/1,248 declared comparisons passing the campaign predicate; raw optimizer convergence is not asserted | Aggregate workflow/physics regression evidence only; not independently reproducible solver-interchangeability or speed evidence |

Authority artifact: `examples/jax/authority_evidence.json` (SHA-256
`13d5843438a93c748234e7cb8a52eb75f7627d832fe8dd3658eea5759e59a521`).

### Reconciliation with the production-architecture blindspot report (2026-08-03)

`docs/jax_solver_production_architecture_blindspot_report.md` (2026-07-28,
snapshot `9fbb569`) assigned smooth deterministic unconstrained scalar
minimization to Optax L-BFGS as production owner. The 2026-08-02 quasi-Newton
closure campaign supersedes that row for this problem family: the SIMSOPT
custom L-BFGS `fused_stepwise` route measured statistical performance par with
the Optax comparator on the pinned RTX 5090 (quiet warm medians 27.65 ms vs
26.03 ms with overlapping spreads; contended 45.9 ms vs 48.9 ms), while
uniquely providing the SciPy-shaped result contract and the post-acceptance
callback hook that the accepted-incumbent Boozer continuation architecture
requires and Optax's scan loop cannot express. The custom provider therefore
remains production owner for this family with Optax retained as an explicit,
receipt-gated comparator, as recorded in the tables above. The blindspot
report's remaining rows are unaffected.

## Migration contracts and live gaps

| Contract | Target requirement | Live status at audited snapshot |
|---|---|---|
| Bounds | Active-set/KKT semantics; clipping is not L-BFGS-B | Public generic JAX bounds unsupported |
| Success | SIMSOPT certificate, not raw library `success` | No single backend-neutral certificate |
| Precision | FP64 authority and explicit mixed-precision fallback | SIMSOPT policy exists; replacement stack not certified |
| Placement | Unsupported device/problem pairs fail explicitly | No target semantic dispatcher |
| Device residency | No implicit NumPy materialization or host callback in inner solve | Optax is host-controlled and synchronizes each iteration; Optimistix materializes the final result; public adapters return host results |
| Counters | Report only measured evaluations | Some adapters synthesize counts |
| Dependencies | Published exact supported stack plus latest-compatible canary | Unmet: no tracked lock/constraints; SciPy, Optax, Optimistix, Lineax, and Equinox use open lower bounds |
| Dependency compatibility | Declared floors satisfy the pinned JAX/JAXlib metadata | Unmet: JAX extra declares `scipy>=1.13`; JAX 0.10.0 requires `scipy>=1.14` |
| Deletion | Clean-revision, tracked differential and failure-mode authority across all call sites | Unmet for generic replacements |

## Primary source locations

| Area | Path |
|---|---|
| Driver enum and legacy mapping | `src/simsopt_jax/solve/driver.py` |
| Public typed dispatch/adapters | `src/simsopt_jax/solve/dispatch.py` |
| Generic SIMSOPT JAX optimizer policy | `src/simsopt_jax/geo/optimizers/optimizer.py` |
| Linear-solve wrappers and certification | `src/simsopt_jax/geo/optimizers/linear_solve.py`; `src/simsopt_jax/geo/optimizers/adjoint_linear_solve.py` |
| Host and private BFGS/L-BFGS-B | `src/simsopt_jax/geo/optimizer_host_lbfgs.py`; `src/simsopt_jax/geo/optimizers/private/` |
| Native SciPy solves | `src/simsopt/solve/serial.py`; `src/simsopt/solve/mpi.py`; `src/simsopt/geo/boozersurface.py` |
| Magnetic-axis ODE | `src/simsopt/field/magnetic_axis_helpers.py`; `src/simsopt_jax/core/magnetic_axis_helpers.py` |
| Field-line ODE/events | `src/simsoptpp/tracing.cpp`; `src/simsopt_jax/core/tracing.py` |
| JAX domain algorithms | `src/simsopt_jax/core/qfm_solver.py`; `src/simsopt_jax/core/pm_optimization.py`; `src/simsopt_jax/core/wireframe_workflow.py` |
| C++ domain algorithms | `src/simsoptpp/permanent_magnet_optimization.cpp`; `src/simsoptpp/wireframe_optimization.cpp` |

## External primary sources

| Library | Source |
|---|---|
| Optax | [Optimizers](https://optax.readthedocs.io/en/latest/api/optimizers.html); [projections](https://optax.readthedocs.io/en/latest/api/projections.html) |
| Optimistix | [Minimization](https://docs.kidger.site/optimistix/api/minimise/); [least squares](https://docs.kidger.site/optimistix/api/least_squares/); [root finding](https://docs.kidger.site/optimistix/api/root_find/) |
| Lineax | [Linear solvers](https://docs.kidger.site/lineax/api/solvers/) |
| SciPy | [`least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html); [`minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) |
| Diffrax | [Events](https://docs.kidger.site/diffrax/api/events/) |
| JAX | [`jax.scipy.optimize.minimize`](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.minimize.html); [JAX 0.10.0 hidden L-BFGS route](https://github.com/jax-ml/jax/blob/jax-v0.10.0/jax/_src/scipy/optimize/minimize.py#L98-L122); [sparse linear algebra](https://docs.jax.dev/en/latest/jax.scipy.html#module-jax.scipy.sparse.linalg) |
| JAXopt | [Maintenance status](https://github.com/google/jaxopt#status); [L-BFGS-B](https://jaxopt.github.io/stable/_autosummary/jaxopt.LBFGSB.html) |
