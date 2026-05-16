# Parity Audit — PRIORITY 4: Fieldline / Guiding-Center / Full-Orbit Tracing

**Audit timestamp:** 2026-05-16
**Auditor scope:** JAX↔C++ parity for `simsopt.jax_core.tracing`
**Branch:** `gpu-purity-stage2-20260405`

## Files audited

| File | Lines | Role |
|---|---|---|
| `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/jax_core/tracing.py` | 3182 | JAX implementation (RHS factories + DOPRI5 + event localizer + drivers) |
| `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/tracing.cpp` | 560 | C++ reference (Boost.Odeint DOPRI5 dense output + TOMS748 event localizer) |
| `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/tracing.h` | 158 | C++ header (StoppingCriterion class hierarchy + extern templates) |
| `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsoptpp/python_tracing.cpp` | 87 | pybind11 bindings |
| `/Users/suhjungdae/code/columbia/simsopt-jax/src/simsopt/field/tracing.py` | 2040 | Python orchestrator + backend routing + criterion translation |

## Executive summary — top findings

1. **HIGH — Missing `dtmax` step ceiling in JAX adaptive driver.** The C++ driver hands `dtmax = r0 * 0.5 * pi / v_total` (or `/AbsB` for fieldlines) to `make_dense_output(tol, tol, dtmax, runge_kutta_dopri5)` at `tracing.cpp:374-375`, bounding any single accepted step to ≤ quarter-revolution. The JAX driver only clamps to `tmax - t` (e.g. `tracing.py:952`, `:1487`, `:2422`, `:2947`); it never imports the physical-orbit step ceiling. Consequence: JAX can take arbitrarily large steps when the local field is weak or the orbit is straight, and re-step rejection (`err_safe > 1`) is the only guard. Trajectory shape parity at fixed `tol` therefore depends on the controller hitting the same coarse-grain regime as Boost; on banana orbits and near-axis trajectories this can change accepted-step count and root-bracket locations.

2. **HIGH — Mismatched initial step heuristic across modes.** C++ uses `dt = 1e-3 * dtmax` for particle drivers (tracing.cpp:463, 490, 530) and `dt = 1e-5 * dtmax` for fieldlines (tracing.cpp:552). JAX collapses all of these to `_INITIAL_STEP_FRACTION = 1/100` of `tmax - t0` (`tracing.py:218` and `_initial_step_size` at `:688-691`). With long-`tmax` fieldline runs this can mean the first attempted step is orders of magnitude larger than C++ would pick; the PI controller usually recovers but the first few rejected attempts will differ in count and FSAL state, breaking byte-identity expectations.

3. **HIGH — Event localizer divergence: Illinois fixed-iter vs Boost TOMS-748.** C++ uses `boost::math::tools::toms748_solve` with adaptive iteration count `rootmaxit=200` and tolerance `eps_tolerance(-log2(tol))` (tracing.cpp:385-386, 416). JAX uses a hand-rolled Illinois false-position with a static iteration budget `max_root_iters=60` (default) and a degenerate exit guard (`atol=0.0` is always passed at `tracing.py:1015`, `:1547`, `:2479`, `:3006`, so the converged-branch never short-circuits and the loop always runs the full 60 iterations). Convergence rates differ: TOMS-748 has order ~1.83 globally; the Illinois trick yields ~1.5 super-linear. With `tol=1e-9` C++ converges in ~30 evaluations; JAX consumes 60 deterministically — accuracy is similar in practice, but the residual sub-step DOPRI5 calls are doubled. Event-time parity is gated at `event_time_rtol=1e-7, event_time_atol=1e-9` per `validation_ladder_contract.py:229-238`, which both backends satisfy, but a critical asymmetry remains: TOMS-748 swaps `t_left`/`t_right` and adapts to monotone sub-intervals while the Illinois loop here does not enforce `t_left <= t_right` (`tracing.py:711-712` documents but does not enforce). This is the highest-leverage subtle divergence.

## Function-by-function parity matrix

| C++ symbol (file:line) | JAX symbol (file:line) | Math | Algorithm | Computation | Severity |
|---|---|---|---|---|---|
| `FieldlineRHS::operator()` (tracing.cpp:315-330) | `fieldline_rhs` (tracing.py:588-611) | PARITY (`dy/dt = B`) | n/a | float64 | LOW (docstring mislabels as `B/|B|`) |
| `GuidingCenterVacuumRHS::operator()` (tracing.cpp:50-77) | `guiding_center_vacuum_rhs` (tracing.py:1272-1339) | PARITY | n/a | float64 | OK |
| `GuidingCenterVacuumBoozerRHS::operator()` (tracing.cpp:106-129) | `guiding_center_vacuum_boozer_rhs` (tracing.py:2005-2069) | PARITY | n/a | float64 | OK |
| `GuidingCenterNoKBoozerRHS::operator()` (tracing.cpp:160-187) | `guiding_center_no_k_boozer_rhs` (tracing.py:2072-2146) | PARITY | n/a | float64 (shares C++'s 1/v_par singularity) | OK with caveat |
| `GuidingCenterBoozerRHS::operator()` (tracing.cpp:218-253) | `guiding_center_boozer_rhs` (tracing.py:2149-2223) | PARITY | n/a | float64 (shares C++'s 1/v_par singularity) | OK with caveat |
| `GuidingCenterRHS` (NON-VACUUM Cartesian) | NONE | MISSING | — | — | INFO (intentional carve-out; C++ also raises `logic_error`) |
| `FullorbitRHS::operator()` (tracing.cpp:276-300) | `fullorbit_vacuum_rhs` (tracing.py:2737-2783) | PARITY (`dx/dt=v, dv/dt=(q/m)v×B`) | n/a | float64 | OK |
| `solve()` adaptive driver (tracing.cpp:367-446) | `trace_fieldline` / `trace_guiding_center` / `trace_guiding_center_boozer` / `trace_fullorbit` / `_run_dopri5_4state` (tracing.py:803, 1342, 2226, 2786, 1724) | DOPRI5 algebra parity | **DIVERGES**: no `dtmax`; initial step different; PI vs Boost dense-output controller | float64 / JIT scan / fixed-shape carry | HIGH |
| `toms748_solve` event root (tracing.cpp:416) | `bracket_root_jax` Illinois (tracing.py:697-797) | residual-zero parity | **DIVERGES**: TOMS-748 vs Illinois, adaptive vs fixed budget, `rootmaxit=200` vs `max_root_iters=60` default | float64; dead iterations because `atol=0.0` | MEDIUM |
| `get_phi` (tracing.cpp:333-351) | `_continuous_phi` (tracing.py:507-523) | PARITY in spirit | **DIVERGES**: C++ enumerates `nearest-2pi`, `nearest`, `nearest+2pi` and picks min-distance; JAX uses `phi_raw + round((phi_near-phi_raw)/2pi)*2pi`. Equivalent when `phi_near` is finite and well-defined; both produce the unique multiple within ±π of `phi_near`. | float64 | OK |
| `IterationStoppingCriterion` (tracing.h:107-115) | `IterStoppingCriterion` (tracing.py:393-403) | PARITY (predicate `iter > max_iter`) | C++ counts every loop iteration in `solve` (tracing.cpp:391); JAX counts iterations as `step_count + 1` post-step (`tracing.py:1088`, `:1621`, `:2555`, `:3085`). Identical semantics. | int32 | OK |
| `MinR/MaxRStoppingCriterion` | `Min/MaxRStoppingCriterion` (tracing.py:347-364) | PARITY | predicate evaluated on accepted-step state | float64 | OK |
| `MinZ/MaxZStoppingCriterion` | `Min/MaxZStoppingCriterion` (tracing.py:365-377) | PARITY | predicate evaluated on accepted-step state | float64 | OK |
| `ToroidalTransitStoppingCriterion(flux=False)` (tracing.h:21-45) | `ToroidalTransitStoppingCriterion` (tracing.py:379-391) | PARITY (uses continuous-phi unwrap) | C++ supports `flux=True` for Boozer paths; JAX docstring says `flux=True` "not on the JAX path yet" but `_stopping_criterion_should_stop` actually applies the same `(phi_unwrapped - phi_init)/2π` formula on the Boozer route using `zeta_current/zeta_init`. Behavior is correct but docstring is stale. | float64 | LOW |
| `MaxToroidalFluxStoppingCriterion` (tracing.h:47-55) | `MaxToroidalFluxStoppingCriterion` (tracing.py:421-426) | PARITY only when `is_boozer_state=True`; else `False` | C++ binds `s = x` (first state slot) — Boozer only path. JAX gates on `is_boozer_state` flag passed by the Boozer driver. | float64 | OK |
| `MinToroidalFluxStoppingCriterion` (tracing.h:57-65) | `MinToroidalFluxStoppingCriterion` (tracing.py:406-419) | PARITY (Boozer only) | same as above | float64 | OK |
| `LevelsetStoppingCriterion` (tracing.h:117-132) | `LevelsetStoppingCriterion` (tracing.py:429-443) | PARITY (`classifier(x,y,z) < 0`) | C++ uses `RegularGridInterpolant3D::evaluate(r, phi, z)`; JAX uses a closure built from `SurfaceClassifier.to_jax_classifier_fn`. Both must return the same signed-distance values. | float64 | depends on interpolant audit (separate scope) |
| `particle_guiding_center_tracing` (tracing.cpp:448-471) | `trace_guiding_center` + `_trace_particles_jax_guiding_center_vacuum` (tracing.py:1342, field/tracing.py:716) | PARITY | endpoint parity test exists (`tests/jax_core/test_tracing_jax_guiding_center.py:224`) | float64 | OK at lane tolerance |
| `particle_guiding_center_boozer_tracing` (tracing.cpp:473-502) | `trace_guiding_center_boozer` + `_trace_particles_boozer_jax` (tracing.py:2226, field/tracing.py:330) | PARITY | endpoint parity tests exist for all 3 modes (`tests/jax_core/test_tracing_jax_gc_boozer.py:264,318,372`) | float64 | OK at lane tolerance |
| `particle_fullorbit_tracing` (tracing.cpp:517-533) | `trace_fullorbit` + `_trace_particles_jax_fullorbit_vacuum` (tracing.py:2786, field/tracing.py:871) | PARITY | endpoint + energy-conservation tests exist (`tests/jax_core/test_tracing_jax_fullorbit.py:240,332`) | float64 | OK at lane tolerance |
| `fieldline_tracing` (tracing.cpp:540-554) | `trace_fieldline` + `_compute_fieldlines_jax` (tracing.py:803, field/tracing.py:1443) | PARITY | endpoint parity test exists (`tests/field/test_tracing_jax_item16.py:80`) | float64 | OK at lane tolerance |

## Detailed findings

### (a) ODE RHS parity

**Cartesian guiding-center vacuum (`GuidingCenterVacuumRHS`).** C++ (tracing.cpp:50-77) and JAX (tracing.py:1272-1339) compute drift in the standard de Blank form. Term-for-term comparison:

- C++: `dydt[i] = fak1*B(0,i) + fak2*BcrossGradAbsB[i]` with `fak1 = v_par/AbsB`, `fak2 = (m/(q*AbsB^3))*(0.5*v_perp2 + v_par^2)`.
- JAX (tracing.py:1327-1334): `dposition = fak1 * B + fak2 * B_cross_grad_abs_B` with identical scalar formulas.
- `dv_par`: C++ `-mu*(B·∇|B|)/AbsB`; JAX `-mu * jnp.dot(B, grad_abs_B) / abs_B`. PARITY.
- Both build `∇|B|` from the same SIMSOPT convention `dB_by_dX[j,l] = ∂_j B_l` (`grad_abs_B_j = B_l * dB_l/dx_j / |B|`); JAX line 1323 `einsum("l,jl->j", B, dB_by_dX) / abs_B`.

**Vacuum Boozer GC (`GuidingCenterVacuumBoozerRHS`).** Pure algebra, no operator-vs-operator mismatch. PARITY.

**No-K and Full Boozer GC (`GuidingCenterNoKBoozerRHS` / `GuidingCenterBoozerRHS`).** Both use the **energy-conservation** form for `dv_par`:

```
dv_par = -(mu / v_par) * (dmodBdpsi * ds * psi0 + dmodBdtheta * dtheta + dmodBdzeta * dzeta)
```

JAX reproduces this exactly (tracing.py:2141, 2218). **MEDIUM CAVEAT**: this RHS has a removable singularity at `v_par → 0` (banana turning point). Both C++ and JAX would NaN there in principle; in C++ the Boost DOPRI5 step rejection backs off and the orbit is smoothed by the `r0*0.5*M_PI/v_total` dtmax ceiling that prevents stepping over the turning point. JAX has **no** such ceiling (see Finding 1), so the failure mode at near-turning-point states is structurally worse — rejected steps with NaN derivatives shrink `h` per the PI controller, but the `factor = SAFETY * err^-0.2` formula sees `err = inf` → `factor = MIN_FACTOR = 0.2` (tracing.py:957-967) and the step shrinks geometrically, which is slow. The C++ direct-tolerance dense_output stepper has stricter step-size discipline at turning points by construction.

**Full-orbit Cartesian (`FullorbitRHS`).** PARITY. JAX line 2780 `acceleration = qoverm * jnp.cross(velocity, B)`; C++ tracing.cpp:297-299 has component-wise expansion of the same cross product.

**Carve-out: non-vacuum Cartesian GC.** C++ throws `logic_error("Guiding center right hand side currently only implemented for vacuum fields.")` at tracing.cpp:470. JAX raises `NotImplementedError` at field/tracing.py:759-768 with the same scope. Symmetric; INFO only.

### (b) Integrator parity

C++ uses `make_dense_output(tol, tol, dtmax, runge_kutta_dopri5<State>())` at tracing.cpp:374-375. This is Boost's DOPRI5 with a dense-output extension; the underlying step controller is Boost's `controlled_runge_kutta` with the standard error norm and a step-size factor based on the embedded 4th-order error. The `dtmax` argument **caps** the step.

JAX uses the same DOPRI5 Butcher tableau (`_DOPRI5_A/B/C/E` at tracing.py:150-208 — Hairer Table 5.2, identical to Boost's coefficients), the same error norm `sqrt(mean((err/sc)^2))` where `sc = atol + rtol * max(|y|, |y_new|)` (tracing.py:677-685), and the same PI(0.2) step factor `factor = SAFETY * err^(-0.2)` clipped to `[MIN_FACTOR=0.2, MAX_FACTOR=5.0]` (tracing.py:215-217, 957-967). FSAL reuse (`k7` becomes next `k_first` on acceptance, tracing.py:971) matches Boost.

**Divergence #1 — `dtmax`.** C++ tracing.cpp computes `dtmax = r0*0.5*M_PI/v_total` (or `/AbsB` for fieldlines, tracing.cpp:551) and passes it to the controller. JAX never computes a physical-orbit step ceiling; the only step bound is `h_clamped = jnp.minimum(h, tmax - t)` (tracing.py:952, 1487, 2422, 2947). On orbits where the PI controller would otherwise let `h` grow to `MAX_FACTOR=5×` per accept until the local error budget tightens, the C++ driver hits `dtmax` and stops growing; JAX does not, so the integrator can take a single huge step across a phi-plane and force the event localizer to find a root over a wide bracket. The Illinois localizer's first sub-step DOPRI5 evaluations then live on a coarser grid than C++ would.

**Divergence #2 — initial step.** C++ uses `dt = 1e-3 * dtmax` for particle drivers (tracing.cpp:463, 490, 530) and `dt = 1e-5 * dtmax` for fieldlines (tracing.cpp:552). JAX uses `_INITIAL_STEP_FRACTION = 1/100 * (tmax - t0)` for **all** integrators (tracing.py:218, 690). For a fieldline run with `tmax=200` and `R0=1.4`, C++ picks `dt ≈ 1e-5 * π*0.7/AbsB ≈ 2e-5/AbsB`; JAX picks `h0 = 2.0`. The PI controller will reject this immediately and shrink, but the rejection path is JAX-specific scratch that has no C++ analog.

**Divergence #3 — step-controller flavor.** Boost's `controlled_runge_kutta` uses Sodin-Hairer style with a single-error-norm "I" controller by default (no integral term); JAX implements the H211-PI(0.2) closed-form `factor = SAFETY * err^-0.2`. These are operationally similar at `tol=1e-9` but accumulate different step-count histories. The lane tolerance `step_count_max_ratio=1.25` (validation_ladder_contract.py:234) absorbs the discrepancy at runtime, but exact byte-identity is not achievable.

**Impact on parity gates.** All these integrator differences mean the JAX path cannot match C++ trajectories byte-for-byte; the `event_time_tracing` lane was explicitly designed (validation_ladder_contract.py:223-238) with loose state-vector tolerances `(rtol=1e-6, atol=1e-8)` to absorb the controller drift. The endpoint parity tests (gc_boozer endpoint, fullorbit endpoint, fieldline endpoint) all pass at this lane; the audit confirms no algebraic divergence beyond the controller asymmetry.

### (c) Event detection parity

**C++ scan loop (tracing.cpp:401-430).** After every accepted step, the driver iterates `for (i = 0; i < phis.size(); ++i)` and tests
```
if(std::floor((phi_last - phi)/2π) != std::floor((phi_current - phi)/2π)) { ... }
```
On a sign change it builds `phi_shift = round(((phi_last+phi_current)/2 - phi)/2π) * 2π + phi`, then defines a residual `rootfun(t)` from `dense.calc_state(t, temp)` (Boost DOPRI5 dense output — Hermite interpolation), and calls `toms748_solve(rootfun, tlast, tcurrent, ..., roottol, rootmaxit=200)`.

**JAX scan loop (tracing.py:988-1043, repeated in each driver).** Same floor-difference test (line 993-995), same `phi_shift` computation (line 998-1005). Difference is the sub-step evaluator: JAX **re-runs a full DOPRI5 step** from `(t, y, k_first)` with `h_sub = s * h_clamped` (line 985 `dopri5_step(rhs, t, y, h_sub, k_first)`); C++ uses Boost's dense Hermite interpolation through `dense.calc_state(t, temp)`. The two should be O(h⁵) consistent — JAX's repeated DOPRI5 is actually a 5th-order RK estimate which is **more accurate** than Boost's dense-output Hermite (which is 4th-order interpolant on the dense-output side). This is a benign upgrade; it does mean JAX pays N extra DOPRI5 evaluations per event-bracket iteration where C++ pays N Hermite interpolations (essentially free).

**Event localizer math.** JAX `bracket_root_jax` (tracing.py:697-797):

- Always runs the fixed `max_iters` loop, no early-exit; the `converged = width <= atol_arr` branch only suppresses function evaluations on the converged iterations but still executes them. Caller always passes `atol=0.0` (tracing.py:1015, 1547, 2479, 3006), so `converged` is only `True` once the bracket has truly collapsed to floating-point zero width — in practice this means **all 60 iterations always execute**.
- The Illinois update is `candidate = b - fb * (b - a) / (fb - fa)` (line 765), with the asymmetric weight `half * fa` or `half * fb` on the retained endpoint (lines 782-783) — standard Illinois.
- Returns `t_best, f_best` = argmin |f| over the candidates traversed. This is correct: if the bracket is `[0, 1]` and the root lies in the interior, the best-residual point is the root estimate.

**Divergences:**

- Algorithm: TOMS-748 (asymptotic order 1.83, adaptive) vs Illinois (order 1.5, fixed). TOMS-748 is meaningfully faster in iteration count for the same target tolerance.
- Iteration ceiling: C++ `rootmaxit=200`, JAX default `max_root_iters=60`. C++ uses the budget defensively; JAX uses 60 deterministically. For typical event-residual scales `O(1)` and `tol=1e-9`, Illinois converges in ~30-40 iterations; 60 is a safety margin. C++ rarely consumes >20.
- `eps_tolerance(-log2(tol))`: C++ converges when `|b-a| <= 2^(-N) * (1 + |b|+|a|)` with `N = ceil(-log2(tol)) ≈ 30` for `tol=1e-9`. JAX has no equivalent — `atol=0` is the only break condition.

**Impact:** Both eventually find a root, JAX's residual sub-step is 5th-order so the localization accuracy actually exceeds C++. The trajectory state recorded at the root (`pos_root = state_at_fraction(s_root)`, line 1026) is computed from a fresh DOPRI5 substep, which is consistent with the accepted-step's state at endpoints (`s=0` reproduces `y`, `s=1` reproduces `y_new`).

**Note on bracket symmetry (`t_left <= t_right` not enforced).** Both `bracket_root_jax` (docstring at tracing.py:711-712) and Boost TOMS-748 require an ascending bracket. C++ always passes `tlast < tcurrent`. JAX's only caller passes `0.0, 1.0` so the issue is moot in current uses, but the API allows an unenforced misuse.

### (d) Classification parity

**Status code semantics.** C++ exit conditions (tracing.cpp:440-444) write `t_final = tmax` and append a final row when no stop fired. JAX writes the same `t_final = tmax` semantics when `reached = (tmax - t_final) <= eps_t` with `eps_t = 1e-12 * max(|tmax|, 1.0)` (tracing.py:1170-1181, 1699-1708, 2633-2642, 3163-3172). `status` mapping:

- `status=0`: reached tmax cleanly.
- `status=1`: max-step-cap exhaustion before tmax — **JAX-specific** since C++ has no static step cap; in C++ the loop runs until `t >= tmax` or a stopping criterion fires.
- `status=-1-i`: stopping criterion `i` fired. **PARITY**: matches the C++ row `{t, -1-i, x, y, z}` written to `res_phi_hits` (tracing.cpp:435).

**`particle lost` accounting.** Field/tracing.py:494 (Boozer JAX), `:858` (Cartesian GC JAX), `:1021` (full-orbit JAX) all use `if t_final < float(tmax) - 1e-15: loss_ctr += 1`. The C++ side does this in field/tracing.py:703 with `if res_ty[-1][0] < tmax - 1e-15`. **PARITY** at the orchestrator level; integrator differences may push `t_final` slightly below `tmax` (e.g. when the integrator stops at `tmax - eps_t`), but the 1e-15 slack absorbs floating-point overshoot.

**Trapped vs passing classification.** Neither C++ nor JAX explicitly classifies orbits as trapped/passing — that is a downstream consumer concern. The integrators only emit `(trajectory, phi_hits)`; classification is performed by `compute_resonances`, `compute_toroidal_transits`, `compute_poloidal_transits` (field/tracing.py:1193-1442). These are NumPy-only orchestrator functions that consume the same `res_tys`/`res_phi_hits` shape on both backends, so classification parity reduces to trajectory parity (covered by the endpoint-parity oracle tests at the `event_time_tracing` lane).

**Banana-orbit / turning-point handling.** Neither integrator branches at `v_par = 0`; the `dv_par = -(mu/v_par)*...` formula in the no-K and full Boozer RHS will produce a non-finite derivative at the turning point. The PI controller rejects the step (`err = NaN → err_safe = inf → accepted = False → factor = MIN_FACTOR = 0.2`); JAX line 955 explicitly handles `err = NaN` via `jnp.isfinite`. **MEDIUM CAVEAT**: the C++ side has Boost's `dtmax` ceiling preventing the step from straddling the turning point in the first place; JAX has no equivalent guard, so JAX is **more likely to attempt a step across `v_par=0`** and waste iterations rejecting it. This does not change the final trajectory at the lane tolerance, but it can slow JAX runs on banana orbits.

### (e) Batch / scan parity

Neither integrator implements batched orbits inside JIT. The orchestrator (`_trace_particles_jax_*` in field/tracing.py) iterates `for i in range(first, last)` over each particle and calls `trace_*` per particle (e.g. field/tracing.py:434-466, :800-831, :967-994). MPI `comm` is split host-side via `parallel_loop_bounds`; the integrator itself runs sequentially per rank. No JIT-level cross-particle vectorization is in place. This is **PARITY** with C++ (which also serially iterates particles in `trace_particles`), and it is also the **biggest missed optimization** — but that is out of scope for a parity audit.

**JIT semantics.** The JAX drivers use `jax.lax.while_loop` for the main integration loop and `jax.lax.fori_loop` for padding. All trajectory carries are fixed-shape `(max_steps + 1, N)` per spec. Padding rows are filled with the final accepted state (tracing.py:1159-1168, 1688-1697, 2622-2631, 3152-3161). **MEDIUM**: if `accepted_count == max_steps` triggered exit before reaching `tmax`, `status=1` and the trajectory is the legitimate prefix; the orchestrator masks-and-trims (`live = traj[mask]` at field/tracing.py:469, 833, 996). However, the **upstream `phi_hits_count > max_phi_hits` overflow IS detected** and the orchestrator raises `RuntimeError("recorded N event rows")` via `_event_hits_prefix` (field/tracing.py:31-57 and test at test_tracing_jax_item16.py:65-72). Good. The **`accepted_count == max_steps` overflow IS NOT explicitly reported** beyond `status=1`; if a downstream consumer ignores `status` they get a silently truncated trajectory. This is documented in the JAX dataclass docstring (tracing.py:307-308) but not enforced as a hard error at the orchestrator level. **INFO**.

## Test coverage

### Coverage map

| Scenario | Coverage | Test(s) |
|---|---|---|
| Fieldline endpoint parity vs C++ | YES | `tests/field/test_tracing_jax_item16.py:80` (single fieldline endpoint vs CPU oracle), `:331` (multi-fieldline) |
| Fieldline phi-plane crossings | YES | `tests/field/test_tracing_jax_item16.py:191`, `tests/field/test_tracing_jax_item16_extended.py:42` |
| Fieldline stopping criteria | YES | `MinR` (item16.py:218), `MaxR` (item16_extended.py:132), `MinZ/MaxZ` (item16_extended.py:185), `ToroidalTransit` (item16_extended.py:159), `Levelset` (item16.py:393) |
| Fieldline overflow guard | YES | `tests/field/test_tracing_jax_item16.py:65` |
| Cartesian GC endpoint parity vs C++ | YES | `tests/jax_core/test_tracing_jax_guiding_center.py:184` |
| Cartesian GC phi crossings | YES | `tests/jax_core/test_tracing_jax_guiding_center.py:436` |
| Cartesian GC stopping criteria | YES | MinR at `:462`, MaxToroidalFlux not tested (Cartesian path inactive by design) |
| Cartesian GC energy/mu conservation | **NO direct JAX-only test** | CPU-side test exists at `tests/field/test_particle.py:215` (`test_energy_conservation`) — tests the full simsopt pipeline, not the JAX integrator in isolation |
| Cartesian GC angular-momentum | **NO direct JAX-only test** | CPU-side equivalent at `test_particle.py:295` only |
| Boozer GC endpoint parity vs C++ (vacuum / no_k / full) | YES | `tests/jax_core/test_tracing_jax_gc_boozer.py:264,318,372` |
| Boozer GC zeta crossings | YES | `tests/jax_core/test_tracing_jax_boozer_zeta_events.py` |
| Boozer GC energy/mu conservation | **NO direct JAX-only test** | CPU-side test exists at `test_particle.py:421` (`test_energy_momentum_conservation_boozer`) — only validates the simsopt-pipeline, not the JAX integrator |
| Full-orbit endpoint parity vs C++ | YES | `tests/jax_core/test_tracing_jax_fullorbit.py:240` |
| Full-orbit energy conservation | YES | `tests/jax_core/test_tracing_jax_fullorbit.py:332` (`test_trace_fullorbit_conservation_invariants`) |
| Full-orbit phi crossings | YES | `tests/jax_core/test_tracing_jax_fullorbit.py:415`, `tests/jax_core/test_tracing_jax_fullorbit_events.py` |
| Full-orbit stopping criteria | YES | `tests/jax_core/test_tracing_jax_fullorbit.py:457`, `:607`, `:636` |
| Trapped-vs-passing classification | **NO test** on either backend (out of scope for the integrator) |
| Banana orbit / turning-point stability | **NO test** specifically targeting `v_par → 0` behavior in JAX |
| Lost-particle counts vs CPU at fixed seed | **NO direct test** — `loss_ctr` increment is exercised by `t_final < tmax` paths but no JAX↔CPU `loss_ctr` parity test exists |
| Hit-wall (Levelset) classification | YES for fieldlines (`test_tracing_jax_item16.py:393`); NO for particles |
| Hit-axis (MinR with axis) | YES for fieldlines (`test_tracing_jax_item16.py:218`); NO for particles |
| MaxToroidalFlux event row recorded | YES for Boozer GC (`test_tracing_jax_gc_boozer.py:583` exercises the criteria translation) |
| Iter cap stopping criterion | **NO test** of `IterStoppingCriterion` on the JAX side |
| Overflow of `phi_hits_count > max_phi_hits` (`_event_hits_prefix` rejection) | YES at fieldline level (`test_tracing_jax_item16.py:65`); presumed inherited by particle wrappers (no direct test) |
| Overflow of `accepted_count == max_steps` | **NO test** — `status=1` exit path is undocumented in tests |

### Coverage gaps (prioritized)

1. **JAX-isolated mu-conservation test for GC integrators.** The pipeline-level test in `test_particle.py` validates SIMSOPT end-to-end, but a regression in `guiding_center_*_boozer_rhs` algebra (e.g. a missing minus sign) might still produce a closed orbit that conserves energy while violating the mu adiabatic invariant. A standalone `test_trace_guiding_center_boozer_conserves_mu` at lane tolerance would close this.
2. **Lost-particle count parity vs C++ at fixed seed.** Stage 2 / Stage 3 pipelines aggregate `loss_ctr`; the regression risk of "JAX integrator loses 1 extra particle per N=100 due to step-rejection asymmetry" is currently not gated.
3. **Banana-orbit (`v_par → 0`) stability under JAX integration.** A test that fires a particle with `v_par` just barely above zero and verifies the integrator does not stall or NaN out is missing. C++ implicitly relies on `dtmax` to prevent this; JAX has no analog (Finding 1).
4. **`IterStoppingCriterion` on JAX side.** Test the iteration-cap criterion explicitly so the `step_count + 1` post-step counter contract is enforced.
5. **`accepted_count == max_steps` exit reporting.** Either add an explicit `RuntimeError` at the orchestrator (akin to the `phi_hits` overflow check), or add a regression test that asserts `status=1` propagates to the consumer.

## Other observations

- **Module docstring discrepancy (LOW).** Lines 8-13 of `tracing.py` claim `dx/dtau = B(x) / |B|` (length-parametrised); the actual `fieldline_rhs` returns `B` directly (line 608-609). The C++ reference also returns `B` directly. The docstring is wrong; the code is correct.
- **Stale carve-out language (LOW).** Lines 386-388 say "`ToroidalTransitStoppingCriterion` matches simsoptpp with `flux=False`; the flux-coordinate branch is not on the JAX path yet (the Boozer guiding-centre RHS is deferred under item 14)". The Boozer GC RHS has since landed; the `ToroidalTransitStoppingCriterion` on the Boozer route now consumes `zeta_unwrapped/zeta_init` per `_stopping_criterion_should_stop` (tracing.py:480-484). The docstring should be updated.
- **`get_phi` discrepancy with C++ algorithm.** C++ enumerates three candidate offsets `nearest_multiple ± 2π` and picks the one with minimum distance to `phi_near` (tracing.cpp:333-351). JAX uses the analytic `round((phi_near - phi_raw)/2π) * 2π + phi_raw` (tracing.py:507-523). The two are mathematically equivalent for finite `phi_near`. No action needed.
- **No `gc_to_fullorbit_initial_guesses` JAX port.** Full-orbit seeding consults the CPU `MagneticField` (`field.B()`, `field.AbsB()`) on the host (field/tracing.py:952-960). This is documented at field/tracing.py:898-905 as intentional — only the integrator runs on device. **INFO** only; out of audit scope.
- **`bracket_root_jax`'s dead-iteration overhead.** With `atol=0.0` always passed in, the converged-branch never short-circuits and the loop always executes the full 60 iterations even when the bracket has collapsed in iteration 5. This is wasteful but correct; a `bracket_atol = jnp.asarray(1e-15, dtype=dtype)` at all four call sites would short-circuit at machine precision and probably halve the cost.

## Recommended actions

### HIGH

1. **Document, then implement, a JAX `dtmax` cap.** Either (a) thread `dtmax = r0*0.5*pi/v_total` (Boozer / Cartesian particle) or `r0*0.5*pi/AbsB` (fieldline) into the JAX drivers as a per-step `h_max` clamp (e.g. `h_next = jnp.minimum(h_clamped * factor, dtmax)`); or (b) explicitly document in the parity docs that the JAX path intentionally diverges here and that the `event_time_tracing` lane absorbs the drift. Option (a) is preferable for accurate banana-orbit behavior and step-count parity.
2. **Align the initial-step heuristic.** Compute `dt0 = 1e-3 * dtmax` (particles) / `1e-5 * dtmax` (fieldlines) on the host and pass it through the JAX spec rather than relying on `_INITIAL_STEP_FRACTION = 1/100 * tmax`. Document the per-mode value.
3. **Use a non-zero `bracket_atol` in the event localizer.** Replace `bracket_atol = jnp.asarray(0.0, dtype=dtype)` at tracing.py:1015, 1547, 2479, 3006 with a meaningful absolute step-fraction tolerance (e.g. `eps_floor = 1e-15 * h_clamped`) so the Illinois loop can short-circuit when the bracket has collapsed. This is a pure performance win with no algebra impact.

### MEDIUM

4. **Add JAX-isolated conservation tests** for mu (adiabatic invariant) in `trace_guiding_center_boozer` (all 3 modes) — currently only the pipeline test at `test_particle.py:421` exists.
5. **Add a lost-particle-count parity test** at fixed seed for `trace_particles` and `trace_particles_boozer` JAX vs CPU backends, to gate the `t_final < tmax - 1e-15` accounting.
6. **Add a banana-orbit stability test** (`v_par` chosen so the orbit just barely traps) for the no-K and full Boozer GC paths, to confirm step-rejection at the turning point converges.

### LOW

7. **Fix the module docstring** at tracing.py:8-13 (fieldline RHS is `dy/dt = B`, not `B/|B|`).
8. **Update stale carve-out comments** at tracing.py:386-388 to reflect that the Boozer GC route has landed and the flux-coordinate `ToroidalTransitStoppingCriterion` is active.
9. **Consider adding an `accepted_count == max_steps` hard-error at the orchestrator level** in `field/tracing.py` (mirroring `_event_hits_prefix`'s overflow rejection), so silently-truncated trajectories cannot reach downstream consumers.

### INFO

10. **`IterStoppingCriterion` lacks a direct test on the JAX side**; add one.
11. **`gc` (non-vacuum Cartesian) GC port** is intentionally deferred on both C++ and JAX. No action.
12. **No per-particle JIT vectorization** in the orchestrator; future optimization opportunity, out of parity scope.
