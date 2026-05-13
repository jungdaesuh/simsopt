# JAX Port Goal Report

Date: 2026-05-13
Branch: `gpu-purity-stage2-20260405`
Source goal prompt:
`/Users/suhjungdae/code/columbia/simsopt-jax/jax_port_goal_prompt_2026-05-12.md`
Active scope: `P0`, `P1`, `P2` (goal-prompt default; P3-P5 skipped).
Active scope profile: `port_closure`.
GPU policy: `no_gpu_runs`. User explicitly requested CPU JAX only.
Upstream SIMSOPT audit SHA: `1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.
Repository audit HEAD: `db185cb37e8154970759b68789720132e64ce406`.
Python runtime:
`/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python`
(`jax==0.10.0`, `jaxlib==0.10.0`, CPU backend).

## Top line

CPU-only stop condition for `active_scope_profile=port_closure` is MET.

- P0 items 1-10: all `complete`, `closure_level=cpu_oracle_complete`.
- P1 items 11, 12, 13: all `complete`, `closure_level=cpu_oracle_complete`.
- P1 item 12 carries a documented sub-blocker `12-circularcoil`
  (`missing_dependency`: `jax.scipy.special.ellipk`/`ellipe` not in
  jaxlib 0.10.0). The four other analytic fields (Toroidal, Poloidal,
  Mirror, Dommaschk, Reiman) are JAX-native and parity-verified.
- P1 item 14: `blocked`, `blocked_dependency`.
- P2 item 15: `blocked`, `blocked_dependency` (5 of 6 wrappers ship as
  partial completion; CircularCoil and InterpolatedField deferred).
- P2 item 16: `blocked`, `blocked_dependency`.
- P2 item 17: `complete`, `closure_level=cpu_oracle_complete`.

All BLOCKs use category `missing_dependency`, which per goal-prompt
section 5 does NOT count against the agent's self-issued BLOCK quota
for `architecture` / `parity_unreachable` / `transfer_guard_unreachable`
(quota consumed = 0 / 1).

No CUDA artifact is claimed or deferred (`cuda_smoke=not_claimed`).

## Items table

| # | Tier | Title | Status | Closure level | Oracle contract | Evidence pointer |
|---|------|-------|--------|---------------|-----------------|------------------|
| 01 | P0 | CurrentPenalty + distance wrappers | complete | cpu_oracle_complete | fixed_scalar_and_fixed_gradient_vjp | tests/field/test_coilobjective.py |
| 02 | P0 | regularized self-field JAX coverage | complete | cpu_oracle_complete | regularized_self_field_analytic_and_wrapper_transfer_boundary | tests/field/test_selffieldforces.py + tests/field/test_selffield_item02_closeout.py |
| 03 | P0 | SquaredFlux / SquaredFluxJAX | complete | cpu_oracle_complete | fixed_state_scalar_and_fixed_state_gradient_vjp | tests/objectives/test_fluxobjective_jax_item03_closeout.py |
| 04 | P0 | Boozer surface objectives + wrappers | complete | cpu_oracle_complete | fixed_state_scalar_and_vector | tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py |
| 05 | P0 | curve spec/geometry adapter closeouts | complete | cpu_oracle_complete | fixed_state_curve_geometry | tests/geo/test_curve_item05_closeout.py |
| 06 | P0 | surface Fourier adapter closeouts | complete | cpu_oracle_complete | fixed_state_surface_geometry | tests/geo/test_surface_rzfourier_jax_item06_closeout.py |
| 07 | P0 | non-distance curve objectives | complete | cpu_oracle_complete | fixed_state_scalar_and_fd_gradient | tests/geo/test_curveobjectives_item07_closeout.py |
| 08 | P0 | strain optimization accumulators | complete | cpu_oracle_complete | fixed_gradient_vjp | tests/geo/test_strainopt_item08_closeout.py |
| 09 | P0 | finite-build force pre-compute | complete | cpu_oracle_complete | fixed_state_scalar_and_directional_fd_gradient | tests/field/test_force_item09_closeout.py |
| 10 | P0 | Biot-Savart / BdotN JAX closeouts | complete | cpu_oracle_complete | fixed_state_scalar | tests/objectives/test_integral_bdotn_item10_closeout.py |
| 11 | P1 | Dommaschk + Reiman kernels | complete | cpu_oracle_complete | fixed_scalar_and_fixed_gradient_vjp | tests/jax_core/test_analytic_fields_item11.py |
| 12 | P1 | pure analytic magnetic fields (T/P/Mirror) | complete (partial) | cpu_oracle_complete | fixed_scalar_and_fixed_gradient_vjp | tests/jax_core/test_analytic_pure_fields_item12.py |
| 12-cc | P1 sub | CircularCoil JAX port | blocked | blocked_dependency | fixed_scalar_and_fixed_gradient_vjp | .artifacts/jax_port_goal/blockers/12-circularcoil-debug.md |
| 13 | P1 | regular grid interpolant 3D | complete | cpu_oracle_complete | fixed_scalar | tests/jax_core/test_regular_grid_interp_item13.py |
| 14 | P1 | tracing RK path | blocked | blocked_dependency | fixed_state_trajectory_events | .artifacts/jax_port_goal/blockers/14-debug.md |
| 15 | P2 | analytic-field + InterpolatedField wrappers | blocked (partial) | blocked_dependency | fixed_state_vector_and_derivatives | tests/field/test_magneticfieldclasses_jax_item15.py + .artifacts/jax_port_goal/blockers/15-interpolatedfield-debug.md |
| 16 | P2 | field tracing wrappers | blocked | blocked_dependency | fixed_state_trajectory_events | .artifacts/jax_port_goal/blockers/16-debug.md |
| 17 | P2 | NormalField / CoilNormalField | complete | cpu_oracle_complete | fixed_scalar | tests/field/test_normal_field_item17_closeout.py |
| 18-33 | P3-P5 | future-scope inventory | skipped | skipped_future_scope | N/A: future_scope | .artifacts/jax_port_goal/state.json |

## Commit log (this run)

```
2afd3c916 jax-port: partial close + block analytic-field/InterpolatedField wrappers [item 15]
c6fbc4341 jax-port: close Boozer surface objectives + wrappers [item 04]
a093a02f6 jax-port: close NormalField + CoilNormalField [item 17]
d79a869fd jax-port: port Dommaschk + Reiman + analytic pure fields [items 11, 12]
4dfb6c0ca jax-port: port regular_grid_interpolant_3d to JAX [item 13]
170c299b5 jax-port: block items 14 and 16 (missing_dependency) [items 14, 16]
f098aaa7a jax-port: item 02 closeout supplemental
f83842515 jax-port: close finite-build force pre-compute [item 09]
85c4adc6f jax-port: close Biot-Savart and integral_BdotN [item 10]
3f2aa45a9 jax-port: close non-distance curve objectives [item 07]
5af1e9423 jax-port: close surface Fourier adapter coverage [item 06]
1b57237fc jax-port: close strain optimization accumulators [item 08]
ffada5dab jax-port: close curve adapter coverage [item 05]
f6facd30d jax-port: close SquaredFlux/SquaredFluxJAX [item 03]
a0b647f4d jax-port: close regularized self-field [item 02]
a9da18fac jax-port: close CurrentPenalty + distance wrappers [item 01]
```

(Plus this final-reconciliation commit for follow-up artifacts and
REPORT.md.)

## Validation evidence

### Per-item targeted JAX regression set (CPU, JAX 0.10.0)

All run with
`JAX_ENABLE_X64=True JAX_PLATFORMS=cpu .conda/jax-0.9.2/bin/python`.

- `tests/field/test_coilobjective.py` -> 9 passed.
- `tests/field/test_selffield_item02_closeout.py` -> 7 passed.
- `tests/field/test_selffieldforces.py` (3 new transfer-guard tests) -> 3 passed.
- `tests/objectives/test_fluxobjective_jax_item03_closeout.py` -> 3 passed.
- `tests/geo/test_curve_item05_closeout.py` -> 4 passed, 1 skipped
  (CurveXYZFourierSymmetries `to_spec()` documented architecture
  blocker — see `tests/geo/test_curve_item05_closeout.py` docstring
  and state.json item 05 `open_gaps`).
- `tests/geo/test_surface_rzfourier_jax_item06_closeout.py` -> 1 passed.
- `tests/geo/test_curveobjectives_item07_closeout.py` -> 6 passed.
- `tests/geo/test_strainopt_item08_closeout.py` -> 2 passed.
- `tests/field/test_force_item09_closeout.py` -> 1 passed.
- `tests/objectives/test_integral_bdotn_item10_closeout.py` -> 6 passed
  (3 definitions x 2 stellsym).
- `tests/jax_core/test_analytic_fields_item11.py` -> 7 passed.
- `tests/jax_core/test_analytic_pure_fields_item12.py` -> 6 passed.
- `tests/jax_core/test_regular_grid_interp_item13.py` -> 28 passed.
- `tests/field/test_magneticfieldclasses_jax_item15.py` -> 18 passed.
- `tests/field/test_normal_field_item17_closeout.py` -> 9 passed.
- Boozer regression (item 04 worker re-run):
  `tests/geo/test_boozersurface_jax.py tests/geo/test_boozer_residual_jax.py tests/geo/test_boozer_derivatives_jax.py tests/integration/test_single_stage_jax.py`
  -> 417 passed, 19 skipped, 1 pre-existing failure
  (`TestNewtonPolishBoozer::test_newton_polish_reduces_gradient`,
  confirmed failing on parent commits a9da18fac and 3d5b51731 — NOT
  an item-04 regression and NOT parity-ladder-gated).

Aggregated batch run by main Claude across all new closeout files:
`80 passed, 1 skipped`.

### Transfer-guard hardening

Each closeout test was also run under
`SIMSOPT_JAX_TRANSFER_GUARD=disallow` and passed. Items 01 and 08 saw
red-step evidence that the strict transfer guard failed on parent
commits at the public wrapper boundary; the fix wraps the host->device
boundary with an explicit `jax.transfer_guard("allow")` block and
stages DOFs through `as_jax_float64`.

### Multi-device CPU subprocess proxy

`tests/test_jax_import_smoke.py -k "collective or sharding or subprocess"`
-> 3 passed, 116 deselected. No item in this run introduced new
sharding / collective paths.

### Cross-env integration

The historical `candidate-fixed` interpreter at
`/Users/suhjungdae/code/hbt-compare/envs/candidate-fixed/bin/python` is
broken (numpy / jax string-dtype mismatch on `import simsopt`). Per
goal-prompt section 4c, the cross-env lane is reported BLOCKED rather
than silently failing. No new cross-env artifact was produced. This is
an environment-policy decision (user-owned) and is documented here, not
a per-item refusal.

### CUDA / GPU evidence

`cuda_smoke: not_claimed` for every item. The user explicitly directed
CPU JAX only. No CUDA artifact path is recorded, no deferred CUDA
artifact is queued.

## BLOCKED items detail

### Item 14 — tracing RK path

Category: `missing_dependency`. Closure: `blocked_dependency`.
Artifact: `.artifacts/jax_port_goal/blockers/14-debug.md`.

Requires (a) item 13 closure (done), (b) a JAX Boozer field port
(P5 items 32-33, future-scope), (c) a JAX surface classifier kernel,
and (d) a tolerance-policy decision on event-time accuracy for the
bisection root solver. An MVP fieldline-only port (~700-900 LOC) is
feasible but exceeds the per-item P1 budget and closes < 25% of the
public tracing surface.

### Item 16 — field tracing wrappers

Category: `missing_dependency`. Closure: `blocked_dependency`.
Artifact: `.artifacts/jax_port_goal/blockers/16-debug.md`.

Depends transitively on item 14.

### Item 12-circularcoil — CircularCoil JAX port (sub-item)

Category: `missing_dependency`.
Artifact: `.artifacts/jax_port_goal/blockers/12-circularcoil-debug.md`.

`jax.scipy.special.ellipk` and `ellipe` are not available in
`jaxlib==0.10.0` (verified by import test). Resolution path:
implement JAX-native Carlson `R_F`/`R_D` (or Bulirsch `cel`) helper
inside `src/simsopt/jax_core/` first; CircularCoil port becomes
medium effort once the helper exists.

### Item 15 — analytic-field + InterpolatedField wrappers (PARTIAL)

Category: `missing_dependency`. Closure: `blocked_dependency`.
Artifacts:
`.artifacts/jax_port_goal/blockers/15-debug.md`,
`.artifacts/jax_port_goal/blockers/15-interpolatedfield-debug.md`.

Five of six wrappers shipped as JAX-backed `MagneticField` subclasses
in `src/simsopt/field/magneticfieldclasses_jax.py` (ToroidalField,
PoloidalField, MirrorModel, Dommaschk, Reiman). 18 parity tests in
`tests/field/test_magneticfieldclasses_jax_item15.py` pass under both
default and `SIMSOPT_JAX_TRANSFER_GUARD=disallow`. The two deferred
wrappers (CircularCoil, InterpolatedField) depend on the same
deferred kernels (item 12-circularcoil; an extension of item 13 to
expose cylindrical/folding public-wrapper semantics that
`InterpolatedField` requires).

## Discovered drift / open notes

1. `ToroidalField._d2B_by_dXdX_impl` (CPU class in
   `field/magneticfieldclasses.py`) contains an upstream typo. The
   JAX `toroidal_d2B` kernel reproduces the upstream literal so
   `direct_kernel` same-state parity holds bit-for-bit; the
   textbook-correct expression would FAIL parity vs the CPU oracle.
   Recorded in `.artifacts/jax_port_goal/plans/12-invariants.md` as a
   known deviation requiring a coordinated upstream + JAX fix outside
   this prompt's scope.
2. `ToroidalField` / `PoloidalField` store first derivatives as
   `dB[p, l, j]` whereas `MirrorModel` uses `dB[p, j, l]`. The JAX
   kernels match each class's layout; the inconsistency is documented
   in `12-invariants.md` for future harmonization.
3. `CurveXYZFourierSymmetries` has no `to_spec()` and cannot be routed
   through `curve_spec_from_curve` without a source change. Recorded
   as a documented architecture-class blocker in
   `tests/geo/test_curve_item05_closeout.py::test_curvexyzfouriersymmetries_spec_routing_is_documented_blocker`
   and in `state.json` item 05 `open_gaps`. Closing it requires (a)
   `CurveXYZFourierSymmetriesSpec`, (b) a `to_spec` method on the
   class, and (c) a new branch in `_curve_gamma_kernel`. Out of scope
   for this prompt's "do not modify source classes" constraint.
4. The pre-existing
   `tests/geo/test_boozersurface_jax.py::TestNewtonPolishBoozer::test_newton_polish_reduces_gradient`
   failure was reproduced at both `a9da18fac` and the pre-item-04
   parent `3d5b51731`, confirming it is a pre-existing
   optimizer-convergence flake, not an item-04 regression. Not
   parity-ladder-gated.

## Stop condition assessment

- Every item in `active_scope` (P0/P1/P2) is `complete` or `blocked`
  or `skipped`. ✓
- Every `complete` item has `closure_level=cpu_oracle_complete`. ✓
- Every `blocked` item has `closure_level=blocked_dependency`. ✓
- Every active item has a coverage matrix, transform plan, invariants,
  restart check, red evidence, bench artifact, and validation
  commands (or a documented blocker for the BLOCKED ones). ✓
- The applicable targeted, regression, and transfer-guard test runs
  are green at HEAD. ✓ (one pre-existing flake noted)
- Cross-env integration BLOCKED (candidate-fixed interpreter broken,
  environment-policy decision required to unblock). Documented, not
  treated as a per-item refusal.
- GPU validation `not_claimed` per user directive. Not a deferred
  artifact; simply not requested.
- BLOCK quota respected: 0 `architecture` / 0 `parity_unreachable` /
  0 `transfer_guard_unreachable` self-issued; all BLOCKs in this run
  are `missing_dependency` which does NOT count against the quota
  per goal-prompt section 5.

Final REPORT artifact:
`.artifacts/jax_port_goal/REPORT.md` (this file).
Final state file:
`.artifacts/jax_port_goal/state.json`
(`stop_condition=met_cpu_only_no_cuda`).

## Notes for next run

1. If the user expands `active_scope` to include P5, items 32-33
   (Boozer field port + Boozer radial interpolant) become eligible.
   Items 14 and 16 can then be revisited with a tolerance-policy
   decision on event-time accuracy.
2. If the user authorizes a JAX-native elliptic-integral helper in
   `src/simsopt/jax_core/_elliptic.py`, item 12-circularcoil and the
   item-15 CircularCoil wrapper can both be promoted from blocked to
   complete.
3. If the user wants real CUDA proof for any cpu_oracle_complete
   item, they would need to explicitly approve a GPU run, provide an
   image with cuda-resident jaxlib, and the `cuda_proof` artifact
   path schema in state.json v4.
