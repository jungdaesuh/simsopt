# Item 09 Math/Physics Invariants

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

## Units and scales

| Quantity | Units | Where realized | Notes |
| --- | --- | --- | --- |
| Lorentz force per unit length (per-quadpoint, pre-MN conversion) | N / m | `_lorentz_force_density_pure` at `src/simsopt/field/force.py:214` | tangents are unit vectors; force density is `I * tangent x B` in SI. |
| Magnetic field on a coil from other coils | T | `_mutual_B_field_at_point_pure` at `force.py:148`; `_B_at_point_from_coil_set_pure` at `force.py:95` | uses `mu_0/(4*pi)` Biot-Savart prefactor at `force.py:41`. |
| Self magnetic field on a regularized coil | T | `B_regularized_pure` (imported from `selffield.py`) | regularization controls singular limit. |
| `LpCurveForce` objective | (MN / m)^p | `lp_force_pure` returns scaled `Lp` integral; conversion factor `1e-6` applied per N -> MN | scalar non-negative if `p > 0`. |
| `SquaredMeanForce` objective | (MN / m)^2 | `squared_mean_force_pure` at `force.py:1696` applies `* 1e-12` to convert from `(N/m)^2` to `(MN/m)^2` | scalar non-negative. |
| Torque per unit length | N * m / m = N | per-coil torque integrand | torque is `r x F` per unit length, in N. |
| `LpCurveTorque` objective | (MN)^p | `lp_torque_pure` returns scaled `Lp` integral | scalar non-negative if `p > 0`. |
| `SquaredMeanTorque` objective | (MN)^2 | `squared_mean_torque` at `force.py:2666` | scalar non-negative. |
| Magnetic energy `B^2` | J (Joule) | `b2energy_pure` returns scalar energy `(1 / 2 mu_0) integral B^2 dV` | Energy is positive. The fork keeps the upstream `b2energy_pure` math; the public class is `B2Energy`. |
| Net external flux | Wb (Weber) | `net_ext_fluxes_pure` at `force.py:1411` | flux through each coil cross-section. |
| Mutual / self inductance | H (Henry) | `_coil_coil_inductances_pure` at `force.py:979` | the docstring already states units explicitly. |

## Sign conventions

- Lorentz force density `f = I * tangent x B`. Sign follows the right-hand rule
  and matches upstream `lp_force_pure` line ordering at `force.py:1685-1686`.
- Net force per coil is `sum(force * gammadash_norm) / npts`; positive
  components correspond to the net Lorentz force in the laboratory frame.
- Torque is `(r - r_center) x F`. Sign convention follows the right-hand rule
  with `r_center` defined as the centroid of the curve.
- Magnetic energy `B^2/(2 mu_0)` is always non-negative.

## Orientation and `stellsym` coverage

- All six wrappers route through `coils_via_symmetries` upstream so the
  expanded coil set covers the full torus. The fork preserves the upstream
  symmetry expansion: `coils_via_symmetries(ncoils=4, nfp=3, stellsym=True)`
  yields `2 * 3 * 4 = 24` coils. The new closeout test exercises exactly this
  configuration and therefore covers `stellsym=True` plus `nfp=3` rotation
  symmetry.
- `stellsym=False` paths are exercised by upstream-style tests in
  `tests/field/test_selffieldforces.py` (e.g., line 1539
  `test_force_and_torque_objectives_with_different_quadpoints` and the
  upstream Taylor sweep). Item 09 does not introduce new `stellsym=False`
  coverage because the parity contract is shared with the existing tests.

## Singular and near-coil regimes excluded

- The mutual `_B_at_point_from_coil_set_pure` kernel uses `exclude_index` to
  skip self-contribution and uses `eps` for numerical regularization at
  `force.py:1031` (and the wrapping coil-self distance term). The new closeout
  fixture stays well clear of any pair coincident with the target coil.
- The self-field path delegates to `B_regularized_pure` from `selffield.py`
  which handles the on-coil singularity via the circ/rect regularization
  documented in `regularization_circ`/`regularization_rect` at
  `src/simsopt/field/selffield.py`. Item 09 fixture uses
  `regularization_circ(0.05)`, well above the singular limit.

## Production scale

- `ncoils_base = 4` and `numquadpoints = 64` in the new closeout test. After
  `coils_via_symmetries(ncoils=4, nfp=3, stellsym=True)` expansion the
  effective coil count is `24` with `numquadpoints=64` each.
- Prompt floor `ncoils >= 4` (base coil count) and `nquadpoints >= 64` are
  both satisfied.
- The existing `tests/field/test_selffieldforces.py::test_force_objectives`
  fixture is `ncoils=4`, `nfp=3`, `stellsym=True`, and default
  `numquadpoints=15*order=90` (24 expanded coils at 90 quadpoints each). The
  new test deliberately picks `numquadpoints=64` to make production scale
  explicit and to avoid drifting fixture defaults.

## Derivative shapes and projection

- `_assemble_curve_current_derivative` at `force.py:781` projects the JAX
  per-coil derivative blocks back through `curve.dgamma_by_dcoeff_vjp`,
  `curve.dgammadash_by_dcoeff_vjp`, `curve.dgammadashdash_by_dcoeff_vjp`, and
  `current.vjp` to produce a `Derivative` instance whose underlying shape
  matches `Optimizable.x.shape[0]` for the active wrapper.
- The closeout test asserts `dJ.shape == (ncoils * len(coil.x),)` which is the
  upstream public derivative shape contract for `Coil`-typed `Optimizable`
  parents.

## Transfer-guard discipline

- All host-to-device staging happens inside `_J_args` ->
  `_CoilStateGroupCache.arrays` -> `_build_shared_coil_state` ->
  `_curve_state_from_entry` at `force.py:593` and
  `_current_value_from_entry` at `force.py:589`. These call sites are outside
  the module-level compiled functions at `force.py:953-976`.
- Cited existing transfer-guard discipline tests:
  - `tests/field/test_selffieldforces.py:158`
    `test_regularization_functions_transform_under_strict_transfer_guard`.
  - `tests/field/test_selffieldforces.py:191`
    `test_b_regularized_pure_jit_vmap_strict_transfer_guard_matches_wrapper`.
  - `tests/field/test_selffieldforces.py:237`
    `test_regularized_coil_self_field_methods_use_strict_transfer_boundary`.
  - `tests/test_jax_import_smoke.py:1127`
    `LpCurveForce` strict-transfer-guard subprocess smoke that runs
    `tests/subprocess/import_smoke_cases.py:1337`.
- The new closeout test exercises the same wrapper under
  `jax.transfer_guard("disallow")` and runs the full pytest invocation under
  `SIMSOPT_JAX_TRANSFER_GUARD=disallow` to confirm that the production-scale
  LpCurveForce path holds no implicit host transfer once the host-side
  `_J_args` boundary has populated the cache.

## Parity contract

- Item 09 parity is **fixed-state scalar value** plus **fixed-state
  directional FD gradient**. Not a final optimizer envelope. Not a
  trajectory-by-trajectory optimizer contract.
- The Taylor parity proof is delivered as a single random direction sampled
  with `numpy.random.default_rng(1729)` (matching the lane direction-seed
  contract) and a central five-step FD with the lane `directional_fd_rtol` /
  `directional_fd_atol`. The lane tolerance values are imported from
  `parity_ladder_tolerances("fd_gradient")` and are not inlined.
