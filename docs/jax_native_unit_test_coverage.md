# JAX Coverage of Native SIMSOPT Unit-Test Capabilities

**Generated file — do not hand-edit.** Regenerate with `python scripts/jax_native_unit_coverage.py --write`.

Coverage contract: docs/jax_native_unit_test_coverage_implementation_plan.md, commit 6fec6e4ca. That plan (Draft, 2026-07-29) owns the schema, the disposition vocabulary, and the fail-closed rules; this document reports them.

Executing wave: docs/jax_native_test_mirror_wave_implementation_plan.md, commit 2221b542a, Implementation Plan item 6.

## Baseline

- Fork baseline commit: `7781d707a51c6e83173a76bf80b2d51646d65fdc`
- Upstream authority commit: `377cf665158f47a9bed4a8b03a00352457ea27c8` (merge base of this fork's HEAD with hiddenSymmetries/simsopt master; every file it lists is present in the working tree by construction)
- Native test surface: 70 files
- Source-tree hash: `sha256:25f057c4fc97465ff59a758e88b8d2ec3b0a39f73e49f21e1ca628f023beedfb`
- Hash recipe: sha256 over, for each enumerated native test file in ascending path order, the utf-8 encoding of '{path}\n{sha256_hexdigest(working tree file bytes)}\n'.

## How to read this report

- Rows are **file-level**. The 2026-07-29 plan's full per-function ledger over every native test definition is deferred to its own later phases and is not claimed here.
- `unclassified` is a valid planning state: the file is listed, and it is **never** counted as covered.
- `jax_partial` and `jax_missing` are valid planning states that fail final completion. They are **not** converted into a percent-covered claim, per the contract.
- A classified file row's `reason` states which capabilities of that file this slice enumerated; anything it does not name is still open.
- 1 capability record(s) reach `jax_equivalent`: native CPU, JAX CPU and strict JAX GPU evidence for every declared observable.

## Native test files by disposition

| Disposition | Files |
| --- | --- |
| `hybrid_boundary` | 1 |
| `jax_missing` | 1 |
| `jax_partial` | 5 |
| `native_only` | 11 |
| `unclassified` | 52 |
| **total** | **70** |

*Scoped rollup: 2 of the 18 classified rows above enumerate only PART of their file's native test functions (see each row's `reason` for exactly which capabilities it covers), so 18/70 classified/total must not be read as "18 files fully covered":*
- `tests/geo/test_curve.py`
- `tests/geo/test_surface_objectives.py`

## Capability records by domain and disposition

| Domain | Disposition | Capabilities |
| --- | --- | --- |
| coilset | `native_only` | 1 |
| curveperturbed | `jax_missing` | 3 |
| curveperturbed | `jax_partial` | 1 |
| curves | `jax_missing` | 2 |
| curves | `jax_partial` | 1 |
| force | `jax_equivalent` | 1 |
| force | `jax_missing` | 3 |
| force | `jax_partial` | 2 |
| force | `native_only` | 1 |
| force | `shared_python` | 1 |
| fourier_interpolation | `native_only` | 1 |
| mgrid | `native_only` | 1 |
| normal_field | `native_only` | 1 |
| ports | `hybrid_boundary` | 1 |
| quasisymmetry | `jax_partial` | 1 |
| spec | `native_only` | 1 |
| strain | `jax_missing` | 1 |
| strain | `jax_partial` | 2 |
| virtual_casing | `native_only` | 1 |
| vmec | `native_only` | 1 |
| **total** | | **27** |

## Dated decisions

| Capability | Disposition | Proposed | Proposed by | Frozen | By |
| --- | --- | --- | --- | --- | --- |
| CS-1 — CoilSet and ReducedCoilSet | `native_only` | 2026-08-23 | 2026-08-23 mirror-wave plan (docs/jax_native_test_mirror_wave_implementation_plan.md, commit 2221b542a), unit 6 | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| FI-1 — fourier_interpolation utility | `native_only` | 2026-08-23 | 2026-08-23 mirror-wave plan (docs/jax_native_test_mirror_wave_implementation_plan.md, commit 2221b542a), unit 6 | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| MF-1 — Self-field reduced model kernels | `shared_python` | 2026-08-24 | unit 1 slice (shared-implementation rule: a jax.numpy module imported directly by both lanes needs no duplicate port) | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| MF-6 — Objective wall-clock timing and error-decay plotting sweep | `native_only` | 2026-08-24 | unit 1 slice (performance-measurement-ownership rule: wall-clock timing and plotting sweeps belong to benchmarks/, not parity tests) | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| MG-1 — MGrid file reader and writer | `native_only` | 2026-08-24 | unit 6 slice (project exclusion rule: VMEC-, SPEC-, third-party-dependent code needs no mirror) | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| NF-1 — NormalField and CoilSet-derived normal field | `native_only` | 2026-08-24 | unit 6 slice (project exclusion rule: VMEC-, SPEC-, third-party-dependent code needs no mirror) | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| PT-1 — PortSet collision geometry and wireframe segment constraints | `hybrid_boundary` | 2026-08-23 | 2026-08-23 mirror-wave plan (docs/jax_native_test_mirror_wave_implementation_plan.md, commit 2221b542a), unit 6 | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| SP-1 — SPEC equilibrium driver | `native_only` | 2026-08-24 | unit 6 slice (project exclusion rule: VMEC-, SPEC-, third-party-dependent code needs no mirror) | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| VC-1 — Virtual casing principle | `native_only` | 2026-08-24 | unit 6 slice (project exclusion rule: VMEC-, SPEC-, third-party-dependent code needs no mirror) | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |
| VM-1 — VMEC equilibrium driver and diagnostics | `native_only` | 2026-08-24 | unit 6 slice (project exclusion rule: VMEC-, SPEC-, third-party-dependent code needs no mirror) | 2026-08-24 | session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending |

## Capability records

### coilset

- **CS-1 — CoilSet and ReducedCoilSet**
  - disposition: `native_only`
  - reason: The sole in-repo consumer of the CoilSet class is simsopt.field.normal_field, which belongs to the SPEC family and is excluded by the project rule. No example and no JAX-side module imports it, and the JAX stage-two objectives express coil collections in their own grouped-spec idiom. Porting 647 LOC with zero JAX-side consumers is YAGNI. Revisit only if a JAX example adopts the CoilSet API.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/field/coilset.py:18` (anchor: `class CoilSet(Optimizable):`), `src/simsopt/field/coilset.py:383` (anchor: `class ReducedCoilSet(CoilSet):`), `src/simsopt/field/normal_field.py:7` (anchor: `from .coilset import CoilSet`), `src/simsopt/field/normal_field.py:549` (anchor: `self._coilset = CoilSet()`)
  - tolerance owner: `not_applicable`

### curveperturbed

- **CP-1 — CurvePerturbed.resample in-place redraw**
  - disposition: `jax_missing`
  - reason: Native CurvePerturbed.resample redraws the perturbation in place and invalidates caches. The JAX reformulation materializes samples into a frozen dataclass whose fields are set once in __post_init__ and fingerprinted, so in-place redraw is structurally impossible by design; a new draw means a new bundle.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt_jax/examples/stochastic_samples.py:35-91` (anchor: `class StochasticPerturbationBundle:`)
  - blocker: Immutability is the reformulation's design premise; adding mutation would break the fingerprint contract the stochastic lane relies on.
  - tolerance owner: `not_applicable`
- **CP-2 — CurvePerturbed GSON serialization round trip**
  - disposition: `jax_missing`
  - reason: The native test round-trips a CurvePerturbed through the GSON object graph. The JAX bundle carries plain FP64 tensor data plus a SHA-256 fingerprint and is not GSONable, so there is no equivalent graph round trip; identity is proven by the fingerprint instead.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt_jax/examples/stochastic_samples.py:98` (anchor: `def _fingerprint(self) -> str:`)
  - JAX-side tests: `tests/jax/examples/test_stochastic_samples.py::test_materialized_sample_hash_binds_values_and_metadata`
  - blocker: Serialization formats are an explicit non-goal of the coverage contract; the fingerprint is the JAX-side identity contract.
  - tolerance owner: `not_applicable`
- **CP-3 — LpCurveTorsion evaluated through perturbed curves**
  - disposition: `jax_missing`
  - reason: torsion_pure needs the first three curve derivatives. The JAX perturbation bundle carries gamma and gammadash only, so a torsion objective cannot be evaluated through it without extending the bundle to second and third derivatives.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt_jax/objectives/stochastic_stage_two.py:16-35` (anchor: `class StochasticCoilPerturbations`), `src/simsopt/geo/curveobjectives.py:137-154` (anchor: `def Lp_torsion_pure(torsion, gammadash, p, threshold):`)
  - blocker: Widening StochasticCoilPerturbations to carry d2/d3 is a production change with a real memory cost; no JAX example needs it today.
  - tolerance owner: `not_applicable`
- **CP-4 — Curve-curve distance objective through perturbed curves**
  - disposition: `jax_partial`
  - reason: The distance objective evaluated on the materialized bundle matches the native CurveCurveDistance through nested CurvePerturbed wrappers to 5.7e-16 relative, which is jax_equivalent evidence on the native-CPU and JAX-CPU lanes. The stochastic sample tests are not parity_lane-parametrized, so there is no GPU lane at all here; the row stays jax_partial under the contract's strict definition. Native's own test_perturbed_objective_distance also Taylor-tests the objective's derivative (dJ = J.dJ(), shrinking finite-difference steps); test_perturbed_curve_distance_objective_matches_native_through_bundle reproduces only the scalar J() value, because the materialized bundle exposes no curve-DOF gradient graph to Taylor-test dJ() against, so that thresholded-gradient path is uncovered here too.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=pending
  - evidence: `tests/jax/examples/test_stochastic_samples.py` (anchor: `def test_perturbed_curve_distance_objective_matches_native_through_bundle`), `tests/jax/examples/test_stochastic_samples.py` (anchor: `Measured relative error 5.7e-16 (pure recombination of the same terms`), `src/simsopt_jax/core/curve_kernels.py:65` (anchor: `def curve_curve_distance_penalty_pure(`)
  - JAX-side tests: `tests/jax/examples/test_stochastic_samples.py::test_perturbed_curve_distance_objective_matches_native_through_bundle`, `tests/jax/examples/test_stochastic_samples.py::test_materialized_samples_reproduce_native_perturbed_coil_geometry`, `tests/jax/examples/test_stochastic_samples.py::test_materialized_gammadash_matches_finite_difference_of_gamma`, `tests/jax/examples/test_stochastic_samples.py::test_materialized_samples_preserve_periodicity_of_perturbation`
  - blocker: No GPU lane exists for tests/jax/examples/test_stochastic_samples.py; adding parity_lane parametrization is a filed follow-up.
  - tolerance owner: `local_test_tolerances`

### curves

- **CV-1 — Named Fourier coefficient accessors on curve subclasses**
  - disposition: `jax_missing`
  - reason: Native CurveRZFourier and CurveHelical expose named DOF access (.rc/.rs/.zc/.zs arrays, set('rc(i)'), get('A_0')) backed by local_dof_names. The JAX curve kernels take a flat dofs vector with positional slicing and publish no name-to-index mapping, so the named-accessor contract has no JAX counterpart. The bound tests' geometry comparisons (via _assert_spec_geometry_matches_native) use module-local constants (_GAMMA_RTOL=1e-10, _DERIV_RTOL=1e-9) declared in test_curve_subclasses_parity.py itself; that file never imports src/simsopt_jax/parity_tolerances.py, so the tolerance owner is local, not not_applicable -- a real numeric comparison is made, it is just not centrally owned.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `tests/geo/test_curve.py:995-1019`, `tests/geo/test_curve_helical.py:7-30`, `src/simsopt_jax/core/curve_rz_fourier.py:14-20` (anchor: `def curverzfourier_pure(dofs, quadpoints, order, nfp, stellsym):`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_curve_subclasses_parity.py::test_curve_rzfourier_dof_round_trip_matches_native`, `tests/jax/native_unit_parity/test_curve_subclasses_parity.py::test_curve_helical_dof_round_trip_matches_native`
  - blocker: Named-DOF plumbing belongs to the Optimizable layer, which the JAX kernels deliberately do not own; the mirror pins the flat-vector round trip instead.
  - tolerance owner: `local_test_tolerances`
- **CV-2 — CurvePlanarFourier near-zero-quaternion regularization**
  - disposition: `jax_partial`
  - reason: Native regularizes a degenerate quaternion as q/(norm+1e-8), which keeps the direction; the JAX kernel returns the zero quaternion instead. The two lanes therefore diverge behaviorally in a non-physical degenerate case that neither test suite currently pins. Ordinary (normalized) quaternions agree.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=passing
  - evidence: `src/simsopt/geo/curveplanarfourier.py:116-119` (anchor: `q / (norm_q + 1e-8)`), `src/simsopt_jax/core/curve_planar_fourier.py:14-21` (anchor: `jnp.where(norm_sq > zero, normalized, zero_quaternion)`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_curve_subclasses_parity.py::test_curve_planarfourier_pure_kernel_matches_native_position`, `tests/jax/native_unit_parity/test_curve_subclasses_parity.py::test_curve_planarfourier_position_and_derivatives_match_native`
  - blocker: Deciding which degenerate-quaternion convention is correct is a numerics ruling, not a test gap; recorded here so the divergence is visible rather than discovered by a future caller.
  - tolerance owner: `local_test_tolerances`
- **CV-3 — RotatedCurve wrapping a direct curve subclass**
  - disposition: `jax_missing`
  - reason: curve_spec_from_curve refuses RotatedCurve by design and says so in its own docstring: rotation/reflection placement is a wrapper transform with no owned DOFs, routed through CoilSymmetrySpec instead. Standalone rotated-curve geometry stays a documented CPU-only wrapper, so there is no JAX spec to mirror.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt_jax/core/curve_geometry.py:196-214` (anchor: `rotation/reflection placement is a wrapper transform with no owned DOFs.`)
  - blocker: Representing RotatedCurve as a CurveSpec would duplicate placement knowledge that CoilSymmetrySpec already owns.
  - tolerance owner: `not_applicable`

### force

- **MF-1 — Self-field reduced model kernels**
  - disposition: `shared_python`
  - reason: The reduced self-field model (_rectangular_xsection_k/_delta, regularization_circ/rect, B_regularized_pure) has exactly one jax.numpy implementation, in simsopt.field.selffield, and the JAX force adapter imports B_regularized_pure from it rather than reimplementing it. The algebra is backend-independent Python exercised once by both lanes; a duplicate JAX port would create a second source of truth for the same formula. Every comparison in the bound tests uses a literal tolerance declared in test_force_parity.py itself (rtol=1e-10, rtol=1e-3, rtol=1e-6, rtol=1e-12), never src/simsopt_jax/parity_tolerances.py, so the tolerance owner is local, not central. test_regularizations_stay_traceable_under_jit_and_grad exercises jit and grad on these kernels but never vmap, so jit_vmap_autodiff_compatibility is not claimed as a covered observable here.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=passing
  - evidence: `src/simsopt/field/selffield.py:18` (anchor: `import jax.numpy as jnp`), `src/simsopt/field/selffield.py:23` (anchor: `__all__ = ['B_regularized_pure', 'regularization_rect', 'regularization_circ']`), `src/simsopt_jax_adapters/field/force.py:22` (anchor: `from simsopt.field.selffield import B_regularized_pure`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_force_parity.py::test_rectangular_xsection_k_matches_published_square_value`, `tests/jax/native_unit_parity/test_force_parity.py::test_rectangular_xsection_delta_matches_published_square_value`, `tests/jax/native_unit_parity/test_force_parity.py::test_rectangular_xsection_functions_are_symmetric_in_a_and_b`, `tests/jax/native_unit_parity/test_force_parity.py::test_rectangular_xsection_functions_match_thin_strip_limits`, `tests/jax/native_unit_parity/test_force_parity.py::test_regularization_circ_matches_closed_form_and_scales_quadratically`, `tests/jax/native_unit_parity/test_force_parity.py::test_regularization_rect_is_area_times_delta_and_symmetric`, `tests/jax/native_unit_parity/test_force_parity.py::test_regularizations_stay_traceable_under_jit_and_grad`
  - tolerance owner: `local_test_tolerances`
- **MF-2 — RegularizedCoil force and torque vector methods**
  - disposition: `jax_missing`
  - reason: Native RegularizedCoil exposes a coil-method layer returning force and torque VECTORS per quadrature point (B_regularized, self_force, force, torque, net_force, net_torque). The JAX adapter's public surface publishes magnitudes and integrated objectives only, so no JAX API returns the vector fields the native layer promises.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/field/coil.py:119-231` (anchor: `def B_regularized(self):`), `src/simsopt_jax_adapters/field/force.py:55-67` (anchor: `"curve_force_norms_pure",`)
  - blocker: Adding a per-quadrature-point force/torque vector API to src/simsopt_jax_adapters/field/force.py is a production-surface change outside this wave's scope; it needs its own capability review.
  - tolerance owner: `not_applicable`
- **MF-3 — Circular and rectangular RegularizedCoil subclasses**
  - disposition: `jax_missing`
  - reason: CircularRegularizedCoil and RectangularRegularizedCoil build the regularization from the cross-section dimensions. The JAX adapter has no subclass equivalent: it reads .regularization off native instances, so subclass construction and its dof plumbing remain native-owned behavior with no JAX counterpart.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/field/coil.py:253` (anchor: `class CircularRegularizedCoil(RegularizedCoil):`), `src/simsopt/field/coil.py:272` (anchor: `class RectangularRegularizedCoil(RegularizedCoil):`), `src/simsopt_jax_adapters/field/force.py:1379-1380` (anchor: `self.regularizations = _as_jax_float64(`), `src/simsopt_jax_adapters/field/force.py:2290` (anchor: `self.regularizations = _as_jax_float64([c.regularization for c in target_coils])`)
  - blocker: A JAX regularized-coil subclass would duplicate the native constructor; the adapter deliberately consumes native instances instead. Revisit only if a JAX example needs to build one.
  - tolerance owner: `not_applicable`
- **MF-4 — Regularization guard when force methods are called on a plain Coil**
  - disposition: `jax_missing`
  - reason: The native AttributeError guard belongs to the MF-2 vector-method layer, which has no JAX counterpart, so the guard itself cannot be mirrored. The adapter's own ValueError analogue on the objective constructors IS mirrored, so the failure contract is covered everywhere a JAX API actually exists: test_self_field_objectives_require_regularized_coils exercises that substitute guard on jax_cpu (and would on jax_gpu once a CUDA lane runs). That passing run is evidence for the substitute ValueError contract, not for the missing native AttributeError guard itself, so both jax lanes on this row are recorded not_applicable rather than passing/pending.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/field/coil.py:119-231` (anchor: `def B_regularized(self):`), `src/simsopt_jax_adapters/field/force.py:1375-1376` (anchor: `raise ValueError("B2Energy can only be used with RegularizedCoil objects")`), `src/simsopt_jax_adapters/field/force.py:2286-2289` (anchor: `"LpCurveForce can only be used with RegularizedCoil objects"`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_force_parity.py::test_self_field_objectives_require_regularized_coils`
  - blocker: Blocked by MF-2: the guarded methods do not exist on the JAX lane.
  - tolerance owner: `not_applicable`
- **MF-5 — HSX coil-1 F_x component against the CoilForces.jl benchmark**
  - disposition: `jax_partial`
  - reason: The native test pins the x-component of the self-force vector against CoilForces.jl for circular and rectangular cross sections. The JAX lane exposes the force norm only, so the mirror compares curve_force_norms_pure against the native self-force magnitudes on the same HSX coil; the per-component benchmark stays native-only. The comparison uses a literal rtol=1e-12 declared in test_force_parity.py itself, not src/simsopt_jax/parity_tolerances.py, so the tolerance owner is local, not central.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=passing
  - evidence: `tests/field/test_selffieldforces.py:482-510`, `src/simsopt_jax_adapters/field/force.py:2006` (anchor: `def curve_force_norms_pure(`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_force_parity.py::test_jax_force_norms_on_hsx_coil_match_native_self_force`
  - blocker: Per-component parity requires the MF-2 vector API.
  - tolerance owner: `local_test_tolerances`
- **MF-6 — Objective wall-clock timing and error-decay plotting sweep**
  - disposition: `native_only`
  - reason: test_objectives_time measures wall-clock time and draws a matplotlib figure. Performance measurement in this repository is owned by benchmarks/ under the measurement contract (matched inputs, warmup policy, synchronization, provenance); mirroring a bare timing loop into a parity test would mint an unprovenanced speed signal.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `tests/field/test_selffieldforces.py:1154`, `tests/field/test_selffieldforces.py:1156`
  - tolerance owner: `not_applicable`
- **MF-7 — Taylor-test outer configuration sweep**
  - disposition: `jax_partial`
  - reason: The native Taylor test sweeps use_jax_curve (jax_flag_list), nfp, threshold, downsample and both regularization types for every objective. The JAX mirror runs one configuration per objective and covers the remaining axes indirectly through value/gradient parity and the curve-subclass mirror, so the full cross product is not reproduced on the JAX lane. Only test_force_and_torque_objective_gradients_match_native draws its rtol from src/simsopt_jax/parity_tolerances.py (via _OBJECTIVE_GRADIENT_RTOL = parity_ladder_tolerances("derivative_heavy")["first_derivative_rtol"]); test_taylor_test_confirms_jax_objective_gradients and test_downsampled_objectives_track_full_resolution use literal tolerances declared in the test file. Since not all of this row's bound tests draw their tolerances from the central module, the owner is recorded local, not central.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=passing
  - evidence: `tests/field/test_selffieldforces.py:1017`, `tests/field/test_selffieldforces.py:1033-1044`, `tests/jax/native_unit_parity/test_force_parity.py:796` (anchor: `LpCurveTorque ds=2 drifts ~9.3e-5 (a tight ~8% margin against its`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_force_parity.py::test_taylor_test_confirms_jax_objective_gradients`, `tests/jax/native_unit_parity/test_force_parity.py::test_force_and_torque_objective_gradients_match_native`, `tests/jax/native_unit_parity/test_force_parity.py::test_downsampled_objectives_track_full_resolution`
  - blocker: Reproducing the full axis cross product would multiply mirror runtime; the axes are individually covered, the product is not.
  - tolerance owner: `local_test_tolerances`
- **MF-8 — curve_force_norms_pure, a JAX-only public export**
  - disposition: `jax_equivalent`
  - reason: curve_force_norms_pure has no native counterpart: it is the JAX lane's public force-magnitude kernel. It is validated on CPU by analytic hoop force, quadrature-resolution convergence, native self_force with mutual sources, native-shape/finiteness, and a Stage-II diagnostics LP-force reconstruction, which is jax_equivalent evidence on the native-CPU and JAX-CPU lanes; every one of these comparisons uses a literal tolerance declared in its own test file (e.g. rtol=1e-10, an absolute reference_budget, rtol=1e-8, rtol=1e-12), never src/simsopt_jax/parity_tolerances.py, so the tolerance owner is local, not central. Three of these tests (the two bound above plus test_regularized_coil_force_quantities_have_native_shapes_and_are_finite) live in tests/jax/native_unit_parity/test_force_parity.py, which every test in the file runs under the parity_lane fixture (conftest.py:512-514, params cpu/gpu) and would exercise its GPU case automatically once CUDA is present -- under the DEFAULT backend mode, not strict SIMSOPT_BACKEND_MODE=jax_gpu_parity, which classifies this file adapter_boundary (conftest.py:596-598) and skips it entirely. The remaining two tests (test_public_force_norms_reconstruct_lp_force_objective and test_force_stage_two_diagnostics_slice_on_device_under_transfer_guard) live in tests/jax/objectives/test_force_stage_two.py, which is NOT parity_lane-parametrized and so has no GPU lane at all, strict or otherwise. The 5090 box is fenced by the concurrent nested-LS B37 campaign, so this wave validated CPU only; under the contract's strict definition of jax_equivalent (native CPU + JAX CPU + strict JAX GPU evidence) the row stays jax_partial until a CUDA-present run lands via that gpu_parity route. GPU lane discharged 2026-08-24: gpu_parity lane green on the RTX 5090 (force suite 75/75, JAX_PLATFORMS=cuda,cpu with the CUDA venv; the default-platform pytest invocation silently falls back to CPU and must not be used as GPU evidence).
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=passing
  - evidence: `src/simsopt_jax_adapters/field/force.py:2006` (anchor: `def curve_force_norms_pure(`), `src/simsopt_jax_adapters/field/force.py:62` (anchor: `"curve_force_norms_pure",`), `tests/conftest.py:512-514` (anchor: `ids=("cpu_parity", "gpu_parity")`), `tests/conftest.py:596-598` (anchor: `"jax/native_unit_parity/test_force_parity.py",`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_force_parity.py::test_jax_self_force_on_circular_coil_matches_analytic_hoop_force`, `tests/jax/native_unit_parity/test_force_parity.py::test_jax_force_norms_converge_with_quadrature_resolution`, `tests/jax/native_unit_parity/test_force_parity.py::test_jax_force_norms_match_native_coil_force_with_mutual_sources`, `tests/jax/native_unit_parity/test_force_parity.py::test_regularized_coil_force_quantities_have_native_shapes_and_are_finite`, `tests/jax/objectives/test_force_stage_two.py::test_public_force_norms_reconstruct_lp_force_objective`, `tests/jax/objectives/test_force_stage_two.py::test_force_stage_two_diagnostics_slice_on_device_under_transfer_guard`
  - tolerance owner: `local_test_tolerances`

### fourier_interpolation

- **FI-1 — fourier_interpolation utility**
  - disposition: `native_only`
  - reason: 54 LOC with zero in-repo consumers: the only occurrence of the name outside its own module is the definition itself. Porting a utility nothing calls would add a JAX surface with no caller and no way to keep it honest.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/util/fourier_interpolation.py:16` (anchor: `def fourier_interpolation(fk, x):`)
  - tolerance owner: `not_applicable`

### mgrid

- **MG-1 — MGrid file reader and writer**
  - disposition: `native_only`
  - reason: MGrid reads and writes the NetCDF mgrid file format for free-boundary VMEC. File-format behavior is an explicit non-goal of the coverage contract, and the consumer is VMEC, excluded by the project rule.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/field/mgrid.py:22` (anchor: `class MGrid():`), `src/simsopt/field/mgrid.py:24` (anchor: `This class reads and writes mgrid (NetCDF) files`)
  - tolerance owner: `not_applicable`

### normal_field

- **NF-1 — NormalField and CoilSet-derived normal field**
  - disposition: `native_only`
  - reason: NormalField is the SPEC free-boundary computational-boundary representation, stores its harmonics in the SPEC convention, and loads through py_spec. It is SPEC-family code, excluded by the project rule that no JAX mirror is required for VMEC-, SPEC-, or third-party-dependent behavior.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/field/normal_field.py:12` (anchor: `import py_spec`), `src/simsopt/field/normal_field.py:23` (anchor: `computational boundary of SPEC free-boundary`), `src/simsopt/field/normal_field.py:101` (anchor: `def from_spec(cls, filename):`)
  - tolerance owner: `not_applicable`

### ports

- **PT-1 — PortSet collision geometry and wireframe segment constraints**
  - disposition: `hybrid_boundary`
  - reason: Port-collision geometry is one-shot host-side work with no gradient and no GPU value: it runs once to decide which wireframe segments are constrained, then the JAX RCLS lane consumes the resulting free segment index set. Porting 1032 LOC of collision geometry would buy nothing, so the boundary is retained and tested instead. The wireframe grid is 12x22 at every scale _scale_configuration defines (only the plasma quadrature varies by scale), so the JAX lane's consumed index set is the exact complement of the native wireframe's constrained segments (528 segments, 31 constrained, 497 free) at both scales, not just "bounded". The boundary is now verified two ways: an independent collision oracle re-derives the constrained-segment set from public wireframe geometry and a freshly built PortSet.collides predicate (never calling constrained_segments()/unconstrained_segments() internally, unlike the prior version of this test), and a second test calls the real rcls_wireframe_jax adapter, compares every free-segment current with an independent native reduced-system solve, requires every native reference free current to be nonzero so a dropped free index cannot pass, and confirms all constrained currents remain zero. Nothing in either test batches or broadcasts, so batching_broadcasting_edge_cases is not claimed as a covered observable.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=not_applicable
  - evidence: `examples/jax/parity/cases/native_wireframe_rcls_with_ports.py:41-110` (anchor: `def _ports_on_surface(surface):`), `src/simsopt_jax_adapters/solve/wireframe.py:314` (anchor: `free_segs_host = np.asarray(wframe.unconstrained_segments(), dtype=np.intp)`), `src/simsopt/geo/wireframe_toroidal.py:773-831` (anchor: `def constrained_segments(self, include='all', update=True):`), `src/simsopt/geo/ports.py` (anchor: `class PortSet(object):`)
  - JAX-side tests: `tests/integration/test_jax_rcls_ports_boundary.py::test_native_constrained_segments_match_independent_collision_oracle`, `tests/integration/test_jax_rcls_ports_boundary.py::test_jax_rcls_solution_currents_match_native_reduced_system`
  - tolerance owner: `local_test_tolerances`

### quasisymmetry

- **QS-1 — Non-quasi-symmetric residual ratio kernel**
  - disposition: `jax_partial`
  - reason: The JAX kernel exists and is consumed in production by NonQuasiSymmetricRatioJAX and by the single-stage fullspace objective, but it owns no dedicated unit test. The only tests that name NonQuasiSymmetricRatioJAX build it with object.__new__ and patched attributes to check gradient plumbing, which the coverage contract explicitly refuses as scientific parity evidence, and no test file contains both the native and the JAX class. Scientific reach is therefore example-level only, through the BoozerQA and single-stage Boozer-vacuum parity mirrors. test_exact_boozerqa_workflow_matches_native_and_jax_cpu's tolerances (rtol=1e-10/atol=1e-12, rtol=1e-3/atol=1e-8, rtol=0.0/atol=2e-3) are numerically identical to entries in src/simsopt_jax/parity_tolerances.py, but the test hardcodes them as literals and never imports or calls parity_ladder_tolerances, so the owner is recorded local, not central: a future change to the central ladder would not automatically flow into this test.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=pending
  - evidence: `src/simsopt_jax/core/quasisymmetry.py:14` (anchor: `def non_quasi_symmetric_residual_primitives(`), `src/simsopt_jax/core/quasisymmetry.py:37` (anchor: `def non_quasi_symmetric_ratio(`), `src/simsopt_jax_adapters/geo/surface_objectives.py:2574` (anchor: `return non_quasi_symmetric_ratio(`), `src/simsopt_jax_adapters/geo/surface_objectives.py:3071` (anchor: `class NonQuasiSymmetricRatioJAX(_BoozerObjectiveBase):`), `src/simsopt_jax/objectives/single_stage_fullspace.py:594` (anchor: `return non_quasi_symmetric_residual_primitives(`)
  - JAX-side tests: `tests/integration/test_jax_mirror_boozerqa_parity.py::test_exact_boozerqa_workflow_matches_native_and_jax_cpu`
  - blocker: A dedicated frozen-data unit test for non_quasi_symmetric_ratio and non_quasi_symmetric_residual_primitives is deferred to the 2026-07-29 plan's later phases (ruled 2026-08-24 not to be a seventh unit of this wave).
  - tolerance owner: `local_test_tolerances`

### spec

- **SP-1 — SPEC equilibrium driver**
  - disposition: `native_only`
  - reason: The SPEC tests drive an external Fortran equilibrium executable through py_spec. Replacing an external scientific solver is an explicit non-goal of the coverage contract, and the project rule excludes SPEC-dependent code from mirroring.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/mhd/spec.py` (anchor: `This module provides a class that handles the SPEC equilibrium code.`), `tests/mhd/test_spec.py`
  - tolerance owner: `not_applicable`

### strain

- **ST-1 — Frenet-frame branch of the strain objective**
  - disposition: `jax_partial`
  - reason: The JAX strain-optimization example evaluates the centroid frame only. The rotated-Frenet kernel exists in simsopt_jax and is wired for finite-build and framed-curve consumers, but the strain example never selects it, so the native FrameRotation='frenet' strain branch has no JAX-side evaluation path. test_strain_optimization_vanishes_matches_native_centroid_frame also drives the vanishing-strain claim through zero-threshold LPTorsionalStrainPenalty/LPBinormalCurvatureStrainPenalty oracles (threshold=0), not native's own test_torsion/test_binormal_curvature defaults (threshold=1e-8 and threshold=1e-4 respectively); every literal tolerance in that test (1e-12, 1e-9) is declared in the test file, not src/simsopt_jax/parity_tolerances.py, so the tolerance owner is local, not central.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=passing
  - evidence: `src/simsopt_jax/examples/strain_optimization.py:101` (anchor: `_tangent, _normal, binormal = rotated_centroid_frame(`), `src/simsopt_jax/core/framedcurve.py:174` (anchor: `def rotated_frenet_frame(`), `src/simsopt_jax/core/finitebuild.py:97` (anchor: `if spec.frame_kind == "frenet":`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_strain_parity.py::test_strain_optimization_vanishes_matches_native_centroid_frame`
  - blocker: Wiring the Frenet branch into the strain example is a production change to src/simsopt_jax, outside this wave's non-goals.
  - tolerance owner: `local_test_tolerances`
- **ST-2 — Curve-shape-DOF gradients of the strain penalties**
  - disposition: `jax_missing`
  - reason: The JAX strain program differentiates with respect to the rotation DOFs only (value_and_grad argnums=4). Native LPBinormalCurvatureStrainPenalty.dJ also contributes the curve-shape term through dgammadash_by_dcoeff_vjp, so the curve-DOF half of the native gradient has no JAX equivalent.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt_jax/examples/strain_optimization.py:160-168` (anchor: `jax.value_and_grad(_strain_objective, argnums=4)`), `src/simsopt/geo/strain_optimization.py:61-62` (anchor: `dgammadash_by_dcoeff_vjp(grad1)`)
  - blocker: Extending the JAX program to differentiate the curve DOFs is a production change to src/simsopt_jax, outside this wave.
  - tolerance owner: `not_applicable`
- **ST-3 — General per-component Lp strain penalty for arbitrary p**
  - disposition: `jax_partial`
  - reason: The JAX side ships one fused combined objective with the p=2 excess-squared form hard-wired. Native exposes a general Lp penalty per strain component for arbitrary p, so the JAX lane covers the shipped configuration but not the general exponent. Both bound tests Taylor-check the gradient of the RAW summed strain (torsional_strain/binormal_curvature_strain), not native's LP-thresholded penalties (LPTorsionalStrainPenalty threshold=1e-8, LPBinormalCurvatureStrainPenalty threshold=1e-4); that thresholded-gradient path has no JAX mirror. The value comparison draws rtol/atol from src/simsopt_jax/parity_tolerances.py's direct_kernel lane, but the gradient checks use a fixed local floor (1e-12 for torsion) or a locally-declared shrink factor (0.3 for binormal curvature) that is never drawn from the central module, so since not all of this row's tolerances are central, the owner is recorded local.
  - lanes: native_cpu=passing, jax_cpu=passing, jax_gpu=passing
  - evidence: `src/simsopt_jax/examples/strain_optimization.py:128-157` (anchor: `def _strain_objective(`), `src/simsopt/geo/curveobjectives.py:137-154` (anchor: `def Lp_torsion_pure(torsion, gammadash, p, threshold):`)
  - JAX-side tests: `tests/jax/native_unit_parity/test_strain_parity.py::test_torsional_strain_matches_native_and_gradient_taylor_checks`, `tests/jax/native_unit_parity/test_strain_parity.py::test_binormal_curvature_strain_matches_native_and_gradient_taylor_checks`
  - blocker: A general-p JAX penalty is a production change, outside this wave.
  - tolerance owner: `local_test_tolerances`

### virtual_casing

- **VC-1 — Virtual casing principle**
  - disposition: `native_only`
  - reason: VirtualCasing is driven from a VMEC equilibrium and delegates the surface integral to the third-party virtual_casing package, writing and reading NetCDF along the way. It is VMEC- and third-party-dependent, excluded by the project rule.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/mhd/virtual_casing.py:24` (anchor: `from .vmec import Vmec`), `src/simsopt/mhd/virtual_casing.py:124` (anchor: `def from_vmec(cls, vmec, src_nphi, src_ntheta=None`)
  - tolerance owner: `not_applicable`

### vmec

- **VM-1 — VMEC equilibrium driver and diagnostics**
  - disposition: `native_only`
  - reason: These files construct and run the external VMEC executable (or import the f90wrap `vmec` module directly) and read its output files. Replacing VMEC is an explicit non-goal of the coverage contract, and the project rule excludes VMEC-dependent code. The portable pieces that were ported already live behind their own JAX-named tests.
  - lanes: native_cpu=passing, jax_cpu=not_applicable, jax_gpu=not_applicable
  - evidence: `src/simsopt/mhd/vmec.py` (anchor: `This module provides a class that handles the VMEC equilibrium code.`), `tests/mhd/test_vmec_f90wrap.py:9`
  - tolerance owner: `not_applicable`

## Classified native test files

| Native test file | Domain | Disposition | Capabilities |
| --- | --- | --- | --- |
| `tests/field/test_coilset.py` | coilset | `native_only` | CS-1 |
| `tests/field/test_mgrid.py` | mgrid | `native_only` | MG-1 |
| `tests/field/test_normal_field.py` | normal_field | `native_only` | NF-1 |
| `tests/field/test_selffieldforces.py` | force | `jax_partial` | MF-1, MF-2, MF-3, MF-4, MF-5, MF-6, MF-7, MF-8 |
| `tests/geo/test_curve.py` | curves | `jax_partial` | CV-1, CV-2, CV-3 |
| `tests/geo/test_curve_helical.py` | curves | `jax_missing` | CV-1 |
| `tests/geo/test_curveperturbed.py` | curveperturbed | `jax_partial` | CP-1, CP-2, CP-3, CP-4 |
| `tests/geo/test_ports.py` | ports | `hybrid_boundary` | PT-1 |
| `tests/geo/test_strainopt.py` | strain | `jax_partial` | ST-1, ST-2, ST-3 |
| `tests/geo/test_surface_objectives.py` | quasisymmetry | `jax_partial` | QS-1 |
| `tests/mhd/test_bootstrap.py` | vmec | `native_only` | VM-1 |
| `tests/mhd/test_integrated_mpi.py` | vmec | `native_only` | VM-1, SP-1 |
| `tests/mhd/test_spec.py` | spec | `native_only` | SP-1 |
| `tests/mhd/test_virtual_casing.py` | virtual_casing | `native_only` | VC-1 |
| `tests/mhd/test_vmec.py` | vmec | `native_only` | VM-1 |
| `tests/mhd/test_vmec_diagnostics.py` | vmec | `native_only` | VM-1 |
| `tests/mhd/test_vmec_f90wrap.py` | vmec | `native_only` | VM-1 |
| `tests/util/test_fourier_interpolation.py` | fourier_interpolation | `native_only` | FI-1 |

## Unclassified native test files (52)

These files are on the pinned native surface and have no capability mapping yet. They are listed here so the gap is visible, and they are not counted as covered.

- `tests/configs/test_LHD_like.py`
- `tests/configs/test_quasr_integration.py`
- `tests/configs/test_zoo.py`
- `tests/configs/test_zoo_mock_quasr.py`
- `tests/core/test_derivative.py`
- `tests/core/test_descriptor.py`
- `tests/core/test_dev.py`
- `tests/core/test_dofs.py`
- `tests/core/test_finite_difference.py`
- `tests/core/test_integrated.py`
- `tests/core/test_json.py`
- `tests/core/test_optimizable.py`
- `tests/core/test_util.py`
- `tests/field/test_biotsavart.py`
- `tests/field/test_boozermagneticfields.py`
- `tests/field/test_coil.py`
- `tests/field/test_fieldline.py`
- `tests/field/test_interpolant.py`
- `tests/field/test_magnetic_axis_helpers.py`
- `tests/field/test_magneticfields.py`
- `tests/field/test_magneticfields_optimization.py`
- `tests/field/test_mpi_tracing.py`
- `tests/field/test_particle.py`
- `tests/field/test_sampling.py`
- `tests/field/test_wireframefield.py`
- `tests/geo/test_boozersurface.py`
- `tests/geo/test_curve_objectives.py`
- `tests/geo/test_curve_optimizable.py`
- `tests/geo/test_finitebuild.py`
- `tests/geo/test_plot.py`
- `tests/geo/test_pm_grid.py`
- `tests/geo/test_qfm.py`
- `tests/geo/test_surface.py`
- `tests/geo/test_surface_garabedian.py`
- `tests/geo/test_surface_rzfourier.py`
- `tests/geo/test_surface_taylor.py`
- `tests/geo/test_surface_xyzfourier.py`
- `tests/geo/test_surfacehenneberg.py`
- `tests/geo/test_wireframe_toroidal.py`
- `tests/mhd/test_boozer.py`
- `tests/mhd/test_profiles.py`
- `tests/objectives/test_constrained.py`
- `tests/objectives/test_fluxobjective.py`
- `tests/objectives/test_least_squares.py`
- `tests/objectives/test_utilities.py`
- `tests/solve/test_constrained.py`
- `tests/solve/test_least_squares.py`
- `tests/solve/test_mpi.py`
- `tests/solve/test_pm_optimization.py`
- `tests/solve/test_wf_optimization.py`
- `tests/util/test_coil_optimization_helper_functions.py`
- `tests/util/test_mpi_partition.py`

## Follow-ups

- DONE 2026-08-24: GPU evidence for the force/strain/curve-subclass mirror suites landed — gpu_parity lane green on the RTX 5090 (3/12/75 per suite) under PYTHONPATH=src:build/cp311-cp311-linux_x86_64 JAX_ENABLE_X64=1 JAX_PLATFORMS=cuda,cpu with .venv-qn-gpu. JAX_PLATFORMS must name cuda explicitly: with it unset, pytest initializes CPU-only and the gpu lane skip-silently vanishes.
- Register the `jax_native_unit_parity` pytest marker in pyproject.toml and require each jax_equivalent capability to be collected by the CPU and default-backend-mode gpu_parity lanes (2026-07-29 plan step 6). Deferred here because pyproject.toml is outside this unit's file ownership.
- Populate the full per-function native ledger — every test definition in every file on the pinned surface mapped to a capability ID — per the 2026-07-29 plan's own checklist. The counts tables in the generated report are this slice's state; do not restate them here.
- Add a dedicated frozen-data unit test for simsopt_jax.core.quasisymmetry (QS-1). Ruled 2026-08-24 not to be a seventh unit of the 2026-08-23 wave.
- Parametrize tests/jax/examples/test_stochastic_samples.py over the parity_lane fixture so the CurvePerturbed reformulation gains a GPU lane (CP-4).
- Move the capabilities whose tolerance owner is `local_test_tolerances` onto src/simsopt_jax/parity_tolerances.py, which the contract names as the centralized owner: CV-1, CV-2, CP-4, MF-1, MF-5, MF-7, MF-8, ST-1, ST-3, QS-1.
- Upstream drift: hiddenSymmetries master at 4ad6fd99189b99d9722ad33aaeb5d30adc81680f adds tests/util/test_logger.py, which this fork does not contain. It is outside the pinned merge-base baseline and must be classified when the baseline is advanced.
- Wire this coverage validator into a PR-blocking CI job. No CI invocation of scripts/jax_native_unit_coverage.py exists today (verified); this needs a workflow edit, which is outside this slice's file ownership.
- Add the new files this slice touches (scripts/jax_native_unit_coverage.py, tests/jax/test_native_unit_coverage_manifest.py) to pyproject.toml's curated pyright include list. pyproject.toml is outside this slice's file ownership.
- Obtain named maintainer countersignature for every native_only/hybrid_boundary/shared_python decision recorded in this manifest. The contract requires a named reviewer; every decision.decided_by in this slice currently reads 'session orchestrator under the user's 2026-08-24 execute-all directive; named-maintainer countersignature pending'.
