# JAX Parity Manifest

Base parity matrix snapshot as of 2026-04-10.

Boozer rows refreshed on 2026-05-05 after the CPU closure in
[`boozer_full_parity_plan_2026-05-04.md`](boozer_full_parity_plan_2026-05-04.md).
This refresh does not claim CUDA hardware parity.

Exact parity means the mirrored JAX test runs with:

- `jax_enable_x64=True` before test arrays are created
- strict backend parity lanes
- explicit host materialization / device sync before parity assertions
- fixed seeded fixtures across CPU and JAX lanes

Solver-level parity stays contract-based: convergence success, residual norm,
final objective, and final physics quantities must match within the documented
acceptance envelope, but iterate-by-iterate identity is not required.

## Parity Test Matrix

This matrix is the SSOT for the mirrored parity surface requested by
`jax_parity_reduction_todos_2026-04-10.md`.

Current Boozer CPU closure is tracked in
[`boozer_full_parity_plan_2026-05-04.md`](boozer_full_parity_plan_2026-05-04.md),
including the explicit mutable-identity exclusions and the current pass/fail
watermark.

| Upstream test | JAX test | Status | Notes |
| --- | --- | --- | --- |
| `tests/field/test_biotsavart.py` | `tests/field/test_biotsavart_jax_parity.py` | exact | Pure-kernel mirror for `A/B`, spatial derivatives, Hessians, and VJP identities. |
| `tests/objectives/test_fluxobjective.py` | `tests/objectives/test_fluxobjective_jax_parity.py` | partial | Dedicated mirrored wrapper coverage for definitions, derivatives, target handling, degenerate normals, singular zero-field behavior, and native-contract rejection. Value/gradient parity is exact where the upstream wrapper contract is defined, but the CPU wrapper still inherits `simsoptpp.integral_BdotN` `nan` boundaries on some degenerate surfaces where the JAX wrapper intentionally returns the stabilized zero-area/singular contract. |
| `tests/objectives/test_fluxobjective.py` | `tests/integration/test_stage2_jax.py` | partial | Integration coverage for mixed quadrature and native-spec rejection behavior complements the dedicated object-level parity file. |
| `tests/objectives/test_integral_bdotn_jax.py` | `tests/objectives/test_integral_bdotn_jax.py` | partial | Exact on regular inputs; direct `simsoptpp.integral_BdotN` still returns `nan` on some degenerate/singular boundaries where the JAX path normalizes to the higher-level wrapper contract (`0.0` or `inf`). |
| `tests/geo/test_surface_rzfourier.py` | `tests/geo/test_surface_rzfourier_jax.py` | partial | Strict tolerance-based CPU/JAX parity for the JAX geometry/object API surface (`surface_spec`/`to_spec`, `*_jax`, DOF round-trips, gradients, loaders, and `copy`). This path intentionally avoids reproducible-summation complexity; full upstream class parity would still require broader object/I/O utility mirroring beyond the current JAX geometry contract. |
| `tests/geo/test_surface_objectives.py::ToroidalFlux*` | `tests/geo/test_surface_objectives_jax.py` | partial | Upstream `ToroidalFlux` constant / first-derivative / Hessian / coil-derivative tests are mirrored across the surface-type and `stellsym` sweep, but this family intentionally remains on tolerance-based CPU/JAX parity rather than exact arithmetic parity. |
| `tests/geo/test_boozersurface.py` | `tests/geo/test_boozersurface_jax.py` | cpu-contract-complete | Boozer CPU parity closure is complete for math kernels, solver results, guard behavior, derivatives/adjoints, and supported public APIs. Parity remains contract-based: solved-state quality and public result semantics are the oracle, not mutable object identity or iterate-by-iterate solver trajectory. |
| `tests/integration/test_single_stage_example.py` and single-stage Boozer integration slices | `tests/integration/test_single_stage_jax_cpu_reference.py` | cpu-contract-complete | Dedicated CPU/JAX Boozer integration tests compare convergence success, residual norms, final solver objective, and final physics quantities (`iota`, `G`, label value/error, anchored axis-z). CUDA Boozer parity is not claimed by this CPU closure and still requires the optional hardware validation gate in the Boozer plan. |
