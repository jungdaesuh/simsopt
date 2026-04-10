# JAX Parity Manifest

Status snapshot as of 2026-04-10.

Exact parity means the mirrored JAX test runs with:

- `jax_enable_x64=True` before test arrays are created
- strict backend parity lanes
- explicit host materialization / device sync before parity assertions
- fixed seeded fixtures across CPU and JAX lanes

Solver-level parity stays contract-based: convergence success, residual norm,
final objective, and final physics quantities must match within the documented
acceptance envelope, but iterate-by-iterate identity is not required.

| Upstream test | JAX test | Status | Notes |
| --- | --- | --- | --- |
| `tests/field/test_biotsavart.py` | `tests/field/test_biotsavart_jax_parity.py` | exact | Pure-kernel mirror for `A/B`, spatial derivatives, Hessians, and VJP identities. |
| `tests/objectives/test_fluxobjective.py` | `tests/objectives/test_fluxobjective_jax_parity.py` | partial | Dedicated mirrored wrapper coverage for definitions, derivatives, target handling, degenerate normals, singular zero-field behavior, and native-only fallback rejection. Value/gradient parity is exact where the upstream wrapper contract is defined, but the CPU wrapper still inherits `simsoptpp.integral_BdotN` `nan` boundaries on some degenerate surfaces where the JAX wrapper intentionally returns the stabilized zero-area/singular contract. |
| `tests/objectives/test_fluxobjective.py` | `tests/integration/test_stage2_jax.py` | partial | Integration coverage for mixed quadrature and native/fallback lane behavior complements the dedicated object-level parity file. |
| `tests/objectives/test_integral_bdotn_jax.py` | `tests/objectives/test_integral_bdotn_jax.py` | partial | Exact on regular inputs; direct `simsoptpp.integral_BdotN` still returns `nan` on some degenerate/singular boundaries where the JAX path normalizes to the higher-level wrapper contract (`0.0` or `inf`). |
| `tests/geo/test_surface_rzfourier.py` | `tests/geo/test_surface_rzfourier_jax.py` | exact | Mirrored CPU/JAX geometry parity under strict CPU/GPU parity lanes. |
| `tests/geo/test_surface_objectives.py::ToroidalFlux*` | `tests/geo/test_surface_objectives_jax.py` | partial | Upstream `ToroidalFlux` constant / first-derivative / Hessian / coil-derivative tests are mirrored across the surface-type and `stellsym` sweep, but this family intentionally remains on tolerance-based CPU/JAX parity rather than exact arithmetic parity. |
| `tests/geo/test_boozersurface.py` | `tests/geo/test_boozersurface_jax.py` | partial | JAX-specific solver and residual contracts are covered; parity is defined by solved-state quality rather than iterate identity. |
| `tests/integration/test_single_stage_example.py` and single-stage Boozer integration slices | `tests/integration/test_single_stage_jax_cpu_reference.py` | partial | CPU/JAX and CPU/GPU solver parity assertions cover convergence success, residual norms, final objective values, iota, and final physics quantities. |
