# Item 06 JAX Transform And Memory Plan

## Compiled Boundaries

Item 06 covers the SurfaceRZFourier, SurfaceXYZFourier, and
SurfaceXYZTensorFourier kernels in `src/simsopt/jax_core/`. Compiled
entrypoints used by the production-scale closeout fixture:

- `surface_rz_fourier_gamma_from_spec(spec)` and
  `surface_rz_fourier_gamma_from_dofs(spec, dofs)`
- `surface_rz_fourier_gammadash1_from_spec(spec)` and `_from_dofs(...)`
- `surface_rz_fourier_gammadash2_from_spec(spec)` and `_from_dofs(...)`
- `surface_rz_fourier_normal_from_spec(spec)` and `_from_dofs(...)`
- `surface_rz_fourier_area_from_spec(spec)` and `_from_dofs(...)`
- `surface_rz_fourier_volume_from_spec(spec)` and `_from_dofs(...)`
- adapter forwarders: `SurfaceRZFourier.gamma_jax`,
  `.gammadash1_jax`, `.gammadash2_jax`, `.normal_jax`, `.area_jax`,
  `.volume_jax`, and matching higher-order entrypoints in
  `tests/geo/test_surface_rzfourier_jax.py`.

For SurfaceXYZ(Tensor)Fourier the compiled entrypoints in
`jax_core/surface_fourier.py` are `surface_gamma`,
`surface_gammadash1`, `surface_gammadash2`, `surface_normal`,
`surface_area`, `surface_volume`, and `build_{theta,phi}_basis`. The
stellsym branch uses `stellsym_scatter_indices(mpol, ntor)` to
gather/scatter Fourier coefficients without dynamic indexing.

## Transforms

- `jit`: applied to `surface_gamma_from_dofs`,
  `surface_normal_from_dofs`, `surface_area_from_dofs`,
  `surface_volume_from_dofs`, and to the
  `surface_rz_fourier_*_from_spec` / `_from_dofs` entrypoints.
  Static-shape strategy: `nphi`, `ntheta`, `mpol`, `ntor`, `nfp`, and
  `stellsym` are treated as static (carried on the
  `SurfaceRZFourierSpec` / `SurfaceXYZFourierSpec` /
  `SurfaceXYZTensorFourierSpec` pytrees) so changing them triggers a
  recompilation rather than dynamic-shape paths.
- `vmap`: only where the kernel naturally requires per-quadrature
  fanout (existing `TestSurfaceFourierJaxCppParity` rows). The
  closeout fixture does not introduce any new `vmap` use.
- `scan` / `fori_loop`: N/A. Surface kernels are uniform over the
  `(nphi, ntheta)` grid and rely on broadcasted Fourier evaluations,
  not Python loops over trace-time-sized axes.
- `checkpoint` / `remat`: N/A for the closeout fixture. The forward
  surface kernel is shallow (one Fourier evaluation + reductions) and
  does not require activation rematerialization to stay within the
  dense materialization budget.
- `shard_map` / `pmap` / collectives: N/A. Item 06 does not introduce
  any sharded or collective lowering; the closeout test runs on a
  single CPU device and only checks single-device parity.
- CPU-ordered census twins
  (`surface_fourier_jax_cpu_ordered.py`): same compiled-jit lowering
  as the standard JAX entrypoints but with reduction order forced to
  match the C++ census order. Item 06 closeout does not touch this
  path; it is already covered by
  `tests/geo/test_surface_fourier_jax_cpu_ordered.py`.

## Memory And Donation

Largest array shapes for the production-scale closeout fixture
(`nphi=32, ntheta=16, mpol=4, ntor=3, nfp=2, stellsym=False`):

- `gamma` / `gammadash1` / `gammadash2` / `normal`: `float64[32, 16, 3]`
  = `12,288 bytes` each.
- DOF vector: `float64[<= 144]` (non-stellsym SurfaceRZFourier with
  `mpol=4, ntor=3`).
- Spec arrays (`rc`, `rs`, `zc`, `zs`): `float64[mpol+1, 2*ntor+1] =
  float64[5, 7] = 280 bytes` each.

No buffer donation is used. `donate_argnums` / `donate_argnames`
remain N/A: the closeout fixture reuses the CPU `SurfaceRZFourier`
object after the JAX call to evaluate the CPU oracle (`surface.gamma()`,
etc.). Donating the spec or DOF arrays into the JAX kernel would
preclude reuse and is not required at this scale.

Dense materialization budget: no new dense tensor is materialized
beyond the existing `(nphi, ntheta, 3)` output arrays already
materialized by the SurfaceRZFourier JAX path.

## HLO / Benchmark Artifact

Item 06 ships test-only; no new kernel was introduced and no new
production hot path was created. The benchmark JSON is recorded as
N/A at `.artifacts/jax_port_goal/bench/06.json` per section 4c with
the rationale "test-only closeout; no hot-path source change". HLO
proxy work for the surface kernels is already covered by
`test_surface_rzfourier_fused_geometry_reduces_hlo_work`,
`test_surface_rzfourier_scalar_gamma_hlo_stays_single_output`,
`test_surface_rzfourier_geometry_avoids_jnp_arange`, and
`test_surface_rz_geometry_hlo_probe_entrypoint_uses_local_package`
in `tests/geo/test_surface_rzfourier_jax.py`.
