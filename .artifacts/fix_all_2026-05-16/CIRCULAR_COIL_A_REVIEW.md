# CircularCoil vector-potential review

Date: 2026-05-18

## Verdict

The `CircularCoil._A_impl` concern is not a JAX-port bug. The JAX kernel in
`src/simsopt/jax_core/circular_coil.py` intentionally mirrors the CPU formula in
`src/simsopt/field/magneticfieldclasses.py`, including the unusual additive
`2 * r0` term in the numerator.

Do not change only the JAX implementation. Any correction would need to be an
upstream `CircularCoil` API/physics decision and should update both CPU and JAX
or explicitly document the gauge convention.

## Evidence

- CPU source: `src/simsopt/field/magneticfieldclasses.py::_A_impl`
- JAX source: `src/simsopt/jax_core/circular_coil.py::_A_local_pointwise`
- Current parity tests compare JAX `A` against CPU `CircularCoil.A()` in
  `tests/field/test_circular_coil_jax.py`.
- Existing CPU tests compare `CircularCoil.B()` and `dB_by_dX()` against a
  BiotSavart coil, but they do not compare `CircularCoil.A()` against
  `BiotSavart.A()`.

## Numerical probe

For a representative radius/center/current and three off-coil points,
`CircularCoil.A()` and `BiotSavart.A()` differed by relative norm about `0.75`.
That is not by itself proof of a bug, because vector potential is gauge
dependent. It is enough to keep the item classified as upstream review rather
than local JAX-port remediation.

## Follow-up

If this becomes a physics-correctness task rather than a JAX-port parity task,
derive and pin the intended `A_phi(rho, z)` gauge convention, then add a CPU
oracle test before changing either implementation.
