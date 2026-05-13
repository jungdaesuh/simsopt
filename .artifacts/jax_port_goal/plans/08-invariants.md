# Item 08 Math And Physics Invariants

## Units And Scales

- Torsion `tau` of a curve has units of `1/m` (curvature units).
- Curvature vector `kappa` has units of `1/m`; binormal unit vector
  `b_hat` is dimensionless.
- Tape width `w` has units of `m`.
- Pointwise strain definitions are dimensionless ratios:
  - Torsional strain: `epsilon_tor = tau^2 * w^2 / 12` (`(1/m)^2 * m^2`).
  - Binormal curvature strain: `epsilon_bend = w * |b_hat . kappa| / 2`
    (`m * 1/m`).
- Arc length element `dl = ||gamma'(t)||_2 dt` is in meters; the
  `Lp_torsion_pure` integrand internally divides by the number of
  quadrature points (`jnp.mean`), so the reported scalar is a curve-
  averaged strain^p with units of (strain)^p · m / quadrature count.
  This convention matches upstream's pure function exactly.

## `Lp` Norm Convention

`Lp_torsion_pure(strain_like, gammadash, p, threshold)` returns

```
(1 / p) * mean_t(max(|strain_like(t)| - threshold, 0)**p * ||gammadash(t)||_2).
```

This is the upstream form from
`/Users/suhjungdae/code/opensource/simsopt/src/simsopt/geo/curveobjectives.py`
and the current repo's `src/simsopt/geo/curveobjectives.py:338`. Item 08
re-uses this exact integrand for both `LPTorsionalStrainPenalty` and
`LPBinormalCurvatureStrainPenalty` by passing the appropriate strain
quantity as `strain_like`.

## Strain Definitions

- Torsional: `epsilon_tor = tau^2 * w^2 / 12`. Implemented as
  `torstrain_pure(torsion, width) = torsion**2 * width**2 / 12`
  (`strain_optimization.py:230`).
- Binormal curvature: `epsilon_bend = w * |b_hat . kappa| / 2`.
  Implemented as `binormstrain_pure(binorm, width) = (width / 2) *
  jnp.abs(binorm)` (`strain_optimization.py:238`), where `binorm` is
  the upstream `framedcurve.frame_binormal_curvature()` scalar field
  along the curve.

These match the Paz-Soldan 2020 definitions cited in the public
`CoilStrain` docstring (`strain_optimization.py:181-210`).

## Derivative Shape

- `LPBinormalCurvatureStrainPenalty.dJ()` and
  `LPTorsionalStrainPenalty.dJ()` return a `Derivative` projection
  built from two contributions: `framedcurve.dframe_*_by_dcoeff_vjp`
  for the strain part and `curve.dgammadash_by_dcoeff_vjp` for the
  arc-length part. The public `dJ()` boundary returns the host-side
  derivative projection consumed by `Optimizable.x` / scipy.
- For the production-scale NCSX `coil_order=6, points_per_period=120`
  fixture used in item 08, the projected gradient has shape
  `(objective.x.size,)`, where `objective.x.size` matches the curve's
  free Fourier DOFs plus the rotation Fourier DOFs (42 for the
  `subtest_torsion` config with `coil_order=6` curve + `order=1`
  rotation).

## Stellsym

Not applicable. Strain is a per-curve, per-quadrature-point invariant
computed from local frame torsion and binormal curvature. There is no
surface symmetry seam in this module.

## Excluded Regimes

- The integrand uses `max(|strain| - threshold, 0)**p` for `p >= 1`,
  which is `C^{p-1}` smooth in `strain` but has a kink at the
  threshold. For `p=2` (the default in the production-scale tests), the
  central-difference FD test in the existing `test_strainopt.py` is
  expected to contract; that contract is unchanged by item 08.
- A torsion-free Frenet frame produces `binorm = 0` identically, so
  `LPBinormalCurvatureStrainPenalty.J()` evaluates to 0 at machine
  precision on a circular `CurveXYZFourier` + `ZeroRotation` fixture.
  This is the negative control exercised by
  `test_lp_binormal_penalty_zero_twist_circle_vanishes_in_frenet_frame`.

## Parity Lane

`direct-kernel`. The item-08 new tests pin tolerance through
`benchmarks.validation_ladder_contract.parity_ladder_tolerances("direct_kernel")`,
which provides `rtol=1e-10`, `atol=1e-12`. Both the host NumPy
reference for `Lp_torsion_pure` and the zero-twist control's floor are
expressed in these tolerances.
