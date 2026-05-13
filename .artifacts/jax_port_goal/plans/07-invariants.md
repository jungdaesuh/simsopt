# Item 07 Math And Physics Invariants

## Units

- All curves expose `gamma()` in meters, `gammadash()` in meters per
  parametric unit, and `kappa()` in inverse meters. `torsion()` is in
  inverse meters. `incremental_arclength()` is meters per parametric
  unit.
- `CurveLength` has units of meters; the mean-of-arclength
  normalisation cancels the trapezoidal factor so the value matches
  `integral_0^1 |gammadash| dphi`.
- `LpCurveCurvature` has units of `m^{1-p} * m = m^{2-p}` when
  multiplied by the local arclength weight; for the default `p=2,
  threshold=0`, the scalar is dimensionless to leading order in `m`
  (the per-quadrature integrand is `kappa^2 * |gammadash|`).
- `LpCurveCurvatureBarrier` is dimensionless inside the log and is
  multiplied by `|gammadash|`, so the integrated value has units of
  meters.
- `LpCurveTorsion` mirrors `LpCurveCurvature` with `tau` in place of
  `kappa`.
- `ArclengthVariation` has units of `m^2` (variance of mean
  incremental arclength).
- `MeanSquaredCurvature` is dimensionless (mean of `kappa^2 *
  |gammadash|` divided by mean of `|gammadash|`).
- `LinkingNumber` is dimensionless and integer-valued.
- `FramedCurveTwist` is in radians at the lp/net/max boundary; the
  range mode returns `max(profile) - min(profile)` in radians.

## Lp Convention

- `Lp_curvature_pure(kappa, gammadash, p, desired_kappa) = (1/p) *
  mean(max(kappa - desired_kappa, 0)^p * arc_length)`.
- `Lp_torsion_pure(torsion, gammadash, p, threshold) = (1/p) *
  mean(max(|torsion| - threshold, 0)^p * arc_length)`.
- `frametwist_lp_pure(profile, gammadash, p) = (mean(profile^p *
  arc_length) / mean(arc_length))^{1/p}`. Note: this is the
  *generalized* mean with the arclength weight, not the canonical
  `(1/p) * integral` form used by the curvature / torsion penalties.
  This is the as-shipped contract in `curveobjectives.py` and is what
  the FD test pins.

## ArclengthVariation Invariants

- Definition (full mode, `nintervals == nquadpoints`):
  `J = Var(mat @ l)` where `mat` is the row-stochastic indicator
  matrix that maps quadrature samples to intervals and `l` is
  `incremental_arclength()`.
- For a uniform-arclength curve (`l` constant) the value is exactly 0.
  This is exercised by the existing `test_arclength_variation_circle`
  and `test_arclength_variation_circle_planar` tests.
- nfp factor: not present in this objective; the mean is over
  parametric `[0, 1]` and is independent of `nfp`.

## MeanSquaredCurvature Invariants

- Definition: `J = mean(kappa^2 * arc_length) / mean(arc_length)`.
- Independent of the parametrization speed.
- Equivalent to `(1/L) * integral kappa^2 dl` where `L = integral dl`.

## LinkingNumber Invariants

- Integer-valued by topology (`>= 1` if interlocked, `0` otherwise).
- Gauss linking integral:
  `Link(c1, c2) = (1/(4*pi)) * |oint oint (r1 - r2)/|r1 - r2|^3 dot
  (dr1 cross dr2)|`.
- `LinkingNumber.dJ() == Derivative({})` by source contract; the value
  is locally constant in dof space (changes only at link-flip
  topological transitions, which are unreachable by smooth
  perturbations).
- For `create_equally_spaced_curves(ncoils, nfp, stellsym=True)` the
  ring of equally-spaced TF coils is unlinked: every pair lies in
  disjoint half-planes, so the integer linking number is exactly 0.
  The new ncoils=4 production-scale test pins this contract.

## FramedCurveTwist Invariants

- Modes:
  - `f="net"` returns `profile[-1] - profile[0]` (net winding in
    radians).
  - `f="range"` returns `max(profile) - min(profile)`.
  - `f="max"` returns `max(|profile|)`.
  - `f="lp"` returns the weighted lp mean defined above.
- Differentiability:
  - The lp mode is smooth in the curve and rotation dofs whenever the
    twist profile remains bounded; the existing source path uses
    `jax.grad` on `frametwist_lp_pure` and `jax.vjp` on
    `frametwist_pure`.
  - The `{net, range, max}` modes intentionally return
    `Derivative({})` from `dJ()` (see source at
    `src/simsopt/geo/curveobjectives.py:1375`). These are
    reporting-only scalars in the optimization stack; the project
    convention is that the empty derivative block carries no projected
    gradient.
- Centroid frame: the inner reference frame
  `self.framedcurve_centroid = FramedCurveCentroid(framedcurve.curve)`
  is constructed with `.rotation.fix_all()`, so the centroid rotation
  dofs do not vary during optimization.

## Excluded Regimes

- `LpCurveCurvatureBarrier` is infeasible (`J = +inf`) when any
  sampled curvature reaches the threshold from below; the existing
  barrier test pins this.
- `FramedCurveTwist`: dofs that make `dot3` (the dot product of the
  base rotated normal with the centroid normal) cross zero are
  excluded; the small Taylor fixture stays well inside the regular
  regime and does not test the singularity.
- `LinkingNumber`: dof perturbations that cross a topological
  link-flip threshold are excluded; the production-scale test uses an
  unlinked ring and a small perturbation regime.

## Negative Controls / Red Evidence

The red step against parent commit `a9da18fac` is documented in
`.artifacts/jax_port_goal/red/07.txt`. The new tests pass on the
parent because the JAX-native source path was already correct; this is
the expected "no source change required" outcome and is recorded
honestly rather than being claimed as a red->green change.
