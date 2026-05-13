# Item 01 Math And Physics Invariants

## CurrentPenalty

- Units: current values are in amps when constructed from SIMSOPT current
  objects; the scalar objective has units `A^2` if `threshold` is in amps.
- Current sign convention: the penalty is even in `I`; the derivative is odd
  above threshold and zero below threshold.
- Scalar contract:
  `J(I) = max(abs(I) - threshold, 0)^2`.
- Derivative contract for `abs(I) > threshold`:
  `dJ/dI = 2 * (abs(I) - threshold) * sign(I)`.
- Projection contract:
  - `Current(8), threshold=5` gives gradient `[6]`.
  - `Current(-8), threshold=5` gives gradient `[-6]`.
  - `2 * Current(4), threshold=5` projects to base-current gradient `[12]`.
  - `Current(2) + Current(4), threshold=5` projects to `[2, 2]`.
- Excluded singular regime: `abs(I) == threshold` is nondifferentiable. No
  equality-threshold derivative claim is made.
- Infinite-current behavior: `I = inf` stays `inf`, not `nan`.

## Distance Wrappers

- Units: distances are in the coordinate units of the curve/surface geometry;
  penalties are weighted by curve arc-length and surface normal magnitudes.
- Orientation: curve-curve candidate culling preserves upstream lower-triangle
  semantics and `num_basecurves`; curve-surface culling preserves rectangular
  cross-collection semantics.
- Symmetry: `CurveCurveDistance` coverage includes `num_basecurves`; broader
  `stellsym=True` / `stellsym=False` consumers remain covered by existing
  downstream tests and item-specific downstream fixtures, not by a new item 01
  optimizer trajectory claim.
- Derivative shape: public wrapper `dJ()` returns legacy `Derivative`-projected
  host gradients; pure helper gradients return JAX array cotangents before
  wrapper projection.
- Contract type: fixed-state scalar and fixed-state gradient/VJP parity.
  Optimizer traces are not the item 01 oracle.
- Excluded regimes: exact point collisions at the barrier boundary are
  intentionally `inf` for `CurveCurveDistanceBarrier`; near-coil singular
  magnetic-field physics is outside this distance-objective item.

## Negative Controls / Red Evidence

The detached HEAD red run with the new tests caught:

- 0-D gradient blocks passed into `Derivative`, raising `IndexError`.
- Python scalar zero literals in `current_penalty_pure`, raising strict
  host-to-device transfer errors under `jax.transfer_guard("disallow")`.
- Strict wrapper calls that crossed the host/device boundary implicitly.

The post-fix tests also include an infinite-current assertion to prevent the
strict-transfer zero construction from turning `inf` into `nan`.
