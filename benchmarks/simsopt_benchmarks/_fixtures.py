"""
Shared fixtures for simsopt benchmarks.

Every benchmark constructs its inputs from these helpers to keep shapes
consistent across benchmarks and between runs. Each fixture accepts a
``size`` label so we can run both production-sized ("prod") and small
("small") variants per the regression-prevention plan in
PERFORMANCE_AUDIT.md.
"""
from __future__ import annotations

import numpy as np


SIZES = {
    # Label  →  (ncoils, n_quad_per_coil, n_eval_points, n_dofs_per_coil)
    "prod":  (16, 200, 10_000, 30),
    "small": (1,  50,  100,    10),
}


def make_coils(size: str = "prod"):
    """Build a Biot–Savart setup: coils + currents + evaluation points.

    Returns a dict with keys: ``bs`` (BiotSavart), ``coils``, ``points``.
    The field has *not* been set yet; caller runs ``bs.set_points(points)``.
    """
    from simsopt.field.coil import Coil, Current
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.geo.curvexyzfourier import CurveXYZFourier

    ncoils, n_quad, npoints, order = SIZES[size]
    rng = np.random.default_rng(0)

    coils = []
    for k in range(ncoils):
        curve = CurveXYZFourier(n_quad, order)
        x = curve.get_dofs()
        # small random perturbation around a circular coil
        x[0] = 1.0 + 0.1 * k
        x[2 * order + 1] = 0.3
        x[4 * order + 2] = 0.1 * (k + 1) / ncoils
        x[:] += 0.01 * rng.standard_normal(x.shape)
        curve.set_dofs(x)
        coils.append(Coil(curve, Current(1e5 * (1.0 + 0.05 * k))))

    bs = BiotSavart(coils)
    points = rng.uniform(-0.5, 0.5, size=(npoints, 3)) + np.array([1.0, 0.0, 0.0])
    return {"bs": bs, "coils": coils, "points": points}


def make_surface(size: str = "prod"):
    """Build a SurfaceRZFourier with representative resolution."""
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier

    if size == "prod":
        mpol, ntor, nphi, ntheta = 8, 8, 64, 64
    else:
        mpol, ntor, nphi, ntheta = 2, 2, 16, 16

    s = SurfaceRZFourier(
        nfp=3, stellsym=True, mpol=mpol, ntor=ntor,
        quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    s.rc[0, ntor] = 1.0
    s.rc[1, ntor] = 0.1
    s.zs[1, ntor] = 0.1
    return s


def output_fingerprint(arr: np.ndarray) -> float:
    """A cheap scalar fingerprint of a result array.

    Catches benchmarks that accidentally start returning zeros / NaNs
    by tracking a stable scalar across runs. The exact value is not
    meaningful; only its stability matters.
    """
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr.ravel()))
