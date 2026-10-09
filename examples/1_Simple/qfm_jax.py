#!/usr/bin/env python3
"""The native QFM example with --jax and --device cpu|gpu selection.

Volume, ToroidalFlux and Area are minimized in that order, each with native
L-BFGS-B penalty followed by squared-equality SLSQP. SciPy stays on the host.
"""

import argparse

import numpy as np

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import Area, QfmSurface, SurfaceRZFourier, ToroidalFlux, Volume
from simsopt_jax.backend import set_backend
from simsopt_jax_adapters.field import JaxBiotSavart
from simsopt_jax_adapters.geo.qfm import JaxQfmSurface


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax", action="store_true", help="evaluate QFM on JAX")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--maxiter", type=int, default=1000)
    args = parser.parse_args()
    if args.jax:
        set_backend("jax", device=args.device, precision="fp64")
    _, _, axis, nfp, native_field = get_data("ncsx")
    field = JaxBiotSavart(native_field.coils) if args.jax else native_field
    # A separate field preserves native QfmResidual's surface-point buffer.
    flux_field = BiotSavart(native_field.coils)
    surface = SurfaceRZFourier(
        mpol=5, ntor=5, stellsym=True, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1/nfp, 25, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 25, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.2, flip_theta=True)
    solver_class = JaxQfmSurface if args.jax else QfmSurface
    for label in (Volume(surface), ToroidalFlux(surface, flux_field), Area(surface)):
        target = label.J()
        solver = solver_class(field, surface, label, target)
        for method in ("LBFGS", "SLSQP"):
            result = solver.minimize_qfm(method=method, tol=1e-12, maxiter=args.maxiter)
            print(f"{type(label).__name__} {method}: success={result['success']}, "
                  f"constraint={0.5*(label.J()-target)**2:.8e}, qfm={solver.qfm.J():.8e}")


if __name__ == "__main__":
    main()
