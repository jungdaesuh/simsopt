#!/usr/bin/env python3

import argparse
import os
import numpy as np
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import QfmResidual, QfmSurface, SurfaceRZFourier, ToroidalFlux, Area, Volume

"""
This example demonstrate how to compute a quadratic flux minimizing surfaces
from a magnetic field induced by coils. We start with an initial guess that
is just a tube around the magnetic axis using the NCSX coils. We first reduce
the penalty objective function to target a given toroidal flux using LBFGS.
The equality constrained objective is then reduced using SLSQP. This is repeated
for fixing the area and toroidal flux.
"""
try:
    from simsopt_jax.backend import set_backend
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.qfm import JaxQfmResidual, JaxQfmSurface
except ImportError as error:
    jax_import_error = str(error)
else:
    jax_import_error = None

parser = argparse.ArgumentParser(description="Compute quadratic flux minimizing surfaces for NCSX")
parser.add_argument("--use-jax", action="store_true", help="Use optional JAX QFM evaluation with native host optimizers")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
parser.add_argument("--maxiter", type=int, help="Override the native 1000-iteration limit")
args = parser.parse_args()
if args.use_jax:
    if jax_import_error is not None:
        parser.error(f"--use-jax requires Python >= 3.11 and jax/jaxlib >= 0.10: {jax_import_error}")
    set_backend("jax", device="gpu" if args.device == "gpu" else "cpu", intent="parity")
maxiter = 1000 if args.maxiter is None else args.maxiter

print("Running 1_Simple/qfm.py")
print("=======================")

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
if args.use_jax:
    bs = JaxBiotSavart(bs.coils)
bs_tf = BiotSavart(bs.coils)

mpol = 5
ntor = 5
stellsym = True
constraint_weight = 1e0

phis = np.linspace(0, 1/nfp, 25, endpoint=False)
thetas = np.linspace(0, 1, 25, endpoint=False)
s = SurfaceRZFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis,
    quadpoints_theta=thetas)
s.fit_to_curve(ma, 0.2, flip_theta=True)

# First optimize at fixed volume

qfm_class = JaxQfmResidual if args.use_jax else QfmResidual
qfm_surface_class = JaxQfmSurface if args.use_jax else QfmSurface
qfm = qfm_class(s, bs)
qfm.J()

vol = Volume(s)
vol_target = vol.J()

qfm_surface = qfm_surface_class(bs, s, vol, vol_target)

res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=maxiter,
                                                         constraint_weight=constraint_weight)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=maxiter)
print(f"||vol constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

# Now optimize at fixed toroidal flux

tf = ToroidalFlux(s, bs_tf)
tf_target = tf.J()

qfm_surface = qfm_surface_class(bs, s, tf, tf_target)

res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=maxiter,
                                                         constraint_weight=constraint_weight)
print(f"||tf constraint||={0.5*(s.volume()-vol_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=maxiter)
print(f"||tf constraint||={0.5*(tf.J()-tf_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

# Check that volume is not changed
print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

# Now optimize at fixed area

ar = Area(s)
ar_target = ar.J()

qfm_surface = qfm_surface_class(bs, s, ar, ar_target)

res = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(tol=1e-12, maxiter=maxiter,
                                                         constraint_weight=constraint_weight)
print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

res = qfm_surface.minimize_qfm_exact_constraints_SLSQP(tol=1e-12, maxiter=maxiter)
print(f"||area constraint||={0.5*(ar.J()-ar_target)**2:.8e}, ||residual||={np.linalg.norm(qfm.J()):.8e}")

# Check that volume is not changed
print(f"||vol constraint||={0.5*(vol.J()-vol_target)**2:.8e}")

if "DISPLAY" in os.environ:
    s.plot()
print("End of 1_Simple/qfm.py")
print("=======================")
