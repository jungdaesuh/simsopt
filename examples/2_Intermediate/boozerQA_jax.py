#!/usr/bin/env python3
r"""
``boozerQA.py`` with a JAX switch: the same QA optimization of the NCSX coils
on one Boozer surface, with the Boozer surface solve and its adjoint
(``JaxBoozerSurface``), the non-quasisymmetry ratio
(``JaxNonQuasiSymmetricRatio``) and their Biot-Savart fields in JAX.
``Iotas``, ``MajorRadius``, ``CurveLength``, ``QuadraticPenalty`` and SciPy's
BFGS are upstream's. Install with ``pip install '.[jax]'`` and run from the
repository root::

    python examples/2_Intermediate/boozerQA_jax.py                # JAX on the CPU
    python examples/2_Intermediate/boozerQA_jax.py --device gpu   # JAX on a CUDA GPU
    python examples/2_Intermediate/boozerQA_jax.py --native       # upstream's classes
    python examples/2_Intermediate/boozerQA_jax.py --fused        # one program, native BFGS

``--fused`` combines the exact solve, all four objective terms and one
implicit adjoint. Its immutable warm start is passed in and returned. All
routes use native host BFGS with the same coil coordinates and options.

For the GPU, prepend
``XLA_FLAGS="${XLA_FLAGS:+${XLA_FLAGS} }--xla_gpu_exclude_nondeterministic_ops=true"``
to make XLA's GPU reductions deterministic. The JAX lane matches the native one
to round-off per evaluation. The default Boozer tolerance, 1e-13, sits at
float64's round-off floor of the residual, so a solve can take one Newton step
more or fewer than natively, and BFGS magnifies such differences over many
iterations. Run with ``CI=true`` for 50 iterations.

More details on this work can be found at doi:10.1017/S0022377822000563 or arxiv:2203.03753.
"""

import argparse
import os

import numpy as np
from scipy.optimize import minimize

from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt_jax.backend import set_backend
from simsopt_jax.runtime.host_boundary import host_array
from simsopt_jax_adapters.field import JaxBiotSavart
from simsopt_jax_adapters.geo.boozer_surface import JaxBoozerSurface
from simsopt_jax_adapters.geo.surface_objectives import JaxNonQuasiSymmetricRatio
from simsopt_jax_adapters.geo.single_stage_exact import JaxExactSingleStage

parser = argparse.ArgumentParser(description="boozerQA.py with the Boozer surface and the non-QS ratio in JAX")
parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
route = parser.add_mutually_exclusive_group()
route.add_argument("--native", action="store_true", help="run upstream's native classes instead")
route.add_argument("--fused", action="store_true", help="fuse the exact solve/objective/adjoint; native host BFGS")
args = parser.parse_args()
if not args.native:
    set_backend("jax", device=args.device)

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA_jax.py")
print("====================================")

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
# bs.coils includes all coils after symmetry expansion (not just the base coils).
# You can access them directly like this:
all_curves = [c.curve for c in bs.coils]
current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = 6
ntor = 6
stellsym = True

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

# To generate an initial guess for the surface computation, start with the magnetic axis and extrude outward
s.fit_to_curve(ma, 0.1, flip_theta=True)
iota = -0.406

# Use a volume surface label
vol = Volume(s)
vol_target = vol.J()

## compute the surface
# The JAX switch, part 1: the Boozer surface and its field.
if args.native:
    boozer_surface = BoozerSurface(bs, s, vol, vol_target)
else:
    boozer_surface = JaxBoozerSurface(JaxBiotSavart(bs.coils), s, vol, vol_target)
res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)

out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}")
## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
mr = MajorRadius(boozer_surface)
ls = [CurveLength(c) for c in base_curves]

J_major_radius = QuadraticPenalty(mr, float(np.asarray(mr.J())), 'identity')  # target major radius is that computed on the initial surface
J_iotas = QuadraticPenalty(Iotas(boozer_surface), res['iota'], 'identity')  # target rotational transform is that computed on the initial surface
# The JAX switch, part 2: the non-quasisymmetry ratio and its field.
if args.native:
    J_nonQSRatio = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(bs.coils))
else:
    J_nonQSRatio = JaxNonQuasiSymmetricRatio(boozer_surface, JaxBiotSavart(bs.coils))
total_length = ls[0] + ls[1] + ls[2]
Jls = QuadraticPenalty(total_length, float(total_length.J()), 'max')

# sum the objectives together
JF = J_nonQSRatio + J_iotas + J_major_radius + Jls

curves_to_vtk(all_curves, OUT_DIR + "curves_init")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_init")

# let's fix the coil current
base_currents[0].fix_all()

if args.fused:
    evaluator = JaxExactSingleStage.from_boozer_surface(
        boozer_surface, JaxBiotSavart(bs.coils), base_curves,
        iota_target=float(res['iota']), major_radius_target=J_major_radius.cons,
        length_target=Jls.cons,
    )
    fused_state = evaluator.initial_state


def fun(dofs):
    global fused_state
    if args.fused:
        evaluation = evaluator.evaluate(dofs, fused_state)
        fused_state = evaluation.state
        nonqs, solved_iota, radius, length = evaluation.terms
        print(f"J={evaluation.value:.1e}, J_nonQSRatio={nonqs:.2e}, iota={solved_iota:.2e}, "
              f"mr={radius:.2e}, Len={length:.1f}, ║∇J║={np.linalg.norm(evaluation.gradient):.1e}")
        return evaluation.value, evaluation.gradient

    # save these as a backup in case the boozer surface Newton solve fails
    sdofs_prev = boozer_surface.surface.x
    iota_prev = boozer_surface.res['iota']
    G_prev = boozer_surface.res['G']

    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    if not boozer_surface.res['success']:
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        boozer_surface.surface.x = sdofs_prev
        boozer_surface.res['iota'] = iota_prev
        boozer_surface.res['G'] = G_prev

    cl_string = ", ".join([f"{J.J():.1f}" for J in ls])
    outstr = f"J={J:.1e}, J_nonQSRatio={J_nonQSRatio.J():.2e}, iota={boozer_surface.res['iota']:.2e}, mr={mr.J():.2e}"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in ls):.1f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = np.asarray(JF.x, dtype=np.float64)
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    J1, _ = f(dofs + 2*eps*h)
    J2, _ = f(dofs + eps*h)
    J3, _ = f(dofs - eps*h)
    J4, _ = f(dofs - 2*eps*h)
    print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))

print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
MAXITER = 50 if in_github_actions else 1e3

res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15)
if args.fused:
    evaluation = evaluator.evaluate(np.asarray(res.x, dtype=np.float64), fused_state)
    JF.x = res.x
    final_inner = host_array(evaluation.state.x)
    boozer_surface.surface.set_dofs(final_inner[:-2])
    boozer_surface.res['iota'] = float(final_inner[-2])
    boozer_surface.res['G'] = float(final_inner[-1])
curves_to_vtk(all_curves, OUT_DIR + "curves_opt")
boozer_surface.surface.to_vtk(OUT_DIR + "surf_opt")

print("End of 2_Intermediate/boozerQA_jax.py")
print("====================================")
