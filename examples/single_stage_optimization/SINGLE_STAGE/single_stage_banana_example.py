import argparse
import os
import io
import json
import numpy as np
from shapely.geometry import Polygon
from scipy.optimize import minimize

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, CurveLength, LpCurveCurvature
from simsopt.geo.surfaceobjectives import (
    Volume,
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    SurfaceSurfaceDistance,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import BiotSavart, Coil, Current
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.objectives.utilities import forward_backward
from simsopt._core.optimizable import load, save
from simsopt.field.coil import ScaledCurrent
import matplotlib.pyplot as plt
from simsopt._core.derivative import derivative_dec


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = DATABASE_EQUILIBRIA_DIR if os.path.isdir(DATABASE_EQUILIBRIA_DIR) else os.path.join(EXAMPLE_ROOT, "equilibria")
DEFAULT_LOCAL_STAGE2_ROOT = os.path.join(EXAMPLE_ROOT, "STAGE_2")
DEFAULT_DATABASE_STAGE2_ROOT = os.path.join(REPO_ROOT, "DATABASE", "COIL_OPTIMIZATION", "outputs")
DEFAULT_SINGLE_STAGE_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
DEFAULT_STAGE2_SEEDS_BY_PLASMA = {
    "wout_nfp22ginsburg_000_014417_iota15.nc": {
        "major_radius": 0.915,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "curvature_weight": 0.0001,
        "banana_surf_radius": 0.22,
        "order": 2,
    },
    "wout_nfp22ginsburg_000_002084_iota20.nc": {
        "major_radius": 0.975,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "curvature_weight": 0.0001,
        "banana_surf_radius": 0.22,
        "order": 2,
    },
}


def format_compact_float(value):
    return f"{value:g}"


def format_local_stage2_seed_dir(major_radius, toroidal_flux, length_weight, cc_weight, curvature_weight, banana_surf_radius, order):
    return (
        f"R0={format_compact_float(major_radius)}"
        f"-s={format_compact_float(toroidal_flux)}"
        f"-LW={format_compact_float(length_weight)}"
        f"-CCW={format_compact_float(cc_weight)}"
        f"-CW={format_compact_float(curvature_weight)}"
        f"-SR={banana_surf_radius:0.3f}"
        f"-Order={order}"
    )


def format_database_stage2_seed_dir(major_radius, toroidal_flux, length_weight, cc_weight, curvature_weight, banana_surf_radius, order):
    return (
        f"MR={format_compact_float(major_radius)}"
        f"-TF={format_compact_float(toroidal_flux)}"
        f"-LW={format_compact_float(length_weight)}"
        f"-CCW={format_compact_float(cc_weight)}"
        f"-CW={format_compact_float(curvature_weight)}"
        f"-SR={format_compact_float(banana_surf_radius)}"
        f"-Order={order}"
    )


def build_stage2_bs_path(args):
    if args.stage2_bs_path:
        return args.stage2_bs_path

    if args.stage2_source == "database":
        seed_dir = format_database_stage2_seed_dir(
            args.stage2_seed_major_radius,
            args.stage2_seed_toroidal_flux,
            args.stage2_seed_length_weight,
            args.stage2_seed_cc_weight,
            args.stage2_seed_curvature_weight,
            args.stage2_seed_banana_surf_radius,
            args.stage2_seed_order,
        )
        return os.path.join(
            args.database_stage2_root,
            f"outputs-{args.plasma_surf_filename}",
            seed_dir,
            "biot_savart_opt.json",
        )

    seed_dir = format_local_stage2_seed_dir(
        args.stage2_seed_major_radius,
        args.stage2_seed_toroidal_flux,
        args.stage2_seed_length_weight,
        args.stage2_seed_cc_weight,
        args.stage2_seed_curvature_weight,
        args.stage2_seed_banana_surf_radius,
        args.stage2_seed_order,
    )
    return os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        seed_dir,
        "biot_savart_opt.json",
    )


def load_stage2_results(stage2_bs_path):
    stage2_results_path = os.path.join(os.path.dirname(stage2_bs_path), "results.json")
    with open(stage2_results_path, "r", encoding="utf-8") as infile:
        stage2_results = json.load(infile)
    return stage2_results_path, stage2_results


def build_equilibrium_path(args):
    if args.equilibrium_path is not None:
        return args.equilibrium_path

    candidate_paths = [
        os.path.join(args.equilibria_dir, args.plasma_surf_filename),
        os.path.join(DATABASE_EQUILIBRIA_DIR, args.plasma_surf_filename),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return candidate_paths[0]


def apply_default_stage2_seed_args(args):
    default_seed = DEFAULT_STAGE2_SEEDS_BY_PLASMA.get(args.plasma_surf_filename, {})
    if args.stage2_seed_major_radius is None:
        args.stage2_seed_major_radius = default_seed.get("major_radius", 0.915)
    if args.stage2_seed_toroidal_flux is None:
        args.stage2_seed_toroidal_flux = default_seed.get("toroidal_flux", 0.24)
    if args.stage2_seed_length_weight is None:
        args.stage2_seed_length_weight = default_seed.get("length_weight", 0.0005)
    if args.stage2_seed_cc_weight is None:
        args.stage2_seed_cc_weight = default_seed.get("cc_weight", 100.0)
    if args.stage2_seed_curvature_weight is None:
        args.stage2_seed_curvature_weight = default_seed.get("curvature_weight", 0.0001)
    if args.stage2_seed_banana_surf_radius is None:
        args.stage2_seed_banana_surf_radius = default_seed.get("banana_surf_radius", 0.22)
    if args.stage2_seed_order is None:
        args.stage2_seed_order = default_seed.get("order", 2)
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-stage Boozer/quasi-symmetry optimization from a Stage 2 seed.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=os.environ.get("PLASMA_SURF_FILENAME", "wout_nfp22ginsburg_000_014417_iota15.nc"),
        help="VMEC wout filename under the equilibria directory.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=os.environ.get("EQUILIBRIA_DIR", DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains the equilibrium wout files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=os.environ.get("EQUILIBRIUM_PATH"),
        help="Explicit path to the equilibrium file. Overrides --equilibria-dir.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get("SINGLE_STAGE_OUTPUT_ROOT", DEFAULT_SINGLE_STAGE_OUTPUT_ROOT),
        help="Directory where the single-stage output family will be written.",
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=float(os.environ["BANANA_SURF_RADIUS"]) if "BANANA_SURF_RADIUS" in os.environ else None,
        help="Coil surface minor radius. Defaults to the Stage 2 seed radius when omitted.",
    )
    parser.add_argument("--nphi", type=int, default=int(os.environ.get("NPHI", "255")))
    parser.add_argument("--ntheta", type=int, default=int(os.environ.get("NTHETA", "64")))
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build the initial Boozer surface, write init artifacts, and skip the optimizer.",
    )
    parser.add_argument("--mpol", type=int, default=int(os.environ.get("MPOL", "8")))
    parser.add_argument("--ntor", type=int, default=int(os.environ.get("NTOR", "6")))
    parser.add_argument("--vol-target", type=float, default=float(os.environ.get("VOL_TARGET", "0.10")))
    parser.add_argument("--constraint-weight", type=float, default=float(os.environ.get("CONSTRAINT_WEIGHT", "1.0")))
    parser.add_argument("--maxiter", type=int, default=int(os.environ.get("MAXITER", "300")))
    parser.add_argument("--iota-target", type=float, default=float(os.environ.get("IOTA_TARGET", "0.15")))
    parser.add_argument("--num-tf-coils", type=int, default=int(os.environ.get("NUM_TF_COILS", "20")))
    parser.add_argument(
        "--boozer-stage",
        choices=["initial", "final"],
        default=os.environ.get("BOOZER_STAGE", "initial"),
        help="Use least-squares Boozer residual during initial stage or exact residual during final stage.",
    )
    parser.add_argument("--cc-dist", type=float, default=float(os.environ.get("CC_DIST", "0.05")))
    parser.add_argument("--curvature-threshold", type=float, default=float(os.environ.get("CURVATURE_THRESHOLD", "20")))
    parser.add_argument("--cc-weight", type=float, default=float(os.environ.get("CC_WEIGHT", "100")))
    parser.add_argument("--curvature-weight", type=float, default=float(os.environ.get("CURVATURE_WEIGHT", "0.1")))
    parser.add_argument(
        "--stage2-source",
        choices=["database", "local"],
        default=os.environ.get("STAGE2_SOURCE", "database"),
        help="Resolve the Stage 2 seed from the archive database or from local STAGE_2 outputs.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=os.environ.get("STAGE2_BS_PATH"),
        help="Explicit path to the Stage 2 biot_savart_opt.json seed. Overrides all derived seed settings.",
    )
    parser.add_argument(
        "--local-stage2-root",
        default=os.environ.get("LOCAL_STAGE2_ROOT", DEFAULT_LOCAL_STAGE2_ROOT),
        help="Directory that contains local STAGE_2 outputs-[plasma]/... runs.",
    )
    parser.add_argument(
        "--database-stage2-root",
        default=os.environ.get("DATABASE_STAGE2_ROOT", DEFAULT_DATABASE_STAGE2_ROOT),
        help="Directory that contains DATABASE/COIL_OPTIMIZATION/outputs.",
    )
    parser.add_argument(
        "--stage2-seed-major-radius",
        type=float,
        default=float(os.environ["STAGE2_SEED_MAJOR_RADIUS"]) if "STAGE2_SEED_MAJOR_RADIUS" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-toroidal-flux",
        type=float,
        default=float(os.environ["STAGE2_SEED_TOROIDAL_FLUX"]) if "STAGE2_SEED_TOROIDAL_FLUX" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-length-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_LENGTH_WEIGHT"]) if "STAGE2_SEED_LENGTH_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-cc-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CC_WEIGHT"]) if "STAGE2_SEED_CC_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-curvature-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CURVATURE_WEIGHT"]) if "STAGE2_SEED_CURVATURE_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-banana-surf-radius",
        type=float,
        default=float(os.environ["STAGE2_SEED_BANANA_SURF_RADIUS"]) if "STAGE2_SEED_BANANA_SURF_RADIUS" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-order",
        type=int,
        default=int(os.environ["STAGE2_SEED_ORDER"]) if "STAGE2_SEED_ORDER" in os.environ else None,
    )
    return parser.parse_args()


class BoozerResidualExact(Optimizable):
    r"""
    This term returns the Boozer residual penalty term
    
    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.
    
    where
    
    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]
    
    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        # same number of points as on the solved surface
        nphis = in_surface.quadpoints_phi.size
        phis = np.linspace(0,1./in_surface.nfp,nphis*4,endpoint=False)
        nthetas = in_surface.quadpoints_theta.size
        thetas = np.linspace(0,1,nthetas*4,endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        print("warning: constraint weight set to 0")
        self.constraint_weight = 0.0
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """
        
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))
 
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=True)
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)
        
        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil
        
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil
        
    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        
        surface = self.surface
        res = self.boozer_surface.res
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=True)

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)
        
        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB

def initialize_boozer_surface(surf_prev, mpol, ntor, bs, vol_target, constraint_weight, iota, G0):
    """
    This initializes the boozer surface, using either the boozer "exact" algorithm, or the boozer "least squares" algorithm
    
    surf_prev: Any instance of simsopt.geo.Surface. This is the initial guess for the boozer surface solver
    mpol: SurfaceXYZTensorFourier resolution (both toroidal and poloidal)
    bs: simsopt.field.BiotSavart instance
    vol_target: target volume to be enclosed by the boozer surface
    constraint_weight: Set to 1.0 to use Boozer least square, None to use Boozer exact
    iota: initial guess for iota value on the surface
    G0: Value of net current going through the torus hole
    """
    surf = SurfaceXYZTensorFourier(
          mpol=mpol,ntor=ntor,nfp=5,stellsym=True,
          quadpoints_theta=surf_prev.quadpoints_theta,
          quadpoints_phi=surf_prev.quadpoints_phi
          )
    surf.least_squares_fit(surf_prev.gamma())

    if constraint_weight is not None:
        # Boozer least square approach
        print("Generating Boozer least squares surface...")
        vol = Volume(surf)
        boozer_surface = BoozerSurface(bs, surf, vol, vol_target, constraint_weight, options={'verbose':True})
    else:
        # Boozer exact approach
        print("Generating Boozer exact surface...")
        surf_exact = SurfaceXYZTensorFourier(
              mpol=mpol,ntor=ntor,nfp=5,stellsym=True,
              quadpoints_theta=np.linspace(0,1,2*mpol+1,endpoint=False),
              quadpoints_phi=np.linspace(0,1./surf.nfp,2*ntor+1,endpoint=False),
              dofs=surf.dofs
              )
    
        vol = Volume(surf_exact)
        boozer_surface = BoozerSurface(bs, surf_exact, vol, vol_target, None, options={'verbose':True})

    # Run boozer surface algorithm
    res = boozer_surface.run_code(iota, G0)
    print(f"G0 from solve: {res['G']}")
    print(f"iota from solve: {res['iota']}")

    # Check if boozer algo is successful
    success1 = res['success'] # True if the boozer surface algo converged
    success2 = not boozer_surface.surface.is_self_intersecting() # True if surface is not self intersecting
    success = success1 and success2
    if not success:
        print(
            "Boozer initialization failed: "
            f"solve_success={success1}, "
            f"self_intersecting={not success2}, "
            f"volume={boozer_surface.surface.volume()}, "
            f"iota_guess={iota}, "
            f"iota_solved={res['iota']}"
        )
        raise RuntimeError("Something went wrong with the Boozer solve...")

    return boozer_surface

def normPlot(surf, bs, filename):
    # Plot the normal magnetic field on the plasma surface
    theta = surf.quadpoints_theta
    phi = surf.quadpoints_phi
    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    unitn = n * (1./absn)[:,:,None]
    sqrt_area = np.sqrt(absn.reshape((-1,1))/float(absn.size))
    surf_area = sqrt_area**2
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bfinal = bs.B().reshape(n.shape)
    Bfinal_norm = np.sum(Bfinal * unitn, axis=2)[:, :, None]
    modBfinal = np.sqrt(np.sum(Bfinal**2, axis=2))[:, :, None]
    relBfinal_norm = Bfinal_norm / modBfinal
    abs_relBfinal_norm_dA = np.abs(relBfinal_norm.reshape((-1, 1))) * surf_area
    mean_abs_relBfinal_norm = np.sum(abs_relBfinal_norm_dA) / np.sum(surf_area)
    max_rnorm = np.max(np.abs(relBfinal_norm))
    relBfinal_norm = np.sum(bs.B().reshape((nphi, ntheta, 3)) * surf.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3))**2, axis=2))[:, :, None]
    fig, ax = plt.subplots()
    contour = ax.contourf(phi, theta, np.squeeze(relBfinal_norm).T, levels=50, cmap='seismic', vmin=-max_rnorm, vmax=max_rnorm)
    ax.set_xlabel(r'$\phi/2\pi$', fontsize=18, fontweight='bold')
    ax.set_ylabel(r'$\theta/2\pi$', fontsize=18, fontweight='bold')
    cbar = fig.colorbar(contour, ax=ax)
    cbar.ax.set_ylabel(r'$\mathbf{B}\cdot\mathbf{n}/|\mathbf{B}|$', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(axis='y', which='major', labelsize=14)
    ax.set_title(f'Surface-averaged \n |Bn|/|B| = {mean_abs_relBfinal_norm:.4e}', fontsize=18, fontweight='bold')
    plt.savefig(f"{filename}.png")
    plt.close()
    return mean_abs_relBfinal_norm

def crossSectionPlot(surf_coils, surf, banana_curve, filename):
    # plots cross section of plasma at a few toroidal locations and relevant HBT cross sections
    plt.figure(figsize=(7,6))
    cs2 = surf_coils.cross_section(0)
    rs2 = np.sqrt(cs2[:,0]**2 + cs2[:,1]**2); rs2 = np.append(rs2, rs2[0])
    zs2 = cs2[:,2]; zs2 = np.append(zs2, zs2[0])    
    plt.plot(rs2, zs2, label='Banana Surface')
    cs3 = hbt.cross_section(0)
    rs3 = np.sqrt(cs3[:,0]**2 + cs3[:,1]**2); rs3 = np.append(rs3, rs3[0])
    zs3 = cs3[:,2]; zs3 = np.append(zs3, zs3[0])
    hbt_poly = Polygon(zip(rs3, zs3))
    plt.plot(rs3, zs3, label='HBT LCFS')
    cs_vv = VV.cross_section(0)
    rs_vv = np.sqrt(cs_vv[:, 0]**2 + cs_vv[:, 1]**2); zs_vv = cs_vv[:, 2]
    rs_vv = np.append(rs_vv, rs_vv[0]); zs_vv = np.append(zs_vv, zs_vv[0])
    plt.plot(rs_vv, zs_vv, label='Vacuum Vessel')
    phi_array = np.linspace(0, 2*np.pi / surf_coils.nfp * 4/5, 5)
    for phi_slice in phi_array:
        cs = surf.cross_section(phi_slice * 2 * np.pi)
        rs = np.sqrt(cs[:,0]**2 + cs[:,1]**2); rs = np.append(rs, rs[0])
        zs = cs[:,2]; zs = np.append(zs, zs[0])
        plt.plot(rs, zs, label=f'Φ={phi_slice/np.pi:0.2f}π')
    plt.xlabel('R [m]', fontsize=18, fontweight='bold')
    plt.ylabel('Z [m]', fontsize=18, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1), fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.minorticks_on()
    plt.grid(True)
    plt.savefig(f"{filename}.png")
    plt.close()
    return True


def fun(x):
    """
    Objective function for L-BFGS-B optimization.

    Evaluates the total objective function and its gradient for a given set of
    degrees of freedom (coil parameters). Attempts to solve for a valid Boozer
    surface; if unsuccessful (solver failure or self-intersection), returns an
    elevated objective value with the last accepted gradient to trigger line
    search backtracking without corrupting the L-BFGS-B Hessian approximation.

    Args:
        x: Current degrees of freedom (coil parameters)

    Returns:
        J: Objective function value
        dJ: Gradient of objective function
    """
    dx = np.linalg.norm(x - run_dict['x_prev'])
    run_dict['x_prev'] = x.copy()
    print(f"Step size: {dx:.2e}")

    run_dict['lscount']+=1

    # initialize to last accepted surface values
    boozer_surface.surface.x = run_dict['sdofs']
    boozer_surface.res['iota'] = run_dict['iota']
    boozer_surface.res['G'] = run_dict['G']

    # Set new coil dofs
    JF.x = x

    # Run boozer surface
    res = boozer_surface.run_code(run_dict['iota'], run_dict['G'])

    # Check success
    try:
        success1 = boozer_surface.res['success']
        success2 = not boozer_surface.surface.is_self_intersecting()
    except Exception as e:
        print("Surface check failed:", e)
        success2 = False
    success = success1 and success2

    if success:
        J = JF.J()
        dJ = JF.dJ()

        print(f"Volume: {boozer_surface.surface.volume()}")
        print(f"Iota: {Iotas(boozer_surface).J()}")

    else:
        print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not success1:
            print("Boozer solver failed")
        if not success2:
            print("Surface is self-intersecting")

        # Elevated J violates Armijo, so the line search backtracks.
        # Returning dJ_old (not negated) avoids the old -dJ corruption path
        # and produces y_k=0 if the step is ever accepted, safely skipping
        # the BFGS Hessian update.
        J = run_dict['J'] + max(abs(run_dict['J']), 1.0)
        dJ = run_dict['dJ'].copy()
        boozer_surface.surface.x = run_dict['sdofs']
        boozer_surface.res['iota'] = run_dict['iota']
        boozer_surface.res['G'] = run_dict['G']

    return J, dJ

def callback(x):
    """
    Callback function executed after each successful optimization iteration.

    Stores the accepted state (surface DOFs, iota, G), evaluates and prints
    detailed diagnostics for all objective function components, and logs the
    iteration summary to file. Used for monitoring optimization progress and
    recording convergence history.

    Args:
        x: Current degrees of freedom (coil parameters) from accepted step
    """
    # Update count for tracking
    run_dict['lscount'] = 0

    # Store last accepted state
    run_dict['sdofs'] = boozer_surface.surface.x.copy()
    run_dict['iota'] = boozer_surface.res['iota']
    run_dict['G'] = boozer_surface.res['G']
    run_dict['J'] = JF.J()
    run_dict['dJ'] = JF.dJ().copy()

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']
    
    J_QS = JnonQSRatio.J()
    dJ_QS = np.linalg.norm(JnonQSRatio.dJ())
    J_Boozer = JBoozerResidual.J()
    dJ_Boozer = np.linalg.norm(JBoozerResidual.dJ())
    J_iota = Jiota.J()
    dJ_iota = np.linalg.norm(Jiota.dJ())
    J_len = JCurveLength.J()
    dJ_len = np.linalg.norm(JCurveLength.dJ())
    J_cc = JCurveCurve.J()
    dJ_cc = np.linalg.norm(JCurveCurve.dJ())
    J_cs = JCurveSurface.J()
    dJ_cs = np.linalg.norm(JCurveSurface.dJ())
    J_surf = JSurfSurf.J()
    dJ_surf = np.linalg.norm(JSurfSurf.dJ())
    J_curvature = JCurvature.J()
    dJ_curvature = np.linalg.norm(JCurvature.dJ())

    iota_str = f"{iota.J():.4f}"
    volume_str = f"{boozer_surface.surface.volume():.4f}"

    max_r = np.max(np.sqrt(banana_curve.gamma()[:,0]**2 + banana_curve.gamma()[:,1]**2))
    max_z = np.max(np.abs(banana_curve.gamma()[:,2]))
    max_curvature = np.max(banana_curve.kappa())
    length = curvelength.J()
    curvecurve_min = JCurveCurve.shortest_distance()
    curvesurf_min = JCurveSurface.shortest_distance()

    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * boozer_surface.surface.unitnormal(), axis=2)))
    run_dict['intersecting'] = boozer_surface.surface.is_self_intersecting()

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    print(f"{'ι Penalty':{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'Curve Length Penalty':{width}} = {J_len:.6e} (dJ = {dJ_len:.6e})", file=buffer)
    print(f"{'Curve-Curve Penalty':{width}} = {J_cc:.6e} (min={curvecurve_min:.3e}) (dJ = {dJ_cc:.6e})", file=buffer)
    print(f"{'Curve-Surface Penalty':{width}} = {J_cs:.6e} (min={curvesurf_min:.3e}) (dJ = {dJ_cs:.6e})", file=buffer)
    print(f"{'Surf-Vessel Penalty':{width}} = {J_surf:.6e} (dJ = {dJ_surf:.6e})", file=buffer) 
    print(f"{'Curvature Penalty':{width}} = {J_curvature:.6e} (dJ = {dJ_curvature:.6e})", file=buffer) 
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)

    print(f"{'Intersecting':{width}} = {run_dict['intersecting']}", file=buffer)
    print(f"{'Max Curve R':{width}} = {max_r:.6e}", file=buffer)
    print(f"{'Max Curve Z':{width}} = {max_z:.6e}", file=buffer)
    print(f"{'Max Curvature':{width}} = {max_curvature:.6e}", file=buffer)
    print(f"{'Curve Length':{width}} = {length:.6e}", file=buffer)
    print("="*70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    filename = OUT_DIR_ITER + "/log.txt"
    with open(filename, "a") as f:
        f.write(output_str + "\n")

    # Advance iteration counter
    run_dict['it'] += 1


if __name__ == "__main__":
    # ==============================================================================
    # CONFIGURATION PARAMETERS
    # ==============================================================================
    args = apply_default_stage2_seed_args(parse_args())
    stage2_bs_path = build_stage2_bs_path(args)
    stage2_results_path, stage2_results = load_stage2_results(stage2_bs_path)
    R0 = float(stage2_results["MAJOR_RADIUS"])
    s = float(stage2_results["TOROIDAL_FLUX"])
    order = int(stage2_results.get("order", args.stage2_seed_order))

    banana_surf_radius = args.banana_surf_radius if args.banana_surf_radius is not None else float(stage2_results["banana_surf_radius"])
    banana_surf_nfp = 5
    nphi = args.nphi
    ntheta = args.ntheta
    mpol = args.mpol
    ntor = args.ntor

    # Optimization targets and weights
    vol_target = args.vol_target
    CONSTRAINT_WEIGHT = args.constraint_weight
    MAXITER = args.maxiter
    iota_target = args.iota_target
    num_tf_coils = args.num_tf_coils

    # Convergence tolerances for different mpol values
    ftol_by_mpol = {8: 1e-5, 9: 5e-6, 10: 1e-6, 11: 5e-7, 12: 1e-7, 13: 5e-8, 14: 1e-8, 15: 5e-9, 16: 1e-9, 17: 5e-10, 18: 1e-10}
    gtol_by_mpol = {8: 1e-2, 9: 5e-3, 10: 1e-3, 11: 5e-4, 12: 1e-4, 13: 5e-5, 14: 1e-5, 15: 5e-6, 16: 1e-6, 17: 5e-7, 18: 1e-7}

    # Output directory setup
    OUT_DIR = args.output_root
    os.makedirs(OUT_DIR, exist_ok=True)
    boozer_type = {'initial': 'least_squares', 'final': 'exact'}  # example
    stage = args.boozer_stage

    # ==============================================================================
    # SURFACE GEOMETRY DEFINITIONS
    # ==============================================================================
    # The outer vacuum vessel of HBT, R0 = 0.976, a = 0.222
    # Solely for visualization purposes
    VV = SurfaceRZFourier(nfp=5, stellsym=True)
    VV.set_rc(0, 0, 0.976)
    VV.set_rc(1, 0, 0.222)
    VV.set_zs(1, 0, 0.222)

    # The proposed new HBT LCFS
    hbt = SurfaceRZFourier(nfp=5, stellsym=True)
    hbt.set_rc(0, 0, 0.9115)    # R0 of LCFS semi-circle center
    hbt.set_rc(1, 0, 0.1605)    # Minor radius (thick metal walls)
    hbt.set_zs(1, 0, 0.152)    # Z extent = ±0.152 m (flat top/bottom)

    # The surface the coils can lie on from Jeff - R0 = 0.976 and a=0.22
    surf_coils = SurfaceRZFourier(nfp=banana_surf_nfp, stellsym=True)
    surf_coils.set_rc(0, 0, 0.976)
    surf_coils.set_rc(1, 0, banana_surf_radius)
    surf_coils.set_zs(1, 0, banana_surf_radius)

    # ==============================================================================
    # LOAD EQUILIBRIUM AND COILS
    # ==============================================================================
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)
    bs = load(stage2_bs_path)

    # Initialize the boundary magnetic surface and scale it to the target major radius
    surf = SurfaceRZFourier.from_wout(file_loc, range="half period", nphi=nphi, ntheta=ntheta, s=s)
    # scale the surface down to the target appropriate major radius
    surf.set_dofs(surf.get_dofs()*R0/surf.major_radius())

    # Extract coil information
    coils = bs.coils
    curves = [c.curve for c in coils]
    tf_coils = coils[:num_tf_coils]
    tf_curves = [c.curve for c in tf_coils]
    banana_coils = coils[num_tf_coils:]
    banana_curves = [c.curve for c in banana_coils]
    banana_curve = banana_curves[0]
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)

    # Calculate G0 parameter from TF coil currents
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    # ==============================================================================
    # OPTIMIZATION SETUP
    # ==============================================================================
    print(f"\n===== Starting single stage optimization for mpol = {mpol} =====")

    OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # Initialize Boozer surface with target parameters
    boozer_surface = initialize_boozer_surface(surf, mpol, ntor, bs, vol_target, CONSTRAINT_WEIGHT, iota_target, G0)

    # ==============================================================================
    # SAVE INITIAL STATE
    # ==============================================================================
    # Save initial coil configurations
    curves_to_vtk(curves, OUT_DIR_ITER + f"/curves_init", close=True)
    bs.save(OUT_DIR_ITER + f"/biot_savart_init.json")

    # Save initial surface with magnetic field normal component data
    pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
        boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3))**2, axis=2))[:, :, None]}
    boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_init", extra_data=pointData)
    boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_init.json")
    print(f"Volume: {boozer_surface.surface.volume()}")

    # Generate initial diagnostic plots
    initial_field_error = normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotInitial")
    crossSectionPlot(surf_coils, boozer_surface.surface, banana_curve, OUT_DIR_ITER + "/CrossSectionInitial")
    initial_volume = boozer_surface.surface.volume()
    initial_iota = Iotas(boozer_surface).J()
    initial_max_curvature = np.max(banana_curve.kappa())

    # ==============================================================================
    # DEFINE OBJECTIVE FUNCTION COMPONENTS
    # ==============================================================================
    # Biot-Savart field calculation
    bs_obj = BiotSavart(coils)

    # Quasi-symmetry and Boozer coordinate residuals
    nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
    if boozer_type[stage] == 'exact':
        brs = [BoozerResidualExact(boozer_surface, bs_obj)]
    else:
        brs = [BoozerResidual(boozer_surface, bs_obj)]

    # Objective function weights and parameters
    LENGTH_WEIGHT = 1
    RES_WEIGHT = 1e3
    IOTAS_WEIGHT = 1e2
    CC_WEIGHT = args.cc_weight
    CC_DIST = args.cc_dist
    CS_WEIGHT = 1
    CS_DIST = 0.02
    SURF_DIST_WEIGHT = 1e3
    SS_DIST = 0.04
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = args.curvature_threshold
    phi_list = np.linspace(0, 1 / boozer_surface.surface.nfp, 5)

    # Individual objective terms
    iota = Iotas(boozer_surface)
    curvelength = CurveLength(banana_curves[0])
    length_target = curvelength.J()

    Jiota = QuadraticPenalty(iota, iota_target)
    JnonQSRatio = sum(nonQSs)
    JBoozerResidual = sum(brs)
    JCurveLength = QuadraticPenalty(curvelength, length_target, 'max')
    JCurveCurve = CurveCurveDistance(curves, CC_DIST)
    JCurveSurface = CurveSurfaceDistance(curves, boozer_surface.surface, CS_DIST)
    JSurfSurf = SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)
    JCurvature = LpCurveCurvature(banana_curves[0], 2, CURVATURE_THRESHOLD)

    # Combined objective function
    JF = JnonQSRatio + RES_WEIGHT * JBoozerResidual + IOTAS_WEIGHT * Jiota \
      + LENGTH_WEIGHT * JCurveLength + CC_WEIGHT * JCurveCurve \
        + CS_WEIGHT * JCurveSurface + SURF_DIST_WEIGHT * JSurfSurf \
        + CURVATURE_WEIGHT * JCurvature

    # Extract degrees of freedom
    dofs = JF.x

    # ==============================================================================
    # INITIALIZE OPTIMIZATION STATE
    # ==============================================================================
    # Initialize run_dict after JF and boozer_surface are ready
    run_dict = {
        'sdofs': boozer_surface.surface.x.copy(),
        'iota': boozer_surface.res['iota'],
        'G': boozer_surface.res['G'],
        'J': JF.J(),
        'dJ': JF.dJ().copy(),
        'it': 1,
        'lscount': 0,
        'x_prev': dofs.copy(),
        'intersecting': False,
    }

    # ==============================================================================
    # RUN OPTIMIZATION
    # ==============================================================================
    # Get convergence tolerances for current mpol
    ftol = ftol_by_mpol.get(mpol)
    gtol = gtol_by_mpol.get(mpol)

    if args.init_only:
        res_nit = 0
        final_volume = initial_volume
        final_iota = initial_iota
        final_max_curvature = initial_max_curvature
        fieldError = initial_field_error
        print("Skipping single-stage optimizer because --init-only was provided.")
    else:
        # Run L-BFGS-B optimization
        res = minimize(fun, dofs, jac=True, method='L-BFGS-B', callback=callback, options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': ftol, 'gtol': gtol})
        res_nit = res.nit
        print(res.message)

        # ==============================================================================
        # SAVE OPTIMIZED STATE
        # ==============================================================================
        # Save optimized coil configurations
        curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True)
        bs.save(OUT_DIR_ITER + "/biot_savart_opt.json")

        # Save optimized surface with magnetic field normal component data
        pointData = {"B_N/B": np.sum(bs.B().reshape((nphi, ntheta, 3)) *
            boozer_surface.surface.unitnormal(), axis=2)[:, :, None] / np.sqrt(np.sum(bs.B().reshape((nphi, ntheta, 3))**2, axis=2))[:, :, None]}

        # Print final results
        boozer_surface.surface.to_vtk(OUT_DIR_ITER + f"/surf_opt", extra_data=pointData)
        boozer_surface.surface.save(OUT_DIR_ITER + f"/surf_opt.json")

        final_volume = boozer_surface.surface.volume()
        final_iota = Iotas(boozer_surface).J()
        final_max_curvature = np.max(banana_curve.kappa())
        print(f"Volume: {final_volume}")
        print(f"Iota: {final_iota}")
        print(f"Max Curvature: {final_max_curvature}")

        # Generate final diagnostic plots
        fieldError = normPlot(boozer_surface.surface, bs, OUT_DIR_ITER + "/NormPlotOptimized")
        crossSectionPlot(surf_coils, boozer_surface.surface, banana_curve, OUT_DIR_ITER + "/CrossSectionOptimized")

    # Save the results of optimization to a separate file
    results = {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "STAGE2_SOURCE": args.stage2_source,
        "STAGE2_BS_PATH": stage2_bs_path,
        "STAGE2_RESULTS_PATH": stage2_results_path,
        "STAGE2_SEED_MAJOR_RADIUS": R0,
        "STAGE2_SEED_TOROIDAL_FLUX": s,
        "STAGE2_SEED_BANANA_SURF_RADIUS": float(stage2_results["banana_surf_radius"]),
        "STAGE2_SEED_ORDER": order,
        "CC_DIST": CC_DIST,
        "CC_WEIGHT": CC_WEIGHT,
        "CURVATURE_WEIGHT": CURVATURE_WEIGHT,
        "CURVATURE_THRESHOLD": CURVATURE_THRESHOLD,
        "LENGTH_WEIGHT": LENGTH_WEIGHT,
        "MAJOR_RADIUS": R0,
        "TOROIDAL_FLUX": s,
        "banana_surf_radius": banana_surf_radius,
        "order": order,
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "iterations": res_nit,
        "TARGET_VOLUME": float(vol_target),
        "TARGET_IOTA": float(iota_target),
        "FINAL_VOLUME": float(final_volume),
        "FINAL_IOTA": float(final_iota),
        "FIELD_ERROR": float(fieldError),
        "SELF_INTERSECTING": run_dict.get('intersecting', False),
        "MAX_CURVATURE": float(final_max_curvature),
        "INITIAL_VOLUME": float(initial_volume),
        "INITIAL_IOTA": float(initial_iota),
        "INITIAL_FIELD_ERROR": float(initial_field_error),
        "INITIAL_MAX_CURVATURE": float(initial_max_curvature),
    }
    with open(os.path.join(OUT_DIR_ITER, "results.json"), "w") as outfile:
        json.dump(results, outfile, indent=2)
