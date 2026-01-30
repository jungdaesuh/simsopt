from simsopt._core.optimizable import load
from simsopt.field import InterpolatedField
from simsopt.field import SurfaceClassifier, \
    compute_fieldlines, LevelsetStoppingCriterion#, plot_poincare_data
from simsopt.geo import SurfaceRZFourier
from simsopt.geo import curves_to_vtk
import numpy as np
import matplotlib.pyplot as plt

nfieldlines = 150 # Number of field lines for integration 
tmax_fl = 10000 # Maximum toroidal angle for integration
tol = 1e-8 # Tolerance for field line integration
interpolate = True # If True, then the BiotSavart magnetic field is interpolated 
                   # on a grid for the magnetic field evaluation
nr = 20 # Number of radial points for interpolation
nphi = 10 # Number of toroidal angle points for interpolation
ntheta = 64
nz = 10 # Number of vertical points for interpolation
degree = 3 # Degree for interpolation
#nfp = 4
nfp = 5
filename='wout_nfp22ginsburg_000_001490.nc'
#filename = 'wout_NAS_n4_AR6.2.01.nc'

def plot_poincare_data(fieldlines_phi_hits, phis, filename, mark_lost=False, aspect='equal', dpi=300, xlims=None,
                       ylims=None, surf=None, s=2, marker='o'):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    from math import ceil, sqrt
    nrowcol = ceil(sqrt(len(phis)))
    plt.figure()
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    color = None
    for i in range(len(phis)):
        row = i//nrowcol
        col = i % nrowcol
        if i != len(phis) - 1:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc='left', y=0.0)
        else:
            axs[row, col].set_title(f"$\\phi = {phis[i]/np.pi:.2f}\\pi$ ", loc='right', y=0.0)
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$r$")
        if col == 0:
            axs[row, col].set_ylabel("$z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            if mark_lost:
                color = 'r' if lost else 'g'
            data_this_phi = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
            axs[row, col].scatter(r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color)

        plt.rc('axes', axisbelow=True)
        axs[row, col].grid(True, linewidth=0.5)

        # if passed a surface, plot the plasma surface outline
        if surf is not None:
            phi_new = phis[i] * 1 / (2 * np.pi)
            print(phi_new)
            cross_section = surf.cross_section(phi=phi_new)
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c='k')

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()

def plot_equilibrium_surfaces(surf, phis, filename="equilibrium_cross_sections.png"):
    plt.figure(figsize=(6,6))
    for phi in phis:
        cross_section = surf.cross_section(phi=phi)  # shape (N, 3)
        # Append first point at the end to close the loop
        cross_section_closed = np.vstack([cross_section, cross_section[0]])
        x = cross_section_closed[:, 0]
        y = cross_section_closed[:, 1]
        z = cross_section_closed[:, 2]

        R = np.sqrt(x**2 + y**2)

        plt.plot(R, z, label=f"phi={phi:.3f}")

    plt.xlabel("R (cylindrical radius)")
    plt.ylabel("Z (vertical coordinate)")
    plt.axis("equal")
    plt.legend()
    plt.title("Equilibrium Surface Cross-Sections")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def create_surface_from_fit(eq_name_full, surf_range, plas_nPhi, plas_nTheta, surf_s):
    surf_nfp1 = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range=surf_range, nphi=plas_nPhi, ntheta=plas_nTheta)
    surf_plas = SurfaceRZFourier(mpol=surf_nfp1.mpol,ntor=surf_nfp1.ntor,nfp=nfp,stellsym=True,
                                    quadpoints_theta=surf_nfp1.quadpoints_theta,
                                    quadpoints_phi=surf_nfp1.quadpoints_phi)
    surf_plas.least_squares_fit(surf_nfp1.gamma())

    surf_plas.set_dofs(surf_plas.get_dofs()*0.885/surf_plas.major_radius())
    return surf_plas



#bs = load('biotsavart_'+config_name+'.json')
#bs = load('outputs-nfp=4/MR=0.925-TF=0.24-LW=1e-05-CCW=100-CW=0.0001-SR=0.22-Order=2/biot_savart_opt.json')
bs = load('outputs-iota30-poincare_test/MR=0.885-TF=0.24-LW=0.000775-CCW=100-CW=0.000775-SR=0.22-Order=2/biot_savart_opt.json')
surf = create_surface_from_fit(filename, "full torus", 255, 64, 0.24)
#surf = SurfaceRZFourier.from_wout(filename, range="full torus", nphi=255, ntheta=64, s=0.24)
# scale the surface down to the target appropriate major radius
#surf.set_dofs(surf.get_dofs()*0.925/surf.major_radius())
#surf = load('surf_'+config_name+'.json')
# Extend surface, since we want to look at field lines beyond it
#surf_extended = load('surf_'+config_name+'.json')

surf_extended = create_surface_from_fit(filename, "full torus", 255, 64, 0.24)
#surf_extended = SurfaceRZFourier.from_wout(filename, range="full torus", nphi=255, ntheta=64, s=0.24)
# scale the surface down to the target appropriate major radius
#surf_extended.set_dofs(surf_extended.get_dofs()*0.925/surf_extended.major_radius())

surf_extended.extend_via_normal(0.05)

# Use extended surface to determine initial conditions
gamma = surf_extended.gamma()
R = np.sqrt(gamma[:,:,0]**2 + gamma[:,:,1]**2)
Z = gamma[:,:,2]

#nfp = surf.nfp
#nfp = 4

Rmin = np.min(R)
Rmax = np.max(R)
Zmax = np.max(Z)


#phis_to_plot = [(i / 4) * (2 * np.pi / nfp) for i in range(4)]
#phis_to_plot = [phi / 8 % (2*np.pi) for phi in [0, 1, 2, 3]]
phis_to_plot = [phi / 20 % (2*np.pi) for phi in [0, 1, 2, 3]]
#phis_to_plot = [phi % (2*np.pi) for phi in [0, 1, 2, 3]]
plot_equilibrium_surfaces(surf, phis_to_plot)


# The parameter h sets the grid size for the classifier, 
# and p is the order. These parameters are not too critical
# to the Poincare calculation. 
sc_fieldline = SurfaceClassifier(surf_extended, h=0.02, p=2)

def trace_fieldlines(bfield):
    # Set up initial conditions 
    R0 = np.linspace(Rmin, Rmax, nfieldlines)
    Z0 = np.zeros(nfieldlines)
    phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
    #phis = [2*np.pi*i / 5 % (2*np.pi) for i in range(4)]
    #phis = np.linspace(0, 2*np.pi/nfp, 4, endpoint=False) % (2*np.pi)
    #phis = [(phi * np.pi / 2) % (2*np.pi) for phi in [0, 1, 2, 3]]
    # Wider φ sampling
    #phis = [(i/16)*(2*np.pi/nfp) for i in range(16)]
    #phis = [phi % (2*np.pi) for phi in phis]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=tol,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    plot_poincare_data(fieldlines_phi_hits, phis, f'poincare_fieldline.png', dpi=300,surf=surf,mark_lost=False)
    return fieldlines_phi_hits

rrange = (Rmin, Rmax, nr)
phirange = (0, 2*np.pi/nfp, nphi)
# exploit stellarator symmetry and only consider positive z values:
zrange = (0, Zmax, nz)

if interpolate:
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )

    bsh.set_points(surf.gamma().reshape((-1, 3)))
    bs.set_points(surf.gamma().reshape((-1, 3)))
    Bh = bsh.B()
    B = bs.B()
    print("Maximum field interpolation error: ", np.max(np.abs(B-Bh)))
else:
    bsh = bs

hits = trace_fieldlines(bsh)
