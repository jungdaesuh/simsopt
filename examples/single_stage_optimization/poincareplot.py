#!/usr/bin/env python3

from simsopt._core.optimizable import load
from simsopt.field import InterpolatedField
from simsopt.field import SurfaceClassifier, \
    compute_fieldlines, LevelsetStoppingCriterion#, plot_poincare_data
from simsopt.geo import SurfaceRZFourier
import os
import pandas as pd
import glob
import json
import numpy as np


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
    import matplotlib.pyplot as plt
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
            cross_section = surf.cross_section(phi=phi_new)
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c='k')

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()

def create_surface_from_fit(eq_name_full, surf_range, plas_nPhi, plas_nTheta, surf_s, R0):
    surf_nfp1 = SurfaceRZFourier.from_wout(eq_name_full, s=surf_s, range=surf_range, nphi=plas_nPhi, ntheta=plas_nTheta)
    surf_plas = SurfaceRZFourier(mpol=surf_nfp1.mpol,ntor=surf_nfp1.ntor,nfp=nfp,stellsym=True,
                                    quadpoints_theta=surf_nfp1.quadpoints_theta,
                                    quadpoints_phi=surf_nfp1.quadpoints_phi)
    surf_plas.least_squares_fit(surf_nfp1.gamma())

    surf_plas.set_dofs(surf_plas.get_dofs()*R0/surf_plas.major_radius())
    return surf_plas

def poincarePlot(dir, R0, s): 
    #surf = SurfaceRZFourier.from_wout(filename, range="full torus", nphi=255, ntheta=64, s=s)
    surf = create_surface_from_fit(filename, "full torus", 255, 64, s, R0)
    # scale the surface down to the target appropriate major radius
    #surf.set_dofs(surf.get_dofs()*R0/surf.major_radius())

    bs = load(dir + '/biot_savart_opt.json')
    surf_extended = create_surface_from_fit(filename, "full torus", 255, 64, s, R0)
    #surf_extended = SurfaceRZFourier.from_wout(filename, range="full torus", nphi=255, ntheta=64, s=s)
    # scale the surface down to the target appropriate major radius
    #surf_extended.set_dofs(surf_extended.get_dofs()*R0/surf_extended.major_radius())
    # Extend surface, since we want to look at field lines beyond it
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

    # The parameter h sets the grid size for the classifier, 
    # and p is the order. These parameters are not too critical
    # to the Poincare calculation. 
    sc_fieldline = SurfaceClassifier(surf_extended, h=0.02, p=2)


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

    def trace_fieldlines(bfield, dir):
        # Set up initial conditions 
        print(f"Creating Poincare plot for directory {dir}")
        R0 = np.linspace(Rmin, Rmax, nfieldlines)
        Z0 = np.zeros(nfieldlines)
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=tol,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        plot_poincare_data(fieldlines_phi_hits, phis, f'{dir}/PoincarePlot.png', dpi=300,surf=surf,mark_lost=False)
        return fieldlines_phi_hits

    hits = trace_fieldlines(bsh, dir)


nfieldlines = 150 # Number of field lines for integration 
tmax_fl = 10000 # Maximum toroidal angle for integration
tol = 1e-8 # Tolerance for field line integration
interpolate = True # If True, then the BiotSavart magnetic field is interpolated 
# on a grid for the magnetic field evaluation
nr = 20 # Number of radial points for interpolation
nphi = 10 # Number of toroidal angle points for interpolation
nfp = 5
nz = 10 # Number of vertical points for interpolation
degree = 3 # Degree for interpolation

#filename = 'wout_NAS_n4_AR6.2.01.nc'
filename = 'wout_nfp22ginsburg_000_014417_iota15.nc'
# Define the common parent directory
parent_dir = f'outputs-{filename}/'
#parent_dir = f'outputs-iota30-poincare_test'
'''parent_dir = f'outputs-nfp=4/MR=0.925-TF=0.24-LW=1e-05-CCW=100-CW=0.0001-SR=0.22-Order=2'
file = parent_dir + '/results.json'
with open(file, 'r') as f:
    data = json.load(f)
R0 = data['MAJOR_RADIUS']
s = data['TOROIDAL_FLUX']
poincarePlot(parent_dir, R0, s)'''



# Get all result directories under the parent directory
result_dirs = [d for d in glob.glob(os.path.join(parent_dir, '*')) if os.path.isdir(d)]

df = pd.DataFrame()
# Loop through directories and read results.json
for result_dir in sorted(result_dirs):
    result_file = os.path.join(result_dir, 'results.json')
    if not os.path.isfile(result_file):
        continue  # Skip if results.json does not exist
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        # Extract the directory name relative to the parent directory
        relative_dirname = os.path.relpath(result_dir, parent_dir)
        # Add relative directory name to data
        data['dirname'] = relative_dirname
        # Normalize nested lists
        df = pd.concat([df, pd.json_normalize(data)], ignore_index=True)
        
        R0 = data['MAJOR_RADIUS']
        s = data['TOROIDAL_FLUX']
        poincarePlot(result_dir, R0, s)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error reading {result_file}: {e}")
        continue


if df.empty: raise Exception("Dataframe is empty: check that the parent directory name is correct")

