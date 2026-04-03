import glob
import json
import os
import sys

# SIMSOPT imports
from simsopt._core.optimizable import load
from simsopt.field import InterpolatedField
from simsopt.field import compute_fieldlines
import numpy as np
from simsopt.field import (
    LevelsetStoppingCriterion,
    MaxZStoppingCriterion,
    MinZStoppingCriterion,
    MaxRStoppingCriterion,
    MinRStoppingCriterion,
)
from simsopt.geo import SurfaceClassifier, curves_to_vtk

# Shared topology scorer — single source of truth for helpers and metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from topology_scorer import (
    midplane_seed_radii as _midplane_seed_radii,
    padded_bounds as _padded_bounds,
    trace_metrics as _trace_metrics,
    phi_hit_counts as _phi_hit_counts,
    build_stopping_criteria,
    STOP_LABELS_VALIDATION,
    STOP_LABELS_DIAGNOSTIC,
)


def _closed_rz(cross_section):
    """Convert a Cartesian cross-section to closed (R, Z) arrays."""
    r = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    z = cross_section[:, 2]
    return np.append(r, r[0]), np.append(z, z[0])


def plot_poincare_data(fieldlines_phi_hits, phis, filename, mark_lost=False, aspect='equal', dpi=600, xlims=None,
                       ylims=None, surf=None, surf1=None, s=2, marker='o'):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    import matplotlib.pyplot as plt
    from math import ceil, sqrt
    nrowcol = ceil(sqrt(len(phis)))
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
        if col > 0:
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
            phi_new = phis[i] / (2 * np.pi)
            cross_section = surf.cross_section(phi=phi_new, thetas=256)
            r_interp, z_interp = _closed_rz(cross_section)
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c='k')
        if surf1 is not None:
            phi_new = phis[i] / (2 * np.pi)
            cross_section = surf1.cross_section(phi=phi_new, thetas=256)
            r_interp, z_interp = _closed_rz(cross_section)
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c='r')

    # Hide unused subplots for non-square phi counts
    for idx in range(len(phis), nrowcol * nrowcol):
        axs[idx // nrowcol, idx % nrowcol].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    nfieldlines = 50 # Number of field lines for integration 
    tmax_fl = 7000 # Maximum toroidal angle for integration
    tol = 1e-7 # Tolerance for field line integration
    interpolate = True # If True, then the BiotSavart magnetic field is interpolated 
                       # on a grid for the magnetic field evaluation
    nr = 40 # Number of radial points for interpolation
    nphi = 40 # Number of toroidal angle points for interpolation
    nz = 20 # Number of vertical points for interpolation
    degree = 3 # Degree for interpolation

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    if os.environ.get("POINCARE_OUT_DIR"):
        OUT_DIR = os.environ["POINCARE_OUT_DIR"]
    else:
        outputs_root = os.path.join(EXAMPLE_ROOT, "SINGLE_STAGE", "outputs")
        candidates = sorted(glob.glob(os.path.join(outputs_root, "mpol=*")), key=os.path.getmtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"No single-stage output found in {outputs_root}. Set POINCARE_OUT_DIR.")
        OUT_DIR = candidates[0]
        print(f"Auto-selected: {os.path.basename(OUT_DIR)}")
    # Load optimized field + surface if both available, otherwise fall back to init for both
    opt_bs_path = OUT_DIR + '/biot_savart_opt.json'
    init_bs_path = OUT_DIR + '/biot_savart_init.json'
    opt_surf_path = OUT_DIR + '/surf_opt.json'
    init_surf_path = OUT_DIR + '/surf_init.json'

    # Load field and surface from the same stage (both opt or both init)
    has_opt = os.path.exists(opt_bs_path) and os.path.exists(opt_surf_path)
    if has_opt:
        bs = load(opt_bs_path)
        surf = load(opt_surf_path)
        field_label = "opt"
        print(f"Loaded OPTIMIZED field + surface")
    else:
        bs = load(init_bs_path)
        surf = load(init_surf_path)
        field_label = "init"
        if os.path.exists(opt_bs_path) != os.path.exists(opt_surf_path):
            print(f"WARNING: mismatched opt files (bs={os.path.exists(opt_bs_path)}, surf={os.path.exists(opt_surf_path)}). Using init for both.")
        else:
            print(f"Loaded INITIAL field + surface (no opt found)")

    # Export the exact geometry used by this Poincare run for ParaView inspection.
    curves_to_vtk([coil.curve for coil in bs.coils], os.path.join(OUT_DIR, f"curves_{field_label}_poincare"), close=True)
    surf.to_vtk(os.path.join(OUT_DIR, f"surf_{field_label}_poincare"))

    # Build stopping criteria from the shared module (single source of truth)
    nfp = surf.nfp
    stop_crit_validation, stop_labels_validation = build_stopping_criteria(surf, include_surface_exit=True)
    stop_crit_box, stop_labels_diagnostic = build_stopping_criteria(surf, include_surface_exit=False)


    def trace_fieldlines(bfield):
        # Seed from the midplane slightly inside the surface so any outward
        # excursion is meaningful rather than an artifact of boundary seeding.
        R0 = _midplane_seed_radii(surf, nfieldlines)
        Z0 = np.zeros((nfieldlines,))
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]

        def trace_and_plot(stopping_criteria, stop_labels, suffix, label, mode):
            fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
                bfield,
                R0,
                Z0,
                tmax=tmax_fl,
                tol=tol,
                phis=phis,
                stopping_criteria=stopping_criteria,
            )
            metrics = _trace_metrics(
                fieldlines_tys,
                fieldlines_phi_hits,
                phis,
                stop_labels,
                mode,
            )
            filename = OUT_DIR + f'/PoincarePlot_{field_label}{suffix}.png'
            plot_poincare_data(
                fieldlines_phi_hits,
                phis,
                filename,
                dpi=600,
                surf=surf,
                mark_lost=False,
            )
            print(
                f"Saved: {os.path.basename(filename)} "
                f"({label}; phi hit counts={metrics['per_phi_hit_counts']}; "
                f"status={metrics['validation_status']}; "
                f"survival={metrics['survived_lines']}/{metrics['nfieldlines']})"
            )
            metrics["plot_filename"] = os.path.basename(filename)
            return metrics

        validation_metrics = trace_and_plot(
            stop_crit_validation,
            stop_labels_validation,
            "",
            "validation: stop on Boozer-surface exit",
            "validation",
        )
        diagnostic_metrics = trace_and_plot(
            stop_crit_box,
            stop_labels_diagnostic,
            "_diagnostic",
            "diagnostic: box-bounded only",
            "diagnostic",
        )
        metrics_path = os.path.join(OUT_DIR, f"PoincareMetrics_{field_label}.json")
        artifact = {
            "field_label": field_label,
            "nfieldlines": nfieldlines,
            "tmax": tmax_fl,
            "tol": tol,
            "phis": [float(phi) for phi in phis],
            "validation": validation_metrics,
            "diagnostic": diagnostic_metrics,
            "validation_status": validation_metrics["validation_status"],
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        print(f"Saved: {os.path.basename(metrics_path)}")
        return artifact

    # Determine interpolation bounds independently from the validation surface
    # so the field interpolant is not clipped at the exact Boozer outline.
    gamma = surf.gamma()
    Rmin = np.min(np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2))
    Rmax = np.max(np.sqrt(gamma[:, :, 0]**2 + gamma[:, :, 1]**2))
    Zmax = np.max(np.abs(gamma[:, :, 2]))
    interp_rmin, interp_rmax, interp_zmax = _padded_bounds(Rmin, Rmax, Zmax)
    rrange = (interp_rmin, interp_rmax, nr)
    phirange = (0, 2*np.pi/nfp, nphi)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, interp_zmax, nz)

    if interpolate:
        bsh = InterpolatedField(
            bs,
            degree,
            rrange,
            phirange,
            zrange,
            True,
            nfp=nfp,
            stellsym=True,
        )

        bsh.set_points(surf.gamma().reshape((-1, 3)))
        bs.set_points(surf.gamma().reshape((-1, 3)))
        Bh = bsh.B()
        B = bs.B()
        print("Maximum field interpolation error: ", np.max(np.abs(B-Bh)))
    else:
        bsh = bs

    trace_fieldlines(bsh)
