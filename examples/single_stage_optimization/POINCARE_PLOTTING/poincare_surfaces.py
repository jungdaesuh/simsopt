import glob
import json
import os
import sys

# SIMSOPT imports
from simsopt.field import compute_fieldlines
import numpy as np
from simsopt.geo import curves_to_vtk

# Shared topology scorer — single source of truth for helpers and metrics
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from banana_opt.json_compat import load_boozer_finite_i as load
from topology_scorer import (
    trace_metrics as _trace_metrics,
    extended_surface_seed_radii,
    build_extended_surface_seed_contract,
    midplane_seed_radii,
    build_midplane_seed_contract,
    build_stopping_criteria,
    prepare_topology_field,
    surface_trace_domain,
    topology_iteration_limit,
)


SUPPORTED_POINCARE_FIELD_POLICIES = frozenset({"auto", "always", "never"})


def resolve_poincare_field_policy(env=None):
    """Read and validate the field policy used by this plotting script."""
    environ = os.environ if env is None else env
    field_policy = str(environ.get("POINCARE_FIELD_POLICY", "auto"))
    if field_policy not in SUPPORTED_POINCARE_FIELD_POLICIES:
        raise ValueError(
            f"Unsupported POINCARE_FIELD_POLICY={field_policy!r}; expected "
            "'auto', 'always', or 'never'"
        )
    return field_policy


def build_poincare_render_modes(
    surf,
    nfieldlines,
    *,
    seed_inset_fraction,
    default_extend_distance,
    stopping_criteria,
    stop_labels,
):
    """Build the three Poincare render contracts without tracing field lines."""
    midplane_radii = midplane_seed_radii(
        surf,
        nfieldlines,
        inset_fraction=seed_inset_fraction,
    )
    default_radii = extended_surface_seed_radii(
        surf,
        nfieldlines,
        extend_distance=default_extend_distance,
    )
    validation_seed_contract = build_midplane_seed_contract(
        nfieldlines,
        seed_inset_fraction,
        midplane_radii,
    )
    default_seed_contract = build_extended_surface_seed_contract(
        nfieldlines,
        default_extend_distance,
        default_radii,
    )
    midplane_trace_domain = surface_trace_domain(surf, seed_radii=midplane_radii)
    default_trace_domain = surface_trace_domain(surf, seed_radii=default_radii)
    field_trace_domain = surface_trace_domain(
        surf,
        seed_radii=np.concatenate([midplane_radii, default_radii]),
    )
    return [
        {
            "radii": midplane_radii,
            "z0": np.zeros(int(nfieldlines)),
            "stopping_criteria": stopping_criteria,
            "stop_labels": stop_labels,
            "plot_suffix": "",
            "metrics_suffix": "_validation",
            "label": "validation: stop on Boozer-surface exit",
            "mode": "validation",
            "seed_contract": validation_seed_contract,
            "trace_domain": midplane_trace_domain,
        },
        {
            "radii": midplane_radii,
            "z0": np.zeros(int(nfieldlines)),
            "stopping_criteria": stopping_criteria,
            "stop_labels": stop_labels,
            "plot_suffix": "_diagnostic",
            "metrics_suffix": "_diagnostic",
            "label": "diagnostic: midplane seeds with Boozer-surface exit guard",
            "mode": "diagnostic",
            "seed_contract": validation_seed_contract,
            "trace_domain": midplane_trace_domain,
        },
        {
            "radii": default_radii,
            "z0": np.zeros(int(nfieldlines)),
            "stopping_criteria": stopping_criteria,
            "stop_labels": stop_labels,
            "plot_suffix": "_default",
            "metrics_suffix": "_default",
            "label": "default: extended-surface seeds with Boozer-surface exit guard",
            "mode": "default",
            "seed_contract": default_seed_contract,
            "trace_domain": default_trace_domain,
        },
    ], field_trace_domain


def plot_poincare_data(
    fieldlines_phi_hits,
    phis,
    filename,
    mark_lost=False,
    aspect="equal",
    dpi=600,
    xlims=None,
    ylims=None,
    surf=None,
    surf1=None,
    s=2,
    marker="o",
):
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
        row = i // nrowcol
        col = i % nrowcol
        if i != len(phis) - 1:
            axs[row, col].set_title(
                f"$\\phi = {phis[i] / np.pi:.2f}\\pi$ ", loc="left", y=0.0
            )
        else:
            axs[row, col].set_title(
                f"$\\phi = {phis[i] / np.pi:.2f}\\pi$ ", loc="right", y=0.0
            )
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
                color = "r" if lost else "g"
            data_this_phi = fieldlines_phi_hits[j][
                np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :
            ]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
            axs[row, col].scatter(
                r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color
            )

        plt.rc("axes", axisbelow=True)
        axs[row, col].grid(True, linewidth=0.5)

        # if passed a surface, plot the plasma surface outline
        if surf is not None:
            phi_new = phis[i] * 1 / (2 * np.pi)
            cross_section = surf.cross_section(phi=phi_new)
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c="k")
        if surf1 is not None:
            phi_new = phis[i] * 1 / (2 * np.pi)
            cross_section = surf1.cross_section(phi=phi_new)
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c="r")

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()


if __name__ == "__main__":
    nfieldlines = 50  # Number of field lines for integration
    tmax_fl = 7000  # Maximum toroidal angle for integration
    tol = 1e-7  # Tolerance for field line integration
    nr = 40  # Number of radial points for interpolation
    nphi = 40  # Number of toroidal angle points for interpolation
    nz = 20  # Number of vertical points for interpolation
    degree = 3  # Degree for interpolation
    field_model_policy = resolve_poincare_field_policy()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
    if os.environ.get("POINCARE_OUT_DIR"):
        OUT_DIR = os.environ["POINCARE_OUT_DIR"]
    else:
        outputs_root = os.path.join(EXAMPLE_ROOT, "SINGLE_STAGE", "outputs")
        candidates = sorted(
            glob.glob(os.path.join(outputs_root, "mpol=*")),
            key=os.path.getmtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No single-stage output found in {outputs_root}. Set POINCARE_OUT_DIR."
            )
        OUT_DIR = candidates[0]
        print(f"Auto-selected: {os.path.basename(OUT_DIR)}")
    # Load optimized field + surface if both available, otherwise fall back to init for both
    opt_bs_path = OUT_DIR + "/biot_savart_opt.json"
    init_bs_path = OUT_DIR + "/biot_savart_init.json"
    opt_surf_path = OUT_DIR + "/surf_opt.json"
    init_surf_path = OUT_DIR + "/surf_init.json"

    # Load field and surface from the same stage (both opt or both init)
    has_opt = os.path.exists(opt_bs_path) and os.path.exists(opt_surf_path)
    if has_opt:
        bs = load(opt_bs_path)
        surf = load(opt_surf_path)
        field_label = "opt"
        print("Loaded OPTIMIZED field + surface")
    else:
        bs = load(init_bs_path)
        surf = load(init_surf_path)
        field_label = "init"
        if os.path.exists(opt_bs_path) != os.path.exists(opt_surf_path):
            print(
                f"WARNING: mismatched opt files (bs={os.path.exists(opt_bs_path)}, surf={os.path.exists(opt_surf_path)}). Using init for both."
            )
        else:
            print("Loaded INITIAL field + surface (no opt found)")

    # Export the exact geometry used by this Poincare run for ParaView inspection.
    curves_to_vtk(
        [coil.curve for coil in bs.coils],
        os.path.join(OUT_DIR, f"curves_{field_label}_poincare"),
        close=True,
    )
    surf.to_vtk(os.path.join(OUT_DIR, f"surf_{field_label}_poincare"))

    # Build stopping criteria from the shared module (single source of truth)
    nfp = surf.nfp
    iteration_limit = topology_iteration_limit(tmax_fl)
    stopping_criteria, stop_labels = build_stopping_criteria(
        surf,
        include_surface_exit=True,
        max_iterations=iteration_limit,
    )
    seed_inset_fraction = 0.05
    default_extend_distance = 0.05
    render_modes, field_trace_domain = build_poincare_render_modes(
        surf,
        nfieldlines,
        seed_inset_fraction=seed_inset_fraction,
        default_extend_distance=default_extend_distance,
        stopping_criteria=stopping_criteria,
        stop_labels=stop_labels,
    )

    def trace_fieldlines(bfield):
        phis = [(i / 4) * (2 * np.pi / nfp) for i in range(4)]

        traced_hits_by_mode = {}

        def trace_and_record(
            render_mode,
        ):
            print(f"Tracing {render_mode['mode']} Poincare mode...")
            fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
                bfield,
                render_mode["radii"],
                render_mode["z0"],
                tmax=tmax_fl,
                tol=tol,
                phis=phis,
                stopping_criteria=render_mode["stopping_criteria"],
            )
            metrics = _trace_metrics(
                fieldlines_tys,
                fieldlines_phi_hits,
                phis,
                render_mode["stop_labels"],
                render_mode["mode"],
            )
            plot_filename = (
                OUT_DIR + f"/PoincarePlot_{field_label}{render_mode['plot_suffix']}.png"
            )
            metrics["plot_filename"] = os.path.basename(plot_filename)
            metrics["seed_contract"] = render_mode["seed_contract"]
            traced_hits_by_mode[render_mode["mode"]] = (
                fieldlines_phi_hits,
                plot_filename,
            )
            metrics_sidecar_path = os.path.join(
                OUT_DIR,
                f"PoincareMetrics_{field_label}{render_mode['metrics_suffix']}.json",
            )
            mode_artifact = {
                "field_label": field_label,
                "render_mode": render_mode["mode"],
                "nfieldlines": nfieldlines,
                "tmax": tmax_fl,
                "tol": tol,
                "phis": [float(phi) for phi in phis],
                "seed_contract": render_mode["seed_contract"],
                "trace_domain": render_mode["trace_domain"].as_metadata(),
                "stop_labels": list(render_mode["stop_labels"]),
                "field_model": field_model,
                "plot_filename": metrics["plot_filename"],
                "metrics": metrics,
                "validation_status": metrics["validation_status"],
            }
            with open(metrics_sidecar_path, "w", encoding="utf-8") as f:
                json.dump(mode_artifact, f, indent=2)
            metrics["metrics_filename"] = os.path.basename(metrics_sidecar_path)
            print(
                f"Saved: {os.path.basename(metrics_sidecar_path)} "
                f"({render_mode['label']}; phi hit counts={metrics['per_phi_hit_counts']}; "
                f"status={metrics['validation_status']}; "
                f"survival={metrics['survived_lines']}/{metrics['nfieldlines']})"
            )
            return metrics

        metrics_by_mode = {
            render_mode["mode"]: trace_and_record(render_mode)
            for render_mode in render_modes
        }
        for render_mode in render_modes:
            fieldlines_phi_hits, plot_filename = traced_hits_by_mode[
                render_mode["mode"]
            ]
            plot_poincare_data(
                fieldlines_phi_hits,
                phis,
                plot_filename,
                dpi=600,
                surf=surf,
                mark_lost=False,
            )
            print(f"Saved: {os.path.basename(plot_filename)}")
        metrics_path = os.path.join(OUT_DIR, f"PoincareMetrics_{field_label}.json")
        artifact = {
            "field_label": field_label,
            "nfieldlines": nfieldlines,
            "tmax": tmax_fl,
            "tol": tol,
            "phis": [float(phi) for phi in phis],
            "seed_contract": render_modes[0]["seed_contract"],
            "default_seed_contract": render_modes[2]["seed_contract"],
            "trace_domain": field_trace_domain.as_metadata(),
            "field_model": field_model,
            "validation": metrics_by_mode["validation"],
            "diagnostic": metrics_by_mode["diagnostic"],
            "default": metrics_by_mode["default"],
            "validation_status": metrics_by_mode["validation"]["validation_status"],
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=2)
        print(f"Saved: {os.path.basename(metrics_path)}")
        return artifact

    bsh, field_model = prepare_topology_field(
        surf,
        bs,
        tmax_fl,
        field_policy=field_model_policy,
        interpolation_grid={
            "degree": degree,
            "nr": nr,
            "nphi": nphi,
            "nz": nz,
        },
        trace_domain=field_trace_domain,
    )
    if field_model["selected_mode"] == "interpolated":
        print("Maximum field interpolation error: ", field_model["max_abs_error"])

    trace_fieldlines(bsh)
