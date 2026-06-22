"""Honest run-dir hygiene for post-hoc reshaped-coil fields (no stale plants).

When a coil field is modified post-hoc (e.g. a centerline reshape) and re-traced,
the new run dir must NOT inherit the seed run's DERIVED verdict artifacts. The
classic bug: ``for f in seed.glob("*.json"): shutil.copy2(f, out)`` plants the
seed's ``results.json`` (its solver verdict) and ``confinement_verdict.json`` into
a dir presented as the reshaped result; ``copy2`` preserves the source mtime so the
plant reads as authoritative. The physics is never contaminated -- every metric
selector picks coils by type/current and the field-line tracer sums the full field
order-independently -- but the dir then misrepresents the reshaped result, and a
naive reader (or a judge that reads solver keys top-level) takes a seed REJECT as
the reshape's verdict.

This module is the fix + recurrence prevention:

* ``is_derived_output`` / ``populate_inputs`` -- whitelist-copy ONLY the tracer
  inputs; never a derived output. Gate any run-dir copy loop with
  ``is_derived_output`` so a seed's results/verdict/metrics are never planted.
* ``compute_reshape_metrics`` -- measure the reshaped field's geometry reusing
  production primitives: high-resolution scalar curvature (carrying the CWS
  secular terms ``G``/``H``, which ``get_dofs()`` does NOT round-trip),
  :func:`max_poloidal_extent_rad`, and :class:`CurveHardwareSdfKeepout`.
  ``winding_r0`` and the SDF manifest are REQUIRED: the swept-channel keepout
  frame (and the poloidal frame) must be the run's winding torus, not the 0.903
  fallback -- a free-3D ``CurveXYZFourier`` coil has no ``.surf`` to supply it,
  so an omitted ``winding_r0`` silently mis-frames the clearance. Survival is read
  from the dir's own default-mode ``PoincareMetrics`` (a trace output).
* ``write_reshape_provenance`` -- write ``RESHAPE_PROVENANCE.json`` (NOT
  ``results.json``): the measured metrics, the shared (unchanged) inputs, a
  coil-group manifest, and pointers to the authoritative artifacts. A post-hoc
  reshape has no full-solver ``results.json`` and we do NOT fabricate one -- a
  partial ``results.json`` would make a downstream judge fail-closed on the
  missing solver keys, whereas an absent one lets it return a clean "no results"
  and defer to the certificate. The provenance file is consumed by no gate.
* ``clean_existing_run_dir`` -- fix an already-built dir: DELETE the stale
  ``results.json`` + ``confinement_verdict.json``, then write the provenance.

The caller supplies the hardware-contract values (``winding_r0``,
``curvature_cap_inv_m`` from :func:`finite_build_frame_aware_curvature_limit_inv_m`,
the SDF manifest path); this module hardcodes none of them.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from banana_opt.coil_groups import COIL_GROUPS_RESULTS_KEY, build_contiguous_manifest
from banana_opt.hardware_contracts import POLOIDAL_EXTENT_HALF_WIDTH_RAD
from banana_opt.hardware_keepout import CurveHardwareSdfKeepout, load_hardware_sdf
from banana_opt.json_compat import load_boozer_finite_i
from banana_opt.poloidal_extent import max_poloidal_extent_rad
from simsopt import load as sload
from simsopt.geo import CurveXYZFourier
from simsopt.geo.curvecwsfourier import CurveCWSFourierCPP

# The U-turn curvature peak is sharp; default curve quadpoints under-resolve it
# (a 256-pt grid can read ~0.3/m low on an order-64 curve). Rebuild dense.
KAPPA_RESOLUTION = 4096
# |I| below this => banana coil, above => TF; an order-independent split that
# does not depend on coil list position.
CURRENT_SPLIT_A = 5.0e4
PROVENANCE_BASENAME = "RESHAPE_PROVENANCE.json"

# Tracer/viewer inputs a reshaped run dir legitimately needs copied from its seed.
INPUT_BASENAMES = (
    "surf_opt.json", "surf_opt_boozer_surface.json", "surf_opt_boozer_state.json",
    "surf_opt_outer.json", "biot_savart_init.json",
)
# results.json keys that a centerline reshape leaves unchanged (currents, winding,
# contract) and so may be carried through verbatim into the provenance record.
SHARED_INPUT_KEYS = (
    "NUM_TF_COILS", "NUM_BANANA_COILS", "NUM_PROXY_COILS", "NUM_VF_COILS", "TOTAL_COILS",
    "COIL_WINDING_SURFACE_MAJOR_RADIUS_M", "BANANA_CWS_EMBEDDED_WINDING_MINOR_RADIUS_M",
    "MAJOR_RADIUS", "banana_surf_radius", "TOROIDAL_FLUX", "NFP",
    "FINITE_CURRENT_MODE", "EFFECTIVE_CURRENT_MODE", "BOOZER_I", "PLASMA_CURRENT_A",
    "STRICT_VACUUM_CURRENT", "CURRENT_LINEAGE", "CONTRACT_HASH", "CURVATURE_THRESHOLD",
    "STAGE2_DESIGN_ONLY_NO_TOPOLOGY_GATE", "STAGE2_DESIGN_ONLY_REASON",
)
WINDING_R0_RESULTS_KEY = "COIL_WINDING_SURFACE_MAJOR_RADIUS_M"


def is_derived_output(name: str) -> bool:
    """True for files a run PRODUCES (must never be copied from a sibling run)."""
    if name in ("results.json", "confinement_verdict.json"):
        return True
    derived_prefixes = ("PoincareMetrics", "PoincarePlot", "confinement_verdict",
                        "curves_", "TransitNormalization", "results_best")
    if name.startswith(derived_prefixes):
        return True
    if name.endswith((".viewer.json", "_poincare.vts", "_poincare.vtu")):
        return True
    return False


def populate_inputs(source_dir: Path, out_dir: Path) -> list[str]:
    """Copy ONLY whitelisted tracer inputs from ``source_dir`` into ``out_dir``.

    Returns the basenames copied. Never copies a derived output. Use this in place
    of a copy-all glob when assembling a reshaped run dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in INPUT_BASENAMES:
        src = source_dir / name
        if src.exists() and not is_derived_output(name):
            shutil.copy2(src, out_dir / name)
            copied.append(name)
    return copied


def _max_kappa_highres(curve, n: int = KAPPA_RESOLUTION) -> float:
    """Max scalar curvature at high resolution (default quadpoints under-resolve
    the sharp U-turn peak). Rebuilds the curve of its own type at ``n`` quadpoints
    from the same dofs; pure geometry, no optimizer path."""
    qp = np.linspace(0.0, 1.0, n, endpoint=False)
    name = type(curve).__name__
    if name == "CurveCWSFourierCPP":
        # G/H are secular winding terms NOT carried by get_dofs(); they must be
        # passed explicitly or the rebuilt curve loses them (a G!=0 coil would be
        # mis-measured, fail-open past the cap). Matches the CWS-rebuild SSOT.
        hi = CurveCWSFourierCPP(qp, curve.order, curve.surf, G=curve.G, H=curve.H)
    elif name == "CurveXYZFourier":
        hi = CurveXYZFourier(qp, curve.order)
    else:
        raise TypeError(f"unsupported banana curve type for high-res kappa: {name}")
    hi.set_dofs(curve.get_dofs())
    return float(hi.kappa().max())


def _banana_tf(field) -> tuple[list, list]:
    banana = [c for c in field.coils if abs(c.current.get_value()) < CURRENT_SPLIT_A]
    tf = [c for c in field.coils if abs(c.current.get_value()) >= CURRENT_SPLIT_A]
    return banana, tf


@dataclass(frozen=True)
class ReshapeMetrics:
    n_banana: int
    n_tf: int
    max_kappa: float
    max_coil_length_m: float
    max_poloidal_extent_rad: float
    keepout_clearance_m: float
    banana_current_abs_max_a: float
    default_survived_lines: int
    default_nfieldlines: int
    volume: float | None
    aspect_ratio: float | None
    winding_r0_m: float


def compute_reshape_metrics(field_path: Path, reconverged_surface_path: Path | None,
                            default_metrics_path: Path, *, winding_r0: float,
                            sdf_manifest: str | Path) -> ReshapeMetrics:
    """Measure the reshaped field's geometry, reusing production primitives.

    ``winding_r0`` and ``sdf_manifest`` are required: keepout/poloidal frames are
    oriented about ``winding_r0`` (the run's winding torus), never the 0.903
    fallback. Volume/aspect come from the reconverged Boozer surface if provided,
    else ``None``. iota/residual are not recomputed here (they need a Boozer solve).
    """
    field = load_boozer_finite_i(str(field_path))
    banana, tf = _banana_tf(field)
    kappas, lengths, polext = [], [], []
    for c in banana:
        cur = c.curve
        gd = np.asarray(cur.gammadash())
        kappas.append(_max_kappa_highres(cur))
        lengths.append(float(np.sum(np.linalg.norm(gd, axis=1)) / gd.shape[0]))
        polext.append(max_poloidal_extent_rad(cur, winding_r0))
    sdf = load_hardware_sdf(str(sdf_manifest))
    keepout = float(CurveHardwareSdfKeepout(
        [c.curve for c in banana], sdf, winding_r0=winding_r0).shortest_distance())

    metrics = json.loads(Path(default_metrics_path).read_text())["metrics"]
    volume = aspect = None
    if reconverged_surface_path is not None and Path(reconverged_surface_path).exists():
        surf = sload(str(reconverged_surface_path))
        sg = surf.surface if hasattr(surf, "surface") else surf
        volume, aspect = float(sg.volume()), float(sg.aspect_ratio())
    return ReshapeMetrics(
        n_banana=len(banana), n_tf=len(tf),
        max_kappa=max(kappas), max_coil_length_m=max(lengths),
        max_poloidal_extent_rad=max(polext), keepout_clearance_m=keepout,
        banana_current_abs_max_a=max(abs(c.current.get_value()) for c in banana),
        default_survived_lines=int(metrics["survived_lines"]),
        default_nfieldlines=int(metrics["nfieldlines"]),
        volume=volume, aspect_ratio=aspect, winding_r0_m=float(winding_r0),
    )


def write_reshape_provenance(out_dir: Path, base_results_path: Path, m: ReshapeMetrics,
                             source_label: str, *, reconverged_surface_name: str | None,
                             curvature_cap_inv_m: float,
                             poloidal_cap_rad: float = POLOIDAL_EXTENT_HALF_WIDTH_RAD,
                             certificate_name: str | None = None,
                             notes: dict | None = None) -> Path:
    """Write ``RESHAPE_PROVENANCE.json`` describing the reshaped field.

    An audit/human record, deliberately NOT named ``results.json`` so no gate
    (e.g. a judge that reads solver keys top-level and fail-closes on a partial
    file) consumes it as a solver verdict. ``curvature_cap_inv_m`` is the run's
    finite-build cap (see :func:`finite_build_frame_aware_curvature_limit_inv_m`).
    """
    base = json.loads(Path(base_results_path).read_text())
    manifest = build_contiguous_manifest(
        num_tf_coils=m.n_tf, num_banana_coils=m.n_banana, num_proxy_coils=0, num_vf_coils=0)
    note_block = {
        "measurement": ("Geometry MEASURED on the reshaped field; this is NOT a solver pass. "
                        "The decisive confinement verdict is the default-mode "
                        "PoincareMetrics_opt_default.json in this dir."),
        "iota_residual": ("iota/Boozer-residual are not recomputed here; if a reconverged "
                          "surface is present it was solved by a separate exact-Newton step."),
    }
    if notes:
        note_block.update(notes)
    payload: dict[str, object] = {
        "RESHAPE_PROVENANCE": (
            f"Post-hoc reshape of seed '{source_label}'. NOT a solver pass; there is no full "
            "results.json (any stale seed copy was removed). Top-level blocks describe THIS field."),
        "is_full_solver_results": False,
        "source_seed": source_label,
        "decisive_artifacts": {
            "confinement_default_mode": "PoincareMetrics_opt_default.json",
            "reconverged_boozer_surface": reconverged_surface_name,
            "certificate": certificate_name,
        },
        "RESHAPE_MEASURED": {
            "MAX_CURVATURE": m.max_kappa, "CURVATURE_CAP": curvature_cap_inv_m,
            "FINITEBUILD_CURVATURE_OK": m.max_kappa <= curvature_cap_inv_m,
            "MAX_COIL_LENGTH_M": m.max_coil_length_m,
            "POLOIDAL_EXTENT_RAD": m.max_poloidal_extent_rad,
            "POLOIDAL_EXTENT_OK": m.max_poloidal_extent_rad <= poloidal_cap_rad,
            "HARDWARE_KEEPOUT_SDF_CLEARANCE_M": m.keepout_clearance_m,
            "HARDWARE_KEEPOUT_OK": m.keepout_clearance_m >= 0.0,
            "BANANA_CURRENT_ABS_MAX_A": m.banana_current_abs_max_a,
            "WINDING_R0_M": m.winding_r0_m,
            "DEFAULT_TOPOLOGY_SURVIVED_LINES": m.default_survived_lines,
            "DEFAULT_TOPOLOGY_NFIELDLINES": m.default_nfieldlines,
            "DEFAULT_TOPOLOGY_SURVIVAL_FRACTION": m.default_survived_lines / m.default_nfieldlines,
            "VOLUME": m.volume, "ASPECT_RATIO": m.aspect_ratio,
        },
        "SHARED_INPUTS": {k: base[k] for k in SHARED_INPUT_KEYS if k in base},
        COIL_GROUPS_RESULTS_KEY: manifest.to_json_payload(),
        "NOTES": note_block,
    }
    path = out_dir / PROVENANCE_BASENAME
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def clean_existing_run_dir(run_dir: Path, base_results_path: Path,
                           reconverged_surface_path: Path | None, source_label: str, *,
                           sdf_manifest: str | Path, curvature_cap_inv_m: float,
                           field_basename: str = "biot_savart_opt.json",
                           default_metrics_basename: str = "PoincareMetrics_opt_default.json",
                           certificate_name: str | None = None,
                           notes: dict | None = None) -> dict:
    """Fix an already-built reshape dir: DELETE the stale baseline ``results.json``
    + ``confinement_verdict.json`` and write ``RESHAPE_PROVENANCE.json``.

    ``winding_r0`` is read from ``base_results`` (``COIL_WINDING_SURFACE_MAJOR_RADIUS_M``,
    the realized winding torus) and passed through to the keepout/poloidal frames.
    A non-design-only field traces without ``results.json``; the decisive verdict
    lives in the default-mode ``PoincareMetrics`` and the certificate.
    """
    run_dir = Path(run_dir)
    deleted: list[str] = []
    for name in ("confinement_verdict.json", "results.json"):
        p = run_dir / name
        if p.exists():
            p.unlink()
            deleted.append(name)

    base = json.loads(Path(base_results_path).read_text())
    winding_r0 = float(base[WINDING_R0_RESULTS_KEY])
    m = compute_reshape_metrics(run_dir / field_basename, reconverged_surface_path,
                                run_dir / default_metrics_basename,
                                winding_r0=winding_r0, sdf_manifest=sdf_manifest)
    surf_name = Path(reconverged_surface_path).name if reconverged_surface_path is not None else None
    write_reshape_provenance(run_dir, base_results_path, m, source_label,
                             reconverged_surface_name=surf_name,
                             curvature_cap_inv_m=curvature_cap_inv_m,
                             certificate_name=certificate_name, notes=notes)
    return {
        "run_dir": str(run_dir), "deleted": deleted, "wrote_provenance": PROVENANCE_BASENAME,
        "metrics": {"max_kappa": round(m.max_kappa, 3),
                    "keepout_mm": round(m.keepout_clearance_m * 1e3, 2),
                    "curvature_ok": m.max_kappa <= curvature_cap_inv_m,
                    "default_survival": f"{m.default_survived_lines}/{m.default_nfieldlines}"},
    }
