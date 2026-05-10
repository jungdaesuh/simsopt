"""Phase 4 P4.5/P4.5b boundary-pinned residual byte arbiter.

This standalone benchmark drives the byte-tier acceptance gate for the
residual derivative bit-identity zeroing slice. It feeds the canonical
CPU-oracle residual-input bundle (P4.1) to BOTH the C++ residual
(``simsopt.geo.boozersurface._call_boozer_residual_ds``) and the JAX
CPU-ordered residual
(``simsopt.geo.boozer_residual_jax.boozer_residual_scalar_and_grad_cpu_ordered``)
and reports byte-level disagreement on the value and gradient outputs.

Two arbiters are produced:

* **residual_only (P4.5)** — the inner residual kernel. Loaded entirely from
  the canonical bundle, no fixture rebuild needed.
* **full_penalty (P4.5b)** — extends P4.5 by rebuilding the same synthetic
  NCSX fixture used during census capture (see
  ``benchmarks/parity/boozer_derivative_input_repro.py::_build_synthetic_fixture``)
  to exercise the label term and the rz-axis penalty that BFGS sees.

See ``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` §10
P4.5 / P4.5b for the contract. This script is diagnostic only and lives
under ``benchmarks/parity/``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


PLAN_DOC = "docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md"
DEFAULT_BUNDLE = Path(".artifacts/parity/20260508-residual-pinned-inputs")
DEFAULT_RESULTS_SUBDIR = "byte_arbiter_results"
DEFAULT_RESULTS_FILENAME = "byte_arbiter_results.json"


_CANONICAL_ARRAY_NAMES = (
    "gamma",
    "xphi",
    "xtheta",
    "dx_ds",
    "dxphi_ds",
    "dxtheta_ds",
    "B",
    "dB_dX",
)
_CANONICAL_SCALAR_NAMES = ("G_value", "iota", "weight_inv_modB")


def _bytewise_unequal_double_count(a: np.ndarray, b: np.ndarray) -> int:
    """Count float64 lanes whose 8-byte representation differs.

    Mirrors
    :func:`benchmarks.parity.boozer_derivative_input_census._bytewise_unequal_double_count`
    so the byte-level metric matches the existing census reporter.
    """
    if a.shape != b.shape:
        return -1
    a_view = np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)
    b_view = np.ascontiguousarray(b, dtype=np.float64).view(np.uint64)
    return int(np.count_nonzero(a_view != b_view))


def _byte_identical(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    a_bytes = np.ascontiguousarray(a, dtype=np.float64).view(np.uint64)
    b_bytes = np.ascontiguousarray(b, dtype=np.float64).view(np.uint64)
    return bool(np.array_equal(a_bytes, b_bytes))


def _argmax_index(diff_abs: np.ndarray) -> tuple[int, ...] | None:
    """Return the multi-dim index of the maximum-magnitude difference.

    Returns ``None`` for empty arrays or when no nonzero diff exists.
    """
    if diff_abs.size == 0:
        return None
    flat = int(np.argmax(diff_abs))
    if diff_abs.flat[flat] == 0.0:
        return None
    return tuple(int(i) for i in np.unravel_index(flat, diff_abs.shape))


def _capture_host_environment() -> dict[str, Any]:
    """Snapshot the host fingerprint relevant for byte-identity claims."""
    try:
        import jax  # noqa: PLC0415

        jax_version = jax.__version__
    except ImportError:
        jax_version = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "jax_version": jax_version,
        "executable": sys.executable,
    }


def _load_canonical_bundle(bundle_dir: Path) -> dict[str, np.ndarray]:
    """Load every ``canonical_<name>.npy`` listed in ``manifest.json``.

    Returns a dict keyed by canonical name (matching
    ``CENSUS_BOUNDARY_ARRAY_ORDER`` / ``CENSUS_BOUNDARY_SCALAR_ORDER``).
    """
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.json missing under {bundle_dir!s}; "
            "regenerate via boozer_derivative_input_repro.py "
            "--census --dump-arrays-as-npy"
        )
    manifest = json.loads(manifest_path.read_text())
    canonical_entries = [
        e for e in manifest.get("files", []) if e["role"] == "canonical"
    ]
    if not canonical_entries:
        raise RuntimeError(
            f"manifest at {manifest_path!s} has no canonical role entries"
        )
    arrays: dict[str, np.ndarray] = {}
    for entry in canonical_entries:
        path = bundle_dir / entry["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical file: {path!s}")
        arr = np.load(path, allow_pickle=False)
        if arr.dtype != np.float64:
            raise RuntimeError(
                f"canonical {entry['name']} dtype={arr.dtype}; expected float64"
            )
        arrays[entry["name"]] = arr
    return arrays


def _run_residual_only(canonical: dict[str, np.ndarray]) -> dict[str, Any]:
    """Drive both residual implementations on canonical inputs and diff outputs.

    The canonical bundle was generated with ``optimize_G=True`` (see
    ``benchmarks/parity/boozer_derivative_input_repro.py::_build_synthetic_fixture``);
    the JAX gradient size therefore matches the CPU gradient size at
    ``nsurfdofs + 2``. Both outputs are normalized by ``num_res = 3 * nphi *
    ntheta`` before comparison: the CPU C++ binding returns the un-normalized
    sum, while the JAX wrapper applies the ``/ num_res`` step inside the
    function body.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)

    from simsopt.geo.boozer_residual_jax import (  # noqa: PLC0415
        boozer_residual_scalar_and_grad_cpu_ordered,
    )
    from simsopt.geo.boozersurface import _call_boozer_residual_ds  # noqa: PLC0415

    B = canonical["B"]
    nphi, ntheta = B.shape[:2]
    num_res = int(3 * nphi * ntheta)

    weight_inv_modB = bool(np.asarray(canonical["weight_inv_modB"]))
    G_value = float(np.asarray(canonical["G_value"]))
    iota = float(np.asarray(canonical["iota"]))

    # CPU residual: returns (val, dval) UN-NORMALIZED.
    val_cpu_raw, dval_cpu_raw = _call_boozer_residual_ds(
        G_value,
        iota,
        canonical["B"],
        canonical["dB_dX"],
        canonical["xphi"],
        canonical["xtheta"],
        canonical["dx_ds"],
        canonical["dxphi_ds"],
        canonical["dxtheta_ds"],
        weight_inv_modB,
    )
    val_cpu = float(val_cpu_raw) / num_res
    grad_cpu = np.asarray(dval_cpu_raw, dtype=np.float64) / num_res

    # JAX residual: returns (value, gradient) ALREADY NORMALIZED.
    val_jax_jax, grad_jax_jax = boozer_residual_scalar_and_grad_cpu_ordered(
        jnp.asarray(G_value, dtype=jnp.float64),
        jnp.asarray(iota, dtype=jnp.float64),
        jnp.asarray(canonical["B"], dtype=jnp.float64),
        jnp.asarray(canonical["dB_dX"], dtype=jnp.float64),
        jnp.asarray(canonical["xphi"], dtype=jnp.float64),
        jnp.asarray(canonical["xtheta"], dtype=jnp.float64),
        jnp.asarray(canonical["dx_ds"], dtype=jnp.float64),
        jnp.asarray(canonical["dxphi_ds"], dtype=jnp.float64),
        jnp.asarray(canonical["dxtheta_ds"], dtype=jnp.float64),
        optimize_G=True,
        weight_inv_modB=weight_inv_modB,
    )
    val_jax = float(np.asarray(jax.device_get(val_jax_jax), dtype=np.float64))
    grad_jax = np.asarray(jax.device_get(grad_jax_jax), dtype=np.float64)

    val_cpu_arr = np.asarray(val_cpu, dtype=np.float64)
    val_jax_arr = np.asarray(val_jax, dtype=np.float64)
    diff_grad = grad_jax - grad_cpu

    return {
        "num_res": num_res,
        "optimize_G": True,
        "weight_inv_modB": weight_inv_modB,
        "value_cpu_normalized": val_cpu,
        "value_jax_normalized": val_jax,
        "max_abs_diff_value": float(abs(val_jax - val_cpu)),
        "byte_identical_value": _byte_identical(val_cpu_arr, val_jax_arr),
        "grad_shape": list(int(s) for s in grad_jax.shape),
        "max_abs_diff_grad": float(np.max(np.abs(diff_grad)))
        if diff_grad.size
        else 0.0,
        "byte_identical_grad": _byte_identical(grad_cpu, grad_jax),
        "n_bytewise_unequal_grad_doubles": _bytewise_unequal_double_count(
            grad_cpu, grad_jax
        ),
        "argmax_grad_diff_index": _argmax_index(np.abs(diff_grad)),
    }


def _run_full_penalty() -> dict[str, Any]:
    """Drive both full-penalty implementations on the synthetic NCSX fixture.

    The full penalty cannot be reconstructed from the canonical bundle alone:
    it depends on ``BoozerSurface`` / ``BoozerSurfaceJAX`` objects, label
    geometry, coil_set_spec, etc. We rebuild the same fixture used by
    ``boozer_derivative_input_repro.py`` so the comparison feeds identical
    inputs to both CPU and JAX paths.
    """
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)

    from benchmarks.parity.boozer_derivative_input_repro import (  # noqa: PLC0415
        _build_synthetic_fixture,
    )
    from simsopt.geo.boozersurface_jax import (  # noqa: PLC0415
        _boozer_penalty_value_and_grad_cpu_ordered,
        _hostify_tree,
        _resolved_coil_set_spec,
    )

    fixture = _build_synthetic_fixture()
    booz_cpu = fixture["boozer_cpu"]
    booz_jax = fixture["boozer_jax"]
    sdofs = np.asarray(fixture["sdofs"], dtype=np.float64)
    iota = float(fixture["iota"])
    G = float(fixture["G"])
    optimize_G = bool(fixture["optimize_G"])
    weight_inv_modB = bool(fixture["weight_inv_modB"])
    constraint_weight = 100.0  # matches BoozerSurfaceJAX construction below

    pieces = [sdofs, np.asarray([iota], dtype=np.float64)]
    if optimize_G:
        pieces.append(np.asarray([G], dtype=np.float64))
    dofs = np.concatenate(pieces)

    val_cpu_raw, grad_cpu_raw = booz_cpu.boozer_penalty_constraints_vectorized(
        dofs,
        derivatives=1,
        constraint_weight=constraint_weight,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )
    val_cpu = float(val_cpu_raw)
    grad_cpu = np.asarray(grad_cpu_raw, dtype=np.float64)

    coil_set_spec = _hostify_tree(_resolved_coil_set_spec(booz_jax.coil_set_spec))
    val_jax_jax, grad_jax_jax = _boozer_penalty_value_and_grad_cpu_ordered(
        jnp.asarray(dofs, dtype=jnp.float64),
        coil_arrays=None,
        coil_set_spec=coil_set_spec,
        quadpoints_phi=_hostify_tree(booz_jax.quadpoints_phi),
        quadpoints_theta=_hostify_tree(booz_jax.quadpoints_theta),
        mpol=booz_jax.mpol,
        ntor=booz_jax.ntor,
        nfp=booz_jax.nfp,
        stellsym=booz_jax.stellsym,
        scatter_indices=_hostify_tree(booz_jax.scatter_indices),
        surface_kind=booz_jax._surface_geometry_kind,
        label_quadpoints_phi=_hostify_tree(booz_jax.label_quadpoints_phi),
        label_quadpoints_theta=_hostify_tree(booz_jax.label_quadpoints_theta),
        label_mpol=booz_jax.label_mpol,
        label_ntor=booz_jax.label_ntor,
        label_nfp=booz_jax.label_nfp,
        label_stellsym=booz_jax.label_stellsym,
        label_scatter_indices=_hostify_tree(booz_jax.label_scatter_indices),
        label_surface_kind=booz_jax._label_surface_geometry_kind,
        targetlabel=booz_jax.targetlabel,
        constraint_weight=constraint_weight,
        label_type=booz_jax.label_type,
        phi_idx=booz_jax.phi_idx,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
        parity_policy="cpu_ordered",
    )
    val_jax = float(np.asarray(jax.device_get(val_jax_jax), dtype=np.float64))
    grad_jax = np.asarray(jax.device_get(grad_jax_jax), dtype=np.float64)

    val_cpu_arr = np.asarray(val_cpu, dtype=np.float64)
    val_jax_arr = np.asarray(val_jax, dtype=np.float64)
    diff_grad = grad_jax - grad_cpu

    return {
        "fixture": "synthetic-ncsx-volume",
        "constraint_weight": float(constraint_weight),
        "optimize_G": optimize_G,
        "weight_inv_modB": weight_inv_modB,
        "value_cpu": val_cpu,
        "value_jax": val_jax,
        "max_abs_diff_value": float(abs(val_jax - val_cpu)),
        "byte_identical_value": _byte_identical(val_cpu_arr, val_jax_arr),
        "grad_shape": list(int(s) for s in grad_jax.shape),
        "max_abs_diff_grad": float(np.max(np.abs(diff_grad)))
        if diff_grad.size
        else 0.0,
        "byte_identical_grad": _byte_identical(grad_cpu, grad_jax),
        "n_bytewise_unequal_grad_doubles": _bytewise_unequal_double_count(
            grad_cpu, grad_jax
        ),
        "argmax_grad_diff_index": _argmax_index(np.abs(diff_grad)),
    }


def _print_summary(payload: dict[str, Any]) -> None:
    print("=== Boozer residual pinned-input byte arbiter ===")
    print(f"bundle: {payload['bundle']}")
    print(f"mode:   {payload['mode']}")
    print(f"host:   {payload['host']['machine']} / {payload['host']['platform']}")
    print(f"jax:    {payload['host']['jax_version']!r}")
    if "residual_only" in payload:
        ro = payload["residual_only"]
        print("--- residual_only (P4.5) ---")
        print(f"  num_res:                       {ro['num_res']}")
        print(f"  max_abs_diff_value:            {ro['max_abs_diff_value']!r}")
        print(f"  byte_identical_value:          {ro['byte_identical_value']}")
        print(f"  max_abs_diff_grad:             {ro['max_abs_diff_grad']!r}")
        print(f"  byte_identical_grad:           {ro['byte_identical_grad']}")
        print(
            f"  n_bytewise_unequal_grad:       {ro['n_bytewise_unequal_grad_doubles']}"
        )
        print(f"  argmax_grad_diff_index:        {ro['argmax_grad_diff_index']}")
    if "full_penalty" in payload:
        fp = payload["full_penalty"]
        print("--- full_penalty (P4.5b) ---")
        print(f"  fixture:                       {fp['fixture']}")
        print(f"  max_abs_diff_value:            {fp['max_abs_diff_value']!r}")
        print(f"  byte_identical_value:          {fp['byte_identical_value']}")
        print(f"  max_abs_diff_grad:             {fp['max_abs_diff_grad']!r}")
        print(f"  byte_identical_grad:           {fp['byte_identical_grad']}")
        print(
            f"  n_bytewise_unequal_grad:       {fp['n_bytewise_unequal_grad_doubles']}"
        )
        print(f"  argmax_grad_diff_index:        {fp['argmax_grad_diff_index']}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 4 P4.5/P4.5b byte arbiter for the Boozer residual "
            "derivative bit-identity zeroing slice. See " + PLAN_DOC + "."
        )
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help=(
            "Directory with the canonical residual-input bundle "
            "(canonical_<name>.npy + manifest.json). Default: "
            f"{DEFAULT_BUNDLE!s}."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("residual_only", "full_penalty", "both"),
        default="both",
        help=(
            "Which arbiter to run. 'residual_only' is P4.5 (canonical bundle "
            "only); 'full_penalty' is P4.5b (rebuilds the synthetic NCSX "
            "fixture); 'both' (default) runs both arbiters."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output directory for byte_arbiter_results.json. Defaults to "
            f"<bundle>/{DEFAULT_RESULTS_SUBDIR}/."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    bundle = args.bundle
    if not bundle.is_dir():
        parser.error(f"bundle directory not found: {bundle!s}")

    out_dir = args.out if args.out is not None else (bundle / DEFAULT_RESULTS_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "bundle": str(bundle),
        "mode": args.mode,
        "plan_doc": PLAN_DOC,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": _capture_host_environment(),
    }

    if args.mode in ("residual_only", "both"):
        canonical = _load_canonical_bundle(bundle)
        missing = [n for n in _CANONICAL_ARRAY_NAMES if n not in canonical]
        missing_scalars = [n for n in _CANONICAL_SCALAR_NAMES if n not in canonical]
        if missing or missing_scalars:
            raise RuntimeError(
                f"canonical bundle missing names: arrays={missing!r} "
                f"scalars={missing_scalars!r}"
            )
        payload["residual_only"] = _run_residual_only(canonical)

    if args.mode in ("full_penalty", "both"):
        payload["full_penalty"] = _run_full_penalty()

    out_path = out_dir / DEFAULT_RESULTS_FILENAME
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _print_summary(payload)
    print(f"\nwrote: {out_path!s}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
