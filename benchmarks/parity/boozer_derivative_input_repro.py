"""Phase 0 reproducer for Boozer derivative input bit-identity census.

The 2026-05-07 BFGS pre-Newton contract slice landed CPU-ordered Boozer
value/gradient routes but the strict acceptance gate stayed red. This script
isolates ONE outer candidate (the first one to diverge in the failing artifact)
and runs ONE inner Boozer LS on each backend (CPU and JAX), producing the raw
derivative inputs that downstream phases of
``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` compare
byte-for-byte.

Phase 0 scope here is the skeleton plus candidate extraction. Phase 1 will
plumb the census via ``--census`` to dump per-array NDJSON.

Usage
-----

    python benchmarks/parity/boozer_derivative_input_repro.py \
        --candidate-source .artifacts/parity/20260507-bfgs-prenewton-cpuordered-vg-m1/cases/cpu_outputs/mpol=2-ntor=2-cfe4b0b6/outer_optimizer_progress.json \
        --line-search-evaluation 4 \
        --dump-arrays .artifacts/parity/20260507-boozer-deriv-input-repro-m1/

Notes
-----

* This file lives under ``benchmarks/parity/``; it is diagnostic, not in the
  production import path.
* The script intentionally records (does not rewrite) ``XLA_FLAGS``. Fast-math
  toggles belong to the FMA ablation side track (plan §5).
* ``OMP_NUM_THREADS`` and ``JAX_ENABLE_X64`` defaults are pinned at the
  subprocess level for the deterministic local variant.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


PLAN_DOC = "docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md"
DEFAULT_ARTIFACT_DIR = Path(".artifacts/parity/20260507-boozer-deriv-input-repro-m1")
DEFAULT_FAILING_ARTIFACT = Path(
    ".artifacts/parity/20260507-bfgs-prenewton-cpuordered-vg-m1/cases/cpu_outputs/"
    "mpol=2-ntor=2-cfe4b0b6/outer_optimizer_progress.json"
)
DEFAULT_LINE_SEARCH_EVALUATION = 4  # First-divergence pair from the failing artifact.
RECORDED_ENV_VARS = (
    "JAX_ENABLE_X64",
    "JAX_PLATFORMS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "XLA_FLAGS",
    "SIMSOPT_BACKEND",
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "CUDA_VISIBLE_DEVICES",
)


@dataclasses.dataclass(frozen=True)
class CandidateRecord:
    """Identifies a single outer-optimizer candidate vector.

    The ``pair_index`` matches ``parity_bug_census.first_divergence.pair_index``
    in the failing artifact; ``line_search_evaluation`` and
    ``accepted_iteration_target`` come from the same event in
    ``outer_optimizer_progress.json``.
    """

    source: Path
    pair_index: int
    line_search_evaluation: int
    accepted_iteration_target: int
    optimizer_method: str
    backend: str
    candidate: tuple[float, ...]
    candidate_size: int
    objective: float | None
    inf_norm: float | None

    @property
    def candidate_id(self) -> str:
        return (
            f"{self.source.name}#pair={self.pair_index}"
            f"#linesearch={self.line_search_evaluation}"
            f"#accepted={self.accepted_iteration_target}"
        )


def _capture_runtime_environment() -> dict[str, Any]:
    """Snapshot the env vars and interpreter state that affect bit-identity."""
    env_snapshot = {name: os.environ.get(name) for name in RECORDED_ENV_VARS}
    return {
        "executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "env": env_snapshot,
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
    }


def _load_objective_evaluation_events(progress_path: Path) -> list[dict[str, Any]]:
    """Filter outer-optimizer progress events to ``objective_evaluation``s.

    Mirrors ``benchmarks/single_stage_init_parity.py::
    _load_objective_evaluation_events_from_case`` so the same selector applies
    here.
    """
    if not progress_path.exists():
        raise FileNotFoundError(
            f"Candidate source progress JSON not found: {progress_path}"
        )
    with progress_path.open() as fh:
        payload = json.load(fh)
    events = payload.get("events", [])
    return [
        dict(event) for event in events if event.get("label") == "objective_evaluation"
    ]


def extract_candidate(
    progress_path: Path,
    *,
    line_search_evaluation: int,
) -> CandidateRecord:
    """Extract a candidate vector from an outer-optimizer progress JSON.

    Args:
        progress_path: Path to ``outer_optimizer_progress.json`` from a case
            artifact.
        line_search_evaluation: ``line_search_evaluation`` field on the target
            ``objective_evaluation`` event. This matches the
            ``parity_bug_census.first_divergence.line_search_evaluation``
            recorded in the failing baseline artifact, so the same value
            selects the same candidate across runs.

    Returns:
        A :class:`CandidateRecord` capturing the candidate vector plus
        provenance fields.
    """
    obj_events = _load_objective_evaluation_events(progress_path)
    matches = [
        (idx, event)
        for idx, event in enumerate(obj_events)
        if int(event.get("line_search_evaluation", -1)) == line_search_evaluation
    ]
    if not matches:
        recorded = sorted(
            {int(event.get("line_search_evaluation", -1)) for event in obj_events}
        )
        raise LookupError(
            f"No objective_evaluation event with "
            f"line_search_evaluation={line_search_evaluation} in {progress_path}; "
            f"recorded values: {recorded}"
        )
    if len(matches) > 1:
        raise LookupError(
            f"Multiple objective_evaluation events match "
            f"line_search_evaluation={line_search_evaluation} in {progress_path}; "
            "supply a more specific selector."
        )
    pair_index, event = matches[0]
    cand = event.get("candidate_optimizer_dofs", {})
    values = cand.get("values")
    if not isinstance(values, list):
        raise ValueError(
            f"Event line_search_evaluation={line_search_evaluation} in "
            f"{progress_path} missing 'candidate_optimizer_dofs.values'"
        )
    objective = event.get("objective", {}).get("value")
    return CandidateRecord(
        source=progress_path,
        pair_index=pair_index,
        line_search_evaluation=int(event.get("line_search_evaluation", -1)),
        accepted_iteration_target=int(event.get("accepted_iteration_target", -1)),
        optimizer_method=str(event.get("optimizer_method", "")),
        backend=str(event.get("backend", "")),
        candidate=tuple(float(v) for v in values),
        candidate_size=int(cand.get("size", len(values))),
        objective=float(objective) if objective is not None else None,
        inf_norm=cand.get("inf_norm"),
    )


def _write_candidate_summary(
    record: CandidateRecord,
    runtime_env: dict[str, Any],
    artifact_dir: Path,
) -> Path:
    """Persist the resolved candidate to the artifact directory.

    Phase 0 only writes a summary; Phase 1 will write per-array NDJSON via
    ``--census``.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out = artifact_dir / "candidate.json"
    payload = {
        "candidate_id": record.candidate_id,
        "candidate_source": str(record.source),
        "pair_index": record.pair_index,
        "line_search_evaluation": record.line_search_evaluation,
        "accepted_iteration_target": record.accepted_iteration_target,
        "optimizer_method": record.optimizer_method,
        "backend": record.backend,
        "candidate_size": record.candidate_size,
        "candidate": list(record.candidate),
        "objective": record.objective,
        "inf_norm": record.inf_norm,
        "runtime": runtime_env,
        "plan_doc": PLAN_DOC,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out


def _build_synthetic_fixture(*, label: str = "Volume", weight_inv_modB: bool = False):
    """Build paired CPU/JAX BoozerSurface objects on an NCSX fixture.

    Phase 1 of ``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md``
    requires a deterministic reproducer that drives both backends past the
    Boozer LS callback boundary. The artifact-driven mode (loading the
    failing ``cpu_outputs/`` seed and replaying the decision vector) belongs
    to Phase 7 regression validation; the synthetic NCSX fixture exercises
    the same boundary helpers and reproduces the same byte-divergence
    ladder. Diagnostic-only.

    Returns a dict of objects required by the census helpers.
    """
    repo_root = Path(__file__).resolve().parents[2]
    helpers_dir = repo_root / "tests" / "geo"
    if str(helpers_dir) not in sys.path:
        sys.path.insert(0, str(helpers_dir))
    from surface_test_helpers import get_boozer_surface  # noqa: PLC0415
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX  # noqa: PLC0415
    from simsopt.geo import (  # noqa: PLC0415
        Volume,
        Area,
        ToroidalFlux,
        SurfaceXYZTensorFourier,
    )
    from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX  # noqa: PLC0415

    bs_cpu, booz_cpu = get_boozer_surface(
        label=label,
        boozer_type="ls",
        optimize_G=True,
        converge=False,
        weight_inv_modB=weight_inv_modB,
    )
    surf_cpu = booz_cpu.surface
    surf_jax = SurfaceXYZTensorFourier(
        mpol=surf_cpu.mpol,
        ntor=surf_cpu.ntor,
        nfp=surf_cpu.nfp,
        stellsym=surf_cpu.stellsym,
        clamped_dims=[False, False, False],
        quadpoints_phi=surf_cpu.quadpoints_phi.copy(),
        quadpoints_theta=surf_cpu.quadpoints_theta.copy(),
    )
    surf_jax.set_dofs(surf_cpu.get_dofs().copy())
    label_cls = type(booz_cpu.label).__name__
    if "Volume" in label_cls:
        label_jax = Volume(surf_jax)
    elif "Area" in label_cls:
        label_jax = Area(surf_jax)
    elif "ToroidalFlux" in label_cls:
        label_jax = ToroidalFlux(surf_jax, BiotSavartJAX(bs_cpu.coils))
    else:
        raise ValueError(f"Unsupported label class for fixture: {label_cls}")
    booz_jax = BoozerSurfaceJAX(
        BiotSavartJAX(bs_cpu.coils),
        surf_jax,
        label_jax,
        booz_cpu.targetlabel,
        constraint_weight=100.0,
        options={"optimizer_backend": "scipy", "weight_inv_modB": weight_inv_modB},
    )
    import numpy as np  # noqa: PLC0415

    mu0 = 4 * np.pi * 1e-7
    G0 = mu0 * sum(abs(c.current.get_value()) for c in bs_cpu.coils)
    return {
        "boozer_cpu": booz_cpu,
        "boozer_jax": booz_jax,
        "biotsavart_cpu": bs_cpu,
        "sdofs": surf_cpu.get_dofs().copy(),
        "iota": -0.406,
        "G": float(G0),
        "weight_inv_modB": weight_inv_modB,
        "optimize_G": True,
    }


def _run_census(
    *,
    artifact_dir: Path,
    runtime_env: dict[str, Any],
    candidate_id: str,
    parity_policy: str = "production",
) -> dict[str, Any]:
    """Run the synthetic-fixture census and emit NDJSON.

    Returns a small summary suitable for printing / assertion in tests.
    """

    from benchmarks.parity.boozer_derivative_input_census import (  # noqa: PLC0415
        capture_cpu_boozer_inputs,
        capture_jax_boozer_inputs,
        compare_boundary_inputs,
        first_divergence,
        write_ndjson,
        CENSUS_BOUNDARY_ARRAY_ORDER,
    )

    fixture = _build_synthetic_fixture()
    cpu_arrays_dict = _materialize_cpu_arrays_dict(fixture)
    jax_arrays_dict = _materialize_jax_arrays_dict(fixture, parity_policy=parity_policy)
    cpu_array_records, cpu_scalar_records = capture_cpu_boozer_inputs(
        fixture["boozer_cpu"],
        sdofs=fixture["sdofs"],
        iota=fixture["iota"],
        G=fixture["G"],
        weight_inv_modB=fixture["weight_inv_modB"],
    )
    jax_array_records, jax_scalar_records = capture_jax_boozer_inputs(
        fixture["boozer_jax"],
        sdofs=fixture["sdofs"],
        iota=fixture["iota"],
        G=fixture["G"],
        weight_inv_modB=fixture["weight_inv_modB"],
        optimize_G=fixture["optimize_G"],
        parity_policy=parity_policy,
    )
    diffs = compare_boundary_inputs(
        cpu_array_records=cpu_array_records,
        cpu_scalar_records=cpu_scalar_records,
        jax_array_records=jax_array_records,
        jax_scalar_records=jax_scalar_records,
        cpu_arrays=cpu_arrays_dict,
        jax_arrays=jax_arrays_dict,
    )
    fd = first_divergence(diffs)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    ndjson_path = artifact_dir / "census.ndjson"
    records = (
        list(cpu_array_records)
        + list(jax_array_records)
        + list(cpu_scalar_records)
        + list(jax_scalar_records)
        + list(diffs)
    )
    write_ndjson(ndjson_path, records)

    summary_path = artifact_dir / "census_summary.json"
    summary = {
        "candidate_id": candidate_id,
        "fixture": "synthetic-ncsx-volume",
        "parity_policy": parity_policy,
        "n_arrays_compared": sum(1 for d in diffs if hasattr(d, "array_name")),
        "n_byte_identical_arrays": sum(
            1 for d in diffs if hasattr(d, "array_name") and d.byte_identical
        ),
        "first_divergence": (
            None
            if fd is None
            else {
                "kind": "array" if hasattr(fd, "array_name") else "scalar",
                "name": getattr(fd, "array_name", None) or fd.name,
                "stage": fd.stage,
                "max_abs_diff": getattr(fd, "max_abs_diff", None)
                or getattr(fd, "abs_diff", None),
            }
        ),
        "ladder": list(CENSUS_BOUNDARY_ARRAY_ORDER),
        "runtime": runtime_env,
        "ndjson": str(ndjson_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _materialize_cpu_arrays_dict(fixture: dict[str, Any]) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415

    inputs = fixture["boozer_cpu"]._boozer_penalty_vectorized_inputs(
        np.asarray(fixture["sdofs"]), 1
    )
    return {
        "gamma": np.asarray(inputs["gamma"]),
        "xphi": np.asarray(inputs["xphi"]),
        "xtheta": np.asarray(inputs["xtheta"]),
        "dx_ds": np.asarray(inputs["dx_dc"]),
        "dxphi_ds": np.asarray(inputs["dxphi_dc"]),
        "dxtheta_ds": np.asarray(inputs["dxtheta_dc"]),
        "B": np.asarray(inputs["B"]),
        "dB_dX": np.asarray(inputs["dB_dx"]),
    }


def _materialize_jax_arrays_dict(
    fixture: dict[str, Any],
    *,
    parity_policy: str = "production",
) -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from simsopt.geo.boozersurface_jax import (  # noqa: PLC0415
        _boozer_penalty_value_and_grad_inputs_cpu_ordered,
        _hostify_tree,
        _resolved_coil_set_spec,
    )

    booz_jax = fixture["boozer_jax"]
    pieces = [
        np.asarray(fixture["sdofs"], dtype=np.float64),
        np.asarray([float(fixture["iota"])], dtype=np.float64),
    ]
    if fixture["optimize_G"]:
        pieces.append(np.asarray([float(fixture["G"])], dtype=np.float64))
    x = jnp.asarray(np.concatenate(pieces), dtype=jnp.float64)
    coil_set_spec = _hostify_tree(_resolved_coil_set_spec(booz_jax.coil_set_spec))
    _, geometry, _, inputs = _boozer_penalty_value_and_grad_inputs_cpu_ordered(
        x,
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
        optimize_G=fixture["optimize_G"],
        parity_policy=parity_policy,
    )

    def _np(value: Any) -> np.ndarray:
        return np.asarray(jax.device_get(value))

    return {
        "gamma": _np(geometry.gamma),
        "xphi": _np(inputs.xphi),
        "xtheta": _np(inputs.xtheta),
        "dx_ds": _np(inputs.dx_ds),
        "dxphi_ds": _np(inputs.dxphi_ds),
        "dxtheta_ds": _np(inputs.dxtheta_ds),
        "B": _np(inputs.B),
        "dB_dX": _np(inputs.dB_dX),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 0 reproducer for Boozer derivative input bit-identity "
            "census. See " + PLAN_DOC + "."
        )
    )
    parser.add_argument(
        "--candidate-source",
        type=Path,
        default=DEFAULT_FAILING_ARTIFACT,
        help=(
            "Path to outer_optimizer_progress.json with objective_evaluation "
            "events. Default points at the failing baseline artifact "
            "(20260507-bfgs-prenewton-cpuordered-vg-m1)."
        ),
    )
    parser.add_argument(
        "--line-search-evaluation",
        type=int,
        default=DEFAULT_LINE_SEARCH_EVALUATION,
        help=(
            "Selects the objective_evaluation event whose "
            "line_search_evaluation field matches this integer. Default "
            f"{DEFAULT_LINE_SEARCH_EVALUATION} matches the first-divergence "
            "pair_index in the failing baseline census."
        ),
    )
    parser.add_argument(
        "--dump-arrays",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=(
            "Output directory for the resolved candidate JSON (and, in "
            "Phase 1, the census NDJSON + per-array .npy dumps)."
        ),
    )
    parser.add_argument(
        "--census",
        action="store_true",
        help=(
            "Phase 1: capture the boundary-input census from a synthetic NCSX "
            "fixture, write census.ndjson + census_summary.json into "
            "--dump-arrays, and report the first non-byte-identical array. "
            "This exercises the same boundary helpers the failing artifact "
            "drives at runtime; the NDJSON ladder is what Phases 2/3 close."
        ),
    )
    parser.add_argument(
        "--print-candidate",
        action="store_true",
        help="Echo the candidate identifier to stdout after extraction.",
    )
    parser.add_argument(
        "--parity-policy",
        choices=("production", "cpu_ordered"),
        default="production",
        help=(
            "Selects which JAX surface kernel feeds the census. "
            "'production' (default) is the matmul/jacfwd hot path; "
            "'cpu_ordered' routes through the surface_fourier_jax_cpu_ordered "
            "twins introduced in Phase 2 of "
            + PLAN_DOC
            + ". Only affects the JAX side of the diff; the CPU side is the "
            "C++ oracle either way."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    runtime_env = _capture_runtime_environment()
    record = extract_candidate(
        args.candidate_source,
        line_search_evaluation=args.line_search_evaluation,
    )
    summary_path = _write_candidate_summary(record, runtime_env, args.dump_arrays)

    print(f"resolved candidate id: {record.candidate_id}")
    if args.print_candidate:
        print(f"candidate_size={record.candidate_size}")
        print(f"objective={record.objective}")
        print(f"inf_norm={record.inf_norm}")
    print(f"wrote candidate summary: {summary_path}")
    if args.census:
        summary = _run_census(
            artifact_dir=args.dump_arrays,
            runtime_env=runtime_env,
            candidate_id=record.candidate_id,
            parity_policy=args.parity_policy,
        )
        fd = summary.get("first_divergence")
        if fd is None:
            print("census: ALL boundary arrays byte-identical")
        else:
            print(
                "census: first divergence "
                f"{fd['kind']}={fd['name']} "
                f"max_abs_diff={fd['max_abs_diff']!r}"
            )
        print(f"census ndjson: {summary['ndjson']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
