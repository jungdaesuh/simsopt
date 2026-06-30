import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt.desc_bridge.objective_factory import (  # noqa: E402
    BOUNDARY_FIDELITY_FIX_HIGH_MODES,
    BOUNDARY_FIDELITY_OFF,
    COIL_SET_MIN_DISTANCE_OBJECTIVE,
    FULL_DESC_OBJECTIVE_ABLATION_POLICY,
    DescObjectiveRuntimeAssemblyError,
    DescObjectiveRuntimeEvaluationError,
    DescObjectiveStackEntry,
    FINITE_NONBINDING_HARDWARE_UPPER_BOUND,
    FIX_BOUNDARY_R_CONSTRAINT,
    FIX_BOUNDARY_Z_CONSTRAINT,
    FIX_COIL_CURRENT_CONSTRAINT,
    FORCE_BALANCE_CONSTRAINT,
    HARD_HARDWARE_AND_FORCE_BALANCE_POLICY,
    HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY,
    HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
    HARDWARE_SDF_KEEPOUT_OBJECTIVE,
    LINKING_CURRENT_GRID_N_CAP,
    NO_COIL_CURVATURE_OBJECTIVE_ABLATION_POLICY,
    NO_COIL_GEOMETRY_OBJECTIVE_ABLATION_POLICY,
    NO_COIL_SET_DISTANCE_OBJECTIVE_ABLATION_POLICY,
    NO_LINKING_CURRENT_OBJECTIVE_ABLATION_POLICY,
    NO_PLASMA_COIL_DISTANCE_OBJECTIVE_ABLATION_POLICY,
    PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY,
    PROXIMAL_FORCE_BALANCE_POLICY,
    VOLUME_OBJECTIVE,
    assemble_desc_objective_stack_runtime,
    build_desc_objective_stack_plan,
    evaluate_desc_objective_stack_runtime,
    validate_objective_stack_for_mode,
)
from banana_opt.desc_bridge.runtime_coilset import (  # noqa: E402
    DescRuntimeCoilsetBuildError,
    _desc_simsopt_field_sample_delta_T,
    _validate_desc_simsopt_field_sample_delta,
    build_desc_runtime_coilset_from_simsopt_field,
    load_desc_runtime_coilset_checkpoint,
    scope_desc_coilset_optimization_to_groups,
)
from banana_opt.desc_bridge.runtime_solve import (  # noqa: E402
    DescFixedPolishRuntimeSolveError,
    DescJointRuntimeSolveError,
    build_desc_optimizer_controls,
    run_desc_fixed_equilibrium_polish_runtime,
    run_desc_joint_optimization_runtime,
)
from banana_opt.desc_bridge.coil_report_utils import (  # noqa: E402
    coil_convention_report,
)
from banana_opt.hardware_contracts import (  # noqa: E402
    TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M,
)
from banana_opt.desc_bridge.runtime_export import (  # noqa: E402
    DescOptimizedSimsoptExportError,
    materialize_optimized_desc_coil_artifact_simsopt_export,
    materialize_optimized_desc_equilibrium_surface_simsopt_export,
)
from banana_opt.desc_bridge.equilibrium_seed import (  # noqa: E402
    DESC_EQUILIBRIUM_SEED_SCHEMA_VERSION,
    DescEquilibriumRuntimeLoadError,
    load_desc_equilibrium_seed_runtime,
    load_desc_equilibrium_seed_spec,
)
from banana_opt.desc_joint_hardware_spec import (  # noqa: E402
    DESC_JOINT_HARDWARE_SPEC_SCHEMA_VERSION,
    load_desc_joint_hardware_spec,
)
from banana_opt.desc_joint_field_inventory import (  # noqa: E402
    load_desc_joint_field_inventory,
)
from banana_opt.desc_joint_result_schema import (  # noqa: E402
    build_preflight_result_payload,
    validate_desc_joint_result_payload,
)
from banana_opt.desc_joint_simsopt_validation import (  # noqa: E402
    DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION,
    build_desc_joint_simsopt_physics_report,
    materialize_desc_joint_simsopt_validation,
)
import banana_opt.desc_joint_validation_launcher as validation_launcher_module  # noqa: E402
from banana_opt.desc_joint_validation_launcher import (  # noqa: E402
    DESC_JOINT_SIMSOPT_VALIDATION_LAUNCH_SCHEMA_VERSION,
    infer_desc_joint_exported_artifact_paths,
    launch_desc_joint_simsopt_validation,
)
import banana_opt.desc_joint_hardware_oracle_launcher as hardware_oracle_module  # noqa: E402
from banana_opt.desc_joint_hardware_oracle_launcher import (  # noqa: E402
    DESC_JOINT_HARDWARE_ORACLE_LAUNCH_SCHEMA_VERSION,
    launch_desc_joint_hardware_oracle,
)
from banana_opt.desc_joint_outer_loop import (  # noqa: E402
    DESC_JOINT_OUTER_LOOP_DECISION_SCHEMA_VERSION,
    materialize_desc_joint_outer_loop_decision,
)
from banana_opt.desc_joint_validation import (  # noqa: E402
    DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
    build_desc_joint_validation_manifest,
    render_desc_joint_validation_report,
    validate_desc_joint_validation_manifest,
)
from banana_opt.desc_joint_seed_manifest import (  # noqa: E402
    DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
    load_desc_joint_seed_manifest,
)
from banana_opt.hardware_contracts import (  # noqa: E402
    BANANA_CURRENT_HARD_LIMIT_A,
    BANANA_WIDTH_MAX_M,
    BANANA_WIDTH_MIN_M,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_HARD_LIMIT_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    TF_CURRENT_HARD_LIMIT_A,
)
from simsopt import load  # noqa: E402
from simsopt.field import BiotSavart  # noqa: E402
from simsopt.field.coil import Coil, Current  # noqa: E402
from simsopt.geo import CurveXYZFourier  # noqa: E402
from simsopt.geo import SurfaceRZFourier  # noqa: E402
from simsopt.geo import SurfaceXYZTensorFourier  # noqa: E402


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_nonnegative_timing(
    timings: dict[str, object],
    timing_key: str,
) -> None:
    value = timings[timing_key]
    assert isinstance(value, float)
    assert value >= 0.0


def _assert_runner_inventory(
    result_payload: dict[str, object],
    *,
    result_path: Path,
) -> dict[str, object]:
    inventory_path = Path(result_payload["run_inventory_path"])
    assert inventory_path == result_path.parent / "desc_joint_run_inventory.json"
    assert inventory_path.is_file()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == "desc_joint_run_inventory_v1"
    assert inventory["result_path"] == str(result_path)
    assert inventory["run_mode"] == result_payload["run_mode"]
    assert inventory["run_configuration"] == result_payload["run_configuration"]
    assert inventory["run_timing_seconds"] == result_payload["run_timing_seconds"]
    assert inventory["objective_stack"] == result_payload["objective_stack"]
    assert inventory["validation_artifacts"] == {
        "manifest": str(result_path.parent / "desc_joint_validation_manifest.json"),
        "report": str(result_path.parent / "desc_joint_validation_report.md"),
    }
    selected_seed = result_payload["input_contract"]["selected_seed"]
    assert inventory["input_artifact_checksums"] == selected_seed["source_checksums"]
    return inventory


def _runner_env_without_desc_runtime_device() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("DESC_JOINT_DESC_DEVICE", None)
    return env


def _assert_legacy_hardware_fields_fail_closed(
    result_payload: dict[str, object],
) -> None:
    assert result_payload["HARDWARE_CONSTRAINTS_OK"] is None
    assert result_payload["HARDWARE_CONSTRAINT_VIOLATIONS"] is None
    assert result_payload["BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK"] is None
    assert result_payload["BEST_FEASIBLE_HARDWARE_CONSTRAINT_VIOLATIONS"] is None
    assert result_payload["BEST_FEASIBLE_AVAILABLE"] is False
    assert result_payload["FINAL_FEASIBILITY_OK"] is False


def _export_binding_payload(exported_artifact_paths: tuple[Path, ...]) -> dict[str, object]:
    resolved_paths = tuple(path.resolve() for path in exported_artifact_paths)
    return {
        "exported_artifact_paths": [str(path) for path in resolved_paths],
        "exported_artifact_checksums": {
            str(path): _sha256(path) for path in resolved_paths
        },
    }


def _write_final_oracle_evidence(
    path: Path,
    *,
    exported_artifact_paths: tuple[str, ...],
    source_artifact_checksums: dict[str, str] | None = None,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
            "source": "direct_loaded_artifact_hardware_contact_oracle",
            "passed": True,
            "exported_artifact_paths": list(exported_artifact_paths),
            "exported_artifact_checksums": {
                artifact_path: _sha256(Path(artifact_path))
                for artifact_path in exported_artifact_paths
            },
            "source_artifact_checksums": (
                {"exported_artifact": "a" * 64}
                if source_artifact_checksums is None
                else source_artifact_checksums
            ),
        },
    )


def _write_poincare_metrics(
    path: Path,
    *,
    exported_artifact_paths: tuple[Path, ...],
    validation_status: str = "validated",
    design_only_override: bool = False,
    nfieldlines: int = 50,
    survived_lines: int = 50,
) -> Path:
    return _write_json(
        path,
        {
            "field_label": "desc_export",
            "render_mode": "validation",
            "validation_status": validation_status,
            "design_only_override": design_only_override,
            "plot_filename": "PoincarePlot_desc_export.png",
            **_export_binding_payload(exported_artifact_paths),
            "metrics": {
                "mode": "validation",
                "nfieldlines": nfieldlines,
                "survived_lines": survived_lines,
                "validation_status": validation_status,
            },
        },
    )


def _write_boozer_state(
    path: Path,
    *,
    exported_artifact_paths: tuple[Path, ...],
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": 1,
            "surface_path": "surf_desc_export_boozer_surface.json",
            "iota": 0.091,
            "G": -2.01062,
            **_export_binding_payload(exported_artifact_paths),
        },
    )


def _write_biot_savart_fixture(
    path: Path,
    currents_A: tuple[float, ...] = (-8.0e4, 1.2e4, 0.0),
    curve_classes: tuple[str | None, ...] | None = None,
) -> Path:
    if curve_classes is not None and len(curve_classes) != len(currents_A):
        raise ValueError("curve_classes length must match currents_A length.")
    objects: dict[str, object] = {}
    coil_refs: list[dict[str, str]] = []
    for index, current_A in enumerate(currents_A, start=1):
        current_name = f"Current{index}"
        coil_name = f"Coil{index}"
        objects[current_name] = {
            "@class": "Current",
            "current": current_A,
        }
        coil_payload: dict[str, object] = {
            "@class": "Coil",
            "current": {"$type": "ref", "value": current_name},
        }
        if curve_classes is not None:
            curve_class = curve_classes[index - 1]
            if curve_class is not None:
                curve_name = f"Curve{index}"
                objects[curve_name] = {"@class": curve_class}
                coil_payload["curve"] = {"$type": "ref", "value": curve_name}
        objects[coil_name] = coil_payload
        coil_refs.append({"$type": "ref", "value": coil_name})
    objects["BiotSavart1"] = {
        "@class": "BiotSavart",
        "coils": coil_refs,
    }
    return _write_json(
        path,
        {
            "@class": "SIMSON",
            "simsopt_objs": objects,
        },
    )


def _write_loadable_biot_savart_fixture(
    path: Path,
    currents_A: tuple[float, ...] = (-8.0e4, 1.2e4, 0.0),
) -> Path:
    coils = [
        _coil(current_A, x_offset=0.8 + 0.05 * index)
        for index, current_A in enumerate(currents_A)
    ]
    BiotSavart(coils).save(str(path))
    return path


def _write_loadable_simsopt_surface_fixture(path: Path) -> Path:
    surface = SurfaceRZFourier(
        nfp=5,
        stellsym=True,
        mpol=2,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0 / 5.0, 7, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 9, endpoint=False),
    )
    surface.set_rc(0, 0, 0.905)
    surface.set_rc(1, 0, 0.053)
    surface.set_zs(1, 0, 0.053)
    surface.set_rc(0, 1, 0.002)
    surface.set_zs(1, 1, 0.001)
    surface.save(str(path))
    return path


def _coil(current_A: float, *, x_offset: float) -> Coil:
    curve = CurveXYZFourier(32, 1)
    dofs = curve.get_dofs().copy()
    dofs[:] = 0.0
    dofs[0] = x_offset
    dofs[1] = 0.02
    dofs[4] = 0.02
    dofs[8] = 0.01
    curve.set_dofs(dofs)
    return Coil(curve, Current(current_A))


def _write_boozer_surface_fixture(path: Path) -> Path:
    return _write_json(
        path,
        {
            "@class": "BoozerSurface",
            "field": {"@class": "BiotSavart"},
        },
    )


def _write_bare_surface_fixture(path: Path) -> Path:
    return _write_json(path, {"@class": "SurfaceXYZTensorFourier"})


def _hardware_spec_fixture(tmp_path: Path) -> Path:
    glb_path = tmp_path / "hardware.glb"
    glb_path.write_bytes(b"live glb")
    keepout_path = _write_json(
        tmp_path / "hardware_keepout.json",
        {
            "schema_version": 1,
            "frame": "machine_metres_zup",
            "units": "m",
            "provenance": {
                "glb": str(glb_path),
                "glb_sha256": _sha256(glb_path),
            },
            "groups": [{"label": "sensor_mounts"}],
            "excluded": {},
        },
    )
    final_oracle_path = tmp_path / "hardware_contact_report.json"
    final_oracle_path.write_text('{"status": "fixture"}\n', encoding="utf-8")
    return _write_json(
        tmp_path / "desc_joint_hardware_spec.json",
        {
            "schema_version": DESC_JOINT_HARDWARE_SPEC_SCHEMA_VERSION,
            "hardware_sources": {
                "glb": str(glb_path),
                "hardware_keepout_json": str(keepout_path),
                "final_oracle": str(final_oracle_path),
            },
            "coil_group_policy": {
                "tf": "fixed",
                "banana": "optimized",
                "proxy": "fixed",
                "vf": "fixed",
            },
        },
    )


def _seed_manifest_fixture(
    tmp_path: Path,
    *,
    loadable_field: bool = False,
    loadable_surface: bool = False,
    source_nfp: int = 5,
    source_stellarator_symmetry: bool | None = True,
) -> Path:
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    source_results_path = tmp_path / "results.json"
    state_path = tmp_path / "surf_chomp_boozer_state.json"
    png_path = tmp_path / "PoincareDefault_slidclean_chomp.png"
    metrics_path = tmp_path / "PoincareDefault_slidclean_chomp.json"
    surface_kind = "bare_surface" if loadable_surface else "boozer_surface"
    if loadable_surface:
        _write_loadable_simsopt_surface_fixture(surface_path)
    else:
        _write_boozer_surface_fixture(surface_path)
    if loadable_field:
        _write_loadable_biot_savart_fixture(field_path)
    else:
        _write_biot_savart_fixture(field_path)
    source_results_payload: dict[str, object] = {
        "COIL_GROUPS": [
            {"role": "tf", "start": 0, "count": 1},
            {"role": "banana", "start": 1, "count": 2},
        ],
        "FINITE_BUILD_ENABLED": True,
        "FINITEBUILD_FILAMENTS_PER_BANANA": 2,
        "FINITEBUILD_NUMFILAMENTS_B": 2,
        "FINITEBUILD_NUMFILAMENTS_N": 1,
        "NFP": source_nfp,
        "TOTAL_COILS": 3,
    }
    if source_stellarator_symmetry is not None:
        source_results_payload["STELLSYM"] = source_stellarator_symmetry
    _write_json(source_results_path, source_results_payload)
    _write_json(state_path, {"iota": 0.091, "G": -2.01062})
    metrics_path.write_text("{}\n", encoding="utf-8")
    png_path.write_bytes(b"png")
    return _write_json(
        tmp_path / "desc_joint_seed_manifest.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "slidclean_chomp",
                    "group": "slid_clean",
                    "surface": str(surface_path),
                    "field": str(field_path),
                    "surface_kind": surface_kind,
                    "source_results": str(source_results_path),
                    "state": str(state_path),
                    "poincare_metrics": str(metrics_path),
                    "poincare_png": str(png_path),
                }
            ],
        },
    )


def _equilibrium_seed_fixture(
    tmp_path: Path,
    *,
    source_kind: str = "simsopt_surface",
    source_filename: str = "seed_surface.json",
) -> Path:
    source_path = tmp_path / source_filename
    source_path.write_text("{}\n", encoding="utf-8")
    return _write_json(
        tmp_path / "equilibrium_seed.json",
        {
            "schema_version": DESC_EQUILIBRIUM_SEED_SCHEMA_VERSION,
            "source_kind": source_kind,
            "source_path": str(source_path),
            "nfp": 5,
            "stellarator_symmetry": True,
            "handedness": "right_handed",
            "angular_convention": "simsopt_theta_phi",
            "major_radius_m": 0.905,
            "minor_radius_m": 0.053,
            "lcfs_mpol": 10,
            "lcfs_ntor": 10,
        },
    )


def _drop_loaded_desc_modules() -> None:
    for module_name in tuple(sys.modules):
        if module_name == "desc" or module_name.startswith("desc."):
            del sys.modules[module_name]


def _fake_desc_source_root(
    tmp_path: Path,
    *,
    supports_linking_grid: bool = True,
    equilibrium_volume: float = 0.0496,
) -> Path:
    _drop_loaded_desc_modules()
    source_root = tmp_path / "fake_desc_source"
    package_root = source_root / "desc"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text(
        "\n".join(
            (
                "import os",
                "",
                '__version__ = "fixture-desc"',
                "config = {'kind': None}",
                "",
                "def set_device(kind='cpu', gpuid=None):",
                "    config['kind'] = kind",
                "    marker = os.environ.get('FAKE_DESC_SET_DEVICE_MARKER')",
                "    if marker:",
                "        with open(marker, 'a', encoding='utf-8') as stream:",
                "            stream.write(f'{kind}\\n')",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "backend.py").write_text(
        "\n".join(
            (
                "import numpy as jnp",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "derivatives.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "",
                "class Derivative:",
                "    def __init__(self, fun, argnum, mode='grad'):",
                "        self.fun = fun",
                "        if isinstance(argnum, tuple):",
                "            self.argnums = argnum",
                "        else:",
                "            self.argnums = (argnum,)",
                "        self.mode = mode",
                "",
                "    def __call__(self, *args):",
                "        value = float(self.fun(*args))",
                "        gradients = []",
                "        for argnum in self.argnums:",
                "            arg = np.asarray(args[argnum], dtype=float)",
                "            gradients.append(np.ones_like(arg, dtype=float) * value)",
                "        if len(gradients) == 1:",
                "            return gradients[0]",
                "        return tuple(gradients)",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "io.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "",
                "class LoadedBoundaryBasis:",
                "    def __init__(self, modes):",
                "        self.modes = np.asarray(modes, dtype=int)",
                "",
                "class LoadedDescSurface:",
                "    R_basis = LoadedBoundaryBasis(",
                "        [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 2], [0, 1, 1]]",
                "    )",
                "    Z_basis = LoadedBoundaryBasis(",
                "        [[0, 1, 0], [0, 2, 0], [0, 0, -2], [0, 1, -1]]",
                "    )",
                "",
                "    def compute(self, names, grid=None, basis='xyz'):",
                "        nodes = np.asarray(grid.nodes, dtype=float)",
                "        theta = nodes[:, 1]",
                "        zeta = nodes[:, 2]",
                "        radius = 1.0 + 0.1 * np.cos(theta)",
                "        z = 0.1 * np.sin(theta)",
                "        xyz = np.column_stack((",
                "            radius * np.cos(zeta),",
                "            radius * np.sin(zeta),",
                "            z,",
                "        ))",
                "        return {'x': xyz}",
                "",
                "class FakeProfile:",
                "    def __init__(self, value):",
                "        self.value = float(value)",
                "",
                "    def __call__(self, rho):",
                "        return np.ones_like(np.asarray(rho, dtype=float)) * self.value",
                "",
                "class LoadedEquilibrium:",
                "    def __init__(self, path):",
                "        self.path = path",
                "        self.L = 18",
                "        self.M = 18",
                "        self.N = 18",
                "        self.L_grid = 36",
                "        self.M_grid = 36",
                "        self.N_grid = 36",
                "        self.surface = LoadedDescSurface()",
                "        self.surface_refresh_rhos = []",
                "        self.change_resolution_calls = []",
                "        self.pressure = FakeProfile(22098.0)",
                "        self.current = FakeProfile(3688.0)",
                "",
                "    def get_surface_at(self, rho):",
                "        self.surface_refresh_rhos.append(float(rho))",
                "        return LoadedDescSurface()",
                "",
                "    def change_resolution(self, L, M, N, L_grid, M_grid, N_grid):",
                "        self.change_resolution_calls.append(",
                "            (L, M, N, L_grid, M_grid, N_grid)",
                "        )",
                "        self.L = L",
                "        self.M = M",
                "        self.N = N",
                "        self.L_grid = L_grid",
                "        self.M_grid = M_grid",
                "        self.N_grid = N_grid",
                "",
                "    def save(self, path):",
                "        with open(path, 'w', encoding='utf-8') as stream:",
                "            stream.write('fake desc equilibrium\\n')",
                "",
                "    def compute(self, names):",
                f"        return {{'V': {equilibrium_volume!r}}}",
                "",
                "def load(path):",
                "    if str(path).endswith('coils.h5'):",
                "        from desc.coils import CoilSet, FourierXYZCoil",
                "        class LoadedFourierXYZCoil(FourierXYZCoil):",
                "            def _compute_position(self, grid=None, basis='xyz'):",
                "                coords = np.asarray(self.coords, dtype=float)",
                "                sample_count = coords.shape[0]",
                "                if grid is not None:",
                "                    sample_count = (",
                "                        int(grid.num_nodes)",
                "                        if hasattr(grid, 'num_nodes')",
                "                        else int(grid)",
                "                    )",
                "                return coords[:sample_count].reshape((1, sample_count, 3))",
                "        base = np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False)",
                "        coils = []",
                "        for index, current in enumerate((0.0, -8.0e4, 1.2e4)):",
                "            coords = np.column_stack((",
                "                0.8 + 0.05 * index + 0.02 * np.cos(base),",
                "                0.02 * np.sin(base),",
                "                0.01 * np.sin(2.0 * base),",
                "            ))",
                "            coils.append(LoadedFourierXYZCoil(",
                "                current, coords, N=3, basis='xyz', name=f'loaded_{index:03d}'",
                "            ))",
                "        return CoilSet(",
                "            *coils, NFP=np.int64(5), sym=np.bool_(True), name='loaded'",
                "        )",
                "    return LoadedEquilibrium(path)",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "vmec.py").write_text(
        "\n".join(
            (
                "class LoadedVmecEquilibrium:",
                "    def __init__(self, path, L, M, N, spectral_indexing, profile):",
                "        self.path = path",
                "        self.L = L",
                "        self.M = M",
                "        self.N = N",
                "        self.spectral_indexing = spectral_indexing",
                "        self.profile = profile",
                "",
                "    def save(self, path):",
                "        with open(path, 'w', encoding='utf-8') as stream:",
                "            stream.write('fake vmec equilibrium\\n')",
                "",
                "    def compute(self, names):",
                f"        return {{'V': {equilibrium_volume!r}}}",
                "",
                "class VMECIO:",
                "    @staticmethod",
                "    def load(path, L, M, N, spectral_indexing, profile):",
                "        return LoadedVmecEquilibrium(",
                "            path, L, M, N, spectral_indexing, profile",
                "        )",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "grid.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "",
                "class Grid:",
                "    def __init__(self, nodes, sort=False, jitable=True):",
                "        self.nodes = np.asarray(nodes, dtype=float)",
                "        self.sort = sort",
                "        self.jitable = jitable",
                "",
                "class LinearGrid:",
                "    def __init__(self, **kwargs):",
                "        self.kwargs = kwargs",
                "        if 'zeta' in kwargs:",
                "            zeta = np.asarray(kwargs['zeta'], dtype=float)",
                "            node_count = int(zeta.size)",
                "        else:",
                "            N = int(kwargs.get('N', 0) or 0)",
                "            node_count = 2 * N + 1 if N > 0 else 1",
                "            zeta = np.linspace(0.0, 2.0 * np.pi, node_count, endpoint=False)",
                "        self.nodes = np.column_stack((",
                "            np.ones(node_count),",
                "            np.zeros(node_count),",
                "            zeta,",
                "        ))",
                "        self.num_nodes = node_count",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "geometry.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "",
                "class FourierRZToroidalSurface:",
                "    def __init__(",
                "        self, coords_rpz, theta, M, N, NFP, sym, check_orientation",
                "    ):",
                "        self.coords_rpz = np.asarray(coords_rpz, dtype=float)",
                "        self.theta = np.asarray(theta, dtype=float)",
                "        self.M = M",
                "        self.N = N",
                "        self.NFP = NFP",
                "        self.sym = sym",
                "        self.check_orientation = check_orientation",
                "",
                "    @classmethod",
                "    def from_values(",
                "        cls, coords, theta, M, N, NFP, sym, check_orientation=True",
                "    ):",
                "        return cls(coords, theta, M, N, NFP, sym, check_orientation)",
                "",
                "    def compute(self, names, grid=None, basis='xyz'):",
                "        coords = self.coords_rpz",
                "        if grid is not None and len(grid.nodes) != len(coords):",
                "            raise ValueError('fixture grid/source sample mismatch')",
                "        radius = coords[:, 0]",
                "        phi = coords[:, 1]",
                "        z = coords[:, 2]",
                "        xyz = np.column_stack(",
                "            (radius * np.cos(phi), radius * np.sin(phi), z)",
                "        )",
                "        return {'x': xyz}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "equilibrium.py").write_text(
        "\n".join(
            (
                "class Equilibrium:",
                "    def __init__(",
                "        self, surface, NFP, L, M, N, L_grid, M_grid, N_grid,",
                "        sym, spectral_indexing, check_orientation, ensure_nested,",
                "        **kwargs",
                "    ):",
                "        self.surface = surface",
                "        self.NFP = NFP",
                "        self.L = L",
                "        self.M = M",
                "        self.N = N",
                "        self.L_grid = L_grid",
                "        self.M_grid = M_grid",
                "        self.N_grid = N_grid",
                "        self.sym = sym",
                "        self.spectral_indexing = spectral_indexing",
                "        self.check_orientation = check_orientation",
                "        self.ensure_nested = ensure_nested",
                "        self.Psi = float(kwargs.get('Psi', 1.0))",
                "        self.unscaled_lcfs_G = -20.0",
                "",
                "    def save(self, path):",
                "        with open(path, 'w', encoding='utf-8') as stream:",
                "            stream.write('fake desc simsopt-surface equilibrium\\n')",
                "",
                "    def compute(self, names, grid=None):",
                "        if names == 'G':",
                "            return {'G': [self.unscaled_lcfs_G * self.Psi]}",
                "        return {'V': 0.0496}",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "coils.py").write_text(
        "\n".join(
            (
                "import numpy as np",
                "",
                "class FourierXYZCoil:",
                "    def __init__(self, current, coords, N, basis, name):",
                "        self.current = current",
                "        self.coords = coords",
                "        self.N = N",
                "        self.basis = basis",
                "        self.name = name",
                "",
                "    @property",
                "    def optimizable_params(self):",
                "        return ['current']",
                "",
                "    @property",
                "    def params_dict(self):",
                "        return {'current': np.asarray([self.current], dtype=float)}",
                "",
                "    @params_dict.setter",
                "    def params_dict(self, params):",
                "        self.current = float(np.asarray(params['current']).reshape(-1)[0])",
                "",
                "    @property",
                "    def dimensions(self):",
                "        return {'current': 1}",
                "",
                "    @property",
                "    def x_idx(self):",
                "        return {'current': np.asarray([0])}",
                "",
                "    @property",
                "    def dim_x(self):",
                "        return 1",
                "",
                "    @classmethod",
                "    def from_values(cls, current, coords, N=10, basis='xyz', name=''):",
                "        return cls(current, coords, N, basis, name)",
                "",
                "    def pack_params(self, params):",
                "        return np.asarray(params['current'], dtype=float).reshape(-1)",
                "",
                "    def unpack_params(self, x):",
                "        return {'current': np.asarray(x, dtype=float).reshape(-1)}",
                "",
                "    def _compute_position(self, grid=None, basis='xyz'):",
                "        if grid is None:",
                "            sample_count = len(self.coords)",
                "        elif hasattr(grid, 'num_nodes'):",
                "            sample_count = int(grid.num_nodes)",
                "        else:",
                "            sample_count = int(grid)",
                "        coords = np.asarray(self.coords, dtype=float)",
                "        if coords.shape[0] < sample_count:",
                "            raise ValueError('fixture sample_count mismatch')",
                "        return coords[:sample_count].reshape((1, sample_count, 3))",
                "",
                "    def copy(self, deepcopy=True):",
                "        return FourierXYZCoil(",
                "            self.current, self.coords.copy(), self.N, self.basis, self.name",
                "        )",
                "",
                "class CoilSet:",
                "    def __init__(",
                "        self, *coils, NFP=1, sym=False, name='', check_intersection=True",
                "    ):",
                "        self.coils = tuple(coils)",
                "        self.NFP = NFP",
                "        self.sym = sym",
                "        self.name = name",
                "        self.check_intersection = check_intersection",
                "",
                "    def __len__(self):",
                "        return len(self.coils)",
                "",
                "    def __getitem__(self, index):",
                "        return self.coils[index]",
                "",
                "    @property",
                "    def num_coils(self):",
                "        return len(self.coils) * (int(self.sym) + 1) * self.NFP",
                "",
                "    @property",
                "    def current(self):",
                "        return [coil.current for coil in self.coils]",
                "",
                "    @property",
                "    def params_dict(self):",
                "        return [coil.params_dict for coil in self.coils]",
                "",
                "    @params_dict.setter",
                "    def params_dict(self, params):",
                "        for coil, coil_params in zip(self.coils, params):",
                "            coil.params_dict = coil_params",
                "",
                "    @property",
                "    def dimensions(self):",
                "        return [coil.dimensions for coil in self.coils]",
                "",
                "    @property",
                "    def dim_x(self):",
                "        return sum(coil.dim_x for coil in self.coils)",
                "",
                "    def pack_params(self, params):",
                "        return np.concatenate(",
                "            [coil.pack_params(coil_params) for coil, coil_params in zip(self.coils, params)]",
                "        )",
                "",
                "    def unpack_params(self, x):",
                "        split = np.split(np.asarray(x, dtype=float), np.arange(1, len(self.coils)))",
                "        return [coil.unpack_params(piece) for coil, piece in zip(self.coils, split)]",
                "",
                "    def _all_currents(self, currents=None):",
                "        if currents is None:",
                "            currents = self.current",
                "        currents = np.asarray(currents, dtype=float).reshape(-1)",
                "        if self.sym:",
                "            currents = np.concatenate((currents, -currents[::-1]))",
                "        return np.tile(currents, self.NFP)",
                "",
                "    def compute(",
                "        self, names, grid=None, params=None, transforms=None,",
                "        data=None, **kwargs",
                "    ):",
                "        return [{'length': 1.0, 'curvature': 1.0} for _coil in self.coils]",
                "",
                "    def _compute_position(self, params=None, grid=None, dx1=False, **kwargs):",
                "        out = np.vstack([coil._compute_position(grid=grid) for coil in self.coils])",
                "        if dx1:",
                "            return out, np.ones_like(out)",
                "        return out",
                "",
                "    def save(self, path):",
                "        with open(path, 'w', encoding='utf-8') as stream:",
                "            stream.write('fake desc coilset\\n')",
                "",
                "    def compute_magnetic_field(",
                "        self, coords, params=None, basis='xyz', source_grid=None,",
                "        transforms=None, chunk_size=None",
                "    ):",
                "        coords = np.asarray(coords, dtype=float)",
                "        field = np.zeros_like(coords)",
                "        for index, coil in enumerate(self.coils, start=1):",
                "            scale = float(coil.current) * 1.0e-12 * index",
                "            field[:, 0] += scale",
                "            field[:, 1] += 0.5 * scale",
                "            field[:, 2] -= 0.25 * scale",
                "        return field",
                "",
                "    def compute_magnetic_vector_potential(",
                "        self, coords, params=None, basis='xyz', source_grid=None,",
                "        transforms=None, chunk_size=None",
                "    ):",
                "        return self.compute_magnetic_field(",
                "            coords, params, basis, source_grid, transforms, chunk_size",
                "        )",
                "",
                "    def copy(self, deepcopy=True):",
                "        return CoilSet(",
                "            *[coil.copy(deepcopy=deepcopy) for coil in self.coils],",
                "            NFP=self.NFP, sym=self.sym, name=self.name,",
                "            check_intersection=self.check_intersection",
                "        )",
                "",
            )
        ),
        encoding="utf-8",
    )
    (package_root / "optimize.py").write_text(
        "\n".join(
            (
                "optimizers = {",
                "    'fixture-success': {'equality_constraints': True},",
                "    'fixture-fail': {'equality_constraints': True},",
                "    'fixture-assert-x0': {'equality_constraints': True},",
                "    'fixture-interrupt': {'equality_constraints': True},",
                "    'fixture-no-eq': {'equality_constraints': False},",
                "    'lsq-auglag': {'equality_constraints': True},",
                "}",
                "",
                "class Optimizer:",
                "    _wrappers = [None, 'prox', 'proximal']",
                "",
                "    def __init__(self, method):",
                "        self.method = method",
                "",
                "    def optimize(",
                "        self, things, objective, constraints=(),",
                "        ftol=None, xtol=None, gtol=None, ctol=None,",
                "        verbose=1, maxiter=None, options=None, copy=False",
                "    ):",
                "        if isinstance(things, (list, tuple)):",
                "            optimized_things = list(things)",
                "        else:",
                "            optimized_things = [things]",
                "        expected_order = getattr(",
                "            objective, 'expected_optimizer_thing_order', None",
                "        )",
                "        if expected_order is not None:",
                "            actual_order = tuple(type(thing).__name__ for thing in things)",
                "            assert actual_order == tuple(expected_order), actual_order",
                "        constraint_tuple = tuple(constraints)",
                "        constraint_types = tuple(",
                "            type(constraint).__name__",
                "            for constraint in constraint_tuple",
                "        )",
                "        if self.method == 'fixture-fail':",
                "            return optimized_things, {",
                "                'success': False,",
                "                'status': 2,",
                "                'message': 'fixture requested failure',",
                "                'nit': 1,",
                "                'nfev': 2,",
                "            }",
                "        if self.method == 'fixture-assert-x0':",
                "            raise AssertionError('x0 is infeasible')",
                "        if self.method == 'fixture-interrupt':",
                "            raise KeyboardInterrupt('fixture optimizer interrupted')",
                "        for thing in optimized_things:",
                "            thing.optimized_by = self.method",
                "            thing.optimizer_maxiter = maxiter",
                "            thing.optimizer_verbose = verbose",
                "            thing.optimizer_tolerances = {",
                "                'ftol': ftol,",
                "                'xtol': xtol,",
                "                'gtol': gtol,",
                "                'ctol': ctol,",
                "            }",
                "            thing.optimizer_options = {} if options is None else dict(options)",
                "            thing.optimizer_copy = copy",
                "            thing.optimizer_constraint_count = len(constraint_tuple)",
                "            thing.optimizer_constraint_types = constraint_types",
                "        return optimized_things, {",
                "            'success': True,",
                "            'status': 1,",
                "            'message': 'fixture success',",
                "            'nit': 3,",
                "            'nfev': 4,",
                "        }",
                "",
            )
        ),
        encoding="utf-8",
    )
    linking_current_class = (
        (
            "class LinkingCurrentConsistency(_ObjectiveTerm):",
            "    def __init__(",
            "        self,",
            "        equilibrium,",
            "        coilset,",
            "        *,",
                "        eq_fixed=False,",
                "        linking_grid,",
                "        weight=1.0,",
                "        normalize=True,",
                "        jac_chunk_size=1,",
                "    ):",
                "        things = (coilset,) if eq_fixed else (coilset, equilibrium)",
                "        super().__init__(",
                "            *things,",
                "            eq_fixed=eq_fixed,",
                "            linking_grid=linking_grid,",
                "            weight=weight,",
                "            normalize=normalize,",
                "            jac_chunk_size=jac_chunk_size,",
                "        )",
        )
        if supports_linking_grid
        else (
            "class LinkingCurrentConsistency(_ObjectiveTerm):",
            "    def __init__(",
            "        self,",
            "        equilibrium,",
            "        coilset,",
                "        *,",
                "        eq_fixed=False,",
                "        weight=1.0,",
                "        normalize=True,",
                "        jac_chunk_size=1,",
                "    ):",
                "        things = (coilset,) if eq_fixed else (coilset, equilibrium)",
                "        super().__init__(",
                "            *things,",
                "            eq_fixed=eq_fixed,",
                "            weight=weight,",
                "            normalize=normalize,",
                "            jac_chunk_size=jac_chunk_size,",
                "        )",
        )
    )
    (package_root / "objectives.py").write_text(
        "\n".join(
            (
                "class _ObjectiveTerm:",
                "    def __init__(self, *args, **kwargs):",
                "        self.args = args",
                "        self.kwargs = kwargs",
                "        self.things = []",
                "        for arg in args:",
                "            if arg not in self.things:",
                "                self.things.append(arg)",
                "        for key in ('field', 'coil', 'coilset'):",
                "            if key in kwargs and kwargs[key] not in self.things:",
                "                self.things.append(kwargs[key])",
                "",
                "    def build(self, use_jit=False, verbose=0):",
                "        self.use_jit = use_jit",
                "        self.verbose = verbose",
                "        for objective in self.objectives:",
                "            objective.build(use_jit=use_jit, verbose=verbose)",
                "        self.built = True",
                "",
                "    def xs(self, *things):",
                "        self.xs_things = tuple(things)",
                "        return tuple(float(index + 1) for index in range(len(things)))",
                "",
                "    def compute_scaled_error(self, *args):",
                "        return [float(len(args) + 1)]",
                "",
                "    def compute_quadratic_scalar(self, *args):",
                "        scaled_error = self.compute_scaled_error(*args)",
                "        return 0.5 * sum(value * value for value in scaled_error)",
                "",
                "class ObjectiveFunction:",
                "    def __init__(",
                "        self,",
                "        objectives,",
                "        use_jit=False,",
                "        deriv_mode='auto',",
                "        jac_chunk_size='auto',",
                "    ):",
                "        self.objectives = tuple(objectives)",
                "        self.constructor_use_jit = use_jit",
                "        self.constructor_deriv_mode = deriv_mode",
                "        self.constructor_jac_chunk_size = jac_chunk_size",
                "        self._things = []",
                "        for objective in self.objectives:",
                "            for thing in objective.things:",
                "                if thing not in self._things:",
                "                    self._things.append(thing)",
                "        self.built = False",
                "",
                "    @property",
                "    def things(self):",
                "        return tuple(self._things)",
                "",
                "    def build(self, use_jit=False, verbose=0):",
                "        self.use_jit = use_jit",
                "        self.verbose = verbose",
                "        self.built = True",
                "",
                "    @property",
                "    def dim_x(self):",
                "        return sum(getattr(thing, 'dim_x', 1) for thing in self._things)",
                "",
                "    def x(self, *things):",
                "        self.x_things = tuple(things)",
                "        dim_x = sum(getattr(thing, 'dim_x', 1) for thing in things)",
                "        return [float(index + 1) for index in range(dim_x)]",
                "",
                "    def compute_scaled_error(self, x):",
                "        raise AssertionError('combined value path must not run')",
                "",
                "    def jac_scaled_error(self, x):",
                "        rows = []",
                "        for index, _objective in enumerate(self.objectives):",
                "            rows.append([float(index + 1) for _ in x])",
                "        return rows",
                "",
                "class QuadraticFlux(_ObjectiveTerm):",
                "    def __init__(self, equilibrium, field, **kwargs):",
                "        super().__init__(field, **kwargs)",
                *linking_current_class,
                "class CoilSetMinDistance(_ObjectiveTerm):",
                "    pass",
                "class CoilSetSDFDistance(_ObjectiveTerm):",
                "    pass",
                "class PlasmaCoilSetMinDistance(_ObjectiveTerm):",
                "    def __init__(",
                "        self, equilibrium, coilset, *args, eq_fixed=False, **kwargs",
                "    ):",
                "        things = (coilset,) if eq_fixed else (coilset, equilibrium)",
                "        super().__init__(*things, eq_fixed=eq_fixed, **kwargs)",
                "class CoilLength(_ObjectiveTerm):",
                "    pass",
                "class CoilCurvature(_ObjectiveTerm):",
                "    pass",
                "class BoundaryError(_ObjectiveTerm):",
                "    pass",
                "class VacuumBoundaryError(_ObjectiveTerm):",
                "    pass",
                "class ForceBalance(_ObjectiveTerm):",
                "    pass",
                "class FixBoundaryR(_ObjectiveTerm):",
                "    pass",
                "class FixBoundaryZ(_ObjectiveTerm):",
                "    pass",
                "class FixCoilCurrent(_ObjectiveTerm):",
                "    pass",
                "class Volume(_ObjectiveTerm):",
                "    pass",
                "",
            )
        ),
        encoding="utf-8",
    )
    return source_root


def _desc_solve_passed_payload(payload: dict[str, object]) -> dict[str, object]:
    passed_payload = dict(payload)
    passed_payload["desc_solve_status"] = {
        "state": "passed",
        "reason": "fixture DESC optimization passed",
        "artifact_paths": [],
    }
    return passed_payload


def _write_fixed_polish_predecessor_manifest(
    path: Path,
    *,
    source_artifact_checksums: dict[str, str],
) -> Path:
    exported_artifact_path = (
        path.parent / f"{path.stem}_fixed_exported_biot_savart.json"
    )
    poincare_metrics_path = (
        path.parent / f"{path.stem}_fixed_poincare_metrics.json"
    )
    physics_evidence_path = path.parent / f"{path.stem}_physics_validation.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    _write_poincare_metrics(
        poincare_metrics_path,
        exported_artifact_paths=(exported_artifact_path,),
    )
    physics_report = build_desc_joint_simsopt_physics_report(
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(poincare_metrics_path,),
    )
    _write_json(physics_evidence_path, physics_report)
    fixed_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={
                "selected_seed": {"source_checksums": source_artifact_checksums},
            },
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )
    manifest = build_desc_joint_validation_manifest(
        result_payload=fixed_payload,
        exported_artifact_paths=(str(exported_artifact_path),),
        expected_source_artifact_checksums=source_artifact_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=None,
        search_hardware_passed=None,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
        physics_validation_evidence_paths=(str(physics_evidence_path),),
    )
    return _write_json(path, manifest)


def _write_lane_b_predecessor_manifest(
    path: Path,
    *,
    source_artifact_checksums: dict[str, str],
    physics_passed: bool = True,
    run_mode: str = "vacuum_joint",
) -> Path:
    exported_artifact_path = path.parent / f"{path.stem}_lane_b_exported.json"
    poincare_metrics_path = path.parent / f"{path.stem}_lane_b_poincare.json"
    physics_evidence_path = path.parent / f"{path.stem}_lane_b_physics.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    _write_poincare_metrics(
        poincare_metrics_path,
        exported_artifact_paths=(exported_artifact_path,),
        validation_status="validated" if physics_passed else "fails_validation",
        survived_lines=50 if physics_passed else 0,
    )
    physics_report = build_desc_joint_simsopt_physics_report(
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(poincare_metrics_path,),
    )
    _write_json(physics_evidence_path, physics_report)
    lane_b_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode=run_mode,
            input_contract={
                "selected_seed": {"source_checksums": source_artifact_checksums},
            },
            objective_stack=[
                "VacuumBoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    manifest = build_desc_joint_validation_manifest(
        result_payload=lane_b_payload,
        exported_artifact_paths=(str(exported_artifact_path),),
        expected_source_artifact_checksums=source_artifact_checksums,
        physics_validation_passed=physics_report["passed"] is True,
        artifact_hardware_passed=None,
        search_hardware_passed=None,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
        physics_validation_evidence_paths=(str(physics_evidence_path),),
    )
    return _write_json(path, manifest)


def _record_result_exported_artifact_paths(
    result_payload: dict[str, object],
    exported_artifact_paths: tuple[str, ...],
) -> None:
    artifact_hardware_status = result_payload["artifact_hardware_status"]
    assert isinstance(artifact_hardware_status, dict)
    artifact_hardware_status["artifact_paths"] = list(exported_artifact_paths)


def test_desc_joint_hardware_spec_resolves_thresholds_through_hardware_schema(tmp_path):
    spec = load_desc_joint_hardware_spec(_hardware_spec_fixture(tmp_path))

    thresholds = spec.threshold_by_name()

    assert thresholds["coil_length"] == COIL_LENGTH_HARD_LIMIT_M
    assert thresholds["coil_coil_spacing"] == COIL_COIL_MIN_DIST_M
    assert thresholds["coil_surface_spacing"] == COIL_PLASMA_MIN_DIST_M
    assert thresholds["max_curvature"] == MAX_CURVATURE_INV_M
    assert thresholds["banana_current"] == BANANA_CURRENT_HARD_LIMIT_A
    assert thresholds["tf_current"] == TF_CURRENT_HARD_LIMIT_A
    assert thresholds["width_min"] == BANANA_WIDTH_MIN_M
    assert thresholds["width_max"] == BANANA_WIDTH_MAX_M
    assert "HARDWARE_CONSTRAINTS_OK" in spec.artifact_payload_field_names()
    assert spec.hardware_metadata["HARDWARE_KEEPOUT_BACKEND"] == "point_cloud"


def test_desc_joint_hardware_spec_fails_closed_on_stale_keepout_glb(tmp_path):
    spec_path = _hardware_spec_fixture(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    keepout_path = Path(payload["hardware_sources"]["hardware_keepout_json"])
    keepout = json.loads(keepout_path.read_text(encoding="utf-8"))
    keepout["provenance"]["glb_sha256"] = "not-the-live-sha"
    _write_json(keepout_path, keepout)

    with pytest.raises(ValueError, match="stale hardware keep-out cloud"):
        load_desc_joint_hardware_spec(spec_path)


def test_desc_joint_hardware_spec_rejects_removed_required_constraint(tmp_path):
    spec_path = _hardware_spec_fixture(tmp_path)
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    payload["constraint_names"] = [
        "coil_length",
        "coil_length_min",
        "coil_coil_spacing",
        "coil_surface_spacing",
        "max_curvature",
        "banana_current",
        "tf_current",
        "width_min",
        "width_max",
    ]
    _write_json(spec_path, payload)

    with pytest.raises(ValueError, match="hardware_keepout"):
        load_desc_joint_hardware_spec(spec_path)


def test_desc_joint_seed_manifest_preserves_full_and_bare_surface_distinction(tmp_path):
    bare_surface_path = tmp_path / "baseline_m36_converged.json"
    bare_field_path = tmp_path / "slid_cws_field.json"
    full_surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    full_field_path = tmp_path / "slid_cws_field_chomp.json"
    _write_bare_surface_fixture(bare_surface_path)
    _write_biot_savart_fixture(bare_field_path)
    _write_boozer_surface_fixture(full_surface_path)
    _write_biot_savart_fixture(full_field_path)
    manifest_path = _write_json(
        tmp_path / "seed_manifest.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "m36_baseline",
                    "group": "m36",
                    "surface": str(bare_surface_path),
                    "field": str(bare_field_path),
                    "surface_kind": "bare_surface",
                    "coil_groups": [
                        {"name": "tf", "count": 1},
                        {"name": "banana", "count": 2},
                    ],
                },
                {
                    "label": "slidclean_chomp",
                    "group": "slid_clean",
                    "surface": str(full_surface_path),
                    "field": str(full_field_path),
                    "surface_kind": "boozer_surface",
                    "coil_groups": [
                        {"name": "tf", "count": 1},
                        {"name": "banana", "count": 2},
                    ],
                },
            ],
        },
    )

    manifest = load_desc_joint_seed_manifest(manifest_path)

    assert manifest.candidate_by_label("m36_baseline").surface_kind == "bare_surface"
    assert manifest.candidate_by_label("m36_baseline").source_checksums() == {
        "surface": _sha256(bare_surface_path),
        "field": _sha256(bare_field_path),
    }
    assert [
        coil_group.to_json_dict()
        for coil_group in manifest.candidate_by_label("m36_baseline").coil_groups
    ] == [{"name": "tf", "count": 1}, {"name": "banana", "count": 2}]
    assert manifest.candidate_by_label("m36_baseline").coil_group_source == "manifest"
    assert (
        manifest.candidate_by_label("slidclean_chomp").surface_kind
        == "boozer_surface"
    )
    assert manifest.to_input_contract()["candidates"][0]["source_checksums"] == {
        "surface": _sha256(bare_surface_path),
        "field": _sha256(bare_field_path),
    }


def test_desc_joint_seed_manifest_resolves_groups_from_source_results(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    source_results_path = tmp_path / "results.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(field_path)
    _write_json(
        source_results_path,
        {
            "COIL_GROUPS": [
                {"role": "tf", "start": 0, "count": 1},
                {"role": "banana", "start": 1, "count": 2},
            ],
        },
    )
    manifest_path = _write_json(
        tmp_path / "seed_manifest.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "slidclean_chomp",
                    "group": "slid_clean",
                    "surface": str(surface_path),
                    "field": str(field_path),
                    "surface_kind": "boozer_surface",
                    "source_results": str(source_results_path),
                },
            ],
        },
    )

    candidate = load_desc_joint_seed_manifest(manifest_path).candidate_by_label(
        "slidclean_chomp"
    )

    assert candidate.coil_group_source == "source_results"
    assert [coil_group.to_json_dict() for coil_group in candidate.coil_groups] == [
        {"name": "tf", "count": 1},
        {"name": "banana", "count": 2},
    ]
    assert candidate.source_checksums()["source_results"] == _sha256(
        source_results_path
    )


def test_desc_joint_seed_manifest_rejects_materialized_cws_flattened_field(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    source_results_path = tmp_path / "results.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveXYZFourier",
            "CurveXYZFourier",
        ),
    )
    _write_json(
        source_results_path,
        {
            "SINGLE_STAGE_BANANA_GEOMETRY_MODE": "materialized_cws",
            "COIL_GROUPS": [
                {"role": "tf", "start": 0, "count": 1},
                {"role": "banana", "start": 1, "count": 2},
            ],
        },
    )
    manifest_path = _write_json(
        tmp_path / "seed_manifest.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "slidclean_chomp",
                    "group": "slid_clean",
                    "surface": str(surface_path),
                    "field": str(field_path),
                    "surface_kind": "boozer_surface",
                    "source_results": str(source_results_path),
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="0 CurveCWSFourierCPP"):
        load_desc_joint_seed_manifest(manifest_path)


def test_desc_joint_seed_manifest_ignores_unreferenced_cws_objects(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    source_results_path = tmp_path / "results.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveXYZFourier",
            "CurveXYZFourier",
        ),
    )
    field_payload = json.loads(field_path.read_text(encoding="utf-8"))
    field_objects = field_payload["simsopt_objs"]
    assert isinstance(field_objects, dict)
    field_objects["UnusedCurveCWSFourierCPP"] = {
        "@class": "CurveCWSFourierCPP",
    }
    _write_json(field_path, field_payload)
    _write_json(
        source_results_path,
        {
            "SINGLE_STAGE_BANANA_GEOMETRY_MODE": "materialized_cws",
            "COIL_GROUPS": [
                {"role": "tf", "start": 0, "count": 1},
                {"role": "banana", "start": 1, "count": 2},
            ],
        },
    )
    manifest_path = _write_json(
        tmp_path / "seed_manifest.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "slidclean_chomp",
                    "group": "slid_clean",
                    "surface": str(surface_path),
                    "field": str(field_path),
                    "surface_kind": "boozer_surface",
                    "source_results": str(source_results_path),
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="0 CurveCWSFourierCPP"):
        load_desc_joint_seed_manifest(manifest_path)


def test_desc_joint_seed_manifest_accepts_materialized_cws_field(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "slid_cws_field_chomp.json"
    source_results_path = tmp_path / "results.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveCWSFourierCPP",
            "CurveCWSFourierCPP",
        ),
    )
    _write_json(
        source_results_path,
        {
            "SINGLE_STAGE_BANANA_GEOMETRY_MODE": "materialized_cws",
            "COIL_GROUPS": [
                {"role": "tf", "start": 0, "count": 1},
                {"role": "banana", "start": 1, "count": 2},
            ],
        },
    )
    manifest_path = _write_json(
        tmp_path / "seed_manifest.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "slidclean_chomp",
                    "group": "slid_clean",
                    "surface": str(surface_path),
                    "field": str(field_path),
                    "surface_kind": "boozer_surface",
                    "source_results": str(source_results_path),
                },
            ],
        },
    )

    candidate = load_desc_joint_seed_manifest(manifest_path).candidate_by_label(
        "slidclean_chomp"
    )

    assert candidate.coil_group_source == "source_results"
    assert [coil_group.to_json_dict() for coil_group in candidate.coil_groups] == [
        {"name": "tf", "count": 1},
        {"name": "banana", "count": 2},
    ]


def test_desc_joint_seed_manifest_rejects_group_count_mismatch(tmp_path):
    surface_path = tmp_path / "surf_chomp_boozer_surface.json"
    field_path = tmp_path / "biot_savart_opt.json"
    _write_boozer_surface_fixture(surface_path)
    _write_biot_savart_fixture(field_path)
    manifest_path = _write_json(
        tmp_path / "seed_manifest.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "bad_groups",
                    "group": "fixture",
                    "surface": str(surface_path),
                    "field": str(field_path),
                    "surface_kind": "boozer_surface",
                    "coil_groups": [{"name": "tf", "count": 1}],
                },
            ],
        },
    )

    with pytest.raises(ValueError, match="does not match field coil count"):
        load_desc_joint_seed_manifest(manifest_path)


def test_desc_joint_seed_manifest_rejects_surface_kind_mismatch(tmp_path):
    bare_surface_path = tmp_path / "bare_surface.json"
    full_surface_path = tmp_path / "boozer_surface.json"
    field_path = tmp_path / "biot_savart.json"
    _write_bare_surface_fixture(bare_surface_path)
    _write_boozer_surface_fixture(full_surface_path)
    _write_biot_savart_fixture(field_path)

    full_declared_bare_path = _write_json(
        tmp_path / "full_declared_bare.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "bad_full",
                    "group": "fixture",
                    "surface": str(full_surface_path),
                    "field": str(field_path),
                    "surface_kind": "bare_surface",
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="declares surface_kind='bare_surface'"):
        load_desc_joint_seed_manifest(full_declared_bare_path)

    bare_declared_full_path = _write_json(
        tmp_path / "bare_declared_full.json",
        {
            "schema_version": DESC_JOINT_SEED_MANIFEST_SCHEMA_VERSION,
            "candidates": [
                {
                    "label": "bad_bare",
                    "group": "fixture",
                    "surface": str(bare_surface_path),
                    "field": str(field_path),
                    "surface_kind": "boozer_surface",
                },
            ],
        },
    )
    with pytest.raises(ValueError, match="does not contain a BoozerSurface object"):
        load_desc_joint_seed_manifest(bare_declared_full_path)


def test_desc_joint_field_inventory_resolves_current_signs(tmp_path):
    field_path = tmp_path / "biot_savart.json"
    _write_json(
        field_path,
        {
            "@class": "SIMSON",
            "simsopt_objs": {
                "Current1": {"@class": "Current", "current": 1.0},
                "ScaledCurrent1": {
                    "@class": "ScaledCurrent",
                    "current_to_scale": {"$type": "ref", "value": "Current1"},
                    "scale": -8.0e4,
                },
                "Coil1": {
                    "@class": "Coil",
                    "current": {"$type": "ref", "value": "ScaledCurrent1"},
                },
                "Current2": {"@class": "Current", "current": 1.2e4},
                "Coil2": {
                    "@class": "Coil",
                    "current": {"$type": "ref", "value": "Current2"},
                },
                "Current3": {"@class": "Current", "current": 0.0},
                "Coil3": {
                    "@class": "Coil",
                    "current": {"$type": "ref", "value": "Current3"},
                },
                "BiotSavart1": {
                    "@class": "BiotSavart",
                    "coils": [
                        {"$type": "ref", "value": "Coil1"},
                        {"$type": "ref", "value": "Coil2"},
                        {"$type": "ref", "value": "Coil3"},
                    ],
                },
            },
        },
    )

    inventory = load_desc_joint_field_inventory(field_path)

    assert inventory.coil_count == 3
    assert inventory.cws_curve_count == 0
    assert inventory.xyz_curve_count == 0
    assert inventory.current_values_A == (-8.0e4, 1.2e4, 0.0)
    assert inventory.current_signs == ("negative", "positive", "zero")
    assert inventory.to_json_dict()["current_sign_counts"] == {
        "negative": 1,
        "zero": 1,
        "positive": 1,
    }
    assert inventory.to_json_dict()["coil_conventions"] == coil_convention_report()


def test_desc_joint_field_inventory_counts_only_referenced_field_curves(tmp_path):
    field_path = tmp_path / "biot_savart.json"
    _write_biot_savart_fixture(
        field_path,
        curve_classes=(
            "CurveXYZFourier",
            "CurveCWSFourierCPP",
            None,
        ),
    )
    field_payload = json.loads(field_path.read_text(encoding="utf-8"))
    field_objects = field_payload["simsopt_objs"]
    assert isinstance(field_objects, dict)
    field_objects["UnusedCurveCWSFourierCPP"] = {
        "@class": "CurveCWSFourierCPP",
    }
    field_objects["UnusedCurveXYZFourier"] = {"@class": "CurveXYZFourier"}
    _write_json(field_path, field_payload)

    inventory = load_desc_joint_field_inventory(field_path)

    assert inventory.coil_count == 3
    assert inventory.cws_curve_count == 1
    assert inventory.xyz_curve_count == 1


def test_desc_joint_objective_stack_rejects_quadratic_flux_in_joint_modes():
    fixed_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "fixed_equilibrium_polish",
            include_hardware_keepout=False,
        )
    ]
    vacuum_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "vacuum_joint",
            include_hardware_keepout=False,
        )
    ]
    finite_beta_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "finite_beta_joint",
            include_hardware_keepout=True,
        )
    ]

    assert "QuadraticFlux" in fixed_names
    assert COIL_SET_MIN_DISTANCE_OBJECTIVE in fixed_names
    assert "QuadraticFlux" not in vacuum_names
    assert "VacuumBoundaryError" in vacuum_names
    assert VOLUME_OBJECTIVE in vacuum_names
    assert FORCE_BALANCE_CONSTRAINT in vacuum_names
    assert COIL_SET_MIN_DISTANCE_OBJECTIVE in vacuum_names
    assert "BoundaryError" in finite_beta_names
    assert VOLUME_OBJECTIVE in finite_beta_names
    assert FORCE_BALANCE_CONSTRAINT in finite_beta_names
    assert COIL_SET_MIN_DISTANCE_OBJECTIVE in finite_beta_names
    assert HARDWARE_SDF_KEEPOUT_OBJECTIVE in finite_beta_names
    with pytest.raises(ValueError, match="must not include QuadraticFlux"):
        validate_objective_stack_for_mode(
            "vacuum_joint",
            (
                DescObjectiveStackEntry(
                    "QuadraticFlux",
                    "physics",
                    True,
                    "regression fixture",
                ),
            ),
        )


def test_desc_joint_objective_stack_applies_physics_only_ablation():
    stack = build_desc_objective_stack_plan(
        "vacuum_joint",
        include_hardware_keepout=True,
        joint_constraint_policy=PROXIMAL_FORCE_BALANCE_POLICY,
        objective_ablation_policy=PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY,
    )

    assert [entry.name for entry in stack] == [
        "VacuumBoundaryError",
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]


def test_desc_joint_objective_stack_marks_hardware_constraints():
    stack = build_desc_objective_stack_plan(
        "vacuum_joint",
        include_hardware_keepout=True,
        joint_constraint_policy=HARD_HARDWARE_AND_FORCE_BALANCE_POLICY,
    )

    entries_by_name = {entry.name: entry for entry in stack}
    assert entries_by_name[VOLUME_OBJECTIVE].runtime_constraint is True
    assert entries_by_name[FORCE_BALANCE_CONSTRAINT].runtime_constraint is True
    assert entries_by_name[COIL_SET_MIN_DISTANCE_OBJECTIVE].runtime_constraint is True
    assert entries_by_name["PlasmaCoilSetMinDistance"].runtime_constraint is True
    assert entries_by_name["CoilLength"].runtime_constraint is True
    assert entries_by_name["CoilCurvature"].runtime_constraint is True
    assert entries_by_name[HARDWARE_SDF_KEEPOUT_OBJECTIVE].runtime_constraint is True
    assert entries_by_name[FIX_COIL_CURRENT_CONSTRAINT].runtime_constraint is True
    assert entries_by_name["VacuumBoundaryError"].runtime_constraint is False
    assert entries_by_name["LinkingCurrentConsistency"].runtime_constraint is False


def test_desc_joint_objective_stack_marks_linking_current_constraint():
    stack = build_desc_objective_stack_plan(
        "vacuum_joint",
        include_hardware_keepout=True,
        joint_constraint_policy=HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY,
    )

    entries_by_name = {entry.name: entry for entry in stack}
    assert entries_by_name["LinkingCurrentConsistency"].runtime_constraint is True
    assert entries_by_name[FORCE_BALANCE_CONSTRAINT].runtime_constraint is True
    assert entries_by_name[VOLUME_OBJECTIVE].runtime_constraint is False
    assert entries_by_name[COIL_SET_MIN_DISTANCE_OBJECTIVE].runtime_constraint is False
    assert entries_by_name["PlasmaCoilSetMinDistance"].runtime_constraint is False
    assert entries_by_name["CoilLength"].runtime_constraint is False
    assert entries_by_name["CoilCurvature"].runtime_constraint is False
    assert entries_by_name[HARDWARE_SDF_KEEPOUT_OBJECTIVE].runtime_constraint is False
    assert entries_by_name[FIX_COIL_CURRENT_CONSTRAINT].runtime_constraint is True


def test_desc_joint_objective_stack_marks_boundary_fidelity_constraints():
    stack = build_desc_objective_stack_plan(
        "vacuum_joint",
        include_hardware_keepout=False,
        boundary_fidelity_policy=BOUNDARY_FIDELITY_FIX_HIGH_MODES,
    )

    entries_by_name = {entry.name: entry for entry in stack}
    assert entries_by_name[FIX_BOUNDARY_R_CONSTRAINT].runtime_constraint is True
    assert entries_by_name[FIX_BOUNDARY_Z_CONSTRAINT].runtime_constraint is True
    assert entries_by_name[FIX_BOUNDARY_R_CONSTRAINT].role == "physics"
    assert entries_by_name[FIX_BOUNDARY_Z_CONSTRAINT].role == "physics"

    with pytest.raises(ValueError, match="only supported for DESC joint modes"):
        build_desc_objective_stack_plan(
            "fixed_equilibrium_polish",
            include_hardware_keepout=False,
            boundary_fidelity_policy=BOUNDARY_FIDELITY_FIX_HIGH_MODES,
        )


def test_desc_joint_objective_stack_applies_coil_family_ablations():
    no_curvature_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "vacuum_joint",
            include_hardware_keepout=True,
            objective_ablation_policy=NO_COIL_CURVATURE_OBJECTIVE_ABLATION_POLICY,
        )
    ]
    no_plasma_distance_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "vacuum_joint",
            include_hardware_keepout=True,
            objective_ablation_policy=(
                NO_PLASMA_COIL_DISTANCE_OBJECTIVE_ABLATION_POLICY
            ),
        )
    ]
    no_coil_distance_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "vacuum_joint",
            include_hardware_keepout=True,
            objective_ablation_policy=NO_COIL_SET_DISTANCE_OBJECTIVE_ABLATION_POLICY,
        )
    ]
    no_coil_geometry_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "vacuum_joint",
            include_hardware_keepout=True,
            objective_ablation_policy=NO_COIL_GEOMETRY_OBJECTIVE_ABLATION_POLICY,
        )
    ]
    no_linking_names = [
        entry.name
        for entry in build_desc_objective_stack_plan(
            "vacuum_joint",
            include_hardware_keepout=True,
            objective_ablation_policy=NO_LINKING_CURRENT_OBJECTIVE_ABLATION_POLICY,
        )
    ]

    assert "CoilCurvature" not in no_curvature_names
    assert COIL_SET_MIN_DISTANCE_OBJECTIVE in no_curvature_names
    assert "PlasmaCoilSetMinDistance" in no_curvature_names
    assert "LinkingCurrentConsistency" in no_curvature_names

    assert "PlasmaCoilSetMinDistance" not in no_plasma_distance_names
    assert COIL_SET_MIN_DISTANCE_OBJECTIVE in no_plasma_distance_names

    assert COIL_SET_MIN_DISTANCE_OBJECTIVE not in no_coil_distance_names
    assert "PlasmaCoilSetMinDistance" in no_coil_distance_names

    assert "LinkingCurrentConsistency" in no_coil_geometry_names
    assert "CoilLength" not in no_coil_geometry_names
    assert "CoilCurvature" not in no_coil_geometry_names
    assert COIL_SET_MIN_DISTANCE_OBJECTIVE not in no_coil_geometry_names
    assert "PlasmaCoilSetMinDistance" not in no_coil_geometry_names
    assert HARDWARE_SDF_KEEPOUT_OBJECTIVE not in no_coil_geometry_names

    assert "LinkingCurrentConsistency" not in no_linking_names
    assert COIL_SET_MIN_DISTANCE_OBJECTIVE in no_linking_names


def test_desc_objective_runtime_assembles_fixed_polish_terms(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    equilibrium = object()
    coilset = object()

    assembly = assemble_desc_objective_stack_runtime(
        mode="fixed_equilibrium_polish",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
    )

    objective_function = assembly.objective_function
    objective_terms = tuple(objective_function.objectives)
    objective_names = tuple(type(term).__name__ for term in objective_terms)
    constraint_terms = tuple(assembly.constraints)
    assert tuple(type(term).__name__ for term in constraint_terms) == (
        FIX_COIL_CURRENT_CONSTRAINT,
    )
    assert objective_names == (
        "QuadraticFlux",
        "LinkingCurrentConsistency",
        "CoilLength",
        "CoilCurvature",
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
    )
    quadratic_flux = objective_terms[0]
    assert quadratic_flux.args == (coilset,)
    assert quadratic_flux.kwargs["vacuum"] is True
    assert quadratic_flux.kwargs["field_grid"].kwargs == {"N": 7}
    assert objective_terms[1].kwargs["eq_fixed"] is True
    assert objective_terms[1].kwargs["linking_grid"].kwargs == {"N": 7}
    assert objective_terms[2].kwargs["bounds"] == (0.0, COIL_LENGTH_HARD_LIMIT_M)
    assert objective_terms[3].kwargs["target"] == 0.0
    assert "bounds" not in objective_terms[3].kwargs
    assert objective_terms[3].kwargs["normalize_target"] is True
    assert objective_terms[4].kwargs["bounds"] == (
        COIL_COIL_MIN_DIST_M,
        np.inf,
    )
    assert objective_terms[5].kwargs["bounds"] == (
        COIL_PLASMA_MIN_DIST_M,
        np.inf,
    )
    assert objective_terms[5].kwargs["eq_fixed"] is True
    assert constraint_terms[0].args == (coilset,)
    assert constraint_terms[0].kwargs["indices"] is True
    report = assembly.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["desc_version"] == "fixture-desc"
    assert report["objective_function_type"] == "desc.objectives.ObjectiveFunction"
    assert report["constraint_names"] == [FIX_COIL_CURRENT_CONSTRAINT]
    assert report["linking_current_grid_n"] == 7
    assert report["hardware_thresholds_m"]["coil_coil_min_dist_m"] == (
        COIL_COIL_MIN_DIST_M
    )

    capped_assembly = assemble_desc_objective_stack_runtime(
        mode="fixed_equilibrium_polish",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=LINKING_CURRENT_GRID_N_CAP + 13,
    )
    capped_linking_term = capped_assembly.objective_function.objectives[1]
    assert capped_linking_term.kwargs["linking_grid"].kwargs == {
        "N": LINKING_CURRENT_GRID_N_CAP,
    }
    assert (
        capped_assembly.report.to_json_dict()["linking_current_grid_n"]
        == LINKING_CURRENT_GRID_N_CAP
    )


def test_desc_objective_runtime_assembles_joint_volume_anchor(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    equilibrium = object()
    coilset = object()

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
    )

    objective_terms = tuple(assembly.objective_function.objectives)
    constraint_terms = tuple(assembly.constraints)
    objective_names = tuple(type(term).__name__ for term in objective_terms)
    constraint_names = tuple(type(term).__name__ for term in constraint_terms)
    assert objective_names == (
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    )
    assert constraint_names == (
        "Volume",
        "ForceBalance",
        FIX_COIL_CURRENT_CONSTRAINT,
    )
    volume_term = constraint_terms[0]
    assert volume_term.args == (equilibrium,)
    assert volume_term.kwargs["target"] == -0.0496
    assert volume_term.kwargs["normalize"] is True
    assert volume_term.kwargs["jac_chunk_size"] == 5
    force_balance_term = constraint_terms[1]
    assert force_balance_term.args == (equilibrium,)
    assert force_balance_term.kwargs["target"] == 0
    assert force_balance_term.kwargs["jac_chunk_size"] == 5
    fixed_current_term = constraint_terms[2]
    assert fixed_current_term.args == (coilset,)
    assert fixed_current_term.kwargs["indices"] is True
    report = assembly.report.to_json_dict()
    assert VOLUME_OBJECTIVE not in report["objective_names"]
    assert report["constraint_names"] == [
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert report["volume_target_m3"] == -0.0496
    assert (
        report["joint_constraint_policy"]
        == HARD_VOLUME_AND_FORCE_BALANCE_POLICY
    )
    assert report["objective_ablation_policy"] == FULL_DESC_OBJECTIVE_ABLATION_POLICY
    assert report["weights"][VOLUME_OBJECTIVE] == 1.0
    assert report["weights"][FORCE_BALANCE_CONSTRAINT] == 1.0


def test_desc_objective_runtime_assembles_hard_hardware_policy(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    equilibrium = object()
    coilset = object()

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
        joint_constraint_policy=HARD_HARDWARE_AND_FORCE_BALANCE_POLICY,
    )

    objective_terms = tuple(assembly.objective_function.objectives)
    constraint_terms = tuple(assembly.constraints)
    assert tuple(type(term).__name__ for term in objective_terms) == (
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
    )
    assert tuple(type(term).__name__ for term in constraint_terms) == (
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        COIL_SET_MIN_DISTANCE_OBJECTIVE,
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
        FIX_COIL_CURRENT_CONSTRAINT,
    )
    assert constraint_terms[2].kwargs["bounds"] == (
        COIL_COIL_MIN_DIST_M,
        FINITE_NONBINDING_HARDWARE_UPPER_BOUND,
    )
    assert constraint_terms[3].kwargs["bounds"] == (
        COIL_PLASMA_MIN_DIST_M,
        FINITE_NONBINDING_HARDWARE_UPPER_BOUND,
    )
    assert constraint_terms[3].kwargs["eq_fixed"] is False
    assert constraint_terms[4].kwargs["bounds"] == (
        0.0,
        COIL_LENGTH_HARD_LIMIT_M,
    )
    assert constraint_terms[5].kwargs["bounds"] == (
        0.0,
        MAX_CURVATURE_INV_M,
    )
    assert "target" not in constraint_terms[5].kwargs
    report = assembly.report.to_json_dict()
    assert report["objective_names"] == [
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
    ]
    assert report["constraint_names"] == [
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        COIL_SET_MIN_DISTANCE_OBJECTIVE,
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert (
        report["joint_constraint_policy"]
        == HARD_HARDWARE_AND_FORCE_BALANCE_POLICY
    )


def test_desc_objective_runtime_assembles_proximal_force_balance_policy(
    tmp_path,
):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    equilibrium = object()
    coilset = object()

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
        joint_constraint_policy=PROXIMAL_FORCE_BALANCE_POLICY,
    )

    objective_terms = tuple(assembly.objective_function.objectives)
    constraint_terms = tuple(assembly.constraints)
    objective_names = tuple(type(term).__name__ for term in objective_terms)
    constraint_names = tuple(type(term).__name__ for term in constraint_terms)
    assert objective_names == (
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
        "Volume",
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    )
    assert constraint_names == (
        "ForceBalance",
        FIX_COIL_CURRENT_CONSTRAINT,
    )
    volume_term = objective_terms[2]
    assert volume_term.args == (equilibrium,)
    assert volume_term.kwargs["target"] == -0.0496
    assert volume_term.kwargs["normalize"] is True
    assert volume_term.kwargs["jac_chunk_size"] == 5
    force_balance_term = constraint_terms[0]
    assert force_balance_term.args == (equilibrium,)
    assert force_balance_term.kwargs["target"] == 0
    assert force_balance_term.kwargs["jac_chunk_size"] == 5
    fixed_current_term = constraint_terms[1]
    assert fixed_current_term.args == (coilset,)
    assert fixed_current_term.kwargs["indices"] is True
    report = assembly.report.to_json_dict()
    assert report["objective_names"] == [
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
        VOLUME_OBJECTIVE,
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    ]
    assert report["constraint_names"] == [
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert report["volume_target_m3"] == -0.0496
    assert report["joint_constraint_policy"] == PROXIMAL_FORCE_BALANCE_POLICY
    assert report["objective_ablation_policy"] == FULL_DESC_OBJECTIVE_ABLATION_POLICY
    assert report["weights"][VOLUME_OBJECTIVE] == 1.0
    assert report["weights"][FORCE_BALANCE_CONSTRAINT] == 1.0


def test_desc_objective_runtime_assembles_hard_linking_current_policy(
    tmp_path,
):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    equilibrium = object()
    coilset = object()

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
        joint_constraint_policy=HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY,
    )

    objective_terms = tuple(assembly.objective_function.objectives)
    constraint_terms = tuple(assembly.constraints)
    assert tuple(type(term).__name__ for term in objective_terms) == (
        "VacuumBoundaryError",
        VOLUME_OBJECTIVE,
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    )
    assert tuple(type(term).__name__ for term in constraint_terms) == (
        "LinkingCurrentConsistency",
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    )
    linking_current_term = constraint_terms[0]
    assert linking_current_term.args == (coilset, equilibrium)
    assert linking_current_term.kwargs["eq_fixed"] is False
    assert linking_current_term.kwargs["normalize"] is True
    assert linking_current_term.kwargs["jac_chunk_size"] == 5
    volume_term = objective_terms[1]
    assert volume_term.args == (equilibrium,)
    assert volume_term.kwargs["target"] == -0.0496
    report = assembly.report.to_json_dict()
    assert report["objective_names"] == [
        "VacuumBoundaryError",
        VOLUME_OBJECTIVE,
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    ]
    assert report["constraint_names"] == [
        "LinkingCurrentConsistency",
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert (
        report["joint_constraint_policy"]
        == HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY
    )


def test_desc_objective_runtime_assembles_boundary_fidelity_guard(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)

    class BoundaryBasis:
        def __init__(self, modes: list[list[int]]) -> None:
            self.modes = np.asarray(modes, dtype=int)

    class BoundarySurface:
        R_basis = BoundaryBasis(
            [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 2], [0, 1, 1]]
        )
        Z_basis = BoundaryBasis(
            [[0, 1, 0], [0, 2, 0], [0, 0, -2], [0, 1, -1]]
        )

    class BoundaryEquilibrium:
        surface = BoundarySurface()

    equilibrium = BoundaryEquilibrium()
    coilset = object()

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
        boundary_fidelity_policy=BOUNDARY_FIDELITY_FIX_HIGH_MODES,
        boundary_fidelity_free_mode_sum=1,
    )

    constraint_terms = tuple(assembly.constraints)
    constraint_names = tuple(type(term).__name__ for term in constraint_terms)
    assert constraint_names == (
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        FIX_BOUNDARY_R_CONSTRAINT,
        FIX_BOUNDARY_Z_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    )
    boundary_R_term = constraint_terms[2]
    boundary_Z_term = constraint_terms[3]
    assert boundary_R_term.args == (equilibrium,)
    assert boundary_Z_term.args == (equilibrium,)
    assert boundary_R_term.kwargs["modes"].tolist() == [
        [0, 2, 0],
        [0, 0, 2],
        [0, 1, 1],
    ]
    assert boundary_Z_term.kwargs["modes"].tolist() == [
        [0, 2, 0],
        [0, 0, -2],
        [0, 1, -1],
    ]

    report = assembly.report.to_json_dict()
    assert report["constraint_names"] == [
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        FIX_BOUNDARY_R_CONSTRAINT,
        FIX_BOUNDARY_Z_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert report["boundary_fidelity"] == {
        "policy": BOUNDARY_FIDELITY_FIX_HIGH_MODES,
        "free_mode_sum": 1,
        "free_selector": "|m| + |n| <= free_mode_sum",
        "fixed_selector": "|m| + |n| > free_mode_sum",
        "R_mode_count": 5,
        "Z_mode_count": 4,
        "fixed_R_mode_count": 3,
        "fixed_Z_mode_count": 3,
        "free_R_mode_count": 2,
        "free_Z_mode_count": 1,
        "fixed_R_modes": [[0, 2, 0], [0, 0, 2], [0, 1, 1]],
        "fixed_Z_modes": [[0, 2, 0], [0, 0, -2], [0, 1, -1]],
    }

    default_assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
        boundary_fidelity_policy=BOUNDARY_FIDELITY_OFF,
    )
    assert default_assembly.report.to_json_dict()["boundary_fidelity"] is None


def test_desc_objective_runtime_assembles_physics_only_ablation(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    equilibrium = object()
    coilset = object()

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=True,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
        joint_constraint_policy=PROXIMAL_FORCE_BALANCE_POLICY,
        objective_ablation_policy=PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY,
    )

    objective_terms = tuple(assembly.objective_function.objectives)
    constraint_terms = tuple(assembly.constraints)
    assert tuple(type(term).__name__ for term in objective_terms) == (
        "VacuumBoundaryError",
        "Volume",
    )
    assert tuple(type(term).__name__ for term in constraint_terms) == (
        "ForceBalance",
        FIX_COIL_CURRENT_CONSTRAINT,
    )
    report = assembly.report.to_json_dict()
    assert report["objective_names"] == [
        "VacuumBoundaryError",
        VOLUME_OBJECTIVE,
    ]
    assert report["constraint_names"] == [
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert report["joint_constraint_policy"] == PROXIMAL_FORCE_BALANCE_POLICY
    assert (
        report["objective_ablation_policy"]
        == PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY
    )


def test_desc_objective_runtime_scales_scoped_joint_linking_current_by_full_current(
    tmp_path,
):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    equilibrium = object()

    class ScopedOptimizationScope:
        optimized_group_names = ("banana",)
        fixed_group_names = ("tf",)
        optimized_unique_coil_indices = (2,)
        fixed_unique_coil_indices = (0, 1)
        unique_coil_count = 3

    class ScopedCoilset:
        current = [10.0, -20.0, 30.0]
        desc_joint_optimization_scope = ScopedOptimizationScope()

        def __init__(self):
            self.all_currents_calls = []

        def _all_currents(self, currents=None):
            if currents is None:
                self.all_currents_calls.append(None)
                return np.array([10.0, -20.0, 30.0])
            scoped_currents = np.asarray(currents, dtype=float).reshape(-1)
            self.all_currents_calls.append(tuple(scoped_currents))
            assert scoped_currents.shape == (1,)
            merged_currents = np.array([10.0, -20.0, 30.0])
            merged_currents[2] = scoped_currents[0]
            return merged_currents

    coilset = ScopedCoilset()

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=equilibrium,
        coilset=coilset,
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=-0.0496,
    )

    linking_term = assembly.objective_function.objectives[1]
    assert type(linking_term).__name__ == "LinkingCurrentConsistency"
    assert linking_term.kwargs["normalize"] is False
    assert linking_term.kwargs["weight"] == pytest.approx(1.0 / 60.0)
    report = assembly.report.to_json_dict()
    normalization = report["linking_current_normalization"]
    assert normalization["source"] == "scoped_full_current_abs_sum"
    assert normalization["full_current_abs_sum_A"] == pytest.approx(60.0)
    assert normalization[
        "full_current_abs_sum_from_scoped_current_merge_A"
    ] == pytest.approx(60.0)
    assert normalization["optimized_unique_current_abs_sum_A"] == pytest.approx(30.0)
    assert normalization["fixed_unique_current_abs_sum_A"] == pytest.approx(30.0)
    assert normalization["effective_weight"] == pytest.approx(1.0 / 60.0)
    assert normalization["normalize"] is False
    geometry_weighting = report["coil_geometry_weighting"]
    assert geometry_weighting["source"] == "scoped_optimized_coil_groups"
    assert geometry_weighting["unit_weight_vector_by_unique_coil"] == [
        0.0,
        0.0,
        1.0,
    ]
    assert geometry_weighting["optimized_unique_coil_indices"] == [2]
    assert geometry_weighting["fixed_unique_coil_indices"] == [0, 1]
    assert geometry_weighting["weighted_objective_names"] == [
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    ]
    assert assembly.objective_function.objectives[2].kwargs["weight"] == [
        0.0,
        0.0,
        1.0,
    ]
    assert assembly.objective_function.objectives[3].kwargs["weight"] == [
        0.0,
        0.0,
        1.0,
    ]
    assert assembly.objective_function.objectives[4].kwargs["weight"] == [
        0.0,
        0.0,
        1.0,
    ]
    assert assembly.objective_function.objectives[5].kwargs["weight"] == [
        0.0,
        0.0,
        1.0,
    ]
    assert coilset.all_currents_calls == [None, (30.0,)]


def test_desc_objective_runtime_requires_patched_linking_grid_api(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path, supports_linking_grid=False)

    with pytest.raises(DescObjectiveRuntimeAssemblyError) as exc_info:
        assemble_desc_objective_stack_runtime(
            mode="fixed_equilibrium_polish",
            equilibrium=object(),
            coilset=object(),
            include_hardware_keepout=False,
            desc_source_root=fake_desc_root,
            grid_n=7,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert "LinkingCurrentConsistency" in report["reason"]
    assert "linking_grid" in report["reason"]


def test_desc_source_root_overrides_cached_desc_modules(tmp_path):
    patched_desc_root = _fake_desc_source_root(
        tmp_path / "patched",
        supports_linking_grid=True,
    )
    stale_desc_root = _fake_desc_source_root(
        tmp_path / "stale",
        supports_linking_grid=False,
    )
    stale_desc_root_str = str(stale_desc_root)
    sys.path.insert(0, stale_desc_root_str)
    try:
        from desc.objectives import LinkingCurrentConsistency

        assert "linking_grid" not in inspect.signature(
            LinkingCurrentConsistency
        ).parameters
    finally:
        sys.path.remove(stale_desc_root_str)

    assembly = assemble_desc_objective_stack_runtime(
        mode="fixed_equilibrium_polish",
        equilibrium=object(),
        coilset=object(),
        include_hardware_keepout=False,
        desc_source_root=patched_desc_root,
        grid_n=7,
    )

    report = assembly.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["desc_source_root"] == str(patched_desc_root)
    assert report["objective_use_jit"] is False
    assert report["objective_deriv_mode"] == "blocked"
    assert assembly.objective_function.constructor_use_jit is False
    assert assembly.objective_function.constructor_deriv_mode == "blocked"
    assert "linking_grid" in inspect.signature(
        type(assembly.objective_function.objectives[1])
    ).parameters


def test_desc_objective_runtime_evaluates_value_and_jacobian(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    assembly = assemble_desc_objective_stack_runtime(
        mode="fixed_equilibrium_polish",
        equilibrium=object(),
        coilset=object(),
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
    )

    report = evaluate_desc_objective_stack_runtime(
        assembly.objective_function,
        use_jit=False,
        compute_jacobian=True,
    ).to_json_dict()

    assert report["status"] == "passed"
    assert report["use_jit"] is False
    assert report["evaluation_mode"] == "sequential_terms"
    assert report["dim_x"] == 1
    assert report["dim_f"] == 6
    assert report["scaled_error_all_finite"] is True
    assert report["jacobian_shape"] == [6, 1]
    assert report["jacobian_all_finite"] is True
    assert report["gradient_all_finite"] is None
    assert report["gradient_seconds"] is None
    assert [term["name"] for term in report["objective_term_reports"]] == [
        "QuadraticFlux",
        "LinkingCurrentConsistency",
        "CoilLength",
        "CoilCurvature",
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
    ]


def test_desc_objective_runtime_evaluates_term_gradients(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    assembly = assemble_desc_objective_stack_runtime(
        mode="fixed_equilibrium_polish",
        equilibrium=object(),
        coilset=object(),
        include_hardware_keepout=False,
        desc_source_root=fake_desc_root,
        grid_n=7,
    )

    report = evaluate_desc_objective_stack_runtime(
        assembly.objective_function,
        use_jit=False,
        compute_gradient=True,
    ).to_json_dict()

    assert report["status"] == "passed"
    assert report["evaluation_mode"] == "sequential_terms"
    assert report["gradient_all_finite"] is True
    assert report["gradient_seconds"] >= 0.0
    assert report["gradient_progress_path"] is None
    assert report["jacobian_shape"] is None
    assert report["jacobian_all_finite"] is None
    for term_report in report["objective_term_reports"]:
        assert term_report["gradient_all_finite"] is True
        assert term_report["gradient_l2"] >= 0.0
        assert term_report["gradient_seconds"] >= 0.0
        assert term_report["gradient_size"] == 1
        assert term_report["gradient_block_shapes"] == [[]]


def test_desc_objective_runtime_rejects_nonfinite_value_or_jacobian():
    class NonFiniteObjectiveTerm:
        def __init__(
            self,
            *,
            scaled_error: list[float],
        ) -> None:
            self.scaled_error = scaled_error
            self.things = (object(),)
            self.built = False

        def build(self, *, use_jit: bool, verbose: int) -> None:
            self.use_jit = use_jit
            self.verbose = verbose
            self.built = True

        def xs(self, *things):
            return tuple(float(index + 1) for index in range(len(things)))

        def compute_scaled_error(self, *args):
            return self.scaled_error

    class NonFiniteObjectiveFunction:
        def __init__(
            self,
            *,
            scaled_error: list[float],
            jacobian: list[list[float]],
        ) -> None:
            self.objectives = (NonFiniteObjectiveTerm(scaled_error=scaled_error),)
            self.things = (object(),)
            self.jacobian = jacobian
            self.dim_x = 2

        def build(self, *, use_jit: bool, verbose: int) -> None:
            self.use_jit = use_jit
            self.verbose = verbose
            for objective in self.objectives:
                objective.build(use_jit=use_jit, verbose=verbose)

        def x(self, *things):
            return [1.0, 2.0]

        def compute_scaled_error(self, x):
            raise AssertionError("combined value path must not run")

        def jac_scaled_error(self, x):
            return self.jacobian

    with pytest.raises(DescObjectiveRuntimeEvaluationError) as bad_value:
        evaluate_desc_objective_stack_runtime(
            NonFiniteObjectiveFunction(
                scaled_error=[float("nan")],
                jacobian=[[1.0, 0.0]],
            )
        )
    value_report = bad_value.value.report.to_json_dict()
    assert value_report["status"] == "failed"
    assert value_report["scaled_error_all_finite"] is False

    with pytest.raises(DescObjectiveRuntimeEvaluationError) as bad_jacobian:
        evaluate_desc_objective_stack_runtime(
            NonFiniteObjectiveFunction(
                scaled_error=[0.0],
                jacobian=[[float("inf"), 0.0]],
            ),
            compute_jacobian=True,
        )
    jacobian_report = bad_jacobian.value.report.to_json_dict()
    assert jacobian_report["status"] == "failed"
    assert jacobian_report["scaled_error_all_finite"] is True
    assert jacobian_report["jacobian_all_finite"] is False


def test_desc_objective_runtime_requires_sdf_manifest_for_hardware_keepout(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)

    with pytest.raises(DescObjectiveRuntimeAssemblyError) as exc_info:
        assemble_desc_objective_stack_runtime(
            mode="vacuum_joint",
            equilibrium=object(),
            coilset=object(),
            include_hardware_keepout=True,
            desc_source_root=fake_desc_root,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert HARDWARE_SDF_KEEPOUT_OBJECTIVE in report["objective_names"]
    assert "requires hardware_sdf path" in report["reason"]


def test_desc_objective_runtime_assembles_hardware_sdf_keepout(
    tmp_path,
    monkeypatch,
):
    class FakePatch:
        def __init__(self) -> None:
            self.grid = np.ones((2, 2, 2), dtype=float) * 0.08
            self.origin_m = np.asarray([0.25, 0.0, 0.0], dtype=float)
            self.spacing_m = 0.025

    class FakeGroup:
        def __init__(self) -> None:
            self.label = "typekk"
            self.grid = np.ones((3, 3, 3), dtype=float) * 0.05
            self.origin_m = np.asarray([0.0, 0.0, 0.0], dtype=float)
            self.spacing_m = 0.05
            self.patches = (FakePatch(),)

    class FakeHardwareSdfData:
        def __init__(self, manifest_path: Path) -> None:
            self.manifest_path = str(manifest_path)
            self.data_path = str(manifest_path.with_suffix(".npz"))
            self.manifest_sha256 = "manifest-sha"
            self.data_sha256 = "data-sha"
            self.groups = (FakeGroup(),)
            self.safety_margin_m = 0.003
            self.error_budget_m = {"e_total_m": 0.002}
            self.documented_gate_only = {"vessel": "final CAD oracle"}
            self.covered_by_other_in_loop = {"coil_coil": "CoilSetMinDistance"}

        @property
        def group_labels(self) -> tuple[str, ...]:
            return tuple(group.label for group in self.groups)

        @property
        def patch_count(self) -> int:
            return sum(len(group.patches) for group in self.groups)

        @property
        def effective_margin_m(self) -> float:
            return 0.005

    fake_desc_root = _fake_desc_source_root(tmp_path)
    manifest_path = tmp_path / "hardware_sdf.json"
    glb_path = tmp_path / "hardware.glb"
    load_calls = []

    def fake_load_hardware_sdf(path, glb_path=None):
        load_calls.append((Path(path), None if glb_path is None else Path(glb_path)))
        return FakeHardwareSdfData(Path(path))

    monkeypatch.setattr(
        "banana_opt.desc_bridge.objective_factory.load_hardware_sdf",
        fake_load_hardware_sdf,
    )

    assembly = assemble_desc_objective_stack_runtime(
        mode="vacuum_joint",
        equilibrium=object(),
        coilset=object(),
        include_hardware_keepout=True,
        hardware_sdf_manifest_path=manifest_path,
        hardware_glb_path=glb_path,
        desc_source_root=fake_desc_root,
        grid_n=7,
        volume_target_m3=0.0496,
    )

    assert load_calls == [(manifest_path, glb_path)]
    objective_terms = tuple(assembly.objective_function.objectives)
    objective_names = tuple(type(term).__name__ for term in objective_terms)
    assert HARDWARE_SDF_KEEPOUT_OBJECTIVE in assembly.report.objective_names
    assert "CoilSetSDFDistance" in objective_names
    keepout_term = objective_terms[objective_names.index("CoilSetSDFDistance")]
    expected_centerline_clearance = 0.005 + TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M
    assert keepout_term.kwargs["minimum_clearance"] == expected_centerline_clearance
    assert keepout_term.kwargs["outside_value"] == expected_centerline_clearance + 1.0
    assert keepout_term.kwargs["bounds"] == (
        expected_centerline_clearance,
        FINITE_NONBINDING_HARDWARE_UPPER_BOUND,
    )
    assert keepout_term.kwargs["grid"].kwargs == {"N": 7}
    assert keepout_term.kwargs["normalize_target"] is False
    assert keepout_term.kwargs["sdf_gridsets"][0][0][0].shape == (3, 3, 3)
    assert keepout_term.kwargs["sdf_gridsets"][0][1][0].shape == (2, 2, 2)
    report = assembly.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["hardware_keepout"] == {
        "source": "hardware_sdf_manifest",
        "manifest_path": str(manifest_path),
        "data_path": str(manifest_path.with_suffix(".npz")),
        "manifest_sha256": "manifest-sha",
        "data_sha256": "data-sha",
        "group_labels": ["typekk"],
        "patch_count": 1,
        "minimum_clearance_m": expected_centerline_clearance,
        "outside_value_m": expected_centerline_clearance + 1.0,
        "sdf_effective_margin_m": 0.005,
        "type_kk_centerline_padding_m": TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M,
        "sampling_policy": (
            "DESC CoilSetSDFDistance samples coil centerlines; the banana "
            "bridge pads by Type-KK outer-channel corner reach. Final "
            "promotion remains bound to the SIMSOPT/CAD swept-solid oracle."
        ),
        "safety_margin_m": 0.003,
        "effective_margin_m": 0.005,
        "error_budget_m": {"e_total_m": 0.002},
        "documented_gate_only_groups": ["vessel"],
        "covered_by_other_in_loop_groups": ["coil_coil"],
    }


def test_desc_fixed_polish_runtime_solve_saves_desc_coil_artifact(tmp_path):
    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc coilset\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"
    optimizer_controls = build_desc_optimizer_controls(
        ftol=1e-4,
        xtol=2e-5,
        gtol=3e-6,
        ctol=4e-3,
        max_nfev=17,
        max_dx=0.25,
        initial_trust_radius=0.02,
        max_trust_radius=0.05,
        min_trust_radius=0.0,
    )

    result = run_desc_fixed_equilibrium_polish_runtime(
        coilset=SaveableCoilSet(),
        objective_function=object(),
        output_root=output_root,
        desc_source_root=fake_desc_root,
        optimizer_method="fixture-success",
        maxiter=5,
        verbose=0,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=True,
    )

    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["desc_version"] == "fixture-desc"
    assert report["allow_high_memory_optimizer"] is True
    assert report["optimizer_controls"] == {
        "tolerances": {
            "ftol": 1e-4,
            "xtol": 2e-5,
            "gtol": 3e-6,
            "ctol": 4e-3,
        },
        "options": {
            "max_nfev": 17,
            "max_dx": 0.25,
            "initial_trust_radius": 0.02,
            "max_trust_radius": 0.05,
            "min_trust_radius": 0.0,
        },
    }
    assert report["optimizer_success"] is True
    assert report["optimizer_nit"] == 3
    assert result.optimized_coilset.optimizer_tolerances == {
        "ftol": 1e-4,
        "xtol": 2e-5,
        "gtol": 3e-6,
        "ctol": 4e-3,
    }
    assert result.optimized_coilset.optimizer_options == {
        "max_nfev": 17,
        "max_dx": 0.25,
        "initial_trust_radius": 0.02,
        "max_trust_radius": 0.05,
        "min_trust_radius": 0.0,
    }
    assert report["optimized_coilset_path"] == str(output_root / "desc_coils.h5")
    assert (output_root / "desc_coils.h5").read_text(encoding="utf-8") == (
        "optimized desc coilset\n"
    )


def test_desc_optimizer_controls_reject_invalid_values():
    with pytest.raises(ValueError, match="ctol"):
        build_desc_optimizer_controls(ctol=0.0)
    with pytest.raises(ValueError, match="max_nfev"):
        build_desc_optimizer_controls(max_nfev=0)
    with pytest.raises(ValueError, match="max_dx"):
        build_desc_optimizer_controls(max_dx=0.0)
    with pytest.raises(ValueError, match="initial_trust_radius"):
        build_desc_optimizer_controls(initial_trust_radius=0.0)
    with pytest.raises(ValueError, match="max_trust_radius"):
        build_desc_optimizer_controls(max_trust_radius=0.0)
    with pytest.raises(ValueError, match="min_trust_radius"):
        build_desc_optimizer_controls(min_trust_radius=-1.0)
    with pytest.raises(ValueError, match="proximal_perturb_order"):
        build_desc_optimizer_controls(proximal_perturb_order=0)
    with pytest.raises(ValueError, match="proximal_solve_maxiter"):
        build_desc_optimizer_controls(proximal_solve_maxiter=0)
    with pytest.raises(ValueError, match="proximal_solve_during_build"):
        build_desc_optimizer_controls(proximal_solve_during_build=1)


def test_desc_fixed_polish_runtime_solve_fails_closed_on_optimizer_failure(tmp_path):
    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    with pytest.raises(DescFixedPolishRuntimeSolveError) as exc_info:
        run_desc_fixed_equilibrium_polish_runtime(
            coilset=SaveableCoilSet(),
            objective_function=object(),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="fixture-fail",
            maxiter=5,
            verbose=0,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["allow_high_memory_optimizer"] is True
    assert report["optimizer_success"] is False
    assert report["optimizer_message"] == "fixture requested failure"
    assert report["optimized_coilset_path"] is None
    assert report["failed_optimizer_coilset_checkpoint_path"] == str(
        output_root / "desc_failed_optimizer_coils.h5"
    )
    assert not (output_root / "desc_coils.h5").exists()
    assert (output_root / "desc_failed_optimizer_coils.h5").is_file()


def test_desc_fixed_polish_runtime_solve_fails_closed_on_partial_save(tmp_path):
    class PartialSaveCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("partial desc coilset\n", encoding="utf-8")
            raise RuntimeError("fixture partial coilset save")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    with pytest.raises(DescFixedPolishRuntimeSolveError) as exc_info:
        run_desc_fixed_equilibrium_polish_runtime(
            coilset=PartialSaveCoilSet(),
            objective_function=object(),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="fixture-success",
            maxiter=5,
            verbose=0,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["optimized_coilset_path"] is None
    assert "fixture partial coilset save" in report["reason"]
    assert not (output_root / "desc_coils.h5").exists()
    assert list(output_root.glob(".desc_coils.h5.*.tmp")) == []


def test_desc_joint_runtime_solve_saves_equilibrium_and_coils(tmp_path):
    class FixtureConstraint:
        pass

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc equilibrium\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc coilset\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"
    optimizer_controls = build_desc_optimizer_controls(
        ftol=1e-5,
        xtol=2e-6,
        gtol=3e-7,
        ctol=4e-4,
        max_nfev=23,
        max_dx=0.5,
        initial_trust_radius=0.03,
        max_trust_radius=0.07,
        min_trust_radius=1e-14,
    )

    result = run_desc_joint_optimization_runtime(
        equilibrium=SaveableEquilibrium(),
        coilset=SaveableCoilSet(),
        objective_function=object(),
        constraints=(FixtureConstraint(),),
        output_root=output_root,
        desc_source_root=fake_desc_root,
        optimizer_method="fixture-success",
        maxiter=5,
        verbose=0,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=True,
    )

    report = result.report.to_json_dict()
    assert report["schema_version"] == "desc_joint_runtime_solve_report_v1"
    assert report["status"] == "passed"
    assert report["desc_version"] == "fixture-desc"
    assert report["allow_high_memory_optimizer"] is True
    assert len(report["constraint_types"]) == 1
    assert report["constraint_types"][0].endswith(
        "test_desc_joint_runtime_solve_saves_equilibrium_and_coils."
        "<locals>.FixtureConstraint"
    )
    assert report["optimizer_success"] is True
    assert report["optimizer_nit"] == 3
    assert report["optimizer_controls"] == {
        "tolerances": {
            "ftol": 1e-5,
            "xtol": 2e-6,
            "gtol": 3e-7,
            "ctol": 4e-4,
        },
        "options": {
            "max_nfev": 23,
            "max_dx": 0.5,
            "initial_trust_radius": 0.03,
            "max_trust_radius": 0.07,
            "min_trust_radius": 1e-14,
        },
    }
    assert result.optimized_equilibrium.optimizer_tolerances == {
        "ftol": 1e-5,
        "xtol": 2e-6,
        "gtol": 3e-7,
        "ctol": 4e-4,
    }
    assert result.optimized_coilset.optimizer_options == {
        "max_nfev": 23,
        "max_dx": 0.5,
        "initial_trust_radius": 0.03,
        "max_trust_radius": 0.07,
        "min_trust_radius": 1e-14,
    }
    assert result.optimized_equilibrium.optimizer_constraint_count == 1
    assert result.optimized_equilibrium.optimizer_constraint_types == (
        "FixtureConstraint",
    )
    assert result.optimized_coilset.optimizer_constraint_count == 1
    assert result.optimized_coilset.optimizer_constraint_types == (
        "FixtureConstraint",
    )
    assert report["optimized_equilibrium_path"] == str(
        output_root / "desc_equilibrium.h5"
    )
    assert report["optimized_coilset_path"] == str(output_root / "desc_coils.h5")
    assert (output_root / "desc_equilibrium.h5").read_text(
        encoding="utf-8"
    ) == "optimized desc equilibrium\n"
    assert (output_root / "desc_coils.h5").read_text(encoding="utf-8") == (
        "optimized desc coilset\n"
    )


def test_desc_joint_runtime_solve_uses_desc_objective_thing_order(tmp_path):
    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc equilibrium\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc coilset\n", encoding="utf-8")

    equilibrium = SaveableEquilibrium()
    coilset = SaveableCoilSet()

    class CoilFirstObjectiveFunction:
        things = (coilset, equilibrium)
        expected_optimizer_thing_order = (
            "SaveableCoilSet",
            "SaveableEquilibrium",
        )

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    result = run_desc_joint_optimization_runtime(
        equilibrium=equilibrium,
        coilset=coilset,
        objective_function=CoilFirstObjectiveFunction(),
        output_root=output_root,
        desc_source_root=fake_desc_root,
        optimizer_method="fixture-success",
        maxiter=5,
        verbose=0,
        allow_high_memory_optimizer=True,
    )

    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert type(result.optimized_equilibrium).__name__ == "SaveableEquilibrium"
    assert type(result.optimized_coilset).__name__ == "SaveableCoilSet"
    assert (output_root / "desc_equilibrium.h5").read_text(
        encoding="utf-8"
    ) == "optimized desc equilibrium\n"
    assert (output_root / "desc_coils.h5").read_text(encoding="utf-8") == (
        "optimized desc coilset\n"
    )


def test_desc_joint_runtime_solve_writes_constraint_feasibility_report(tmp_path):
    class FixtureThing:
        dim_x = 2

    fixture_thing = FixtureThing()

    class DiagnosticObjectiveFunction:
        things = (fixture_thing,)

        def x(self, *things):
            return [0.0, 1.0]

    class DiagnosticConstraint:
        things = (fixture_thing,)
        bounds = (0.0, 1.0)
        target = None
        dim_f = 3
        normalization = 1.0
        _normalize_target = True
        built = False

        def build(self, *, use_jit: bool = False, verbose: int = 0) -> None:
            self.use_jit = use_jit
            self.verbose = verbose
            self.built = True

        def xs(self, *things):
            self.xs_things = tuple(things)
            return tuple()

        def _scale(self, values):
            return values

        def compute_scaled(self, *args):
            return [0.5, 1.25, -0.1]

        def compute_scaled_error(self, *args):
            return [0.0, 0.25, -0.1]

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc equilibrium\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc coilset\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    result = run_desc_joint_optimization_runtime(
        equilibrium=SaveableEquilibrium(),
        coilset=SaveableCoilSet(),
        objective_function=DiagnosticObjectiveFunction(),
        constraints=(DiagnosticConstraint(),),
        output_root=output_root,
        desc_source_root=fake_desc_root,
        optimizer_method="fixture-success",
        maxiter=5,
        verbose=0,
        allow_high_memory_optimizer=True,
    )

    report = result.report.to_json_dict()
    feasibility_report_path = Path(report["constraint_feasibility_report_path"])
    feasibility_report = json.loads(feasibility_report_path.read_text(encoding="utf-8"))

    assert feasibility_report_path == output_root / (
        "desc_constraint_feasibility_report.json"
    )
    assert feasibility_report["schema_version"] == (
        "desc_joint_constraint_feasibility_report_v1"
    )
    assert feasibility_report["status"] == "passed"
    assert feasibility_report["constraint_terms_all_finite"] is True
    assert feasibility_report["state_vector"]["dim_x"] == 2
    assert feasibility_report["violation_count"] == 2
    assert feasibility_report["max_abs_scaled_error"] == 0.25
    assert feasibility_report["constraint_terms"][0]["name"] == "DiagnosticConstraint"
    assert feasibility_report["constraint_terms"][0]["violation_count"] == 2
    assert feasibility_report["constraint_terms"][0]["worst_rows"][0] == {
        "index": 1,
        "scaled_error": 0.25,
        "scaled_lower_bound": 0.0,
        "scaled_upper_bound": 1.0,
        "scaled_value": 1.25,
    }


def test_desc_joint_runtime_solve_preserves_feasibility_report_on_optimizer_exception(
    tmp_path,
):
    class FixtureThing:
        dim_x = 1

    fixture_thing = FixtureThing()

    class DiagnosticObjectiveFunction:
        things = (fixture_thing,)

        def x(self, *things):
            return [0.0]

    class DiagnosticConstraint:
        things = (fixture_thing,)
        target = 0.0
        bounds = None
        dim_f = 1
        normalization = 1.0
        _normalize_target = True
        built = False

        def build(self, *, use_jit: bool = False, verbose: int = 0) -> None:
            self.use_jit = use_jit
            self.verbose = verbose
            self.built = True

        def xs(self, *things):
            return tuple()

        def _scale(self, values):
            return values

        def compute_scaled(self, *args):
            return [0.0]

        def compute_scaled_error(self, *args):
            return [0.0]

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    with pytest.raises(DescJointRuntimeSolveError) as exc_info:
        run_desc_joint_optimization_runtime(
            equilibrium=SaveableEquilibrium(),
            coilset=SaveableCoilSet(),
            objective_function=DiagnosticObjectiveFunction(),
            constraints=(DiagnosticConstraint(),),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="fixture-assert-x0",
            maxiter=5,
            verbose=0,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    feasibility_report_path = Path(report["constraint_feasibility_report_path"])
    feasibility_report = json.loads(feasibility_report_path.read_text(encoding="utf-8"))

    assert report["status"] == "failed"
    assert report["reason"] == "AssertionError: x0 is infeasible"
    assert feasibility_report_path == output_root / (
        "desc_constraint_feasibility_report.json"
    )
    assert feasibility_report["status"] == "passed"
    assert feasibility_report["constraint_terms_all_finite"] is True
    assert feasibility_report["constraint_terms"][0]["name"] == "DiagnosticConstraint"
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()


def test_desc_joint_runtime_solve_rejects_proximal_constraint_wrapper(tmp_path):
    class FixtureConstraint:
        pass

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc equilibrium\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc coilset\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    with pytest.raises(DescJointRuntimeSolveError) as exc_info:
        run_desc_joint_optimization_runtime(
            equilibrium=SaveableEquilibrium(),
            coilset=SaveableCoilSet(),
            objective_function=object(),
            constraints=(FixtureConstraint(),),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="proximal-fixture-no-eq",
            maxiter=5,
            verbose=0,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["optimizer_method"] == "proximal-fixture-no-eq"
    assert report["optimizer_success"] is None
    assert "uses a proximal constraint wrapper" in report["reason"]
    assert report["optimized_equilibrium_path"] is None
    assert report["optimized_coilset_path"] is None
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()


def test_desc_joint_runtime_solve_rejects_proximal_controls_without_wrapper(
    tmp_path,
):
    class FixtureConstraint:
        pass

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"
    optimizer_controls = build_desc_optimizer_controls(
        proximal_perturb_order=1,
    )

    with pytest.raises(DescJointRuntimeSolveError) as exc_info:
        run_desc_joint_optimization_runtime(
            equilibrium=SaveableEquilibrium(),
            coilset=SaveableCoilSet(),
            objective_function=object(),
            constraints=(FixtureConstraint(),),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="fixture-success",
            maxiter=5,
            verbose=0,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["optimizer_method"] == "fixture-success"
    assert "proximal optimizer controls require" in report["reason"]
    assert report["optimized_equilibrium_path"] is None
    assert report["optimized_coilset_path"] is None
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()


def test_desc_joint_runtime_solve_accepts_proximal_equilibrium_constraint(
    tmp_path,
):
    class FixtureEquilibriumConstraint:
        _equilibrium = True

    class FixtureLinearConstraint:
        linear = True

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc equilibrium\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc coilset\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"
    optimizer_controls = build_desc_optimizer_controls(
        proximal_perturb_order=1,
        proximal_solve_maxiter=2,
        proximal_solve_during_build=False,
    )

    result = run_desc_joint_optimization_runtime(
        equilibrium=SaveableEquilibrium(),
        coilset=SaveableCoilSet(),
        objective_function=object(),
        constraints=(FixtureEquilibriumConstraint(), FixtureLinearConstraint()),
        output_root=output_root,
        desc_source_root=fake_desc_root,
        optimizer_method="proximal-fixture-no-eq",
        maxiter=5,
        verbose=0,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=True,
    )

    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["optimizer_method"] == "proximal-fixture-no-eq"
    assert report["optimizer_success"] is True
    assert len(report["constraint_types"]) == 2
    assert report["constraint_types"][0].endswith(
        "test_desc_joint_runtime_solve_accepts_proximal_equilibrium_constraint."
        "<locals>.FixtureEquilibriumConstraint"
    )
    assert report["constraint_types"][1].endswith(
        "test_desc_joint_runtime_solve_accepts_proximal_equilibrium_constraint."
        "<locals>.FixtureLinearConstraint"
    )
    assert report["optimizer_controls"]["options"] == {
        "perturb_options": {"order": 1},
        "solve_options": {
            "maxiter": 2,
            "solve_during_proximal_build": False,
        },
    }
    assert result.optimized_equilibrium.optimizer_options == {
        "perturb_options": {"order": 1},
        "solve_options": {
            "maxiter": 2,
            "solve_during_proximal_build": False,
        },
    }
    assert (output_root / "desc_equilibrium.h5").is_file()
    assert (output_root / "desc_coils.h5").is_file()


def test_desc_joint_runtime_solve_rejects_optimizer_without_constraint_support(
    tmp_path,
):
    class FixtureConstraint:
        pass

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    with pytest.raises(DescJointRuntimeSolveError) as exc_info:
        run_desc_joint_optimization_runtime(
            equilibrium=SaveableEquilibrium(),
            coilset=SaveableCoilSet(),
            objective_function=object(),
            constraints=(FixtureConstraint(),),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="fixture-no-eq",
            maxiter=5,
            verbose=0,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["allow_high_memory_optimizer"] is True
    assert report["optimizer_method"] == "fixture-no-eq"
    assert report["optimizer_success"] is None
    assert len(report["constraint_types"]) == 1
    assert "does not support equality constraints" in report["reason"]
    assert report["optimized_equilibrium_path"] is None
    assert report["optimized_coilset_path"] is None
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()


def test_desc_joint_runtime_solve_accepts_lsq_auglag_constraints(tmp_path):
    class FixtureConstraint:
        pass

    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc equilibrium\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc coilset\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    result = run_desc_joint_optimization_runtime(
        equilibrium=SaveableEquilibrium(),
        coilset=SaveableCoilSet(),
        objective_function=object(),
        constraints=(FixtureConstraint(),),
        output_root=output_root,
        desc_source_root=fake_desc_root,
        optimizer_method="lsq-auglag",
        maxiter=5,
        verbose=0,
        allow_high_memory_optimizer=True,
    )

    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["optimizer_method"] == "lsq-auglag"
    assert report["optimizer_success"] is True
    assert (output_root / "desc_equilibrium.h5").is_file()
    assert (output_root / "desc_coils.h5").is_file()


def test_desc_joint_runtime_solve_fails_closed_on_optimizer_failure(tmp_path):
    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    class SaveableCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("should not be written\n", encoding="utf-8")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    with pytest.raises(DescJointRuntimeSolveError) as exc_info:
        run_desc_joint_optimization_runtime(
            equilibrium=SaveableEquilibrium(),
            coilset=SaveableCoilSet(),
            objective_function=object(),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="fixture-fail",
            maxiter=5,
            verbose=0,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["allow_high_memory_optimizer"] is True
    assert report["optimizer_success"] is False
    assert report["optimizer_message"] == "fixture requested failure"
    assert report["optimized_equilibrium_path"] is None
    assert report["optimized_coilset_path"] is None
    assert report["failed_optimizer_equilibrium_checkpoint_path"] == str(
        output_root / "desc_failed_optimizer_equilibrium.h5"
    )
    assert report["failed_optimizer_coilset_checkpoint_path"] == str(
        output_root / "desc_failed_optimizer_coils.h5"
    )
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()
    assert (output_root / "desc_failed_optimizer_equilibrium.h5").is_file()
    assert (output_root / "desc_failed_optimizer_coils.h5").is_file()


def test_desc_joint_runtime_solve_fails_closed_on_partial_save(tmp_path):
    class SaveableEquilibrium:
        def save(self, path: str) -> None:
            Path(path).write_text("optimized desc equilibrium\n", encoding="utf-8")

    class PartialSaveCoilSet:
        def save(self, path: str) -> None:
            Path(path).write_text("partial desc coilset\n", encoding="utf-8")
            raise RuntimeError("fixture partial coilset save")

    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    with pytest.raises(DescJointRuntimeSolveError) as exc_info:
        run_desc_joint_optimization_runtime(
            equilibrium=SaveableEquilibrium(),
            coilset=PartialSaveCoilSet(),
            objective_function=object(),
            output_root=output_root,
            desc_source_root=fake_desc_root,
            optimizer_method="fixture-success",
            maxiter=5,
            verbose=0,
            allow_high_memory_optimizer=True,
        )

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["optimized_equilibrium_path"] is None
    assert report["optimized_coilset_path"] is None
    assert report["failed_optimizer_equilibrium_checkpoint_path"] is None
    assert report["failed_optimizer_coilset_checkpoint_path"] is None
    assert "fixture partial coilset save" in report["reason"]
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()
    assert list(output_root.glob(".desc_equilibrium.h5.*.tmp")) == []
    assert list(output_root.glob(".desc_coils.h5.*.tmp")) == []


def test_saved_desc_coil_artifact_exports_to_loadable_simsopt_biot_savart(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    source_field_path = _write_loadable_biot_savart_fixture(
        tmp_path / "biot_savart_opt.json",
    )
    source_surface_path = tmp_path / "surf_seed.json"
    source_surface_path.write_text("{}\n", encoding="utf-8")
    optimized_coilset_path = tmp_path / "desc_coils.h5"
    optimized_coilset_path.write_text("fixture desc coilset artifact\n", encoding="utf-8")
    output_root = tmp_path / "out"

    result = materialize_optimized_desc_coil_artifact_simsopt_export(
        optimized_coilset_path=optimized_coilset_path,
        source_artifacts={
            "seed_surface": source_surface_path,
            "seed_field": source_field_path,
        },
        coil_group_counts={"tf": 1, "banana": 2},
        output_root=output_root,
        sample_count=16,
        simsopt_fourier_order=3,
        desc_source_root=fake_desc_root,
    )

    assert result.exported_biot_savart_path == output_root / "biot_savart_desc_export.json"
    assert result.import_report_path == output_root / "desc_coil_import_report.json"
    assert result.export_report_path == (
        output_root / "desc_optimized_simsopt_export_report.json"
    )
    loaded_export = load(str(result.exported_biot_savart_path))
    assert isinstance(loaded_export, BiotSavart)
    assert len(loaded_export.coils) == 3
    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["optimized_coilset_type"] == "desc.coils.CoilSet"
    assert report["optimized_coilset_source_path"] == str(
        optimized_coilset_path.resolve()
    )
    assert report["artifact_metadata"]["source_artifact_checksums"] == {
        "seed_surface": _sha256(source_surface_path),
        "seed_field": _sha256(source_field_path),
    }
    import_report = json.loads(result.import_report_path.read_text(encoding="utf-8"))
    assert import_report["group_counts"] == {"tf": 1, "banana": 2}
    assert import_report["exported_artifact_paths"] == [
        str(output_root / "biot_savart_desc_export.json")
    ]


def test_saved_desc_coil_artifact_export_fails_closed_when_artifact_missing(tmp_path):
    source_field_path = _write_loadable_biot_savart_fixture(
        tmp_path / "biot_savart_opt.json",
    )
    source_surface_path = tmp_path / "surf_seed.json"
    source_surface_path.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "out"

    with pytest.raises(DescOptimizedSimsoptExportError) as exc_info:
        materialize_optimized_desc_coil_artifact_simsopt_export(
            optimized_coilset_path=tmp_path / "missing_desc_coils.h5",
            source_artifacts={
                "seed_surface": source_surface_path,
                "seed_field": source_field_path,
            },
            coil_group_counts={"tf": 1, "banana": 2},
            output_root=output_root,
            sample_count=16,
            simsopt_fourier_order=3,
            desc_source_root=None,
        )

    report_path = output_root / "desc_optimized_simsopt_export_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert "DESC optimized coil artifact does not exist" in report["reason"]
    assert report["optimized_coilset_source_path"] == str(
        (tmp_path / "missing_desc_coils.h5").resolve()
    )
    assert exc_info.value.report.to_json_dict() == report


def test_saved_desc_equilibrium_exports_to_loadable_simsopt_surface(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    optimized_equilibrium_path = tmp_path / "desc_equilibrium.h5"
    optimized_equilibrium_path.write_text(
        "fixture desc equilibrium artifact\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "out"

    result = materialize_optimized_desc_equilibrium_surface_simsopt_export(
        optimized_equilibrium_path=optimized_equilibrium_path,
        output_root=output_root,
        desc_source_root=fake_desc_root,
        nfp=5,
        stellarator_symmetry=True,
        mpol=1,
        ntor=1,
    )

    assert result.exported_surface_path == output_root / "surf_desc_equilibrium_export.json"
    assert result.export_report_path == (
        output_root / "desc_optimized_surface_export_report.json"
    )
    loaded_surface = load(str(result.exported_surface_path))
    assert isinstance(loaded_surface, SurfaceXYZTensorFourier)
    assert loaded_surface.nfp == 5
    assert loaded_surface.stellsym is True
    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["optimized_equilibrium_type"] == "desc.io.LoadedEquilibrium"
    assert report["optimized_equilibrium_source_path"] == str(
        optimized_equilibrium_path.resolve()
    )
    assert report["exported_surface_path"] == str(result.exported_surface_path.resolve())
    assert report["mpol"] == 1
    assert report["ntor"] == 1
    assert report["sample_count_phi"] == 3
    assert report["sample_count_theta"] == 9
    assert isinstance(report["max_fit_residual_m"], float)
    assert report["max_fit_residual_m"] >= 0.0


def test_desc_runtime_coilset_builds_from_simsopt_seed_field(tmp_path):
    source_field_path = _write_loadable_biot_savart_fixture(
        tmp_path / "biot_savart_opt.json",
    )
    source_surface_path = tmp_path / "surf_seed.json"
    source_surface_path.write_text("{}\n", encoding="utf-8")
    fake_desc_root = _fake_desc_source_root(tmp_path)

    result = build_desc_runtime_coilset_from_simsopt_field(
        source_field_path=source_field_path,
        source_artifacts={
            "seed_surface": source_surface_path,
            "seed_field": source_field_path,
        },
        coil_group_counts={"tf": 1, "banana": 2},
        desc_fourier_order=3,
        sample_count=16,
        source_nfp=5,
        source_stellarator_symmetry=True,
        desc_source_root=fake_desc_root,
    )

    coilset = result.coilset
    assert coilset.NFP == 1
    assert coilset.sym is False
    assert coilset.check_intersection is False
    assert [coil.name for coil in coilset.coils] == [
        "tf_000",
        "banana_000",
        "banana_001",
    ]
    assert [coil.N for coil in coilset.coils] == [3, 3, 3]
    assert [coil.basis for coil in coilset.coils] == ["xyz", "xyz", "xyz"]
    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["desc_version"] == "fixture-desc"
    assert report["coilset_type"] == "desc.coils.CoilSet"
    assert report["field_sample_source_grid"] == 16
    assert report["field_sample_chunk_size"] == 10
    assert report["field_sample_probe_points_xyz"] == [
        [0.78, 0.0, 0.0],
        [0.84, 0.02, 0.01],
        [0.91, -0.02, -0.01],
        [0.95, 0.03, 0.0],
    ]
    assert report["max_desc_simsopt_field_sample_delta_T"] >= 0.0
    assert report["mean_desc_simsopt_field_sample_delta_T"] >= 0.0
    assert report["max_desc_simsopt_field_sample_delta_threshold_T"] == 2.0
    assert report["source_nfp"] == 5
    assert report["source_stellarator_symmetry"] is True
    assert report["coilset_nfp"] == 1
    assert report["coilset_stellarator_symmetry"] is False
    assert report["nfp"] == 5
    assert report["stellarator_symmetry"] is True
    assert report["export_report"]["group_counts"] == {"tf": 1, "banana": 2}
    assert report["export_report"]["artifact_metadata"][
        "source_artifact_checksums"
    ] == {
        "seed_surface": _sha256(source_surface_path),
        "seed_field": _sha256(source_field_path),
    }


def test_desc_runtime_coilset_loads_failed_optimizer_checkpoint_for_continuation(
    tmp_path,
):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    checkpoint_path = tmp_path / "desc_failed_optimizer_coils.h5"
    checkpoint_path.write_text("fixture failed optimizer coilset\n", encoding="utf-8")

    result = load_desc_runtime_coilset_checkpoint(
        checkpoint_path=checkpoint_path,
        coil_group_counts={"tf": 1, "banana": 2},
        source_nfp=5,
        source_stellarator_symmetry=True,
        desc_source_root=fake_desc_root,
        desc_fourier_order=3,
        sample_count=16,
    )
    scoped = scope_desc_coilset_optimization_to_groups(
        coilset=result.coilset,
        coil_group_counts={"tf": 1, "banana": 2},
        optimized_group_names=("banana",),
        desc_source_root=fake_desc_root,
    )

    report = result.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["source_field_path"] == str(checkpoint_path.resolve())
    assert report["reason"] == "DESC runtime coilset loaded from DESC checkpoint."
    assert report["coilset_type"] == "desc.coils.CoilSet"
    assert report["export_report"] is None
    assert report["max_desc_simsopt_field_sample_delta_T"] is None
    assert report["coilset_nfp"] == 5
    assert report["coilset_stellarator_symmetry"] is True
    assert scoped.scope.group_counts == {"tf": 1, "banana": 2}
    assert scoped.scope.optimized_unique_coil_indices == (1, 2)


def test_desc_scoped_coilset_accepts_full_params_for_desc_internal_geometry(
    tmp_path,
):
    source_field_path = _write_loadable_biot_savart_fixture(
        tmp_path / "biot_savart_opt.json",
    )
    source_surface_path = tmp_path / "surf_seed.json"
    source_surface_path.write_text("{}\n", encoding="utf-8")
    fake_desc_root = _fake_desc_source_root(tmp_path)
    build_result = build_desc_runtime_coilset_from_simsopt_field(
        source_field_path=source_field_path,
        source_artifacts={
            "seed_surface": source_surface_path,
            "seed_field": source_field_path,
        },
        coil_group_counts={"tf": 1, "banana": 2},
        desc_fourier_order=3,
        sample_count=16,
        source_nfp=5,
        source_stellarator_symmetry=True,
        desc_source_root=fake_desc_root,
    )

    scoped = scope_desc_coilset_optimization_to_groups(
        coilset=build_result.coilset,
        coil_group_counts={"tf": 1, "banana": 2},
        optimized_group_names=("banana",),
        desc_source_root=fake_desc_root,
    )

    assert scoped.coilset.dim_x == 2
    assert len(scoped.coilset.params_dict) == 2
    assert scoped.coilset.desc_joint_optimization_scope is scoped.scope
    assert "_desc_joint_optimization_scope" not in scoped.coilset.__dict__
    assert "_desc_joint_optimization_scope" not in scoped.coilset._static_attrs
    full_params = build_result.coilset.params_dict
    scoped_points = scoped.coilset._compute_position(params=full_params, grid=16)
    unscoped_points = build_result.coilset._compute_position(params=full_params, grid=16)
    assert np.asarray(scoped_points).shape == np.asarray(unscoped_points).shape
    np.testing.assert_allclose(scoped_points, unscoped_points)


class _ParityFixtureSimsoptField:
    def __init__(self) -> None:
        self.points = np.zeros((0, 3))

    def set_points(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=float)

    def B(self) -> np.ndarray:
        return _parity_fixture_field(self.points)


class _ParityFixtureDescCoilSet:
    def compute_magnetic_field(
        self,
        coords: np.ndarray,
        *,
        basis: str,
        source_grid: int,
        chunk_size: int,
    ) -> np.ndarray:
        assert basis == "xyz"
        assert source_grid == 16
        assert chunk_size == 4
        return _parity_fixture_field(np.asarray(coords, dtype=float)) + 1.0e-13


def _parity_fixture_field(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            2.0 * points[:, 0],
            -3.0 * points[:, 1],
            0.5 + points[:, 2],
        )
    )


def test_desc_simsopt_field_parity_helper_compares_fixed_points_with_tolerance():
    max_delta, mean_delta = _desc_simsopt_field_sample_delta_T(
        biot_savart=_ParityFixtureSimsoptField(),
        coilset=_ParityFixtureDescCoilSet(),
        source_grid=16,
        chunk_size=4,
    )

    assert max_delta <= 2.0e-13
    assert mean_delta <= 2.0e-13


def test_desc_simsopt_field_parity_guard_rejects_symmetry_blowup():
    with pytest.raises(ValueError, match="field parity exceeded"):
        _validate_desc_simsopt_field_sample_delta((3.2, 2.8))


def test_desc_runtime_coilset_rejects_missing_nfp_or_symmetry(tmp_path):
    source_field_path = _write_loadable_biot_savart_fixture(
        tmp_path / "biot_savart_opt.json",
    )
    source_surface_path = tmp_path / "surf_seed.json"
    source_surface_path.write_text("{}\n", encoding="utf-8")
    fake_desc_root = _fake_desc_source_root(tmp_path)
    kwargs = {
        "source_field_path": source_field_path,
        "source_artifacts": {
            "seed_surface": source_surface_path,
            "seed_field": source_field_path,
        },
        "coil_group_counts": {"tf": 1, "banana": 2},
        "desc_fourier_order": 3,
        "sample_count": 16,
        "desc_source_root": fake_desc_root,
    }

    with pytest.raises(DescRuntimeCoilsetBuildError) as missing_nfp:
        build_desc_runtime_coilset_from_simsopt_field(
            **kwargs,
            source_nfp=None,
            source_stellarator_symmetry=True,
        )
    assert "explicit seed source NFP" in missing_nfp.value.report.reason

    with pytest.raises(DescRuntimeCoilsetBuildError) as missing_symmetry:
        build_desc_runtime_coilset_from_simsopt_field(
            **kwargs,
            source_nfp=5,
            source_stellarator_symmetry=None,
        )
    assert "explicit seed source stellarator_symmetry" in (
        missing_symmetry.value.report.reason
    )

    with pytest.raises(DescRuntimeCoilsetBuildError) as invalid_nfp:
        build_desc_runtime_coilset_from_simsopt_field(
            **kwargs,
            source_nfp="bad",
            source_stellarator_symmetry=True,
        )
    assert "source NFP must be a positive integer" in invalid_nfp.value.report.reason
    assert invalid_nfp.value.report.nfp == 0
    assert invalid_nfp.value.report.coilset_nfp == 1
    assert invalid_nfp.value.report.coilset_stellarator_symmetry is False

    with pytest.raises(DescRuntimeCoilsetBuildError) as invalid_symmetry:
        build_desc_runtime_coilset_from_simsopt_field(
            **kwargs,
            source_nfp=5,
            source_stellarator_symmetry="bad",
        )
    assert "source stellarator_symmetry must be boolean" in (
        invalid_symmetry.value.report.reason
    )
    assert invalid_symmetry.value.report.stellarator_symmetry is False
    assert invalid_symmetry.value.report.source_nfp == 5
    assert invalid_symmetry.value.report.coilset_nfp == 1
    assert invalid_symmetry.value.report.coilset_stellarator_symmetry is False


def test_desc_joint_result_schema_separates_status_sections():
    payload = build_preflight_result_payload(
        mode="vacuum_joint",
        input_contract={"seed": "fixture"},
        objective_stack=[
            "VacuumBoundaryError",
            "LinkingCurrentConsistency",
            VOLUME_OBJECTIVE,
        ],
    )

    assert payload["desc_solve_status"]["state"] == "preflight_passed"
    assert payload["artifact_hardware_status"]["state"] == "not_run"
    assert payload["physics_validation_status"]["state"] == "not_run"
    assert payload["promotion_status"]["state"] == "not_requested"

    payload["promotion_status"]["state"] = "preflight_passed"
    with pytest.raises(ValueError, match="promotion_status"):
        validate_desc_joint_result_payload(payload)


def test_desc_joint_result_schema_rejects_forged_promotion_pass(tmp_path):
    payload = build_preflight_result_payload(
        mode="fixed_equilibrium_polish",
        input_contract={"seed": "fixture"},
        objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
    )

    payload["promotion_status"]["state"] = "passed"
    with pytest.raises(ValueError, match="desc_solve_status.state"):
        validate_desc_joint_result_payload(payload)

    payload["desc_solve_status"]["state"] = "passed"
    payload["artifact_hardware_status"]["state"] = "passed"
    payload["physics_validation_status"]["state"] = "passed"
    with pytest.raises(ValueError, match="exported artifact paths"):
        validate_desc_joint_result_payload(payload)

    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    exported_artifact_path.write_text('{"field": "fixture"}\n', encoding="utf-8")
    payload["artifact_hardware_status"]["artifact_paths"] = [
        str(exported_artifact_path)
    ]
    with pytest.raises(ValueError, match="direct hardware/contact oracle evidence"):
        validate_desc_joint_result_payload(payload)

    payload["promotion_status"]["artifact_paths"] = [
        str(tmp_path / "missing_oracle.json")
    ]
    with pytest.raises(ValueError, match="existing direct hardware/contact oracle"):
        validate_desc_joint_result_payload(payload)

    unbound_oracle_path = _write_final_oracle_evidence(
        tmp_path / "unbound_hardware_contact_oracle.json",
        exported_artifact_paths=(),
    )
    payload["promotion_status"]["artifact_paths"] = [str(unbound_oracle_path)]
    with pytest.raises(ValueError, match="exported_artifact_paths"):
        validate_desc_joint_result_payload(payload)

    oracle_path = _write_final_oracle_evidence(
        tmp_path / "direct_hardware_contact_oracle.json",
        exported_artifact_paths=(str(exported_artifact_path),),
    )
    payload["promotion_status"]["artifact_paths"] = [str(oracle_path)]
    validate_desc_joint_result_payload(payload)


def test_equilibrium_seed_spec_requires_explicit_source_kind(tmp_path):
    seed_path = _equilibrium_seed_fixture(tmp_path)
    seed = load_desc_equilibrium_seed_spec(seed_path)

    assert seed.source_kind == "simsopt_surface"
    assert seed.nfp == 5
    assert seed.to_input_contract()["source_kind"] == "simsopt_surface"
    assert seed.target_lcfs_G is None
    low_resolution_seed = seed.with_lcfs_resolution(lcfs_mpol=4, lcfs_ntor=4)
    assert low_resolution_seed.lcfs_mpol == 4
    assert low_resolution_seed.lcfs_ntor == 4
    assert seed.lcfs_mpol == 10
    assert seed.lcfs_ntor == 10

    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    payload.pop("source_kind")
    _write_json(seed_path, payload)
    with pytest.raises(ValueError, match="source_kind"):
        load_desc_equilibrium_seed_spec(seed_path)


def test_equilibrium_seed_runtime_loads_desc_h5_with_source_root(tmp_path):
    seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    seed = load_desc_equilibrium_seed_spec(seed_path)

    loaded_seed = load_desc_equilibrium_seed_runtime(
        seed,
        desc_source_root=fake_desc_root,
    )

    report = loaded_seed.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["loader"] == "desc.io.load"
    assert report["desc_version"] == "fixture-desc"
    assert report["equilibrium_type"] == "desc.io.LoadedEquilibrium"
    assert report["requested_resolution"] == {
        "L": 10,
        "M": 10,
        "N": 10,
        "L_grid": 20,
        "M_grid": 20,
        "N_grid": 20,
    }
    assert report["lcfs_parity"]["comparison_sample_count"] == 441
    assert report["lcfs_parity"]["max_xyz_delta_m"] == 0.0
    equilibrium = loaded_seed.equilibrium
    assert getattr(equilibrium, "surface_refresh_rhos") == [1.0]
    assert getattr(equilibrium, "change_resolution_calls") == [
        (10, 10, 10, 20, 20, 20)
    ]
    assert getattr(equilibrium, "L") == 10
    assert getattr(equilibrium, "M") == 10
    assert getattr(equilibrium, "N") == 10


def test_equilibrium_seed_runtime_loads_vmec_wout_resolution_contract(tmp_path):
    seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="vmec_wout",
        source_filename="wout_fixture.nc",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    seed = load_desc_equilibrium_seed_spec(seed_path)

    loaded_seed = load_desc_equilibrium_seed_runtime(
        seed,
        desc_source_root=fake_desc_root,
    )

    equilibrium = loaded_seed.equilibrium
    assert getattr(equilibrium, "L") == 10
    assert getattr(equilibrium, "M") == 10
    assert getattr(equilibrium, "N") == 10
    assert getattr(equilibrium, "spectral_indexing") == "ansi"
    assert getattr(equilibrium, "profile") == "iota"
    report = loaded_seed.report.to_json_dict()
    assert report["loader"] == "desc.vmec.VMECIO.load"
    assert report["requested_resolution"] == {
        "L": 10,
        "M": 10,
        "N": 10,
        "profile": "iota",
        "spectral_indexing": "ansi",
    }


def test_equilibrium_seed_runtime_loads_simsopt_surface_with_lcfs_parity(tmp_path):
    seed_path = _equilibrium_seed_fixture(tmp_path)
    seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    _write_loadable_simsopt_surface_fixture(Path(seed_payload["source_path"]))
    fake_desc_root = _fake_desc_source_root(tmp_path)
    seed = load_desc_equilibrium_seed_spec(seed_path)

    loaded_seed = load_desc_equilibrium_seed_runtime(
        seed,
        desc_source_root=fake_desc_root,
    )

    equilibrium = loaded_seed.equilibrium
    assert getattr(equilibrium, "NFP") == 5
    assert getattr(equilibrium, "M") == 10
    assert getattr(equilibrium, "N") == 10
    assert getattr(equilibrium, "spectral_indexing") == "ansi"
    report = loaded_seed.report.to_json_dict()
    assert report["status"] == "passed"
    assert report["loader"] == "simsopt_surface_to_desc_equilibrium"
    assert report["desc_version"] == "fixture-desc"
    assert report["equilibrium_type"] == "desc.equilibrium.Equilibrium"
    assert report["requested_resolution"] == {
        "lcfs_mpol": 10,
        "lcfs_ntor": 10,
    }
    assert report["lcfs_parity"]["sample_count_phi"] == 7
    assert report["lcfs_parity"]["sample_count_theta"] == 9
    assert report["lcfs_parity"]["comparison_sample_count"] == 63
    assert report["lcfs_parity"]["max_xyz_delta_m"] == pytest.approx(0.0)
    assert report["lcfs_G_scaling"] is None


def test_equilibrium_seed_runtime_scales_simsopt_surface_psi_to_target_G(tmp_path):
    seed_path = _equilibrium_seed_fixture(tmp_path)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    payload["target_lcfs_G"] = -2.0
    _write_json(seed_path, payload)
    _write_loadable_simsopt_surface_fixture(Path(payload["source_path"]))
    fake_desc_root = _fake_desc_source_root(tmp_path)
    seed = load_desc_equilibrium_seed_spec(seed_path)

    loaded_seed = load_desc_equilibrium_seed_runtime(
        seed,
        desc_source_root=fake_desc_root,
    )

    report = loaded_seed.report.to_json_dict()
    assert getattr(loaded_seed.equilibrium, "Psi") == pytest.approx(0.1)
    assert report["lcfs_G_scaling"] == {
        "target_lcfs_G": -2.0,
        "unscaled_lcfs_G": -20.0,
        "scaled_lcfs_G": -2.0,
        "psi_before_Wb": 1.0,
        "psi_after_Wb": 0.1,
        "psi_scale_factor": 0.1,
        "relative_G_error": 0.0,
    }
    assert report["lcfs_parity"]["mean_xyz_delta_m"] == pytest.approx(0.0)
    assert report["lcfs_parity"]["rms_xyz_delta_m"] == pytest.approx(0.0)


def test_equilibrium_seed_runtime_fails_closed_for_invalid_simsopt_surface(tmp_path):
    fake_desc_root = _fake_desc_source_root(tmp_path)
    seed = load_desc_equilibrium_seed_spec(_equilibrium_seed_fixture(tmp_path))

    with pytest.raises(DescEquilibriumRuntimeLoadError) as exc_info:
        load_desc_equilibrium_seed_runtime(seed, desc_source_root=fake_desc_root)

    report = exc_info.value.report.to_json_dict()
    assert report["status"] == "failed"
    assert report["loader"] == "simsopt_surface_to_desc_equilibrium"
    assert report["lcfs_parity"] is None


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    (
        ("nfp", True, "integer"),
        ("nfp", 5.5, "integer"),
        ("lcfs_mpol", 10.1, "integer"),
        ("major_radius_m", True, "numeric"),
        ("minor_radius_m", float("nan"), "finite and positive"),
        ("target_lcfs_G", float("nan"), "finite"),
    ),
)
def test_equilibrium_seed_spec_rejects_unsafe_numeric_values(
    tmp_path,
    field_name,
    bad_value,
    message,
):
    seed_path = _equilibrium_seed_fixture(tmp_path)
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    payload[field_name] = bad_value
    _write_json(seed_path, payload)

    with pytest.raises(ValueError, match=message):
        load_desc_equilibrium_seed_spec(seed_path)


def test_desc_joint_runner_preflight_writes_contract_payload(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path)
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--output-root",
            str(output_root),
            "--preflight-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_runner_env_without_desc_runtime_device(),
    )
    preflight_path = Path(completed.stdout.strip())
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))

    assert preflight_path == output_root / "desc_joint_preflight.json"
    assert payload["run_mode"] == "vacuum_joint"
    assert payload["input_contract"]["selected_seed"]["label"] == "slidclean_chomp"
    assert payload["input_contract"]["selected_seed_coil_group_counts"] == {
        "tf": 1,
        "banana": 2,
    }
    assert payload["input_contract"]["selected_seed_coil_group_source"] == (
        "source_results"
    )
    assert payload["input_contract"]["selected_seed_field_inventory"]["coil_count"] == 3
    assert payload["input_contract"]["selected_seed_field_inventory"][
        "current_sign_counts"
    ] == {
        "negative": 1,
        "zero": 1,
        "positive": 1,
    }
    assert payload["input_contract"]["equilibrium_seed"]["source_kind"] == (
        "simsopt_surface"
    )
    assert payload["input_contract"]["equilibrium_seed"]["target_lcfs_G"] == (
        pytest.approx(-2.01062 / (2.0 * math.pi))
    )
    assert "QuadraticFlux" not in payload["objective_stack"]


def test_desc_joint_runner_preflight_records_predecessor_manifests(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path)
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)
    fixed_polish_manifest = _write_json(
        tmp_path / "fixed_polish_validation_manifest.json",
        {"schema_version": "desc_joint_validation_manifest_v1"},
    )
    lane_b_manifest = _write_json(
        tmp_path / "lane_b_validation_manifest.json",
        {"schema_version": "desc_joint_validation_manifest_v1"},
    )
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "finite_beta_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--output-root",
            str(output_root),
            "--fixed-polish-predecessor-manifest",
            str(fixed_polish_manifest),
            "--lane-b-predecessor-manifest",
            str(lane_b_manifest),
            "--preflight-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_runner_env_without_desc_runtime_device(),
    )
    preflight_path = Path(completed.stdout.strip())
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))

    assert payload["fixed_polish_predecessor_status"] == {
        "state": "passed",
        "reason": "fixed-polish predecessor validation manifest supplied",
        "artifact_paths": [str(fixed_polish_manifest.resolve())],
    }
    assert payload["lane_b_predecessor_status"] == {
        "state": "passed",
        "reason": "Lane B predecessor validation manifest supplied",
        "artifact_paths": [str(lane_b_manifest.resolve())],
    }
    assert payload["run_configuration"]["predecessors"] == {
        "fixed_polish_predecessor_manifest": str(fixed_polish_manifest.resolve()),
        "lane_b_predecessor_manifest": str(lane_b_manifest.resolve()),
    }


def test_desc_joint_runner_preflight_applies_resolution_preset_and_overrides(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path)
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--output-root",
            str(output_root),
            "--resolution-preset",
            "production",
            "--desc-grid-n",
            "13",
            "--conversion-sample-count",
            "17",
            "--preflight-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_runner_env_without_desc_runtime_device(),
    )
    preflight_path = Path(completed.stdout.strip())
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))
    run_configuration = payload["run_configuration"]

    assert run_configuration["resolution_preset"] == "production"
    assert run_configuration["desc_runtime"] == {
        "desc_source_root": None,
        "desc_runtime_device": None,
        "bootstrapped_desc_runtime_device": None,
        "desc_coilset_checkpoint": None,
        "desc_grid_n": 13,
        "desc_equilibrium_lcfs_mpol": 10,
        "desc_equilibrium_lcfs_ntor": 10,
        "desc_bs_chunk_size": 25,
        "desc_dist_chunk_size": 8,
        "desc_jac_chunk_size": 10,
        "desc_objective_use_jit": False,
        "desc_objective_deriv_mode": "blocked",
        "desc_joint_constraint_policy": HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
        "desc_objective_ablation_policy": FULL_DESC_OBJECTIVE_ABLATION_POLICY,
    }
    assert run_configuration["conversion"] == {
        "desc_fourier_order": 10,
        "conversion_sample_count": 17,
        "simsopt_fourier_order": 10,
    }


def test_desc_joint_runner_bootstraps_desc_runtime_device_before_imports(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path)
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"
    marker_path = tmp_path / "desc_device_marker.txt"
    env = os.environ.copy()
    env["FAKE_DESC_SET_DEVICE_MARKER"] = str(marker_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--desc-runtime-device",
            "gpu",
            "--output-root",
            str(output_root),
            "--preflight-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    preflight_path = Path(completed.stdout.strip())
    payload = json.loads(preflight_path.read_text(encoding="utf-8"))

    assert marker_path.read_text(encoding="utf-8") == "gpu\n"
    assert payload["run_configuration"]["desc_runtime"]["desc_runtime_device"] == "gpu"
    assert (
        payload["run_configuration"]["desc_runtime"][
            "bootstrapped_desc_runtime_device"
        ]
        == "gpu"
    )


def test_desc_joint_runner_equilibrium_load_only_writes_runtime_report(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--equilibrium-load-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_runner_env_without_desc_runtime_device(),
    )

    report_path = Path(completed.stdout.strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == output_root / "desc_equilibrium_load_report.json"
    assert (output_root / "desc_joint_preflight.json").is_file()
    assert report["status"] == "passed"
    assert report["loader"] == "desc.io.load"
    assert report["desc_version"] == "fixture-desc"
    assert report["equilibrium_type"] == "desc.io.LoadedEquilibrium"
    assert report["requested_resolution"] == {
        "L": 4,
        "M": 4,
        "N": 4,
        "L_grid": 8,
        "M_grid": 8,
        "N_grid": 8,
    }
    assert report["lcfs_parity"]["comparison_sample_count"] == 81
    assert report["lcfs_parity"]["max_xyz_delta_m"] == 0.0
    assert report["mode_profile_adjustment"] == {
        "mode": "fixed_equilibrium_polish",
        "action": "preserved",
        "reason": "mode does not use DESC VacuumBoundaryError.",
        "before_max_abs": {
            "pressure": 22098.0,
            "current": 3688.0,
        },
        "after_max_abs": {
            "pressure": 22098.0,
            "current": 3688.0,
        },
    }


def test_desc_joint_runner_equilibrium_load_only_accepts_simsopt_surface_seed(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path)
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)
    equilibrium_seed_payload = json.loads(
        equilibrium_seed_path.read_text(encoding="utf-8")
    )
    _write_loadable_simsopt_surface_fixture(
        Path(equilibrium_seed_payload["source_path"])
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--equilibrium-load-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    report_path = Path(completed.stdout.strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == output_root / "desc_equilibrium_load_report.json"
    assert report["status"] == "passed"
    assert report["loader"] == "simsopt_surface_to_desc_equilibrium"
    assert report["equilibrium_type"] == "desc.equilibrium.Equilibrium"
    assert report["lcfs_parity"]["comparison_sample_count"] == 63
    assert report["lcfs_parity"]["max_xyz_delta_m"] == pytest.approx(0.0)
    assert report["lcfs_G_scaling"]["target_lcfs_G"] == pytest.approx(
        -2.01062 / (2.0 * math.pi)
    )
    assert report["lcfs_G_scaling"]["unscaled_lcfs_G"] == -20.0
    assert report["lcfs_G_scaling"]["scaled_lcfs_G"] == pytest.approx(
        -2.01062 / (2.0 * math.pi)
    )


def test_desc_joint_runner_objective_assembly_only_writes_runtime_reports(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    objective_report_path = Path(completed.stdout.strip())
    objective_report = json.loads(objective_report_path.read_text(encoding="utf-8"))
    coilset_report = json.loads(
        (output_root / "desc_runtime_coilset_build_report.json").read_text(
            encoding="utf-8"
        )
    )
    optimizer_scope_report = json.loads(
        (output_root / "desc_runtime_optimizer_scope_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert objective_report_path == output_root / "desc_objective_assembly_report.json"
    assert (output_root / "desc_joint_preflight.json").is_file()
    assert (output_root / "desc_equilibrium_load_report.json").is_file()
    assert not (output_root / "desc_result.json").exists()
    assert coilset_report["status"] == "passed"
    assert coilset_report["coilset_type"] == "desc.coils.CoilSet"
    assert coilset_report["field_sample_source_grid"] == 16
    assert coilset_report["field_sample_chunk_size"] == 10
    assert coilset_report["max_desc_simsopt_field_sample_delta_T"] >= 0.0
    assert coilset_report["export_report"]["group_counts"] == {
        "tf": 1,
        "banana": 2,
    }
    assert optimizer_scope_report["status"] == "passed"
    assert optimizer_scope_report["scope"]["optimized_group_names"] == ["banana"]
    assert optimizer_scope_report["scope"]["fixed_group_names"] == ["tf"]
    assert optimizer_scope_report["scope"]["optimized_unique_coil_indices"] == [1, 2]
    assert optimizer_scope_report["scope"]["fixed_unique_coil_indices"] == [0]
    assert objective_report["status"] == "passed"
    assert objective_report["grid_n"] == 7
    assert objective_report["linking_current_grid_n"] == 7
    assert objective_report["bs_chunk_size"] == 10
    assert objective_report["dist_chunk_size"] == 2
    assert objective_report["jac_chunk_size"] == 5
    assert objective_report["objective_names"] == [
        "QuadraticFlux",
        "LinkingCurrentConsistency",
        "CoilLength",
        "CoilCurvature",
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
    ]


def test_desc_joint_runner_preserves_profiles_for_finite_beta_assembly(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "finite_beta_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    objective_report_path = Path(completed.stdout.strip())
    load_report = json.loads(
        (output_root / "desc_equilibrium_load_report.json").read_text(
            encoding="utf-8"
        )
    )
    objective_report = json.loads(objective_report_path.read_text(encoding="utf-8"))

    assert objective_report_path == output_root / "desc_objective_assembly_report.json"
    assert load_report["mode_profile_adjustment"] == {
        "mode": "finite_beta_joint",
        "action": "preserved",
        "reason": "mode does not use DESC VacuumBoundaryError.",
        "before_max_abs": {
            "pressure": 22098.0,
            "current": 3688.0,
        },
        "after_max_abs": {
            "pressure": 22098.0,
            "current": 3688.0,
        },
    }
    assert objective_report["status"] == "passed"
    assert "BoundaryError" in objective_report["objective_names"]
    assert "VacuumBoundaryError" not in objective_report["objective_names"]


def test_desc_joint_runner_objective_assembly_can_restart_from_desc_coilset_checkpoint(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="desc_failed_optimizer_equilibrium.h5",
    )
    coilset_checkpoint_path = tmp_path / "desc_failed_optimizer_coils.h5"
    coilset_checkpoint_path.write_text(
        "fixture failed optimizer coilset\n",
        encoding="utf-8",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--desc-coilset-checkpoint",
            str(coilset_checkpoint_path),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert Path(completed.stdout.strip()) == (
        output_root / "desc_objective_assembly_report.json"
    )
    preflight = json.loads(
        (output_root / "desc_joint_preflight.json").read_text(encoding="utf-8")
    )
    coilset_report = json.loads(
        (output_root / "desc_runtime_coilset_build_report.json").read_text(
            encoding="utf-8"
        )
    )
    optimizer_scope_report = json.loads(
        (output_root / "desc_runtime_optimizer_scope_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert preflight["input_contract"]["desc_coilset_checkpoint"] == str(
        coilset_checkpoint_path.resolve()
    )
    assert preflight["run_configuration"]["desc_runtime"][
        "desc_coilset_checkpoint"
    ] == str(coilset_checkpoint_path.resolve())
    assert coilset_report["status"] == "passed"
    assert coilset_report["source_field_path"] == str(coilset_checkpoint_path.resolve())
    assert coilset_report["reason"] == (
        "DESC runtime coilset loaded from DESC checkpoint."
    )
    assert coilset_report["export_report"] is None
    assert coilset_report["max_desc_simsopt_field_sample_delta_T"] is None
    assert optimizer_scope_report["status"] == "passed"
    assert optimizer_scope_report["scope"]["group_counts"] == {
        "tf": 1,
        "banana": 2,
    }
    assert optimizer_scope_report["scope"]["optimized_unique_coil_indices"] == [1, 2]
    assert not (output_root / "desc_result.json").exists()
    assert not (output_root / "biot_savart_desc_export.json").exists()


def test_desc_joint_runner_joint_objective_assembly_uses_seed_surface_volume(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    seed_surface_path = Path(seed_manifest["candidates"][0]["surface"])
    expected_volume_m3 = abs(float(load(str(seed_surface_path)).volume()))
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path, equilibrium_volume=-1.0)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    objective_report_path = Path(completed.stdout.strip())
    objective_report = json.loads(objective_report_path.read_text(encoding="utf-8"))
    assert objective_report_path == output_root / "desc_objective_assembly_report.json"
    assert objective_report["status"] == "passed"
    assert objective_report["objective_names"] == [
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    ]
    assert objective_report["constraint_names"] == [
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert (
        objective_report["joint_constraint_policy"]
        == HARD_VOLUME_AND_FORCE_BALANCE_POLICY
    )
    assert (
        objective_report["objective_ablation_policy"]
        == FULL_DESC_OBJECTIVE_ABLATION_POLICY
    )
    assert objective_report["volume_target_m3"] == pytest.approx(
        -expected_volume_m3
    )
    preflight = json.loads(
        (output_root / "desc_joint_preflight.json").read_text(encoding="utf-8")
    )
    assert (
        preflight["run_configuration"]["desc_runtime"][
            "desc_joint_constraint_policy"
        ]
        == HARD_VOLUME_AND_FORCE_BALANCE_POLICY
    )
    assert (
        preflight["run_configuration"]["desc_runtime"][
            "desc_objective_ablation_policy"
        ]
        == FULL_DESC_OBJECTIVE_ABLATION_POLICY
    )


def test_desc_joint_runner_objective_assembly_supports_proximal_policy(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    seed_surface_path = Path(seed_manifest["candidates"][0]["surface"])
    expected_volume_m3 = abs(float(load(str(seed_surface_path)).volume()))
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path, equilibrium_volume=-1.0)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "proximal-lsq-exact",
            "--desc-joint-constraint-policy",
            PROXIMAL_FORCE_BALANCE_POLICY,
            "--desc-proximal-perturb-order",
            "1",
            "--desc-proximal-solve-maxiter",
            "2",
            "--no-desc-proximal-solve-during-build",
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    objective_report_path = Path(completed.stdout.strip())
    objective_report = json.loads(objective_report_path.read_text(encoding="utf-8"))
    assert objective_report_path == output_root / "desc_objective_assembly_report.json"
    assert objective_report["status"] == "passed"
    assert objective_report["objective_names"] == [
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
        VOLUME_OBJECTIVE,
        "CoilSetMinDistance",
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    ]
    assert objective_report["constraint_names"] == [
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert (
        objective_report["joint_constraint_policy"]
        == PROXIMAL_FORCE_BALANCE_POLICY
    )
    assert (
        objective_report["objective_ablation_policy"]
        == FULL_DESC_OBJECTIVE_ABLATION_POLICY
    )
    assert objective_report["volume_target_m3"] == pytest.approx(
        -expected_volume_m3
    )
    preflight = json.loads(
        (output_root / "desc_joint_preflight.json").read_text(encoding="utf-8")
    )
    assert (
        preflight["run_configuration"]["desc_runtime"][
            "desc_joint_constraint_policy"
        ]
        == PROXIMAL_FORCE_BALANCE_POLICY
    )
    assert (
        preflight["run_configuration"]["desc_runtime"][
            "desc_objective_ablation_policy"
        ]
        == FULL_DESC_OBJECTIVE_ABLATION_POLICY
    )
    assert preflight["run_configuration"]["optimizer"]["method"] == (
        "proximal-lsq-exact"
    )
    assert preflight["run_configuration"]["optimizer"]["controls"]["options"] == {
        "perturb_options": {"order": 1},
        "solve_options": {
            "maxiter": 2,
            "solve_during_proximal_build": False,
        },
    }


def test_desc_joint_runner_objective_assembly_supports_boundary_fidelity_guard(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path, equilibrium_volume=-1.0)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-boundary-fidelity-policy",
            BOUNDARY_FIDELITY_FIX_HIGH_MODES,
            "--desc-boundary-fidelity-free-mode-sum",
            "1",
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    objective_report_path = Path(completed.stdout.strip())
    objective_report = json.loads(objective_report_path.read_text(encoding="utf-8"))
    assert objective_report_path == output_root / "desc_objective_assembly_report.json"
    assert objective_report["status"] == "passed"
    assert objective_report["constraint_names"] == [
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        FIX_BOUNDARY_R_CONSTRAINT,
        FIX_BOUNDARY_Z_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert objective_report["boundary_fidelity"] == {
        "policy": BOUNDARY_FIDELITY_FIX_HIGH_MODES,
        "free_mode_sum": 1,
        "free_selector": "|m| + |n| <= free_mode_sum",
        "fixed_selector": "|m| + |n| > free_mode_sum",
        "R_mode_count": 5,
        "Z_mode_count": 4,
        "fixed_R_mode_count": 3,
        "fixed_Z_mode_count": 3,
        "free_R_mode_count": 2,
        "free_Z_mode_count": 1,
        "fixed_R_modes": [[0, 2, 0], [0, 0, 2], [0, 1, 1]],
        "fixed_Z_modes": [[0, 2, 0], [0, 0, -2], [0, 1, -1]],
    }
    preflight = json.loads(
        (output_root / "desc_joint_preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["objective_stack"] == [
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        COIL_SET_MIN_DISTANCE_OBJECTIVE,
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
        FIX_BOUNDARY_R_CONSTRAINT,
        FIX_BOUNDARY_Z_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    desc_runtime_config = preflight["run_configuration"]["desc_runtime"]
    assert (
        desc_runtime_config["desc_boundary_fidelity_policy"]
        == BOUNDARY_FIDELITY_FIX_HIGH_MODES
    )
    assert desc_runtime_config["desc_boundary_fidelity_free_mode_sum"] == 1


def test_desc_joint_runner_objective_assembly_supports_hard_linking_current_policy(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path, equilibrium_volume=-1.0)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-joint-constraint-policy",
            HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY,
            "--desc-boundary-fidelity-policy",
            BOUNDARY_FIDELITY_FIX_HIGH_MODES,
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    objective_report_path = Path(completed.stdout.strip())
    objective_report = json.loads(objective_report_path.read_text(encoding="utf-8"))
    assert objective_report["status"] == "passed"
    assert objective_report["objective_names"] == [
        "VacuumBoundaryError",
        VOLUME_OBJECTIVE,
        COIL_SET_MIN_DISTANCE_OBJECTIVE,
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    ]
    assert objective_report["constraint_names"] == [
        "LinkingCurrentConsistency",
        FORCE_BALANCE_CONSTRAINT,
        FIX_BOUNDARY_R_CONSTRAINT,
        FIX_BOUNDARY_Z_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert (
        objective_report["joint_constraint_policy"]
        == HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY
    )
    preflight = json.loads(
        (output_root / "desc_joint_preflight.json").read_text(encoding="utf-8")
    )
    desc_runtime_config = preflight["run_configuration"]["desc_runtime"]
    assert (
        desc_runtime_config["desc_joint_constraint_policy"]
        == HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY
    )


def test_desc_joint_runner_objective_assembly_supports_objective_ablation(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path, equilibrium_volume=-1.0)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-joint-constraint-policy",
            PROXIMAL_FORCE_BALANCE_POLICY,
            "--desc-objective-ablation-policy",
            PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY,
            "--objective-assembly-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    objective_report_path = Path(completed.stdout.strip())
    objective_report = json.loads(objective_report_path.read_text(encoding="utf-8"))
    assert objective_report["status"] == "passed"
    assert objective_report["objective_names"] == [
        "VacuumBoundaryError",
        VOLUME_OBJECTIVE,
    ]
    assert objective_report["constraint_names"] == [
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert objective_report["joint_constraint_policy"] == PROXIMAL_FORCE_BALANCE_POLICY
    assert (
        objective_report["objective_ablation_policy"]
        == PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY
    )
    preflight = json.loads(
        (output_root / "desc_joint_preflight.json").read_text(encoding="utf-8")
    )
    assert preflight["objective_stack"] == [
        "VacuumBoundaryError",
        VOLUME_OBJECTIVE,
        FORCE_BALANCE_CONSTRAINT,
        FIX_COIL_CURRENT_CONSTRAINT,
    ]
    assert (
        preflight["run_configuration"]["desc_runtime"][
            "desc_objective_ablation_policy"
        ]
        == PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY
    )


def test_desc_joint_runner_objective_eval_only_writes_smoke_report(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"
    objective_eval_args = [
        sys.executable,
        str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
        "--mode",
        "fixed_equilibrium_polish",
        "--hardware-spec",
        str(hardware_spec_path),
        "--seed-manifest",
        str(seed_manifest_path),
        "--seed-label",
        "slidclean_chomp",
        "--equilibrium-seed",
        str(equilibrium_seed_path),
        "--desc-source-root",
        str(fake_desc_root),
        "--desc-fourier-order",
        "3",
        "--conversion-sample-count",
        "16",
        "--desc-grid-n",
        "7",
        "--objective-eval-only",
    ]

    completed = subprocess.run(
        [
            *objective_eval_args,
            "--output-root",
            str(output_root),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    evaluation_report_path = Path(completed.stdout.strip())
    evaluation_report = json.loads(
        evaluation_report_path.read_text(encoding="utf-8")
    )
    assert evaluation_report_path == (
        output_root / "desc_objective_evaluation_report.json"
    )
    assert (output_root / "desc_equilibrium_load_report.json").is_file()
    assert (output_root / "desc_runtime_coilset_build_report.json").is_file()
    assert (output_root / "desc_runtime_optimizer_scope_report.json").is_file()
    assert (output_root / "desc_objective_assembly_report.json").is_file()
    assert not (output_root / "desc_result.json").exists()
    assert evaluation_report["status"] == "passed"
    assert evaluation_report["evaluation_mode"] == "sequential_terms"
    assert evaluation_report["dim_x"] == 2
    assert evaluation_report["dim_f"] == 6
    assert evaluation_report["jacobian_shape"] is None
    assert evaluation_report["jacobian_all_finite"] is None
    assert evaluation_report["gradient_all_finite"] is None
    assert len(evaluation_report["objective_term_reports"]) == 6
    assert evaluation_report["build_seconds"] >= 0.0
    assert evaluation_report["value_seconds"] >= 0.0
    assert evaluation_report["jacobian_seconds"] is None
    assert evaluation_report["gradient_seconds"] is None

    jacobian_output_root = tmp_path / "out_jacobian"
    jacobian_completed = subprocess.run(
        [
            *objective_eval_args,
            "--output-root",
            str(jacobian_output_root),
            "--objective-eval-jacobian",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    jacobian_report_path = Path(jacobian_completed.stdout.strip())
    jacobian_report = json.loads(jacobian_report_path.read_text(encoding="utf-8"))
    assert jacobian_report_path == (
        jacobian_output_root / "desc_objective_evaluation_report.json"
    )
    assert jacobian_report["status"] == "passed"
    assert jacobian_report["evaluation_mode"] == "sequential_terms"
    assert jacobian_report["dim_x"] == 2
    assert jacobian_report["dim_f"] == 6
    assert jacobian_report["jacobian_shape"] == [6, 2]
    assert jacobian_report["jacobian_all_finite"] is True
    assert jacobian_report["jacobian_seconds"] >= 0.0

    gradient_output_root = tmp_path / "out_gradient"
    gradient_completed = subprocess.run(
        [
            *objective_eval_args,
            "--output-root",
            str(gradient_output_root),
            "--objective-eval-gradient",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    gradient_report_path = Path(gradient_completed.stdout.strip())
    gradient_report = json.loads(gradient_report_path.read_text(encoding="utf-8"))
    gradient_preflight = json.loads(
        (gradient_output_root / "desc_joint_preflight.json").read_text(
            encoding="utf-8"
        )
    )
    gradient_configuration = gradient_preflight["run_configuration"]
    assert gradient_report_path == (
        gradient_output_root / "desc_objective_evaluation_report.json"
    )
    assert gradient_report["status"] == "passed"
    assert gradient_report["evaluation_mode"] == "sequential_terms"
    assert gradient_report["dim_x"] == 2
    assert gradient_report["dim_f"] == 6
    assert gradient_report["jacobian_shape"] is None
    assert gradient_report["jacobian_all_finite"] is None
    assert gradient_report["gradient_all_finite"] is True
    assert gradient_report["gradient_seconds"] >= 0.0
    assert gradient_report["gradient_progress_path"] == str(
        gradient_output_root / "desc_objective_gradient_progress.jsonl"
    )
    assert gradient_configuration["objective_eval"]["gradient"] is True
    gradient_progress = [
        json.loads(line)
        for line in (
            gradient_output_root / "desc_objective_gradient_progress.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in gradient_progress] == [
        "start",
        "finish",
        "start",
        "finish",
        "start",
        "finish",
        "start",
        "finish",
        "start",
        "finish",
        "start",
        "finish",
    ]
    for term_report in gradient_report["objective_term_reports"]:
        assert term_report["gradient_all_finite"] is True
        assert term_report["gradient_size"] == 1
        assert term_report["gradient_block_shapes"] == [[]]


def test_desc_joint_runner_fixed_polish_only_writes_desc_result_and_artifact(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-success",
            "--desc-maxiter",
            "5",
            "--desc-optimizer-verbose",
            "0",
            "--desc-optimizer-ftol",
            "0.01",
            "--desc-optimizer-xtol",
            "0.0002",
            "--desc-optimizer-gtol",
            "0.000003",
            "--desc-optimizer-ctol",
            "0.004",
            "--desc-optimizer-max-nfev",
            "19",
            "--desc-optimizer-max-dx",
            "0.25",
            "--desc-optimizer-initial-trust-radius",
            "0.02",
            "--desc-optimizer-max-trust-radius",
            "0.05",
            "--desc-optimizer-min-trust-radius",
            "0",
            "--allow-high-memory-desc-optimizer",
            "--fixed-polish-only",
        ],
        check=True,
        text=True,
        capture_output=True,
        env=_runner_env_without_desc_runtime_device(),
    )

    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(
        (output_root / "desc_fixed_polish_solve_report.json").read_text(
            encoding="utf-8"
        )
    )
    validation_manifest = json.loads(
        (output_root / "desc_joint_validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert result_path == output_root / "desc_result.json"
    assert (output_root / "desc_coils.h5").is_file()
    _assert_legacy_hardware_fields_fail_closed(result_payload)
    run_configuration = result_payload["run_configuration"]
    assert run_configuration["desc_runtime"] == {
        "desc_source_root": str(fake_desc_root),
        "desc_runtime_device": None,
        "bootstrapped_desc_runtime_device": None,
        "desc_coilset_checkpoint": None,
        "desc_grid_n": 7,
        "desc_equilibrium_lcfs_mpol": 4,
        "desc_equilibrium_lcfs_ntor": 4,
        "desc_bs_chunk_size": 10,
        "desc_dist_chunk_size": 2,
        "desc_jac_chunk_size": 5,
        "desc_objective_use_jit": False,
        "desc_objective_deriv_mode": "blocked",
        "desc_joint_constraint_policy": HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
        "desc_objective_ablation_policy": FULL_DESC_OBJECTIVE_ABLATION_POLICY,
    }
    assert run_configuration["conversion"] == {
        "desc_fourier_order": 3,
        "conversion_sample_count": 16,
        "simsopt_fourier_order": 3,
    }
    assert run_configuration["optimizer"] == {
        "method": "fixture-success",
        "maxiter": 5,
        "verbose": 0,
        "controls": {
            "tolerances": {
                "ftol": 0.01,
                "xtol": 0.0002,
                "gtol": 0.000003,
                "ctol": 0.004,
            },
            "options": {
                "max_nfev": 19,
                "max_dx": 0.25,
                "initial_trust_radius": 0.02,
                "max_trust_radius": 0.05,
                "min_trust_radius": 0.0,
            },
        },
        "allow_high_memory_optimizer": True,
        "optimized_coil_groups": ["banana"],
    }
    run_timings = result_payload["run_timing_seconds"]
    _assert_nonnegative_timing(run_timings, "preflight")
    _assert_nonnegative_timing(run_timings, "equilibrium_load")
    _assert_nonnegative_timing(run_timings, "coilset_build")
    _assert_nonnegative_timing(run_timings, "objective_assembly")
    _assert_nonnegative_timing(run_timings, "optimizer")
    _assert_nonnegative_timing(run_timings, "optimized_simsopt_export")
    _assert_nonnegative_timing(run_timings, "result_materialization")
    _assert_nonnegative_timing(run_timings, "validation_manifest")
    inventory = _assert_runner_inventory(result_payload, result_path=result_path)
    assert inventory["output_artifacts"]["desc_runtime_desc_coils"] == str(
        output_root / "desc_coils.h5"
    )
    assert inventory["output_artifacts"]["desc_runtime_exported_biot_savart"] == str(
        output_root / "biot_savart_desc_export.json"
    )
    assert result_payload["desc_solve_status"]["state"] == "passed"
    assert result_payload["desc_solve_status"]["artifact_paths"] == [
        str(output_root / "desc_coils.h5")
    ]
    assert result_payload["artifact_hardware_status"]["state"] == "blocked"
    assert result_payload["artifact_hardware_status"]["artifact_paths"] == [
        str(output_root / "biot_savart_desc_export.json")
    ]
    assert result_payload["physics_validation_status"]["state"] == "not_run"
    assert result_payload["promotion_status"]["state"] == "blocked"
    assert result_payload["desc_runtime_artifacts"]["desc_coils"] == str(
        output_root / "desc_coils.h5"
    )
    assert result_payload["desc_runtime_artifacts"]["exported_biot_savart"] == str(
        output_root / "biot_savart_desc_export.json"
    )
    assert result_payload["desc_runtime_artifacts"]["desc_coil_import_report"] == str(
        output_root / "desc_coil_import_report.json"
    )
    assert result_payload["desc_runtime_artifacts"][
        "optimized_simsopt_export_report"
    ] == str(output_root / "desc_optimized_simsopt_export_report.json")
    assert (output_root / "biot_savart_desc_export.json").is_file()
    assert (output_root / "desc_coil_import_report.json").is_file()
    assert (output_root / "desc_optimized_simsopt_export_report.json").is_file()
    optimized_export_report = json.loads(
        (output_root / "desc_optimized_simsopt_export_report.json").read_text(
            encoding="utf-8"
        )
    )
    selected_seed_field = Path(result_payload["input_contract"]["selected_seed"]["field"])
    assert optimized_export_report["optimized_coilset_source_path"] == str(
        (output_root / "desc_coils.h5").resolve()
    )
    assert optimized_export_report["artifact_metadata"][
        "source_artifact_checksums"
    ]["seed_field"] == _sha256(selected_seed_field)
    loaded_export = load(str(output_root / "biot_savart_desc_export.json"))
    assert isinstance(loaded_export, BiotSavart)
    assert len(loaded_export.coils) == 3
    assert solve_report["status"] == "passed"
    assert solve_report["allow_high_memory_optimizer"] is True
    assert solve_report["constraint_types"] == [
        "desc.objectives.FixCoilCurrent",
    ]
    assert solve_report["optimizer_controls"] == run_configuration["optimizer"][
        "controls"
    ]
    assert result_payload["desc_optimizer_result"]["controls"] == (
        run_configuration["optimizer"]["controls"]
    )
    assert solve_report["optimizer_success"] is True
    assert solve_report["maxiter"] == 5
    assert validation_manifest["exported_artifact_paths"] == [
        str(output_root / "biot_savart_desc_export.json")
    ]
    assert validation_manifest["promotion_status"]["state"] == "blocked"


def test_desc_joint_runner_fixed_polish_blocks_optimizer_without_memory_opt_in(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-optimizer-method",
            "fixture-success",
            "--fixed-polish-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(
        (output_root / "desc_fixed_polish_solve_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert result_path == output_root / "desc_result.json"
    assert solve_report["status"] == "failed"
    assert solve_report["allow_high_memory_optimizer"] is False
    assert "blocked by default" in solve_report["reason"]
    assert result_payload["run_configuration"]["optimizer"][
        "allow_high_memory_optimizer"
    ] is False
    assert result_payload["desc_solve_status"]["state"] == "failed"
    assert result_payload["desc_runtime_artifacts"]["desc_coils"] is None
    assert not (output_root / "desc_runtime_coilset_build_report.json").exists()
    assert not (output_root / "desc_objective_assembly_report.json").exists()
    assert not (output_root / "desc_coils.h5").exists()


def test_desc_joint_runner_fixed_polish_export_preserves_canonical_desc_group_order(
    tmp_path,
):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    manifest_payload = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    source_results_path = Path(manifest_payload["candidates"][0]["source_results"])
    source_results = json.loads(source_results_path.read_text(encoding="utf-8"))
    source_results["COIL_GROUPS"] = [
        {"role": "banana", "start": 0, "count": 2},
        {"role": "tf", "start": 2, "count": 1},
    ]
    _write_json(source_results_path, source_results)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-success",
            "--desc-optimizer-verbose",
            "0",
            "--allow-high-memory-desc-optimizer",
            "--fixed-polish-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert Path(completed.stdout.strip()) == output_root / "desc_result.json"
    import_report = json.loads(
        (output_root / "desc_coil_import_report.json").read_text(encoding="utf-8")
    )
    export_report = json.loads(
        (output_root / "desc_runtime_coilset_build_report.json").read_text(
            encoding="utf-8"
        )
    )["export_report"]
    assert export_report["group_order"] == ["tf", "banana"]
    assert import_report["group_order"] == ["tf", "banana"]
    assert [entry["group"] for entry in import_report["entries"]] == [
        "tf",
        "banana",
        "banana",
    ]
    assert [entry["current_A"] for entry in import_report["entries"]] == [
        0.0,
        -8.0e4,
        1.2e4,
    ]


def test_desc_joint_runner_fixed_polish_only_records_optimizer_failure(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-fail",
            "--desc-maxiter",
            "5",
            "--desc-optimizer-verbose",
            "0",
            "--allow-high-memory-desc-optimizer",
            "--fixed-polish-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(
        (output_root / "desc_fixed_polish_solve_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert result_path == output_root / "desc_result.json"
    assert not (output_root / "desc_coils.h5").exists()
    assert result_payload["desc_solve_status"]["state"] == "failed"
    assert result_payload["desc_solve_status"]["artifact_paths"] == []
    assert result_payload["desc_runtime_artifacts"][
        "failed_optimizer_checkpoint_desc_coils"
    ] == str(output_root / "desc_failed_optimizer_coils.h5")
    assert result_payload["artifact_hardware_status"]["state"] == "blocked"
    assert result_payload["artifact_hardware_status"]["artifact_paths"] == []
    assert result_payload["physics_validation_status"]["state"] == "blocked"
    assert result_payload["promotion_status"]["state"] == "blocked"
    assert result_payload["desc_optimizer_result"]["success"] is False
    assert solve_report["status"] == "failed"
    assert solve_report["optimizer_message"] == "fixture requested failure"
    assert solve_report["failed_optimizer_coilset_checkpoint_path"] == str(
        output_root / "desc_failed_optimizer_coils.h5"
    )
    assert (output_root / "desc_failed_optimizer_coils.h5").is_file()


def test_desc_joint_runner_fixed_polish_pre_writes_failed_result_contract(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-interrupt",
            "--desc-maxiter",
            "5",
            "--desc-optimizer-verbose",
            "0",
            "--allow-high-memory-desc-optimizer",
            "--fixed-polish-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    result_path = output_root / "desc_result.json"
    solve_report_path = output_root / "desc_fixed_polish_solve_report.json"
    assert result_path.is_file()
    assert solve_report_path.is_file()
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(solve_report_path.read_text(encoding="utf-8"))

    assert solve_report["status"] == "failed"
    assert solve_report["optimizer_success"] is None
    assert (
        "optimizer execution has started but has not returned yet"
        in solve_report["reason"]
    )
    assert solve_report["optimized_coilset_path"] is None
    assert solve_report["failed_optimizer_coilset_checkpoint_path"] is None
    assert result_payload["desc_solve_status"]["state"] == "failed"
    assert result_payload["desc_solve_status"]["artifact_paths"] == []
    assert result_payload["desc_runtime_artifacts"]["fixed_polish_solve_report"] == str(
        solve_report_path
    )
    assert result_payload["desc_runtime_artifacts"]["desc_coils"] is None
    assert result_payload["desc_runtime_artifacts"]["exported_biot_savart"] is None
    assert result_payload["artifact_hardware_status"]["state"] == "blocked"
    assert result_payload["physics_validation_status"]["state"] == "blocked"
    assert result_payload["promotion_status"]["state"] == "blocked"
    assert result_payload["desc_optimizer_result"]["success"] is None
    assert not (output_root / "desc_coils.h5").exists()
    assert not (output_root / "biot_savart_desc_export.json").exists()


def test_desc_joint_runner_fixed_polish_setup_failure_writes_result_contract(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        source_stellarator_symmetry=None,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-success",
            "--allow-high-memory-desc-optimizer",
            "--fixed-polish-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(
        (output_root / "desc_fixed_polish_solve_report.json").read_text(
            encoding="utf-8"
        )
    )
    setup_report = json.loads(
        (output_root / "desc_runtime_coilset_build_report.json").read_text(
            encoding="utf-8"
        )
    )
    validation_manifest = json.loads(
        (output_root / "desc_joint_validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert result_path == output_root / "desc_result.json"
    assert setup_report["status"] == "failed"
    assert "explicit seed source stellarator_symmetry" in setup_report["reason"]
    assert solve_report["status"] == "failed"
    assert "setup failed before optimizer execution" in solve_report["reason"]
    assert solve_report["optimizer_success"] is None
    assert solve_report["objective_function_type"] is None
    assert not (output_root / "desc_coils.h5").exists()
    assert result_payload["desc_solve_status"]["state"] == "failed"
    assert result_payload["desc_runtime_artifacts"]["setup_failure_report"] == str(
        output_root / "desc_runtime_coilset_build_report.json"
    )
    assert result_payload["desc_runtime_artifacts"]["desc_coils"] is None
    assert validation_manifest["promotion_status"]["state"] == "blocked"


def test_desc_joint_runner_joint_run_only_writes_desc_result_and_artifacts(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-success",
            "--desc-maxiter",
            "5",
            "--desc-optimizer-verbose",
            "0",
            "--allow-high-memory-desc-optimizer",
            "--joint-run-only",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(
        (output_root / "desc_joint_runtime_solve_report.json").read_text(
            encoding="utf-8"
        )
    )
    load_report = json.loads(
        (output_root / "desc_equilibrium_load_report.json").read_text(
            encoding="utf-8"
        )
    )
    validation_manifest = json.loads(
        (output_root / "desc_joint_validation_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert result_path == output_root / "desc_result.json"
    assert result_payload["run_mode"] == "vacuum_joint"
    assert "QuadraticFlux" not in result_payload["objective_stack"]
    assert load_report["mode_profile_adjustment"] == {
        "mode": "vacuum_joint",
        "action": "zeroed_pressure_current",
        "reason": (
            "vacuum_joint uses DESC VacuumBoundaryError, so loaded DESC "
            "equilibrium pressure and toroidal-current profiles are set to zero "
            "before objective assembly."
        ),
        "before_max_abs": {
            "pressure": 22098.0,
            "current": 3688.0,
        },
        "after_max_abs": {
            "pressure": 0.0,
            "current": 0.0,
        },
    }
    _assert_legacy_hardware_fields_fail_closed(result_payload)
    run_configuration = result_payload["run_configuration"]
    assert run_configuration["lanes"]["joint_run_only"] is True
    assert run_configuration["desc_runtime"]["desc_grid_n"] == 7
    assert run_configuration["desc_runtime"]["desc_bs_chunk_size"] == 10
    assert run_configuration["desc_runtime"]["desc_dist_chunk_size"] == 2
    assert run_configuration["desc_runtime"]["desc_jac_chunk_size"] == 5
    assert run_configuration["conversion"]["conversion_sample_count"] == 16
    run_timings = result_payload["run_timing_seconds"]
    _assert_nonnegative_timing(run_timings, "preflight")
    _assert_nonnegative_timing(run_timings, "equilibrium_load")
    _assert_nonnegative_timing(run_timings, "coilset_build")
    _assert_nonnegative_timing(run_timings, "objective_assembly")
    _assert_nonnegative_timing(run_timings, "optimizer")
    _assert_nonnegative_timing(run_timings, "optimized_simsopt_export")
    _assert_nonnegative_timing(run_timings, "optimized_simsopt_surface_export")
    inventory = _assert_runner_inventory(result_payload, result_path=result_path)
    assert inventory["output_artifacts"]["desc_runtime_desc_equilibrium"] == str(
        output_root / "desc_equilibrium.h5"
    )
    assert inventory["output_artifacts"]["desc_runtime_exported_surface"] == str(
        output_root / "surf_desc_equilibrium_export.json"
    )
    assert (output_root / "desc_equilibrium.h5").is_file()
    assert (output_root / "desc_coils.h5").is_file()
    assert result_payload["desc_solve_status"]["state"] == "passed"
    assert result_payload["desc_solve_status"]["artifact_paths"] == [
        str(output_root / "desc_equilibrium.h5"),
        str(output_root / "desc_coils.h5"),
    ]
    assert result_payload["artifact_hardware_status"]["state"] == "blocked"
    assert result_payload["artifact_hardware_status"]["artifact_paths"] == [
        str(output_root / "biot_savart_desc_export.json")
    ]
    assert result_payload["physics_validation_status"]["state"] == "not_run"
    assert result_payload["physics_validation_status"]["artifact_paths"] == [
        str(output_root / "biot_savart_desc_export.json"),
        str(output_root / "surf_desc_equilibrium_export.json"),
    ]
    assert result_payload["promotion_status"]["state"] == "blocked"
    assert result_payload["desc_runtime_artifacts"]["joint_solve_report"] == str(
        output_root / "desc_joint_runtime_solve_report.json"
    )
    assert result_payload["desc_runtime_artifacts"]["desc_equilibrium"] == str(
        output_root / "desc_equilibrium.h5"
    )
    assert result_payload["desc_runtime_artifacts"]["desc_coils"] == str(
        output_root / "desc_coils.h5"
    )
    assert result_payload["desc_runtime_artifacts"]["exported_biot_savart"] == str(
        output_root / "biot_savart_desc_export.json"
    )
    assert result_payload["desc_runtime_artifacts"]["exported_surface"] == str(
        output_root / "surf_desc_equilibrium_export.json"
    )
    assert (output_root / "biot_savart_desc_export.json").is_file()
    assert (output_root / "surf_desc_equilibrium_export.json").is_file()
    assert (output_root / "desc_coil_import_report.json").is_file()
    assert (output_root / "desc_optimized_simsopt_export_report.json").is_file()
    assert (output_root / "desc_optimized_surface_export_report.json").is_file()
    optimized_export_report = json.loads(
        (output_root / "desc_optimized_simsopt_export_report.json").read_text(
            encoding="utf-8"
        )
    )
    selected_seed_field = Path(result_payload["input_contract"]["selected_seed"]["field"])
    assert optimized_export_report["optimized_coilset_source_path"] == str(
        (output_root / "desc_coils.h5").resolve()
    )
    assert optimized_export_report["artifact_metadata"][
        "source_artifact_checksums"
    ]["seed_field"] == _sha256(selected_seed_field)
    loaded_export = load(str(output_root / "biot_savart_desc_export.json"))
    assert isinstance(loaded_export, BiotSavart)
    assert len(loaded_export.coils) == 3
    loaded_surface = load(str(output_root / "surf_desc_equilibrium_export.json"))
    assert isinstance(loaded_surface, SurfaceXYZTensorFourier)
    optimized_surface_export_report = json.loads(
        (output_root / "desc_optimized_surface_export_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert optimized_surface_export_report["status"] == "passed"
    assert optimized_surface_export_report["optimized_equilibrium_source_path"] == str(
        (output_root / "desc_equilibrium.h5").resolve()
    )
    assert optimized_surface_export_report["exported_surface_path"] == str(
        (output_root / "surf_desc_equilibrium_export.json").resolve()
    )
    assert solve_report["schema_version"] == "desc_joint_runtime_solve_report_v1"
    assert solve_report["status"] == "passed"
    assert solve_report["allow_high_memory_optimizer"] is True
    assert solve_report["optimizer_success"] is True
    assert solve_report["optimized_equilibrium_path"] == str(
        output_root / "desc_equilibrium.h5"
    )
    assert solve_report["optimized_coilset_path"] == str(output_root / "desc_coils.h5")
    assert validation_manifest["exported_artifact_paths"] == [
        str(output_root / "biot_savart_desc_export.json")
    ]
    assert validation_manifest["promotion_status"]["state"] == "blocked"


def test_desc_joint_runner_joint_run_pre_writes_failed_result_contract(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-interrupt",
            "--desc-maxiter",
            "5",
            "--desc-optimizer-verbose",
            "0",
            "--allow-high-memory-desc-optimizer",
            "--joint-run-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    result_path = output_root / "desc_result.json"
    solve_report_path = output_root / "desc_joint_runtime_solve_report.json"
    assert result_path.is_file()
    assert solve_report_path.is_file()
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(solve_report_path.read_text(encoding="utf-8"))

    assert solve_report["status"] == "failed"
    assert solve_report["optimizer_success"] is None
    assert (
        "optimizer execution has started but has not returned yet"
        in solve_report["reason"]
    )
    assert solve_report["optimized_equilibrium_path"] is None
    assert solve_report["optimized_coilset_path"] is None
    assert solve_report["failed_optimizer_equilibrium_checkpoint_path"] is None
    assert solve_report["failed_optimizer_coilset_checkpoint_path"] is None
    assert result_payload["desc_solve_status"]["state"] == "failed"
    assert result_payload["desc_solve_status"]["artifact_paths"] == []
    assert result_payload["desc_runtime_artifacts"]["joint_solve_report"] == str(
        solve_report_path
    )
    assert result_payload["desc_runtime_artifacts"]["desc_equilibrium"] is None
    assert result_payload["desc_runtime_artifacts"]["desc_coils"] is None
    assert result_payload["desc_runtime_artifacts"]["exported_biot_savart"] is None
    assert result_payload["desc_runtime_artifacts"]["exported_surface"] is None
    assert result_payload["artifact_hardware_status"]["state"] == "blocked"
    assert result_payload["physics_validation_status"]["state"] == "blocked"
    assert result_payload["promotion_status"]["state"] == "blocked"
    assert result_payload["desc_optimizer_result"]["success"] is None
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()
    assert not (output_root / "biot_savart_desc_export.json").exists()
    assert not (output_root / "surf_desc_equilibrium_export.json").exists()


def test_desc_joint_runner_joint_run_blocks_optimizer_without_memory_opt_in(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-optimizer-method",
            "fixture-success",
            "--joint-run-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(
        (output_root / "desc_joint_runtime_solve_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert result_path == output_root / "desc_result.json"
    assert solve_report["status"] == "failed"
    assert solve_report["allow_high_memory_optimizer"] is False
    assert "blocked by default" in solve_report["reason"]
    assert result_payload["run_configuration"]["optimizer"][
        "allow_high_memory_optimizer"
    ] is False
    assert result_payload["desc_solve_status"]["state"] == "failed"
    assert result_payload["desc_runtime_artifacts"]["desc_equilibrium"] is None
    assert result_payload["desc_runtime_artifacts"]["desc_coils"] is None
    assert not (output_root / "desc_runtime_coilset_build_report.json").exists()
    assert not (output_root / "desc_objective_assembly_report.json").exists()
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()


def test_desc_joint_runner_joint_run_only_records_optimizer_failure(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        loadable_surface=True,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--desc-grid-n",
            "7",
            "--desc-optimizer-method",
            "fixture-fail",
            "--desc-maxiter",
            "5",
            "--desc-optimizer-verbose",
            "0",
            "--allow-high-memory-desc-optimizer",
            "--joint-run-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    solve_report = json.loads(
        (output_root / "desc_joint_runtime_solve_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert result_path == output_root / "desc_result.json"
    assert not (output_root / "desc_equilibrium.h5").exists()
    assert not (output_root / "desc_coils.h5").exists()
    assert result_payload["desc_solve_status"]["state"] == "failed"
    assert result_payload["desc_solve_status"]["artifact_paths"] == []
    assert result_payload["desc_runtime_artifacts"][
        "failed_optimizer_checkpoint_desc_equilibrium"
    ] == str(output_root / "desc_failed_optimizer_equilibrium.h5")
    assert result_payload["desc_runtime_artifacts"][
        "failed_optimizer_checkpoint_desc_coils"
    ] == str(output_root / "desc_failed_optimizer_coils.h5")
    assert result_payload["artifact_hardware_status"]["state"] == "blocked"
    assert result_payload["artifact_hardware_status"]["artifact_paths"] == []
    assert result_payload["physics_validation_status"]["state"] == "blocked"
    assert result_payload["promotion_status"]["state"] == "blocked"
    assert result_payload["desc_optimizer_result"]["success"] is False
    assert solve_report["status"] == "failed"
    assert solve_report["optimizer_message"] == "fixture requested failure"
    assert solve_report["failed_optimizer_equilibrium_checkpoint_path"] == str(
        output_root / "desc_failed_optimizer_equilibrium.h5"
    )
    assert solve_report["failed_optimizer_coilset_checkpoint_path"] == str(
        output_root / "desc_failed_optimizer_coils.h5"
    )
    assert (output_root / "desc_failed_optimizer_equilibrium.h5").is_file()
    assert (output_root / "desc_failed_optimizer_coils.h5").is_file()


def test_desc_joint_runner_joint_run_only_rejects_fixed_mode(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--joint-run-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "--joint-run-only requires vacuum_joint" in completed.stderr
    assert not (output_root / "desc_result.json").exists()


def test_desc_joint_runner_objective_assembly_rejects_missing_seed_symmetry(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        source_stellarator_symmetry=None,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(
        tmp_path,
        source_kind="desc_h5",
        source_filename="seed_equilibrium.h5",
    )
    fake_desc_root = _fake_desc_source_root(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--desc-source-root",
            str(fake_desc_root),
            "--output-root",
            str(output_root),
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "16",
            "--objective-assembly-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    report_path = Path(completed.stdout.strip())
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_path == output_root / "desc_runtime_coilset_build_report.json"
    assert report["status"] == "failed"
    assert "explicit seed source stellarator_symmetry" in report["reason"]
    assert not (output_root / "desc_result.json").exists()


def test_desc_joint_runner_conversion_only_writes_loadable_artifacts(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)
    output_root = tmp_path / "out"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--output-root",
            str(output_root),
            "--conversion-only",
            "--desc-fourier-order",
            "3",
            "--conversion-sample-count",
            "32",
            "--simsopt-fourier-order",
            "3",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    result_path = Path(completed.stdout.strip())
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert result_path == output_root / "desc_result.json"
    assert result_payload["desc_solve_status"]["state"] == "blocked"
    assert "DESC optimizer has not run" in result_payload["desc_solve_status"]["reason"]
    assert result_payload["artifact_hardware_status"]["state"] == "blocked"
    assert result_payload["physics_validation_status"]["state"] == "not_run"
    assert result_payload["promotion_status"]["state"] == "blocked"
    _assert_legacy_hardware_fields_fail_closed(result_payload)
    run_configuration = result_payload["run_configuration"]
    assert run_configuration["lanes"]["conversion_only"] is True
    assert run_configuration["conversion"] == {
        "desc_fourier_order": 3,
        "conversion_sample_count": 32,
        "simsopt_fourier_order": 3,
    }
    run_timings = result_payload["run_timing_seconds"]
    _assert_nonnegative_timing(run_timings, "preflight")
    _assert_nonnegative_timing(run_timings, "conversion")
    assert run_timings["optimizer"] is None
    inventory = _assert_runner_inventory(result_payload, result_path=result_path)
    assert inventory["output_artifacts"]["conversion_exported_biot_savart"] == str(
        output_root / "biot_savart_desc_export.json"
    )

    conversion_artifacts = result_payload["conversion_artifacts"]
    assert isinstance(conversion_artifacts, dict)
    desc_coils_path = Path(conversion_artifacts["desc_coils"])
    export_report_path = Path(conversion_artifacts["export_report"])
    exported_biot_savart_path = Path(conversion_artifacts["exported_biot_savart"])
    import_report_path = Path(conversion_artifacts["import_report"])
    assert desc_coils_path.is_file()
    assert export_report_path.is_file()
    assert exported_biot_savart_path.is_file()
    assert import_report_path.is_file()

    loaded_export = load(str(exported_biot_savart_path))
    assert isinstance(loaded_export, BiotSavart)
    assert [coil.current.get_value() for coil in loaded_export.coils] == [
        -8.0e4,
        1.2e4,
        0.0,
    ]
    source_field = load(str(tmp_path / "biot_savart_opt.json"))
    assert isinstance(source_field, BiotSavart)
    probe_points = [
        [0.78, 0.0, 0.0],
        [0.84, 0.02, 0.01],
        [0.91, -0.02, -0.01],
        [0.95, 0.03, 0.0],
    ]
    source_field.set_points(probe_points)
    loaded_export.set_points(probe_points)
    np.testing.assert_allclose(
        loaded_export.B(),
        source_field.B(),
        atol=1.0e-12,
        rtol=0.0,
    )

    desc_coils_payload = json.loads(desc_coils_path.read_text(encoding="utf-8"))
    assert desc_coils_payload["schema_version"] == (
        "desc_joint_conversion_only_coils_v1"
    )
    assert desc_coils_payload["group_counts"] == {"banana": 2, "tf": 1}
    assert [coil["group"] for coil in desc_coils_payload["coils"]] == [
        "tf",
        "banana",
        "banana",
    ]
    expected_source_identity = {
        "coil_names": ["Coil1", "Coil2", "Coil3"],
        "coil_group_manifest": [
            {"role": "tf", "start": 0, "count": 1},
            {"role": "banana", "start": 1, "count": 2},
        ],
        "nfp": 5,
        "stellarator_symmetry": True,
        "banana_pack_metadata": {
            "finite_build_enabled": True,
            "filaments_per_banana": 2,
            "numfilaments_n": 1,
            "numfilaments_b": 2,
        },
    }
    assert desc_coils_payload["artifact_metadata"]["source_identity"] == (
        expected_source_identity
    )

    export_report = json.loads(export_report_path.read_text(encoding="utf-8"))
    assert export_report["artifact_metadata"]["source_artifact_checksums"].keys() >= {
        "seed_field",
        "seed_surface",
        "seed_source_results",
    }
    assert export_report["artifact_metadata"]["source_identity"] == (
        expected_source_identity
    )
    assert (
        export_report["artifact_metadata"]["conversion_residuals"][
            "max_field_sample_delta_T"
        ]
        <= 1.0e-12
    )

    validation_manifest_path = output_root / "desc_joint_validation_manifest.json"
    validation_manifest = json.loads(
        validation_manifest_path.read_text(encoding="utf-8")
    )
    assert validation_manifest["physics_validation_status"]["passed"] is None
    assert validation_manifest["artifact_hardware_status"]["passed"] is None
    assert validation_manifest["promotion_status"]["state"] == "blocked"
    assert "DESC optimization" in validation_manifest["promotion_status"]["reason"]
    assert (output_root / "desc_joint_validation_report.md").is_file()


def test_desc_joint_runner_rejects_seed_equilibrium_nfp_mismatch(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(
        tmp_path,
        loadable_field=True,
        source_nfp=4,
    )
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "fixed_equilibrium_polish",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--output-root",
            str(tmp_path / "out"),
            "--conversion-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "seed source NFP mismatch" in completed.stderr


def test_desc_joint_runner_conversion_only_rejects_joint_modes(tmp_path):
    hardware_spec_path = _hardware_spec_fixture(tmp_path)
    seed_manifest_path = _seed_manifest_fixture(tmp_path, loadable_field=True)
    equilibrium_seed_path = _equilibrium_seed_fixture(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "run_desc_joint_banana.py"),
            "--mode",
            "vacuum_joint",
            "--hardware-spec",
            str(hardware_spec_path),
            "--seed-manifest",
            str(seed_manifest_path),
            "--seed-label",
            "slidclean_chomp",
            "--equilibrium-seed",
            str(equilibrium_seed_path),
            "--output-root",
            str(tmp_path / "out"),
            "--conversion-only",
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "fixed_equilibrium_polish smoke path" in completed.stderr


def test_simsopt_validation_wrapper_materializes_physics_evidence(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    poincare_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_validation.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    boozer_state_path = _write_boozer_state(
        tmp_path / "surf_desc_export_state.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={"seed": "fixture"},
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )

    artifacts = materialize_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(poincare_metrics_path,),
        boozer_state_paths=(boozer_state_path,),
        require_boozer_state=True,
        output_root=tmp_path / "validation",
    )

    physics_report = json.loads(
        artifacts.physics_report_path.read_text(encoding="utf-8")
    )
    assert physics_report["schema_version"] == (
        DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION
    )
    assert physics_report["passed"] is True
    assert physics_report["exported_artifact_checksums"] == {
        str(exported_artifact_path.resolve()): _sha256(exported_artifact_path),
    }
    assert physics_report["poincare_metrics"][0]["validation_status"] == "validated"
    assert physics_report["boozer_states"][0]["iota"] == 0.091

    manifest = json.loads(
        artifacts.validation_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["physics_validation_status"]["passed"] is True
    assert manifest["physics_validation_status"]["evidence_paths"] == [
        str(artifacts.physics_report_path),
    ]
    assert manifest["artifact_hardware_status"]["passed"] is None
    assert manifest["promotion_status"]["state"] == "blocked"
    assert "hardware validation" in manifest["promotion_status"]["reason"]
    assert artifacts.validation_report_path.is_file()


def test_simsopt_validation_wrapper_binds_joint_surface_to_exported_surface(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    optimized_surface_path = tmp_path / "surf_desc_equilibrium_export.json"
    wrong_surface_path = tmp_path / "seed_surface.json"
    desc_equilibrium_path = tmp_path / "desc_equilibrium.h5"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    optimized_surface_path.write_text('{"surface": "optimized"}\n', encoding="utf-8")
    wrong_surface_path.write_text('{"surface": "seed"}\n', encoding="utf-8")
    desc_equilibrium_path.write_text("optimized desc equilibrium\n", encoding="utf-8")
    poincare_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_validation.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    wrong_surface_boozer_state_path = _write_json(
        tmp_path / "wrong_surface_boozer_state.json",
        {
            "schema_version": 1,
            "surface_path": str(wrong_surface_path),
            "iota": 0.091,
            "G": -2.01062,
            **_export_binding_payload((exported_artifact_path,)),
        },
    )
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="vacuum_joint",
            input_contract={"selected_seed": {"surface": str(wrong_surface_path)}},
            objective_stack=[
                "VacuumBoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    result_payload["desc_runtime_artifacts"] = {
        "desc_equilibrium": str(desc_equilibrium_path),
        "exported_surface": str(optimized_surface_path),
    }

    artifacts = materialize_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(poincare_metrics_path,),
        output_root=tmp_path / "joint_validation",
    )
    physics_report = json.loads(
        artifacts.physics_report_path.read_text(encoding="utf-8")
    )
    assert physics_report["validated_surface_path"] == str(
        optimized_surface_path.resolve()
    )
    assert physics_report["joint_equilibrium_artifact_path"] == str(
        desc_equilibrium_path.resolve()
    )

    with pytest.raises(ValueError, match="validated_surface_path must match"):
        materialize_desc_joint_simsopt_validation(
            result_payload=result_payload,
            exported_artifact_paths=(exported_artifact_path,),
            poincare_metrics_paths=(poincare_metrics_path,),
            validated_surface_path=wrong_surface_path,
            output_root=tmp_path / "joint_validation_wrong_explicit",
        )

    with pytest.raises(ValueError, match="Boozer state surface_path must match"):
        materialize_desc_joint_simsopt_validation(
            result_payload=result_payload,
            exported_artifact_paths=(exported_artifact_path,),
            poincare_metrics_paths=(poincare_metrics_path,),
            boozer_state_paths=(wrong_surface_boozer_state_path,),
            output_root=tmp_path / "joint_validation_wrong_boozer",
        )


def test_simsopt_validation_wrapper_records_failed_boozer_evidence(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    poincare_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_validation.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    failed_boozer_state_path = _write_json(
        tmp_path / "surf_desc_export_state_failed.json",
        {
            "schema_version": 1,
            "surface_path": "surf_desc_export_boozer_surface.json",
            "passed": False,
            "reason": "fixture Boozer failure",
            **_export_binding_payload((exported_artifact_path,)),
        },
    )
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={"seed": "fixture"},
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )

    artifacts = materialize_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(poincare_metrics_path,),
        boozer_state_paths=(failed_boozer_state_path,),
        require_boozer_state=True,
        output_root=tmp_path / "validation",
    )

    physics_report = json.loads(
        artifacts.physics_report_path.read_text(encoding="utf-8")
    )
    assert physics_report["passed"] is False
    assert "Boozer state validation failed" in physics_report["reason"]
    assert physics_report["boozer_states"][0]["passed"] is False
    assert physics_report["boozer_states"][0]["reason"] == "fixture Boozer failure"
    manifest = json.loads(
        artifacts.validation_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["physics_validation_status"]["passed"] is False
    assert manifest["promotion_status"]["state"] == "failed"


def test_validate_desc_joint_export_cli_consumes_sidecar_artifacts(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    poincare_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_validation.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={"seed": "fixture"},
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )
    result_path = _write_json(tmp_path / "desc_result.json", result_payload)
    output_root = tmp_path / "validation_cli"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "validate_desc_joint_export.py"),
            "--result",
            str(result_path),
            "--exported-artifact",
            str(exported_artifact_path),
            "--poincare-metrics",
            str(poincare_metrics_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    manifest_path = Path(completed.stdout.strip())
    assert manifest_path == output_root / "desc_joint_validation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["physics_validation_status"]["passed"] is True
    assert manifest["artifact_hardware_status"]["passed"] is None
    assert manifest["search_hardware_status"]["passed"] is None
    assert manifest["final_oracle_status"]["passed"] is False
    assert manifest["promotion_status"]["state"] == "blocked"
    assert (output_root / "desc_joint_simsopt_physics_validation.json").is_file()
    assert (output_root / "desc_joint_validation_report.md").is_file()

    unsafe_flag = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "validate_desc_joint_export.py"),
            "--result",
            str(result_path),
            "--exported-artifact",
            str(exported_artifact_path),
            "--poincare-metrics",
            str(poincare_metrics_path),
            "--output-root",
            str(tmp_path / "unsafe_cli"),
            "--final-oracle-passed",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert unsafe_flag.returncode != 0
    assert "unrecognized arguments" in unsafe_flag.stderr


def test_desc_joint_validation_launcher_dry_run_records_commands_without_claims(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    surface_path = tmp_path / "surf_opt_boozer_surface.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    surface_path.write_text('{"surface": true}\n', encoding="utf-8")
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={
                "selected_seed": {
                    "surface": str(surface_path),
                    "state": None,
                }
            },
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )
    result_payload["artifact_hardware_status"]["artifact_paths"] = [
        str(exported_artifact_path)
    ]

    artifacts = launch_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=infer_desc_joint_exported_artifact_paths(
            result_payload
        ),
        output_root=tmp_path / "validation_launch",
        dry_run=True,
    )

    assert artifacts.physics_artifacts is None
    launch_report = json.loads(
        artifacts.launch_report_path.read_text(encoding="utf-8")
    )
    assert launch_report["schema_version"] == (
        DESC_JOINT_SIMSOPT_VALIDATION_LAUNCH_SCHEMA_VERSION
    )
    assert launch_report["status"] == "prepared"
    assert launch_report["validation_manifest_path"] is None
    assert launch_report["poincare"]["env_overrides"]["POINCARE_RENDER_MODES"] == (
        "validation"
    )
    assert Path(launch_report["prepared_inputs"]["biot_savart_opt"]).is_file()
    assert Path(launch_report["prepared_inputs"]["surf_opt"]).is_file()


def test_desc_joint_validation_launcher_binds_joint_surface_to_exported_surface(
    tmp_path,
):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    seed_surface_path = tmp_path / "seed_surface.json"
    validation_surface_path = tmp_path / "validated_surface.json"
    wrong_surface_path = tmp_path / "wrong_surface.json"
    desc_equilibrium_path = tmp_path / "desc_equilibrium.h5"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    seed_surface_path.write_text('{"surface": "seed"}\n', encoding="utf-8")
    validation_surface_path.write_text('{"surface": "validated"}\n', encoding="utf-8")
    wrong_surface_path.write_text('{"surface": "wrong"}\n', encoding="utf-8")
    desc_equilibrium_path.write_text("optimized desc equilibrium\n", encoding="utf-8")
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="vacuum_joint",
            input_contract={
                "selected_seed": {
                    "surface": str(seed_surface_path),
                    "state": None,
                }
            },
            objective_stack=[
                "VacuumBoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    result_payload["artifact_hardware_status"]["artifact_paths"] = [
        str(exported_artifact_path)
    ]
    result_payload["desc_runtime_artifacts"] = {
        "desc_equilibrium": str(desc_equilibrium_path),
        "exported_biot_savart": str(exported_artifact_path),
        "exported_surface": str(validation_surface_path),
    }

    with pytest.raises(ValueError, match="must match"):
        launch_desc_joint_simsopt_validation(
            result_payload=result_payload,
            exported_artifact_paths=infer_desc_joint_exported_artifact_paths(
                result_payload
            ),
            output_root=tmp_path / "validation_launch_wrong_surface",
            surface_path=wrong_surface_path,
            dry_run=True,
        )

    artifacts = launch_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=infer_desc_joint_exported_artifact_paths(
            result_payload
        ),
        output_root=tmp_path / "validation_launch_with_exported_surface",
        dry_run=True,
    )

    launch_report = json.loads(
        artifacts.launch_report_path.read_text(encoding="utf-8")
    )
    assert launch_report["source_surface_path"] == str(
        validation_surface_path.resolve()
    )
    assert launch_report["joint_equilibrium_artifact_path"] == str(
        desc_equilibrium_path.resolve()
    )
    assert launch_report["joint_equilibrium_artifact_sha256"] == _sha256(
        desc_equilibrium_path
    )


def test_desc_joint_validation_launcher_keeps_joint_surface_with_boozer_wrapper(
    tmp_path,
    monkeypatch,
):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    seed_surface_path = tmp_path / "seed_surface.json"
    exported_surface_path = tmp_path / "surf_desc_equilibrium_export.json"
    desc_equilibrium_path = tmp_path / "desc_equilibrium.h5"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    seed_surface_path.write_text('{"surface": "seed"}\n', encoding="utf-8")
    exported_surface_path.write_text('{"surface": "joint"}\n', encoding="utf-8")
    desc_equilibrium_path.write_text("optimized desc equilibrium\n", encoding="utf-8")
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="vacuum_joint",
            input_contract={
                "selected_seed": {
                    "surface": str(seed_surface_path),
                    "state": None,
                }
            },
            objective_stack=[
                "VacuumBoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    result_payload["artifact_hardware_status"]["artifact_paths"] = [
        str(exported_artifact_path)
    ]
    result_payload["desc_runtime_artifacts"] = {
        "desc_equilibrium": str(desc_equilibrium_path),
        "exported_biot_savart": str(exported_artifact_path),
        "exported_surface": str(exported_surface_path),
    }

    def fake_run(
        command,
        *,
        cwd,
        check,
        timeout,
        env,
        text,
        capture_output,
    ):
        metrics_path = Path(env["POINCARE_OUT_DIR"]) / (
            "PoincareMetrics_opt_validation.json"
        )
        _write_poincare_metrics(
            metrics_path,
            exported_artifact_paths=(exported_artifact_path,),
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    def fake_boozer_state(
        *,
        biot_savart_path,
        surface_path,
        output_root,
        exported_artifact_checksums,
        iota_guess,
        G_guess,
    ):
        boozer_wrapper_path = output_root / "surf_desc_export_boozer_surface.json"
        boozer_wrapper_path.write_text(
            '{"surface": "generated boozer wrapper"}\n',
            encoding="utf-8",
        )
        return _write_json(
            output_root / "surf_desc_export_boozer_state.json",
            {
                "schema_version": 1,
                "surface_path": str(boozer_wrapper_path),
                "iota": 0.134,
                "G": -2.01,
                "exported_artifact_paths": list(exported_artifact_checksums),
                "exported_artifact_checksums": dict(exported_artifact_checksums),
                "boozer_validation": {
                    "source": "desc_joint_export_boozer_resolve",
                    "input_surface": str(surface_path),
                },
            },
        )

    monkeypatch.setattr(validation_launcher_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validation_launcher_module,
        "materialize_desc_joint_boozer_validation_state",
        fake_boozer_state,
    )

    artifacts = launch_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=infer_desc_joint_exported_artifact_paths(
            result_payload
        ),
        output_root=tmp_path / "validation_launch",
        iota_guess=0.149,
        G_guess=-2.01,
    )

    assert artifacts.physics_artifacts is not None
    physics_report = json.loads(
        artifacts.physics_artifacts.physics_report_path.read_text(encoding="utf-8")
    )
    assert physics_report["passed"] is True
    assert physics_report["validated_surface_path"] == str(
        exported_surface_path.resolve()
    )
    assert physics_report["boozer_states"][0]["surface_path"] == str(
        (
            tmp_path
            / "validation_launch"
            / "simsopt_validation_run"
            / "surf_desc_export_boozer_surface.json"
        )
    )


def test_desc_joint_validation_launcher_requires_joint_boozer_warm_start(
    tmp_path,
):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    seed_surface_path = tmp_path / "seed_surface.json"
    validation_surface_path = tmp_path / "validated_surface.json"
    desc_equilibrium_path = tmp_path / "desc_equilibrium.h5"
    seed_state_path = tmp_path / "seed_state.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    seed_surface_path.write_text('{"surface": "seed"}\n', encoding="utf-8")
    validation_surface_path.write_text('{"surface": "validated"}\n', encoding="utf-8")
    desc_equilibrium_path.write_text("optimized desc equilibrium\n", encoding="utf-8")
    _write_json(seed_state_path, {"iota": 0.11, "G": -2.01})
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="vacuum_joint",
            input_contract={
                "selected_seed": {
                    "surface": str(seed_surface_path),
                    "state": str(seed_state_path),
                }
            },
            objective_stack=[
                "VacuumBoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    result_payload["artifact_hardware_status"]["artifact_paths"] = [
        str(exported_artifact_path)
    ]
    result_payload["desc_runtime_artifacts"] = {
        "desc_equilibrium": str(desc_equilibrium_path),
        "exported_biot_savart": str(exported_artifact_path),
        "exported_surface": str(validation_surface_path),
    }

    with pytest.raises(ValueError, match="requires explicit --iota and --G"):
        launch_desc_joint_simsopt_validation(
            result_payload=result_payload,
            exported_artifact_paths=infer_desc_joint_exported_artifact_paths(
                result_payload
            ),
            output_root=tmp_path / "validation_launch_seed_state_rejected",
            run_poincare=False,
            run_boozer=True,
        )


def test_desc_joint_validation_launcher_runs_poincare_and_binds_metrics(
    tmp_path,
    monkeypatch,
):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    surface_path = tmp_path / "surf_opt_boozer_surface.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    surface_path.write_text('{"surface": true}\n', encoding="utf-8")
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={
                "selected_seed": {
                    "surface": str(surface_path),
                    "state": None,
                }
            },
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )
    calls: list[dict[str, object]] = []

    def fake_run(
        command,
        *,
        cwd,
        check,
        timeout,
        env,
        text,
        capture_output,
    ):
        calls.append(
            {
                "command": list(command),
                "cwd": cwd,
                "check": check,
                "timeout": timeout,
                "env": dict(env),
                "text": text,
                "capture_output": capture_output,
            }
        )
        metrics_path = Path(env["POINCARE_OUT_DIR"]) / (
            "PoincareMetrics_opt_validation.json"
        )
        _write_json(
            metrics_path,
            {
                "field_label": "opt",
                "render_mode": "validation",
                "validation_status": "validated",
                "design_only_override": False,
                "plot_filename": "PoincarePlot_opt_validation.png",
                "metrics": {
                    "mode": "validation",
                    "nfieldlines": 50,
                    "survived_lines": 49,
                    "validation_status": "validated",
                },
            },
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="poincare ok",
            stderr="",
        )

    monkeypatch.setattr(validation_launcher_module.subprocess, "run", fake_run)

    artifacts = launch_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        output_root=tmp_path / "validation_launch",
        python_executable=Path(".conda-env") / "bin" / "python",
        run_boozer=False,
        require_boozer_state=False,
    )

    assert len(calls) == 1
    assert calls[0]["command"][0] == str(
        (Path(".conda-env") / "bin" / "python").resolve()
    )
    assert calls[0]["env"]["POINCARE_OUT_DIR"] == str(
        tmp_path / "validation_launch" / "simsopt_validation_run"
    )
    assert calls[0]["env"]["POINCARE_RENDER_MODES"] == "validation"
    metrics_path = (
        tmp_path
        / "validation_launch"
        / "simsopt_validation_run"
        / "PoincareMetrics_opt_validation.json"
    )
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["exported_artifact_paths"] == [
        str(exported_artifact_path.resolve())
    ]
    assert metrics_payload["exported_artifact_checksums"] == {
        str(exported_artifact_path.resolve()): _sha256(exported_artifact_path),
    }
    assert artifacts.physics_artifacts is not None
    physics_report = json.loads(
        artifacts.physics_artifacts.physics_report_path.read_text(encoding="utf-8")
    )
    assert physics_report["passed"] is True
    manifest = json.loads(
        artifacts.physics_artifacts.validation_manifest_path.read_text(
            encoding="utf-8"
        )
    )
    assert manifest["physics_validation_status"]["passed"] is True
    assert manifest["artifact_hardware_status"]["passed"] is None
    assert manifest["promotion_status"]["state"] == "blocked"
    launch_report = json.loads(
        artifacts.launch_report_path.read_text(encoding="utf-8")
    )
    assert launch_report["status"] == "completed"
    assert launch_report["poincare_subprocess"]["stdout"] == "poincare ok"


def test_desc_joint_validation_launcher_materializes_failed_boozer_evidence(
    tmp_path,
    monkeypatch,
):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    surface_path = tmp_path / "surf_opt_boozer_surface.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    surface_path.write_text('{"surface": true}\n', encoding="utf-8")
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={
                "selected_seed": {
                    "surface": str(surface_path),
                    "state": None,
                }
            },
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )

    def fake_run(
        command,
        *,
        cwd,
        check,
        timeout,
        env,
        text,
        capture_output,
    ):
        metrics_path = Path(env["POINCARE_OUT_DIR"]) / (
            "PoincareMetrics_opt_validation.json"
        )
        _write_json(
            metrics_path,
            {
                "field_label": "opt",
                "render_mode": "validation",
                "validation_status": "validated",
                "design_only_override": False,
                "plot_filename": "PoincarePlot_opt_validation.png",
                "metrics": {
                    "mode": "validation",
                    "nfieldlines": 50,
                    "survived_lines": 50,
                    "validation_status": "validated",
                },
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    def fake_boozer_state(**kwargs):
        raise RuntimeError("fixture Boozer failed")

    monkeypatch.setattr(validation_launcher_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        validation_launcher_module,
        "materialize_desc_joint_boozer_validation_state",
        fake_boozer_state,
    )

    artifacts = launch_desc_joint_simsopt_validation(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        output_root=tmp_path / "validation_launch",
        require_boozer_state=True,
        iota_guess=0.1,
        G_guess=-2.0,
    )

    assert artifacts.physics_artifacts is not None
    failed_state_path = (
        tmp_path
        / "validation_launch"
        / "simsopt_validation_run"
        / "surf_desc_export_boozer_state_failed.json"
    )
    failed_state = json.loads(failed_state_path.read_text(encoding="utf-8"))
    assert failed_state["passed"] is False
    assert "fixture Boozer failed" in failed_state["reason"]
    physics_report = json.loads(
        artifacts.physics_artifacts.physics_report_path.read_text(encoding="utf-8")
    )
    assert physics_report["passed"] is False
    assert physics_report["boozer_states"][0]["passed"] is False
    launch_report = json.loads(
        artifacts.launch_report_path.read_text(encoding="utf-8")
    )
    assert launch_report["status"] == "completed"
    assert launch_report["validation_manifest_path"] == str(
        artifacts.physics_artifacts.validation_manifest_path
    )


def test_desc_joint_validation_launcher_cli_dry_run(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    surface_path = tmp_path / "surf_opt_boozer_surface.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    surface_path.write_text('{"surface": true}\n', encoding="utf-8")
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={
                "selected_seed": {
                    "surface": str(surface_path),
                    "state": None,
                }
            },
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )
    result_payload["artifact_hardware_status"]["artifact_paths"] = [
        str(exported_artifact_path)
    ]
    result_path = _write_json(tmp_path / "desc_result.json", result_payload)
    output_root = tmp_path / "validation_cli_launch"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "launch_desc_joint_validation.py"),
            "--result",
            str(result_path),
            "--output-root",
            str(output_root),
            "--poincare-render-mode",
            "validation",
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    launch_report_path = Path(completed.stdout.strip())
    assert launch_report_path == (
        output_root / "desc_joint_simsopt_validation_launch_report.json"
    )
    launch_report = json.loads(launch_report_path.read_text(encoding="utf-8"))
    assert launch_report["status"] == "prepared"
    assert launch_report["validation_manifest_path"] is None
    assert launch_report["poincare"]["env_overrides"]["POINCARE_RENDER_MODES"] == (
        "validation"
    )


def _write_physics_report(
    path: Path,
    *,
    exported_artifact_path: Path,
    passed: bool = True,
    validated_surface_path: Path | None = None,
    joint_equilibrium_artifact_path: Path | None = None,
) -> Path:
    return _write_json(
        path,
        {
            "schema_version": DESC_JOINT_SIMSOPT_PHYSICS_VALIDATION_SCHEMA_VERSION,
            "source": "simsopt_boozer_poincare_sidecars",
            "passed": passed,
            "reason": "fixture physics validation",
            "exported_artifact_paths": [str(exported_artifact_path.resolve())],
            "exported_artifact_checksums": {
                str(exported_artifact_path.resolve()): _sha256(exported_artifact_path),
            },
            "validated_surface_path": (
                None
                if validated_surface_path is None
                else str(validated_surface_path.resolve())
            ),
            "validated_surface_sha256": (
                None
                if validated_surface_path is None
                else _sha256(validated_surface_path)
            ),
            "joint_equilibrium_artifact_path": (
                None
                if joint_equilibrium_artifact_path is None
                else str(joint_equilibrium_artifact_path.resolve())
            ),
            "joint_equilibrium_artifact_sha256": (
                None
                if joint_equilibrium_artifact_path is None
                else _sha256(joint_equilibrium_artifact_path)
            ),
            "poincare_metrics": [],
            "boozer_states": [],
            "require_boozer_state": False,
        },
    )


def _result_payload_with_export_report(
    tmp_path: Path,
    *,
    exported_artifact_path: Path,
    source_checksums: dict[str, str] | None = None,
) -> dict[str, object]:
    checksums = {"seed_field": "a" * 64} if source_checksums is None else source_checksums
    export_report_path = _write_json(
        tmp_path / "desc_optimized_simsopt_export_report.json",
        {
            "artifact_metadata": {
                "source_artifact_checksums": checksums,
            }
        },
    )
    payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={"selected_seed": {"surface": "fixture"}},
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )
    payload["artifact_hardware_status"]["artifact_paths"] = [
        str(exported_artifact_path)
    ]
    payload["desc_runtime_artifacts"] = {
        "optimized_simsopt_export_report": str(export_report_path),
    }
    return payload


def test_desc_joint_hardware_oracle_launcher_dry_run_records_command(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    oracle_source_path = tmp_path / "surf_desc_export_boozer_surface.json"
    audit_script_path = tmp_path / "audit_hardware_contacts.py"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    oracle_source_path.write_text('{"surface": true}\n', encoding="utf-8")
    audit_script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    result_payload = _result_payload_with_export_report(
        tmp_path,
        exported_artifact_path=exported_artifact_path,
    )

    artifacts = launch_desc_joint_hardware_oracle(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        oracle_source_artifact_path=oracle_source_path,
        output_root=tmp_path / "hardware_oracle",
        audit_script_path=audit_script_path,
        dry_run=True,
    )

    assert artifacts.final_oracle_evidence_path is None
    launch_report = json.loads(
        artifacts.launch_report_path.read_text(encoding="utf-8")
    )
    assert launch_report["schema_version"] == (
        DESC_JOINT_HARDWARE_ORACLE_LAUNCH_SCHEMA_VERSION
    )
    assert launch_report["status"] == "prepared"
    assert launch_report["validation_manifest_path"] is None
    assert launch_report["command"][-2:] == ["--artifact", str(oracle_source_path.resolve())]


def test_desc_joint_hardware_oracle_launcher_materializes_passing_evidence(
    tmp_path,
    monkeypatch,
):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    oracle_source_path = tmp_path / "surf_desc_export_boozer_surface.json"
    audit_script_path = tmp_path / "audit_hardware_contacts.py"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    oracle_source_path.write_text('{"surface": true}\n', encoding="utf-8")
    audit_script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    result_payload = _result_payload_with_export_report(
        tmp_path,
        exported_artifact_path=exported_artifact_path,
    )
    physics_report_path = _write_physics_report(
        tmp_path / "desc_joint_simsopt_physics_validation.json",
        exported_artifact_path=exported_artifact_path,
    )
    calls: list[dict[str, object]] = []

    def fake_run(
        command,
        *,
        cwd,
        check,
        timeout,
        text,
        capture_output,
    ):
        calls.append(
            {
                "command": list(command),
                "cwd": cwd,
                "check": check,
                "timeout": timeout,
                "text": text,
                "capture_output": capture_output,
            }
        )
        _write_json(
            oracle_source_path.parent / "hardware_contact_audit.json",
            {
                "schema_version": 1,
                "source_artifact": oracle_source_path.name,
                "source_sha256": _sha256(oracle_source_path),
                "viewer_artifact": "surf_desc_export_boozer_surface.viewer.json",
                "viewer_sha256": "b" * 64,
                "checker": "fixture hardware_contact_report",
                "hardware_groups": [
                    "vessel",
                    "shells",
                    "sensors",
                    "solenoid",
                    "remc",
                    "frame",
                    "sample",
                    "limiter",
                    "quartz-spools-generic",
                    "quartz-spools-thick",
                ],
                "vessel_shell_included": True,
                "total_coils": 3,
                "hits": 0,
                "contacts": [],
            },
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="hardware contacts: 0/3 CLEAR",
            stderr="",
        )

    monkeypatch.setattr(hardware_oracle_module.subprocess, "run", fake_run)

    artifacts = launch_desc_joint_hardware_oracle(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        oracle_source_artifact_path=oracle_source_path,
        output_root=tmp_path / "hardware_oracle",
        physics_report_path=physics_report_path,
        audit_script_path=audit_script_path,
    )

    assert len(calls) == 1
    assert calls[0]["check"] is False
    assert artifacts.final_oracle_evidence_path is not None
    evidence = json.loads(
        artifacts.final_oracle_evidence_path.read_text(encoding="utf-8")
    )
    assert evidence["schema_version"] == (
        DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION
    )
    assert evidence["source"] == "direct_loaded_artifact_hardware_contact_oracle"
    assert evidence["passed"] is True
    assert evidence["exported_artifact_paths"] == [
        str(exported_artifact_path.resolve())
    ]
    assert evidence["exported_artifact_checksums"] == {
        str(exported_artifact_path.resolve()): _sha256(exported_artifact_path),
    }
    manifest = json.loads(
        artifacts.validation_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["physics_validation_status"]["passed"] is True
    assert manifest["artifact_hardware_status"]["passed"] is True
    assert manifest["final_oracle_status"]["passed"] is True
    assert manifest["promotion_status"]["state"] == "passed"
    launch_report = json.loads(
        artifacts.launch_report_path.read_text(encoding="utf-8")
    )
    assert launch_report["status"] == "passed"


def test_desc_joint_hardware_oracle_launcher_requires_joint_surface_binding(
    tmp_path,
    monkeypatch,
):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    oracle_source_path = tmp_path / "surf_desc_export_boozer_surface.json"
    other_surface_path = tmp_path / "other_surface.json"
    desc_equilibrium_path = tmp_path / "desc_equilibrium.h5"
    audit_script_path = tmp_path / "audit_hardware_contacts.py"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    oracle_source_path.write_text('{"surface": true}\n', encoding="utf-8")
    other_surface_path.write_text('{"surface": "other"}\n', encoding="utf-8")
    desc_equilibrium_path.write_text("optimized desc equilibrium\n", encoding="utf-8")
    audit_script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    result_payload = _result_payload_with_export_report(
        tmp_path,
        exported_artifact_path=exported_artifact_path,
    )
    result_payload["run_mode"] = "vacuum_joint"
    result_payload["objective_stack"] = [
        "VacuumBoundaryError",
        "LinkingCurrentConsistency",
        VOLUME_OBJECTIVE,
    ]
    result_payload["desc_runtime_artifacts"]["desc_equilibrium"] = str(
        desc_equilibrium_path
    )
    fixed_polish_predecessor_path = tmp_path / "fixed_polish_validation_manifest.json"
    source_checksums = {"seed_field": "a" * 64}
    _write_fixed_polish_predecessor_manifest(
        fixed_polish_predecessor_path,
        source_artifact_checksums=source_checksums,
    )
    result_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(fixed_polish_predecessor_path)],
    }
    mismatched_physics_report_path = _write_physics_report(
        tmp_path / "mismatched_physics.json",
        exported_artifact_path=exported_artifact_path,
        validated_surface_path=other_surface_path,
        joint_equilibrium_artifact_path=desc_equilibrium_path,
    )

    with pytest.raises(ValueError, match="oracle source must match"):
        launch_desc_joint_hardware_oracle(
            result_payload=result_payload,
            exported_artifact_paths=(exported_artifact_path,),
            oracle_source_artifact_path=oracle_source_path,
            output_root=tmp_path / "joint_hardware_oracle_mismatch",
            physics_report_path=mismatched_physics_report_path,
            audit_script_path=audit_script_path,
            dry_run=True,
        )

    physics_report_path = _write_physics_report(
        tmp_path / "joint_physics.json",
        exported_artifact_path=exported_artifact_path,
        validated_surface_path=oracle_source_path,
        joint_equilibrium_artifact_path=desc_equilibrium_path,
    )

    def fake_run(
        command,
        *,
        cwd,
        check,
        timeout,
        text,
        capture_output,
    ):
        _write_json(
            oracle_source_path.parent / "hardware_contact_audit.json",
            {
                "schema_version": 1,
                "source_artifact": oracle_source_path.name,
                "source_sha256": _sha256(oracle_source_path),
                "hits": 0,
                "contacts": [],
            },
        )
        return subprocess.CompletedProcess(command, 0, stdout="CLEAR", stderr="")

    monkeypatch.setattr(hardware_oracle_module.subprocess, "run", fake_run)

    artifacts = launch_desc_joint_hardware_oracle(
        result_payload=result_payload,
        exported_artifact_paths=(exported_artifact_path,),
        oracle_source_artifact_path=oracle_source_path,
        output_root=tmp_path / "joint_hardware_oracle",
        physics_report_path=physics_report_path,
        audit_script_path=audit_script_path,
    )

    assert artifacts.final_oracle_evidence_path is not None
    evidence = json.loads(
        artifacts.final_oracle_evidence_path.read_text(encoding="utf-8")
    )
    assert evidence["joint_equilibrium_artifact_path"] == str(
        desc_equilibrium_path.resolve()
    )
    assert evidence["joint_equilibrium_artifact_sha256"] == _sha256(
        desc_equilibrium_path
    )
    manifest = json.loads(
        artifacts.validation_manifest_path.read_text(encoding="utf-8")
    )
    assert manifest["fixed_polish_predecessor_status"]["passed"] is True
    assert manifest["promotion_status"]["state"] == "passed"


def test_desc_joint_hardware_oracle_launcher_cli_dry_run(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    oracle_source_path = tmp_path / "surf_desc_export_boozer_surface.json"
    audit_script_path = tmp_path / "audit_hardware_contacts.py"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    oracle_source_path.write_text('{"surface": true}\n', encoding="utf-8")
    audit_script_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    result_payload = _result_payload_with_export_report(
        tmp_path,
        exported_artifact_path=exported_artifact_path,
    )
    result_path = _write_json(tmp_path / "desc_result.json", result_payload)
    output_root = tmp_path / "hardware_oracle_cli"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "launch_desc_joint_hardware_oracle.py"),
            "--result",
            str(result_path),
            "--oracle-source-artifact",
            str(oracle_source_path),
            "--output-root",
            str(output_root),
            "--audit-script",
            str(audit_script_path),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    launch_report_path = Path(completed.stdout.strip())
    assert launch_report_path == (
        output_root / "desc_joint_hardware_oracle_launch_report.json"
    )
    launch_report = json.loads(launch_report_path.read_text(encoding="utf-8"))
    assert launch_report["status"] == "prepared"


def test_simsopt_validation_wrapper_fails_closed_on_nonvalidated_poincare(tmp_path):
    exported_artifact_path = tmp_path / "biot_savart_desc_export.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    diagnostic_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_default.json",
        exported_artifact_paths=(exported_artifact_path,),
        validation_status="diagnostic_only",
        nfieldlines=50,
        survived_lines=42,
    )

    physics_report = build_desc_joint_simsopt_physics_report(
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(diagnostic_metrics_path,),
    )

    assert physics_report["passed"] is False
    assert physics_report["reason"] == "No strict Poincare validation sidecar was supplied."
    assert physics_report["poincare_metrics"][0]["passed"] is False

    strict_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_validation.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    strict_with_diagnostic_report = build_desc_joint_simsopt_physics_report(
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(strict_metrics_path, diagnostic_metrics_path),
    )

    assert strict_with_diagnostic_report["passed"] is True
    assert strict_with_diagnostic_report["reason"] == (
        "SIMSOPT Poincare/Boozer validation sidecars passed."
    )
    assert strict_with_diagnostic_report["poincare_metrics"][0]["passed"] is True
    assert strict_with_diagnostic_report["poincare_metrics"][1]["passed"] is False

    design_only_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_design_only.json",
        exported_artifact_paths=(exported_artifact_path,),
        validation_status="validated",
        design_only_override=True,
    )
    design_only_report = build_desc_joint_simsopt_physics_report(
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(design_only_metrics_path,),
    )

    assert design_only_report["passed"] is False
    assert design_only_report["poincare_metrics"][0]["design_only_override"] is True

    other_export_path = tmp_path / "other_biot_savart_desc_export.json"
    other_export_path.write_text('{"field": "other"}\n', encoding="utf-8")
    wrong_path_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_wrong_path.json",
        exported_artifact_paths=(other_export_path,),
    )
    with pytest.raises(ValueError, match="exported_artifact_paths"):
        build_desc_joint_simsopt_physics_report(
            exported_artifact_paths=(exported_artifact_path,),
            poincare_metrics_paths=(wrong_path_metrics_path,),
        )

    wrong_checksum_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_wrong_checksum.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    wrong_checksum_payload = json.loads(
        wrong_checksum_metrics_path.read_text(encoding="utf-8")
    )
    wrong_checksum_payload["exported_artifact_checksums"] = {
        str(exported_artifact_path.resolve()): "0" * 64,
    }
    _write_json(wrong_checksum_metrics_path, wrong_checksum_payload)
    with pytest.raises(ValueError, match="exported_artifact_checksums"):
        build_desc_joint_simsopt_physics_report(
            exported_artifact_paths=(exported_artifact_path,),
            poincare_metrics_paths=(wrong_checksum_metrics_path,),
        )

    with pytest.raises(ValueError, match="existing file"):
        build_desc_joint_simsopt_physics_report(
            exported_artifact_paths=(tmp_path / "missing_biot_savart.json",),
            poincare_metrics_paths=(diagnostic_metrics_path,),
        )
    with pytest.raises(ValueError, match="sequence of paths"):
        build_desc_joint_simsopt_physics_report(
            exported_artifact_paths=exported_artifact_path,
            poincare_metrics_paths=(diagnostic_metrics_path,),
        )
    with pytest.raises(ValueError, match="at least one exported SIMSOPT artifact"):
        build_desc_joint_simsopt_physics_report(
            exported_artifact_paths=(),
            poincare_metrics_paths=(diagnostic_metrics_path,),
        )

    null_metrics_path = _write_poincare_metrics(
        tmp_path / "PoincareMetrics_desc_export_null_metrics.json",
        exported_artifact_paths=(exported_artifact_path,),
    )
    null_metrics_payload = json.loads(null_metrics_path.read_text(encoding="utf-8"))
    null_metrics_payload["metrics"] = None
    _write_json(null_metrics_path, null_metrics_payload)
    null_metrics_report = build_desc_joint_simsopt_physics_report(
        exported_artifact_paths=(exported_artifact_path,),
        poincare_metrics_paths=(null_metrics_path,),
    )
    assert null_metrics_report["passed"] is True


def test_validation_manifest_keeps_promotion_oracle_separate(tmp_path):
    exported_artifact_path = tmp_path / "exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    source_checksums = {"seed_field": "a" * 64}
    result_payload = build_preflight_result_payload(
        mode="fixed_equilibrium_polish",
        input_contract={"seed": "fixture"},
        objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
    )
    desc_passed_payload = _desc_solve_passed_payload(result_payload)

    manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
    )

    assert manifest["promotion_status"]["state"] == "blocked"
    assert "DESC optimization" in manifest["promotion_status"]["reason"]
    assert manifest["physics_validation_status"]["passed"] is True
    assert manifest["artifact_hardware_status"]["passed"] is True
    report = render_desc_joint_validation_report(manifest)
    assert "promotion_status: blocked" in report
    assert "final_oracle_status:" in report
    assert "exported_artifact_paths:" in report

    missing_oracle_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=False,
        final_oracle_evidence_path="/tmp/desc_joint_missing_oracle_report.json",
    )
    assert missing_oracle_manifest["promotion_status"]["state"] == "blocked"

    oracle_path = tmp_path / "hardware_contact_report.json"
    oracle_path.write_text("{}\n", encoding="utf-8")
    failing_oracle_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=False,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert failing_oracle_manifest["promotion_status"]["state"] == "failed"

    with pytest.raises(ValueError, match="schema_version"):
        build_desc_joint_validation_manifest(
            result_payload=desc_passed_payload,
            exported_artifact_paths=exported_artifact_paths,
            expected_source_artifact_checksums=source_checksums,
            physics_validation_passed=True,
            artifact_hardware_passed=True,
            search_hardware_passed=True,
            final_oracle_passed=True,
            final_oracle_evidence_path=str(oracle_path),
        )
    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )
    blocked_preflight_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert blocked_preflight_manifest["promotion_status"]["state"] == "blocked"
    assert "DESC optimization" in blocked_preflight_manifest["promotion_status"][
        "reason"
    ]

    passing_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert passing_manifest["promotion_status"]["state"] == "passed"

    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums={"seed_field": "b" * 64},
    )
    with pytest.raises(ValueError, match="source_artifact_checksums"):
        build_desc_joint_validation_manifest(
            result_payload=desc_passed_payload,
            exported_artifact_paths=exported_artifact_paths,
            expected_source_artifact_checksums=source_checksums,
            physics_validation_passed=True,
            artifact_hardware_passed=True,
            search_hardware_passed=True,
            final_oracle_passed=True,
            final_oracle_evidence_path=str(oracle_path),
        )

    missing_exported_artifact_path = "/tmp/desc_joint_missing_exported_artifact.json"
    _write_json(
        oracle_path,
        {
            "schema_version": DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
            "source": "direct_loaded_artifact_hardware_contact_oracle",
            "passed": True,
            "exported_artifact_paths": [missing_exported_artifact_path],
            "exported_artifact_checksums": {
                missing_exported_artifact_path: "a" * 64,
            },
            "source_artifact_checksums": source_checksums,
        },
    )
    missing_export_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=(missing_exported_artifact_path,),
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=False,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert missing_export_manifest["promotion_status"]["state"] == "failed"
    with pytest.raises(ValueError, match="existing files"):
        build_desc_joint_validation_manifest(
            result_payload=desc_passed_payload,
            exported_artifact_paths=(missing_exported_artifact_path,),
            expected_source_artifact_checksums=source_checksums,
            physics_validation_passed=True,
            artifact_hardware_passed=True,
            search_hardware_passed=True,
            final_oracle_passed=True,
            final_oracle_evidence_path=str(oracle_path),
        )
    _write_json(
        oracle_path,
        {
            "schema_version": DESC_JOINT_FINAL_ORACLE_EVIDENCE_SCHEMA_VERSION,
            "source": "direct_loaded_artifact_hardware_contact_oracle",
            "passed": True,
            "exported_artifact_paths": [],
            "exported_artifact_checksums": {},
            "source_artifact_checksums": source_checksums,
        },
    )
    with pytest.raises(ValueError, match="at least one exported artifact"):
        build_desc_joint_validation_manifest(
            result_payload=desc_passed_payload,
            exported_artifact_paths=(),
            expected_source_artifact_checksums=source_checksums,
            physics_validation_passed=True,
            artifact_hardware_passed=True,
            search_hardware_passed=True,
            final_oracle_passed=True,
            final_oracle_evidence_path=str(oracle_path),
        )


def test_validation_manifest_blocks_joint_promotion_without_fixed_polish_predecessor(
    tmp_path,
):
    exported_artifact_path = tmp_path / "joint_exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    source_checksums = {"seed_field": "a" * 64}
    result_payload = build_preflight_result_payload(
        mode="vacuum_joint",
        input_contract={"seed": "fixture"},
        objective_stack=[
            "VacuumBoundaryError",
            "LinkingCurrentConsistency",
            VOLUME_OBJECTIVE,
        ],
    )
    desc_passed_payload = _desc_solve_passed_payload(result_payload)
    oracle_path = tmp_path / "hardware_contact_report.json"
    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )

    blocked_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )

    assert blocked_manifest["fixed_polish_predecessor_status"]["passed"] is False
    assert blocked_manifest["promotion_status"]["state"] == "blocked"
    assert "fixed-equilibrium polish predecessor" in blocked_manifest[
        "promotion_status"
    ]["reason"]

    failed_physics_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=False,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert failed_physics_manifest["promotion_status"]["state"] == "failed"
    assert "SIMSOPT physics validation" in failed_physics_manifest[
        "promotion_status"
    ]["reason"]

    malformed_fixed_polish_path = tmp_path / "malformed_fixed_polish.json"
    malformed_fixed_polish_path.write_text('{"passed": true}\n', encoding="utf-8")
    desc_passed_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(malformed_fixed_polish_path)],
    }
    malformed_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert malformed_manifest["fixed_polish_predecessor_status"]["passed"] is False
    assert "invalid fixed-polish predecessor evidence" in malformed_manifest[
        "fixed_polish_predecessor_status"
    ]["reason"]
    assert malformed_manifest["promotion_status"]["state"] == "blocked"

    invalid_physics_evidence_path = tmp_path / "not_physics_evidence.json"
    invalid_physics_evidence_path.write_text('{"passed": true}\n', encoding="utf-8")
    invalid_fixed_export_path = tmp_path / "invalid_fixed_exported_biot_savart.json"
    invalid_fixed_export_path.write_text('{"field": true}\n', encoding="utf-8")
    invalid_fixed_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="fixed_equilibrium_polish",
            input_contract={"seed": "fixture"},
            objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
        )
    )
    invalid_fixed_manifest = build_desc_joint_validation_manifest(
        result_payload=invalid_fixed_payload,
        exported_artifact_paths=(str(invalid_fixed_export_path),),
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=None,
        search_hardware_passed=None,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
        physics_validation_evidence_paths=(str(invalid_physics_evidence_path),),
    )
    invalid_fixed_manifest_path = tmp_path / "invalid_fixed_polish_manifest.json"
    _write_json(invalid_fixed_manifest_path, invalid_fixed_manifest)
    desc_passed_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(invalid_fixed_manifest_path)],
    }
    invalid_evidence_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert invalid_evidence_manifest["fixed_polish_predecessor_status"][
        "passed"
    ] is False
    assert "physics evidence must use schema_version" in invalid_evidence_manifest[
        "fixed_polish_predecessor_status"
    ]["reason"]
    assert invalid_evidence_manifest["promotion_status"]["state"] == "blocked"

    mismatched_fixed_polish_path = tmp_path / "mismatched_fixed_polish.json"
    _write_fixed_polish_predecessor_manifest(
        mismatched_fixed_polish_path,
        source_artifact_checksums={"seed_field": "b" * 64},
    )
    desc_passed_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(mismatched_fixed_polish_path)],
    }
    mismatched_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert mismatched_manifest["fixed_polish_predecessor_status"]["passed"] is False
    assert "source_artifact_checksums" in mismatched_manifest[
        "fixed_polish_predecessor_status"
    ]["reason"]
    assert mismatched_manifest["promotion_status"]["state"] == "blocked"

    fixed_polish_evidence_path = tmp_path / "fixed_polish_validation_manifest.json"
    _write_fixed_polish_predecessor_manifest(
        fixed_polish_evidence_path,
        source_artifact_checksums=source_checksums,
    )
    desc_passed_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(fixed_polish_evidence_path)],
    }
    passing_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )

    assert passing_manifest["fixed_polish_predecessor_status"]["passed"] is True
    assert passing_manifest["fixed_polish_predecessor_status"]["artifact_paths"] == [
        str(fixed_polish_evidence_path.resolve())
    ]
    assert passing_manifest["promotion_status"]["state"] == "passed"


def test_validation_manifest_blocks_finite_beta_without_lane_b_predecessor(
    tmp_path,
):
    exported_artifact_path = tmp_path / "finite_beta_exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    source_checksums = {"seed_field": "a" * 64}
    fixed_polish_evidence_path = tmp_path / "fixed_polish_validation_manifest.json"
    _write_fixed_polish_predecessor_manifest(
        fixed_polish_evidence_path,
        source_artifact_checksums=source_checksums,
    )
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="finite_beta_joint",
            input_contract={"seed": "fixture"},
            objective_stack=[
                "BoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    result_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(fixed_polish_evidence_path)],
    }
    oracle_path = tmp_path / "hardware_contact_report.json"
    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )

    blocked_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )

    assert blocked_manifest["fixed_polish_predecessor_status"]["passed"] is True
    assert blocked_manifest["lane_b_predecessor_status"]["passed"] is False
    assert blocked_manifest["promotion_status"]["state"] == "blocked"
    assert "Lane B vacuum-joint predecessor" in blocked_manifest[
        "promotion_status"
    ]["reason"]


def test_validation_manifest_requires_passed_vacuum_joint_for_finite_beta_lane_b(
    tmp_path,
):
    exported_artifact_path = tmp_path / "finite_beta_exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    source_checksums = {"seed_field": "a" * 64}
    fixed_polish_evidence_path = tmp_path / "fixed_polish_validation_manifest.json"
    _write_fixed_polish_predecessor_manifest(
        fixed_polish_evidence_path,
        source_artifact_checksums=source_checksums,
    )
    wrong_mode_lane_b_path = tmp_path / "wrong_mode_lane_b_manifest.json"
    _write_lane_b_predecessor_manifest(
        wrong_mode_lane_b_path,
        source_artifact_checksums=source_checksums,
        run_mode="finite_beta_joint",
    )
    failed_physics_lane_b_path = tmp_path / "failed_physics_lane_b_manifest.json"
    _write_lane_b_predecessor_manifest(
        failed_physics_lane_b_path,
        source_artifact_checksums=source_checksums,
        physics_passed=False,
    )
    passing_lane_b_path = tmp_path / "lane_b_validation_manifest.json"
    _write_lane_b_predecessor_manifest(
        passing_lane_b_path,
        source_artifact_checksums=source_checksums,
    )
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="finite_beta_joint",
            input_contract={"seed": "fixture"},
            objective_stack=[
                "BoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    result_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(fixed_polish_evidence_path)],
    }
    oracle_path = tmp_path / "hardware_contact_report.json"
    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )

    result_payload["lane_b_predecessor_status"] = {
        "state": "passed",
        "reason": "Lane B predecessor passed strict SIMSOPT validation",
        "artifact_paths": [str(wrong_mode_lane_b_path)],
    }
    wrong_mode_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert wrong_mode_manifest["lane_b_predecessor_status"]["passed"] is False
    assert "vacuum_joint validation manifest" in wrong_mode_manifest[
        "lane_b_predecessor_status"
    ]["reason"]
    assert wrong_mode_manifest["promotion_status"]["state"] == "blocked"

    result_payload["lane_b_predecessor_status"] = {
        "state": "passed",
        "reason": "Lane B predecessor passed strict SIMSOPT validation",
        "artifact_paths": [str(failed_physics_lane_b_path)],
    }
    failed_physics_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    assert failed_physics_manifest["lane_b_predecessor_status"]["passed"] is False
    assert "SIMSOPT round-trip validation" in failed_physics_manifest[
        "lane_b_predecessor_status"
    ]["reason"]
    assert failed_physics_manifest["promotion_status"]["state"] == "blocked"

    result_payload["lane_b_predecessor_status"] = {
        "state": "passed",
        "reason": "Lane B predecessor passed strict SIMSOPT validation",
        "artifact_paths": [str(passing_lane_b_path)],
    }
    passing_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )

    assert passing_manifest["lane_b_predecessor_status"]["passed"] is True
    assert passing_manifest["lane_b_predecessor_status"]["artifact_paths"] == [
        str(passing_lane_b_path.resolve())
    ]
    assert passing_manifest["promotion_status"]["state"] == "passed"


def test_desc_joint_outer_loop_gate_accepts_only_oracle_backed_joint_candidates(
    tmp_path,
):
    exported_artifact_path = tmp_path / "joint_exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    source_checksums = {"seed_field": "a" * 64}
    fixed_polish_evidence_path = tmp_path / "fixed_polish_validation_manifest.json"
    _write_fixed_polish_predecessor_manifest(
        fixed_polish_evidence_path,
        source_artifact_checksums=source_checksums,
    )
    result_payload = build_preflight_result_payload(
        mode="vacuum_joint",
        input_contract={"seed": "fixture"},
        objective_stack=[
            "VacuumBoundaryError",
            "LinkingCurrentConsistency",
            VOLUME_OBJECTIVE,
        ],
    )
    desc_passed_payload = _desc_solve_passed_payload(result_payload)
    _record_result_exported_artifact_paths(
        desc_passed_payload,
        exported_artifact_paths,
    )
    desc_passed_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(fixed_polish_evidence_path)],
    }
    oracle_path = tmp_path / "hardware_contact_report.json"
    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )
    passing_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    passing_manifest_path = _write_json(
        tmp_path / "desc_joint_validation_manifest.json",
        passing_manifest,
    )

    accepted = materialize_desc_joint_outer_loop_decision(
        result_payload=desc_passed_payload,
        validation_manifest=passing_manifest,
        output_root=tmp_path / "outer_loop_accept",
        validation_manifest_path=passing_manifest_path,
    )

    assert accepted.decision_path == (
        tmp_path / "outer_loop_accept" / "desc_joint_outer_loop_decision.json"
    )
    accepted_payload = json.loads(
        accepted.decision_path.read_text(encoding="utf-8")
    )
    assert accepted_payload["schema_version"] == (
        DESC_JOINT_OUTER_LOOP_DECISION_SCHEMA_VERSION
    )
    assert accepted_payload["decision"] == "accepted"
    assert accepted_payload["eligible_for_next_search_stage"] is True
    assert accepted_payload["eligible_for_promotion"] is True
    assert accepted_payload["rejection_stage"] is None
    assert accepted_payload["validation_manifest_path"] == str(
        passing_manifest_path.resolve()
    )
    assert accepted_payload["result_exported_artifact_paths"] == [
        str(exported_artifact_path)
    ]

    lane_b_evidence_path = tmp_path / "lane_b_validation_manifest.json"
    _write_lane_b_predecessor_manifest(
        lane_b_evidence_path,
        source_artifact_checksums=source_checksums,
    )
    finite_beta_payload = build_preflight_result_payload(
        mode="finite_beta_joint",
        input_contract={"seed": "fixture"},
        objective_stack=[
            "BoundaryError",
            "LinkingCurrentConsistency",
            VOLUME_OBJECTIVE,
        ],
    )
    finite_beta_passed_payload = _desc_solve_passed_payload(finite_beta_payload)
    _record_result_exported_artifact_paths(
        finite_beta_passed_payload,
        exported_artifact_paths,
    )
    finite_beta_passed_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(fixed_polish_evidence_path)],
    }
    finite_beta_passed_payload["lane_b_predecessor_status"] = {
        "state": "passed",
        "reason": "Lane B predecessor passed strict SIMSOPT validation",
        "artifact_paths": [str(lane_b_evidence_path)],
    }
    finite_beta_manifest = build_desc_joint_validation_manifest(
        result_payload=finite_beta_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    finite_beta_accepted = materialize_desc_joint_outer_loop_decision(
        result_payload=finite_beta_passed_payload,
        validation_manifest=finite_beta_manifest,
        output_root=tmp_path / "outer_loop_accept_finite_beta",
    )
    finite_beta_accepted_payload = json.loads(
        finite_beta_accepted.decision_path.read_text(encoding="utf-8")
    )
    assert finite_beta_accepted_payload["decision"] == "accepted"
    assert finite_beta_accepted_payload["lane_b_predecessor_status"]["passed"] is True

    del finite_beta_passed_payload["lane_b_predecessor_status"]
    finite_beta_blocked_manifest = build_desc_joint_validation_manifest(
        result_payload=finite_beta_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    finite_beta_rejected = materialize_desc_joint_outer_loop_decision(
        result_payload=finite_beta_passed_payload,
        validation_manifest=finite_beta_blocked_manifest,
        output_root=tmp_path / "outer_loop_reject_finite_beta",
    )
    finite_beta_rejected_payload = json.loads(
        finite_beta_rejected.decision_path.read_text(encoding="utf-8")
    )
    assert finite_beta_rejected_payload["decision"] == "rejected"
    assert finite_beta_rejected_payload["rejection_stage"] == "lane_b_predecessor"

    other_exported_artifact_path = tmp_path / "other_joint_exported_biot_savart.json"
    other_exported_artifact_path.write_text('{"field": false}\n', encoding="utf-8")
    other_exported_artifact_paths = (str(other_exported_artifact_path),)
    other_oracle_path = tmp_path / "other_hardware_contact_report.json"
    _write_final_oracle_evidence(
        other_oracle_path,
        exported_artifact_paths=other_exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )
    mismatched_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=other_exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(other_oracle_path),
    )
    with pytest.raises(ValueError, match="exported artifact paths"):
        materialize_desc_joint_outer_loop_decision(
            result_payload=desc_passed_payload,
            validation_manifest=mismatched_manifest,
            output_root=tmp_path / "outer_loop_mismatch",
        )

    mode_mismatched_manifest = json.loads(json.dumps(passing_manifest))
    mode_mismatched_manifest["run_mode"] = "finite_beta_joint"
    mode_mismatched_manifest["lane_b_predecessor_status"] = {
        "passed": True,
        "reason": "Lane B predecessor passed strict SIMSOPT validation",
        "artifact_paths": [str(lane_b_evidence_path.resolve())],
    }
    with pytest.raises(ValueError, match="run_mode"):
        materialize_desc_joint_outer_loop_decision(
            result_payload=desc_passed_payload,
            validation_manifest=mode_mismatched_manifest,
            output_root=tmp_path / "outer_loop_mode_mismatch",
        )

    failing_manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        physics_validation_passed=True,
        artifact_hardware_passed=False,
        search_hardware_passed=True,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
    )
    rejected = materialize_desc_joint_outer_loop_decision(
        result_payload=desc_passed_payload,
        validation_manifest=failing_manifest,
        output_root=tmp_path / "outer_loop_reject",
    )
    rejected_payload = json.loads(
        rejected.decision_path.read_text(encoding="utf-8")
    )
    assert rejected_payload["decision"] == "rejected"
    assert rejected_payload["eligible_for_next_search_stage"] is False
    assert rejected_payload["rejection_stage"] == "artifact_hardware"


def test_desc_joint_outer_loop_gate_cli_materializes_decision(tmp_path):
    exported_artifact_path = tmp_path / "joint_exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    fixed_polish_evidence_path = tmp_path / "fixed_polish_validation_manifest.json"
    source_checksums = {"seed_field": "a" * 64}
    _write_fixed_polish_predecessor_manifest(
        fixed_polish_evidence_path,
        source_artifact_checksums=source_checksums,
    )
    oracle_path = tmp_path / "hardware_contact_report.json"
    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )
    result_payload = _desc_solve_passed_payload(
        build_preflight_result_payload(
            mode="vacuum_joint",
            input_contract={"seed": "fixture"},
            objective_stack=[
                "VacuumBoundaryError",
                "LinkingCurrentConsistency",
                VOLUME_OBJECTIVE,
            ],
        )
    )
    _record_result_exported_artifact_paths(result_payload, exported_artifact_paths)
    result_payload["fixed_polish_predecessor_status"] = {
        "state": "passed",
        "reason": "fixed-polish predecessor passed round-trip validation",
        "artifact_paths": [str(fixed_polish_evidence_path)],
    }
    manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )
    result_path = _write_json(tmp_path / "desc_result.json", result_payload)
    manifest_path = _write_json(tmp_path / "desc_joint_validation_manifest.json", manifest)
    output_root = tmp_path / "outer_loop_cli"

    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES_ROOT / "DESC_JOINT" / "gate_desc_joint_candidate.py"),
            "--result",
            str(result_path),
            "--validation-manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    decision_path = Path(completed.stdout.strip())
    assert decision_path == output_root / "desc_joint_outer_loop_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["decision"] == "accepted"
    assert decision["validation_manifest_path"] == str(manifest_path.resolve())


def test_validation_manifest_validator_rejects_forged_promotion_pass(tmp_path):
    exported_artifact_path = tmp_path / "exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    exported_artifact_paths = (str(exported_artifact_path),)
    source_checksums = {"seed_field": "a" * 64}
    result_payload = build_preflight_result_payload(
        mode="fixed_equilibrium_polish",
        input_contract={"seed": "fixture"},
        objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
    )
    desc_passed_payload = _desc_solve_passed_payload(result_payload)
    oracle_path = tmp_path / "hardware_contact_report.json"
    _write_final_oracle_evidence(
        oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )
    manifest = build_desc_joint_validation_manifest(
        result_payload=desc_passed_payload,
        exported_artifact_paths=exported_artifact_paths,
        expected_source_artifact_checksums=source_checksums,
        physics_validation_passed=True,
        artifact_hardware_passed=True,
        search_hardware_passed=True,
        final_oracle_passed=True,
        final_oracle_evidence_path=str(oracle_path),
    )

    manifest["desc_solve_status"]["state"] = "blocked"
    with pytest.raises(ValueError, match="desc_solve_status.state"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["desc_solve_status"]["state"] = "passed"
    manifest["physics_validation_status"]["passed"] = False
    with pytest.raises(ValueError, match="physics_validation_status.passed"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["physics_validation_status"]["passed"] = None
    with pytest.raises(ValueError, match="physics_validation_status.passed"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["physics_validation_status"]["passed"] = True
    manifest["artifact_hardware_status"]["passed"] = False
    with pytest.raises(ValueError, match="artifact_hardware_status.passed"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["artifact_hardware_status"]["passed"] = None
    with pytest.raises(ValueError, match="artifact_hardware_status.passed"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["artifact_hardware_status"]["passed"] = True
    manifest["final_oracle_status"]["passed"] = False
    with pytest.raises(ValueError, match="final_oracle_status.passed"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["final_oracle_status"]["passed"] = True
    manifest["promotion_status"]["final_oracle_evidence_path"] = str(
        tmp_path / "missing_oracle_report.json"
    )
    with pytest.raises(ValueError, match="existing file"):
        validate_desc_joint_validation_manifest(manifest)

    other_oracle_path = tmp_path / "other_hardware_contact_report.json"
    _write_final_oracle_evidence(
        other_oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )
    manifest["promotion_status"]["final_oracle_evidence_path"] = str(other_oracle_path)
    with pytest.raises(ValueError, match="must match"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["promotion_status"]["final_oracle_evidence_path"] = str(oracle_path)
    mismatched_oracle_path = tmp_path / "mismatched_hardware_contact_report.json"
    other_exported_artifact_path = tmp_path / "other_exported.json"
    other_exported_artifact_path.write_text('{"field": false}\n', encoding="utf-8")
    _write_final_oracle_evidence(
        mismatched_oracle_path,
        exported_artifact_paths=(str(other_exported_artifact_path),),
        source_artifact_checksums=source_checksums,
    )
    manifest["final_oracle_status"]["evidence_path"] = str(mismatched_oracle_path)
    with pytest.raises(ValueError, match="exported_artifact_paths"):
        validate_desc_joint_validation_manifest(manifest)
    _write_final_oracle_evidence(
        mismatched_oracle_path,
        exported_artifact_paths=exported_artifact_paths,
        source_artifact_checksums=source_checksums,
    )
    oracle_payload = json.loads(mismatched_oracle_path.read_text(encoding="utf-8"))
    oracle_payload["exported_artifact_checksums"] = {
        str(exported_artifact_path): "b" * 64
    }
    _write_json(mismatched_oracle_path, oracle_payload)
    with pytest.raises(ValueError, match="exported_artifact_checksums"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["final_oracle_status"]["evidence_path"] = str(oracle_path)
    manifest["source_artifact_checksums"] = {"seed_field": "b" * 64}
    with pytest.raises(ValueError, match="source_artifact_checksums"):
        validate_desc_joint_validation_manifest(manifest)


def test_validation_manifest_validator_rejects_malformed_passed_fields(tmp_path):
    exported_artifact_path = tmp_path / "exported_biot_savart.json"
    exported_artifact_path.write_text('{"field": true}\n', encoding="utf-8")
    result_payload = build_preflight_result_payload(
        mode="fixed_equilibrium_polish",
        input_contract={"seed": "fixture"},
        objective_stack=["QuadraticFlux", "LinkingCurrentConsistency"],
    )
    manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=(str(exported_artifact_path),),
        physics_validation_passed=None,
        artifact_hardware_passed=None,
        search_hardware_passed=None,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
    )

    manifest["physics_validation_status"]["passed"] = "yes"
    with pytest.raises(ValueError, match="physics_validation_status.passed"):
        validate_desc_joint_validation_manifest(manifest)

    manifest["physics_validation_status"]["passed"] = None
    manifest["final_oracle_status"]["passed"] = None
    with pytest.raises(ValueError, match="final_oracle_status.passed"):
        validate_desc_joint_validation_manifest(manifest)
