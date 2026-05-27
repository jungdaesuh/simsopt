from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import CurveCWSFourierCPP, CurveXYZFourier

from banana_opt.current_contracts import (
    HBT_PROXY_VF_CURRENT_RATIO,
    MU0,
    validate_jhalpern30_proxy_vf_current_convention,
)
from banana_opt.finite_current_profiles import (
    DEFAULT_JHALPERN30_VF_TEMPLATE_PATH,
    JHALPERN30_BOOZER_CURRENT_CONVENTION,
    JHALPERN30_FINITE_CURRENT_MODE,
    JHALPERN30_G0_POLICY,
    JHALPERN30_NUM_BANANA_COILS,
    JHALPERN30_NUM_PROXY_COILS,
    JHALPERN30_NUM_TF_COILS,
    JHALPERN30_NUM_VF_COILS,
    JHALPERN30_PROFILE,
    JHALPERN30_PROXY_PLACEMENT_MODE,
    JHALPERN30_VF_CURRENT_MUTABILITY,
    JHALPERN30_VF_CURRENT_SIGN_POLICY,
    JHALPERN30_VF_TEMPLATE_SHA256,
)
from banana_opt.hardware_contracts import (
    BANANA_WINDING_MINOR_RADIUS_M,
    MAX_CURVATURE_INV_M,
)
from banana_opt.json_compat import load_boozer_finite_i
from banana_opt.wout_convention import wout_convention_artifact_fields


JHALPERN30_STAGE_STATE_REQUIRED_KEYS = (
    "iota",
    "G",
    "volume",
    "iota_target",
    "stage_idx",
    "stage_mpol",
    "stage_ntor",
    "stage_order",
    "stage_qp",
)
JHALPERN30_STAGE_BSURF_FILENAME = "bsurf_opt.json"
JHALPERN30_STAGE_STATE_FILENAME = "state.json"
JHALPERN30_IMPORTED_RESULTS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Jhalpern30BananaCurrentReplay:
    current_scale_A: float
    banana_current_sign: int
    banana_current_pinned: bool
    banana_i_fixed_s2_kA: float | None


@dataclass(frozen=True)
class Jhalpern30StageBundle:
    stage_dir: Path
    bsurf_path: Path
    state_path: Path
    state: Mapping[str, object]


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_jhalpern30_vf_template_path(vf_template_path: str | None) -> str:
    if vf_template_path not in {None, ""}:
        return str(vf_template_path)
    if not DEFAULT_JHALPERN30_VF_TEMPLATE_PATH.is_file():
        raise FileNotFoundError(
            "Bundled jhalpern30 VF template is missing: "
            f"{DEFAULT_JHALPERN30_VF_TEMPLATE_PATH}."
        )
    return str(DEFAULT_JHALPERN30_VF_TEMPLATE_PATH)


def _load_validated_jhalpern30_vf_template(
    vf_template_path: str | Path,
    *,
    load_fn=load_boozer_finite_i,
):
    template_path = Path(vf_template_path)
    if not template_path.is_file():
        raise FileNotFoundError(f"jhalpern30 VF template not found: {template_path}.")
    if template_path.resolve() == DEFAULT_JHALPERN30_VF_TEMPLATE_PATH.resolve():
        template_sha256 = sha256_file(template_path)
        if template_sha256 != JHALPERN30_VF_TEMPLATE_SHA256:
            raise ValueError(
                "Bundled jhalpern30 VF template hash drifted: "
                f"{template_sha256} != {JHALPERN30_VF_TEMPLATE_SHA256}."
            )
    loaded_template = load_fn(str(template_path))
    template_coils = getattr(loaded_template, "coils", None)
    if template_coils is None:
        raise ValueError(
            f"jhalpern30 VF template {template_path} does not contain coils."
        )
    if len(template_coils) != JHALPERN30_NUM_VF_COILS:
        raise ValueError(
            "jhalpern30 VF template must contain exactly "
            f"{JHALPERN30_NUM_VF_COILS} coils; got {len(template_coils)}."
        )
    signs = tuple(float(np.sign(coil.current.get_value())) for coil in template_coils)
    if any(sign == 0.0 for sign in signs):
        raise ValueError("jhalpern30 VF template currents must be non-zero.")
    return loaded_template, signs


def validate_jhalpern30_vf_template(
    vf_template_path: str | Path,
    *,
    load_fn=load_boozer_finite_i,
) -> tuple[float, ...]:
    _, signs = _load_validated_jhalpern30_vf_template(
        vf_template_path,
        load_fn=load_fn,
    )
    return signs


def jhalpern30_proxy_boozer_I(proxy_current_A: float) -> float:
    return MU0 * float(proxy_current_A)


def jhalpern30_banana_current_sign(*, flip_banana: bool) -> int:
    return -1 if flip_banana else 1


def resolve_jhalpern30_banana_current_replay(
    *,
    flip_banana: bool,
    banana_i_fixed_s2: str | None,
) -> Jhalpern30BananaCurrentReplay:
    banana_current_sign = jhalpern30_banana_current_sign(flip_banana=flip_banana)
    if banana_i_fixed_s2 is not None and banana_i_fixed_s2.strip() != "":
        banana_i_fixed_s2_kA = float(banana_i_fixed_s2)
        current_scale_A = banana_current_sign * banana_i_fixed_s2_kA * 1.0e3
        return Jhalpern30BananaCurrentReplay(
            current_scale_A=current_scale_A,
            banana_current_sign=banana_current_sign,
            banana_current_pinned=True,
            banana_i_fixed_s2_kA=banana_i_fixed_s2_kA,
        )
    return Jhalpern30BananaCurrentReplay(
        current_scale_A=banana_current_sign * -1.0e4,
        banana_current_sign=banana_current_sign,
        banana_current_pinned=False,
        banana_i_fixed_s2_kA=None,
    )


def build_jhalpern30_banana_coils(
    banana_curve: CurveCWSFourierCPP,
    *,
    surf_coils,
    flip_banana: bool,
    banana_i_fixed_s2: str | None,
) -> tuple[list[Coil], Jhalpern30BananaCurrentReplay]:
    replay = resolve_jhalpern30_banana_current_replay(
        flip_banana=flip_banana,
        banana_i_fixed_s2=banana_i_fixed_s2,
    )
    banana_raw_current = Current(1.0)
    banana_scaled_current = ScaledCurrent(
        banana_raw_current,
        replay.current_scale_A,
    )
    if replay.banana_current_pinned:
        banana_raw_current.fix_all()
    banana_coils = coils_via_symmetries(
        [banana_curve],
        [banana_scaled_current],
        surf_coils.nfp,
        surf_coils.stellsym,
    )
    return banana_coils, replay


def build_jhalpern30_proxy_plasma_current_coils(
    surface,
    proxy_current_A: float,
) -> list[Coil]:
    proxy_curve = CurveXYZFourier(128, 1)
    proxy_curve.set("xc(1)", float(surface.major_radius()))
    proxy_curve.set("ys(1)", float(surface.major_radius()))
    proxy_curve.set("zc(0)", 0.0)
    proxy_curve.fix_all()
    proxy_current = Current(float(proxy_current_A))
    proxy_current.fix_all()
    return [Coil(proxy_curve, proxy_current)]


def build_jhalpern30_vf_coils(
    proxy_current_A: float,
    template_path: str | Path,
    *,
    load_fn=load_boozer_finite_i,
) -> list[Coil]:
    resolved_proxy_current_A = float(proxy_current_A)
    vf_current_A = resolved_proxy_current_A * HBT_PROXY_VF_CURRENT_RATIO
    template_path = resolve_jhalpern30_vf_template_path(str(template_path))
    loaded_template, template_signs = _load_validated_jhalpern30_vf_template(
        template_path,
        load_fn=load_fn,
    )
    validate_jhalpern30_proxy_vf_current_convention(
        proxy_plasma_current_A=resolved_proxy_current_A,
        vf_current_A=vf_current_A,
    )
    shared_leaf_current = Current(1.0)
    shared_scaled_current = ScaledCurrent(shared_leaf_current, vf_current_A)
    proxy_sign = float(np.sign(resolved_proxy_current_A))
    vf_coils: list[Coil] = []
    for template_coil, template_sign in zip(loaded_template.coils, template_signs):
        vf_curve = template_coil.curve
        vf_curve.fix_all()
        vf_current = shared_scaled_current * (template_sign * proxy_sign)
        vf_current.unfix_all()
        vf_coils.append(Coil(vf_curve, vf_current))
    return vf_coils


def jhalpern30_flip_from_stage_parent(path: str | Path) -> bool:
    return Path(path).parent.name.endswith("_flip")


def jhalpern30_iota_target_sign(*, flip_banana: bool) -> int:
    return -1 if flip_banana else 1


def validate_jhalpern30_stage_state(state: Mapping[str, object]) -> dict[str, object]:
    missing_keys = [
        key for key in JHALPERN30_STAGE_STATE_REQUIRED_KEYS if state.get(key) is None
    ]
    if missing_keys:
        raise ValueError(
            "jhalpern30 stage state is missing required keys: "
            + ", ".join(missing_keys)
            + "."
        )
    return {key: state[key] for key in JHALPERN30_STAGE_STATE_REQUIRED_KEYS}


def resolve_jhalpern30_stage_bundle(
    bundle_root: str | Path,
    *,
    stage_name: str = "stage00",
) -> Jhalpern30StageBundle:
    stage_dir = Path(bundle_root) / stage_name
    bsurf_path = stage_dir / JHALPERN30_STAGE_BSURF_FILENAME
    state_path = stage_dir / JHALPERN30_STAGE_STATE_FILENAME
    if not bsurf_path.is_file():
        raise FileNotFoundError(f"jhalpern30 stage Boozer surface missing: {bsurf_path}.")
    if not state_path.is_file():
        raise FileNotFoundError(f"jhalpern30 stage state missing: {state_path}.")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"jhalpern30 stage state must be a JSON object: {state_path}.")
    validated_state = validate_jhalpern30_stage_state(state)
    return Jhalpern30StageBundle(
        stage_dir=stage_dir,
        bsurf_path=bsurf_path,
        state_path=state_path,
        state=validated_state,
    )


def _copy_or_save_biot_savart(
    loaded_boozer_surface,
    output_bs_path: Path,
) -> BiotSavart:
    biotsavart = getattr(loaded_boozer_surface, "biotsavart", None)
    if biotsavart is None:
        raise ValueError(
            "jhalpern30 stage importer requires bsurf_opt.json to contain a "
            "BoozerSurface with an attached BiotSavart field."
        )
    if not isinstance(biotsavart, BiotSavart):
        raise TypeError(
            "jhalpern30 stage importer expected a BiotSavart field; got "
            f"{type(biotsavart).__name__}."
        )
    biotsavart.save(str(output_bs_path))
    return biotsavart


def _extract_jhalpern30_tf_current_A(biotsavart: BiotSavart) -> float:
    expected_total_coils = JHALPERN30_PROFILE.default_total_coils
    if len(biotsavart.coils) != expected_total_coils:
        raise ValueError(
            "jhalpern30 stage importer expected "
            f"{expected_total_coils} coils; got {len(biotsavart.coils)}."
        )
    tf_currents_A = tuple(
        float(coil.current.get_value())
        for coil in biotsavart.coils[:JHALPERN30_NUM_TF_COILS]
    )
    first_tf_current_A = tf_currents_A[0]
    if not np.allclose(tf_currents_A, first_tf_current_A, rtol=0.0, atol=1.0e-9):
        raise ValueError("jhalpern30 stage importer requires uniform TF currents.")
    return first_tf_current_A


def import_jhalpern30_stage_bundle(
    bundle_root: str | Path,
    output_dir: str | Path,
    *,
    plasma_surf_path: str | Path,
    stage_name: str = "stage00",
    load_fn=load_boozer_finite_i,
) -> tuple[Path, Path]:
    bundle = resolve_jhalpern30_stage_bundle(bundle_root, stage_name=stage_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    imported_bs_path = output_path / "biot_savart_opt.json"
    imported_results_path = output_path / "results.json"

    loaded_boozer_surface = load_fn(str(bundle.bsurf_path))
    biotsavart = _copy_or_save_biot_savart(loaded_boozer_surface, imported_bs_path)
    shutil.copyfile(bundle.bsurf_path, output_path / bundle.bsurf_path.name)
    shutil.copyfile(bundle.state_path, output_path / bundle.state_path.name)

    state = dict(bundle.state)
    stage_idx = int(state["stage_idx"])
    flip_banana = jhalpern30_flip_from_stage_parent(bundle.stage_dir)
    banana_replay = resolve_jhalpern30_banana_current_replay(
        flip_banana=flip_banana,
        banana_i_fixed_s2=None,
    )
    coil_groups = JHALPERN30_PROFILE.build_default_coil_groups_manifest()
    tf_current_A = _extract_jhalpern30_tf_current_A(biotsavart)
    wout_fields = wout_convention_artifact_fields(
        wout_path=plasma_surf_path,
        tf_current_A=tf_current_A,
    )
    proxy_index = JHALPERN30_NUM_TF_COILS + JHALPERN30_NUM_BANANA_COILS
    proxy_current_A = float(biotsavart.coils[proxy_index].current.get_value())
    results = {
        "JHALPERN30_IMPORT_SCHEMA_VERSION": JHALPERN30_IMPORTED_RESULTS_SCHEMA_VERSION,
        "FINITE_CURRENT_MODE": JHALPERN30_FINITE_CURRENT_MODE,
        "PLASMA_SURF_PATH": str(plasma_surf_path),
        "WOUT_CONVENTION": wout_fields["WOUT_CONVENTION"],
        "WOUT_OFF_SPEC": wout_fields["WOUT_OFF_SPEC"],
        "TF_CURRENT_A": tf_current_A,
        "BOOZER_CURRENT_CONVENTION": JHALPERN30_BOOZER_CURRENT_CONVENTION,
        "BOOZER_I": jhalpern30_proxy_boozer_I(proxy_current_A),
        "PROXY_PLACEMENT_MODE": JHALPERN30_PROXY_PLACEMENT_MODE,
        "G0_POLICY": JHALPERN30_G0_POLICY,
        "PROXY_PLASMA_CURRENT_A": proxy_current_A,
        "VF_CURRENT_A": proxy_current_A * HBT_PROXY_VF_CURRENT_RATIO,
        "VF_TEMPLATE_PATH": str(DEFAULT_JHALPERN30_VF_TEMPLATE_PATH),
        "VF_TEMPLATE_SHA256": JHALPERN30_VF_TEMPLATE_SHA256,
        "VF_CURRENT_SIGN_POLICY": JHALPERN30_VF_CURRENT_SIGN_POLICY,
        "VF_CURRENT_MUTABILITY": JHALPERN30_VF_CURRENT_MUTABILITY,
        "FLIP_BANANA": flip_banana,
        "BANANA_CURRENT_SIGN": banana_replay.banana_current_sign,
        "BANANA_CURRENT_PINNED": banana_replay.banana_current_pinned,
        "BANANA_I_FIXED_S2_KA": banana_replay.banana_i_fixed_s2_kA,
        "IOTA_TARGET_SIGN": jhalpern30_iota_target_sign(flip_banana=flip_banana),
        "NUM_TF_COILS": JHALPERN30_NUM_TF_COILS,
        "NUM_BANANA_COILS": JHALPERN30_NUM_BANANA_COILS,
        "NUM_PROXY_COILS": JHALPERN30_NUM_PROXY_COILS,
        "NUM_VF_COILS": JHALPERN30_NUM_VF_COILS,
        "TOTAL_COILS": coil_groups.total(),
        "COIL_GROUPS": coil_groups.to_json_payload(),
        "STAGE2_BS_PATH": str(imported_bs_path),
        "STAGE2_BS_SHA256": sha256_file(imported_bs_path),
        "JHALPERN30_STAGE_BSURF_PATH": str(bundle.bsurf_path),
        "JHALPERN30_STAGE_BSURF_SHA256": sha256_file(bundle.bsurf_path),
        "JHALPERN30_STAGE_STATE_PATH": str(bundle.state_path),
        "JHALPERN30_STAGE_NAME": f"stage{stage_idx:02d}",
        "JHALPERN30_STAGE_STATE": state,
        "G": float(state["G"]),
        "IOTA": float(state["iota"]),
        "IOTA_TARGET": float(state["iota_target"]),
        "VOLUME": float(state["volume"]),
        "MAJOR_RADIUS": 0.976,
        "TOROIDAL_FLUX": 0.24,
        "banana_surf_radius": BANANA_WINDING_MINOR_RADIUS_M,
        "CURVATURE_THRESHOLD": MAX_CURVATURE_INV_M,
        "order": int(state["stage_order"]),
    }
    imported_results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return imported_bs_path, imported_results_path


__all__ = [
    "JHALPERN30_STAGE_STATE_REQUIRED_KEYS",
    "Jhalpern30BananaCurrentReplay",
    "Jhalpern30StageBundle",
    "build_jhalpern30_banana_coils",
    "build_jhalpern30_proxy_plasma_current_coils",
    "build_jhalpern30_vf_coils",
    "import_jhalpern30_stage_bundle",
    "jhalpern30_banana_current_sign",
    "jhalpern30_flip_from_stage_parent",
    "jhalpern30_iota_target_sign",
    "jhalpern30_proxy_boozer_I",
    "resolve_jhalpern30_banana_current_replay",
    "resolve_jhalpern30_stage_bundle",
    "resolve_jhalpern30_vf_template_path",
    "sha256_file",
    "validate_jhalpern30_stage_state",
    "validate_jhalpern30_vf_template",
]
