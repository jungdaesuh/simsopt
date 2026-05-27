from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from banana_opt.coil_groups import CoilGroupsManifest, build_contiguous_manifest
from banana_opt.current_contracts import (
    BananaReplayPolicy,
    BoozerCurrentConvention,
    DEFAULT_FINITE_CURRENT_MODE,
    FiniteCurrentMode,
    G0Policy,
    HBT_PROXY_VF_CURRENT_RATIO,
    ProxyPlacementPolicy,
    ProxyVfCurrentScalarPolicy,
    VfCurrentMutability,
    VfCurrentSignPolicy,
    resolve_boozer_current_convention,
)


_MODULE_DIR = Path(__file__).resolve().parent

WATARU_FINITE_CURRENT_MODE: FiniteCurrentMode = DEFAULT_FINITE_CURRENT_MODE
JHALPERN30_FINITE_CURRENT_MODE: FiniteCurrentMode = "jhalpern30_proxy_field"

DEFAULT_WATARU_VF_TEMPLATE_PATH = _MODULE_DIR / "wataru_vf_template.json"
DEFAULT_JHALPERN30_VF_TEMPLATE_PATH = _MODULE_DIR / "jhalpern30_vf_biotsavart.json"


@dataclass(frozen=True)
class FiniteCurrentProfile:
    """Finite-current mode policy metadata shared by runners and importers."""

    mode: FiniteCurrentMode
    default_num_tf_coils: int
    default_num_banana_coils: int
    default_num_proxy_coils: int
    default_num_vf_coils: int
    boozer_current_convention: BoozerCurrentConvention
    g0_policy: G0Policy
    proxy_placement_policy: ProxyPlacementPolicy
    proxy_vf_current_scalar_policy: ProxyVfCurrentScalarPolicy
    default_vf_template_path: Path | None
    vf_template_sha256: str | None
    vf_current_ratio: float
    vf_current_sign_policy: VfCurrentSignPolicy
    vf_current_mutability: VfCurrentMutability
    banana_replay_policy: BananaReplayPolicy
    supported_entrypoints: tuple[str, ...]
    rejected_entrypoints: tuple[str, ...]
    required_artifact_metadata_keys: tuple[str, ...]

    def build_default_coil_groups_manifest(self) -> CoilGroupsManifest:
        return build_contiguous_manifest(
            num_tf_coils=self.default_num_tf_coils,
            num_banana_coils=self.default_num_banana_coils,
            num_proxy_coils=self.default_num_proxy_coils,
            num_vf_coils=self.default_num_vf_coils,
        )

    @property
    def default_total_coils(self) -> int:
        return (
            self.default_num_tf_coils
            + self.default_num_banana_coils
            + self.default_num_proxy_coils
            + self.default_num_vf_coils
        )


_COMMON_REQUIRED_ARTIFACT_METADATA_KEYS = (
    "FINITE_CURRENT_MODE",
    "BOOZER_CURRENT_CONVENTION",
    "G0_POLICY",
    "PROXY_PLACEMENT_MODE",
    "PROXY_VF_CURRENT_SCALAR_POLICY",
    "PROXY_PLASMA_CURRENT_A",
    "VF_CURRENT_A",
    "VF_TEMPLATE_PATH",
    "VF_TEMPLATE_SHA256",
    "VF_CURRENT_SIGN_POLICY",
    "VF_CURRENT_MUTABILITY",
    "NUM_TF_COILS",
    "NUM_BANANA_COILS",
    "NUM_PROXY_COILS",
    "NUM_VF_COILS",
    "TOTAL_COILS",
    "COIL_GROUPS",
)

WATARU_PROFILE = FiniteCurrentProfile(
    mode=WATARU_FINITE_CURRENT_MODE,
    default_num_tf_coils=20,
    default_num_banana_coils=10,
    default_num_proxy_coils=1,
    default_num_vf_coils=20,
    boozer_current_convention=resolve_boozer_current_convention(
        WATARU_FINITE_CURRENT_MODE,
    ),
    g0_policy="signed_explicit_tf_current",
    proxy_placement_policy="vmec_axis_zeroth_coefficients",
    proxy_vf_current_scalar_policy="nonnegative_magnitude",
    default_vf_template_path=DEFAULT_WATARU_VF_TEMPLATE_PATH,
    vf_template_sha256="1df87dbe845b014199fb1a4a1a414a2dff922d0ae9da10b1861092d0f326d989",
    vf_current_ratio=HBT_PROXY_VF_CURRENT_RATIO,
    vf_current_sign_policy="template_sign_vf_current_scalar",
    vf_current_mutability="independent_fixed_current",
    banana_replay_policy="stage2_cli_seed_current",
    supported_entrypoints=(
        "STAGE_2/banana_coil_solver.py",
        "SINGLE_STAGE/single_stage_banana_example.py",
        "run_stage2_to_single_stage.py",
    ),
    rejected_entrypoints=(),
    required_artifact_metadata_keys=_COMMON_REQUIRED_ARTIFACT_METADATA_KEYS,
)

JHALPERN30_PROFILE = FiniteCurrentProfile(
    mode=JHALPERN30_FINITE_CURRENT_MODE,
    default_num_tf_coils=20,
    default_num_banana_coils=10,
    default_num_proxy_coils=1,
    default_num_vf_coils=20,
    boozer_current_convention=resolve_boozer_current_convention(
        JHALPERN30_FINITE_CURRENT_MODE,
    ),
    g0_policy="signed_explicit_tf_current",
    proxy_placement_policy="surface_major_radius_z0",
    proxy_vf_current_scalar_policy="signed_physical_scalar",
    default_vf_template_path=DEFAULT_JHALPERN30_VF_TEMPLATE_PATH,
    vf_template_sha256="1df87dbe845b014199fb1a4a1a414a2dff922d0ae9da10b1861092d0f326d989",
    vf_current_ratio=HBT_PROXY_VF_CURRENT_RATIO,
    vf_current_sign_policy="template_sign_abs_proxy_current",
    vf_current_mutability="shared_unfixed_scaled_current",
    banana_replay_policy=(
        "flip_banana_parent_suffix_env_BANANA_CURRENT_SIGN_BANANA_I_FIXED_S2"
    ),
    supported_entrypoints=(
        "import_jhalpern30_replay.py",
        "STAGE_2/banana_coil_solver.py",
        "SINGLE_STAGE/single_stage_banana_example.py",
    ),
    rejected_entrypoints=(
        "run_stage2_to_single_stage.py:pre_boozer_repair",
    ),
    required_artifact_metadata_keys=(
        *_COMMON_REQUIRED_ARTIFACT_METADATA_KEYS,
        "BOOZER_I",
        "FLIP_BANANA",
        "BANANA_CURRENT_SIGN",
        "BANANA_CURRENT_PINNED",
        "BANANA_I_FIXED_S2_KA",
        "IOTA_TARGET_SIGN",
        "JHALPERN30_STAGE_NAME",
        "JHALPERN30_STAGE_STATE",
    ),
)

FINITE_CURRENT_PROFILES: Mapping[FiniteCurrentMode, FiniteCurrentProfile] = {
    WATARU_PROFILE.mode: WATARU_PROFILE,
    JHALPERN30_PROFILE.mode: JHALPERN30_PROFILE,
}


def get_finite_current_profile(mode: str) -> FiniteCurrentProfile:
    profile = FINITE_CURRENT_PROFILES.get(mode)
    if profile is None:
        supported = ", ".join(sorted(FINITE_CURRENT_PROFILES))
        raise ValueError(
            f"Unsupported finite-current profile mode {mode!r}; expected {supported}."
        )
    return profile


JHALPERN30_NUM_TF_COILS = JHALPERN30_PROFILE.default_num_tf_coils
JHALPERN30_NUM_BANANA_COILS = JHALPERN30_PROFILE.default_num_banana_coils
JHALPERN30_NUM_PROXY_COILS = JHALPERN30_PROFILE.default_num_proxy_coils
JHALPERN30_NUM_VF_COILS = JHALPERN30_PROFILE.default_num_vf_coils
JHALPERN30_G0_POLICY = JHALPERN30_PROFILE.g0_policy
JHALPERN30_BOOZER_CURRENT_CONVENTION = JHALPERN30_PROFILE.boozer_current_convention
JHALPERN30_PROXY_PLACEMENT_MODE = JHALPERN30_PROFILE.proxy_placement_policy
JHALPERN30_PROXY_VF_CURRENT_SCALAR_POLICY = (
    JHALPERN30_PROFILE.proxy_vf_current_scalar_policy
)
JHALPERN30_VF_CURRENT_SIGN_POLICY = JHALPERN30_PROFILE.vf_current_sign_policy
JHALPERN30_VF_CURRENT_MUTABILITY = JHALPERN30_PROFILE.vf_current_mutability
JHALPERN30_VF_TEMPLATE_SHA256 = JHALPERN30_PROFILE.vf_template_sha256

__all__ = [
    "DEFAULT_JHALPERN30_VF_TEMPLATE_PATH",
    "DEFAULT_WATARU_VF_TEMPLATE_PATH",
    "FINITE_CURRENT_PROFILES",
    "FiniteCurrentProfile",
    "JHALPERN30_BOOZER_CURRENT_CONVENTION",
    "JHALPERN30_FINITE_CURRENT_MODE",
    "JHALPERN30_G0_POLICY",
    "JHALPERN30_NUM_BANANA_COILS",
    "JHALPERN30_NUM_PROXY_COILS",
    "JHALPERN30_NUM_TF_COILS",
    "JHALPERN30_NUM_VF_COILS",
    "JHALPERN30_PROFILE",
    "JHALPERN30_PROXY_PLACEMENT_MODE",
    "JHALPERN30_PROXY_VF_CURRENT_SCALAR_POLICY",
    "JHALPERN30_VF_CURRENT_MUTABILITY",
    "JHALPERN30_VF_CURRENT_SIGN_POLICY",
    "JHALPERN30_VF_TEMPLATE_SHA256",
    "WATARU_FINITE_CURRENT_MODE",
    "WATARU_PROFILE",
    "get_finite_current_profile",
]
