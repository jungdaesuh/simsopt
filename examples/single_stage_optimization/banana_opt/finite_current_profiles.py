from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
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

VACUUM_FINITE_CURRENT_MODE: FiniteCurrentMode = "vacuum"
WATARU_FINITE_CURRENT_MODE: FiniteCurrentMode = DEFAULT_FINITE_CURRENT_MODE
JHALPERN30_FINITE_CURRENT_MODE: FiniteCurrentMode = "jhalpern30_proxy_field"

DEFAULT_WATARU_VF_TEMPLATE_PATH = _MODULE_DIR / "wataru_vf_template.json"
DEFAULT_JHALPERN30_VF_TEMPLATE_PATH = _MODULE_DIR / "jhalpern30_vf_biotsavart.json"
_PROXY_SIGN_HELP_WIDTH = 88


@dataclass(frozen=True)
class ProxyCurrentSignConvention:
    """Operator-facing proxy-current sign contract for one finite-current mode."""

    key: str
    frame: str
    signedness: str
    replay_reference: str
    operator_warning: str
    empirical_effect_note: str = ""

    def metadata_fields(self) -> dict[str, str]:
        return {
            "PROXY_CURRENT_SIGN_CONVENTION": self.key,
            "PROXY_CURRENT_SIGN_FRAME": self.frame,
            "PROXY_CURRENT_SIGNEDNESS": self.signedness,
            "PROXY_CURRENT_SIGN_REPLAY_REFERENCE": self.replay_reference,
            "PROXY_CURRENT_OPERATOR_WARNING": self.operator_warning,
        }

    def summary_fields(
        self,
        *,
        mode: FiniteCurrentMode,
        scalar_policy: ProxyVfCurrentScalarPolicy,
    ) -> dict[str, str]:
        return {
            "proxy_current_mode": mode,
            "proxy_current_sign_convention": self.key,
            "proxy_current_sign_frame": self.frame,
            "proxy_current_signedness": self.signedness,
            "proxy_current_scalar_policy": scalar_policy,
            "proxy_current_replay_reference": self.replay_reference,
            "proxy_current_operator_warning": self.operator_warning,
        }

    def help_text(
        self,
        *,
        mode: FiniteCurrentMode,
        scalar_policy: ProxyVfCurrentScalarPolicy,
    ) -> str:
        effect_note = (
            f" {self.empirical_effect_note}" if self.empirical_effect_note else ""
        )
        return fill(
            (
                f"{self.operator_warning}{effect_note} "
                f"(scalar_policy={scalar_policy}; convention={self.key})"
            ),
            width=_PROXY_SIGN_HELP_WIDTH,
            initial_indent=f"- {mode}: ",
            subsequent_indent="  ",
        )


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
    proxy_current_sign_convention: ProxyCurrentSignConvention
    supported_entrypoints: tuple[str, ...]
    rejected_entrypoints: tuple[str, ...]
    required_artifact_metadata_keys: tuple[str, ...]

    def proxy_current_sign_metadata_fields(self) -> dict[str, str]:
        return self.proxy_current_sign_convention.metadata_fields()

    def proxy_current_sign_summary_fields(self) -> dict[str, str]:
        return self.proxy_current_sign_convention.summary_fields(
            mode=self.mode,
            scalar_policy=self.proxy_vf_current_scalar_policy,
        )

    def proxy_current_sign_help_line(self) -> str:
        return self.proxy_current_sign_convention.help_text(
            mode=self.mode,
            scalar_policy=self.proxy_vf_current_scalar_policy,
        )

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

VACUUM_PROFILE = FiniteCurrentProfile(
    mode=VACUUM_FINITE_CURRENT_MODE,
    default_num_tf_coils=20,
    default_num_banana_coils=10,
    default_num_proxy_coils=0,
    default_num_vf_coils=0,
    boozer_current_convention=resolve_boozer_current_convention(
        VACUUM_FINITE_CURRENT_MODE,
    ),
    g0_policy="signed_explicit_tf_current",
    proxy_placement_policy="none",
    proxy_vf_current_scalar_policy="none",
    default_vf_template_path=None,
    vf_template_sha256=None,
    vf_current_ratio=0.0,
    vf_current_sign_policy="none",
    vf_current_mutability="none",
    banana_replay_policy="stage2_cli_seed_current",
    proxy_current_sign_convention=ProxyCurrentSignConvention(
        key="none",
        frame="no_proxy_current",
        signedness="none",
        replay_reference="vacuum_tf_banana_only",
        operator_warning=(
            "vacuum mode has no proxy/VF current; do not pass a proxy-current scalar."
        ),
    ),
    supported_entrypoints=(
        "STAGE_2/banana_coil_solver.py",
        "SINGLE_STAGE/single_stage_banana_example.py",
        "run_stage2_to_single_stage.py",
    ),
    rejected_entrypoints=(),
    required_artifact_metadata_keys=_COMMON_REQUIRED_ARTIFACT_METADATA_KEYS,
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
    proxy_current_sign_convention=ProxyCurrentSignConvention(
        key="wataru_nonnegative_proxy_vf_magnitude",
        frame="hbt_wataru_proxy_vf_magnitude",
        signedness="nonnegative_magnitude",
        replay_reference="HBT/Wataru proxy/VF replay contract",
        operator_warning=(
            "wataru_proxy_field accepts nonnegative proxy/VF current "
            "magnitudes; do not infer these signs from TF/banana currents."
        ),
    ),
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
    proxy_current_sign_convention=ProxyCurrentSignConvention(
        key="jhalpern30_signed_upstream_proxy_loop",
        frame="upstream_jhalpern30_proxy_loop",
        signedness="signed_physical_scalar",
        replay_reference="banana_drivers generate_proxy_coils xc(1)=R ys(1)=R",
        operator_warning=(
            "jhalpern30_proxy_field accepts a signed upstream proxy-loop "
            "scalar; do not infer its sign from TF/banana currents or flip "
            "the local proxy winding."
        ),
        empirical_effect_note=(
            "Empirical sign probe: positive raises iota; negative is "
            "counter-current/collapse."
        ),
    ),
    supported_entrypoints=(
        "import_jhalpern30_replay.py",
        "STAGE_2/banana_coil_solver.py",
        "SINGLE_STAGE/single_stage_banana_example.py",
    ),
    rejected_entrypoints=("run_stage2_to_single_stage.py:pre_boozer_repair",),
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
    VACUUM_PROFILE.mode: VACUUM_PROFILE,
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


def format_proxy_current_sign_convention_help(
    modes: tuple[FiniteCurrentMode, ...],
) -> str:
    lines = ["Proxy-current sign conventions:"]
    lines.extend(
        get_finite_current_profile(mode).proxy_current_sign_help_line()
        for mode in modes
    )
    return "\n".join(lines)


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
    "ProxyCurrentSignConvention",
    "VACUUM_FINITE_CURRENT_MODE",
    "VACUUM_PROFILE",
    "WATARU_FINITE_CURRENT_MODE",
    "WATARU_PROFILE",
    "format_proxy_current_sign_convention_help",
    "get_finite_current_profile",
]
