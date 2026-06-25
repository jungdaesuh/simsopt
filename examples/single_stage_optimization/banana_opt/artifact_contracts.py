from __future__ import annotations

from .current_contracts import BANANA_CURRENT_HARD_LIMIT_A
from .hardware_contracts import fixed_stage2_artifact_hardware_contract

DEFAULT_LEGACY_BANANA_INIT_CURRENT_A = 1.0e4


def _upgrade_legacy_banana_current_metadata(
    upgraded_results: dict[str, object],
) -> None:
    banana_current_A = upgraded_results.get("BANANA_CURRENT_A")
    stage2_seed_path = upgraded_results.get("STAGE2_BS_PATH")
    if upgraded_results.get("BANANA_INIT_CURRENT_A") is None:
        if stage2_seed_path in {None, ""}:
            upgraded_results["BANANA_INIT_CURRENT_A"] = (
                DEFAULT_LEGACY_BANANA_INIT_CURRENT_A
            )
        elif upgraded_results.get("init_only") and banana_current_A is not None:
            upgraded_results["BANANA_INIT_CURRENT_A"] = float(banana_current_A)
    if upgraded_results.get("BANANA_CURRENT_MAX_A") is None:
        realized_current_abs_A = (
            0.0 if banana_current_A is None else abs(float(banana_current_A))
        )
        upgraded_results["BANANA_CURRENT_MAX_A"] = max(
            BANANA_CURRENT_HARD_LIMIT_A,
            realized_current_abs_A,
        )


def _upgrade_legacy_stage2_hardware_contract_metadata(
    upgraded_results: dict[str, object],
) -> None:
    for key, value in fixed_stage2_artifact_hardware_contract().items():
        if upgraded_results.get(key) is None:
            upgraded_results[key] = float(value)


def upgrade_legacy_stage2_artifact_results(
    stage2_artifact_results: dict[str, object],
    *,
    known_num_tf_coils: int | None = None,
    known_tf_current_A: float | None = None,
) -> dict[str, object]:
    upgraded_results = dict(stage2_artifact_results)
    if upgraded_results.get("TF_CURRENT_A") is None and known_tf_current_A is not None:
        upgraded_results["TF_CURRENT_A"] = float(known_tf_current_A)
    if upgraded_results.get("NUM_TF_COILS") is None and known_num_tf_coils is not None:
        upgraded_results["NUM_TF_COILS"] = int(known_num_tf_coils)
    if upgraded_results.get("TF_CURRENT_SUM_ABS_A") is None:
        tf_current_A = upgraded_results.get("TF_CURRENT_A")
        num_tf_coils = upgraded_results.get("NUM_TF_COILS")
        if tf_current_A is not None and num_tf_coils is not None:
            upgraded_results["TF_CURRENT_SUM_ABS_A"] = abs(float(tf_current_A)) * float(
                num_tf_coils
            )
    _upgrade_legacy_banana_current_metadata(upgraded_results)
    _upgrade_legacy_stage2_hardware_contract_metadata(upgraded_results)
    return upgraded_results
