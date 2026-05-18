from dataclasses import dataclass
import math
from pathlib import Path
from typing import Literal

from scipy.io import netcdf_file


WoutLane = Literal["signed_cw", "positive_ccw"]
SIGNED_CW: WoutLane = "signed_cw"
POSITIVE_CCW: WoutLane = "positive_ccw"


class WoutConventionReadError(ValueError):
    """The wout cannot provide the convention fields required for handoff."""


class WoutCorruptError(ValueError):
    """The wout carries internally inconsistent convention signs."""


@dataclass(frozen=True)
class WoutConvention:
    lane: WoutLane
    rbtor: float
    phi_edge: float


def _signed_lane(value: float, *, field_name: str) -> WoutLane:
    if not math.isfinite(value):
        raise WoutConventionReadError(f"{field_name} is not finite")
    if value < 0.0:
        return SIGNED_CW
    if value > 0.0:
        return POSITIVE_CCW
    raise WoutConventionReadError(f"{field_name} has zero sign")


def _read_variable_data(handle: netcdf_file, name: str):
    if name not in handle.variables:
        raise WoutConventionReadError(f"missing {name}")
    return handle.variables[name].data.copy()


def _read_scalar(handle: netcdf_file, name: str) -> float:
    data = _read_variable_data(handle, name)
    if data.size != 1:
        raise WoutConventionReadError(f"{name} is not scalar")
    return float(data.reshape(-1)[0])


def _read_edge_value(handle: netcdf_file, name: str) -> float:
    data = _read_variable_data(handle, name)
    if data.shape == ():
        raise WoutConventionReadError(f"{name} is scalar")
    if data.ndim != 1:
        raise WoutConventionReadError(f"{name} is not 1-D")
    return float(data[-1])


def classify_wout_convention(wout_path: str | Path) -> WoutConvention:
    path = Path(wout_path)
    try:
        with netcdf_file(str(path), "r", mmap=False) as handle:
            rbtor = _read_scalar(handle, "rbtor")
            phi_edge = _read_edge_value(handle, "phi")
    except (FileNotFoundError, OSError) as exc:
        raise WoutConventionReadError(f"cannot read wout: {path}") from exc

    rbtor_lane = _signed_lane(rbtor, field_name="rbtor")
    phi_lane = _signed_lane(phi_edge, field_name="phi_edge")
    if rbtor_lane != phi_lane:
        raise WoutCorruptError(
            f"wout convention signs disagree: rbtor={rbtor:g}, phi_edge={phi_edge:g}"
        )
    return WoutConvention(lane=rbtor_lane, rbtor=rbtor, phi_edge=phi_edge)


def expected_wout_lane_for_tf_current(tf_current_A: float) -> WoutLane:
    current = float(tf_current_A)
    if not math.isfinite(current):
        raise ValueError("tf_current_A must be finite")
    if current < 0.0:
        return SIGNED_CW
    if current > 0.0:
        return POSITIVE_CCW
    raise ValueError("tf_current_A must have nonzero sign")


def wout_convention_artifact_fields(
    *,
    wout_path: str | Path,
    tf_current_A: float,
) -> dict[str, object]:
    convention = classify_wout_convention(wout_path)
    expected_lane = expected_wout_lane_for_tf_current(tf_current_A)
    return {
        "WOUT_CONVENTION": convention.lane,
        "WOUT_OFF_SPEC": convention.lane != expected_lane,
    }


def validate_wout_convention_artifact_fields(
    *,
    stage2_results_path: str | Path,
    stage2_artifact_results: dict,
) -> None:
    path = Path(stage2_results_path)
    required = ("PLASMA_SURF_PATH", "TF_CURRENT_A", "WOUT_CONVENTION", "WOUT_OFF_SPEC")
    missing = [key for key in required if key not in stage2_artifact_results]
    if missing:
        raise ValueError(
            f"{path} missing WOUT convention metadata: {', '.join(missing)}"
        )
    wout_off_spec = stage2_artifact_results["WOUT_OFF_SPEC"]
    if wout_off_spec is not True and wout_off_spec is not False:
        raise ValueError(f"{path} WOUT_OFF_SPEC must be boolean.")

    expected_fields = wout_convention_artifact_fields(
        wout_path=stage2_artifact_results["PLASMA_SURF_PATH"],
        tf_current_A=float(stage2_artifact_results["TF_CURRENT_A"]),
    )
    for key, expected_value in expected_fields.items():
        if stage2_artifact_results[key] != expected_value:
            raise ValueError(
                f"{path} {key}={stage2_artifact_results[key]!r} "
                f"does not match wout convention {expected_value!r}."
            )
