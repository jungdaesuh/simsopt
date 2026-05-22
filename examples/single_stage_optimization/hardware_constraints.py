import math

import jax
import numpy as np

from simsopt._core.jax_host_boundary import host_array


_ACCEPTED_OPTIMIZER_STATUSES = frozenset({0, 4})


def apply_hardware_constraint_verdict(
    optimizer_success,
    termination_message,
    hardware_status,
    *,
    init_only,
):
    """Apply hardware-constraint failure semantics to a run result."""
    if init_only or hardware_status["success"]:
        return bool(optimizer_success), termination_message
    if termination_message:
        termination_message = f"{termination_message}; hardware_constraints_failed"
    else:
        termination_message = "hardware_constraints_failed"
    return False, termination_message


def sanitize_diagnostic_payload(value):
    """Convert diagnostic payloads to JSON-safe data, masking NaN/Inf as null."""
    if isinstance(value, dict):
        return {key: sanitize_diagnostic_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_diagnostic_payload(item) for item in value]
    if isinstance(value, jax.Array):
        return sanitize_diagnostic_payload(host_array(value))
    if isinstance(value, np.ndarray):
        return sanitize_diagnostic_payload(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def strict_accepted_payload(value, *, path="payload"):
    """Convert accepted artifact payloads to JSON data and reject NaN/Inf."""
    if isinstance(value, dict):
        return {
            key: strict_accepted_payload(item, path=f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            strict_accepted_payload(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return strict_accepted_payload(value.tolist(), path=path)
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite for an accepted artifact.")
    return value


def _finite_value(value):
    if value is None:
        return False
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, (list, tuple)):
        return bool(np.all(np.isfinite(np.asarray(value))))
    if isinstance(value, (float, int)):
        return math.isfinite(value)
    return False


def _value_at_path(payload, path):
    value = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def accepted_result_rejection_reasons(
    payload,
    *,
    optimizer_success,
    optimizer_status,
    final_objective,
    final_dofs,
    final_gradient=None,
    final_gradient_finite=None,
    require_gradient,
    required_metric_paths=(),
    required_metadata_paths=(),
):
    """Return acceptance-gate failures for a result artifact payload."""
    reasons = []
    if optimizer_success is not True:
        reasons.append("optimizer_success_not_true")
    if optimizer_status is not None:
        status = int(optimizer_status)
        if status not in _ACCEPTED_OPTIMIZER_STATUSES:
            reasons.append(f"optimizer_status_{status}")
    if not _finite_value(final_objective):
        reasons.append("final_objective_nonfinite")
    if not _finite_value(final_dofs):
        reasons.append("final_dofs_nonfinite")
    if require_gradient:
        if final_gradient is not None:
            gradient_is_finite = _finite_value(final_gradient)
        else:
            gradient_is_finite = final_gradient_finite is True
        if not gradient_is_finite:
            reasons.append("final_gradient_nonfinite")
    for metric_path in required_metric_paths:
        metric = _value_at_path(payload, tuple(metric_path))
        if not _finite_value(metric):
            reasons.append(f"required_metric_nonfinite:{'.'.join(metric_path)}")
    for metadata_path in required_metadata_paths:
        metadata = _value_at_path(payload, tuple(metadata_path))
        if metadata is None:
            reasons.append(f"required_metadata_missing:{'.'.join(metadata_path)}")
    return reasons


def accepted_result_payload(
    payload,
    *,
    optimizer_success,
    optimizer_status,
    final_objective,
    final_dofs,
    final_gradient=None,
    final_gradient_finite=None,
    require_gradient,
    required_metric_paths=(),
    required_metadata_paths=(),
):
    """Return strict JSON payload data after proving accepted-result gates."""
    reasons = accepted_result_rejection_reasons(
        payload,
        optimizer_success=optimizer_success,
        optimizer_status=optimizer_status,
        final_objective=final_objective,
        final_dofs=final_dofs,
        final_gradient=final_gradient,
        final_gradient_finite=final_gradient_finite,
        require_gradient=require_gradient,
        required_metric_paths=required_metric_paths,
        required_metadata_paths=required_metadata_paths,
    )
    if reasons:
        raise ValueError(
            "Refusing accepted artifact because result gates failed: "
            + ", ".join(reasons)
        )
    return strict_accepted_payload(payload)
