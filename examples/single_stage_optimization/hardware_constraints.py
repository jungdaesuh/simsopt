import math

import numpy as np


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


def sanitize_json_payload(value):
    """Convert payloads into JSON-safe Python data without NaN/Inf floats."""
    if isinstance(value, dict):
        return {key: sanitize_json_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json_payload(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
