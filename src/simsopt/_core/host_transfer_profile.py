"""Process-local host<->device transfer profiling counters."""

from __future__ import annotations

import os
from time import perf_counter

_ENV_VAR = "SIMSOPT_PROFILE_HOST_TRANSFERS"
_ENABLED = os.environ.get(_ENV_VAR) == "1"

_DEVICE_TO_HOST_COUNT = 0
_DEVICE_TO_HOST_SECONDS = 0.0
_HOST_TO_DEVICE_COUNT = 0
_HOST_TO_DEVICE_SECONDS = 0.0


def host_transfer_profiling_enabled() -> bool:
    return _ENABLED


def transfer_perf_start() -> float | None:
    return perf_counter() if _ENABLED else None


def record_device_to_host(start: float | None) -> None:
    if start is None:
        return
    global _DEVICE_TO_HOST_COUNT, _DEVICE_TO_HOST_SECONDS
    _DEVICE_TO_HOST_COUNT += 1
    _DEVICE_TO_HOST_SECONDS += perf_counter() - start


def record_host_to_device(start: float | None) -> None:
    if start is None:
        return
    global _HOST_TO_DEVICE_COUNT, _HOST_TO_DEVICE_SECONDS
    _HOST_TO_DEVICE_COUNT += 1
    _HOST_TO_DEVICE_SECONDS += perf_counter() - start


def snapshot() -> dict[str, float]:
    return {
        "device_to_host_count": _DEVICE_TO_HOST_COUNT,
        "device_to_host_seconds": _DEVICE_TO_HOST_SECONDS,
        "host_to_device_count": _HOST_TO_DEVICE_COUNT,
        "host_to_device_seconds": _HOST_TO_DEVICE_SECONDS,
    }


def reset() -> None:
    global _DEVICE_TO_HOST_COUNT, _DEVICE_TO_HOST_SECONDS
    global _HOST_TO_DEVICE_COUNT, _HOST_TO_DEVICE_SECONDS
    _DEVICE_TO_HOST_COUNT = 0
    _DEVICE_TO_HOST_SECONDS = 0.0
    _HOST_TO_DEVICE_COUNT = 0
    _HOST_TO_DEVICE_SECONDS = 0.0
