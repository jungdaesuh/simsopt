from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Mapping, Sequence


CURTOR_SIGN_SCHEMA_VERSION: Final[int] = 1
DEFAULT_CURTOR_SIGN_LADDER_A: Final[tuple[float, ...]] = (
    0.0,
    -200.0,
    -400.0,
    -800.0,
    200.0,
    400.0,
    800.0,
)
DEFAULT_CURTOR_CONVERGENCE_NS: Final[tuple[int, ...]] = (51, 99, 151)
_BASE_VMEC_NS_RAMP: Final[tuple[int, ...]] = (13, 25, 51, 99, 151)


def radial_schedule_for_final_ns(final_ns: int) -> tuple[int, ...]:
    requested_ns = int(final_ns)
    if requested_ns not in _BASE_VMEC_NS_RAMP:
        raise ValueError(f"Unsupported VMEC final ns {requested_ns}.")
    return tuple(ns for ns in _BASE_VMEC_NS_RAMP if ns <= requested_ns)


def niter_schedule_for_ns(ns_schedule: Sequence[int]) -> tuple[int, ...]:
    return tuple(4000 if int(ns) < 51 else 6000 for ns in ns_schedule)


def ftol_schedule_for_ns(ns_schedule: Sequence[int]) -> tuple[float, ...]:
    return tuple(1.0e-10 for _ns in ns_schedule)


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, object]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def write_curtor_sign_json(
    path: str | Path,
    artifact: Mapping[str, object],
) -> None:
    Path(path).write_text(
        json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_curtor_sign_artifact(
    *,
    sign_records: Sequence[Mapping[str, object]],
    convergence_records: Sequence[Mapping[str, object]],
    source_input: str,
    current_magnitude_A: float,
    convergence_ns_values: Sequence[int] = DEFAULT_CURTOR_CONVERGENCE_NS,
) -> dict[str, object]:
    lift_records = _lift_records(sign_records)
    raising = _select_iota_raising_record(lift_records)
    ns_values = tuple(int(ns) for ns in convergence_ns_values)
    return {
        "schema_version": CURTOR_SIGN_SCHEMA_VERSION,
        "source_input": str(source_input),
        "vmec_current_contract": {
            "ncurr": 1,
            "ac0": 1.0,
            "curtor_scaled_quantity": "edge enclosed toroidal current",
            "need_to_run_code": True,
        },
        "sign_probe": {
            "current_magnitude_A": abs(float(current_magnitude_A)),
            "iota_raising_curtor_sign": int(raising["curtor_sign"]),
            "iota_raising_curtor_A": float(raising["curtor_A"]),
            "validated_edge_lift": float(raising["lift_edge"]),
            "validated_axis_lift": float(raising["lift_axis"]),
            "lift_records": lift_records,
        },
        "convergence": {
            "ns_values": list(ns_values),
            "reported_metric": "iota_edge_lift_only",
            "edge_lifts": (
                []
                if not convergence_records
                else _edge_lift_convergence(
                    convergence_records,
                    ns_values=ns_values,
                )
            ),
        },
    }


def _lift_records(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    baseline = _baseline_record(records)
    baseline_axis = _required_float(baseline, "iota_axis")
    baseline_edge = _required_float(baseline, "iota_edge")
    lift_records = []
    for record in records:
        curtor_A = _required_float(record, "curtor_A")
        if not bool(record.get("converged", True)):
            continue
        iota_axis = _required_float(record, "iota_axis")
        iota_edge = _required_float(record, "iota_edge")
        lift_records.append(
            {
                "curtor_A": curtor_A,
                "curtor_sign": _current_sign(curtor_A),
                "iota_axis": iota_axis,
                "iota_edge": iota_edge,
                "lift_axis": iota_axis - baseline_axis,
                "lift_edge": iota_edge - baseline_edge,
            }
        )
    return lift_records


def _edge_lift_convergence(
    records: Sequence[Mapping[str, object]],
    *,
    ns_values: Sequence[int],
) -> list[dict[str, object]]:
    rows = []
    for ns in ns_values:
        baseline = _record_for_ns_and_curtor(records, ns=ns, curtor_A=0.0)
        branch = _max_positive_edge_lift_record_for_ns(records, ns=ns, baseline=baseline)
        rows.append(
            {
                "ns": int(ns),
                "curtor_A": _required_float(branch, "curtor_A"),
                "edge_lift": _required_float(branch, "iota_edge")
                - _required_float(baseline, "iota_edge"),
            }
        )
    return rows


def _baseline_record(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    for record in records:
        if _required_float(record, "curtor_A") == 0.0:
            return record
    raise ValueError("curtor sign artifact requires a curtor_A=0 baseline.")


def _record_for_ns_and_curtor(
    records: Sequence[Mapping[str, object]],
    *,
    ns: int,
    curtor_A: float,
) -> Mapping[str, object]:
    for record in records:
        if int(_required_float(record, "vmec_final_ns")) == int(ns):
            if _required_float(record, "curtor_A") == float(curtor_A):
                return record
    raise ValueError(f"Missing VMEC convergence record for ns={ns}, curtor={curtor_A}.")


def _max_positive_edge_lift_record_for_ns(
    records: Sequence[Mapping[str, object]],
    *,
    ns: int,
    baseline: Mapping[str, object],
) -> Mapping[str, object]:
    baseline_edge = _required_float(baseline, "iota_edge")
    candidates = [
        record
        for record in records
        if int(_required_float(record, "vmec_final_ns")) == int(ns)
        and _required_float(record, "curtor_A") != 0.0
    ]
    if not candidates:
        raise ValueError(f"Missing nonzero VMEC convergence record for ns={ns}.")
    return max(
        candidates,
        key=lambda record: _required_float(record, "iota_edge") - baseline_edge,
    )


def _select_iota_raising_record(
    lift_records: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    candidates = [
        record
        for record in lift_records
        if _required_float(record, "curtor_A") != 0.0
    ]
    if not candidates:
        raise ValueError("curtor sign artifact requires nonzero curtor records.")
    raising = max(candidates, key=lambda record: _required_float(record, "lift_edge"))
    if _required_float(raising, "lift_edge") <= 0.0:
        raise ValueError("No curtor branch raises iota_edge above baseline.")
    return raising


def _current_sign(value: float) -> int:
    current = float(value)
    if current > 0.0:
        return 1
    if current < 0.0:
        return -1
    return 0


def _required_float(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if value is None or value == "":
        raise ValueError(f"Missing required curtor artifact field {key}.")
    return float(value)
