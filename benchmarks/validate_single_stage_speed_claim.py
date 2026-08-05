"""Machine-checkable definition of done for the single-stage speed campaign.

Recomputes every median, ratio, and parity comparison from raw per-sample
receipt records; parity tolerances are re-derived from constants deliberately
duplicated from the frozen SSOT rather than read from the receipt. Enforces
the frozen-files contract against the campaign baseline tag. Exit codes:
0 = WIN, 3 = certified TIE, 1 = LOSS, 2 = integrity error (missing,
malformed, inconsistent, or non-finite receipt evidence, or frozen-file
drift). Every unexpected exception is an integrity error, never a LOSS.

TIE evidence lives OUTSIDE the immutable receipt root, in the sibling
directory "<artifact-root>.profile" (at least one non-empty regular file).

Protocol: docs/single_stage_speed_campaign_protocol.md. This file is owned by
the campaign auditor; the campaign worker runs it but must never edit it.

v2 (amendment r2): per-lane backend/driver provenance, required
parity-observable coverage, cross-lane identity of the initial point and
input/configuration fingerprints.
v3 (amendment r5): monotone verdict partition (TIE = any failing ratio at or
under 1.05 with profile evidence; no dead zone), parity recomputation from
duplicated SSOT constants referenced to the native value, native-row
cross-binding, wall-clock versus trajectory consistency, full-budget
trajectory contiguity, effective-construction fingerprint in the shared
identity, adjoint-route presence for custom lanes, four-lane schema presence,
finite-ratio guards, and integrity-safe exception handling.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BASELINE_TAG = "campaign-20260804-frozen-r5"
FROZEN_PATHS = (
    "benchmarks/validate_single_stage_speed_claim.py",
    "docs/single_stage_speed_campaign_protocol.md",
    "src/simsopt_jax/parity_tolerances.py",
    "src/simsopt/optimization_endpoint.py",
    "src/simsopt_jax/solve/endpoint_certificate.py",
    "examples/jax/parity/arbiter.py",
)
NATIVE_LANE = "native_cpu"
GATED_LANES = ("native_cpu", "jax_gpu_custom", "jax_gpu_optax")
ALL_RECEIPT_LANES = ("native_cpu", "jax_gpu_custom", "jax_gpu_optax", "jax_cpu_custom")
# Deliberate duplication of the producer's identity table and of the frozen
# tolerance SSOT (src/simsopt_jax/parity_tolerances.py, key
# mirror_single_stage_final_value): the validator must assert provenance and
# re-derive parity bounds without importing worker-owned code.
EXPECTED_LANE_IDENTITIES = {
    "native_cpu": ("native_cpu", "simsopt_scipy_bfgs_with_boozer_newton"),
    "jax_gpu_custom": (
        "jax_gpu_fast",
        "simsopt_jax_host_lbfgsb_with_traceable_boozer_newton",
    ),
    "jax_gpu_optax": (
        "jax_gpu_fast",
        "simsopt_jax_optax_lbfgs_with_traceable_boozer_newton",
    ),
}
EXPECTED_PARITY_RTOL = 2e-8
EXPECTED_PARITY_ATOL = 2e-12
REQUIRED_PARITY_OBSERVABLES = frozenset(
    {
        "final_objective",
        "final_iota",
        "final_volume",
        "final_non_qs_ratio",
        "final_boozer_residual",
    }
)
SHARED_AUDIT_KEYS = (
    "initial_parameters_sha256",
    "input_fingerprint",
    "configuration_fingerprint",
    "effective_construction_fingerprint",
)
ADJOINT_ROUTE_LANES = ("jax_gpu_custom", "jax_gpu_optax")
SPEED_MARGIN = 0.90
TIE_MAX_RATIO = 1.05
QUALITY_SLACK = 1.001
EXPECTED_WARM_SAMPLES = 7
FINAL_WARM_SAMPLE_INDEX = 6
ENDPOINT_TRAJECTORY_REL_TOL = 1e-6
WALL_CONSISTENCY_SLACK = 1.001
PROFILE_EVIDENCE_SUFFIX = ".profile"
CAMPAIGN_REQUIRED_KEYS = (
    "campaign_id",
    "git_describe",
    "hostname",
    "device_name",
    "python_version",
    "jax_version",
    "iteration_budget",
    "scale",
    "created_utc",
)


class IntegrityError(Exception):
    """Receipt bytes cannot support any verdict (missing, malformed, drifted)."""


@dataclass(frozen=True)
class LaneEvidence:
    lane_id: str
    warm_wall_seconds: tuple[float, ...]
    warm_final_objectives: tuple[float, ...]
    warm_trajectories: tuple[tuple[tuple[int, float, float], ...], ...]
    endpoint: dict
    shared_audit: tuple[str, ...]


def _fail(message: str) -> IntegrityError:
    return IntegrityError(message)


def check_frozen_files(repo_root: Path, baseline_tag: str) -> None:
    """Byte-compare each frozen path against the baseline tag's blob.

    Uses content hashes rather than `git diff` so untracked-but-snapshotted
    files are compared correctly.
    """
    drifted = []
    for relative_path in FROZEN_PATHS:
        baseline = subprocess.run(
            ["git", "rev-parse", f"{baseline_tag}:{relative_path}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if baseline.returncode != 0:
            raise _fail(f"{relative_path} missing from baseline {baseline_tag}")
        current = subprocess.run(
            ["git", "hash-object", "--path", relative_path, "--", relative_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if current.returncode != 0:
            raise _fail(f"{relative_path} missing from working tree")
        if baseline.stdout.strip() != current.stdout.strip():
            drifted.append(relative_path)
    if drifted:
        raise _fail(f"frozen files drifted from {baseline_tag}: {drifted}")


def load_campaign(artifact_root: Path) -> dict:
    campaign_path = artifact_root / "campaign.json"
    if not campaign_path.is_file():
        raise _fail(f"missing {campaign_path}")
    campaign = json.loads(campaign_path.read_text())
    missing = [key for key in CAMPAIGN_REQUIRED_KEYS if not campaign.get(key)]
    if missing:
        raise _fail(f"campaign.json missing/empty keys: {missing}")
    return campaign


def _load_trajectory(
    path: Path, iteration_budget: int
) -> tuple[tuple[int, float, float], ...]:
    if not path.is_file():
        raise _fail(f"missing trajectory {path}")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        record = (
            int(row["iteration"]),
            float(row["objective"]),
            float(row["wall_seconds_from_start"]),
        )
        if not math.isfinite(record[1]) or not math.isfinite(record[2]):
            raise _fail(f"non-finite trajectory record {path}:{line_number}")
        records.append(record)
    if not records:
        raise _fail(f"empty trajectory {path}")
    for position, record in enumerate(records, start=1):
        if record[0] != position:
            raise _fail(
                f"trajectory {path} not contiguous from 1 at position {position}"
            )
    for previous, current in zip(records, records[1:]):
        if current[2] < previous[2]:
            raise _fail(
                f"trajectory {path} wall time decreases at iteration {current[0]}"
            )
    if records[-1][0] != iteration_budget:
        raise _fail(
            f"trajectory {path} ended at iteration {records[-1][0]}, not the "
            f"fixed budget {iteration_budget}"
        )
    return tuple(records)


def load_lane(artifact_root: Path, lane_id: str, iteration_budget: int) -> LaneEvidence:
    lane_dir = artifact_root / "lanes" / lane_id
    measurement_path = lane_dir / "measurement.json"
    endpoint_path = lane_dir / "endpoint.json"
    for path in (measurement_path, endpoint_path):
        if not path.is_file():
            raise _fail(f"missing {path}")
    measurement = json.loads(measurement_path.read_text())
    endpoint = json.loads(endpoint_path.read_text())

    warm_samples = [
        sample
        for sample in measurement.get("samples", [])
        if sample.get("phase") == "warm"
    ]
    if len(warm_samples) != EXPECTED_WARM_SAMPLES:
        raise _fail(
            f"{lane_id}: expected {EXPECTED_WARM_SAMPLES} warm samples, "
            f"found {len(warm_samples)}"
        )

    walls: list[float] = []
    finals: list[float] = []
    trajectories: list[tuple[tuple[int, float, float], ...]] = []
    final_sample_objective: float | None = None
    for sample in warm_samples:
        wall = float(sample["wall_seconds"])
        if not math.isfinite(wall) or wall <= 0.0:
            raise _fail(f"{lane_id}: invalid warm wall_seconds {wall}")
        walls.append(wall)
        sample_index = int(sample["sample_index"])
        trajectory = _load_trajectory(
            lane_dir / f"trajectory-warm-{sample_index}.jsonl", iteration_budget
        )
        if trajectory[-1][2] > wall * WALL_CONSISTENCY_SLACK:
            raise _fail(
                f"{lane_id}: warm sample {sample_index} trajectory wall "
                f"{trajectory[-1][2]} exceeds the recorded sample wall {wall}"
            )
        trajectories.append(trajectory)
        finals.append(trajectory[-1][1])
        if sample_index == FINAL_WARM_SAMPLE_INDEX:
            final_sample_objective = trajectory[-1][1]

    if final_sample_objective is None:
        raise _fail(
            f"{lane_id}: warm sample {FINAL_WARM_SAMPLE_INDEX} (the endpoint "
            "sample) is missing"
        )

    observables = endpoint.get("observables", {})
    if observables.get("inner_solver_success") is not True:
        raise _fail(f"{lane_id}: inner solver not successful in endpoint.json")
    if str(endpoint.get("precision")) != "fp64":
        raise _fail(f"{lane_id}: precision is not fp64")
    audit = endpoint.get("audit", {})
    if lane_id in EXPECTED_LANE_IDENTITIES:
        expected_backend_mode, expected_driver = EXPECTED_LANE_IDENTITIES[lane_id]
        if audit.get("backend_mode") != expected_backend_mode:
            raise _fail(
                f"{lane_id}: backend_mode {audit.get('backend_mode')!r} != "
                f"expected {expected_backend_mode!r}"
            )
        if audit.get("driver") != expected_driver:
            raise _fail(
                f"{lane_id}: driver {audit.get('driver')!r} != "
                f"expected {expected_driver!r}"
            )
    if lane_id in ADJOINT_ROUTE_LANES:
        adjoint_route = audit.get("adjoint_route")
        if not isinstance(adjoint_route, str) or not adjoint_route:
            raise _fail(f"{lane_id}: audit.adjoint_route missing or empty")
    shared_audit_values = []
    for key in SHARED_AUDIT_KEYS:
        value = audit.get(key)
        if not isinstance(value, str) or not value:
            raise _fail(f"{lane_id}: audit.{key} missing or empty")
        shared_audit_values.append(value)
    endpoint_final = float(observables["final_objective"])
    if not math.isfinite(endpoint_final):
        raise _fail(f"{lane_id}: non-finite endpoint final objective")
    denominator = max(abs(endpoint_final), abs(final_sample_objective), 1e-300)
    if (
        abs(endpoint_final - final_sample_objective) / denominator
        > ENDPOINT_TRAJECTORY_REL_TOL
    ):
        raise _fail(
            f"{lane_id}: endpoint final objective {endpoint_final} disagrees "
            f"with warm sample {FINAL_WARM_SAMPLE_INDEX} trajectory final "
            f"{final_sample_objective}"
        )
    return LaneEvidence(
        lane_id=lane_id,
        warm_wall_seconds=tuple(walls),
        warm_final_objectives=tuple(finals),
        warm_trajectories=tuple(trajectories),
        endpoint=endpoint,
        shared_audit=tuple(shared_audit_values),
    )


def check_parity(lane: LaneEvidence, native_observables: dict) -> None:
    """Recompute each parity comparison from the frozen tolerance constants.

    The receipt's own tolerance field is required sane but never trusted; the
    bound is re-derived from the native value, and each row's native value is
    cross-bound to the native lane's endpoint observables.
    """
    rows = lane.endpoint.get("parity", {}).get("rows", [])
    if not rows:
        raise _fail(f"{lane.lane_id}: no parity rows in endpoint.json")
    covered = {row.get("observable") for row in rows}
    missing = REQUIRED_PARITY_OBSERVABLES - covered
    if missing:
        raise _fail(f"{lane.lane_id}: parity rows missing {sorted(missing)}")
    for row in rows:
        observable = row.get("observable", "<unnamed>")
        native_value = float(row["native_value"])
        lane_value = float(row["lane_value"])
        recorded_tolerance = float(row["tolerance"])
        if not (
            math.isfinite(native_value)
            and math.isfinite(lane_value)
            and math.isfinite(recorded_tolerance)
            and recorded_tolerance >= 0.0
        ):
            raise _fail(f"{lane.lane_id}: malformed parity row {observable}")
        if observable in native_observables and native_value != float(
            native_observables[observable]
        ):
            raise _fail(
                f"{lane.lane_id}: parity row {observable} native_value "
                f"{native_value} does not match the native endpoint "
                f"{native_observables[observable]}"
            )
        expected_tolerance = (
            EXPECTED_PARITY_ATOL + EXPECTED_PARITY_RTOL * abs(native_value)
        )
        if abs(lane_value - native_value) > expected_tolerance:
            raise _fail(
                f"{lane.lane_id}: parity failure on {observable}: "
                f"|{lane_value} - {native_value}| > {expected_tolerance} "
                "(recomputed from the frozen tolerance constants)"
            )


def time_to_quality(
    trajectory: tuple[tuple[int, float, float], ...], j_target: float
) -> float:
    for _, objective, wall_seconds in trajectory:
        if objective <= j_target:
            return wall_seconds
    return math.inf


def _profile_evidence(artifact_root: Path) -> list[Path]:
    profile_dir = artifact_root.parent / (artifact_root.name + PROFILE_EVIDENCE_SUFFIX)
    if not profile_dir.is_dir():
        return []
    return sorted(
        path
        for path in profile_dir.iterdir()
        if path.is_file() and path.stat().st_size > 0
    )


def evaluate(artifact_root: Path, repo_root: Path, baseline_tag: str) -> int:
    check_frozen_files(repo_root, baseline_tag)
    campaign = load_campaign(artifact_root)
    iteration_budget = int(campaign["iteration_budget"])
    if campaign["scale"] != "native_default":
        raise _fail(f"scale is {campaign['scale']!r}, not 'native_default'")

    for lane_id in ALL_RECEIPT_LANES:
        if not (artifact_root / "lanes" / lane_id).is_dir():
            raise _fail(f"receipt lane directory missing: {lane_id}")

    lanes = {
        lane_id: load_lane(artifact_root, lane_id, iteration_budget)
        for lane_id in GATED_LANES
    }
    native = lanes[NATIVE_LANE]
    native_observables = {
        key: value
        for key, value in native.endpoint.get("observables", {}).items()
        if isinstance(value, (int, float))
    }
    for lane_id in GATED_LANES:
        if lane_id != NATIVE_LANE:
            check_parity(lanes[lane_id], native_observables)
    native_shared_audit = native.shared_audit
    for lane_id in GATED_LANES:
        if lanes[lane_id].shared_audit != native_shared_audit:
            raise _fail(
                f"{lane_id}: shared audit identity "
                f"{dict(zip(SHARED_AUDIT_KEYS, lanes[lane_id].shared_audit))} "
                "does not match the native lane (initial point or input "
                "bundle differs)"
            )

    native_final_median = statistics.median(native.warm_final_objectives)
    if not (math.isfinite(native_final_median) and native_final_median > 0.0):
        raise _fail(
            f"native warm-final median {native_final_median} violates the "
            "positive-objective protocol assumption"
        )
    j_target = QUALITY_SLACK * native_final_median

    ttq: dict[str, float] = {}
    warm_median: dict[str, float] = {}
    final_median: dict[str, float] = {}
    for lane_id, lane in lanes.items():
        ttq[lane_id] = statistics.median(
            time_to_quality(trajectory, j_target)
            for trajectory in lane.warm_trajectories
        )
        warm_median[lane_id] = statistics.median(lane.warm_wall_seconds)
        final_median[lane_id] = statistics.median(lane.warm_final_objectives)

    custom = "jax_gpu_custom"
    for lane_id in (NATIVE_LANE, custom):
        if not (math.isfinite(ttq[lane_id]) and ttq[lane_id] > 0.0):
            raise _fail(
                f"{lane_id}: TTQ {ttq[lane_id]} is not finite-positive; no "
                "speed verdict is derivable"
            )
    gates = {
        "ttq_vs_native": ttq[custom] / ttq[NATIVE_LANE],
        "ttq_vs_optax": ttq[custom] / ttq["jax_gpu_optax"],
        "budget_vs_native": warm_median[custom] / warm_median[NATIVE_LANE],
    }
    for gate_name, ratio in gates.items():
        if math.isnan(ratio):
            raise _fail(f"gate {gate_name} ratio is NaN")
    quality_ok = final_median[custom] <= j_target

    print(f"campaign: {campaign['campaign_id']} on {campaign['device_name']}")
    print(f"J_target = {j_target:.6e} (native warm-final median × {QUALITY_SLACK})")
    for lane_id in GATED_LANES:
        print(
            f"  {lane_id}: TTQ={ttq[lane_id]:.3f}s "
            f"warm_median={warm_median[lane_id]:.3f}s "
            f"final_median={final_median[lane_id]:.6e}"
        )
    if math.isinf(ttq["jax_gpu_optax"]):
        print(
            "  note: jax_gpu_optax never reached J_target; ttq_vs_optax "
            "passes by domination, not by pace"
        )
    for gate_name, ratio in gates.items():
        print(f"  gate {gate_name}: ratio={ratio:.4f} (pass ≤ {SPEED_MARGIN})")
    print(f"  gate budget_quality: custom final ≤ J_target: {quality_ok}")

    failing = {name: ratio for name, ratio in gates.items() if ratio > SPEED_MARGIN}
    if not failing and quality_ok:
        print("VERDICT: WIN")
        return 0

    tie_eligible = quality_ok and all(
        ratio <= TIE_MAX_RATIO for ratio in failing.values()
    )
    profile_evidence = _profile_evidence(artifact_root)
    if tie_eligible and profile_evidence:
        print(
            "VERDICT: TIE "
            f"(profile evidence: {[path.name for path in profile_evidence]})"
        )
        return 3
    print(f"VERDICT: LOSS (failing gates: {failing}, quality_ok={quality_ok})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path.home() / "simsopt-campaigns" / "single-stage-speed-20260804",
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--baseline-tag", default=BASELINE_TAG)
    arguments = parser.parse_args()
    try:
        return evaluate(
            arguments.artifact_root.expanduser(),
            arguments.repo_root,
            arguments.baseline_tag,
        )
    except IntegrityError as error:
        print(f"INTEGRITY ERROR: {error}")
        return 2
    except (
        json.JSONDecodeError,
        KeyError,
        ValueError,
        TypeError,
        ZeroDivisionError,
        OSError,
    ) as error:
        print(f"INTEGRITY ERROR: unreadable or inconsistent receipt: {error!r}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
