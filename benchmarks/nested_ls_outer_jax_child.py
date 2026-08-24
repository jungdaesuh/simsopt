"""Fresh-process JAX outer lane for the nested-LS eight-term B3/B37 claim.

Normal mode runs scipy L-BFGS-B over the 11 coil DOFs with the reduced
nested-LS inner solve eliminating the 661 surface DOFs, and writes the
endpoint record the parent times as a full process wall. No physics
rejudge happens inside that timed run: the charter runs the gate
untimed. ``--rejudge-endpoint`` is that untimed gate — it reconstructs at
a previously written endpoint, runs the C++ LS Newton no-op judge and the
reduced-gradient check, and writes a verdict. Not F3 7.70x.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import NestedLsOuterState

os.environ.setdefault("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_T0 = time.perf_counter()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(REPO / "src")

from benchmarks.validation_ladder_common import apply_compilation_cache_policy

apply_compilation_cache_policy(os.environ.get("JAX_COMPILATION_CACHE_DIR"))

import jax
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    NESTED_LS_OUTER_REJUDGE_SCHEMA,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "JAX outer L-BFGS-B lane over the 11 coil DOFs, or the untimed "
            "endpoint rejudge gate."
        )
    )
    parser.add_argument("out_json", help="Destination JSON path.")
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="scipy L-BFGS-B maxiter (charter rung: 3 or 37).",
    )
    parser.add_argument(
        "--maxcor",
        type=int,
        required=True,
        help="scipy L-BFGS-B maxcor, identical on both lanes.",
    )
    parser.add_argument(
        "--rejudge-endpoint",
        default=None,
        help="Endpoint JSON from a previous normal-mode run. Untimed gate.",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True, slots=True)
class _OuterEval:
    """One outer evaluation, inner-feasible or rejected. Ledger row only.

    Mirrors the native twin's per-evaluation record so the two lanes'
    receipts line up row for row, including its ``rejection_reason``
    vocabulary: ``iota_branch_guard`` (converged off the anchor's branch)
    and ``inner_solve_failed`` (never converged). ``anchor_distance`` is
    the ``‖Δc‖`` that priced the sealed sentinel.
    """

    eval_index: int
    inner_feasible: bool
    rejection_reason: str | None
    rejection_detail: str | None
    j: float
    grad_l2: float
    grad_inf: float
    anchor_distance: float
    # Measured only where a branch was reached: None on an inner solve
    # that never converged.
    iota_branch_delta: float | None
    seconds: float
    seconds_since_optimize_start: float
    coil_dofs: tuple[float, ...]
    # ``inner_iota`` is the iota this evaluation's inner solve produced;
    # None when it produced none. The rest are None on any rejection: the
    # solve raised before publishing them, and reprinting the previous
    # evaluation's telemetry here would read as if this one measured it.
    inner_iota: float | None
    inner_g: float | None
    inner_iterations: int | None
    inner_grad_l2: float | None
    adjoint_live_eta: float | None
    inner_surface_sha256: str


@dataclass(frozen=True, slots=True)
class _NestedCandidate:
    """One feasible nested point pending outer acceptance.

    The candidate carries the complete state needed to promote or restore
    one point atomically. A successful inner solve alone does not make it
    the outer incumbent; scipy's accepted-step callback does that.
    """

    eval_index: int
    coil_dofs: NDArray[np.float64]
    j: float
    gradient: NDArray[np.float64]
    grad_l2: float
    grad_inf: float
    surface_dofs: NDArray[np.float64]
    iota: float
    G: float
    inner_iterations: int
    inner_grad_l2: float
    adjoint_live_eta: float


def _restore_nested_candidate(
    state: NestedLsOuterState,
    jax_boozer: BoozerSurfaceJAX,
    candidate: _NestedCandidate,
) -> None:
    """Restore one committed candidate as the sole continuation anchor."""

    state.set_anchor(candidate.surface_dofs, candidate.iota, candidate.G)
    state.inner_iterations = int(candidate.inner_iterations)
    state.inner_grad_l2 = float(candidate.inner_grad_l2)
    state.adjoint_live_eta = float(candidate.adjoint_live_eta)
    jax_boozer.surface.set_dofs(candidate.surface_dofs)
    jax_boozer.biotsavart.x = np.array(candidate.coil_dofs, dtype=np.float64, copy=True)
    jax_boozer._refresh_coil_data()


@dataclass(frozen=True, slots=True)
class _OuterIterate:
    """One accepted L-BFGS-B iterate, keyed back to the evaluation ledger."""

    iterate_index: int
    evals_completed: int
    j: float
    grad_l2: float
    coil_dofs: tuple[float, ...]
    seconds_since_optimize_start: float


@dataclass(frozen=True, slots=True)
class _EndpointRecord:
    """One lane's endpoint, normalized out of either child's JSON layout.

    The rejudge gate is charter-symmetric — it judges the native endpoint
    exactly as it judges the JAX one — so it reads both layouts into this
    single record rather than growing a second rejudge path. The native
    child nests its endpoint under ``endpoint``; the JAX child writes
    flat ``endpoint_*`` keys.
    """

    lane: str
    schema: str
    coil_dofs: NDArray[np.float64]
    surface_dofs: NDArray[np.float64]
    declared_coil_sha256: str
    declared_surface_sha256: str
    iota: float
    G: float
    j: float


def _load_endpoint_record(endpoint_path: Path) -> _EndpointRecord:
    """Read a JAX-lane or native-lane endpoint JSON into one shape."""

    payload = json.loads(endpoint_path.read_text(encoding="utf-8"))
    schema = str(payload["schema"])
    if schema == NESTED_LS_OUTER_JAX_CHILD_SCHEMA:
        return _EndpointRecord(
            lane="jax",
            schema=schema,
            coil_dofs=np.asarray(payload["endpoint_coil_dofs"], dtype=np.float64),
            surface_dofs=np.asarray(payload["endpoint_surface_dofs"], dtype=np.float64),
            declared_coil_sha256=str(payload["endpoint_coil_sha256"]),
            declared_surface_sha256=str(payload["endpoint_surface_sha256"]),
            iota=float(payload["endpoint_iota"]),
            G=float(payload["endpoint_g"]),
            j=float(payload["endpoint_j"]),
        )
    if schema == NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA:
        endpoint = payload["endpoint"]
        return _EndpointRecord(
            lane="native",
            schema=schema,
            coil_dofs=np.asarray(endpoint["coil_dofs"], dtype=np.float64),
            surface_dofs=np.asarray(endpoint["surface_dofs"], dtype=np.float64),
            declared_coil_sha256=str(endpoint["coil_sha256"]),
            declared_surface_sha256=str(endpoint["surface_sha256"]),
            iota=float(endpoint["iota"]),
            G=float(endpoint["G"]),
            j=float(endpoint["objective"]),
        )
    raise SystemExit(
        f"rejudge cannot read endpoint schema {schema!r}; expected "
        f"{NESTED_LS_OUTER_JAX_CHILD_SCHEMA!r} or "
        f"{NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA!r}."
    )


def _endpoint_declaration_mismatch(
    record: _EndpointRecord,
    *,
    expected_coil_sha256: str,
    expected_surface_sha256: str,
) -> str | None:
    if record.declared_coil_sha256 != expected_coil_sha256:
        return "endpoint_coil_declaration_mismatch"
    if record.declared_surface_sha256 != expected_surface_sha256:
        return "endpoint_surface_declaration_mismatch"
    return None


def _cache_meta() -> dict[str, object]:
    return {
        "jax_default_backend": jax.default_backend(),
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "cache_disabled": os.environ.get("SIMSOPT_DISABLE_JAX_COMPILATION_CACHE"),
    }


def _run_outer(*, budget: int, maxcor: int) -> dict[str, object]:
    """Timed lane: outer L-BFGS-B over coils, inner nested-LS eliminated."""

    if jax.default_backend() != "gpu":
        raise SystemExit(f"expected gpu, got {jax.default_backend()!r}")
    from simsopt_jax_adapters.geo.nested_ls_contract import (
        NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
        NESTED_LS_OUTER_MAX_RESTARTS,
        NestedLsOuterCandidateStore,
        nested_ls_outer_endpoint_success,
        nested_ls_outer_ftol_zero_stop,
        nested_ls_outer_rejection_barrier,
        nested_ls_outer_restart_reason,
    )
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        DEFAULT_F3_B37_GPU_LANE,
        DEFAULT_F3_B37_NATIVE_LANE,
        NestedLsBranchJump,
        NestedLsInnerSolveFailed,
        load_archived_nested_ls_pair,
        load_flat675_lane_blocks,
        nested_ls_outer_value_and_grad,
        nested_ls_runtime_identity,
        prepare_f3_b37_outer_state,
        sha256_float64,
    )

    # One optimizer policy for both lanes: the sealed F3 native-lane
    # knobs, scaled to this rung's budget. The native twin owns the
    # loader, so neither lane can drift into its own ftol/gtol/maxls.
    from benchmarks.nested_ls_outer_native_child import load_outer_optimizer_policy

    import_init_seconds = float(time.perf_counter() - _T0)
    load_started = time.perf_counter()
    outer_policy = load_outer_optimizer_policy(
        DEFAULT_F3_B37_NATIVE_LANE, budget=int(budget), maxcor=int(maxcor)
    )
    coils, surface, lane_meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    _native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    del _native, _target
    problem_load_seconds = float(time.perf_counter() - load_started)
    # Charter Amendment 1 / start symmetry: no archived-endpoint freeze
    # here. ``load_archived_nested_ls_pair`` leaves the raw un-nest lane
    # surface on the Boozer, ``prepare_f3_b37_outer_state`` anchors on it,
    # and the first outer evaluation therefore pays the inner convergence
    # inside the timed wall exactly as the native twin's seed solve does.
    prepare_started = time.perf_counter()
    state = prepare_f3_b37_outer_state(jax_boozer)
    prepare_seconds = float(time.perf_counter() - prepare_started)
    start_coils = np.asarray(jax_boozer.biotsavart.x, dtype=np.float64)
    start_surface = np.array(state.anchor_surface_dofs, dtype=np.float64, copy=True)
    start_iota = float(state.anchor_iota)
    start_g = float(state.anchor_G)

    evals: list[_OuterEval] = []
    feasible: list[_NestedCandidate] = []
    iterates: list[_OuterIterate] = []
    candidates = NestedLsOuterCandidateStore[_NestedCandidate](start_coils)
    optimize_started = time.perf_counter()

    def _restore(candidate: _NestedCandidate) -> None:
        _restore_nested_candidate(state, jax_boozer, candidate)

    def _value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        point = np.array(x, dtype=np.float64, copy=True)
        started = time.perf_counter()
        try:
            value, gradient = nested_ls_outer_value_and_grad(state, point)
        except (NestedLsBranchJump, NestedLsInnerSolveFailed) as signal:
            # Only these two types are caught. Coil motion is deliberately
            # untyped upstream and must stay fatal, and evaluation 0 has no
            # committed anchor to price against, so both re-raise.
            seconds = float(time.perf_counter() - started)
            if not candidates.is_primed:
                raise
            anchor = candidates.committed
            _restore(anchor)
            if candidates.committed_matches(point):
                raise RuntimeError(
                    "the nested inner solve rejected the exact committed outer "
                    "point; no line-search barrier can represent that failure"
                ) from signal
            if isinstance(signal, NestedLsBranchJump):
                rejection_reason = "iota_branch_guard"
                rejected_iota: float | None = signal.iota
                branch_delta: float | None = abs(signal.iota - signal.anchor_iota)
            else:
                # Non-convergence never reached a branch, so there is no
                # iota to report and no branch motion to measure.
                rejection_reason = "inner_solve_failed"
                rejected_iota = None
                branch_delta = None
            distance = float(np.linalg.norm(point - anchor.coil_dofs))
            sentinel, sentinel_grad = nested_ls_outer_rejection_barrier(
                anchor_value=anchor.j,
                anchor_parameters=anchor.coil_dofs,
                trial_parameters=point,
                scale=float(outer_policy.rejection_distance_scale),
            )
            sentinel_grad_l2 = float(np.linalg.norm(sentinel_grad))
            sentinel_grad_inf = float(np.linalg.norm(sentinel_grad, ord=np.inf))
            rejected = _OuterEval(
                eval_index=len(evals),
                inner_feasible=False,
                rejection_reason=rejection_reason,
                rejection_detail=str(signal),
                j=sentinel,
                grad_l2=sentinel_grad_l2,
                grad_inf=sentinel_grad_inf,
                anchor_distance=distance,
                iota_branch_delta=branch_delta,
                seconds=seconds,
                seconds_since_optimize_start=float(
                    time.perf_counter() - optimize_started
                ),
                coil_dofs=tuple(float(entry) for entry in point),
                inner_iota=rejected_iota,
                inner_g=None,
                inner_iterations=None,
                inner_grad_l2=None,
                adjoint_live_eta=None,
                inner_surface_sha256=sha256_float64(anchor.surface_dofs),
            )
            evals.append(rejected)
            print(
                "outer jax eval"
                f" index={rejected.eval_index} REJECTED {rejection_reason}"
                f" sentinel_J={sentinel!r} rejected_iota={rejected_iota!r}"
                f" iota_delta={branch_delta!r}"
                f" anchor_distance={distance!r} seconds={seconds!r}"
                f" detail={str(signal)!r}",
                flush=True,
            )
            return sentinel, sentinel_grad
        seconds = float(time.perf_counter() - started)
        grad = np.asarray(gradient, dtype=np.float64).reshape(-1)
        inner_surface = np.array(state.anchor_surface_dofs, dtype=np.float64, copy=True)
        warm_start_iota = (
            candidates.committed.iota if candidates.is_primed else start_iota
        )
        record = _OuterEval(
            eval_index=len(evals),
            inner_feasible=True,
            rejection_reason=None,
            rejection_detail=None,
            j=float(value),
            grad_l2=float(np.linalg.norm(grad)),
            grad_inf=float(np.linalg.norm(grad, ord=np.inf)),
            anchor_distance=0.0,
            iota_branch_delta=abs(float(state.anchor_iota) - warm_start_iota),
            seconds=seconds,
            seconds_since_optimize_start=float(time.perf_counter() - optimize_started),
            coil_dofs=tuple(float(entry) for entry in point),
            inner_iota=float(state.anchor_iota),
            inner_g=float(state.anchor_G),
            inner_iterations=int(state.inner_iterations),
            inner_grad_l2=float(state.inner_grad_l2),
            adjoint_live_eta=float(state.adjoint_live_eta),
            inner_surface_sha256=sha256_float64(inner_surface),
        )
        evals.append(record)
        candidate = _NestedCandidate(
            eval_index=record.eval_index,
            coil_dofs=point,
            j=record.j,
            gradient=np.array(grad, dtype=np.float64, copy=True),
            grad_l2=record.grad_l2,
            grad_inf=record.grad_inf,
            surface_dofs=inner_surface,
            iota=float(state.anchor_iota),
            G=float(state.anchor_G),
            inner_iterations=int(state.inner_iterations),
            inner_grad_l2=float(state.inner_grad_l2),
            adjoint_live_eta=float(state.adjoint_live_eta),
        )
        feasible.append(candidate)
        primed = candidates.record(point, candidate)
        if not primed:
            _restore(candidates.committed)
        print(
            "outer jax eval"
            f" index={record.eval_index} J={record.j!r}"
            f" grad_l2={record.grad_l2!r} iota={record.inner_iota!r}"
            f" iota_delta={record.iota_branch_delta!r}"
            f" inner_iter={record.inner_iterations}"
            f" inner_grad_l2={record.inner_grad_l2!r}"
            f" live_eta={record.adjoint_live_eta!r} seconds={record.seconds!r}",
            flush=True,
        )
        return float(value), grad

    def _callback(xk: np.ndarray) -> None:
        committed = candidates.accept(np.asarray(xk, dtype=np.float64))
        _restore(committed)
        iterates.append(
            _OuterIterate(
                iterate_index=len(iterates),
                evals_completed=len(evals),
                j=committed.j,
                grad_l2=committed.grad_l2,
                coil_dofs=tuple(float(entry) for entry in np.asarray(xk)),
                seconds_since_optimize_start=float(
                    time.perf_counter() - optimize_started
                ),
            )
        )

    # Recovery only: a qualifying early stop restarts from the transaction's
    # complete incumbent with fresh L-BFGS memory. Trial evaluations cannot
    # alter that incumbent; the identical rule lives in the native twin.
    budget_total = int(budget)
    consumed_nit = 0
    total_nfev = 0
    total_njev = 0
    restart_nits: list[int] = []
    restart_attempts: list[dict[str, object]] = []
    attempt_x0 = start_coils
    while True:
        rejected_before = len(evals) - len(feasible)
        attempt_options = outer_policy.as_scipy_options()
        attempt_options["maxiter"] = budget_total - consumed_nit
        result = minimize(
            _value_and_grad,
            attempt_x0,
            jac=True,
            method=outer_policy.method,
            options=attempt_options,
            callback=_callback,
        )
        attempt_nit = int(result.nit)
        grad_inf = float(np.max(np.abs(np.asarray(result.jac, dtype=np.float64))))
        rejected_this_attempt = (len(evals) - len(feasible)) - rejected_before
        # The shared classifier restarts an abnormal line search or an FTOL
        # stop that the sealed ftol=0 policy explicitly disabled.
        restart_reason = nested_ls_outer_restart_reason(
            ftol=float(outer_policy.ftol),
            status=int(result.status),
            message=str(result.message),
        )
        restart_nits.append(attempt_nit)
        restart_attempts.append(
            {
                "nit": attempt_nit,
                "nfev": int(result.nfev),
                "njev": int(result.njev),
                "optimizer_success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "fun": float(result.fun),
                "grad_inf": grad_inf,
                "rejected_this_attempt": int(rejected_this_attempt),
                "restart_reason": restart_reason,
            }
        )
        consumed_nit += attempt_nit
        total_nfev += int(result.nfev)
        total_njev += int(result.njev)
        if (
            consumed_nit >= budget_total
            or attempt_nit == 0
            or restart_reason is None
            or len(restart_nits) > NESTED_LS_OUTER_MAX_RESTARTS
        ):
            break
        attempt_x0 = np.array(
            candidates.committed.coil_dofs,
            dtype=np.float64,
            copy=True,
        )
    optimize_seconds = float(time.perf_counter() - optimize_started)
    optimizer_x = np.asarray(result.x, dtype=np.float64)
    endpoint_anchor = candidates.committed
    endpoint_is_optimizer_x = candidates.committed_matches(optimizer_x)
    ftol_zero_stop = nested_ls_outer_ftol_zero_stop(
        ftol=float(outer_policy.ftol),
        message=str(result.message),
    )
    endpoint = np.array(endpoint_anchor.coil_dofs, dtype=np.float64, copy=True)
    endpoint_surface = endpoint_anchor.surface_dofs
    process_elapsed_seconds = float(time.perf_counter() - _T0)
    return {
        "schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
        "mode": "outer",
        "budget": int(budget),
        "maxcor": int(maxcor),
        # A fixed-budget run stops on maxiter, which scipy reports as
        # success=False, so the child's own flag is the usable-endpoint
        # predicate the driver gates on — the reported endpoint is the
        # optimizer's own final iterate and the committed transaction point.
        # An FTOL stop is never success under the sealed ftol=0 policy;
        # scipy's raw verdict remains separately visible.
        "success": nested_ls_outer_endpoint_success(
            endpoint_matches=endpoint_is_optimizer_x,
            ftol=float(outer_policy.ftol),
            status=int(result.status),
            message=str(result.message),
        ),
        "optimizer_success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "ftol_zero_stop": ftol_zero_stop,
        "nit": int(consumed_nit),
        "nfev": int(total_nfev),
        "njev": int(total_njev),
        "restart_count": int(len(restart_nits) - 1),
        "restart_nits": [int(value) for value in restart_nits],
        "restart_attempts": restart_attempts,
        "result_fun": float(result.fun),
        "endpoint_is_optimizer_x": endpoint_is_optimizer_x,
        "optimizer_x": [float(value) for value in optimizer_x],
        "outer_policy": outer_policy.as_payload(),
        "iota_branch_guard": float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
        "feasible_evaluations": int(len(feasible)),
        "rejected_evaluations": int(len(evals) - len(feasible)),
        # Charter Amendment 1 / start symmetry: this lane opens on the raw
        # un-nest archived lane surface, so its first outer evaluation pays
        # the inner convergence inside the timed wall, exactly as the
        # native twin's start-point solve does.
        "start_policy": "raw_lane_surface_unnest",
        "start_coil_dofs": [float(entry) for entry in start_coils],
        "start_coil_sha256": sha256_float64(start_coils),
        "start_surface_sha256": sha256_float64(start_surface),
        "start_iota": start_iota,
        "start_g": start_g,
        "endpoint_eval_index": int(endpoint_anchor.eval_index),
        "endpoint_coil_dofs": [float(entry) for entry in endpoint],
        "endpoint_coil_sha256": sha256_float64(endpoint),
        "endpoint_surface_dofs": [float(entry) for entry in endpoint_surface],
        "endpoint_surface_sha256": sha256_float64(endpoint_surface),
        "endpoint_j": endpoint_anchor.j,
        "endpoint_grad_l2": endpoint_anchor.grad_l2,
        "endpoint_grad_inf": endpoint_anchor.grad_inf,
        "endpoint_iota": endpoint_anchor.iota,
        "endpoint_g": endpoint_anchor.G,
        "endpoint_inner_iterations": endpoint_anchor.inner_iterations,
        "endpoint_inner_grad_l2": endpoint_anchor.inner_grad_l2,
        "endpoint_adjoint_live_eta": endpoint_anchor.adjoint_live_eta,
        "outer_evals": [asdict(record) for record in evals],
        "outer_iterates": [asdict(record) for record in iterates],
        "wall_splits": {
            "import_init_seconds": import_init_seconds,
            "problem_load_seconds": problem_load_seconds,
            "prepare_state_seconds": prepare_seconds,
            "optimize_seconds": optimize_seconds,
            "process_elapsed_seconds": process_elapsed_seconds,
        },
        "lane": lane_meta,
        "runtime": nested_ls_runtime_identity(),
        **_cache_meta(),
    }


def _rejudge(*, endpoint_path: Path, budget: int, maxcor: int) -> dict[str, object]:
    """Untimed physics gate at a written endpoint. Never inside a claim clock."""

    from simsopt_jax_adapters.geo.nested_ls_contract import (
        NESTED_LS_NEWTON_TOL,
        nested_ls_physics_newton_kwargs,
    )
    from simsopt_jax_adapters.geo.nested_ls_reduced import (
        nested_ls_reduced_closures,
        reduced_penalty_gradient_envelope,
        require_full_y_rank,
        solve_projected_y,
    )
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        _native_ls_gradient_norms,
        load_archived_nested_ls_pair,
        nested_ls_runtime_identity,
        sha256_file,
        sha256_float64,
    )

    record = _load_endpoint_record(endpoint_path)
    source_child_payload_sha = sha256_file(endpoint_path)
    # The declared sha is the producer's own witness; the written array is
    # what this gate reconstructs from. Both must agree, and the reload
    # must reproduce the array, or the judge is not judging the endpoint.
    expected_coil_sha = sha256_float64(record.coil_dofs)
    expected_surface_sha = sha256_float64(record.surface_dofs)
    declaration_mismatch = _endpoint_declaration_mismatch(
        record,
        expected_coil_sha256=expected_coil_sha,
        expected_surface_sha256=expected_surface_sha,
    )
    if declaration_mismatch is not None:
        raise SystemExit(
            f"rejudge endpoint declaration failed closed: {declaration_mismatch}"
        )
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=record.coil_dofs,
        surface_coordinates=record.surface_dofs,
    )
    del _target
    coils_before = np.asarray(native.biotsavart.x, dtype=np.float64)
    reloaded_coil_sha = sha256_float64(coils_before)
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    loaded = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    reloaded_surface_sha = sha256_float64(loaded)
    endpoint_iota = record.iota
    endpoint_g = record.G
    solution = solve_projected_y(
        residual_fn, loaded, np.array([endpoint_iota, endpoint_g], dtype=np.float64)
    )
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    reduced_gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, loaded, y_star),
        dtype=np.float64,
    )
    reduced_grad_l2 = float(np.linalg.norm(reduced_gradient))
    native.need_to_run_code = True
    started = time.perf_counter()
    judged = native.minimize_boozer_penalty_constraints_newton(
        iota=endpoint_iota,
        G=endpoint_g,
        **nested_ls_physics_newton_kwargs(),
    )
    rejudge_seconds = float(time.perf_counter() - started)
    coils_after = np.asarray(native.biotsavart.x, dtype=np.float64)
    surface_after = np.asarray(native.surface.get_dofs(), dtype=np.float64)
    grad_l2, grad_inf = _native_ls_gradient_norms(judged)
    coil_delta_inf = float(np.linalg.norm(coils_after - coils_before, ord=np.inf))
    surface_delta_inf = float(np.linalg.norm(surface_after - loaded, ord=np.inf))
    rejudge_iter = int(judged["iter"])
    rejudge_success = bool(judged["success"])
    grad_tol = float(NESTED_LS_NEWTON_TOL)
    rejudge_noop = bool(
        rejudge_success
        and rejudge_iter == 0
        and coil_delta_inf == 0.0
        and surface_delta_inf == 0.0
        and np.isfinite(grad_l2)
        and grad_l2 <= grad_tol
    )
    reduced_grad_ok = bool(np.isfinite(reduced_grad_l2) and reduced_grad_l2 <= grad_tol)
    if reloaded_coil_sha != expected_coil_sha:
        reason: str | None = "endpoint_coil_reload_mismatch"
    elif reloaded_surface_sha != expected_surface_sha:
        reason = "endpoint_surface_reload_mismatch"
    elif not rejudge_success:
        reason = "rejudge_failed"
    elif rejudge_iter != 0:
        reason = "rejudge_not_noop"
    elif coil_delta_inf != 0.0:
        reason = "rejudge_coil_moved"
    elif surface_delta_inf != 0.0:
        reason = "rejudge_surface_moved"
    elif not (np.isfinite(grad_l2) and grad_l2 <= grad_tol):
        reason = "rejudge_grad_tol"
    elif not reduced_grad_ok:
        reason = "reduced_grad_tol"
    else:
        reason = None
    print(
        "outer rejudge"
        f" lane={record.lane} noop={rejudge_noop} iter={rejudge_iter}"
        f" native_grad_l2={grad_l2!r} reduced_grad_l2={reduced_grad_l2!r}"
        f" reason={reason!r}",
        flush=True,
    )
    return {
        "schema": NESTED_LS_OUTER_REJUDGE_SCHEMA,
        "mode": "rejudge",
        "judged_lane": record.lane,
        "judged_schema": record.schema,
        "budget": int(budget),
        "maxcor": int(maxcor),
        "source_child_payload_sha256": source_child_payload_sha,
        "source_child_schema": record.schema,
        "declared_coil_sha256": record.declared_coil_sha256,
        "declared_surface_sha256": record.declared_surface_sha256,
        "endpoint_coil_sha256": expected_coil_sha,
        "endpoint_surface_sha256": expected_surface_sha,
        "reloaded_coil_sha256": reloaded_coil_sha,
        "reloaded_surface_sha256": reloaded_surface_sha,
        "endpoint_j": record.j,
        "grad_tol": grad_tol,
        "endpoint_iota": endpoint_iota,
        "endpoint_g": endpoint_g,
        "y_star_iota": float(y_star[0]),
        "y_star_g": float(y_star[1]),
        "y_star_vs_endpoint_iota": float(y_star[0]) - endpoint_iota,
        "y_star_vs_endpoint_g": float(y_star[1]) - endpoint_g,
        "y_rank": int(np.asarray(jax.device_get(solution.numerical_rank))),
        "reduced_grad_l2": reduced_grad_l2,
        "reduced_grad_ok": reduced_grad_ok,
        "native_rejudge_success": rejudge_success,
        "native_rejudge_iter": rejudge_iter,
        "native_rejudge_iota": float(judged["iota"]),
        "native_rejudge_g": float(judged["G"]),
        "native_rejudge_grad_l2": grad_l2,
        "native_rejudge_grad_inf": grad_inf,
        "native_rejudge_coil_delta_inf": coil_delta_inf,
        "native_rejudge_surface_delta_inf": surface_delta_inf,
        "native_rejudge_seconds": rejudge_seconds,
        "rejudge_noop": rejudge_noop,
        "fail_closed_reason": reason,
        "process_elapsed_seconds": float(time.perf_counter() - _T0),
        "runtime": nested_ls_runtime_identity(),
        **_cache_meta(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out_json)
    if args.rejudge_endpoint is None:
        payload = _run_outer(budget=int(args.budget), maxcor=int(args.maxcor))
    else:
        payload = _rejudge(
            endpoint_path=Path(args.rejudge_endpoint),
            budget=int(args.budget),
            maxcor=int(args.maxcor),
        )
    out_path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
