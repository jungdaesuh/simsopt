"""Fresh-process native-twin outer L-BFGS-B over the 11 flat-675 coil DOFs.

One process is one complete NATIVE-lane outer optimization under
``docs/jax_nested_ls_outer_charter.md``: a nested banana ``run_code`` inner
solve at every outer evaluation, the frozen bundle's eight-term outer ``J``
at ``(c, v_frozen, s*(c))``, and ``dJ/dc`` through the C++ Boozer-LS
adjoint. The parent pins ``OMP_NUM_THREADS`` and owns the claim clock; the
walls reported here are informational splits. Not a speed claim, not F3
7.70x.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# The adapter modules below pull JAX in transitively even though nothing on
# this lane traces. That import is the only JAX cost left in the native
# child, so it is timed and published rather than left unaccounted: an
# unmeasured cost inside the native denominator biases every later ratio.
_IMPORT_STARTED = time.perf_counter()

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(REPO / "src")

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    SurfaceRZFourier,
)
from simsopt.objectives.utilities import forward_backward
from simsopt_jax.core.specs import SurfaceRZFourierSpec
from simsopt_jax_adapters.geo.flat675.manifest import (
    load_flat675_input_manifest,
    load_flat675_vessel_template,
)
from simsopt_jax_adapters.geo.flat675.policy import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    Flat675ObjectivePolicy,
    flat675_outer_objective_config,
)
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
    NESTED_LS_OUTER_MAX_RESTARTS,
    NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    NestedLsOuterAcceptWithoutCandidate,
    NestedLsOuterCandidateStore,
    nested_ls_outer_attempt_fun_is_objective,
    nested_ls_outer_endpoint_success,
    nested_ls_outer_ftol_zero_stop,
    nested_ls_outer_rejection_barrier,
    nested_ls_outer_restart_reason,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE,
    DEFAULT_F3_B37_NATIVE_LANE,
    DEFAULT_FLAT675_BUNDLE_ROOT,
    _run_native_banana_bfgs_then_newton,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    nested_ls_omp_threads_pinned,
    nested_ls_runtime_identity,
    nested_ls_threading_env,
    sha256_float64,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    _TRACEABLE_SINGLE_STAGE_OUTER_TERM_WEIGHT_KEYS,
    SurfaceSurfaceDistance,
)

from benchmarks.nested_ls_shamanskii_attribution import write_strict_json

MODULE_IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED

# The sealed F3 lane names are source provenance. The transactional containment
# below deliberately supersedes the four state/rejection behaviours and records
# that override separately in every new child payload.
IMPLEMENTED_LANE_POLICY_NAMES = {
    "accepted_state_policy": "rolling_last_accepted_anchor",
    "method": "L-BFGS-B",
    "rejection_gradient_policy": "accepted_anchor_gradient",
    "rejection_rollback_policy": "retain_accepted_anchor",
    "rejection_value_policy": "anchor_plus_offset_plus_distance_v1",
}
TRANSACTION_POLICY_NAME_ITEMS: Final[tuple[tuple[str, str], ...]] = (
    ("accepted_state_policy", "commit_on_scipy_accept"),
    ("rejection_gradient_policy", "quadratic_barrier_derivative_v2"),
    ("rejection_rollback_policy", "restore_committed_anchor"),
    (
        "rejection_value_policy",
        "anchor_plus_half_scaled_distance_squared_v2",
    ),
)
PUBLICATION = (
    "Native-twin outer L-BFGS-B over the 11 flat-675 coil DOFs with a nested "
    "banana run_code inner solve per evaluation. Charter "
    "docs/jax_nested_ls_outer_charter.md. Not F3 7.70x."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One complete native-lane flat-675 outer optimization."
    )
    parser.add_argument(
        "out_json", type=Path, help="Receipt path written by the child."
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="scipy L-BFGS-B maxiter (charter B3 nit=3, B37 nit=37).",
    )
    parser.add_argument(
        "--maxcor",
        type=int,
        required=True,
        help="scipy L-BFGS-B maxcor, identical on both lanes.",
    )
    return parser.parse_args(argv)


@dataclass(frozen=True, slots=True)
class OuterOptimizerPolicy:
    """The F3 native lane's sealed outer policy plus this run's budget/maxcor.

    ``ftol``, ``gtol``, and ``maxls`` are the sealed lane's; ``maxiter`` and
    ``maxcor`` are the charter rung's. The source lane's rejection scale is
    retained while the transactional containment owns its coherent barrier.
    """

    source: str
    method: str
    ftol: float
    gtol: float
    maxls: int
    maxiter: int
    maxcor: int
    rejection_distance_scale: float
    lane_policy_names: dict[str, str]

    def as_scipy_options(self) -> dict[str, object]:
        return {
            "ftol": self.ftol,
            "gtol": self.gtol,
            "maxls": self.maxls,
            "maxiter": self.maxiter,
            "maxcor": self.maxcor,
        }

    def as_payload(self) -> dict[str, object]:
        return {
            "source": self.source,
            "method": self.method,
            "ftol": self.ftol,
            "gtol": self.gtol,
            "maxls": self.maxls,
            "maxiter": self.maxiter,
            "maxcor": self.maxcor,
            "rejection_distance_scale": self.rejection_distance_scale,
            "iota_branch_guard": float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
            "lane_policy_names": self.lane_policy_names,
            "transaction_policy_names": dict(TRANSACTION_POLICY_NAME_ITEMS),
        }


def load_outer_optimizer_policy(
    lane_path: Path,
    *,
    budget: int,
    maxcor: int,
) -> OuterOptimizerPolicy:
    """Read the sealed outer-optimizer knobs from one F3 lane JSON."""

    policy = json.loads(Path(lane_path).read_text())["policy"]
    names = {key: str(policy[key]) for key in IMPLEMENTED_LANE_POLICY_NAMES}
    if names != IMPLEMENTED_LANE_POLICY_NAMES:
        raise ValueError(
            f"lane {lane_path} declares outer policies this child does not "
            f"implement: {names!r} != {IMPLEMENTED_LANE_POLICY_NAMES!r}."
        )
    return OuterOptimizerPolicy(
        source=str(Path(lane_path)),
        method=str(policy["method"]),
        ftol=float(policy["ftol"]),
        gtol=float(policy["gtol"]),
        maxls=int(policy["maxls"]),
        maxiter=int(budget),
        maxcor=int(maxcor),
        rejection_distance_scale=float(policy["rejection_distance_scale"]),
        lane_policy_names=names,
    )


def load_lane_vessel_coordinates(lane_path: Path) -> NDArray[np.float64]:
    """Return the frozen vessel block of a fused or native F3 lane candidate."""

    result = json.loads(Path(lane_path).read_text())["result"]
    endpoint = result.get("endpoint_candidate")
    candidate = (
        endpoint
        if isinstance(endpoint, dict) and "vessel_coordinates" in endpoint
        else result["final_certificate"]["candidate"]
    )
    return np.asarray(candidate["vessel_coordinates"], dtype=np.float64)


def build_vessel_surface(
    template: SurfaceRZFourierSpec,
    coordinates: NDArray[np.float64],
) -> SurfaceRZFourier:
    """Materialize the frozen vessel the surface-to-vessel term measures against."""

    vessel = SurfaceRZFourier(
        nfp=int(template.nfp),
        stellsym=bool(template.stellsym),
        mpol=int(template.mpol),
        ntor=int(template.ntor),
        quadpoints_phi=np.asarray(template.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(template.quadpoints_theta, dtype=np.float64),
    )
    vessel.x = np.asarray(coordinates, dtype=np.float64)
    vessel.fix_all()
    return vessel


@dataclass(frozen=True, slots=True)
class NativeOuterObjective:
    """Native eight-term flat-675 ``J`` and its 11-DOF coil gradient.

    ``evaluate`` reads the solved inner state off ``boozer.res``, so it is
    only meaningful directly after a converged nested solve: the surface is
    eliminated by that solve's LU factors and coil VJP. Weights arrive in
    :data:`FLAT675_OBJECTIVE_TERM_KEYS` order and are summed in that order.
    """

    boozer: BoozerSurface
    non_qs: NonQuasiSymmetricRatio
    residual: BoozerResidual
    curve_curve: CurveCurveDistance
    curve_surface: CurveSurfaceDistance
    surface_vessel: SurfaceSurfaceDistance
    banana_length: CurveLength
    banana_curvature: LpCurveCurvature
    weights: tuple[float, ...]
    iota_target: float
    length_target_m: float

    def evaluate(self) -> tuple[float, dict[str, float], NDArray[np.float64]]:
        """Return ``(J, raw term values, dJ/dc)`` at the solved inner state."""

        boozer = self.boozer
        inner_state = boozer.res
        iota = float(inner_state["iota"])
        g_value = float(inner_state["G"])
        surface = boozer.surface
        field = boozer.biotsavart
        p_matrix, l_matrix, u_matrix = inner_state["PLU"]
        iota_delta = iota - self.iota_target

        non_qs_value, non_qs_derivative = (
            self.non_qs.fixed_surface_value_and_derivative()
        )
        residual_value, residual_derivative, residual_y_partial = (
            self.residual.fixed_surface_value_derivative_and_y_partial(
                iota,
                g_value,
                weight_inv_modB=bool(inner_state["weight_inv_modB"]),
            )
        )
        length_excess = max(float(self.banana_length.J()) - self.length_target_m, 0.0)
        raw = {
            "non_qs": float(non_qs_value),
            "residual": float(residual_value),
            "iota": 0.5 * iota_delta * iota_delta,
            "length": 0.5 * length_excess * length_excess,
            "curve_curve": float(self.curve_curve.J()),
            "curve_surface": float(self.curve_surface.J()),
            "surface_vessel": float(self.surface_vessel.J()),
            "curvature": float(self.banana_curvature.J()),
        }
        weight = dict(zip(FLAT675_OBJECTIVE_TERM_KEYS, self.weights, strict=True))
        total = (
            weight[FLAT675_OBJECTIVE_TERM_KEYS[0]] * raw[FLAT675_OBJECTIVE_TERM_KEYS[0]]
        )
        for term_key in FLAT675_OBJECTIVE_TERM_KEYS[1:]:
            total = total + weight[term_key] * raw[term_key]

        curve_surface_derivative = self.curve_surface.dJ(partials=True)
        surface_vessel_derivative = self.surface_vessel.dJ(partials=True)
        direct = (
            weight["non_qs"] * non_qs_derivative
            + weight["residual"] * residual_derivative
            + weight["length"] * length_excess * self.banana_length.dJ(partials=True)
            + weight["curve_curve"] * self.curve_curve.dJ(partials=True)
            + weight["curve_surface"] * curve_surface_derivative
            + weight["curvature"] * self.banana_curvature.dJ(partials=True)
        )
        surface_cotangent = (
            weight["non_qs"] * non_qs_derivative(surface)
            + weight["residual"] * residual_derivative(surface)
            + weight["curve_surface"] * curve_surface_derivative(surface)
            + weight["surface_vessel"] * surface_vessel_derivative(surface)
        )
        inner_cotangent = np.zeros(l_matrix.shape[0], dtype=np.float64)
        inner_cotangent[: surface_cotangent.size] = surface_cotangent
        inner_cotangent[-2] = (
            weight["residual"] * float(residual_y_partial[0])
            + weight["iota"] * iota_delta
        )
        inner_cotangent[-1] = weight["residual"] * float(residual_y_partial[1])

        adjoint = forward_backward(p_matrix, l_matrix, u_matrix, inner_cotangent)
        reduction = inner_state["vjp"](adjoint, boozer, iota, g_value)
        gradient = np.asarray(direct(field), dtype=np.float64) - np.asarray(
            reduction(field), dtype=np.float64
        )
        return float(total), raw, gradient


def build_native_outer_objective(
    boozer: BoozerSurface,
    *,
    objective_policy: Flat675ObjectivePolicy,
    vessel: SurfaceRZFourier,
) -> NativeOuterObjective:
    """Wire the eight frozen-bundle terms onto one native ``BoozerSurface``.

    Every weight, target, and threshold is read from ``objective_policy``;
    the non-QS auxiliary grid is checked against the policy's own quadrature
    because a silently different grid would change ``J`` without failing.
    """

    surface = boozer.surface
    if surface.local_dof_size != surface.local_full_dof_size:
        raise ValueError(
            "the native outer twin requires every Boozer surface DOF free; "
            f"got {surface.local_dof_size} of {surface.local_full_dof_size}."
        )
    curves = [coil.curve for coil in boozer.biotsavart.coils]
    banana_curve = curves[objective_policy.optimized_coil_index]
    grid_size = objective_policy.non_qs_grid_size
    if grid_size % 2 != 0:
        raise ValueError(
            f"non_qs_grid_size {grid_size} is not the 2*sDIM grid "
            "NonQuasiSymmetricRatio builds."
        )
    non_qs = NonQuasiSymmetricRatio(
        boozer,
        BiotSavart(boozer.biotsavart.coils),
        sDIM=grid_size // 2,
        quasi_poloidal=False,
    )
    config = flat675_outer_objective_config(
        objective_policy,
        nfp=int(surface.nfp),
        vessel_gamma=vessel.gamma(),
    )
    for axis_name, observed in (
        ("non_qs_quadpoints_phi", non_qs.surface.quadpoints_phi),
        ("non_qs_quadpoints_theta", non_qs.surface.quadpoints_theta),
    ):
        if not np.array_equal(
            np.asarray(observed, dtype=np.float64), config[axis_name]
        ):
            raise ValueError(
                f"NonQuasiSymmetricRatio {axis_name} differs from the frozen policy."
            )
    weights = tuple(
        float(config[_TRACEABLE_SINGLE_STAGE_OUTER_TERM_WEIGHT_KEYS[term_key]])
        for term_key in FLAT675_OBJECTIVE_TERM_KEYS
    )
    return NativeOuterObjective(
        boozer=boozer,
        non_qs=non_qs,
        residual=BoozerResidual(boozer, BiotSavart(boozer.biotsavart.coils)),
        curve_curve=CurveCurveDistance(
            curves, objective_policy.curve_curve_threshold_m
        ),
        curve_surface=CurveSurfaceDistance(
            curves, surface, objective_policy.curve_surface_threshold_m
        ),
        surface_vessel=SurfaceSurfaceDistance(
            surface, vessel, objective_policy.surface_vessel_threshold_m
        ),
        banana_length=CurveLength(banana_curve),
        banana_curvature=LpCurveCurvature(
            banana_curve,
            objective_policy.curvature_p_norm,
            threshold=objective_policy.curvature_threshold_inverse_m,
        ),
        weights=weights,
        iota_target=objective_policy.iota_target,
        length_target_m=objective_policy.length_target_m,
    )


@dataclass(frozen=True, slots=True)
class InnerWarmStart:
    """Surface and ``(iota, G)`` one nested solve is started from."""

    surface_dofs: NDArray[np.float64]
    iota: float
    G: float


@dataclass(frozen=True, slots=True)
class OuterAnchor:
    """Last accepted outer iterate: the state every solve warm-starts from."""

    coil_dofs: NDArray[np.float64]
    warm_start: InnerWarmStart
    objective: float
    gradient: NDArray[np.float64]
    terms: dict[str, float]


class NativeOuterRun:
    """scipy-facing objective with transactional nested candidates.

    An evaluation is accepted only when its nested solve converges *and*
    stays on the warm start's Boozer branch: charter Amendment 1 rules that
    ``s*(c)`` is local, so an ``iota`` moving more than
    ``NESTED_LS_OUTER_IOTA_BRANCH_GUARD`` is a failed evaluation however
    well it converged. Later feasible evaluations remain pending until scipy's
    callback accepts
    their exact coil bytes. Rejections restore the incumbent and return the
    coherent quadratic containment barrier. Evaluation 0 commits immediately
    because scipy never calls its callback for ``x0``.
    """

    def __init__(
        self,
        objective: NativeOuterObjective,
        *,
        seed: InnerWarmStart,
        rejection_distance_scale: float,
        start_coil_dofs: NDArray[np.float64],
    ) -> None:
        self.objective = objective
        self.seed = seed
        self.rejection_distance_scale = rejection_distance_scale
        self.candidates = NestedLsOuterCandidateStore[OuterAnchor](start_coil_dofs)
        self.records: list[dict[str, object]] = []
        self.feasible: list[OuterAnchor] = []

    @property
    def anchor(self) -> OuterAnchor | None:
        """The committed outer incumbent, absent only before solving ``x0``."""

        return self.candidates.committed if self.candidates.is_primed else None

    def endpoint_at(self, coil_dofs: NDArray[np.float64]) -> OuterAnchor | None:
        """Return the accepted iterate whose coil DOFs are exactly ``coil_dofs``."""

        return self.anchor if self.candidates.committed_matches(coil_dofs) else None

    def _restore_anchor(self, anchor: OuterAnchor) -> None:
        boozer = self.objective.boozer
        boozer.biotsavart.x = np.array(anchor.coil_dofs, dtype=np.float64, copy=True)
        boozer.surface.set_dofs(np.array(anchor.warm_start.surface_dofs, copy=True))
        boozer.need_to_run_code = True

    def accept(self, coil_dofs: NDArray[np.float64]) -> None:
        """Promote exactly the candidate named by scipy's accepted callback."""

        parameters = np.asarray(coil_dofs, dtype=np.float64)
        # ``accept`` raises the contract's own typed
        # ``NestedLsOuterAcceptWithoutCandidate`` when the announced bytes
        # match neither the incumbent nor a staged candidate; the restart
        # loop catches it so the run publishes its evidence and exits
        # nonzero instead of dying inside scipy. No retry, no fallback.
        committed = self.candidates.accept(parameters)
        self._restore_anchor(committed)

    def _reinstate_warm_start(self) -> InnerWarmStart:
        warm_start = self.seed if self.anchor is None else self.anchor.warm_start
        boozer = self.objective.boozer
        boozer.surface.set_dofs(np.array(warm_start.surface_dofs, copy=True))
        boozer.need_to_run_code = True
        return warm_start

    def __call__(self, x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        boozer = self.objective.boozer
        coil_dofs = np.asarray(x, dtype=np.float64)
        boozer.biotsavart.x = coil_dofs
        warm_start = self._reinstate_warm_start()
        solve = _run_native_banana_bfgs_then_newton(
            boozer,
            iota=warm_start.iota,
            G=warm_start.G,
        )
        # Branch motion is measured against the state the solve was warm
        # started from, which is the accepted anchor once one exists and the
        # archived QR seed at evaluation 0.
        iota_branch_delta = abs(float(solve["iota"]) - warm_start.iota)
        if not bool(solve["success"]):
            reason: str | None = "inner_solve_failed"
        elif iota_branch_delta > NESTED_LS_OUTER_IOTA_BRANCH_GUARD:
            reason = "iota_branch_guard"
        else:
            reason = None
        record: dict[str, object] = {
            "evaluation": len(self.records),
            "coil_dofs": [float(value) for value in coil_dofs],
            "inner_success": bool(solve["success"]),
            "inner_bfgs_iter": int(solve["bfgs_iter"]),
            "inner_newton_iter": int(solve["newton_iter"]),
            "inner_bfgs_seconds": float(solve["bfgs_seconds"]),
            "inner_newton_seconds": float(solve["newton_seconds"]),
            "inner_seconds": float(solve["seconds"]),
            "inner_coil_delta_inf": float(solve["coil_delta_inf"]),
            "iota": float(solve["iota"]),
            "G": float(solve["G"]),
            "iota_branch_delta": iota_branch_delta,
            # The state this evaluation's nested solve warm-started from, and
            # the state a refusal restores to — the same anchor
            # ``iota_branch_delta`` and ``anchor_distance`` are measured
            # against. Published under a name that says whose surface it is,
            # on every row, exactly as the JAX twin does.
            "anchor_surface_sha256": sha256_float64(warm_start.surface_dofs),
            "rejection_reason": reason,
        }
        if reason is not None:
            anchor = self.anchor
            if anchor is None:
                raise ValueError(
                    "the native outer twin's start-point evaluation was rejected "
                    f"({reason}); there is no accepted anchor to fall back to."
                )
            self._restore_anchor(anchor)
            if self.candidates.committed_matches(coil_dofs):
                raise RuntimeError(
                    "the native nested inner solve rejected the exact committed "
                    "outer point; no line-search barrier can represent that failure"
                )
            distance = float(np.linalg.norm(coil_dofs - anchor.coil_dofs))
            value, gradient = nested_ls_outer_rejection_barrier(
                anchor_value=anchor.objective,
                anchor_parameters=anchor.coil_dofs,
                trial_parameters=coil_dofs,
                scale=self.rejection_distance_scale,
            )
            record.update(
                {
                    "inner_feasible": False,
                    # Barrier row: ``objective`` and ``gradient_l2`` below are
                    # the containment barrier and its derivative, not the
                    # eight-term outer objective. See the ``value_is_valid``
                    # field comment on ``_OuterEval`` in
                    # benchmarks/nested_ls_outer_jax_child.py for the full
                    # semantics the two lanes share. Nothing else on this row
                    # is the anchor's either: ``iota``/``G``/the iteration
                    # counts are this evaluation's own refused solve, and the
                    # refused solve produced no surface of its own, so the
                    # row names none — the anchor's stays under
                    # ``anchor_surface_sha256`` above.
                    "value_is_valid": False,
                    "inner_surface_sha256": None,
                    "objective": value,
                    "objective_seconds": 0.0,
                    "gradient_l2": float(np.linalg.norm(gradient)),
                    "anchor_distance": distance,
                    "terms": None,
                }
            )
            self.records.append(record)
            print(
                f"outer eval={record['evaluation']} REJECTED {reason}"
                f" sentinel_J={value!r} solved_iota={solve['iota']!r}"
                f" iota_delta={iota_branch_delta!r} inner_s={solve['seconds']!r}",
                flush=True,
            )
            return value, gradient

        objective_started = time.perf_counter()
        value, terms, gradient = self.objective.evaluate()
        objective_seconds = float(time.perf_counter() - objective_started)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ValueError(
                "the native outer objective produced a nonfinite value or gradient "
                f"at evaluation {len(self.records)}."
            )
        candidate = OuterAnchor(
            coil_dofs=np.array(coil_dofs, copy=True),
            warm_start=InnerWarmStart(
                surface_dofs=np.array(
                    boozer.surface.get_dofs(), dtype=np.float64, copy=True
                ),
                iota=float(solve["iota"]),
                G=float(solve["G"]),
            ),
            objective=value,
            gradient=np.array(gradient, copy=True),
            terms=dict(terms),
        )
        self.feasible.append(candidate)
        primed = self.candidates.record(coil_dofs, candidate)
        if not primed:
            self._restore_anchor(self.candidates.committed)
        record.update(
            {
                "inner_feasible": True,
                # Objective row: ``objective`` and ``gradient_l2`` below are
                # the eight-term outer objective and its coil gradient, so
                # this row is the one a physics aggregate may read.
                "value_is_valid": True,
                # The surface THIS solve produced, read from the candidate's
                # captured copy rather than off the Boozer, which the restore
                # above may already have wound back to the incumbent.
                "inner_surface_sha256": sha256_float64(
                    candidate.warm_start.surface_dofs
                ),
                "objective": value,
                "objective_seconds": objective_seconds,
                "gradient_l2": float(np.linalg.norm(gradient)),
                "anchor_distance": 0.0,
                "terms": {
                    key: float(terms[key]) for key in FLAT675_OBJECTIVE_TERM_KEYS
                },
            }
        )
        self.records.append(record)
        print(
            f"outer eval={record['evaluation']} J={value!r}"
            f" iota={solve['iota']!r} inner_s={solve['seconds']!r}"
            f" obj_s={objective_seconds!r} |g|={record['gradient_l2']!r}",
            flush=True,
        )
        return value, gradient


def _term_ledger(
    terms: dict[str, float],
    weights: tuple[float, ...],
) -> list[dict[str, object]]:
    return [
        {
            "term": term_key,
            "weight": float(weight),
            "raw": float(terms[term_key]),
            "weighted": float(weight) * float(terms[term_key]),
        }
        for term_key, weight in zip(FLAT675_OBJECTIVE_TERM_KEYS, weights, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class NativeOuterRunContext:
    """Everything one native outer run needs from the heavy world.

    :func:`_prepare_native_run` is the composition root: the flat-675
    bundle, the native ``BoozerSurface`` pair, the frozen vessel and the
    eight-term objective all resolve there. :func:`_drive_native_run`
    owns the optimizer loop, the transaction and the payload, and reaches
    physics only through ``run``. The split is what lets the rejection
    ledger and the honest-failure endpoint be exercised without the
    bundle on disk.
    """

    run: NativeOuterRun
    outer_policy: OuterOptimizerPolicy
    objective_weights: tuple[float, ...]
    vessel_dofs: NDArray[np.float64]
    lane_meta: dict[str, object]
    seed_iota: float
    seed_G: float
    start_coil_dofs: NDArray[np.float64]
    threading: dict[str, str | None]
    module_import_seconds: float
    build_seconds: float
    child_started: float


def _prepare_native_run(*, budget: int, maxcor: int) -> NativeOuterRunContext:
    """Load the bundle, build the eight-term objective, seed the transaction."""

    child_started = time.perf_counter()
    threading = nested_ls_threading_env()
    outer_policy = load_outer_optimizer_policy(
        DEFAULT_F3_B37_NATIVE_LANE,
        budget=int(budget),
        maxcor=int(maxcor),
    )

    build_started = time.perf_counter()
    coils, surface_block, lane_meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    # The evaluation-0 warm start is the lane's own archived inner state, not
    # a live QR: recomputing it would run JAX inside the native denominator
    # and inflate the bar that the JAX lane is later measured against.
    archived_inner_state = lane_meta["inner_state"]
    if not isinstance(archived_inner_state, list) or len(archived_inner_state) != 2:
        raise ValueError(
            f"lane {DEFAULT_F3_B37_GPU_LANE} publishes no two-component inner "
            f"state to seed the native twin; got {archived_inner_state!r}."
        )
    seed_iota, seed_g = (float(value) for value in archived_inner_state)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface_block,
    )
    del _target, jax_boozer
    native.surface.set_dofs(np.asarray(native.surface.get_dofs(), dtype=np.float64))

    manifest = load_flat675_input_manifest(DEFAULT_FLAT675_BUNDLE_ROOT)
    vessel = build_vessel_surface(
        load_flat675_vessel_template(DEFAULT_FLAT675_BUNDLE_ROOT),
        load_lane_vessel_coordinates(DEFAULT_F3_B37_GPU_LANE),
    )
    objective = build_native_outer_objective(
        native,
        objective_policy=manifest.objective_policy,
        vessel=vessel,
    )
    build_seconds = float(time.perf_counter() - build_started)

    start_coil_dofs = np.asarray(native.biotsavart.x, dtype=np.float64)
    run = NativeOuterRun(
        objective,
        seed=InnerWarmStart(
            surface_dofs=np.asarray(native.surface.get_dofs(), dtype=np.float64),
            iota=seed_iota,
            G=seed_g,
        ),
        rejection_distance_scale=outer_policy.rejection_distance_scale,
        start_coil_dofs=start_coil_dofs,
    )
    return NativeOuterRunContext(
        run=run,
        outer_policy=outer_policy,
        objective_weights=objective.weights,
        # A real snapshot, not ``np.asarray``: ``local_full_x`` hands back the
        # ``DOFs`` object's own ``_x`` buffer, and ``asarray`` on a float64
        # array does not copy, so the frozen context would hold a live alias
        # to the vessel for the whole run and serialize whatever it had become
        # at payload time.
        vessel_dofs=np.array(vessel.local_full_x, dtype=np.float64, copy=True),
        lane_meta=lane_meta,
        seed_iota=seed_iota,
        seed_G=seed_g,
        start_coil_dofs=start_coil_dofs,
        threading=threading,
        module_import_seconds=MODULE_IMPORT_SECONDS,
        build_seconds=build_seconds,
        child_started=child_started,
    )


def _drive_native_run(context: NativeOuterRunContext) -> dict[str, object]:
    """Run the outer optimizer and return this child's complete payload."""

    run = context.run
    outer_policy = context.outer_policy
    solve_started = time.perf_counter()
    # Recovery only: a qualifying early stop restarts from the transaction's
    # complete incumbent with fresh L-BFGS memory. Trial evaluations cannot
    # alter that incumbent; the identical rule lives in the JAX twin.
    budget_total = int(outer_policy.maxiter)
    consumed_nit = 0
    total_nfev = 0
    total_njev = 0
    attempts_started = 0
    restart_nits: list[int] = []
    restart_attempts: list[dict[str, object]] = []
    attempt_x0 = context.start_coil_dofs
    faulted: NestedLsOuterAcceptWithoutCandidate | None = None
    result = None
    accepted_iterates = 0

    def _accept(coil_dofs: NDArray[np.float64]) -> None:
        """Promote the announced iterate and count it.

        The count is the only record of how far an attempt got when scipy
        never returns, which is the accept-without-candidate path. The JAX
        twin reads the same quantity off its ``outer_iterates`` ledger.
        """

        nonlocal accepted_iterates
        run.accept(coil_dofs)
        accepted_iterates += 1

    while True:
        attempts_started += 1
        rejected_before = sum(
            1 for record in run.records if record["rejection_reason"] is not None
        )
        records_before = len(run.records)
        accepted_before = accepted_iterates
        attempt_options = outer_policy.as_scipy_options()
        attempt_options["maxiter"] = budget_total - consumed_nit
        try:
            result = minimize(
                run,
                attempt_x0,
                jac=True,
                method=outer_policy.method,
                options=attempt_options,
                callback=_accept,
            )
        except NestedLsOuterAcceptWithoutCandidate as fault:
            # Fail closed, but not before the evidence is published: this
            # attempt produced no scipy verdict, and every row already in
            # the ledger stays in the payload below.
            #
            # scipy returned nothing to count with, so this attempt's share
            # of the published counters comes from the ledger the child kept
            # itself: one evaluation row per call into ``run``, and
            # ``jac=True`` means every such call returned the gradient too,
            # so nfev and njev are the same count. ``nit`` is the number of
            # accepted iterates this attempt completed — the faulting
            # callback completed none, which is why it can be zero beside a
            # nonzero nfev. No scipy verdict is fabricated: this attempt
            # contributes no ``restart_attempts`` row.
            faulted = fault
            faulted_nfev = len(run.records) - records_before
            consumed_nit += accepted_iterates - accepted_before
            total_nfev += faulted_nfev
            total_njev += faulted_nfev
            break
        attempt_nit = int(result.nit)
        grad_inf = float(np.max(np.abs(np.asarray(result.jac, dtype=np.float64))))
        # ``result.fun`` is whatever this attempt's last evaluation returned,
        # and a wholly rejected line search leaves it holding a containment
        # barrier while ``result.x`` and ``result.jac`` are restored to the
        # incumbent (reproduced by
        # tests/geo/test_nested_ls_outer_child_evidence.py on the shipped
        # fixture: barrier 0.31250004043721513 beside an anchor objective of
        # 0.3125). The raw scipy datum stays published and the shared
        # provenance rule decides, from this lane's own last ledger row,
        # whether it IS the objective. Both lanes call the one rule.
        last_record = run.records[-1]
        attempt_fun_is_objective = nested_ls_outer_attempt_fun_is_objective(
            reported_parameters=np.asarray(result.x, dtype=np.float64),
            last_evaluated_parameters=np.asarray(
                last_record["coil_dofs"], dtype=np.float64
            ),
            last_evaluation_value_is_valid=bool(last_record["value_is_valid"]),
        )
        rejected_this_attempt = (
            sum(1 for record in run.records if record["rejection_reason"] is not None)
            - rejected_before
        )
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
                "value_is_valid": attempt_fun_is_objective,
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
            run.candidates.committed.coil_dofs,
            dtype=np.float64,
            copy=True,
        )
    solve_seconds = float(time.perf_counter() - solve_started)

    if faulted is None:
        optimizer_x = np.asarray(result.x, dtype=np.float64)
        endpoint: OuterAnchor | None = run.endpoint_at(optimizer_x)
        endpoint_is_optimizer_x = endpoint is not None
        status: int | None = int(result.status)
        message = str(result.message)
        optimizer_success = bool(result.success)
        ftol_zero_stop = nested_ls_outer_ftol_zero_stop(
            ftol=float(outer_policy.ftol),
            message=message,
        )
        success = nested_ls_outer_endpoint_success(
            endpoint_matches=endpoint_is_optimizer_x,
            ftol=float(outer_policy.ftol),
            status=int(result.status),
            message=message,
        )
        child_fault_reason: str | None = None
    else:
        # The fault fired inside scipy's accepted-step callback, so this run
        # has no scipy verdict at all: ``status`` and ``message`` say that
        # rather than borrowing a previous attempt's. ``optimizer_x`` is the
        # point scipy announced and the store could not match — the honest
        # subject of the failure — and it is by construction not the
        # committed incumbent, which is why the store refused it. This path
        # names no endpoint of its own; the committed-incumbent fallback
        # below is the one that supplies it.
        optimizer_x = np.array(faulted.parameters, dtype=np.float64, copy=True)
        endpoint = None
        endpoint_is_optimizer_x = False
        status = None
        message = str(faulted)
        optimizer_success = False
        ftol_zero_stop = False
        success = False
        child_fault_reason = NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON
    if endpoint is None:
        endpoint = run.anchor
    if endpoint is None:
        raise ValueError("the native outer twin accepted no outer iterate.")
    endpoint_surface = np.asarray(endpoint.warm_start.surface_dofs, dtype=np.float64)
    payload: dict[str, object] = {
        "schema": NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
        "publication": PUBLICATION,
        # A fixed-budget run stops on maxiter, which scipy reports as
        # success=False, so the child's own flag is the usable-endpoint
        # predicate the parent needs: the reported endpoint is the
        # optimizer's own final iterate and the committed transaction point.
        # An FTOL stop is never success under the sealed ftol=0 policy.
        "success": success,
        # None on every completed run. A named fault means this payload is
        # published evidence of a failure, never a result: read it before
        # reading anything else here. It answers a different question from
        # ``success``, which is the judged outcome of a run that did produce
        # a scipy verdict; this key says the child could not complete and
        # produced no scipy verdict at all.
        "child_fault_reason": child_fault_reason,
        "optimizer_success": optimizer_success,
        "ftol_zero_stop": ftol_zero_stop,
        "feasible_evaluations": len(run.feasible),
        "rejected_evaluations": sum(
            1 for record in run.records if record["rejection_reason"] is not None
        ),
        "rejection_reasons": sorted(
            {
                str(record["rejection_reason"])
                for record in run.records
                if record["rejection_reason"] is not None
            }
        ),
        "budget": int(outer_policy.maxiter),
        "maxcor": int(outer_policy.maxcor),
        "nit": int(consumed_nit),
        "nfev": int(total_nfev),
        "njev": int(total_njev),
        # One restart per attempt after the first. Counted when an attempt
        # begins, not when scipy returns: an attempt that faults at the
        # accepted-step callback records no ``restart_nits`` entry, so
        # counting entries would under-report the restarts that really
        # happened on exactly the path where a reader most needs the truth.
        "restart_count": int(attempts_started - 1),
        "restart_nits": [int(value) for value in restart_nits],
        "restart_attempts": restart_attempts,
        "status": status,
        "message": message,
        "endpoint_is_optimizer_x": endpoint_is_optimizer_x,
        "optimizer_x": [float(value) for value in optimizer_x],
        "outer_policy": outer_policy.as_payload(),
        "start": {
            "lane": context.lane_meta,
            "coil_dofs": [float(value) for value in context.start_coil_dofs],
            "qr_seed_iota": context.seed_iota,
            "qr_seed_G": context.seed_G,
            "seed_provenance": (
                "lane result.endpoint_inner_state, read as archived; the live "
                "QR that produced it is deliberately not rerun on this lane"
            ),
            "evaluation": 0,
        },
        "endpoint": {
            "coil_dofs": [float(value) for value in endpoint.coil_dofs],
            "coil_sha256": sha256_float64(endpoint.coil_dofs),
            "objective": endpoint.objective,
            "terms": _term_ledger(endpoint.terms, context.objective_weights),
            "iota": endpoint.warm_start.iota,
            "G": endpoint.warm_start.G,
            "gradient": [float(value) for value in endpoint.gradient],
            "gradient_l2": float(np.linalg.norm(endpoint.gradient)),
            "surface_dofs": [float(value) for value in endpoint_surface],
            "surface_sha256": sha256_float64(endpoint_surface),
            "vessel_dofs": [float(value) for value in context.vessel_dofs],
        },
        "evaluations": run.records,
        "walls": {
            "clock": "informational_child_splits_only",
            # module_import_seconds bounds the residual JAX cost on this lane:
            # nothing here traces, but the adapter loaders import JAX.
            "module_import_seconds": context.module_import_seconds,
            "problem_build_seconds": context.build_seconds,
            "outer_solve_seconds": solve_seconds,
            "inner_seconds_total": float(
                sum(float(record["inner_seconds"]) for record in run.records)
            ),
            "main_seconds": float(time.perf_counter() - context.child_started),
            "child_total_seconds": float(time.perf_counter() - _IMPORT_STARTED),
        },
        "threading": context.threading,
        "omp_pinned": nested_ls_omp_threads_pinned(context.threading),
        "omp_num_threads": context.threading["OMP_NUM_THREADS"],
        # This lane's numbers come out of the compiled extension, so this
        # lane must bind it. The parent's own record cannot stand in: the
        # children are separate processes launched with rewritten
        # environments, so only the child that loaded a binary can witness
        # which one it loaded. Sampled at payload time because
        # ``nested_ls_runtime_identity`` ends in a wall-clock
        # ``timestamp_utc`` and hashes the extension's bytes as they are when
        # it is called — both are write-time facts. Same function on both
        # lanes, so the two receipts are comparable field for field.
        "runtime": nested_ls_runtime_identity(),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    context = _prepare_native_run(budget=int(args.budget), maxcor=int(args.maxcor))
    payload = _drive_native_run(context)
    write_strict_json(args.out_json, payload)
    endpoint = payload["endpoint"]
    print(
        "native outer"
        f" nit={payload['nit']} nfev={payload['nfev']}"
        f" J={endpoint['objective']!r} iota={endpoint['iota']!r}"
        f" child_fault_reason={payload['child_fault_reason']!r}"
        f" wrote={args.out_json}",
        flush=True,
    )
    # Fail closed only after the receipt is on disk. The parent sees a
    # nonzero child, and the evidence the run did accumulate survives it.
    if payload["child_fault_reason"] is not None:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
