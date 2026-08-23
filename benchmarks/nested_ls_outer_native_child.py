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
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE,
    DEFAULT_F3_B37_NATIVE_LANE,
    DEFAULT_FLAT675_BUNDLE_ROOT,
    _run_native_banana_bfgs_then_newton,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    nested_ls_omp_threads_pinned,
    nested_ls_threading_env,
    sha256_float64,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    _TRACEABLE_SINGLE_STAGE_OUTER_TERM_WEIGHT_KEYS,
    SurfaceSurfaceDistance,
)

from benchmarks.nested_ls_shamanskii_attribution import write_strict_json

MODULE_IMPORT_SECONDS = time.perf_counter() - _IMPORT_STARTED

SCHEMA = "nested-ls-outer-native-child.v1"
# The sealed F3 native-lane behaviours this child reimplements. A lane whose
# policy names differ is a different outer formulation, so the child refuses it
# rather than silently optimizing something else.
IMPLEMENTED_LANE_POLICY_NAMES = {
    "accepted_state_policy": "rolling_last_accepted_anchor",
    "method": "L-BFGS-B",
    "rejection_gradient_policy": "accepted_anchor_gradient",
    "rejection_rollback_policy": "retain_accepted_anchor",
    "rejection_value_policy": "anchor_plus_offset_plus_distance_v1",
}
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
    ``maxcor`` are the charter rung's. The two rejection scalars parameterize
    the lane's ``anchor_plus_offset_plus_distance_v1`` policy.
    """

    source: str
    method: str
    ftol: float
    gtol: float
    maxls: int
    maxiter: int
    maxcor: int
    rejection_value_offset: float
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
            "rejection_value_offset": self.rejection_value_offset,
            "rejection_distance_scale": self.rejection_distance_scale,
            "iota_branch_guard": float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
            "lane_policy_names": self.lane_policy_names,
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
        rejection_value_offset=float(policy["rejection_value_offset"]),
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


@dataclass(slots=True)
class OuterAnchor:
    """Last accepted outer iterate: the state every solve warm-starts from."""

    coil_dofs: NDArray[np.float64]
    warm_start: InnerWarmStart
    objective: float
    gradient: NDArray[np.float64]
    terms: dict[str, float]


class NativeOuterRun:
    """scipy-facing ``fun(x) -> (J, dJ/dc)`` with a rolling accepted anchor.

    An evaluation is accepted only when its nested solve converges *and*
    stays on the warm start's Boozer branch: charter Amendment 1 rules that
    ``s*(c)`` is local, so an ``iota`` moving more than
    ``NESTED_LS_OUTER_IOTA_BRANCH_GUARD`` is a failed evaluation however
    well it converged. Either rejection reinstates the anchor and reports
    its objective raised by ``rejection_value_offset`` plus the scaled
    distance travelled, with the anchor's gradient, so the line search
    shortens instead of stepping onto another branch. Evaluation 0 has no
    anchor to fall back to and fails the run instead of sentinelling — a
    start point that cannot be solved on the archived seed's own branch is
    not this problem.
    """

    def __init__(
        self,
        objective: NativeOuterObjective,
        *,
        seed: InnerWarmStart,
        rejection_value_offset: float,
        rejection_distance_scale: float,
    ) -> None:
        self.objective = objective
        self.seed = seed
        self.anchor: OuterAnchor | None = None
        self.rejection_value_offset = rejection_value_offset
        self.rejection_distance_scale = rejection_distance_scale
        self.records: list[dict[str, object]] = []
        self.accepted: list[OuterAnchor] = []

    def endpoint_at(self, coil_dofs: NDArray[np.float64]) -> OuterAnchor | None:
        """Return the accepted iterate whose coil DOFs are exactly ``coil_dofs``."""

        for candidate in reversed(self.accepted):
            if np.array_equal(candidate.coil_dofs, coil_dofs):
                return candidate
        return None

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
            "rejection_reason": reason,
        }
        if reason is not None:
            anchor = self.anchor
            if anchor is None:
                raise ValueError(
                    "the native outer twin's start-point evaluation was rejected "
                    f"({reason}); there is no accepted anchor to fall back to."
                )
            self._reinstate_warm_start()
            distance = float(np.linalg.norm(coil_dofs - anchor.coil_dofs))
            value = (
                anchor.objective
                + self.rejection_value_offset
                + self.rejection_distance_scale * distance
            )
            gradient = np.array(anchor.gradient, copy=True)
            record.update(
                {
                    "accepted": False,
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
        self.anchor = OuterAnchor(
            coil_dofs=np.array(coil_dofs, copy=True),
            warm_start=InnerWarmStart(
                surface_dofs=np.asarray(boozer.surface.get_dofs(), dtype=np.float64),
                iota=float(solve["iota"]),
                G=float(solve["G"]),
            ),
            objective=value,
            gradient=np.array(gradient, copy=True),
            terms=terms,
        )
        self.accepted.append(self.anchor)
        record.update(
            {
                "accepted": True,
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    child_started = time.perf_counter()
    threading = nested_ls_threading_env()
    outer_policy = load_outer_optimizer_policy(
        DEFAULT_F3_B37_NATIVE_LANE,
        budget=int(args.budget),
        maxcor=int(args.maxcor),
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
        rejection_value_offset=outer_policy.rejection_value_offset,
        rejection_distance_scale=outer_policy.rejection_distance_scale,
    )

    solve_started = time.perf_counter()
    result = minimize(
        run,
        start_coil_dofs,
        jac=True,
        method=outer_policy.method,
        options=outer_policy.as_scipy_options(),
    )
    solve_seconds = float(time.perf_counter() - solve_started)

    optimizer_x = np.asarray(result.x, dtype=np.float64)
    endpoint = run.endpoint_at(optimizer_x)
    endpoint_is_optimizer_x = endpoint is not None
    if endpoint is None:
        endpoint = run.anchor
    if endpoint is None:
        raise ValueError("the native outer twin accepted no outer iterate.")
    endpoint_surface = np.asarray(endpoint.warm_start.surface_dofs, dtype=np.float64)
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "publication": PUBLICATION,
        # A fixed-budget run stops on maxiter, which scipy reports as
        # success=False, so the child's own flag is the usable-endpoint
        # predicate the parent needs: the reported endpoint is the
        # optimizer's own final iterate *and* an accepted one, since only
        # converged on-branch evaluations enter ``run.accepted``. Rejected
        # line-search trials are the sealed policy working, not a failed
        # run, so they are counted rather than folded into this flag.
        "success": endpoint_is_optimizer_x,
        "optimizer_success": bool(result.success),
        "accepted_evaluations": len(run.accepted),
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
        "budget": int(args.budget),
        "maxcor": int(args.maxcor),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "status": int(result.status),
        "message": str(result.message),
        "endpoint_is_optimizer_x": endpoint_is_optimizer_x,
        "optimizer_x": [float(value) for value in optimizer_x],
        "outer_policy": outer_policy.as_payload(),
        "start": {
            "lane": lane_meta,
            "coil_dofs": [float(value) for value in start_coil_dofs],
            "qr_seed_iota": seed_iota,
            "qr_seed_G": seed_g,
            "seed_provenance": (
                "lane result.endpoint_inner_state, read as archived; the live "
                "QR that produced it is deliberately not rerun on this lane"
            ),
            "evaluation": 0,
        },
        "endpoint": {
            "coil_dofs": [float(value) for value in endpoint.coil_dofs],
            "objective": endpoint.objective,
            "terms": _term_ledger(endpoint.terms, objective.weights),
            "iota": endpoint.warm_start.iota,
            "G": endpoint.warm_start.G,
            "gradient": [float(value) for value in endpoint.gradient],
            "gradient_l2": float(np.linalg.norm(endpoint.gradient)),
            "surface_dofs": [float(value) for value in endpoint_surface],
            "surface_sha256": sha256_float64(endpoint_surface),
            "vessel_dofs": [float(value) for value in np.asarray(vessel.local_full_x)],
        },
        "evaluations": run.records,
        "walls": {
            "clock": "informational_child_splits_only",
            # module_import_seconds bounds the residual JAX cost on this lane:
            # nothing here traces, but the adapter loaders import JAX.
            "module_import_seconds": MODULE_IMPORT_SECONDS,
            "problem_build_seconds": build_seconds,
            "outer_solve_seconds": solve_seconds,
            "inner_seconds_total": float(
                sum(float(record["inner_seconds"]) for record in run.records)
            ),
            "main_seconds": float(time.perf_counter() - child_started),
            "child_total_seconds": float(time.perf_counter() - _IMPORT_STARTED),
        },
        "threading": threading,
        "omp_pinned": nested_ls_omp_threads_pinned(threading),
        "omp_num_threads": threading["OMP_NUM_THREADS"],
    }
    write_strict_json(args.out_json, payload)
    print(
        "native outer"
        f" nit={payload['nit']} nfev={payload['nfev']}"
        f" J={endpoint.objective!r} iota={endpoint.warm_start.iota!r}"
        f" wrote={args.out_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
