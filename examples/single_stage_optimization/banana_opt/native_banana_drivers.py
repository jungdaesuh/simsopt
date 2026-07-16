from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Protocol
import json
import os
import platform

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable, load
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    Iotas,
    LpCurveCurvature,
    NonQuasiSymmetricRatio,
    Surface,
    SurfaceRZFourier,
    SurfaceXYZFourier,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt.objectives import QuadraticPenalty

from .jax_banana_types import (
    BANANA_IDX,
    COMMON_OBJECTIVE_CONTRACT_ID,
    COMMON_OBJECTIVE_TERM_NAMES,
    HARDWARE_LIMITS,
    N_BANANA,
    SingleStageWeights,
    WeightedTerm,
)


SurfaceType = SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier


class Saveable(Protocol):
    def save(self, filename: str) -> None: ...


@dataclass(frozen=True)
class NativeSingleStageConfig:
    biotsavart_file: Path
    surface_path: Path
    boozer_state_path: Path
    output_root: Path
    overwrite: bool
    run_config_sha256: str
    objective_profile: str
    vmec_s: float
    surface_scale: float | None
    mpol: int
    ntor: int
    nphi: int
    ntheta: int
    constraint_weight: float
    iota_target: float
    boozer_bfgs_tol: float
    boozer_bfgs_maxiter: int
    boozer_newton_tol: float
    boozer_newton_maxiter: int
    boozer_limited_memory: bool
    outer_maxiter: int
    outer_maxcor: int
    outer_maxls: int
    outer_ftol: float
    outer_gtol: float
    weights: SingleStageWeights


@dataclass(frozen=True)
class NativeOutputPaths:
    results: Path
    biotsavart: Path
    surface: Path
    boozersurface: Path


@dataclass(frozen=True)
class BoozerSolveState:
    iota: float
    G: float


@dataclass(frozen=True)
class _AcceptedState:
    optimizer_dofs: np.ndarray
    surface_dofs: np.ndarray
    iota: float
    G: float
    objective: float
    gradient: np.ndarray


@dataclass(frozen=True)
class NativeOptimizationStats:
    accepted_iterations: int
    evaluations: int
    rejected_evaluations: int
    outer_inner_solve_s: float
    outer_objective_s: float
    endpoint_reconciliation_inner_solve_s: float
    endpoint_reconciliation_objective_s: float
    final_inner_solve_s: float
    final_objective_s: float


class _RecordedObjective(Optimizable):
    """Record one term value without evaluating the term a second time."""

    def __init__(self, objective: Optimizable) -> None:
        self.objective = objective
        self.last_value: float | None = None
        super().__init__(depends_on=[objective])

    def J(self) -> float:
        self.last_value = float(self.objective.J())
        return self.last_value

    @derivative_dec
    def dJ(self):
        return self.objective.dJ(partials=True)

    return_fn_map = {"J": J, "dJ": dJ}


def _require_inactive_coil_surface_distance(objective: _RecordedObjective) -> None:
    if objective.last_value is None or objective.last_value > 0.0:
        raise RuntimeError(
            "The matched seven-term benchmark requires the coil-surface-distance "
            "hinge to remain inactive because its implicit Boozer-surface derivative "
            "is not implemented"
        )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _float_array_sha256(values: np.ndarray) -> str:
    canonical = np.asarray(values, dtype="<f8").reshape(-1)
    return sha256(canonical.tobytes(order="C")).hexdigest()


def _string_sequence_sha256(values: list[str]) -> str:
    payload = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _save_atomic(saveable: Saveable, path: Path) -> None:
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    saveable.save(str(temporary_path))
    temporary_path.replace(path)


def native_output_paths(output_root: Path) -> NativeOutputPaths:
    return NativeOutputPaths(
        results=output_root / "results.json",
        biotsavart=output_root / "biot_savart_opt.json",
        surface=output_root / "surf_opt.json",
        boozersurface=output_root / "boozersurface_opt.json",
    )


def ensure_writable_native_outputs(
    paths: NativeOutputPaths,
    *,
    overwrite: bool,
) -> None:
    existing = [path for path in paths.__dict__.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite native run artifacts: {joined}")
    paths.results.parent.mkdir(parents=True, exist_ok=True)


def _validate_config(config: NativeSingleStageConfig) -> None:
    if config.objective_profile != "common-seven-term":
        raise ValueError(
            "Native full-loop comparison requires objective_profile=common-seven-term"
        )
    if config.constraint_weight <= 0.0:
        raise ValueError(
            "Native full-loop comparison requires BoozerLS constraint_weight > 0"
        )
    if len(config.run_config_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in config.run_config_sha256
    ):
        raise ValueError("run_config_sha256 must be a lowercase SHA-256 digest")
    positive_integers = {
        "mpol": config.mpol,
        "ntor": config.ntor,
        "nphi": config.nphi,
        "ntheta": config.ntheta,
        "boozer_bfgs_maxiter": config.boozer_bfgs_maxiter,
        "boozer_newton_maxiter": config.boozer_newton_maxiter,
        "outer_maxiter": config.outer_maxiter,
        "outer_maxcor": config.outer_maxcor,
        "outer_maxls": config.outer_maxls,
    }
    invalid_integers = [name for name, value in positive_integers.items() if value < 1]
    if invalid_integers:
        raise ValueError(
            f"Expected positive integer controls: {', '.join(invalid_integers)}"
        )
    positive_tolerances = {
        "boozer_bfgs_tol": config.boozer_bfgs_tol,
        "boozer_newton_tol": config.boozer_newton_tol,
        "outer_ftol": config.outer_ftol,
        "outer_gtol": config.outer_gtol,
    }
    invalid_tolerances = [
        name for name, value in positive_tolerances.items() if value <= 0.0
    ]
    if invalid_tolerances:
        raise ValueError(
            f"Expected positive solver tolerances: {', '.join(invalid_tolerances)}"
        )


def _load_biotsavart(path: Path) -> BiotSavart:
    loaded = load(str(path))
    if not isinstance(loaded, BiotSavart):
        raise TypeError(f"Expected BiotSavart JSON, got {type(loaded)!r}: {path}")
    if len(loaded.coils) < BANANA_IDX + N_BANANA:
        raise ValueError(
            "BiotSavart seed does not contain the expected TF and banana-coil blocks"
        )
    return loaded


def _load_surface(
    path: Path,
    *,
    vmec_s: float,
    scale: float | None,
    nphi: int,
    ntheta: int,
) -> SurfaceType:
    if path.suffix == ".json":
        loaded = load(str(path))
        if not isinstance(
            loaded,
            SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier,
        ):
            raise TypeError(f"Expected a supported surface JSON, got {type(loaded)!r}")
        surface = loaded
    elif path.suffix == ".nc":
        surface = SurfaceRZFourier.from_wout(str(path), s=vmec_s)
        if scale is not None:
            surface.set_dofs(surface.get_dofs() * float(scale) / surface.major_radius())
    else:
        raise ValueError(
            f"Unsupported surface extension {path.suffix!r}; expected .json or .nc"
        )
    return resize_surface(surface, nphi=nphi, ntheta=ntheta)


def resize_surface(
    initial_surface: SurfaceType,
    *,
    mpol: int | None = None,
    ntor: int | None = None,
    nphi: int,
    ntheta: int,
) -> SurfaceType:
    """Copy a seed surface onto an explicit Fourier and quadrature grid."""
    target_mpol = initial_surface.mpol if mpol is None else int(mpol)
    target_ntor = initial_surface.ntor if ntor is None else int(ntor)
    if isinstance(initial_surface, SurfaceRZFourier):
        return initial_surface.copy(
            mpol=target_mpol,
            ntor=target_ntor,
            nphi=nphi,
            ntheta=ntheta,
        )

    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=nphi,
        ntheta=ntheta,
        nfp=initial_surface.nfp,
    )
    if isinstance(initial_surface, SurfaceXYZFourier):
        source = SurfaceXYZFourier(
            mpol=initial_surface.mpol,
            ntor=initial_surface.ntor,
            nfp=initial_surface.nfp,
            stellsym=initial_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        target: SurfaceType = SurfaceXYZFourier(
            mpol=target_mpol,
            ntor=target_ntor,
            nfp=initial_surface.nfp,
            stellsym=initial_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    else:
        source = SurfaceXYZTensorFourier(
            mpol=initial_surface.mpol,
            ntor=initial_surface.ntor,
            nfp=initial_surface.nfp,
            stellsym=initial_surface.stellsym,
            clamped_dims=list(initial_surface.clamped_dims),
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        target = SurfaceXYZTensorFourier(
            mpol=target_mpol,
            ntor=target_ntor,
            nfp=initial_surface.nfp,
            stellsym=initial_surface.stellsym,
            clamped_dims=list(initial_surface.clamped_dims),
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    source.set_dofs(initial_surface.get_dofs())
    target.least_squares_fit(source.gamma())
    return target


def _boozer_quadpoints(
    *,
    nfp: int,
    stellsym: bool,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> tuple[np.ndarray, np.ndarray]:
    if stellsym and nphi == 2 * ntor + 1 and ntheta == 2 * mpol + 1:
        return (
            np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
            np.linspace(0.0, 1.0, ntheta, endpoint=False),
        )
    if stellsym and nphi == 2 * ntor + 1 and ntheta == mpol + 1:
        return (
            np.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
            np.linspace(0.0, 0.5, ntheta, endpoint=False),
        )
    if stellsym and nphi == ntor + 1 and ntheta == 2 * mpol + 1:
        return (
            np.linspace(0.0, 1.0 / (2.0 * nfp), nphi, endpoint=False),
            np.linspace(0.0, 1.0, ntheta, endpoint=False),
        )
    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=nphi,
        ntheta=ntheta,
        nfp=nfp,
    )
    return np.asarray(quadpoints_phi, dtype=float), np.asarray(
        quadpoints_theta, dtype=float
    )


def build_boozer_surface_copy(
    initial_surface: SurfaceType,
    *,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> SurfaceXYZTensorFourier:
    """Fit any supported seed representation into the native Boozer tensor basis."""
    quadpoints_phi, quadpoints_theta = _boozer_quadpoints(
        nfp=initial_surface.nfp,
        stellsym=initial_surface.stellsym,
        mpol=mpol,
        ntor=ntor,
        nphi=nphi,
        ntheta=ntheta,
    )
    if isinstance(initial_surface, SurfaceRZFourier):
        source: SurfaceType = SurfaceRZFourier(
            mpol=initial_surface.mpol,
            ntor=initial_surface.ntor,
            nfp=initial_surface.nfp,
            stellsym=initial_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    elif isinstance(initial_surface, SurfaceXYZFourier):
        source = SurfaceXYZFourier(
            mpol=initial_surface.mpol,
            ntor=initial_surface.ntor,
            nfp=initial_surface.nfp,
            stellsym=initial_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    else:
        source = SurfaceXYZTensorFourier(
            mpol=initial_surface.mpol,
            ntor=initial_surface.ntor,
            nfp=initial_surface.nfp,
            stellsym=initial_surface.stellsym,
            clamped_dims=list(initial_surface.clamped_dims),
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    source.set_dofs(initial_surface.get_dofs())
    boozer_surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=initial_surface.nfp,
        stellsym=initial_surface.stellsym,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    boozer_surface.least_squares_fit(source.gamma())
    return boozer_surface


def _load_initial_boozer_state(path: Path) -> BoozerSolveState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BoozerSolveState(iota=float(payload["iota"]), G=float(payload["G"]))


def build_native_boozer_surface(
    *,
    biotsavart: BiotSavart,
    surface: SurfaceType,
    config: NativeSingleStageConfig,
) -> BoozerSurface:
    boozer_surface = build_boozer_surface_copy(
        surface,
        mpol=config.mpol,
        ntor=config.ntor,
        nphi=config.nphi,
        ntheta=config.ntheta,
    )
    label = Volume(boozer_surface)
    return BoozerSurface(
        biotsavart,
        boozer_surface,
        label,
        boozer_surface.volume(),
        constraint_weight=config.constraint_weight,
        options={
            "verbose": False,
            "bfgs_tol": config.boozer_bfgs_tol,
            "bfgs_maxiter": config.boozer_bfgs_maxiter,
            "newton_tol": config.boozer_newton_tol,
            "newton_maxiter": config.boozer_newton_maxiter,
            "limited_memory": config.boozer_limited_memory,
        },
    )


def _solve_boozer(
    boozer_surface: BoozerSurface,
    state: BoozerSolveState,
) -> dict[str, object]:
    result = boozer_surface.run_code(state.iota, state.G)
    if result is None:
        result = boozer_surface.res
    if not bool(result["success"]):
        raise RuntimeError("Native BoozerLS solve failed")
    return result


def _banana_curves(biotsavart: BiotSavart) -> list[object]:
    return [coil.curve for coil in biotsavart.coils[BANANA_IDX : BANANA_IDX + N_BANANA]]


def _weighted_sum(terms: tuple[WeightedTerm, ...]) -> Optimizable:
    active_terms = tuple(term for term in terms if term.weight != 0.0)
    if not active_terms:
        raise ValueError("The native common objective requires a nonzero term weight")
    objective = active_terms[0].weight * active_terms[0].objective
    for term in active_terms[1:]:
        objective = objective + term.weight * term.objective
    return objective


def build_native_single_stage_objective(
    *,
    boozer_surface: BoozerSurface,
    biotsavart: BiotSavart,
    iota_target: float,
    weights: SingleStageWeights,
) -> tuple[Optimizable, tuple[WeightedTerm, ...], _RecordedObjective]:
    """Build the seven-term native objective shared with the FP64 JAX lane."""
    if weights.csdist <= 0.0:
        raise ValueError(
            "The common seven-term contract requires a positive "
            "coil-surface-distance weight"
        )
    banana_curves = _banana_curves(biotsavart)
    base_curve = banana_curves[0]
    coil_surface_distance = _RecordedObjective(
        CurveSurfaceDistance(
            banana_curves,
            boozer_surface.surface,
            HARDWARE_LIMITS.min_csdist,
        )
    )
    terms = (
        WeightedTerm(
            COMMON_OBJECTIVE_TERM_NAMES[0],
            weights.nonqs,
            NonQuasiSymmetricRatio(boozer_surface, biotsavart),
        ),
        WeightedTerm(
            COMMON_OBJECTIVE_TERM_NAMES[1],
            weights.bres,
            BoozerResidual(boozer_surface, biotsavart),
        ),
        WeightedTerm(
            COMMON_OBJECTIVE_TERM_NAMES[2],
            weights.iota,
            QuadraticPenalty(Iotas(boozer_surface), iota_target),
        ),
        WeightedTerm(
            COMMON_OBJECTIVE_TERM_NAMES[3],
            weights.length,
            QuadraticPenalty(
                CurveLength(base_curve),
                HARDWARE_LIMITS.max_length,
                "max",
            ),
        ),
        WeightedTerm(
            COMMON_OBJECTIVE_TERM_NAMES[4],
            weights.ccdist,
            CurveCurveDistance(banana_curves, HARDWARE_LIMITS.min_ccdist),
        ),
        WeightedTerm(
            COMMON_OBJECTIVE_TERM_NAMES[5],
            weights.csdist,
            coil_surface_distance,
        ),
        WeightedTerm(
            COMMON_OBJECTIVE_TERM_NAMES[6],
            weights.curvature,
            LpCurveCurvature(
                base_curve,
                HARDWARE_LIMITS.banana_curv_p,
                HARDWARE_LIMITS.max_curvature,
            ),
        ),
    )
    return _weighted_sum(terms), terms, coil_surface_distance


def _accepted_state(
    objective: Optimizable,
    boozer_surface: BoozerSurface,
    *,
    objective_value: float,
    gradient: np.ndarray,
) -> _AcceptedState:
    result = boozer_surface.res
    return _AcceptedState(
        optimizer_dofs=np.asarray(objective.x, dtype=np.float64).copy(),
        surface_dofs=np.asarray(
            boozer_surface.surface.get_dofs(), dtype=np.float64
        ).copy(),
        iota=float(result["iota"]),
        G=float(result["G"]),
        objective=float(objective_value),
        gradient=np.asarray(gradient, dtype=np.float64).copy(),
    )


class NativeSingleStageEvaluator:
    """Own accepted-state warm starts and rollback for host SciPy L-BFGS-B.

    Trial Boozer solves always start from the last accepted surface. Failed trials
    return a finite elevated value, while callbacks atomically promote solved trials.
    """

    def __init__(
        self,
        *,
        objective: Optimizable,
        boozer_surface: BoozerSurface,
        initial_state: _AcceptedState,
        coil_surface_distance: _RecordedObjective,
    ) -> None:
        self.objective = objective
        self.boozer_surface = boozer_surface
        self.accepted_state = initial_state
        self.coil_surface_distance = coil_surface_distance
        self.last_trial_state: _AcceptedState | None = initial_state
        self.accepted_iterations = 0
        self.evaluations = 0
        self.rejected_evaluations = 0
        self.outer_inner_solve_s = 0.0
        self.outer_objective_s = 0.0

    def _restore_accepted_warm_start(self, candidate_dofs: np.ndarray) -> None:
        self.objective.x = np.asarray(candidate_dofs, dtype=np.float64)
        self.boozer_surface.surface.set_dofs(self.accepted_state.surface_dofs)
        self.boozer_surface.need_to_run_code = True

    def _reject_trial(self, reason: str) -> tuple[float, np.ndarray]:
        if self.evaluations == 1:
            raise RuntimeError(
                "Initial Boozer value/gradient evaluation failed during native "
                "banana single-stage optimization"
            )
        self.rejected_evaluations += 1
        self.last_trial_state = None
        self.boozer_surface.surface.set_dofs(self.accepted_state.surface_dofs)
        self.boozer_surface.need_to_run_code = True
        penalty_scale = max(abs(self.accepted_state.objective), 1.0)
        rejected_value = self.accepted_state.objective + 2.0 * penalty_scale
        print(
            f"eval={self.evaluations} rejected_{reason}=true J={rejected_value:.16e}",
            flush=True,
        )
        return rejected_value, self.accepted_state.gradient.copy()

    def value_and_gradient(
        self, candidate_dofs: np.ndarray
    ) -> tuple[float, np.ndarray]:
        candidate = np.asarray(candidate_dofs, dtype=np.float64)
        self._restore_accepted_warm_start(candidate)
        solve_started = perf_counter()
        result = self.boozer_surface.run_code(
            self.accepted_state.iota,
            self.accepted_state.G,
        )
        self.outer_inner_solve_s += perf_counter() - solve_started
        if result is None:
            result = self.boozer_surface.res
        self.evaluations += 1
        if not bool(result["success"]):
            return self._reject_trial("boozer_solve")

        objective_started = perf_counter()
        objective_value = float(self.objective.J())
        _require_inactive_coil_surface_distance(self.coil_surface_distance)
        gradient = np.asarray(self.objective.dJ(), dtype=np.float64)
        self.outer_objective_s += perf_counter() - objective_started
        if not np.isfinite(objective_value) or not np.all(np.isfinite(gradient)):
            return self._reject_trial("nonfinite_objective")
        self.last_trial_state = _accepted_state(
            self.objective,
            self.boozer_surface,
            objective_value=objective_value,
            gradient=gradient,
        )
        if self.evaluations == 1 and np.array_equal(
            candidate,
            self.accepted_state.optimizer_dofs,
        ):
            # SciPy evaluates x0 but does not invoke the callback for it. Match the
            # JAX lane by promoting that repeated, successfully solved x0 state.
            self.accepted_state = self.last_trial_state
        print(
            f"eval={self.evaluations} iota={float(result['iota']):.16e} "
            f"J={objective_value:.16e} |grad|={np.linalg.norm(gradient):.16e}",
            flush=True,
        )
        return objective_value, gradient

    def accept(self, accepted_dofs: np.ndarray) -> None:
        accepted = np.asarray(accepted_dofs, dtype=np.float64)
        if self.last_trial_state is None or not np.array_equal(
            accepted,
            self.last_trial_state.optimizer_dofs,
        ):
            self.value_and_gradient(accepted)
        if self.last_trial_state is None:
            raise RuntimeError(
                "SciPy accepted a candidate without a successful Boozer solve"
            )
        self.accepted_state = self.last_trial_state
        self.accepted_iterations += 1
        print(
            f"accepted_iter={self.accepted_iterations} "
            f"J={self.accepted_state.objective:.16e}",
            flush=True,
        )

    def finalize(
        self, final_dofs: np.ndarray
    ) -> tuple[_AcceptedState, NativeOptimizationStats]:
        final = np.asarray(final_dofs, dtype=np.float64)
        endpoint_reconciliation_inner_solve_s = 0.0
        endpoint_reconciliation_objective_s = 0.0
        if not np.array_equal(final, self.accepted_state.optimizer_dofs):
            prior_inner_solve_s = self.outer_inner_solve_s
            prior_objective_s = self.outer_objective_s
            self.value_and_gradient(final)
            endpoint_reconciliation_inner_solve_s = (
                self.outer_inner_solve_s - prior_inner_solve_s
            )
            endpoint_reconciliation_objective_s = (
                self.outer_objective_s - prior_objective_s
            )
            self.outer_inner_solve_s = prior_inner_solve_s
            self.outer_objective_s = prior_objective_s
            if self.last_trial_state is None:
                raise RuntimeError(
                    "Optimizer endpoint has no successful Boozer solved state"
                )
            self.accepted_state = self.last_trial_state

        self._restore_accepted_warm_start(final)
        final_solve_started = perf_counter()
        result = self.boozer_surface.run_code(
            self.accepted_state.iota,
            self.accepted_state.G,
        )
        final_inner_solve_s = perf_counter() - final_solve_started
        if result is None:
            result = self.boozer_surface.res
        if not bool(result["success"]):
            raise RuntimeError(
                "Final native BoozerLS solve failed at the optimizer endpoint"
            )

        final_objective_started = perf_counter()
        objective_value = float(self.objective.J())
        _require_inactive_coil_surface_distance(self.coil_surface_distance)
        gradient = np.asarray(self.objective.dJ(), dtype=np.float64)
        final_objective_s = perf_counter() - final_objective_started
        if not np.isfinite(objective_value) or not np.all(np.isfinite(gradient)):
            raise RuntimeError("Final native objective or gradient is non-finite")
        self.accepted_state = _accepted_state(
            self.objective,
            self.boozer_surface,
            objective_value=objective_value,
            gradient=gradient,
        )
        return self.accepted_state, NativeOptimizationStats(
            accepted_iterations=self.accepted_iterations,
            evaluations=self.evaluations,
            rejected_evaluations=self.rejected_evaluations,
            outer_inner_solve_s=self.outer_inner_solve_s,
            outer_objective_s=self.outer_objective_s,
            endpoint_reconciliation_inner_solve_s=(
                endpoint_reconciliation_inner_solve_s
            ),
            endpoint_reconciliation_objective_s=(endpoint_reconciliation_objective_s),
            final_inner_solve_s=final_inner_solve_s,
            final_objective_s=final_objective_s,
        )


def _term_payload(terms: tuple[WeightedTerm, ...]) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for term in terms:
        raw_value = float(term.objective.J())
        payload[term.name] = {
            "raw": raw_value,
            "weight": float(term.weight),
            "weighted": float(term.weight) * raw_value,
        }
    return payload


def _state_payload(
    state: _AcceptedState,
    *,
    terms: tuple[WeightedTerm, ...],
    volume: float,
) -> dict[str, object]:
    return {
        "objective": state.objective,
        "gradient_norm": float(np.linalg.norm(state.gradient)),
        "iota": state.iota,
        "G": state.G,
        "volume": float(volume),
        "dofs": state.optimizer_dofs.tolist(),
        "dof_count": int(state.optimizer_dofs.size),
        "dofs_sha256": _float_array_sha256(state.optimizer_dofs),
        "gradient": state.gradient.tolist(),
        "gradient_count": int(state.gradient.size),
        "gradient_sha256": _float_array_sha256(state.gradient),
        "terms": _term_payload(terms),
    }


def _input_payload(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def _config_payload(config: NativeSingleStageConfig) -> dict[str, object]:
    return {
        "objective_profile": config.objective_profile,
        "vmec_s": config.vmec_s,
        "surface_scale": config.surface_scale,
        "mpol": config.mpol,
        "ntor": config.ntor,
        "nphi": config.nphi,
        "ntheta": config.ntheta,
        "constraint_weight": config.constraint_weight,
        "iota_target": config.iota_target,
        "boozer_bfgs_tol": config.boozer_bfgs_tol,
        "boozer_bfgs_maxiter": config.boozer_bfgs_maxiter,
        "boozer_newton_tol": config.boozer_newton_tol,
        "boozer_newton_maxiter": config.boozer_newton_maxiter,
        "boozer_limited_memory": config.boozer_limited_memory,
        "outer_maxiter": config.outer_maxiter,
        "outer_maxcor": config.outer_maxcor,
        "outer_maxls": config.outer_maxls,
        "outer_ftol": config.outer_ftol,
        "outer_gtol": config.outer_gtol,
    }


def _optimizer_payload(
    optimizer_result: OptimizeResult,
    stats: NativeOptimizationStats,
    config: NativeSingleStageConfig,
) -> dict[str, object]:
    return {
        "method": "L-BFGS-B",
        "success": bool(optimizer_result.success),
        "status": int(optimizer_result.status),
        "message": str(optimizer_result.message),
        "nit": int(optimizer_result.nit),
        "nfev": int(optimizer_result.nfev),
        "njev": int(optimizer_result.njev),
        "accepted_iterations": stats.accepted_iterations,
        "evaluations": stats.evaluations,
        "rejected_evaluations": stats.rejected_evaluations,
        "maxiter": config.outer_maxiter,
        "maxcor": config.outer_maxcor,
        "maxls": config.outer_maxls,
        "ftol": config.outer_ftol,
        "gtol": config.outer_gtol,
    }


def _common_weights(weights: SingleStageWeights) -> dict[str, float]:
    return {
        COMMON_OBJECTIVE_TERM_NAMES[0]: weights.nonqs,
        COMMON_OBJECTIVE_TERM_NAMES[1]: weights.bres,
        COMMON_OBJECTIVE_TERM_NAMES[2]: weights.iota,
        COMMON_OBJECTIVE_TERM_NAMES[3]: weights.length,
        COMMON_OBJECTIVE_TERM_NAMES[4]: weights.ccdist,
        COMMON_OBJECTIVE_TERM_NAMES[5]: weights.csdist,
        COMMON_OBJECTIVE_TERM_NAMES[6]: weights.curvature,
    }


def run_native_single_stage(config: NativeSingleStageConfig) -> dict[str, object]:
    """Run and persist one native CPU seven-term banana full-loop optimization.

    The function owns input normalization, accepted-state Boozer warm starts,
    SciPy L-BFGS-B, final re-solve, and the reproducibility artifact contract.
    """
    run_started = perf_counter()
    _validate_config(config)
    paths = native_output_paths(config.output_root)
    ensure_writable_native_outputs(paths, overwrite=config.overwrite)

    setup_started = perf_counter()
    biotsavart = _load_biotsavart(config.biotsavart_file)
    input_surface = _load_surface(
        config.surface_path,
        vmec_s=config.vmec_s,
        scale=config.surface_scale,
        nphi=config.nphi,
        ntheta=config.ntheta,
    )
    initial_guess = _load_initial_boozer_state(config.boozer_state_path)
    boozer_surface = build_native_boozer_surface(
        biotsavart=biotsavart,
        surface=input_surface,
        config=config,
    )
    objective, terms, coil_surface_distance = build_native_single_stage_objective(
        boozer_surface=boozer_surface,
        biotsavart=biotsavart,
        iota_target=config.iota_target,
        weights=config.weights,
    )
    setup_s = perf_counter() - setup_started

    initial_solve_started = perf_counter()
    _solve_boozer(boozer_surface, initial_guess)
    initial_inner_solve_s = perf_counter() - initial_solve_started
    initial_payload_started = perf_counter()
    initial_objective = float(objective.J())
    _require_inactive_coil_surface_distance(coil_surface_distance)
    initial_gradient = np.asarray(objective.dJ(), dtype=np.float64)
    if not np.isfinite(initial_objective) or not np.all(np.isfinite(initial_gradient)):
        raise RuntimeError("Initial native objective or gradient is non-finite")
    initial_state = _accepted_state(
        objective,
        boozer_surface,
        objective_value=initial_objective,
        gradient=initial_gradient,
    )
    initial_payload = _state_payload(
        initial_state,
        terms=terms,
        volume=float(boozer_surface.surface.volume()),
    )
    initial_state_payload_s = perf_counter() - initial_payload_started

    evaluator = NativeSingleStageEvaluator(
        objective=objective,
        boozer_surface=boozer_surface,
        initial_state=initial_state,
        coil_surface_distance=coil_surface_distance,
    )
    outer_started = perf_counter()
    optimizer_result = minimize(
        evaluator.value_and_gradient,
        initial_state.optimizer_dofs,
        jac=True,
        method="L-BFGS-B",
        callback=evaluator.accept,
        options={
            "maxiter": config.outer_maxiter,
            "maxcor": config.outer_maxcor,
            "maxls": config.outer_maxls,
            "ftol": config.outer_ftol,
            "gtol": config.outer_gtol,
        },
    )
    outer_optimization_s = perf_counter() - outer_started
    final_state, optimization_stats = evaluator.finalize(optimizer_result.x)
    final_payload = _state_payload(
        final_state,
        terms=terms,
        volume=float(boozer_surface.surface.volume()),
    )

    artifact_started = perf_counter()
    _save_atomic(biotsavart, paths.biotsavart)
    _save_atomic(boozer_surface.surface, paths.surface)
    _save_atomic(boozer_surface, paths.boozersurface)
    artifact_write_s = perf_counter() - artifact_started
    dof_names = [str(name) for name in objective.dof_names]
    payload: dict[str, object] = {
        "schema_version": 1,
        "driver": "single_stage_banana_native",
        "backend": "native-simsopt-cpu",
        "precision": "float64",
        "constraint_method": "soft-penalty",
        "mixed_precision": False,
        "comparison_schema_version": 1,
        "objective_contract": {
            "id": COMMON_OBJECTIVE_CONTRACT_ID,
            "ordered_terms": list(COMMON_OBJECTIVE_TERM_NAMES),
            "weights": _common_weights(config.weights),
            "optimizer_method": "L-BFGS-B",
            "constraint_method": "soft-penalty",
            "dtype": "float64",
            "mixed_precision": False,
            "adjoint_acceptance_policy": "native-plu-finite-gradient",
            "inactive_term_requirements": {"coil_surface_distance": 0.0},
            "dof_names": dof_names,
            "dof_count": len(dof_names),
            "dof_names_sha256": _string_sequence_sha256(dof_names),
        },
        "config": _config_payload(config),
        "inputs": {
            "biotsavart": _input_payload(config.biotsavart_file),
            "surface": _input_payload(config.surface_path),
            "boozer_state": _input_payload(config.boozer_state_path),
        },
        "input_sha256": {
            "biotsavart": _sha256_file(config.biotsavart_file.resolve()),
            "surface": _sha256_file(config.surface_path.resolve()),
            "boozer_state": _sha256_file(config.boozer_state_path.resolve()),
        },
        "run_config_sha256": config.run_config_sha256,
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "pid": os.getpid(),
        },
        "optimizer": _optimizer_payload(optimizer_result, optimization_stats, config),
        "timings": {
            "setup_s": setup_s,
            "initial_inner_solve_s": initial_inner_solve_s,
            "initial_state_payload_s": initial_state_payload_s,
            "outer_optimization_s": outer_optimization_s,
            "outer_inner_solve_s": optimization_stats.outer_inner_solve_s,
            "outer_objective_s": optimization_stats.outer_objective_s,
            "endpoint_reconciliation_inner_solve_s": (
                optimization_stats.endpoint_reconciliation_inner_solve_s
            ),
            "endpoint_reconciliation_objective_s": (
                optimization_stats.endpoint_reconciliation_objective_s
            ),
            "final_inner_solve_s": optimization_stats.final_inner_solve_s,
            "final_objective_s": optimization_stats.final_objective_s,
            "artifact_write_s": artifact_write_s,
            "measured_before_results_write_s": perf_counter() - run_started,
        },
        "initial_state": initial_payload,
        "final_state": final_payload,
        "outputs": {
            "biotsavart": {
                "path": str(paths.biotsavart.resolve()),
                "sha256": _sha256_file(paths.biotsavart),
            },
            "surface": {
                "path": str(paths.surface.resolve()),
                "sha256": _sha256_file(paths.surface),
            },
            "boozersurface": {
                "path": str(paths.boozersurface.resolve()),
                "sha256": _sha256_file(paths.boozersurface),
            },
            "results": str(paths.results.resolve()),
        },
    }
    _write_json_atomic(paths.results, payload)
    return payload


def common_only_weights(
    *,
    nonqs: float,
    bres: float,
    iota: float,
    length: float,
    ccdist: float,
    csdist: float,
    curvature: float,
) -> SingleStageWeights:
    """Return the shared seven-term weights with every custom term disabled."""
    return replace(
        SingleStageWeights(),
        nonqs=float(nonqs),
        bres=float(bres),
        iota=float(iota),
        length=float(length),
        ccdist=float(ccdist),
        csdist=float(csdist),
        curvature=float(curvature),
        poloidal=0.0,
        width=0.0,
        global_curvature_radius=0.0,
        currents=0.0,
    )
