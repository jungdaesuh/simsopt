"""Serial 9-term NCSX nested-LS outer used by the compare harness.

This is not F3/flat-675, not ``examples/jax``, and not a one-to-one
mirror of ``examples/2_Intermediate/boozerQA_ls_mpi.py``. The 9-term
weights and last-surface major-radius identity follow that example's
serial ``MPIObjective`` mean. Inner iteration budgets, Newton
continuation caps, and the B3 restore-reject barrier are caller-owned
harness policy. The shipped example's BoozerLS defaults remain
BFGS 1500 then Newton 40, and its failed-trial contract is
``(J_prev, -dJ_prev)``. Schur Newton is a unit-test inner only.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from weakref import WeakKeyDictionary

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from simsopt.field import BiotSavart
from simsopt.geo import (
    ArclengthVariation,
    BoozerResidual,
    CurveCurveDistance,
    CurveLength,
    Iotas,
    LpCurveCurvature,
    MajorRadius,
    MeanSquaredCurvature,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.objectives import MPIObjective, QuadraticPenalty

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.curve_objectives import (
    ArclengthVariationJAX,
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    LpCurveCurvatureJAX,
    MeanSquaredCurvatureJAX,
)
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_JAX_INNER_STAB,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
    NESTED_LS_WEIGHT_INV_MODB,
)
from simsopt_jax_adapters.geo.flat675.y_solve import solve_flat675_y_qr
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NestedLsSchurNewtonResult,
    NestedLsYSolution,
    nested_ls_runtime_coil_closures,
    pack_surface_and_y,
    require_full_y_rank,
    run_reduced_nested_ls_schur_newton,
    solve_projected_y,
)
from simsopt_jax.geo.optimizers.optimizer import host_jax_minimize_value_and_grad
from simsopt_jax_adapters.geo.surface_objectives import (
    BoozerResidualJAX,
    IotasJAX,
    MajorRadiusJAX,
    NonQuasiSymmetricRatioJAX,
    compute_standard_surface_objective_gradients,
)

NCSX_EXAMPLE_JSON: Final[Path] = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "2_Intermediate"
    / "inputs"
    / "input_ncsx"
    / "ncsx_init.json"
)

NCSX_MIN_DIST_THRESHOLD: Final[float] = 0.15
NCSX_KAPPA_THRESHOLD: Final[float] = 15.0
NCSX_MSC_THRESHOLD: Final[float] = 15.0
NCSX_IOTAS_TARGET: Final[float] = -0.4
NCSX_RES_WEIGHT: Final[float] = 1.0e4
NCSX_LENGTH_WEIGHT: Final[float] = 1.0
NCSX_MR_WEIGHT: Final[float] = 1.0
NCSX_MIN_DIST_WEIGHT: Final[float] = 1.0e2
NCSX_KAPPA_WEIGHT: Final[float] = 1.0
NCSX_MSC_WEIGHT: Final[float] = 1.0
NCSX_IOTAS_WEIGHT: Final[float] = 1.0
NCSX_ARCLENGTH_WEIGHT: Final[float] = 1.0e-2
NCSX_EVAL_TIMING_KEYS: Final[tuple[str, ...]] = (
    "inner",
    "self_intersection",
    "y_surface_jacobian",
    "y_coil_jacobian",
    "direct_partials",
    "implicit_adjoint",
    "coil_terms",
    "total",
)
# Compare-harness land cap. Not the shipped BoozerLS default (1500).
NCSX_NATIVE_BFGS_MAXITER: Final[int] = 20
# Compare-harness continuation cap. Not the shipped Newton default (40).
# Feasible 18² evals finish in 4–5 steps; infeasible coil trials otherwise
# grind the lander's 40-step budget at GPU-negative cost.
NCSX_OUTER_NEWTON_MAXITER: Final[int] = 8
_NCSX_RUNTIME_KERNELS: WeakKeyDictionary = WeakKeyDictionary()


def empty_ncsx_eval_timing() -> dict[str, float]:
    """Zero wall-clock seconds for every published outer-eval bucket."""

    return {key: 0.0 for key in NCSX_EVAL_TIMING_KEYS}


def _accumulate_timing(timing: dict[str, float], key: str, started: float) -> None:
    timing[key] = timing.get(key, 0.0) + (time.perf_counter() - started)


def summarize_ncsx_eval_timings(
    records: Sequence[dict[str, float]],
) -> dict[str, object]:
    """Sum and mean wall-clock seconds across feasible outer evaluations."""

    packed = tuple(records)
    if not packed:
        return {
            "n": 0,
            "keys": list(NCSX_EVAL_TIMING_KEYS),
            "sum": empty_ncsx_eval_timing(),
            "mean": empty_ncsx_eval_timing(),
        }
    sums = empty_ncsx_eval_timing()
    for row in packed:
        for key in NCSX_EVAL_TIMING_KEYS:
            sums[key] += float(row.get(key, 0.0))
    count = float(len(packed))
    return {
        "n": len(packed),
        "keys": list(NCSX_EVAL_TIMING_KEYS),
        "sum": sums,
        "mean": {key: sums[key] / count for key in NCSX_EVAL_TIMING_KEYS},
    }


def _live_ncsx_surface(state: "NcsxNestedLsSurfaceState"):
    if state.jax_boozer is not None:
        return state.jax_boozer.surface
    if state.native is not None:
        return state.native.surface
    raise ValueError("NCSX surface state has neither a JAX nor a native Boozer.")


class NcsxNestedLsInnerSolveFailed(RuntimeError):
    """Inner Schur Newton did not land on the nested branch."""

    def __init__(self, *, iteration_count: int, grad_l2: float, exit_status: str):
        super().__init__(
            "NCSX nested-LS inner failed: "
            f"exit={exit_status!r} iter={int(iteration_count)} "
            f"||g||_2={float(grad_l2)!r}."
        )
        self.iteration_count = int(iteration_count)
        self.grad_l2 = float(grad_l2)
        self.exit_status = str(exit_status)


class NcsxNestedLsBranchJump(RuntimeError):
    """Inner solve converged onto a different iota sheet than the anchor."""

    def __init__(self, *, iota: float, anchor_iota: float, guard: float):
        super().__init__(
            "NCSX nested-LS iota jumped "
            f"{abs(float(iota) - float(anchor_iota))} > {float(guard)}."
        )
        self.iota = float(iota)
        self.anchor_iota = float(anchor_iota)
        self.guard = float(guard)


class NcsxNestedLsSelfIntersecting(RuntimeError):
    """A trial surface is self-intersecting at the default cylindrical cut.

    Native ``boozerQA_ls_mpi.py`` rejects that geometry. This signal takes
    the B3 containment barrier, not the example's flipped-gradient sentinel.
    """

    def __init__(self, *, surface_index: int):
        super().__init__(
            "NCSX nested-LS trial surface "
            f"{int(surface_index)} is self-intersecting."
        )
        self.surface_index = int(surface_index)


def remap_tensor_fourier_index(index: int, m_old: int, m_new: int) -> int:
    """Map a stellsym-aware TensorFourier cosine/sine slot onto a padded grid."""

    if index <= m_old:
        return int(index)
    return int(m_new + (index - m_old))


def upsample_surface_xyz_tensor_fourier(
    old: SurfaceXYZTensorFourier,
    *,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> SurfaceXYZTensorFourier:
    """Pad TensorFourier modes without changing the realized geometry."""

    phi = np.linspace(0.0, 1.0 / old.nfp, nphi, endpoint=False)
    theta = np.linspace(0.0, 1.0, ntheta, endpoint=False)
    new = SurfaceXYZTensorFourier(
        nfp=old.nfp,
        stellsym=old.stellsym,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=phi,
        quadpoints_theta=theta,
        clamped_dims=list(old.clamped_dims),
    )
    for arr in (new.xcs, new.ycs, new.zcs):
        arr[:, :] = 0.0
    for i in range(2 * old.mpol + 1):
        for j in range(2 * old.ntor + 1):
            ii = remap_tensor_fourier_index(i, old.mpol, mpol)
            jj = remap_tensor_fourier_index(j, old.ntor, ntor)
            new.xcs[ii, jj] = old.xcs[i, j]
            new.ycs[ii, jj] = old.ycs[i, j]
            new.zcs[ii, jj] = old.zcs[i, j]
    new.local_full_x = new.get_dofs()
    new.invalidate_cache()
    return new


def clone_surface_xyz_tensor_fourier(
    surface: SurfaceXYZTensorFourier,
) -> SurfaceXYZTensorFourier:
    """Independent TensorFourier copy with the same quadrature and modes."""

    cloned = SurfaceXYZTensorFourier(
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        mpol=surface.mpol,
        ntor=surface.ntor,
        quadpoints_phi=np.asarray(surface.quadpoints_phi, dtype=np.float64).copy(),
        quadpoints_theta=np.asarray(surface.quadpoints_theta, dtype=np.float64).copy(),
        clamped_dims=list(surface.clamped_dims),
    )
    cloned.set_dofs(np.asarray(surface.get_dofs(), dtype=np.float64).copy())
    return cloned


def _host_float64(values) -> NDArray[np.float64]:
    return np.asarray(jax.device_get(values), dtype=np.float64).reshape(-1)


def _identity_quadratic(value: float, target: float) -> tuple[float, float]:
    diff = float(value) - float(target)
    return 0.5 * diff * diff, diff


def _ncsx_runtime_kernels(jax_boozer: BoozerSurfaceJAX):
    """Compile-once residual/HVP/y-chain with coil as a kernel argument."""

    cached = _NCSX_RUNTIME_KERNELS.get(jax_boozer)
    if cached is not None:
        return cached
    residual_rt, objective_rt, _phi = nested_ls_runtime_coil_closures(jax_boozer)
    del _phi

    @jax.jit
    def packed_hvp(packed, tangent, coil):
        def phi(decision: jax.Array) -> jax.Array:
            return objective_rt(decision, coil)

        return jax.jvp(jax.grad(phi), (packed,), (tangent,))[1]

    @jax.jit
    def envelope(surface, coil, y_probe):
        surface = jnp.asarray(surface, dtype=jnp.float64).reshape(-1)
        coil = jnp.asarray(coil, dtype=jnp.float64).reshape(-1)
        probe = jnp.asarray(y_probe, dtype=jnp.float64).reshape(-1)

        def residual_of_y(y: jax.Array) -> jax.Array:
            return residual_rt(pack_surface_and_y(surface, y), coil)

        residual = residual_of_y(probe)
        design = jax.jacfwd(residual_of_y)(probe)
        rhs = design @ probe - residual
        y_raw = solve_flat675_y_qr(design, rhs)
        packed = pack_surface_and_y(surface, y_raw.solution)
        value, full_grad = jax.value_and_grad(
            lambda decision: objective_rt(decision, coil)
        )(packed)
        return (
            value,
            full_grad[:-2],
            y_raw.solution,
            y_raw.singular_values,
            y_raw.numerical_rank,
            y_raw.numerics_finite,
            design,
            rhs,
        )

    @jax.jit
    def y_partial_vjps(surface, coil, y_star, lam):
        """``(∂y/∂s)ᵀ λ`` and ``(∂y/∂c)ᵀ λ`` by VJP through the QR ``y*``."""

        surface = jnp.asarray(surface, dtype=jnp.float64).reshape(-1)
        coil = jnp.asarray(coil, dtype=jnp.float64).reshape(-1)
        probe = jnp.asarray(y_star, dtype=jnp.float64).reshape(-1)
        lam = jnp.asarray(lam, dtype=jnp.float64).reshape(-1)

        def y_of_surface(surface_dofs: jax.Array) -> jax.Array:
            return solve_projected_y(
                lambda packed: residual_rt(packed, coil),
                surface_dofs,
                probe,
            ).solution

        def y_of_coil(coil_vector: jax.Array) -> jax.Array:
            return solve_projected_y(
                lambda packed: residual_rt(packed, coil_vector),
                surface,
                probe,
            ).solution

        surface_bar = jax.vjp(y_of_surface, surface)[1](lam)[0]
        coil_bar = jax.vjp(y_of_coil, coil)[1](lam)[0]
        return surface_bar, coil_bar

    kernels = {
        "residual_rt": residual_rt,
        "objective_rt": objective_rt,
        "packed_hvp": packed_hvp,
        "envelope": envelope,
        "y_partial_vjps": y_partial_vjps,
    }
    _NCSX_RUNTIME_KERNELS[jax_boozer] = kernels
    return kernels


def _ncsx_bound_inner_functions(jax_boozer: BoozerSurfaceJAX, coil: jax.Array):
    kernels = _ncsx_runtime_kernels(jax_boozer)
    y_probe = jnp.zeros((2,), dtype=jnp.float64)

    def residual_fn(packed: jax.Array) -> jax.Array:
        return kernels["residual_rt"](packed, coil)

    def objective_fn(packed: jax.Array) -> jax.Array:
        return kernels["objective_rt"](packed, coil)

    def packed_hvp(packed: jax.Array, tangent: jax.Array) -> jax.Array:
        return kernels["packed_hvp"](packed, tangent, coil)

    def envelope_value_and_grad(_residual_fn, _objective_fn, surface_dofs):
        del _residual_fn, _objective_fn
        (
            value,
            surface_grad,
            y_sol,
            singular_values,
            numerical_rank,
            numerics_finite,
            design,
            rhs,
        ) = kernels["envelope"](
            jnp.asarray(surface_dofs, dtype=jnp.float64).reshape(-1),
            coil,
            y_probe,
        )
        solution = NestedLsYSolution(
            solution=y_sol,
            singular_values=singular_values,
            numerical_rank=numerical_rank,
            numerics_finite=numerics_finite,
            design_matrix=design,
            right_hand_side=rhs,
        )
        require_full_y_rank(solution)
        return (
            float(np.asarray(jax.device_get(value))),
            _host_float64(surface_grad),
            solution,
        )

    return residual_fn, objective_fn, packed_hvp, envelope_value_and_grad, kernels


def run_ncsx_schur_inner(
    jax_boozer: BoozerSurfaceJAX,
    *,
    iota: float,
    G: float,
    constraint_weight: float | None = None,
    maxiter: int = NESTED_LS_BANANA_NEWTON_MAXITER,
    tol: float = NESTED_LS_BANANA_NEWTON_TOL,
    stab: float = NESTED_LS_JAX_INNER_STAB,
) -> NestedLsSchurNewtonResult:
    """Reduced Schur Newton on one NCSX Boozer surface (dense LU).

    Defaults follow the banana ``run_code`` bar (``newton_tol=1e-11``,
    ``maxiter=40``, ``stab=0``), not the 675 reconstruct ``1e-13`` judge.
    Coil DOFs are kernel arguments so L-BFGS outer evals reuse XLA.
    """

    weight = (
        float(jax_boozer.constraint_weight)
        if constraint_weight is None
        else float(constraint_weight)
    )
    coil = jnp.asarray(jax_boozer.biotsavart.x, dtype=jnp.float64).reshape(-1)
    residual_fn, objective_fn, packed_hvp, envelope, _kernels = (
        _ncsx_bound_inner_functions(jax_boozer, coil)
    )
    return run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(iota),
        G=float(G),
        constraint_weight=weight,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
        stab=float(stab),
        tol=float(tol),
        maxiter=int(maxiter),
        linear_solver="dense_lu",
        max_dense_linearization_bytes=None,
        residual_fn=residual_fn,
        objective_fn=objective_fn,
        packed_hvp=packed_hvp,
        envelope_value_and_grad=envelope,
    )


def _projected_y_coil_vjp(residual_rt, surface, coil_dofs, y_star, lam):
    """One-shot ``(∂y/∂c)ᵀ λ`` through the QR ``y*``; production uses the jit."""

    surface_jax = jnp.asarray(surface, dtype=jnp.float64).reshape(-1)
    coil = jnp.asarray(coil_dofs, dtype=jnp.float64).reshape(-1)
    probe = jnp.asarray(y_star, dtype=jnp.float64).reshape(-1)
    lam = jnp.asarray(lam, dtype=jnp.float64).reshape(-1)

    def y_of_coil(coil_vector: jax.Array) -> jax.Array:
        return solve_projected_y(
            lambda packed: residual_rt(packed, coil_vector),
            surface_jax,
            probe,
        ).solution

    return jax.vjp(y_of_coil, coil)[1](lam)[0]


def _coil_term_value_and_grad(
    coil_terms, biotsavart
) -> tuple[float, NDArray[np.float64]]:
    value = float(coil_terms.J())
    gradient = np.asarray(coil_terms.dJ(partials=True)(biotsavart), dtype=np.float64)
    return value, gradient


def _sum_objectives(terms):
    items = list(terms)
    if not items:
        raise ValueError("coil-term list must be nonempty.")
    total = items[0]
    for term in items[1:]:
        total = total + term
    return total


@dataclass
class NcsxNestedLsSurfaceState:
    """One NCSX surface plus the JAX or native objectives that read it.

    Live ``iota`` / ``G`` are the last inner guess. The committed warm
    start is ``anchor_surface_dofs`` with ``anchor_iota`` and
    ``anchor_G``. Only :func:`commit_ncsx_anchor` advances the anchor.
    JAX and native outers both use banana ``run_code``. A state must
    have at least one Boozer.
    """

    jax_boozer: BoozerSurfaceJAX | None
    native: BoozerSurface | None
    iota: float
    G: float
    anchor_iota: float
    anchor_G: float
    anchor_surface_dofs: NDArray[np.float64]
    nonqs: NonQuasiSymmetricRatioJAX | None
    residual: BoozerResidualJAX | None
    major_radius: MajorRadiusJAX | None
    iotas_term: IotasJAX | None
    radius_target: float
    radius_scale: float


@dataclass
class NcsxNestedLsProblem:
    """Coil-outer NCSX 9-term problem with JAX Schur or native banana inners."""

    surfaces: tuple[NcsxNestedLsSurfaceState, ...]
    coil_terms: object
    iota_target: float
    length_target: float
    biotsavart: BiotSavartJAX | None
    last_inner: NestedLsSchurNewtonResult | None = field(default=None, repr=False)
    last_native_inner: dict[str, object] | None = field(default=None, repr=False)
    last_run_code: dict[str, object] | None = field(default=None, repr=False)
    last_eval_timing: dict[str, float] = field(default_factory=dict, repr=False)
    native_objective: object | None = field(default=None, repr=False)
    native_biotsavart: BiotSavart | None = field(default=None, repr=False)


def ncsx_banana_run_code(
    jax_boozer: BoozerSurfaceJAX,
    iota,
    G=None,
    *,
    sdofs=None,
    polish_only: bool = False,
    newton_maxiter_cap: int | None = None,
) -> dict[str, object]:
    """Banana inner with coil geometry as kernel arguments.

    Public ondevice ``BoozerSurfaceJAX.run_code`` closure-converts the
    penalty objective and hashes coil bytes into the BFGS compile key.
    Land uses host BFGS then dense-LU Newton. Outer evals may pass
    ``polish_only=True`` and an explicit ``newton_maxiter_cap``; those
    are harness knobs, not BoozerLS defaults. The Newton ``res`` is
    persisted for solved-state IFT.
    """

    if not jax_boozer.need_to_run_code:
        stored = jax_boozer.res
        if stored is None:
            raise RuntimeError(
                "ncsx_banana_run_code found need_to_run_code=False with no stored res."
            )
        return stored
    if sdofs is not None:
        jax_boozer._set_surface_dofs(sdofs)
    jax_boozer._refresh_coil_data()
    optimize_G = G is not None
    weight_inv_modB = jax_boozer.options["weight_inv_modB"]
    iota_out = iota
    g_out = G
    if not polish_only:
        value_and_grad = jax_boozer._make_penalty_value_and_grad_host_jax_with(
            optimize_G,
            weight_inv_modB,
            jax_boozer.constraint_weight,
        )
        x0 = jax_boozer._pack_decision_vector(iota, G)
        ls_result = host_jax_minimize_value_and_grad(
            value_and_grad,
            x0,
            method="bfgs",
            tol=jax_boozer.options["bfgs_tol"],
            maxiter=int(jax_boozer.options["bfgs_maxiter"]),
            value_and_grad=True,
        )
        accepted_x = getattr(ls_result, "x_device", ls_result.x)
        sdofs_out, iota_out, g_out = jax_boozer._unpack_penalty_optimizer_state(
            accepted_x, optimize_G
        )
        jax_boozer._set_surface_dofs(sdofs_out)
    jax_boozer.need_to_run_code = True
    newton_maxiter = int(jax_boozer.options["newton_maxiter"])
    if newton_maxiter_cap is not None:
        newton_maxiter = min(newton_maxiter, int(newton_maxiter_cap))
    return jax_boozer.minimize_boozer_penalty_constraints_newton(
        constraint_weight=jax_boozer.constraint_weight,
        iota=iota_out,
        G=g_out,
        verbose=jax_boozer.options["verbose"],
        tol=jax_boozer.options["newton_tol"],
        maxiter=newton_maxiter,
        stab=jax_boozer.options["newton_stab"],
        weight_inv_modB=weight_inv_modB,
    )


def ncsx_nested_ls_outer_value_and_grad(
    problem: NcsxNestedLsProblem,
    coil_dofs: object,
) -> tuple[float, NDArray[np.float64]]:
    """Nine-term ``J(c)`` and coil gradient at banana ``s*(c)``.

    Inner is :func:`ncsx_banana_run_code` with ``polish_only=True``
    (dense-LU Newton continuation from the committed banana land,
    coils as kernel arguments). Surface-term gradients use one batched
    solved-state IFT adjoint, not Schur jacrev. Always warm-starts
    from the committed anchor. Failures restore the anchor before
    raising.
    """

    if problem.biotsavart is None or any(
        surface_state.jax_boozer is None for surface_state in problem.surfaces
    ):
        raise ValueError(
            "ncsx_nested_ls_outer_value_and_grad requires JAX Boozer surfaces."
        )
    restore_ncsx_anchor(problem)
    coil = np.asarray(coil_dofs, dtype=np.float64).reshape(-1)
    problem.biotsavart.x = np.array(coil, dtype=np.float64, copy=True)
    timing = empty_ncsx_eval_timing()
    problem.last_eval_timing = timing
    eval_started = time.perf_counter()
    succeeded = False
    try:
        run_codes: list[dict[str, object]] = []
        for surface_state in problem.surfaces:
            jax_boozer = surface_state.jax_boozer
            jax_boozer.biotsavart.x = np.array(coil, dtype=np.float64, copy=True)
            jax_boozer.need_to_run_code = True
            started = time.perf_counter()
            inner = ncsx_banana_run_code(
                jax_boozer,
                surface_state.anchor_iota,
                surface_state.anchor_G,
                polish_only=True,
                newton_maxiter_cap=NCSX_OUTER_NEWTON_MAXITER,
            )
            _accumulate_timing(timing, "inner", started)
            if inner is None:
                raise RuntimeError(
                    "JAX run_code returned None after need_to_run_code=True."
                )
            problem.last_run_code = inner
            if not bool(inner["success"]):
                raise NcsxNestedLsInnerSolveFailed(
                    iteration_count=int(inner.get("iter", -1)),
                    grad_l2=float(np.linalg.norm(inner.get("jacobian", [np.inf]))),
                    exit_status="failed",
                )
            if abs(float(inner["iota"]) - float(surface_state.anchor_iota)) > float(
                NESTED_LS_OUTER_IOTA_BRANCH_GUARD
            ):
                raise NcsxNestedLsBranchJump(
                    iota=float(inner["iota"]),
                    anchor_iota=float(surface_state.anchor_iota),
                    guard=float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
                )
            run_codes.append(inner)

        started = time.perf_counter()
        for index, surface_state in enumerate(problem.surfaces):
            if surface_state.jax_boozer.surface.is_self_intersecting():
                raise NcsxNestedLsSelfIntersecting(surface_index=index)
        _accumulate_timing(timing, "self_intersection", started)

        nsurf = len(run_codes)
        inv_n = 1.0 / float(nsurf)
        total = 0.0
        coil_grad = np.zeros_like(coil)
        iota_values: list[float] = []
        iota_grads: list[NDArray[np.float64]] = []
        started = time.perf_counter()
        for surface_state in problem.surfaces:
            compute_standard_surface_objective_gradients(
                surface_state.residual,
                surface_state.iotas_term,
                surface_state.nonqs,
            )
            qs_j = float(surface_state.nonqs.J())
            qs_dj = np.asarray(
                surface_state.nonqs.dJ_by_dcoil_dofs(), dtype=np.float64
            )
            res_j = float(surface_state.residual.J())
            res_dj = np.asarray(
                surface_state.residual.dJ_by_dcoil_dofs(), dtype=np.float64
            )
            total += inv_n * (qs_j + NCSX_RES_WEIGHT * res_j)
            coil_grad = coil_grad + inv_n * (qs_dj + NCSX_RES_WEIGHT * res_dj)
            if surface_state.radius_scale != 0.0:
                radius_dj = np.asarray(
                    surface_state.major_radius.dJ_by_dcoil_dofs(),
                    dtype=np.float64,
                )
                radius = float(surface_state.major_radius.J())
                radius_j, radius_dval = _identity_quadratic(
                    radius, surface_state.radius_target
                )
                total += surface_state.radius_scale * radius_j
                coil_grad = coil_grad + (
                    surface_state.radius_scale * radius_dval * radius_dj
                )
            iota_values.append(float(surface_state.iotas_term.J()))
            iota_grads.append(
                np.asarray(
                    surface_state.iotas_term.dJ_by_dcoil_dofs(), dtype=np.float64
                )
            )
        mean_iota = float(sum(iota_values) / nsurf)
        iota_j, mean_dval = _identity_quadratic(mean_iota, problem.iota_target)
        total += NCSX_IOTAS_WEIGHT * iota_j
        iota_scale = NCSX_IOTAS_WEIGHT * mean_dval / float(nsurf)
        for iota_dj in iota_grads:
            coil_grad = coil_grad + iota_scale * iota_dj
        _accumulate_timing(timing, "implicit_adjoint", started)

        started = time.perf_counter()
        coil_j, coil_dj = _coil_term_value_and_grad(
            problem.coil_terms, problem.biotsavart
        )
        total += coil_j
        coil_grad = coil_grad + coil_dj
        _accumulate_timing(timing, "coil_terms", started)
        if not np.isfinite(total) or not bool(np.all(np.isfinite(coil_grad))):
            raise RuntimeError(
                f"NCSX nested-LS outer evaluation is not finite: J={total!r}."
            )
        for surface_state, inner in zip(problem.surfaces, run_codes, strict=True):
            surface_state.iota = float(inner["iota"])
            surface_state.G = float(inner["G"])
        succeeded = True
        return total, coil_grad
    finally:
        _accumulate_timing(timing, "total", eval_started)
        if not succeeded:
            restore_ncsx_anchor(problem)


def commit_ncsx_anchor(problem: NcsxNestedLsProblem) -> None:
    """Snapshot the live ``(s, ι, G)`` as the committed warm start."""

    for surface_state in problem.surfaces:
        live = _live_ncsx_surface(surface_state)
        surface_state.anchor_surface_dofs = np.array(
            live.get_dofs(),
            dtype=np.float64,
            copy=True,
        )
        if (
            surface_state.jax_boozer is not None
            and surface_state.native is not None
            and surface_state.native.surface is not live
        ):
            surface_state.native.surface.set_dofs(
                np.array(surface_state.anchor_surface_dofs, dtype=np.float64, copy=True)
            )
        surface_state.anchor_iota = float(surface_state.iota)
        surface_state.anchor_G = float(surface_state.G)


def restore_ncsx_anchor(problem: NcsxNestedLsProblem) -> None:
    """Install the committed ``(s, ι, G)``; discard any unaccepted trial."""

    for surface_state in problem.surfaces:
        dofs = np.array(surface_state.anchor_surface_dofs, dtype=np.float64, copy=True)
        if surface_state.jax_boozer is not None:
            surface_state.jax_boozer.surface.set_dofs(dofs)
        if surface_state.native is not None:
            surface_state.native.surface.set_dofs(
                np.array(dofs, dtype=np.float64, copy=True)
            )
            residual_state = getattr(surface_state.native, "res", None)
            if residual_state is not None:
                residual_state["iota"] = float(surface_state.anchor_iota)
                residual_state["G"] = float(surface_state.anchor_G)
            surface_state.native.need_to_run_code = True
        surface_state.iota = float(surface_state.anchor_iota)
        surface_state.G = float(surface_state.anchor_G)


def ncsx_native_outer_value_and_grad(
    problem: NcsxNestedLsProblem,
    coil_dofs: object,
) -> tuple[float, NDArray[np.float64]]:
    """Nine-term native ``J(c)`` at banana ``s*(c)``, with the B3 restore.

    Always warm-starts from the committed anchor. A successful return
    leaves the trial surface on the native Boozer objects; it does not
    commit the anchor. Failures restore the anchor before raising.
    ``last_eval_timing`` uses the same keys as the JAX twin: native
    ``J()``/``dJ()`` fill ``direct_partials`` (the cached adjoint lives
    inside ``compute()``), and the JAX-only Jacobian buckets stay zero.
    """

    if problem.native_objective is None or problem.native_biotsavart is None:
        raise ValueError(
            "ncsx_native_outer_value_and_grad requires native_objective "
            "and native_biotsavart."
        )
    if any(surface_state.native is None for surface_state in problem.surfaces):
        raise ValueError(
            "ncsx_native_outer_value_and_grad requires a native Boozer "
            "on every surface."
        )
    restore_ncsx_anchor(problem)
    coil = np.asarray(coil_dofs, dtype=np.float64).reshape(-1)
    problem.native_biotsavart.x = np.array(coil, dtype=np.float64, copy=True)
    timing = empty_ncsx_eval_timing()
    problem.last_eval_timing = timing
    eval_started = time.perf_counter()
    succeeded = False
    try:
        native_results: list[dict[str, object]] = []
        for surface_state in problem.surfaces:
            native = surface_state.native
            native.need_to_run_code = True
            started = time.perf_counter()
            inner = native.run_code(
                surface_state.anchor_iota, surface_state.anchor_G
            )
            _accumulate_timing(timing, "inner", started)
            if inner is None:
                raise RuntimeError(
                    "native run_code returned None after need_to_run_code=True."
                )
            problem.last_native_inner = inner
            if not bool(inner["success"]):
                raise NcsxNestedLsInnerSolveFailed(
                    iteration_count=int(inner.get("iter", -1)),
                    grad_l2=float(np.linalg.norm(inner["jacobian"])),
                    exit_status="failed",
                )
            if abs(float(inner["iota"]) - float(surface_state.anchor_iota)) > float(
                NESTED_LS_OUTER_IOTA_BRANCH_GUARD
            ):
                raise NcsxNestedLsBranchJump(
                    iota=float(inner["iota"]),
                    anchor_iota=float(surface_state.anchor_iota),
                    guard=float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
                )
            native_results.append(inner)

        started = time.perf_counter()
        for index, surface_state in enumerate(problem.surfaces):
            if surface_state.native.surface.is_self_intersecting():
                raise NcsxNestedLsSelfIntersecting(surface_index=index)
        _accumulate_timing(timing, "self_intersection", started)

        started = time.perf_counter()
        total = float(problem.native_objective.J())
        coil_grad = np.asarray(problem.native_objective.dJ(), dtype=np.float64)
        _accumulate_timing(timing, "direct_partials", started)
        if coil_grad.shape != coil.shape:
            raise RuntimeError(
                "native NCSX outer gradient shape "
                f"{coil_grad.shape} does not match coil DOFs {coil.shape}."
            )
        if not np.isfinite(total) or not bool(np.all(np.isfinite(coil_grad))):
            raise RuntimeError(
                f"NCSX native outer evaluation is not finite: J={total!r}."
            )
        for surface_state, inner in zip(
            problem.surfaces, native_results, strict=True
        ):
            surface_state.iota = float(inner["iota"])
            surface_state.G = float(inner["G"])
        succeeded = True
        return total, coil_grad
    finally:
        _accumulate_timing(timing, "total", eval_started)
        if not succeeded:
            restore_ncsx_anchor(problem)


def _build_coil_terms(base_curves, curves) -> tuple[object, float]:
    lengths = [CurveLengthJAX(curve) for curve in base_curves]
    length_sum = _sum_objectives(lengths)
    length_target = float(length_sum.J())
    length_penalty = NCSX_LENGTH_WEIGHT * QuadraticPenalty(
        length_sum, length_target, "max"
    )
    distance = NCSX_MIN_DIST_WEIGHT * CurveCurveDistanceJAX(
        list(curves),
        NCSX_MIN_DIST_THRESHOLD,
        num_basecurves=len(base_curves),
    )
    curvature = NCSX_KAPPA_WEIGHT * _sum_objectives(
        LpCurveCurvatureJAX(curve, 2, NCSX_KAPPA_THRESHOLD) for curve in base_curves
    )
    msc_term = NCSX_MSC_WEIGHT * _sum_objectives(
        QuadraticPenalty(MeanSquaredCurvatureJAX(curve), NCSX_MSC_THRESHOLD, "max")
        for curve in base_curves
    )
    arclength = NCSX_ARCLENGTH_WEIGHT * _sum_objectives(
        ArclengthVariationJAX(curve) for curve in base_curves
    )
    return (
        length_penalty + distance + curvature + msc_term + arclength,
        length_target,
    )


def _build_native_coil_terms(base_curves, curves) -> tuple[object, float]:
    lengths = [CurveLength(curve) for curve in base_curves]
    length_sum = _sum_objectives(lengths)
    length_target = float(length_sum.J())
    length_penalty = NCSX_LENGTH_WEIGHT * QuadraticPenalty(
        length_sum, length_target, "max"
    )
    distance = NCSX_MIN_DIST_WEIGHT * CurveCurveDistance(
        list(curves),
        NCSX_MIN_DIST_THRESHOLD,
        num_basecurves=len(base_curves),
    )
    curvature = NCSX_KAPPA_WEIGHT * _sum_objectives(
        LpCurveCurvature(curve, 2, NCSX_KAPPA_THRESHOLD) for curve in base_curves
    )
    msc_term = NCSX_MSC_WEIGHT * _sum_objectives(
        QuadraticPenalty(MeanSquaredCurvature(curve), NCSX_MSC_THRESHOLD, "max")
        for curve in base_curves
    )
    arclength = NCSX_ARCLENGTH_WEIGHT * _sum_objectives(
        ArclengthVariation(curve) for curve in base_curves
    )
    return (
        length_penalty + distance + curvature + msc_term + arclength,
        length_target,
    )


def _assemble_native_nine_term(
    natives: Sequence[BoozerSurface],
    *,
    base_curves,
    curves,
) -> tuple[object, float, object]:
    """Serial MPIObjective mean of the example 9-term native outer."""

    packed = tuple(natives)
    nsurf = len(packed)
    if nsurf == 0:
        raise ValueError("native 9-term objective requires at least one surface.")
    coils = packed[0].biotsavart.coils
    radii = [MajorRadius(boozer) for boozer in packed]
    radius_penalties = [
        (
            nsurf
            * QuadraticPenalty(
                radius, float(radius.boozer_surface.surface.major_radius()), "identity"
            )
            if index == nsurf - 1
            else 0
            * QuadraticPenalty(
                radius, float(radius.boozer_surface.surface.major_radius()), "identity"
            )
        )
        for index, radius in enumerate(radii)
    ]
    iotas = [Iotas(boozer) for boozer in packed]
    non_qs = [NonQuasiSymmetricRatio(boozer, BiotSavart(coils)) for boozer in packed]
    residuals = [BoozerResidual(boozer, BiotSavart(coils)) for boozer in packed]
    mean_iota = MPIObjective(iotas, None, needs_splitting=True)
    coil_terms, length_target = _build_native_coil_terms(base_curves, curves)
    objective = (
        MPIObjective(non_qs, None, needs_splitting=True)
        + NCSX_RES_WEIGHT * MPIObjective(residuals, None, needs_splitting=True)
        + NCSX_IOTAS_WEIGHT
        * QuadraticPenalty(mean_iota, NCSX_IOTAS_TARGET, "identity")
        + NCSX_MR_WEIGHT * MPIObjective(radius_penalties, None, needs_splitting=True)
        + coil_terms
    )
    return objective, length_target, coil_terms


def _ls_inner_options(
    *,
    optimizer_backend: str | None = None,
    bfgs_maxiter: int | None = None,
    newton_maxiter: int | None = None,
) -> dict[str, object]:
    """LS inner options. Omitted maxiters keep BoozerLS constructor defaults."""

    options: dict[str, object] = {
        "verbose": False,
        "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
        "bfgs_tol": 1e-10,
        "newton_tol": NESTED_LS_BANANA_NEWTON_TOL,
    }
    if optimizer_backend is not None:
        options["optimizer_backend"] = optimizer_backend
    if bfgs_maxiter is not None:
        options["bfgs_maxiter"] = int(bfgs_maxiter)
    if newton_maxiter is not None:
        options["newton_maxiter"] = int(newton_maxiter)
    return options


def _make_jax_boozer(
    coils,
    surface: SurfaceXYZTensorFourier,
    *,
    constraint_weight: float,
    optimizer_backend: str,
    bfgs_maxiter: int | None = None,
    newton_maxiter: int | None = None,
) -> BoozerSurfaceJAX:
    label = Volume(surface)
    return BoozerSurfaceJAX(
        BiotSavartJAX(coils),
        surface,
        label,
        float(label.J()),
        constraint_weight=float(constraint_weight),
        options=_ls_inner_options(
            optimizer_backend=optimizer_backend,
            bfgs_maxiter=bfgs_maxiter,
            newton_maxiter=newton_maxiter,
        ),
    )


def _make_native_boozer(
    coils,
    surface: SurfaceXYZTensorFourier,
    *,
    constraint_weight: float,
    bfgs_maxiter: int | None = None,
    newton_maxiter: int | None = None,
) -> BoozerSurface:
    label = Volume(surface)
    return BoozerSurface(
        BiotSavart(coils),
        surface,
        label,
        float(label.J()),
        constraint_weight=float(constraint_weight),
        options=_ls_inner_options(
            bfgs_maxiter=bfgs_maxiter,
            newton_maxiter=newton_maxiter,
        ),
    )


def ncsx_problem_from_jax_boozers(
    jax_boozers: Sequence[BoozerSurfaceJAX],
    *,
    iotas: Sequence[float],
    g_values: Sequence[float],
    base_curves,
    curves,
    natives: Sequence[BoozerSurface | None] | None = None,
) -> NcsxNestedLsProblem:
    """Attach the 9-term outer to already-constructed JAX Boozers."""

    packed = tuple(jax_boozers)
    nsurf = len(packed)
    if nsurf == 0:
        raise ValueError("ncsx_problem_from_jax_boozers requires at least one surface.")
    native_list = (
        tuple(natives)
        if natives is not None
        else tuple(None for _ in packed)
    )
    if len(native_list) != nsurf:
        raise ValueError("natives must match jax_boozers length.")
    surface_states = []
    for index, (jax_boozer, iota, g_value, native) in enumerate(
        zip(packed, iotas, g_values, native_list, strict=True)
    ):
        jax_boozer._refresh_coil_data()
        major_radius = MajorRadiusJAX(jax_boozer)
        radius_target = float(
            major_radius._compute_value(jax_boozer.surface.get_dofs())
        )
        surface_states.append(
            NcsxNestedLsSurfaceState(
                jax_boozer=jax_boozer,
                native=native,
                iota=float(iota),
                G=float(g_value),
                anchor_iota=float(iota),
                anchor_G=float(g_value),
                anchor_surface_dofs=np.array(
                    jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True
                ),
                nonqs=NonQuasiSymmetricRatioJAX(jax_boozer, jax_boozer.biotsavart),
                residual=BoozerResidualJAX(jax_boozer, jax_boozer.biotsavart),
                major_radius=major_radius,
                iotas_term=IotasJAX(jax_boozer),
                radius_target=radius_target,
                radius_scale=(NCSX_MR_WEIGHT if index == nsurf - 1 else 0.0),
            )
        )
    coil_terms, length_target = _build_coil_terms(list(base_curves), list(curves))
    native_objective = None
    native_biotsavart = None
    if native_list and all(native is not None for native in native_list):
        native_objective, _native_length, _native_coil = _assemble_native_nine_term(
            native_list,
            base_curves=base_curves,
            curves=curves,
        )
        del _native_length, _native_coil
        native_biotsavart = native_list[0].biotsavart
    problem = NcsxNestedLsProblem(
        surfaces=tuple(surface_states),
        coil_terms=coil_terms,
        iota_target=NCSX_IOTAS_TARGET,
        length_target=length_target,
        biotsavart=packed[0].biotsavart,
        native_objective=native_objective,
        native_biotsavart=native_biotsavart,
    )
    commit_ncsx_anchor(problem)
    return problem


def ncsx_problem_from_native_boozers(
    natives: Sequence[BoozerSurface],
    *,
    iotas: Sequence[float],
    g_values: Sequence[float],
    base_curves,
    curves,
) -> NcsxNestedLsProblem:
    """Attach the 9-term native outer to already-constructed banana Boozers."""

    packed = tuple(natives)
    nsurf = len(packed)
    if nsurf == 0:
        raise ValueError(
            "ncsx_problem_from_native_boozers requires at least one surface."
        )
    surface_states = []
    for index, (native, iota, g_value) in enumerate(
        zip(packed, iotas, g_values, strict=True)
    ):
        radius_target = float(native.surface.major_radius())
        surface_states.append(
            NcsxNestedLsSurfaceState(
                jax_boozer=None,
                native=native,
                iota=float(iota),
                G=float(g_value),
                anchor_iota=float(iota),
                anchor_G=float(g_value),
                anchor_surface_dofs=np.array(
                    native.surface.get_dofs(), dtype=np.float64, copy=True
                ),
                nonqs=None,
                residual=None,
                major_radius=None,
                iotas_term=None,
                radius_target=radius_target,
                radius_scale=(NCSX_MR_WEIGHT if index == nsurf - 1 else 0.0),
            )
        )
    native_objective, length_target, coil_terms = _assemble_native_nine_term(
        packed,
        base_curves=base_curves,
        curves=curves,
    )
    problem = NcsxNestedLsProblem(
        surfaces=tuple(surface_states),
        coil_terms=coil_terms,
        iota_target=NCSX_IOTAS_TARGET,
        length_target=length_target,
        biotsavart=None,
        native_objective=native_objective,
        native_biotsavart=packed[0].biotsavart,
    )
    commit_ncsx_anchor(problem)
    return problem


def prepare_ncsx_nested_ls_problem(
    *,
    coils,
    base_curves,
    curves,
    surfaces: Sequence[SurfaceXYZTensorFourier],
    iotas: Sequence[float],
    g_values: Sequence[float],
    constraint_weight: float,
    include_native: bool = False,
) -> NcsxNestedLsProblem:
    """Build the 9-term NCSX outer for the compare harness.

    ``surfaces`` are already at the intended Fourier/quadrature scale.
    The last surface carries the major-radius equality. Inner iteration
    budgets are the harness caps, not the shipped BoozerLS defaults.
    """

    packed_surfaces = tuple(surfaces)
    if len(packed_surfaces) == 0:
        raise ValueError(
            "prepare_ncsx_nested_ls_problem requires at least one surface."
        )
    jax_boozers = []
    natives: list[BoozerSurface | None] = []
    for surface in packed_surfaces:
        jax_surface = clone_surface_xyz_tensor_fourier(surface)
        jax_boozers.append(
            _make_jax_boozer(
                coils,
                jax_surface,
                constraint_weight=constraint_weight,
                optimizer_backend="ondevice",
                bfgs_maxiter=NCSX_NATIVE_BFGS_MAXITER,
                newton_maxiter=NESTED_LS_BANANA_NEWTON_MAXITER,
            )
        )
        natives.append(
            _make_native_boozer(
                coils,
                clone_surface_xyz_tensor_fourier(surface),
                constraint_weight=constraint_weight,
                bfgs_maxiter=NCSX_NATIVE_BFGS_MAXITER,
                newton_maxiter=NESTED_LS_BANANA_NEWTON_MAXITER,
            )
            if include_native
            else None
        )
    return ncsx_problem_from_jax_boozers(
        jax_boozers,
        iotas=iotas,
        g_values=g_values,
        base_curves=base_curves,
        curves=curves,
        natives=natives,
    )
