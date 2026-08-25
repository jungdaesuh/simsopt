"""NCSX boozerQA nested-LS: reduced Schur inner and 9-term outer.

This is not F3/flat-675 and does not call ``prepare_f3_b37_outer_state``.
The inner is :func:`run_reduced_nested_ls_schur_newton` with dense LU.
The outer variable is the free coil DOF vector; surface DOFs are
eliminated by the Schur implicit-function theorem. C++ reconstruct
Newton remains the untimed rejudge. MPI multi-rank splitting is out of
scope: one process takes the :class:`~simsopt.objectives.MPIObjective`
mean of each per-surface term, matching serial ``boozerQA_ls_mpi.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, Volume
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.objectives import QuadraticPenalty

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
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NestedLsSchurNewtonResult,
    implicit_adjoint_coil_gradient,
    nested_ls_reduced_closures,
    nested_ls_runtime_coil_closures,
    require_full_y_rank,
    run_reduced_nested_ls_schur_newton,
    solve_projected_y,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    BoozerResidualJAX,
    MajorRadiusJAX,
    NonQuasiSymmetricRatioJAX,
    surface_dmajor_radius_jax_from_dofs,
    surface_major_radius_jax_from_dofs,
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
    """

    weight = (
        float(jax_boozer.constraint_weight)
        if constraint_weight is None
        else float(constraint_weight)
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
    )


def _projected_y_surface_jacobian(residual_fn, surface, y_probe):
    surface_jax = jnp.asarray(surface, dtype=jnp.float64).reshape(-1)
    probe = jnp.asarray(y_probe, dtype=jnp.float64).reshape(-1)

    def y_of_surface(surface_dofs: jax.Array) -> jax.Array:
        return solve_projected_y(residual_fn, surface_dofs, probe).solution

    solution = solve_projected_y(residual_fn, surface_jax, probe)
    require_full_y_rank(solution)
    jacobian = jax.jacrev(y_of_surface)(surface_jax)
    return solution, jacobian


def _projected_y_coil_jacobian(residual_rt, surface, coil_dofs, y_probe):
    surface_jax = jnp.asarray(surface, dtype=jnp.float64).reshape(-1)
    coil = jnp.asarray(coil_dofs, dtype=jnp.float64).reshape(-1)
    probe = jnp.asarray(y_probe, dtype=jnp.float64).reshape(-1)

    def y_of_coil(coil_vector: jax.Array) -> jax.Array:
        def residual_fn(packed: jax.Array) -> jax.Array:
            return residual_rt(packed, coil_vector)

        return solve_projected_y(residual_fn, surface_jax, probe).solution

    return jax.jacrev(y_of_coil)(coil)


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
    """One NCSX surface plus the JAX objectives that read it.

    Live ``iota`` / ``G`` are the last inner guess. The committed warm
    start is ``anchor_surface_dofs`` with ``anchor_iota`` and
    ``anchor_G``. Only :func:`commit_ncsx_anchor` advances the anchor.
    """

    jax_boozer: BoozerSurfaceJAX
    native: BoozerSurface | None
    iota: float
    G: float
    anchor_iota: float
    anchor_G: float
    anchor_surface_dofs: NDArray[np.float64]
    nonqs: NonQuasiSymmetricRatioJAX
    residual: BoozerResidualJAX
    major_radius: MajorRadiusJAX
    radius_target: float
    radius_scale: float


@dataclass
class NcsxNestedLsProblem:
    """Coil-outer NCSX 9-term problem with reduced-Schur inners."""

    surfaces: tuple[NcsxNestedLsSurfaceState, ...]
    coil_terms: object
    iota_target: float
    length_target: float
    biotsavart: BiotSavartJAX
    last_inner: NestedLsSchurNewtonResult | None = field(default=None, repr=False)


def _surface_partials_at_solved(
    state: NcsxNestedLsSurfaceState,
    *,
    coil_dofs: jax.Array,
    inner: NestedLsSchurNewtonResult,
    residual_fn,
    residual_rt,
    iota_dval: float,
    nsurf: int,
) -> tuple[float, NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(J_surface_except_iota, ∂J/∂c, ∂J/∂s)`` including the y chain.

    NonQS and residual enter as the MPIObjective mean over ``nsurf``.
    Major radius is already the last-surface identity penalty (no extra
    ``× nsurf``). ``∂y/∂s`` and ``∂y/∂c|_s`` both multiply ``∂J/∂y``.
    """

    sdofs = jnp.asarray(inner.surface_dofs, dtype=jnp.float64)
    y_probe = np.array([float(inner.iota), float(inner.G)], dtype=np.float64)
    _y_solution, y_jacobian = _projected_y_surface_jacobian(
        residual_fn, sdofs, y_probe
    )
    del _y_solution
    y_jac_s = jnp.asarray(y_jacobian, dtype=jnp.float64)
    y_jac_c = jnp.asarray(
        _projected_y_coil_jacobian(residual_rt, sdofs, coil_dofs, y_probe),
        dtype=jnp.float64,
    )
    inv_n = 1.0 / float(nsurf)

    qs_value, qs_dc, qs_ds = state.nonqs._direct_objective_value_and_gradients(
        coil_dofs,
        sdofs,
    )
    x_inner, optimize_G = state.residual._inner_objective_state(
        float(inner.iota),
        float(inner.G),
        sdofs=sdofs,
    )
    res_value, res_dc, res_dx = state.residual._direct_objective_value_and_gradients(
        coil_dofs,
        x_inner,
        optimize_G,
        NESTED_LS_WEIGHT_INV_MODB,
    )
    res_dx = jnp.asarray(res_dx, dtype=jnp.float64).reshape(-1)
    res_ds = res_dx[: sdofs.size]
    res_dy = res_dx[sdofs.size :]

    radius = surface_major_radius_jax_from_dofs(
        state.major_radius._surface_spec(), sdofs
    )
    radius_ds = surface_dmajor_radius_jax_from_dofs(
        state.major_radius._surface_spec(), sdofs
    )
    radius_j, radius_dval = _identity_quadratic(
        float(_host_float64(radius)[0]), state.radius_target
    )

    qs_j = inv_n * float(_host_float64(qs_value)[0])
    res_j = inv_n * float(_host_float64(res_value)[0])
    value = qs_j + NCSX_RES_WEIGHT * res_j + state.radius_scale * radius_j
    coil_grad = inv_n * (
        _host_float64(qs_dc) + NCSX_RES_WEIGHT * _host_float64(res_dc)
    )
    surface_grad = (
        inv_n
        * (
            jnp.asarray(qs_ds, dtype=jnp.float64).reshape(-1)
            + NCSX_RES_WEIGHT * res_ds
        )
        + state.radius_scale * radius_dval * jnp.asarray(radius_ds, dtype=jnp.float64)
    )
    y_partial = jnp.zeros((2,), dtype=jnp.float64).at[0].set(iota_dval)
    y_partial = y_partial + inv_n * NCSX_RES_WEIGHT * res_dy.reshape(-1)
    surface_grad = surface_grad + y_jac_s.T @ y_partial
    coil_grad = coil_grad + _host_float64(y_jac_c.T @ y_partial)
    return value, coil_grad, _host_float64(surface_grad)


def ncsx_nested_ls_outer_value_and_grad(
    problem: NcsxNestedLsProblem,
    coil_dofs: object,
) -> tuple[float, NDArray[np.float64]]:
    """Nine-term ``J(c)`` and coil gradient at ``s*(c)`` via Schur IFT.

    Always warm-starts from the committed anchor, never from a previous
    trial. A successful return leaves the trial ``s*(c)`` on the Boozer
    objects for the caller to snapshot; it does not commit the anchor.
    Failures, including a self-intersecting trial surface, restore the
    anchor before raising.
    """

    restore_ncsx_anchor(problem)
    coil = np.asarray(coil_dofs, dtype=np.float64).reshape(-1)
    problem.biotsavart.x = np.array(coil, dtype=np.float64, copy=True)
    succeeded = False
    try:
        inners: list[NestedLsSchurNewtonResult] = []
        for surface_state in problem.surfaces:
            jax_boozer = surface_state.jax_boozer
            jax_boozer.biotsavart.x = np.array(coil, dtype=np.float64, copy=True)
            jax_boozer._refresh_coil_data()
            inner = run_ncsx_schur_inner(
                jax_boozer,
                iota=surface_state.anchor_iota,
                G=surface_state.anchor_G,
            )
            problem.last_inner = inner
            if not inner.success:
                raise NcsxNestedLsInnerSolveFailed(
                    iteration_count=int(inner.iteration_count),
                    grad_l2=float(np.linalg.norm(inner.reduced_gradient)),
                    exit_status=str(inner.exit_status),
                )
            if abs(float(inner.iota) - float(surface_state.anchor_iota)) > float(
                NESTED_LS_OUTER_IOTA_BRANCH_GUARD
            ):
                raise NcsxNestedLsBranchJump(
                    iota=float(inner.iota),
                    anchor_iota=float(surface_state.anchor_iota),
                    guard=float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
                )
            inners.append(inner)

        for index, surface_state in enumerate(problem.surfaces):
            if surface_state.jax_boozer.surface.is_self_intersecting():
                raise NcsxNestedLsSelfIntersecting(surface_index=index)

        nsurf = len(inners)
        mean_iota = float(sum(inner.iota for inner in inners) / nsurf)
        iota_j, mean_dval = _identity_quadratic(mean_iota, problem.iota_target)
        iota_dval = NCSX_IOTAS_WEIGHT * mean_dval / float(nsurf)
        total = NCSX_IOTAS_WEIGHT * iota_j
        coil_grad = np.zeros_like(coil)
        coil_jax = jnp.asarray(coil, dtype=jnp.float64)
        for surface_state, inner in zip(problem.surfaces, inners, strict=True):
            jax_boozer = surface_state.jax_boozer
            residual_fn, _objective_fn, _phi = nested_ls_reduced_closures(jax_boozer)
            del _objective_fn, _phi
            residual_rt, objective_rt, _phi_rt = nested_ls_runtime_coil_closures(
                jax_boozer
            )
            del _phi_rt
            term_j, term_dc, term_ds = _surface_partials_at_solved(
                surface_state,
                coil_dofs=coil_jax,
                inner=inner,
                residual_fn=residual_fn,
                residual_rt=residual_rt,
                iota_dval=iota_dval,
                nsurf=nsurf,
            )
            correction = implicit_adjoint_coil_gradient(
                residual_rt,
                objective_rt,
                inner.surface_dofs,
                coil,
                term_ds,
                stab=float(NESTED_LS_JAX_INNER_STAB),
                linear_solver="dense_lu",
                max_dense_linearization_bytes=None,
            )
            total += float(term_j)
            coil_grad = coil_grad + term_dc + _host_float64(correction)

        coil_j, coil_dj = _coil_term_value_and_grad(
            problem.coil_terms, problem.biotsavart
        )
        total += coil_j
        coil_grad = coil_grad + coil_dj
        if not np.isfinite(total) or not bool(np.all(np.isfinite(coil_grad))):
            raise RuntimeError(
                f"NCSX nested-LS outer evaluation is not finite: J={total!r}."
            )
        for surface_state, inner in zip(problem.surfaces, inners, strict=True):
            surface_state.iota = float(inner.iota)
            surface_state.G = float(inner.G)
        succeeded = True
        return total, coil_grad
    finally:
        if not succeeded:
            restore_ncsx_anchor(problem)


def commit_ncsx_anchor(problem: NcsxNestedLsProblem) -> None:
    """Snapshot the live ``(s, ι, G)`` as the committed warm start."""

    for surface_state in problem.surfaces:
        surface_state.anchor_surface_dofs = np.array(
            surface_state.jax_boozer.surface.get_dofs(),
            dtype=np.float64,
            copy=True,
        )
        surface_state.anchor_iota = float(surface_state.iota)
        surface_state.anchor_G = float(surface_state.G)


def restore_ncsx_anchor(problem: NcsxNestedLsProblem) -> None:
    """Install the committed ``(s, ι, G)``; discard any unaccepted trial."""

    for surface_state in problem.surfaces:
        surface_state.jax_boozer.surface.set_dofs(
            np.array(surface_state.anchor_surface_dofs, dtype=np.float64, copy=True)
        )
        surface_state.iota = float(surface_state.anchor_iota)
        surface_state.G = float(surface_state.anchor_G)


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


def _make_jax_boozer(
    coils,
    surface: SurfaceXYZTensorFourier,
    *,
    constraint_weight: float,
    optimizer_backend: str,
) -> BoozerSurfaceJAX:
    label = Volume(surface)
    return BoozerSurfaceJAX(
        BiotSavartJAX(coils),
        surface,
        label,
        float(label.J()),
        constraint_weight=float(constraint_weight),
        options={
            "verbose": False,
            "optimizer_backend": optimizer_backend,
            "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
        },
    )


def _make_native_boozer(
    coils,
    surface: SurfaceXYZTensorFourier,
    *,
    constraint_weight: float,
) -> BoozerSurface:
    label = Volume(surface)
    return BoozerSurface(
        BiotSavart(coils),
        surface,
        label,
        float(label.J()),
        constraint_weight=float(constraint_weight),
        options={
            "verbose": False,
            "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
        },
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
                radius_target=radius_target,
                radius_scale=(NCSX_MR_WEIGHT if index == nsurf - 1 else 0.0),
            )
        )
    coil_terms, length_target = _build_coil_terms(list(base_curves), list(curves))
    problem = NcsxNestedLsProblem(
        surfaces=tuple(surface_states),
        coil_terms=coil_terms,
        iota_target=NCSX_IOTAS_TARGET,
        length_target=length_target,
        biotsavart=packed[0].biotsavart,
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
    """Build the 9-term NCSX outer around reduced-Schur inners.

    ``surfaces`` are already at the intended Fourier/quadrature scale.
    The last surface carries the major-radius equality, matching
    ``boozerQA_ls_mpi.py``.
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
            )
        )
        natives.append(
            _make_native_boozer(
                coils,
                clone_surface_xyz_tensor_fourier(surface),
                constraint_weight=constraint_weight,
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
