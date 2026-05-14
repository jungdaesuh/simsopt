"""VMEC true single-stage objective for HBT (paper-faithful, paper Eq. 13 + 26).

Implements the joint VMEC + coil objective from
``program_cw_poloidal_legacy_vmec_true_single_stage_plan.md`` sections
"Objective Design" and "Gradient contract":

.. code-block:: text

    J  = J1 + w_coils * J2
    J1 = qs_weight * f_QS                         (paper Eq. 13 baseline)
       + aspect_weight * (A - A_target)^2
       + iota_weight   * (iota(s*) - iota_target)^2
       + volume_weight * (V - V_target)^2         (HBT extension)
    J2 = SquaredFlux(local)                        (paper Eq. 26)
       + omega_L         * sum QuadraticPenalty(L_i, L*)
       + omega_kappa_max * sum LpCurveCurvature(c_i, p=2, kappa*)
       + omega_kappa_msc * sum QuadraticPenalty(MSC(c_i), msc*)
       + omega_d         * CurveCurveDistance
       + omega_ell       * sum ArclengthVariation

Gradient contract (plan section "Gradient contract"):

.. code-block:: text

    dJ1/dx_coils   = 0                                    (identity; asserted)
    dJ1/dx_surface = forward FD via MPIFiniteDifference   (boundary DOFs only)
    dJ2/dx_coils   = analytic SIMSOPT                     (.dJ())
    dJ2/dx_surface = analytic mixed derivative            (local SquaredFlux)

Step-clamp barrier (plan section "Optimizer entrypoint"):
the driver clamps proposed steps before any VMEC call. When the proposed
``x`` violates the boundary or coil clamp, ``J_scalar`` returns the
quadratic-barrier value and ``compute_grad`` returns its gradient *without*
running VMEC. This gives scipy's line search a real value to backtrack
against.

This module is intentionally NOT a runnable script. The caller
(``banana_drivers/04_vmec_singlestage_driver.py``) owns the MPI launch,
scratch-directory chdir lifecycle, ``MPIFiniteDifference`` context manager,
and the ``scipy.optimize.minimize`` call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from netCDF4 import Dataset

from simsopt._core.derivative import Derivative
from simsopt._core.finite_difference import MPIFiniteDifference
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    ArclengthVariation,
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
    MeanSquaredCurvature,
    create_equally_spaced_curves,
)
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt.util import MpiPartition

from .vmec_single_stage_config import (
    GRADIENT_NORM_FLOOR_BOUNDARY,
    GRADIENT_NORM_FLOOR_COILS,
    VmecSingleStageConfig,
)
from .vmec_single_stage_exceptions import (
    IotaSurfaceTargetMiss,
    SeedZeroGradientTrap,
)


@dataclass
class VmecSingleStageObjective:
    """Bundle returned by :func:`build_objective`.

    Exposes a single ``J_and_grad`` entrypoint so the scipy driver pays one
    VMEC run per outer iteration (``scipy.optimize.minimize(..., jac=True)``
    unpacks the ``(value, gradient)`` tuple from a single call). The legacy
    ``J_scalar`` and ``compute_grad`` views are kept for unit tests but
    they internally short-circuit to ``J_and_grad`` so they share the
    cached VMEC run when called back-to-back on the same ``x``.

    The driver's step-clamp lives in ``banana_drivers/04_vmec_singlestage_driver.py``;
    this module does not duplicate the clamp.
    """

    J_and_grad: Callable[[np.ndarray], Tuple[float, np.ndarray]]
    J_scalar: Callable[[np.ndarray], float]
    compute_grad: Callable[[np.ndarray], np.ndarray]
    dof_schema: Dict[str, object]
    x0: np.ndarray
    n_coil_dofs: int
    n_current_dofs: int
    n_boundary_dofs: int
    coil_currents_slice: slice
    boundary_slice: slice


# ---------------------------------------------------------------------------
# Iota at promotion surface
# ---------------------------------------------------------------------------


def iota_at_promotion_surface(
    wout, s_target: float
) -> float:
    """Linear interpolation of VMEC ``iotaf`` at the promotion-surface label.

    VMEC stores ``iotaf`` on the full radial grid ``s_full_grid`` (length
    ``ns``, including ``s=0`` and ``s=1``). The plan requires an explicit
    interpolation at the SSOT surface, not a nonexistent ``iota_at(s)`` helper.

    Args:
        wout: VMEC ``wout`` Struct (``vmec.wout`` after a successful run).
        s_target: Normalized toroidal-flux label in ``[0, 1]``.

    Returns:
        Interpolated iota value at ``s_target``.
    """

    s_full = np.linspace(0.0, 1.0, int(wout.ns))
    iotaf = np.asarray(wout.iotaf, dtype=float)
    return float(np.interp(float(s_target), s_full, iotaf))


# ---------------------------------------------------------------------------
# DOF schema
# ---------------------------------------------------------------------------


def assemble_dof_schema(
    *,
    coil_dof_count: int,
    current_dof_count: int,
    boundary_dof_count: int,
    pinned_current_index: int,
    pinned_current_value_A: float,
    fixed_boundary_modes: List[str],
    coils_currents_dof_names: List[str],
    current_dof_names: List[str],
    boundary_dof_names: List[str],
) -> Dict[str, object]:
    """Return the typed ``dof_schema`` manifest dict.

    Per plan section "Artifact Manifest", the schema must list:
        - coil-shape DOFs
        - current DOFs (one pinned per paper page 7)
        - VMEC boundary DOFs
        - fixed DOFs (at minimum ``RBC(0,0)`` and the pinned current index)
        - vector ordering
        - units

    Vector ordering matches the order ``J_scalar`` reads ``x``:
    ``[coils_currents_block, boundary_block]``. SIMSOPT decides the
    internal order of ``coils_currents_block`` via its sorted-by-name
    ancestor walk; ``coils_currents_dof_names`` records that authoritative
    order so the schema is faithful even though the objective body does
    not slice between coil-shape and free-current DOFs.

    The pinned current does *not* appear in ``current_dof_names``; it is
    recorded under ``fixed_dofs.pinned_currents`` with the pinned value.
    """

    return {
        "vector_ordering": [
            {
                "block": "coils_currents",
                "count": int(coil_dof_count + current_dof_count),
                "subblocks": {
                    "coil_shape_count": int(coil_dof_count),
                    "free_current_count": int(current_dof_count),
                },
            },
            {"block": "boundary", "count": int(boundary_dof_count)},
        ],
        "coils_currents": {
            "total_count": int(coil_dof_count + current_dof_count),
            "coil_shape_count": int(coil_dof_count),
            "free_current_count": int(current_dof_count),
            "dof_names_in_order": list(coils_currents_dof_names),
            "coil_shape_units": "dimensionless Fourier coefficients (Cartesian)",
            "current_units": "A",
        },
        "free_current_dof_names": list(current_dof_names),
        "boundary_dofs": {
            "count": int(boundary_dof_count),
            "names": list(boundary_dof_names),
            "units": "m for R*c, dimensionless for normalized Z*c (SIMSOPT SurfaceRZFourier)",
        },
        "fixed_dofs": {
            "boundary_modes": list(fixed_boundary_modes),
            "pinned_currents": [
                {
                    "coil_index": int(pinned_current_index),
                    "value_A": float(pinned_current_value_A),
                }
            ],
        },
    }


# ---------------------------------------------------------------------------
# Objective bundle
# ---------------------------------------------------------------------------


@dataclass
class _ObjectiveBundle:
    """Owns the SIMSOPT objects the objective body needs across calls.

    Held inside the closure returned by ``build_objective``. Mutable because
    SIMSOPT optimizable graphs are mutable; the bundle itself is not exposed
    to the caller, only the typed :class:`VmecSingleStageObjective` view.

    ``JF`` is the *single* composite J2 optimizable. Setting ``JF.x`` writes
    DOFs into every shared coil/current ancestor. Reading ``JF.dJ()`` returns
    one analytic gradient over the full coil+current DOF block. Holding it
    once avoids re-creating composite optimizables on every call.

    The DOF vector layout is ``[coils_currents_block, boundary_block]`` where
    SIMSOPT decides the internal order inside ``coils_currents_block`` based
    on its ancestor walk. ``coil_currents_slice`` and ``boundary_slice`` are
    the only slices the objective body uses; the schema names every DOF
    inside ``coils_currents_block`` for traceability.

    ``prior_axis`` caches the axis-guess arrays (``raxis_cc``, ``zaxis_cs``)
    loaded once from ``prior_wout_path``; refreshing the axis on a moved
    boundary then becomes an in-memory copy, not a netCDF reopen.

    ``_last_x_hash`` / ``_last_J_grad`` cache the most recent ``(J, grad)``
    keyed by ``x.tobytes()``. ``scipy.optimize.minimize(..., jac=True)``
    unpacks ``(value, gradient)`` from a single call into ``J_and_grad``;
    the cache covers the case where unit tests or non-jac=True callers
    invoke ``J_scalar`` and ``compute_grad`` back-to-back on the same ``x``.
    """

    vmec: Vmec
    surf: object  # SurfaceRZFourier (boundary)
    bs: BiotSavart
    Jf: SquaredFlux
    Jls: List[CurveLength]
    Jcc: CurveCurveDistance
    Jkappa: List[LpCurveCurvature]
    Jmsc: List[MeanSquaredCurvature]
    Jal: List[ArclengthVariation]
    JF: object  # composite J2 optimizable
    qs: QuasisymmetryRatioResidual
    base_curves: List[object]
    base_currents: List[Current]
    curves: List[object]
    coils: List[object]
    config: VmecSingleStageConfig
    coil_currents_slice: slice
    boundary_slice: slice
    prior_axis: Optional[Tuple[np.ndarray, np.ndarray]]
    _last_x_hash: Optional[bytes] = None
    _last_J_grad: Optional[Tuple[float, np.ndarray]] = None


# ---------------------------------------------------------------------------
# Mixed surface derivative for local SquaredFlux
# ---------------------------------------------------------------------------


def _mixed_surface_derivative_local_squared_flux(
    *,
    Jf: SquaredFlux,
    surf,
    bs: BiotSavart,
    nphi: int,
    ntheta: int,
) -> np.ndarray:
    """Analytic mixed derivative of local SquaredFlux w.r.t. surface DOFs.

    Reproduces the formula from the SIMSOPT reference example
    ``examples/3_Advanced/single_stage_optimization.py:175`` block, but
    expressed as an isolated function so the gradient assembler can call it
    once per ``compute_grad`` invocation.

    Args:
        Jf: ``SquaredFlux(surf, bs, definition="local")``.
        surf: Boundary surface (``vmec.boundary``).
        bs: ``BiotSavart`` with ``set_points(surf.gamma())`` already called.
        nphi, ntheta: Surface sampling sizes (must match the SquaredFlux
            quadrature grid).

    Returns:
        1D numpy array of the gradient with respect to ``surf.x``.
    """

    if Jf.definition != "local":
        raise ValueError(
            "Mixed surface derivative is implemented only for "
            'SquaredFlux(definition="local"). Got definition='
            f"{Jf.definition!r}."
        )

    n = surf.normal()
    absn = np.linalg.norm(n, axis=2)
    B = bs.B().reshape((nphi, ntheta, 3))
    dB_by_dX = bs.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    Bcoil = B
    unitn = n * (1.0 / absn)[:, :, None]
    Bcoil_n = np.sum(Bcoil * unitn, axis=2)
    mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
    B_N = np.sum(Bcoil * n, axis=2)
    # Bracketed kernels as in single_stage_optimization.py:176-177.
    dJdx = (Bcoil_n / mod_Bcoil**2)[:, :, None] * (
        np.sum(
            dB_by_dX
            * (n - B * (B_N / mod_Bcoil**2)[:, :, None])[:, :, None, :],
            axis=3,
        )
    )
    dJdN = (Bcoil_n / mod_Bcoil**2)[:, :, None] * Bcoil - 0.5 * (
        B_N**2 / absn**3 / mod_Bcoil**2
    )[:, :, None] * n
    deriv = surf.dnormal_by_dcoeff_vjp(
        dJdN / (nphi * ntheta)
    ) + surf.dgamma_by_dcoeff_vjp(dJdx / (nphi * ntheta))
    return np.asarray(Derivative({surf: deriv})(surf), dtype=float)


# ---------------------------------------------------------------------------
# J1 and J2 component evaluators
# ---------------------------------------------------------------------------


def _evaluate_J1(bundle: _ObjectiveBundle) -> float:
    """Paper Eq. 13 baseline + HBT extensions.

    Assumes the boundary surface has been set and ``vmec.run()`` already
    completed successfully in the caller's scratch directory.
    """

    cfg = bundle.config
    aspect = float(bundle.vmec.aspect())
    iota_promo = iota_at_promotion_surface(
        bundle.vmec.wout, cfg.iota_promotion_surface_s
    )
    volume = float(bundle.vmec.volume())
    f_qs = float(bundle.qs.total())
    return (
        cfg.qs_weight * f_qs
        + cfg.aspect_weight * (aspect - cfg.aspect_target) ** 2
        + cfg.iota_weight * (iota_promo - cfg.iota_target) ** 2
        + cfg.volume_weight * (volume - cfg.volume_target) ** 2
    )


def _evaluate_J2(bundle: _ObjectiveBundle) -> float:
    """Paper Eq. 26 coil objective with HBT-tuned regularizer weights.

    Evaluates the stored composite ``JF.J()``; the composite is built once
    in ``build_objective`` from the same weighted-sum recipe used by
    ``_evaluate_dJ2_dxcoils``.
    """

    return float(bundle.JF.J())


def _evaluate_dJ2_dxcoils(bundle: _ObjectiveBundle) -> np.ndarray:
    """Analytic gradient of J2 with respect to coil + current DOFs.

    SIMSOPT ``.dJ()`` is analytic in coil/field DOFs for all three
    ``SquaredFlux`` definitions and for every coil-geometry regularizer in
    the J2 composite. The composite is assembled once in ``build_objective``
    so each Jacobian sweep is a single ``.dJ()`` call.
    """

    return np.asarray(bundle.JF.dJ(), dtype=float)


# ---------------------------------------------------------------------------
# Bundle construction
# ---------------------------------------------------------------------------


def _build_coil_block(
    *,
    nfp: int,
    stellsym: bool,
    R0: float,
    R1: float,
    ncoils: int,
    nmodes_coils: int,
    nquadpoints: int,
    tf_current_pin_A: float,
    pinned_current_coil_index: int,
) -> Tuple[List[object], List[Current], List[object], List[object]]:
    """Build base curves, base currents (one pinned), coils, curves."""

    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=R0,
        R1=R1,
        order=nmodes_coils,
        numquadpoints=nquadpoints,
    )
    base_currents = [Current(1.0) * 1.0e5 for _ in range(ncoils)]
    # Paper page-7: pin exactly one current to avoid the zero-current
    # quadratic-flux trap. Plan: "the pinned TF-current convention satisfies
    # this only if dof_schema.fixed_dofs explicitly records the pinned
    # current index and value."
    pinned = base_currents[pinned_current_coil_index]
    pinned.set_value(float(tf_current_pin_A))
    pinned.fix_all()

    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)
    curves = [c.curve for c in coils]
    return base_curves, base_currents, curves, coils


def build_objective(
    vmec_input_path: str,
    *,
    mpi: MpiPartition,
    config: VmecSingleStageConfig,
    ncoils: int = 3,
    nmodes_coils: int = 7,
    coil_circle_radius_R1: float = 0.6,
    nquadpoints: int = 128,
    prior_wout_path: Optional[str] = None,
) -> "VmecSingleStageObjective":
    """Factory: build SIMSOPT objective graph and return the bundle.

    Args:
        vmec_input_path: Path to a *rerunnable* VMEC input file (must NOT be
            a ``wout_*.nc``). See ``vmec_seed_loader.materialize_rerunnable_input``.
        mpi: ``MpiPartition`` from the caller. The driver owns the partition
            lifecycle. Plan section "MPI partitioning and worker isolation":
            one ``Vmec`` per MPI group.
        config: Frozen run config.
        ncoils, nmodes_coils, coil_circle_radius_R1, nquadpoints: Coil
            initial-condition parameters. Pinned by the caller, not optimized.
        prior_wout_path: Optional path to a prior accepted ``wout`` whose
            axis (``raxis_cc``, ``zaxis_cs``) is reused as the initial axis
            guess on a moved boundary. Plan section "SIMSOPT/VMEC API
            Constraints": "Refresh the VMEC initial axis guess from the
            prior accepted wout before every VMEC call on a moved boundary."
            Loaded once at build time; the runtime closure reuses the
            cached arrays instead of re-opening the netCDF.

    Returns:
        :class:`VmecSingleStageObjective` bundle exposing ``J_and_grad``
        (one VMEC run per call), ``J_scalar`` / ``compute_grad`` (thin
        views over the cached pair), ``dof_schema``, and DOF block sizes.

        - ``J_scalar(x)``: returns ``J = J1 + w_coils * J2``, applying the
          step-clamp barrier before any VMEC call.
        - ``compute_grad(x)``: returns ``dJ/dx`` per the plan gradient
          contract.
        - ``dof_schema``: typed manifest dict for ``artifact_manifest.json``.
    """

    cfg = config

    # ----- Vmec setup -----
    vmec = Vmec(
        vmec_input_path,
        mpi=mpi,
        verbose=False,
        nphi=cfg.nphi_vmec,
        ntheta=cfg.ntheta_vmec,
        range_surface="half period",
    )
    # Pin VMEC-owned numerical schedule. Direct ``indata`` mutations require
    # explicit ``need_to_run_code = True`` per the SIMSOPT VMEC docstring
    # (mhd/vmec.py:153-157).
    vi = vmec.indata
    vi.ns_array[: len(cfg.ns_array)] = np.asarray(cfg.ns_array, dtype=int)
    vi.ftol_array[: len(cfg.ftol_array)] = np.asarray(cfg.ftol_array, dtype=float)
    vi.niter_array[: len(cfg.niter_array)] = np.asarray(cfg.niter_array, dtype=int)
    vi.delt = float(cfg.delt)
    vi.mpol = int(cfg.mpol)
    vi.ntor = int(cfg.ntor)
    # Plan section "SIMSOPT/VMEC API Constraints": all Stage A is vacuum.
    vi.pres_scale = 0.0
    # Plan H9: explicit fixed-boundary commitment. Reject any seed that came
    # in with lfreeb=True; the loader should have rejected it already.
    if bool(vi.lfreeb):
        raise ValueError(
            "Seed VMEC input enables free boundary (lfreeb=.true.); "
            "Stage A pins fixed-boundary mode. See plan H9."
        )
    vmec.need_to_run_code = True

    surf = vmec.boundary
    surf.fix_all()
    # Free only modes (m, n) with m <= mpol and -ntor <= n <= ntor, per the
    # SIMSOPT reference example.
    surf.fixed_range(
        mmin=0, mmax=int(cfg.mpol), nmin=-int(cfg.ntor), nmax=int(cfg.ntor),
        fixed=False,
    )
    surf.fix("rc(0,0)")

    # ----- Coil block -----
    base_curves, base_currents, curves, coils = _build_coil_block(
        nfp=surf.nfp,
        stellsym=True,
        R0=float(cfg.R0),
        R1=float(coil_circle_radius_R1),
        ncoils=int(ncoils),
        nmodes_coils=int(nmodes_coils),
        nquadpoints=int(nquadpoints),
        tf_current_pin_A=float(cfg.tf_current_pin_A),
        pinned_current_coil_index=int(cfg.pinned_current_coil_index),
    )
    bs = BiotSavart(coils)
    bs.set_points(surf.gamma().reshape((-1, 3)))

    # ----- J2 building blocks -----
    Jf = SquaredFlux(surf, bs, definition="local")
    Jls = [CurveLength(c) for c in base_curves]
    Jcc = CurveCurveDistance(curves, float(cfg.cc_threshold), num_basecurves=len(curves))
    Jkappa = [
        LpCurveCurvature(c, 2, float(cfg.kappa_max_threshold)) for c in base_curves
    ]
    Jmsc = [MeanSquaredCurvature(c) for c in base_curves]
    Jal = [ArclengthVariation(c) for c in base_curves]

    # ----- J1 building blocks -----
    qs = QuasisymmetryRatioResidual(
        vmec,
        list(cfg.qs_surfaces),
        helicity_m=int(cfg.qs_helicity_m),
        helicity_n=int(cfg.qs_helicity_n),
    )

    # ----- Composite J2 -----
    # Paper Eq. 26 weighted sum. The composite shares ancestor parameter
    # arrays with every component, so setting ``JF.x = coils_block``
    # propagates one consistent DOF vector across the entire J2 graph.
    JF = (
        Jf
        + cfg.omega_L
        * sum(QuadraticPenalty(Jl, float(cfg.L_threshold)) for Jl in Jls)
        + cfg.omega_kappa_max * sum(Jkappa)
        + cfg.omega_kappa_msc
        * sum(QuadraticPenalty(Jm, float(cfg.kappa_msc_threshold)) for Jm in Jmsc)
        + cfg.omega_d * Jcc
        + cfg.omega_ell * sum(Jal)
    )
    # ``JF.x`` is the combined (coil-shape, current) DOF vector. SIMSOPT
    # decides the internal order via ``_get_ancestors`` (sorted by name);
    # the schema records the full ``JF.dof_names`` list so the order is
    # auditable from artifacts even though the objective body treats
    # ``coils_currents`` as a single opaque block.
    coils_currents_dofs = np.asarray(JF.x, dtype=float).copy()
    n_coils_currents = coils_currents_dofs.size
    n_boundary_dofs = int(surf.x.size)

    # Count free vs pinned currents for the schema. SIMSOPT ``Current`` has a
    # single dof named "x0"; one is pinned by ``_build_coil_block``.
    n_current_dofs_free = sum(
        1 for c in base_currents if not c.is_fixed("x0")
    )
    n_coil_shape_dofs = n_coils_currents - n_current_dofs_free

    coil_currents_slice = slice(0, n_coils_currents)
    boundary_slice = slice(n_coils_currents, n_coils_currents + n_boundary_dofs)

    # Full dof_names of the composite are SIMSOPT's authoritative order.
    coils_currents_dof_names = (
        list(JF.dof_names) if hasattr(JF, "dof_names")
        else [f"jf_dof_{i}" for i in range(n_coils_currents)]
    )
    boundary_dof_names = (
        list(surf.local_dof_names) if hasattr(surf, "local_dof_names")
        else [f"boundary_{i}" for i in range(n_boundary_dofs)]
    )

    dof_schema = assemble_dof_schema(
        coil_dof_count=n_coil_shape_dofs,
        current_dof_count=n_current_dofs_free,
        boundary_dof_count=n_boundary_dofs,
        pinned_current_index=int(cfg.pinned_current_coil_index),
        pinned_current_value_A=float(cfg.tf_current_pin_A),
        fixed_boundary_modes=["rc(0,0)"],
        coils_currents_dof_names=coils_currents_dof_names,
        current_dof_names=[
            f"coil_{i}_current_A"
            for i, c in enumerate(base_currents)
            if not c.is_fixed("x0")
        ],
        boundary_dof_names=boundary_dof_names,
    )

    x0 = np.concatenate(
        [coils_currents_dofs, np.asarray(surf.x, dtype=float)]
    )

    # Pre-load axis from prior wout once (plan section "SIMSOPT/VMEC API
    # Constraints" requires axis refresh on moved boundary; re-opening the
    # netCDF on every J_and_grad call is wasteful).
    prior_axis: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if prior_wout_path is not None:
        with Dataset(prior_wout_path, "r") as nc:
            prior_axis = (
                np.asarray(nc.variables["raxis_cc"][:], dtype=float),
                np.asarray(nc.variables["zaxis_cs"][:], dtype=float),
            )

    bundle = _ObjectiveBundle(
        vmec=vmec,
        surf=surf,
        bs=bs,
        Jf=Jf,
        Jls=Jls,
        Jcc=Jcc,
        Jkappa=Jkappa,
        Jmsc=Jmsc,
        Jal=Jal,
        JF=JF,
        qs=qs,
        base_curves=base_curves,
        base_currents=base_currents,
        curves=curves,
        coils=coils,
        config=cfg,
        coil_currents_slice=coil_currents_slice,
        boundary_slice=boundary_slice,
        prior_axis=prior_axis,
    )

    def _apply_dofs(x: np.ndarray) -> None:
        coils_block = x[: n_coils_currents]
        boundary_block = x[n_coils_currents : n_coils_currents + n_boundary_dofs]
        # Coil-shape + free-current DOFs flow through the composite ``JF``;
        # setting ``JF.x`` writes into every shared coil/current ancestor.
        bundle.JF.x = coils_block
        # Surface DOFs are owned by the Vmec.boundary surface; mutating
        # ``surf.x`` marks ``vmec.need_to_run_code = True`` per the SIMSOPT
        # cache contract.
        bundle.surf.x = boundary_block

    def _refresh_axis() -> None:
        if bundle.prior_axis is None:
            return
        raxis_cc, zaxis_cs = bundle.prior_axis
        n = min(raxis_cc.size, bundle.vmec.indata.raxis_cc.size)
        bundle.vmec.indata.raxis_cc[:n] = raxis_cc[:n]
        m = min(zaxis_cs.size, bundle.vmec.indata.zaxis_cs.size)
        bundle.vmec.indata.zaxis_cs[:m] = zaxis_cs[:m]
        bundle.vmec.need_to_run_code = True

    def J_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Single combined entrypoint. One VMEC run per ``x``.

        ``scipy.optimize.minimize(..., jac=True)`` unpacks the tuple from
        one call; ``J_scalar`` and ``compute_grad`` are thin views that
        return one half of the cached pair. The cache is keyed by the
        ``x.tobytes()`` digest so a back-to-back ``J_scalar``/``compute_grad``
        pair on the same ``x`` does not re-enter VMEC.

        Raises:
            IotaSurfaceTargetMiss: ``iota_at_promotion_surface(wout)``
                deviates from ``config.iota_target`` by more than
                ``config.iota_tol`` after a successful VMEC run.
        """

        x = np.asarray(x, dtype=float)
        if x.shape != x0.shape:
            raise ValueError(
                f"J_and_grad received x of shape {x.shape}; expected "
                f"{x0.shape}"
            )
        x_hash = x.tobytes()
        cached = bundle._last_J_grad
        if bundle._last_x_hash == x_hash and cached is not None:
            return cached[0], cached[1].copy()

        _apply_dofs(x)
        _refresh_axis()
        # Caller is responsible for ``os.chdir(scratch)`` BEFORE invoking
        # this. VMEC writes side files into cwd.
        bundle.vmec.run()
        bundle.bs.set_points(bundle.surf.gamma().reshape((-1, 3)))

        iota_promo = iota_at_promotion_surface(
            bundle.vmec.wout, bundle.config.iota_promotion_surface_s
        )
        if abs(iota_promo - bundle.config.iota_target) > bundle.config.iota_tol:
            raise IotaSurfaceTargetMiss(
                f"|iota(s={bundle.config.iota_promotion_surface_s}) - "
                f"iota_target| = "
                f"{abs(iota_promo - bundle.config.iota_target):.4e} > "
                f"iota_tol={bundle.config.iota_tol:.4e}",
                ier_flag=int(bundle.vmec.wout.ier_flag),
            )

        J1 = _evaluate_J1(bundle)
        J2 = _evaluate_J2(bundle)
        J = float(J1 + bundle.config.w_coils * J2)

        dJ2_dxcoils = _evaluate_dJ2_dxcoils(bundle)
        mixed_dJ2_dxsurface = _mixed_surface_derivative_local_squared_flux(
            Jf=bundle.Jf,
            surf=bundle.surf,
            bs=bundle.bs,
            nphi=bundle.config.nphi_vmec,
            ntheta=bundle.config.ntheta_vmec,
        )

        # dJ1/dx_surface via MPIFiniteDifference over boundary DOFs only.
        # Plan section "Gradient contract": "MPIFiniteDifference only for
        # VMEC/stage-1 boundary terms."
        def _J1_for_fd() -> float:
            bundle.vmec.run()
            return _evaluate_J1(bundle)

        with MPIFiniteDifference(
            _J1_for_fd,
            mpi,
            diff_method="forward",
            abs_step=bundle.config.eps_fd_xsurface_abs,
            rel_step=bundle.config.eps_fd_xsurface_rel,
        ) as fd_block:
            if mpi.proc0_world:
                dJ1_dxsurface = np.asarray(
                    fd_block.jac(bundle.surf.x), dtype=float
                ).ravel()
            else:
                dJ1_dxsurface = np.zeros(n_boundary_dofs, dtype=float)
        mpi.comm_world.Bcast(dJ1_dxsurface, root=0)

        grad = np.zeros_like(x, dtype=float)
        grad[bundle.coil_currents_slice] = (
            bundle.config.w_coils * dJ2_dxcoils
        )
        grad[bundle.boundary_slice] = (
            dJ1_dxsurface + bundle.config.w_coils * mixed_dJ2_dxsurface
        )
        bundle._last_x_hash = x_hash
        bundle._last_J_grad = (J, grad)
        return J, grad.copy()

    def J_scalar(x: np.ndarray) -> float:
        return J_and_grad(x)[0]

    def compute_grad(x: np.ndarray) -> np.ndarray:
        return J_and_grad(x)[1]

    return VmecSingleStageObjective(
        J_and_grad=J_and_grad,
        J_scalar=J_scalar,
        compute_grad=compute_grad,
        dof_schema=dof_schema,
        x0=x0,
        n_coil_dofs=int(n_coil_shape_dofs),
        n_current_dofs=int(n_current_dofs_free),
        n_boundary_dofs=int(n_boundary_dofs),
        coil_currents_slice=coil_currents_slice,
        boundary_slice=boundary_slice,
    )


# ---------------------------------------------------------------------------
# Seed-gradient trap (called once at seed, before optimizer loop)
# ---------------------------------------------------------------------------


def check_seed_zero_gradient_trap(
    *,
    compute_grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    coil_currents_slice: slice,
    boundary_slice: slice,
    w_coils: float,
    boundary_floor: float = GRADIENT_NORM_FLOOR_BOUNDARY,
    coil_floor: float = GRADIENT_NORM_FLOOR_COILS,
) -> None:
    """Run the first-FD gradient-norm check at seed.

    Plan section "Finite-difference step schedule":

    1. Boundary-DOF FD gradient norm of ``J1 + w_coils * J2`` must exceed
       ``GRADIENT_NORM_FLOOR_BOUNDARY``.
    2. When ``w_coils > 0``, the J2-only coil-DOF gradient norm must exceed
       ``GRADIENT_NORM_FLOOR_COILS``.
    3. When ``w_coils == 0`` (boundary-only warmup), condition 2 is skipped.

    Raises:
        SeedZeroGradientTrap: When either condition fails.
    """

    grad = compute_grad(np.asarray(x0, dtype=float))
    coils_block = grad[coil_currents_slice]
    boundary_block = grad[boundary_slice]
    boundary_norm = float(np.linalg.norm(boundary_block))
    coil_norm = float(np.linalg.norm(coils_block))

    if boundary_norm < boundary_floor:
        raise SeedZeroGradientTrap(
            f"Boundary-DOF gradient norm {boundary_norm:.3e} below floor "
            f"{boundary_floor:.3e} at seed.",
            boundary_norm=boundary_norm,
            coil_norm=coil_norm,
        )
    if w_coils > 0.0 and coil_norm < coil_floor:
        raise SeedZeroGradientTrap(
            f"Coil-DOF gradient norm {coil_norm:.3e} below floor "
            f"{coil_floor:.3e} at seed (w_coils={w_coils:.3e}).",
            boundary_norm=boundary_norm,
            coil_norm=coil_norm,
        )
