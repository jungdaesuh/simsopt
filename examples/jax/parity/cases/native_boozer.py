"""Exact matched workflow for ``2_Intermediate/boozer.py``.

Two residual definitions are published at the initial state and they are not
the same quantity:

* ``initial:residual`` / ``initial:jacobian`` are the SOLVER'S penalty residual
  and its Jacobian -- ``(3*nphi*ntheta + 2,)`` entries, divided by
  ``sqrt(3*nphi*ntheta)``, ``weight_inv_modB=True``, with the label and
  ``z(0,0)`` constraint rows appended.  They are the cross-lane parity keys.
* ``initial:boozer_residual`` / ``initial:boozer_jacobian`` are the plain
  Boozer residual ``boozer_surface_residual(surface, iota, G, field,
  derivatives=1)`` -- ``(3*nphi*ntheta,)`` entries, unscaled,
  ``weight_inv_modB=False``, no constraint rows.  That is the definition the
  official capture records, so these keys compare exactly with upstream.  Both
  lanes evaluate them through the native library, so they say nothing about
  JAX-versus-native agreement.

``first:*`` publishes the end state and the provider outcome of the official
first stage (the L-BFGS-B reduction), next to ``area:*`` and ``flux:*``.  Three
of those keys are exactly comparable and ARE compared lane against lane at
``native_default``: ``first:stopping_reason_code``, ``first:nit`` and
``first:solver_success``.  There both lanes get ``OFFICIAL_LBFGS_MAXITER`` from
``_scale_configuration``, and the stage is a budget exit on upstream itself, so
both lanes stop at the cap and the three keys agree with each other and with
the official record (``iteration-limit`` / ``nit 300`` / ``success false``;
upstream's own ``status 1``, ``nit 300``, ``success false``).  At the bounded
scale the JAX lane runs a reduced budget for the measured reason in
``_scale_configuration``, so the same three routes are ``applicable: false``
there.  Neither the manifest nor
``tests/integration/test_jax_mirror_boozer_parity.py`` restates that split:
both derive it from the declared budgets.

The compared first-stage outcome is the NORMALIZED stopping reason, never the
raw provider status, because the two lanes' first stages are two different
emitters: the native lane's is ``scipy.optimize.minimize(method='L-BFGS-B')``
(convention ``scipy-lbfgsb``) and the JAX lane's is the private on-device
L-BFGS-B port reached through ``Driver.SIMSOPT_LBFGSB`` (convention
``private-lbfgsb``, public status from
``simsopt_jax.geo.optimizers.private._lbfgsb_scipy.lbfgsb_public_status_from_state``).
``simsopt_contracts.optimization_endpoint`` owns both vocabularies and says why
no table is shared: the same integer means different things in different
solvers.  The two agree on 1 and 2, but the private port adds ``6`` (non-finite
endpoint) and ``99`` (callback stop), which SciPy never emits -- so a raw
comparison both fails on a status 1 / status 6 pair that means the same stop
and passes on a status 99 / status 99 pair that means two different stops.
``FIRST_STAGE_STATUS_CONVENTION_BY_DRIVER`` names one emitter per lane driver
and ``first_stage_stopping_reason`` classifies through the contract, exactly as
``native_qfm.py`` does for its six SciPy calls.

The other ``first:*`` keys are published diagnostics with no comparator, and
their routes stay ``applicable: false``:

* ``first:status`` is each provider's own raw integer; it is recorded so the
  classification can be audited from the receipt, and it is never judged;
* ``first:nfev`` and ``first:njev`` count line-search trials at the same
  budget, which two implementations of the same algorithm need not match --
  measured 373 (native) versus 363 (JAX) versus 365 (official) at
  ``native_default``, and the native lane alone moved 370 -> 373 between two
  revisions of this branch;
* the six floats are that budget exit's path-dependent end point -- measured
  iota -0.53067 (native) versus -0.53012 (JAX) versus -0.52258 (official).

No official number above is a literal here: they come from the tracked record
``examples/jax/parity/official_reference/9e027eac3/native-boozer.json`` and are
compared against this case's declared configuration in
``tests/integration/test_jax_mirror_boozer_parity.py``; the lane measurements
are in ``A/fix-wave-3/boozer/probe/RESULTS.md``.

``first:``/``area:``/``flux:provider_persisted_iterate`` report whether each
stage left the provider's own iterate behind or restored the stage start, and
they are compared exactly.  Upstream always persists; this repository's native
library reverts a stage that failed without reducing the norm
(``src/simsopt/geo/boozersurface.py:624`` on the L-BFGS route, ``:897`` on the
manual route) while ``BoozerSurfaceJAX`` commits as upstream does
(``src/simsopt_jax_adapters/geo/boozer_surface.py:8162,8619``).  Without this
key the two lanes could chain the next stage from different end states and the
divergence would surface only as an unexplained ``area:``/``flux:`` mismatch.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, get_args

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_contracts.optimization_endpoint import (
    StatusConvention,
    StoppingReason,
    certify_optimization_endpoint,
)
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_CONSTRAINT_WEIGHT,
    OFFICIAL_FLUX_MULTIPLIER,
    OFFICIAL_INITIAL_IOTA,
    OFFICIAL_LBFGS_MAXITER,
    OFFICIAL_LS_MAXITER,
    OFFICIAL_SOLVER_TOLERANCE,
    OFFICIAL_SURFACE_DISTANCE,
    OFFICIAL_SURFACE_RESOLUTION,
    BoozerStageOutcome,
    BoozerStageState,
    boozer_official_options,
    run_boozer_lbfgs_stage,
    run_boozer_manual_stage,
)

WORKFLOW_STAGES = (
    "construct_ncsx_coils_and_tensor_fourier_surface",
    "evaluate_initial_boozer_residual_and_jacobian",
    "reduce_area_constrained_residual_with_lbfgs",
    "polish_area_constrained_residual_with_least_squares",
    "triple_toroidal_flux_label_and_resolve_surface",
)

#: The solver driver each lane runs, end to end.  The first stage's emitter is
#: read from this one name per lane, so the published convention cannot drift
#: from the driver the receipt records.
NATIVE_DRIVER = "simsopt_scipy_lbfgsb_manual_lm"
JAX_DRIVER = "simsopt_jax_lbfgsb_manual_ls"
#: Which status vocabulary each lane's FIRST stage emits.  The native lane
#: reaches ``scipy.optimize.minimize(method='L-BFGS-B')``
#: (``simsopt/geo/boozersurface.py:622``); the JAX lane runs the private
#: on-device L-BFGS-B port, because ``boozer_official_options`` selects
#: ``Driver.SIMSOPT_LBFGSB`` -> ``optimizer_backend='ondevice'`` -> method
#: ``lbfgs-ondevice``, whose public status is
#: ``lbfgsb_public_status_from_state``.  The vocabularies themselves are owned
#: by :mod:`simsopt_contracts.optimization_endpoint`.
FIRST_STAGE_STATUS_CONVENTION_BY_DRIVER: Final[Mapping[str, StatusConvention]] = (
    MappingProxyType(
        {
            NATIVE_DRIVER: "scipy-lbfgsb",
            JAX_DRIVER: "private-lbfgsb",
        }
    )
)
#: The end-state keys judged against upstream's own end states (``official_scatter_contracts``):
#: every area and flux float the workflow compares lane against lane.  Which Boozer surface the
#: workflow lands on is not a function of its input -- the first stage stops at its iteration cap,
#: unconverged, and upstream's own script reaches several surfaces from one-ulp starts -- so these
#: keys are matched against upstream's draws, while ``solver_success``, the persistence flags and
#: every construction, initial and first-stage route stay lane against lane.
END_STATE_OBSERVABLES: Final[tuple[str, ...]] = (
    "area:G",
    "area:iota",
    "area:label",
    "area:residual_norm",
    "flux:G",
    "flux:iota",
    "flux:label",
    "flux:residual_norm",
    "flux:target",
    "flux:surface_dofs",
)
#: Integer code of each normalized stopping reason, for publication as a parity
#: observable: the arbiter compares numeric arrays only (it calls
#: ``np.isfinite`` on every required value), so the classification travels as
#: its index in the contract's own ``StoppingReason`` literal.  The vocabulary
#: is not restated here -- it is read from the contract.
STOPPING_REASON_CODES: Final[Mapping[StoppingReason, int]] = MappingProxyType(
    {reason: index for index, reason in enumerate(get_args(StoppingReason))}
)


def first_stage_stopping_reason(
    outcome: BoozerStageOutcome,
    *,
    max_iterations: int,
    status_convention: StatusConvention,
) -> StoppingReason:
    """Classify one lane's first stage into the contract's shared vocabulary.

    The raw status integers of the two lanes are drawn from two different
    alphabets, so only this classification is comparable across them.  The
    gradient norms :func:`certify_optimization_endpoint` takes feed its
    stationarity fields, which a stopping reason does not read; the endpoint's
    finiteness is passed explicitly, as ``native_qfm.py`` does for the same
    reason.

    ``outcome`` is an L-BFGS stage outcome, the only route that reports a
    status and an iteration count; the manual Levenberg-Marquardt stages report
    neither and are not classified here.
    """
    endpoint_finite = bool(
        np.all(np.isfinite(outcome.state.surface_dofs))
        and np.isfinite(outcome.state.iota)
        and np.isfinite(outcome.state.G)
        and np.isfinite(float(outcome.objective))
        and np.isfinite(outcome.gradient_norm)
    )
    return certify_optimization_endpoint(
        status_convention=status_convention,
        provider_success=outcome.success,
        provider_status=outcome.status,
        iterations=int(outcome.nit),
        max_iterations=max_iterations,
        initial_gradient_inf_norm=0.0,
        final_gradient_inf_norm=0.0,
        parameters_finite=endpoint_finite,
        observables_finite=endpoint_finite,
        inner_success=True,
    ).stopping_reason


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "mpol": OFFICIAL_SURFACE_RESOLUTION if native_scale else 2,
        "ntor": OFFICIAL_SURFACE_RESOLUTION if native_scale else 2,
        "native_bfgs_maxiter": OFFICIAL_LBFGS_MAXITER,
        "native_ls_maxiter": OFFICIAL_LS_MAXITER,
        # The JAX lane runs upstream's own first-stage budget at
        # ``native_default``, where the exactly comparable first-stage facts
        # (status, nit, solver_success) are therefore compared lane against
        # lane.  The bounded scale keeps a reduced JAX budget, and that is a
        # measured constraint, not a saving: with a shared 300-iteration first
        # stage at mpol=ntor=2 the JAX lane ends at iota -0.2109 and stage two
        # then converges onto a DIFFERENT Boozer branch than the native lane
        # (area:iota -0.1938 versus -0.41398), i.e. the cheap scale cannot
        # carry the official budget without changing which surface it solves.
        "jax_bfgs_maxiter": OFFICIAL_LBFGS_MAXITER if native_scale else 60,
        "jax_ls_maxiter": OFFICIAL_LS_MAXITER,
        "solver_tolerance": OFFICIAL_SOLVER_TOLERANCE,
        "constraint_weight": OFFICIAL_CONSTRAINT_WEIGHT,
        "initial_iota": OFFICIAL_INITIAL_IOTA,
        "surface_distance": OFFICIAL_SURFACE_DISTANCE,
        "flux_multiplier": OFFICIAL_FLUX_MULTIPLIER,
    }


def _configuration_int(configuration: Mapping[str, object], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(configuration: Mapping[str, object], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _problem(configuration: Mapping[str, object]):
    from simsopt.configs import get_data
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceXYZTensorFourier

    _, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    field = BiotSavart(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    mpol = _configuration_int(configuration, "mpol")
    ntor = _configuration_int(configuration, "ntor")
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / nfp,
            2 * ntor + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * mpol + 1,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(
        magnetic_axis,
        _configuration_float(configuration, "surface_distance"),
        flip_theta=True,
    )
    return magnetic_axis, native_field, field, surface, float(G0)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze NCSX and initial-surface state for every solver lane."""
    configuration = _scale_configuration(scale)
    magnetic_axis, native_field, _, surface, G0 = _problem(configuration)
    return create_input_bundle(
        root,
        case_id="native-boozer",
        random_seed=0,
        arrays={
            "axis_dofs": np.asarray(
                magnetic_axis.local_full_x,
                dtype=np.float64,
            ),
            "field_dofs": np.asarray(native_field.x, dtype=np.float64),
            "surface_dofs": np.asarray(surface.get_dofs(), dtype=np.float64),
        },
        configuration={**configuration, "initial_G": G0},
        scale=scale,
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _effective_fingerprint(
    bundle: InputBundle,
    axis_dofs: np.ndarray,
    field_dofs: np.ndarray,
    surface_dofs: np.ndarray,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "axis_dofs": _array_digest(axis_dofs),
            "field_dofs": _array_digest(field_dofs),
            "surface_dofs": _array_digest(surface_dofs),
            **bundle.configuration,
        },
    )


def _plain_boozer_residual(surface, iota: float, G: float, field, *, derivatives: int):
    """Official plain Boozer residual: unscaled, unweighted, no constraint rows."""
    from simsopt.geo import boozer_surface_residual

    return boozer_surface_residual(
        surface,
        iota,
        G,
        field,
        derivatives=derivatives,
    )


def _residual_norm(surface, iota: float, G: float, field) -> float:
    residual = _plain_boozer_residual(surface, iota, G, field, derivatives=0)[0]
    return float(np.linalg.norm(np.asarray(residual, dtype=np.float64)))


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt.geo import Area, BoozerSurface, ToroidalFlux

    magnetic_axis, native_field, field, surface, G0 = _problem(bundle.configuration)
    initial_iota = _configuration_float(bundle.configuration, "initial_iota")
    tolerance = _configuration_float(bundle.configuration, "solver_tolerance")
    constraint_weight = _configuration_float(bundle.configuration, "constraint_weight")
    start = BoozerStageState(
        surface_dofs=np.asarray(arrays["surface_dofs"], dtype=np.float64),
        iota=initial_iota,
        G=G0,
    )
    area = Area(surface)
    solver = BoozerSurface(
        native_field,
        surface,
        area,
        float(area.J()),
    )
    initial_x = np.concatenate(
        (arrays["surface_dofs"], np.asarray([initial_iota, G0], dtype=np.float64))
    )
    initial_residual, initial_jacobian = solver._get_residual_vector_and_jacobian(
        initial_x,
        constraint_weight,
        True,
        True,
    )
    initial_boozer_residual, initial_boozer_jacobian = _plain_boozer_residual(
        surface,
        initial_iota,
        G0,
        native_field,
        derivatives=1,
    )
    initial_residual_norm = _residual_norm(
        surface,
        initial_iota,
        G0,
        native_field,
    )
    rough = run_boozer_lbfgs_stage(
        solver,
        start,
        tol=tolerance,
        maxiter=_configuration_int(bundle.configuration, "native_bfgs_maxiter"),
        constraint_weight=constraint_weight,
    )
    rough_label = float(area.J())
    rough_residual_norm = _residual_norm(
        surface,
        rough.state.iota,
        rough.state.G,
        native_field,
    )
    polished = run_boozer_manual_stage(
        solver,
        rough.state,
        tol=tolerance,
        maxiter=_configuration_int(bundle.configuration, "native_ls_maxiter"),
        constraint_weight=constraint_weight,
    )
    area_label = float(area.J())
    area_residual_norm = _residual_norm(
        surface,
        polished.state.iota,
        polished.state.G,
        native_field,
    )

    toroidal_flux = ToroidalFlux(surface, field)
    flux_target = _configuration_float(bundle.configuration, "flux_multiplier") * float(
        toroidal_flux.J()
    )
    flux_solver = BoozerSurface(
        native_field,
        surface,
        toroidal_flux,
        flux_target,
    )
    expanded = run_boozer_manual_stage(
        flux_solver,
        polished.state,
        tol=tolerance,
        maxiter=_configuration_int(bundle.configuration, "native_ls_maxiter"),
        constraint_weight=constraint_weight,
    )
    values = _values(
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_surface_dofs=arrays["surface_dofs"],
        initial_residual=np.asarray(initial_residual, dtype=np.float64),
        initial_jacobian=np.asarray(initial_jacobian, dtype=np.float64),
        initial_boozer_residual=np.asarray(initial_boozer_residual, dtype=np.float64),
        initial_boozer_jacobian=np.asarray(initial_boozer_jacobian, dtype=np.float64),
        initial_residual_norm=initial_residual_norm,
        driver=NATIVE_DRIVER,
        first_stage_max_iterations=_configuration_int(
            bundle.configuration, "native_bfgs_maxiter"
        ),
        rough=rough,
        rough_label=rough_label,
        rough_residual_norm=rough_residual_norm,
        area=polished,
        area_label=area_label,
        area_residual_norm=area_residual_norm,
        flux=expanded,
        flux_target=flux_target,
        flux_label=float(toroidal_flux.J()),
        flux_residual_norm=_residual_norm(
            surface,
            expanded.state.iota,
            expanded.state.G,
            native_field,
        ),
        flux_surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
    )
    return _observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver=NATIVE_DRIVER,
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt.geo import Area, ToroidalFlux
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

    import jax
    import jax.numpy as jnp

    magnetic_axis, native_field, field, surface, G0 = _problem(bundle.configuration)
    initial_iota = _configuration_float(bundle.configuration, "initial_iota")
    tolerance = _configuration_float(bundle.configuration, "solver_tolerance")
    constraint_weight = _configuration_float(bundle.configuration, "constraint_weight")
    start = BoozerStageState(
        surface_dofs=np.asarray(arrays["surface_dofs"], dtype=np.float64),
        iota=initial_iota,
        G=G0,
    )
    jax_field = BiotSavartJAX(native_field.coils)
    options = boozer_official_options(
        rough_maxiter=_configuration_int(bundle.configuration, "jax_bfgs_maxiter"),
        ls_maxiter=_configuration_int(bundle.configuration, "jax_ls_maxiter"),
        tolerance=tolerance,
    )
    area = Area(surface)
    solver = BoozerSurfaceJAX(
        jax_field,
        surface,
        area,
        float(area.J()),
        constraint_weight=constraint_weight,
        options=options,
    )
    initial_x = jnp.concatenate(
        (
            jax.device_put(np.asarray(arrays["surface_dofs"], dtype=np.float64)),
            jax.device_put(np.asarray([initial_iota, G0], dtype=np.float64)),
        )
    )
    kernels = solver._get_penalty_kernel_bundle(
        optimize_G=True,
        weight_inv_modB=True,
        constraint_weight=constraint_weight,
    )
    coil_spec = jax_field.coil_set_spec()
    initial_residual_device, initial_jacobian_device = (
        kernels.residual(initial_x, coil_spec),
        kernels.jacobian(initial_x, coil_spec),
    )
    initial_residual, initial_jacobian = jax.device_get(
        (initial_residual_device, initial_jacobian_device)
    )
    initial_boozer_residual, initial_boozer_jacobian = _plain_boozer_residual(
        surface,
        initial_iota,
        G0,
        native_field,
        derivatives=1,
    )
    initial_residual_norm = _residual_norm(
        surface,
        initial_iota,
        G0,
        native_field,
    )

    rough = run_boozer_lbfgs_stage(
        solver,
        start,
        tol=tolerance,
        maxiter=_configuration_int(bundle.configuration, "jax_bfgs_maxiter"),
        constraint_weight=constraint_weight,
    )
    rough_label = float(area.J())
    rough_residual_norm = _residual_norm(
        surface,
        rough.state.iota,
        rough.state.G,
        native_field,
    )
    polished = run_boozer_manual_stage(
        solver,
        rough.state,
        tol=tolerance,
        maxiter=_configuration_int(bundle.configuration, "jax_ls_maxiter"),
        constraint_weight=constraint_weight,
    )
    area_label = float(area.J())
    area_residual_norm = _residual_norm(
        surface,
        polished.state.iota,
        polished.state.G,
        native_field,
    )

    toroidal_flux = ToroidalFlux(surface, field)
    flux_target = _configuration_float(bundle.configuration, "flux_multiplier") * float(
        toroidal_flux.J()
    )
    flux_field = BiotSavartJAX(native_field.coils)
    flux_solver = BoozerSurfaceJAX(
        flux_field,
        surface,
        toroidal_flux,
        flux_target,
        constraint_weight=constraint_weight,
        options=options,
        surface_runtime_state=solver.surface_runtime_state,
    )
    expanded = run_boozer_manual_stage(
        flux_solver,
        polished.state,
        tol=tolerance,
        maxiter=_configuration_int(bundle.configuration, "jax_ls_maxiter"),
        constraint_weight=constraint_weight,
    )
    values = _values(
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_surface_dofs=arrays["surface_dofs"],
        initial_residual=np.asarray(initial_residual, dtype=np.float64),
        initial_jacobian=np.asarray(initial_jacobian, dtype=np.float64),
        initial_boozer_residual=np.asarray(initial_boozer_residual, dtype=np.float64),
        initial_boozer_jacobian=np.asarray(initial_boozer_jacobian, dtype=np.float64),
        initial_residual_norm=initial_residual_norm,
        driver=JAX_DRIVER,
        first_stage_max_iterations=_configuration_int(
            bundle.configuration, "jax_bfgs_maxiter"
        ),
        rough=rough,
        rough_label=rough_label,
        rough_residual_norm=rough_residual_norm,
        area=polished,
        area_label=area_label,
        area_residual_norm=area_residual_norm,
        flux=expanded,
        flux_target=flux_target,
        flux_label=float(toroidal_flux.J()),
        flux_residual_norm=_residual_norm(
            surface,
            expanded.state.iota,
            expanded.state.G,
            native_field,
        ),
        flux_surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver=JAX_DRIVER,
    )


def _values(
    *,
    axis_dofs: np.ndarray,
    field_dofs: np.ndarray,
    initial_surface_dofs: np.ndarray,
    initial_residual: np.ndarray,
    initial_jacobian: np.ndarray,
    initial_boozer_residual: np.ndarray,
    initial_boozer_jacobian: np.ndarray,
    initial_residual_norm: float,
    driver: str,
    first_stage_max_iterations: int,
    rough: BoozerStageOutcome,
    rough_label: float,
    rough_residual_norm: float,
    area: BoozerStageOutcome,
    area_label: float,
    area_residual_norm: float,
    flux: BoozerStageOutcome,
    flux_target: float,
    flux_label: float,
    flux_residual_norm: float,
    flux_surface_dofs: np.ndarray,
) -> dict[str, np.ndarray]:
    first_stopping_reason = first_stage_stopping_reason(
        rough,
        max_iterations=first_stage_max_iterations,
        status_convention=FIRST_STAGE_STATUS_CONVENTION_BY_DRIVER[driver],
    )
    return {
        "construction:axis_dofs": axis_dofs,
        "construction:field_dofs": field_dofs,
        "initial:surface_dofs": initial_surface_dofs,
        "initial:residual": initial_residual,
        "initial:jacobian": initial_jacobian,
        "initial:boozer_residual": initial_boozer_residual,
        "initial:boozer_jacobian": initial_boozer_jacobian,
        "initial:residual_norm": np.asarray(initial_residual_norm, dtype=np.float64),
        "first:surface_dofs": rough.state.surface_dofs,
        "first:iota": np.asarray(rough.state.iota, dtype=np.float64),
        "first:G": np.asarray(rough.state.G, dtype=np.float64),
        "first:label": np.asarray(rough_label, dtype=np.float64),
        "first:residual_norm": np.asarray(rough_residual_norm, dtype=np.float64),
        "first:objective": np.asarray(rough.objective, dtype=np.float64),
        "first:gradient_norm": np.asarray(rough.gradient_norm, dtype=np.float64),
        "first:stopping_reason_code": np.asarray(
            STOPPING_REASON_CODES[first_stopping_reason],
            dtype=np.int64,
        ),
        "first:status": np.asarray(rough.status, dtype=np.int64),
        "first:nit": np.asarray(rough.nit, dtype=np.int64),
        "first:nfev": np.asarray(rough.nfev, dtype=np.int64),
        "first:njev": np.asarray(rough.njev, dtype=np.int64),
        "first:solver_success": np.asarray(rough.success, dtype=np.bool_),
        "first:provider_persisted_iterate": np.asarray(
            rough.provider_persisted_iterate,
            dtype=np.bool_,
        ),
        "area:iota": np.asarray(area.state.iota, dtype=np.float64),
        "area:G": np.asarray(area.state.G, dtype=np.float64),
        "area:label": np.asarray(area_label, dtype=np.float64),
        "area:residual_norm": np.asarray(area_residual_norm, dtype=np.float64),
        "area:solver_success": np.asarray(area.success, dtype=np.bool_),
        "area:provider_persisted_iterate": np.asarray(
            area.provider_persisted_iterate,
            dtype=np.bool_,
        ),
        "flux:target": np.asarray(flux_target, dtype=np.float64),
        "flux:iota": np.asarray(flux.state.iota, dtype=np.float64),
        "flux:G": np.asarray(flux.state.G, dtype=np.float64),
        "flux:label": np.asarray(flux_label, dtype=np.float64),
        "flux:residual_norm": np.asarray(flux_residual_norm, dtype=np.float64),
        "flux:surface_dofs": flux_surface_dofs,
        "flux:solver_success": np.asarray(flux.success, dtype=np.bool_),
        "flux:provider_persisted_iterate": np.asarray(
            flux.provider_persisted_iterate,
            dtype=np.bool_,
        ),
    }


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    values: dict[str, np.ndarray],
    *,
    platform: str,
    precision: str,
    driver: str,
) -> LaneObservation:
    success = bool(
        bool(values["area:solver_success"])
        and bool(values["flux:solver_success"])
        and np.all(np.isfinite(values["flux:surface_dofs"]))
        and np.isfinite(float(values["flux:residual_norm"]))
        and float(values["flux:residual_norm"]) < float(values["initial:residual_norm"])
        and np.isfinite(float(values["flux:iota"]))
        and np.isfinite(float(values["flux:G"]))
    )
    return LaneObservation(
        lane=lane,
        backend_mode=(
            "native_cpu" if lane == "native-cpu" else os.environ["SIMSOPT_BACKEND_MODE"]
        ),
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle,
            values["construction:axis_dofs"],
            values["construction:field_dofs"],
            values["initial:surface_dofs"],
        ),
        driver=driver,
        normalized_status="converged" if success else "failed",
        raw_status=(
            f"area={bool(values['area:solver_success'])};"
            f"flux={bool(values['flux:solver_success'])}"
        ),
        success=success,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact three-stage Boozer-surface workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
