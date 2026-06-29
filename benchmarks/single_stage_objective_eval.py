"""SSOT helpers to assemble and matched-state-pair the single-stage banana objective.

Both the 1-seed cross-backend parity *test*
(``tests/integration/test_single_stage_objective_parity.py``) and the N-seed
parity *matrix* (``benchmarks/single_stage_objective_parity_matrix.py``) build the
full 8-term single-stage objective ``JF`` through the example's SSOT factory
(``select_single_stage_objective_backend`` + ``build_single_stage_objective``) and
need an identical matched-state fixture pair (one native-C++ lane, one JAX lane
evaluated at the *same* resolved Boozer state).  Keeping that glue in one module
means the test and the matrix exercise byte-identical objective definitions and
the same matched-state contract -- no parallel copy that can drift.

The example module is imported lazily inside each function (not at module top) so
importing this module stays cheap and matches the deferred-but-static import
convention used by ``single_stage_smoke_fixture`` -- it never uses ``importlib``
or dynamic ``import``.
"""

from __future__ import annotations

import sys

import numpy as np

from benchmarks.single_stage_smoke_defaults import (
    DEFAULT_PLASMA_SURF_FILENAME,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_CONSTRAINT_WEIGHT,
    DEFAULT_IOTA_TARGET,
    DEFAULT_NUM_TF_COILS,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_VOL_TARGET,
    build_real_single_stage_init_fixture,
)
from examples.single_stage_optimization.equilibria_paths import (
    DEFAULT_EQUILIBRIA_DIR,
)

# ``main()`` selects the residual kind via ``boozer_type[stage]`` where
# ``boozer_type = {"initial": "least_squares", "final": "exact"}`` and the default
# ``--boozer-stage`` is "initial".  "least_squares" is the constraint_weight=1.0 LS
# Boozer surface the smoke fixture builds; ``select_boozer_residual_class`` only
# special-cases the literal "exact", so this is the correct kind for the fixture.
DEFAULT_BOOZER_KIND = "least_squares"


def objective_weights_from_example_defaults():
    """Build ``SingleStageObjectiveWeights`` from the example's argparse defaults.

    Mirrors ``main()``'s weight assembly (including the ``max``/``min`` clamps
    against the module's hardware-minimum constants) so the objective under test is
    the realistic production objective rather than an ad-hoc one.  Deriving the
    weights from the shared defaults keeps callers drift-proof if the example's
    defaults change.
    """
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    argv_backup = sys.argv
    try:
        sys.argv = ["single_stage_banana_example.py"]
        args = single_stage_example.parse_args()
    finally:
        sys.argv = argv_backup

    cc_dist = max(args.cc_dist, single_stage_example.COIL_COIL_MIN_DIST_M)
    cs_dist = max(args.cs_dist, single_stage_example.COIL_PLASMA_MIN_DIST_M)
    ss_dist = max(args.ss_dist, single_stage_example.PLASMA_VESSEL_MIN_DIST_M)
    length_target = min(
        float(args.length_target), single_stage_example.COIL_LENGTH_TARGET_M
    )
    return single_stage_example.SingleStageObjectiveWeights(
        non_qs=args.non_qs_weight,
        res=args.res_weight,
        iotas=args.iotas_weight,
        length=args.length_weight,
        cc=args.cc_weight,
        cs=args.cs_weight,
        surf_dist=args.surf_dist_weight,
        curvature=args.curvature_weight,
        cc_dist=cc_dist,
        cs_dist=cs_dist,
        ss_dist=ss_dist,
        curvature_threshold=args.curvature_threshold,
        length_target=length_target,
    )


def build_lane_objective(
    fixture,
    *,
    use_jax,
    num_tf_coils=DEFAULT_NUM_TF_COILS,
    boozer_kind=DEFAULT_BOOZER_KIND,
    weights=None,
):
    """Assemble the SSOT single-stage objective for one backend lane.

    Uses exactly the factory entry points ``main()`` uses so the native and JAX
    lanes build byte-identical objective *definitions*; only the term
    implementations differ via ``use_jax``.  ``weights`` defaults to the example's
    argparse defaults; pass an explicit ``SingleStageObjectiveWeights`` to reuse a
    single weights instance across both lanes of a pair (so the only difference is
    the backend, never the weights).
    """
    from examples.single_stage_optimization.SINGLE_STAGE import (
        single_stage_banana_example as single_stage_example,
    )

    bs = fixture["bs"]
    coils = bs.coils
    curves = [c.curve for c in coils]
    # First ``num_tf_coils`` coils are the fixed TF set; the remainder are the
    # banana coils whose curves drive the length / curvature penalties.
    banana_curves = [c.curve for c in coils[num_tf_coils:]]

    backend = single_stage_example.select_single_stage_objective_backend(
        use_jax=use_jax,
        boozer_kind=boozer_kind,
        # ``jax_bs`` is only consumed on the JAX lane; ``coils`` only on the native
        # lane (it builds its own ``BiotSavart(coils)``).
        jax_bs=bs if use_jax else None,
        coils=coils,
    )
    return single_stage_example.build_single_stage_objective(
        boozer_surface=fixture["boozer_surface"],
        backend=backend,
        banana_curves=banana_curves,
        curves=curves,
        vessel_surface=single_stage_example.build_single_stage_vessel_surface(),
        iota_target=fixture["iota_target"],
        weights=(
            weights if weights is not None else objective_weights_from_example_defaults()
        ),
    )


def _recover_consistent_iota_G(cpu_boozer, bs, *, num_tf_coils, iota_guess,
                               max_newton=12, tol=1e-13):
    """Recover the (iota, G) consistent with a FIXED resolved Boozer surface.

    The serialized resolved surface does not store its production (iota, G), and a
    full re-solve from ``(resolved_dofs, FINAL_IOTA, g0)`` drifts to a different
    (degenerate, self-intersecting) Boozer branch because ``g0`` (the abs-current
    seed guess) is not the production G.  But for a *fixed* surface the Boozer
    least-squares penalty is exactly quadratic in ``(iota, G)`` (the field is affine
    in G and the residual affine in iota), so a few Newton steps on the 2x2
    ``(iota, G)`` sub-block of the penalty Hessian recover the unique
    ``(iota*, G*)`` minimizing the Boozer residual at the resolved surface -- the
    pair consistent with that surface.  The surface DOFs are held at their resolved
    values throughout (only the last two entries of the dof vector are updated), so
    the recovery itself cannot move the surface.  ``(resolved_dofs, iota*, G*)`` is
    stationary in ``(iota, G)`` only -- the surface-dof gradient is generally nonzero
    -- so the subsequent full solve (which frees all DOFs) may relax the surface, but
    starting from the consistent ``(iota, G)`` it stays on the resolved Boozer branch
    rather than collapsing to a degenerate one (the caller asserts this in-place
    convergence via a post-solve iota drift check).

    Raises ``RuntimeError`` if the 2x2 Newton does not drive ``||grad_(iota,G)||``
    below ``tol`` within ``max_newton`` steps (the resolved surface is not a Boozer
    least-squares fixed point in ``(iota, G)``), rather than silently returning a
    non-stationary pair that downstream parity could not detect.
    """
    coils = bs.coils
    current_sum = sum(abs(c.current.get_value()) for c in coils[:num_tf_coils])
    g0 = float(2.0 * np.pi * current_sum * (4.0 * np.pi * 1e-7 / (2.0 * np.pi)))
    surf = cpu_boozer.surface
    weight_inv_modB = bool(cpu_boozer.options.get("weight_inv_modB", True))
    x = np.concatenate([np.asarray(surf.x, dtype=float), [float(iota_guess), g0]])
    converged = False
    for _ in range(max_newton):
        _, grad, hess = cpu_boozer.boozer_penalty_constraints_vectorized(
            x,
            derivatives=2,
            constraint_weight=cpu_boozer.constraint_weight,
            optimize_G=True,
            weight_inv_modB=weight_inv_modB,
        )
        sub_grad = np.asarray(grad, dtype=float)[-2:]
        if float(np.linalg.norm(sub_grad)) <= tol:
            converged = True
            break
        sub_hess = np.asarray(hess, dtype=float)[-2:, -2:]
        x[-2:] = x[-2:] - np.linalg.solve(sub_hess, sub_grad)
    if not converged:
        raise RuntimeError(
            f"(iota, G) recovery did not converge in {max_newton} Newton steps "
            f"(||grad_(iota,G)|| > {tol:.1e}); the resolved surface is not a "
            "Boozer least-squares fixed point in (iota, G)."
        )
    return float(x[-2]), float(x[-1])


def build_matched_same_state_fixture_pair(
    *,
    stage2_bs_path,
    plasma_surf_filename=DEFAULT_PLASMA_SURF_FILENAME,
    equilibrium_path=None,
    equilibria_dir=DEFAULT_EQUILIBRIA_DIR,
    nphi=DEFAULT_SMOKE_NPHI,
    ntheta=DEFAULT_SMOKE_NTHETA,
    mpol=DEFAULT_SMOKE_MPOL,
    ntor=DEFAULT_SMOKE_NTOR,
    vol_target=DEFAULT_VOL_TARGET,
    iota_target=DEFAULT_IOTA_TARGET,
    num_tf_coils=DEFAULT_NUM_TF_COILS,
    constraint_weight=DEFAULT_CONSTRAINT_WEIGHT,
    jax_optimizer_backend="ondevice",
    cpu_boozer_surface_dofs_override=None,
    cpu_boozer_iota_override=None,
    cpu_boozer_G_override=None,
    recover_iota_G=False,
):
    """Build CPU + JAX fixtures on the *identical* fully Boozer-solved state.

    The native CPU lane solves the Boozer surface, then the JAX lane's Boozer solve
    is seeded with the CPU lane's converged surface DOFs / iota / G so both
    objectives evaluate at the same converged state (same-state parity, not two
    independent solves).  Both lanes solve to convergence from the *same* warm
    start, so they reach the same physical surface -- this is what makes the
    cross-boundary (field/surface-coupled) terms agree to machine precision.

    With all arguments at their defaults the CPU lane solves *fresh* from the
    Stage-2 seed -- the smoke-config pair the cross-backend parity test uses.

    For a production autoresearch seed neither a fresh solve nor a re-solve from the
    resolved DOFs reaches the production surface: a fresh solve at a tractable
    resolution collapses to a degenerate (near-zero-iota, self-intersecting)
    surface, and a re-solve from ``(resolved_dofs, FINAL_IOTA, g0)`` drifts to a
    different Boozer branch because the production ``G`` is not stored and ``g0`` is
    only the seed guess.  Pass ``recover_iota_G=True`` with the seed's resolved
    surface DOFs (``cpu_boozer_surface_dofs_override``) and resolved iota
    (``cpu_boozer_iota_override``): the CPU lane installs the resolved surface
    value-only (no drift), recovers the consistent ``(iota*, G*)`` for it
    (:func:`_recover_consistent_iota_G`), then full-solves from
    ``(resolved_dofs, iota*, G*)`` -- stationary in ``(iota, G)`` -- so the full solve
    relaxes any residual surface gradient on the resolved Boozer branch (asserted via
    a post-solve iota drift check) rather than collapsing.  The JAX lane then
    full-solves from the CPU lane's converged state, reaching the same surface.
    """
    if recover_iota_G and cpu_boozer_surface_dofs_override is None:
        raise ValueError(
            "recover_iota_G requires cpu_boozer_surface_dofs_override "
            "(the seed's resolved Boozer surface DOFs)."
        )
    # Single fixture build for both paths.  ``reuse_resolved_warm_start_solve``
    # installs the resolved surface value-only (no nonlinear solve -> no drift) when
    # recovering; otherwise it is a no-op and the CPU lane solves fresh.
    cpu_fixture = build_real_single_stage_init_fixture(
        backend="cpu",
        optimizer_backend="scipy",
        plasma_surf_filename=plasma_surf_filename,
        equilibrium_path=equilibrium_path,
        equilibria_dir=equilibria_dir,
        stage2_bs_path=stage2_bs_path,
        nphi=nphi,
        ntheta=ntheta,
        mpol=mpol,
        ntor=ntor,
        vol_target=vol_target,
        iota_target=iota_target,
        num_tf_coils=num_tf_coils,
        constraint_weight=constraint_weight,
        boozer_surface_dofs_override=cpu_boozer_surface_dofs_override,
        boozer_iota_override=cpu_boozer_iota_override,
        boozer_G_override=cpu_boozer_G_override,
        reuse_resolved_warm_start_solve=recover_iota_G,
    )
    cpu_boozer = cpu_fixture["boozer_surface"]
    if recover_iota_G:
        # Recover (iota*, G*) consistent with the fixed resolved surface, then
        # full-solve from the (iota, G)-stationary state (relaxes on the resolved
        # branch; the in-place convergence is asserted just below).
        iota_star, G_star = _recover_consistent_iota_G(
            cpu_boozer,
            cpu_fixture["bs"],
            num_tf_coils=num_tf_coils,
            iota_guess=(
                cpu_boozer_iota_override
                if cpu_boozer_iota_override is not None
                else iota_target
            ),
        )
        # Independent production-state check: the (iota, G) recovery minimises the
        # Boozer penalty at the FIXED resolved surface, so iota* is that surface's
        # natural iota.  Assert it matches the seed's stored resolved iota -- a
        # wrong/drifted serialized surface would differ by O(0.1).  (Newton on the
        # quadratic (iota, G) penalty is seed-independent, so this is a genuine
        # cross-check, not a restatement of the iota guess.)
        if cpu_boozer_iota_override is not None:
            assert np.isclose(
                iota_star, float(cpu_boozer_iota_override), rtol=1e-3, atol=1e-6
            ), (
                f"recovered iota {iota_star!r} disagrees with the seed's resolved "
                f"iota {float(cpu_boozer_iota_override)!r}: the serialized surface "
                "is not the production Boozer surface."
            )
        cpu_boozer.need_to_run_code = True
        cpu_boozer.run_code(iota_star, G_star)
        # Verify the full solve converged IN PLACE: from the fixed point it must not
        # jump to another Boozer branch (which would move iota by O(0.1)).
        solved_iota = float(cpu_boozer.res["iota"])
        assert np.isclose(solved_iota, iota_star, rtol=1e-3, atol=1e-6), (
            f"the full solve drifted iota {iota_star!r} -> {solved_iota!r}: "
            "(resolved_dofs, iota*, G*) was not a Boozer fixed point."
        )

    cpu_result = cpu_boozer.res
    assert cpu_result is not None and cpu_result.get("success", False), (
        "CPU reduced-real fixture did not converge"
    )

    jax_fixture = build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend=jax_optimizer_backend,
        plasma_surf_filename=plasma_surf_filename,
        equilibrium_path=equilibrium_path,
        equilibria_dir=equilibria_dir,
        stage2_bs_path=stage2_bs_path,
        nphi=nphi,
        ntheta=ntheta,
        mpol=mpol,
        ntor=ntor,
        vol_target=vol_target,
        iota_target=iota_target,
        num_tf_coils=num_tf_coils,
        constraint_weight=constraint_weight,
        boozer_surface_dofs_override=np.asarray(
            cpu_boozer.surface.get_dofs(), dtype=float
        ),
        boozer_iota_override=float(cpu_result["iota"]),
        boozer_G_override=float(cpu_result["G"]),
    )
    return cpu_fixture, jax_fixture
