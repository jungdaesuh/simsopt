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
    DEFAULT_STAGE2_BS_PATH,
)
from benchmarks.single_stage_smoke_fixture import (
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


def build_matched_same_state_fixture_pair(
    *,
    plasma_surf_filename=DEFAULT_PLASMA_SURF_FILENAME,
    equilibrium_path=None,
    equilibria_dir=DEFAULT_EQUILIBRIA_DIR,
    stage2_bs_path=DEFAULT_STAGE2_BS_PATH,
    nphi=DEFAULT_SMOKE_NPHI,
    ntheta=DEFAULT_SMOKE_NTHETA,
    mpol=DEFAULT_SMOKE_MPOL,
    ntor=DEFAULT_SMOKE_NTOR,
    vol_target=DEFAULT_VOL_TARGET,
    iota_target=DEFAULT_IOTA_TARGET,
    num_tf_coils=DEFAULT_NUM_TF_COILS,
    jax_optimizer_backend="ondevice",
):
    """Build CPU + JAX fixtures on the *identical* resolved Boozer state.

    The native CPU lane solves the Boozer surface fresh from the given coils /
    equilibrium / config, then the JAX lane's Boozer solve is seeded with the CPU
    lane's converged surface DOFs / iota / G so both objectives evaluate at the
    same converged state (same-state parity, not two independent solves).

    With all arguments at their defaults this reproduces the smoke-config pair the
    cross-backend parity test uses.  For an autoresearch seed, pass the seed's
    config (``stage2_bs_path`` / equilibrium / ``mpol`` / ``ntor`` / grid / targets
    / ``num_tf_coils``); the Boozer surface resolution is a free choice (a uniform
    moderate ``mpol`` keeps the jax-CPU lane tractable).  Ultra-low-|iota| configs
    whose fresh Boozer solve does not converge are out of scope for this matrix.
    """
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
    )
    cpu_boozer = cpu_fixture["boozer_surface"]
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
        boozer_surface_dofs_override=np.asarray(
            cpu_boozer.surface.get_dofs(), dtype=float
        ),
        boozer_iota_override=float(cpu_result["iota"]),
        boozer_G_override=float(cpu_result["G"]),
    )
    return cpu_fixture, jax_fixture
