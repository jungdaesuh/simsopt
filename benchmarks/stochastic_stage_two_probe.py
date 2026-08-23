"""Matched-policy, matched-sample ``stage_two_optimization_stochastic`` A/B probe.

Plan task P1.1 (baseline) and the P1.3 re-probe hook of
``docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md``.
Diagnostic-not-certifying by construction: this script mints no speed claim.
It measures one lane per invocation and publishes one grade-stamped artifact
per leg through ``benchmarks/probe_conventions.py``; the ratio is formed by
whoever reads two artifacts, and a certified ratio needs its own charter.

What "matched" means here, and what it costs:

* **Matched sample.** Both lanes materialize the training perturbations from
  the same PCG64DXSM stream through the same public helper
  (``materialize_stochastic_coil_perturbations``) at the same configuration,
  and every artifact records the bundle ``sha256`` from
  ``src/simsopt_jax/examples/stochastic_samples.py``.  The two lanes cannot
  share one process — they need different interpreters — so byte identity is
  established by that fingerprint, and ``--compare`` refuses to compare two
  endpoints whose fingerprints (or budget/maxcor/scale) differ.
* **Matched policy.** One ``--maxcor`` for both lanes per pair (10 = mirror
  default, 400 = native default), the same iteration budget, and the same
  convergence tolerance: the native leg passes ``tol=1e-15`` to SciPy, which
  sets ``ftol=gtol=1e-15``, and the JAX leg passes those two values
  explicitly.  ``maxfun``/``maxls`` are left at their defaults, which are
  already identical (15000 / 20) on both sides.
* **Matched window.** The timed window is the ``minimize`` call and nothing
  else.  The 5-epsilon Taylor test and the 256-sample out-of-sample loop of
  the native example are outside it on both lanes — the out-of-sample bundle
  is not even materialized here.
* **Single rank.** No MPI anywhere (``MPI4PY_RC_INITIALIZE=false``);
  ``mpi_ranks`` is recorded as 1.  The native example's two-dimensional
  ranks x OMP denominator is a charter problem, not a probe problem.

Every timed leg runs as a re-exec'd child so ``OMP_NUM_THREADS`` and the JAX
platform are set *before* the interpreter starts, and the child verifies it
observed the thread count that was requested.  The child environment is
``probe_conventions.pinned_environment`` — scrub by prefix, then pin — plus the
``SIMSOPT_``/``XLA_`` selectors this probe owns.  ``simsopt.geo`` objectives are
jax-jitted, so the native leg imports JAX transitively: its child pins
``JAX_PLATFORMS=cpu``, ``JAX_ENABLE_X64=1`` and ``CUDA_VISIBLE_DEVICES=""`` so
that import can neither take a CUDA context away from a GPU leg nor silently
evaluate the native objective in fp32.

``--omp`` is mandatory for a timed leg on *both* lanes: an unset
``OMP_NUM_THREADS`` is not a neutral default, it is the collapse this tree's
ledger documents, and a leg nobody pinned has no denominator.  ``--sample-tile``
and ``--compile-cache`` are JAX-lane levers and are refused on ``--lane native``
rather than stamped into an artifact they cannot affect.  A leg is published
only if every timed solve took at least one iteration and either converged or
ran the whole budget (:data:`SOLVE_GATE`): a solve that stopped short without
converging is a broken measurement, not a fast one, and exhausting the budget —
which SciPy reports as ``success=False``, ``status=1`` — is the intended
terminal state of a fixed-budget leg rather than a failure.  Each
executed leg also appends one line to ``--ledger``, so the order the legs really
ran in is provable from the file rather than assumed from the intended
schedule.

``--output`` inside this repository must land under ``docs/receipts/evidence``;
a path entirely outside the repository is a scratch run, accepted and stamped
``published_under_evidence_root: false`` in its own artifact.  ``--endpoint-out``
must end in ``.npz`` and refuses to overwrite: the file it would clobber is the
other half of a ``--compare`` pair, written by a leg that has already run.

The native evaluator is cloned from the parity case's native lane rather than
imported: ``examples/jax/parity/cases/native_stage_two_optimization_stochastic.py``
exposes its geometry and configuration helpers (imported here, so the problem
construction has one owner) but keeps the objective assembly inside its private
``_native``.  Factoring a ``build_native_evaluator`` out of that case is the
plan's named U3 harness-clone gap; until it lands, the objective assembly below
is the clone and must be diffed against ``_native`` when either moves.

Run recipe (from the repository root; interpreters differ per lane)::

    NATIVE_PY=.venv/bin/python
    GPU_PY=.venv-qn-gpu/bin/python
    export PYTHONPATH=src:build/cp311-cp311-linux_x86_64
    export JAX_ENABLE_X64=1
    EV=docs/receipts/evidence

    # shapes and sample fingerprints, no solving
    $NATIVE_PY benchmarks/stochastic_stage_two_probe.py --dry-run

    # fair-native denominator: sweep OMP, one artifact per thread count
    for omp in 2 4 8 16 32 48; do
      $NATIVE_PY benchmarks/stochastic_stage_two_probe.py \
        --lane native --maxcor 400 --budget 400 --omp $omp \
        --output $EV/stochastic_p11_native_omp${omp}_maxcor400.json \
        --endpoint-out $EV/stochastic_p11_native_omp${omp}_maxcor400.npz
    done

    # GPU lane, warm persistent cache; drop --compile-cache for the cold leg
    $GPU_PY benchmarks/stochastic_stage_two_probe.py \
      --lane jax-gpu --maxcor 400 --budget 400 --omp 8 \
      --compile-cache /tmp/stochastic_probe_cache \
      --output $EV/stochastic_p11_jaxgpu_maxcor400.json \
      --endpoint-out $EV/stochastic_p11_jaxgpu_maxcor400.npz

    # P1.3 re-probe: the P1.1 GPU leg unchanged, plus the sample-axis tile
    $GPU_PY benchmarks/stochastic_stage_two_probe.py \
      --lane jax-gpu --maxcor 400 --budget 400 --omp 8 --sample-tile 8 \
      --compile-cache /tmp/stochastic_probe_cache \
      --output $EV/stochastic_p13_jaxgpu_tile8_maxcor400.json \
      --endpoint-out $EV/stochastic_p13_jaxgpu_tile8_maxcor400.npz

    # endpoint agreement, native_workflow tolerance bucket
    $NATIVE_PY benchmarks/stochastic_stage_two_probe.py --compare \
      $EV/stochastic_p11_native_omp32_maxcor400.npz \
      $EV/stochastic_p11_jaxgpu_maxcor400.npz

Alternate the two lanes pair by pair (``probe_conventions.interleave_schedule``
prints the order under ``--dry-run``); position bias in a ratio is otherwise
indistinguishable from the ratio.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.probe_conventions import (
    COMPILATION_CACHE_VARIABLE,
    OMP_SWEEP,
    SCRUBBED_ENVIRONMENT_PREFIXES,
    THREAD_COUNT_VARIABLES,
    ProbeConventionError,
    append_leg_ledger,
    array_sha256,
    interleave_schedule,
    observed_openmp_threads,
    pinned_environment,
    runtime_identity,
    write_probe_artifact,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from simsopt.geo import Curve, GaussianSampler, SurfaceRZFourier
    from simsopt_jax.examples import (
        ExecutionScale,
        StochasticPerturbationBundle,
        StochasticStageTwoConfiguration,
    )

SCHEMA = "stochastic-stage-two-probe.v1"
PLAN = "docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md"
PARITY_CASE = "examples/jax/parity/cases/native_stage_two_optimization_stochastic.py"
TOLERANCE_BUCKET = "native_workflow"
PUBLICATION = (
    "Matched-policy, matched-sample, single-rank stochastic stage-two A/B "
    "probe (plan P1.1/P1.3). One lane per invocation; the timed window is the "
    "minimize call only. Diagnostic-not-certifying: not a speed claim."
)

#: SciPy's ``minimize(..., tol=)`` sets ``ftol`` and ``gtol`` for L-BFGS-B, so
#: the native example's ``tol=1e-15`` is the policy both lanes are pinned to.
MATCHED_TOLERANCE = 1.0e-15

#: The two matched policy points the plan freezes: 10 is the mirror's history
#: size, 400 is the native example's.
MATCHED_MAXCOR = (10, 400)

_LANES = ("native", "jax-gpu")

#: Probe artifacts published inside this repository live here and nowhere else.
#: A run whose ``--output`` lands entirely outside the repository is a scratch
#: run: it is allowed, and it is stamped ``published_under_evidence_root:
#: false`` in its own artifact, matching ``benchmarks/marginal_quartet_probes.py``.
EVIDENCE_ROOT = REPO_ROOT / "docs" / "receipts" / "evidence"

#: Executed order is evidence and lives in a ledger, not in this artifact: one
#: appended line per executed leg, so a degenerate interleave is visible after
#: the fact instead of assumed away by the intended schedule.
DEFAULT_LEDGER = EVIDENCE_ROOT / "probe_leg_ledger.jsonl"

#: The only suffix an endpoint may carry.  ``np.savez`` appends ``.npz`` to a
#: *path* it is handed, so any other suffix names a file ``--compare`` is never
#: given; the write itself opens ``"xb"`` on an already-suffixed path, so the
#: archive lands exactly where the operator said it would.
ENDPOINT_SUFFIX = ".npz"

#: What a reader of this artifact would otherwise have to derive from the source
#: to know what the timed number does *not* contain.  Every artifact carries
#: these; the JAX lane carries its own route disclosure on top.
_SHARED_DISCLOSURES = (
    (
        "Instrument, not example: the timed route is the builder route assembled "
        "in this file (native lane: the parity case's cloned simsopt objective "
        "under scipy.optimize.minimize; JAX lane: "
        "make_stochastic_stage_two_objective -> TraceableScalarProblem -> "
        "simsopt_jax.solve.dispatch.minimize). Neither shipped script runs here: "
        "examples/2_Intermediate/stage_two_optimization_stochastic.py and its "
        "mirror examples/jax/2_Intermediate/stage_two_optimization_stochastic.py "
        "also pay a five-epsilon Taylor test, a 256-sample out-of-sample score and "
        "VTK output, none of which are inside this probe's window."
    ),
    (
        "Shared-sample injection: both lanes are handed one bundle materialized by "
        "materialize_stochastic_coil_perturbations "
        "(src/simsopt_jax/examples/stochastic_samples.py) and recorded here by its "
        "sha256, which is what makes the two lanes' perturbations the same bytes. "
        "The shipped native example instead draws its own samples in-process from "
        "PerturbationSample(sampler, randomgen=...); the native lane here feeds "
        "that same class the materialized sample= arrays."
    ),
    "Probes measure; charters certify. Nothing in this artifact is a claim.",
)

_JAX_LANE_DISCLOSURES = (
    "SOLVE-GATE ASSUMPTION (unverified on GPU): the gate's budget-exhaustion clause assumes the fused JAX driver reports nit == budget at exhaustion, as scipy does. If it reports budget-1 the gate refuses the leg loudly (fail-closed, publishable=false in the ledger) rather than minting a number; the first GPU run must confirm the convention and this line be updated.",
    (
        "The timed JAX window calls simsopt_jax.solve.dispatch.minimize directly on "
        "the cache-marked private problem._solver_value_and_grad_fn -- the same "
        "route serial_solve_jax takes internally "
        "(src/simsopt_jax/solve/serial.py::serial_solve_jax) -- but it "
        "bypasses that function's wrapper. The two whole-objective evaluations "
        "serial_solve_jax takes around the solve and its "
        "_write_bounded_objective_log write of simsopt_<utc>.dat are therefore "
        "neither paid nor timed here, and neither is its _require_success check; "
        "this probe's own publication gate replaces that check."
    ),
)

#: Pinned beside ``--compile-cache``: at JAX's defaults a small/fast program is
#: never written to the persistent cache, so a "warm persistent cache" leg
#: would be cold in disguise.  ``0`` is the tree's unified sentinel
#: (``benchmarks/wireframe_gsco_siblings_reference_scale.py``,
#: ``benchmarks/run_jax_example_execution_mode_benchmark.py``) and means "no
#: floor": every entry qualifies regardless of size or compile time.
PERSISTENT_CACHE_THRESHOLDS = {
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
}

# Recorded in every artifact so a reader can see the runtime selectors the
# probe itself pinned without serializing the inherited shell environment.
# Every name :func:`leg_environment` pins is here: an artifact that omits a pin
# cannot prove what the leg's environment actually was, and a name that is
# pinned in one branch only (the cache thresholds, CUDA_VISIBLE_DEVICES) is
# still attested on the other, where it reads ``null`` and proves the scrub.
# ``JAX_TRANSFER_GUARD`` and ``XLA_FLAGS`` are pinned by nothing here and are
# attested for exactly that reason: the scrub removed them.
_ATTESTED_ENVIRONMENT_NAMES = (
    *THREAD_COUNT_VARIABLES,
    "OMP_DYNAMIC",
    "OMP_SCHEDULE",
    "JAX_PLATFORMS",
    "JAX_ENABLE_X64",
    "JAX_TRANSFER_GUARD",
    COMPILATION_CACHE_VARIABLE,
    *PERSISTENT_CACHE_THRESHOLDS,
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_PRECISION",
    "CUDA_VISIBLE_DEVICES",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_FLAGS",
    "MPI4PY_RC_INITIALIZE",
    "HWLOC_COMPONENTS",
    "PYTHONPATH",
)


@dataclass(frozen=True, slots=True)
class SharedInputs:
    """The one problem construction both lanes consume, plus its samples."""

    scale: ExecutionScale
    configuration: StochasticStageTwoConfiguration
    configuration_mapping: dict[str, object]
    surface: SurfaceRZFourier
    base_curves: list[Curve]
    coils: list[object]
    sampler: GaussianSampler
    training: StochasticPerturbationBundle
    construction_seconds: float


# ---------------------------------------------------------------------------
# Problem construction (one owner: the parity case)
# ---------------------------------------------------------------------------


def build_shared_inputs(scale: ExecutionScale) -> SharedInputs:
    """Build geometry and materialize the training perturbations once."""
    from examples.jax.parity.cases.native_stage_two_optimization_stochastic import (
        _build_geometry,
        _scale_configuration,
        _symmetry_layout,
    )
    from simsopt.geo import GaussianSampler
    from simsopt_jax.examples import (
        materialize_stochastic_coil_perturbations,
        stochastic_stage_two_configuration,
    )

    started = time.perf_counter()
    configuration = stochastic_stage_two_configuration(scale)
    mapping = _scale_configuration(scale)
    surface, base_curves, coils = _build_geometry(mapping)
    source_indices, rotations = _symmetry_layout(coils, base_curves)
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        sigma=configuration.perturbation_sigma,
        length_scale=configuration.perturbation_length_scale,
        n_derivs=1,
    )
    training = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(base_curves),
        sample_count=configuration.training_sample_count,
        seed=configuration.training_seed,
    )
    return SharedInputs(
        scale=scale,
        configuration=configuration,
        configuration_mapping=mapping,
        surface=surface,
        base_curves=base_curves,
        coils=coils,
        sampler=sampler,
        training=training,
        construction_seconds=time.perf_counter() - started,
    )


def sample_identity(shared: SharedInputs) -> dict[str, object]:
    """The matched-sample gate: the bundle fingerprint and how it was drawn."""
    training = shared.training
    return {
        "training_sha256": training.sha256,
        "training_gamma_shape": list(training.gamma.shape),
        "training_gammadash_shape": list(training.gammadash.shape),
        "generator": training.generator,
        "ordering": training.ordering,
        "dtype": training.dtype,
        "byte_order": training.byte_order,
        "seed": int(training.seed),
        "sample_count": int(training.sample_count),
        "out_of_sample_materialized": False,
    }


def workload_shape(shared: SharedInputs) -> dict[str, int]:
    """Per-objective-eval work, the quantity the mechanism law is stated in."""
    configuration = shared.configuration
    surface_points = configuration.surface_nphi * configuration.surface_ntheta
    coil_count = len(shared.coils)
    return {
        "surface_points": surface_points,
        "coil_count": coil_count,
        "curve_quadrature": configuration.curve_quadrature,
        "training_samples": configuration.training_sample_count,
        "pair_evals_per_objective_eval": (
            surface_points
            * coil_count
            * configuration.curve_quadrature
            * configuration.training_sample_count
        ),
    }


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolveRow:
    """One timed solve: the published row, and the unit the publication gate reads.

    ``kind`` splits first-touch from steady state; ``success``/``status``/``nit``
    are the solver's own verdict, which :func:`refused_solve_rows` reads before
    anything is published.
    """

    index: int
    kind: str
    seconds: float
    nit: int
    nfev: int
    njev: int
    objective: float
    status: int
    success: bool
    message: str


def _solve_row(
    index: int,
    *,
    seconds: float,
    nit: int,
    nfev: int,
    njev: int,
    objective: float,
    status: int,
    success: bool,
    message: str,
) -> SolveRow:
    return SolveRow(
        index=index,
        # Index 0 is the first solve in this process: it carries first-touch
        # cost on both lanes (BLAS warmup natively, trace+lowering+compile on
        # the JAX lane).  Warm and cold never fold into one number.
        kind="cold_in_process" if index == 0 else "warm",
        seconds=seconds,
        nit=nit,
        nfev=nfev,
        njev=njev,
        objective=objective,
        status=status,
        success=success,
        message=message,
    )


#: What a timed row must satisfy to be publishable, stated once here and
#: stamped into every artifact so a reader gates the same way this file does.
SOLVE_GATE = (
    "every timed solve must report nit >= 1 and must have either converged "
    "(success) or run the whole --budget (nit >= budget); a solve that stopped "
    "short without converging timed less work than the leg claims"
)


def refused_solve_rows(rows: Sequence[SolveRow], *, budget: int) -> list[SolveRow]:
    """The timed solves that make a leg unpublishable, in order.

    Two ways a row is not a measurement of what this probe claims to time.  A
    solve with ``nit < 1`` produced a wall clock for no optimization at all.  A
    solve that stopped *before* the budget without converging — an abnormal
    line-search termination, a nonfinite gradient, a step the driver refused —
    did less work than the matched policy bought, and publishing its seconds
    reports a broken leg as a fast one.

    ``success`` alone cannot be the test: both lanes run a fixed iteration
    budget, and exhausting it is the *intended* terminal state, reported by
    SciPy L-BFGS-B as ``success=False`` with ``status=1`` ("TOTAL NO. OF
    ITERATIONS REACHED LIMIT").  The lane-neutral condition is therefore
    "converged, or ran every iteration it was given" — no status code and no
    message string is parsed, so the same rule holds for both drivers.
    """
    return [
        row
        for row in rows
        if row.nit < 1 or (row.success is not True and row.nit < budget)
    ]


def run_native_leg(
    shared: SharedInputs, *, budget: int, maxcor: int, repeat: int
) -> tuple[list[SolveRow], dict[str, object]]:
    """SciPy L-BFGS-B over the simsopt objective; timed window = ``minimize``.

    Returns the timed rows — which the caller gates and serializes — and the
    rest of the leg payload.
    """
    import numpy as np
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart, Coil
    from simsopt.geo import (
        ArclengthVariation,
        CurveCurveDistance,
        CurveLength,
        CurvePerturbed,
        LpCurveCurvature,
        MeanSquaredCurvature,
        PerturbationSample,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    configuration = shared.configuration
    surface = shared.surface
    base_curves = shared.base_curves
    coils = shared.coils

    construction_start = time.perf_counter()
    training_fluxes = []
    for gamma_sample, gammadash_sample in zip(
        shared.training.gamma,
        shared.training.gammadash,
        strict=True,
    ):
        perturbed_coils = [
            Coil(
                CurvePerturbed(
                    coil.curve,
                    PerturbationSample(
                        shared.sampler,
                        sample=[gamma_perturbation, gammadash_perturbation],
                    ),
                ),
                coil.current,
            )
            for coil, gamma_perturbation, gammadash_perturbation in zip(
                coils,
                gamma_sample,
                gammadash_sample,
                strict=True,
            )
        ]
        training_fluxes.append(SquaredFlux(surface, BiotSavart(perturbed_coils)))
    training_flux = sum(training_fluxes) * (1.0 / len(training_fluxes))
    lengths = [CurveLength(curve) for curve in base_curves]
    curve_curve = CurveCurveDistance(
        [coil.curve for coil in coils],
        configuration.curve_curve_threshold,
        num_basecurves=configuration.num_base_curves,
    )
    curvatures = [
        LpCurveCurvature(curve, 2, configuration.curvature_threshold)
        for curve in base_curves
    ]
    mean_squared_curvatures = [MeanSquaredCurvature(curve) for curve in base_curves]
    arclength_variations = [ArclengthVariation(curve) for curve in base_curves]
    objective = (
        training_flux
        + configuration.length_weight * sum(lengths)
        + configuration.curve_curve_weight * curve_curve
        + configuration.curvature_weight * sum(curvatures)
        + configuration.mean_squared_curvature_weight
        * sum(
            QuadraticPenalty(
                value,
                configuration.mean_squared_curvature_threshold,
                "max",
            )
            for value in mean_squared_curvatures
        )
        + configuration.arclength_variation_weight * sum(arclength_variations)
    )
    initial_parameters = np.asarray(BiotSavart(coils).x, dtype=np.float64)
    construction_seconds = time.perf_counter() - construction_start

    def value_and_gradient(parameters: NDArray[np.float64]):
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    options = {"maxiter": budget, "maxcor": maxcor}
    rows: list[SolveRow] = []
    result = None
    for index in range(repeat):
        started = time.perf_counter()
        result = minimize(
            value_and_gradient,
            initial_parameters,
            jac=True,
            method="L-BFGS-B",
            options=options,
            tol=MATCHED_TOLERANCE,
        )
        seconds = time.perf_counter() - started
        rows.append(
            _solve_row(
                index,
                seconds=seconds,
                nit=int(result.nit),
                nfev=int(result.nfev),
                njev=int(result.njev),
                objective=float(result.fun),
                status=int(result.status),
                success=bool(result.success),
                message=str(result.message),
            )
        )
        print(
            f"native solve {index} seconds={seconds:.6f} nit={result.nit}", flush=True
        )
    return rows, {
        "driver": "scipy_lbfgsb",
        "policy": {
            "maxiter": budget,
            "maxcor": maxcor,
            "tol": MATCHED_TOLERANCE,
            "scipy_options": dict(options),
            "scipy_tol_sets": ["ftol", "gtol"],
        },
        "construction_seconds": construction_seconds,
        "dof_count": int(initial_parameters.size),
        "initial_parameters": initial_parameters,
        "final_parameters": np.asarray(result.x, dtype=np.float64),
        "final_objective": float(result.fun),
    }


def run_jax_leg(
    shared: SharedInputs,
    *,
    budget: int,
    maxcor: int,
    repeat: int,
    sample_tile: int | None,
) -> tuple[list[SolveRow], dict[str, object]]:
    """The mirror's fused on-device L-BFGS-B lane; timed window = ``minimize``.

    Returns the timed rows — which the caller gates and serializes — and the
    rest of the leg payload.
    """
    import jax
    import numpy as np
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.objectives import (
        StageTwoObjectiveConfig,
        StochasticCoilPerturbations,
        make_stochastic_stage_two_objective,
    )
    from simsopt_jax.solve.dispatch import minimize
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax.solve.serial import TraceableScalarProblem
    from simsopt_jax.solve.simsopt.contracts import SimsoptLBFGSBOptions
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    configuration = shared.configuration
    construction_start = time.perf_counter()
    field = BiotSavartJAX(shared.coils)
    flux = SquaredFluxJAX(shared.surface, field)
    flux_spec = flux.fixed_surface_flux_spec()
    device = get_runtime_jax_device()
    surface_gamma = jax.device_put(
        np.asarray(shared.surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    surface_normal = jax.device_put(
        np.asarray(shared.surface.normal(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    training = StochasticCoilPerturbations(
        gamma=jax.device_put(shared.training.gamma, device),
        gammadash=jax.device_put(shared.training.gammadash, device),
    )
    objective_config = StageTwoObjectiveConfig(
        num_base_curves=configuration.num_base_curves,
        length_weight=configuration.length_weight,
        curve_curve_minimum_distance=configuration.curve_curve_threshold,
        curve_curve_weight=configuration.curve_curve_weight,
        curvature_threshold=configuration.curvature_threshold,
        curvature_weight=configuration.curvature_weight,
        mean_squared_curvature_threshold=(
            configuration.mean_squared_curvature_threshold
        ),
        mean_squared_curvature_weight=configuration.mean_squared_curvature_weight,
        arclength_variation_weight=configuration.arclength_variation_weight,
    )
    # ``sample_tile=None`` is the factory's own default (the sequential scan
    # oracle), and the factory validates the lever against this bundle's sample
    # count at build time, typed — so a bad tile fails here, before the clock.
    objective = make_stochastic_stage_two_objective(
        field,
        flux_spec,
        training,
        surface_gamma,
        surface_normal,
        objective_config,
        sample_tile=sample_tile,
    )
    initial_parameters = np.asarray(field.x, dtype=np.float64)
    initial_device = jax.device_put(initial_parameters, device)
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_device)
    jax.block_until_ready(initial_device)
    construction_seconds = time.perf_counter() - construction_start

    options = SimsoptLBFGSBOptions(
        maxiter=budget,
        maxcor=maxcor,
        ftol=MATCHED_TOLERANCE,
        gtol=MATCHED_TOLERANCE,
    )
    # The cache-marked solver callable, exactly as ``serial_solve_jax`` passes it
    # (``benchmarks/stage_two_finitebuild_native_gpu.py::_leg_jax_solve``): the
    # public bound method is unmarked, so the fused L-BFGS executable cache would
    # re-trace every solve and "warm" would silently include lowering.  No step
    # observer is attached, which is what selects ``fused_stepwise`` in
    # ``simsopt_jax.solve.dispatch._legacy_lbfgsb_options``.
    solver_value_and_grad = problem._solver_value_and_grad_fn

    def solve():
        return minimize(
            solver_value_and_grad,
            initial_device,
            driver=Driver.SIMSOPT_LBFGSB,
            options=options,
        )

    rows: list[SolveRow] = []
    result = None
    for index in range(repeat):
        started = time.perf_counter()
        # ``minimize`` returns ``result.x`` on the host, so the device queue is
        # already drained when the clock stops; the explicit block above covers
        # the staging that precedes it.
        result = solve()
        seconds = time.perf_counter() - started
        rows.append(
            _solve_row(
                index,
                seconds=seconds,
                nit=int(result.nit),
                nfev=int(result.nfev),
                njev=int(result.njev),
                objective=float(result.fun),
                status=int(result.status),
                success=bool(result.success),
                message=str(result.message),
            )
        )
        print(f"jax solve {index} seconds={seconds:.6f} nit={result.nit}", flush=True)
    devices = [
        {"platform": str(item.platform), "kind": str(item.device_kind)}
        for item in jax.local_devices()
    ]
    return rows, {
        "driver": "simsopt_lbfgsb",
        "policy": {
            **asdict(options),
            "step_observer_attached": False,
            "sample_tile": sample_tile,
        },
        "construction_seconds": construction_seconds,
        "dof_count": int(initial_parameters.size),
        "initial_parameters": initial_parameters,
        "final_parameters": np.asarray(result.x, dtype=np.float64),
        "final_objective": float(result.fun),
        "jax_devices": devices,
        "solve_device": None if device is None else str(device.platform),
    }


# ---------------------------------------------------------------------------
# Child environment
# ---------------------------------------------------------------------------


def leg_environment(
    lane: str,
    *,
    omp: int,
    jax_platforms: str,
    compile_cache: Path | None,
) -> dict[str, str]:
    """The child's environment: the shared scrub-then-pin, plus this probe's selectors.

    :func:`~benchmarks.probe_conventions.pinned_environment` owns the scrub, the
    thread-count family, the single-rank pin and ``JAX_ENABLE_X64``; everything
    added below is a selector this probe alone chooses.
    """
    if lane == "native":
        # ``simsopt.geo`` objectives are jax-jitted, so the native leg imports
        # JAX transitively; unpinned, that import can take a CUDA context from
        # a GPU leg and defaults to float32 for the jitted pieces.
        environment = pinned_environment(lane=lane, omp=omp, jax_platforms="cpu")
        environment["SIMSOPT_BACKEND_MODE"] = "native_cpu"
        environment["CUDA_VISIBLE_DEVICES"] = ""
    else:
        environment = pinned_environment(
            lane=lane,
            omp=omp,
            jax_platforms=jax_platforms,
            compile_cache_dir=compile_cache,
        )
        environment["SIMSOPT_BACKEND_MODE"] = (
            "jax_cpu_fast" if jax_platforms == "cpu" else "jax_gpu_fast"
        )
        if jax_platforms == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        else:
            # Never preallocate: this box's GPU is shared with other campaigns.
            environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        if compile_cache is not None:
            # Without these two the cache stays empty for small/fast programs,
            # so a "warm persistent cache" leg would be cold in disguise.
            environment.update(PERSISTENT_CACHE_THRESHOLDS)
    environment["SIMSOPT_BACKEND_STRICT"] = "1"
    environment["SIMSOPT_PRECISION"] = "fp64"
    inherited_pythonpath = environment.get("PYTHONPATH")
    entries = [str(REPO_ROOT), str(REPO_ROOT / "src")]
    if inherited_pythonpath:
        entries.append(inherited_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))
    return environment


def _attested_environment() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in _ATTESTED_ENVIRONMENT_NAMES}


# ---------------------------------------------------------------------------
# Endpoints and comparison
# ---------------------------------------------------------------------------


def validate_output_path(path: Path) -> Path:
    """Where a published artifact may land, mirroring the marginal probe's rule.

    An ``--output`` *inside* this repository must be under
    :data:`EVIDENCE_ROOT`: an artifact written somewhere else in the tree is
    evidence nobody looking for evidence will find, and it is one ``git add``
    away from being committed as source.  An ``--output`` entirely outside the
    repository is a scratch run — allowed, and stamped
    ``published_under_evidence_root: false`` in the artifact itself, so a
    reader never has to infer publication status from a path they cannot see.
    """
    resolved = path.resolve()
    if EVIDENCE_ROOT not in resolved.parents and REPO_ROOT in resolved.parents:
        raise ProbeConventionError(
            f"--output inside the repository must live under {EVIDENCE_ROOT}, got "
            f"{resolved}; a path outside the repository is accepted and is recorded "
            "as published_under_evidence_root: false"
        )
    return resolved


def published_under_evidence_root(path: Path) -> bool:
    """Whether this artifact lands in the published evidence tree."""
    return EVIDENCE_ROOT in path.resolve().parents


def validate_endpoint_path(path: Path) -> Path:
    """Refuse an endpoint that is misnamed or would clobber the other half of a pair.

    Both refusals are about evidence, not tidiness, and mirror
    ``benchmarks/pm_gpmo_probes.py``'s ``_validate_moments_path``.  A suffix
    other than :data:`ENDPOINT_SUFFIX` is a mistyped flag: ``np.savez`` appends
    ``.npz`` to a path it is handed, so ``foo.npy`` becomes ``foo.npy.npz`` and
    ``--compare`` is later pointed at a file that was never written.  An
    existing file is the *other* lane's endpoint — the two lanes cannot share a
    process, so a pair is built one file at a time — and overwriting it
    destroys the half of the comparison that already ran.  The write itself
    opens ``"xb"``, so the refusal is enforced by the kernel too, and two legs
    racing one path cannot both win.
    """
    if path.suffix != ENDPOINT_SUFFIX:
        raise ProbeConventionError(
            f"--endpoint-out must end in {ENDPOINT_SUFFIX}, got {path.name!r}: "
            "numpy.savez appends the suffix itself, so any other path names a "
            "file that is never written and a --compare side nobody can load"
        )
    if path.exists():
        raise ProbeConventionError(
            f"--endpoint-out refuses to overwrite an existing endpoint: {path}; "
            "that file is the other half of a --compare pair, not scratch"
        )
    return path


def write_endpoint(
    path: Path, *, metadata: dict[str, object], leg: dict[str, object]
) -> None:
    """Publish final DOFs and objective for cross-lane comparison.

    ``np.savez`` is handed an open ``"xb"`` handle rather than a path: the
    handle is the overwrite refusal (:func:`validate_endpoint_path` checks it
    early, the kernel enforces it at the write), and passing a handle also
    stops ``np.savez`` appending a second ``.npz`` to the operator's path.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        np.savez(
            handle,
            dofs=leg["final_parameters"],
            initial_dofs=leg["initial_parameters"],
            objective=np.asarray(leg["final_objective"], dtype=np.float64),
            metadata=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        )


#: Fields that must agree before two endpoints are comparable at all.  ``nit``
#: and ``nfev`` are deliberately absent: at one matched policy the two lanes may
#: converge in different iteration counts, and that is a fact to report, not a
#: mismatch to refuse on.
_ENDPOINT_IDENTITY_FIELDS = (
    "scale",
    "budget",
    "maxcor",
    "training_sha256",
    "initial_parameters_sha256",
)

#: Reported per side, never gated.
_ENDPOINT_REPORT_FIELDS = ("nit", "nfev")


def _endpoint_field(metadata: Mapping[str, object], name: str, path: Path) -> object:
    """One endpoint-metadata field, or a typed refusal naming what is missing.

    A missing key here is a schema mismatch — an endpoint written by an older
    probe, by a different probe, or by hand — and the reader is told which
    field and which file, instead of being handed a bare ``KeyError``.
    """
    if name not in metadata:
        raise ProbeConventionError(
            f"{path} carries no {name!r} in its endpoint metadata: expected the "
            f"{SCHEMA} endpoint schema (identity fields "
            f"{list(_ENDPOINT_IDENTITY_FIELDS)}, reported fields "
            f"{list(_ENDPOINT_REPORT_FIELDS)}, plus 'lane'), got "
            f"{sorted(metadata)}; the file was almost certainly written by a "
            "different or older probe schema"
        )
    return metadata[name]


def compare_endpoints(first: Path, second: Path) -> dict[str, object]:
    """Report endpoint agreement against the ``native_workflow`` bucket."""
    import numpy as np
    from simsopt_jax.parity_tolerances import parity_ladder_tolerances

    with np.load(first, allow_pickle=False) as archive:
        first_dofs = np.asarray(archive["dofs"], dtype=np.float64)
        first_objective = float(archive["objective"])
        first_metadata = json.loads(str(archive["metadata"].item()))
    with np.load(second, allow_pickle=False) as archive:
        second_dofs = np.asarray(archive["dofs"], dtype=np.float64)
        second_objective = float(archive["objective"])
        second_metadata = json.loads(str(archive["metadata"].item()))

    mismatched = {
        name: (
            _endpoint_field(first_metadata, name, first),
            _endpoint_field(second_metadata, name, second),
        )
        for name in _ENDPOINT_IDENTITY_FIELDS
        if _endpoint_field(first_metadata, name, first)
        != _endpoint_field(second_metadata, name, second)
    }
    if mismatched:
        raise ProbeConventionError(
            "endpoints are not comparable; these matched-sample/matched-policy "
            f"fields differ (first vs second): {mismatched}"
        )
    if first_dofs.shape != second_dofs.shape:
        raise ProbeConventionError(
            f"DOF shapes differ: {first_dofs.shape} vs {second_dofs.shape}"
        )

    tolerances = parity_ladder_tolerances(TOLERANCE_BUCKET)
    rtol = float(tolerances["whole_solve_value_rtol"])
    atol = float(tolerances["whole_solve_value_atol"])
    dof_absolute = np.abs(first_dofs - second_dofs)
    dof_scale = np.maximum(np.abs(second_dofs), np.finfo(np.float64).tiny)
    objective_absolute = abs(first_objective - second_objective)
    objective_scale = max(abs(second_objective), np.finfo(np.float64).tiny)
    report = {
        "bucket": TOLERANCE_BUCKET,
        "whole_solve_value_rtol": rtol,
        "whole_solve_value_atol": atol,
        "max_abs_dof_diff": float(np.max(dof_absolute)),
        "max_rel_dof_diff": float(np.max(dof_absolute / dof_scale)),
        "dofs_within_bucket": bool(
            np.all(dof_absolute <= atol + rtol * np.abs(second_dofs))
        ),
        "objective_a": first_objective,
        "objective_b": second_objective,
        "abs_objective_diff": objective_absolute,
        "rel_objective_diff": objective_absolute / objective_scale,
        "objective_within_bucket": bool(
            objective_absolute <= atol + rtol * abs(second_objective)
        ),
        "lane_a": _endpoint_field(first_metadata, "lane", first),
        "lane_b": _endpoint_field(second_metadata, "lane", second),
        "matched_fields": {
            name: _endpoint_field(first_metadata, name, first)
            for name in _ENDPOINT_IDENTITY_FIELDS
        },
        # Reported, not gated: the two lanes run one matched policy, and the
        # native lane may legitimately reach its stopping test in fewer
        # iterations.  A reader comparing endpoint agreement needs to see the
        # iteration counts the two endpoints were reached in.
        "iteration_counts": {
            "a": {
                name: _endpoint_field(first_metadata, name, first)
                for name in _ENDPOINT_REPORT_FIELDS
            },
            "b": {
                name: _endpoint_field(second_metadata, name, second)
                for name in _ENDPOINT_REPORT_FIELDS
            },
            "gated": False,
        },
    }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched-policy, matched-sample stochastic stage-two A/B probe "
            "(diagnostic-not-certifying)."
        )
    )
    parser.add_argument("--lane", choices=_LANES)
    parser.add_argument(
        "--scale",
        choices=("native_default", "bounded"),
        default="native_default",
        help="native_default is the shipped scale; bounded is smoke-only.",
    )
    parser.add_argument(
        "--maxcor",
        type=int,
        choices=MATCHED_MAXCOR,
        default=10,
        help="L-BFGS history size, identical on both lanes of a pair.",
    )
    parser.add_argument("--budget", type=int, default=400, help="maxiter, both lanes.")
    parser.add_argument(
        "--omp",
        type=int,
        default=None,
        help=(
            "Required for a timed leg on both lanes: OMP_NUM_THREADS pinned before "
            f"interpreter start; sweep {list(OMP_SWEEP)}."
        ),
    )
    parser.add_argument(
        "--sample-tile",
        type=int,
        default=None,
        help=(
            "JAX lane only, refused on --lane native: sample-axis tile (plan "
            "P1.2/P1.3); omit for the scan oracle."
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="Timed solves per leg; solve 0 is cold-in-process, the rest warm.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Artifact JSON path; inside the repository it must live under "
            f"{EVIDENCE_ROOT.relative_to(REPO_ROOT)}, outside it is a scratch run "
            "stamped published_under_evidence_root: false."
        ),
    )
    parser.add_argument(
        "--endpoint-out",
        type=Path,
        default=None,
        metavar="ENDPOINT.npz",
        help=(
            "NPZ with final DOFs and objective for --compare; refuses a foreign "
            "suffix and refuses to overwrite an existing endpoint."
        ),
    )
    parser.add_argument(
        "--compile-cache",
        type=Path,
        default=None,
        help=(
            "JAX lane only, refused on --lane native: persistent compilation cache "
            "directory (warm leg); omit for cold."
        ),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="JSONL appended once per executed leg; this is the executed-order proof.",
    )
    parser.add_argument(
        "--jax-platforms",
        choices=("cuda", "cpu"),
        default="cuda",
        help="JAX lane platform; cpu is for smoke runs that must not touch the GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configuration, shapes and sample fingerprints; solve nothing.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        default=None,
        metavar=("A.npz", "B.npz"),
        help="Compare two endpoint files at the native_workflow bucket.",
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _dry_run(arguments: argparse.Namespace) -> None:
    import numpy as np
    from simsopt.field import BiotSavart

    shared = build_shared_inputs(arguments.scale)
    identity = sample_identity(shared)
    workload = workload_shape(shared)
    initial_parameters = np.ascontiguousarray(
        np.asarray(BiotSavart(shared.coils).x, dtype=np.float64)
    )
    print(f"schema {SCHEMA} grade diagnostic-not-certifying")
    print(
        f"scale {shared.scale} construction_seconds {shared.construction_seconds:.3f}"
    )
    print(f"configuration {json.dumps(asdict(shared.configuration), sort_keys=True)}")
    print(
        f"surface_input_sha256 {shared.configuration_mapping['surface_input_sha256']}"
    )
    print(f"coils {len(shared.coils)} base_curves {len(shared.base_curves)}")
    print(
        f"dof_count {initial_parameters.size} initial_parameters_sha256 "
        f"{array_sha256(initial_parameters)}"
    )
    print(f"workload {json.dumps(workload, sort_keys=True)}")
    print(f"samples {json.dumps(identity, sort_keys=True)}")
    print(
        f"mpi_ranks 1 omp_sweep {list(OMP_SWEEP)} matched_maxcor {list(MATCHED_MAXCOR)}"
    )
    print(f"matched_tolerance {MATCHED_TOLERANCE}")
    # A suggested alternation for the operator, printed and never published:
    # nothing here observed it being followed.  What ran is the ledger's.
    print(
        f"interleave (plan) {interleave_schedule(3, *_LANES)}"
        "  -- suggestion only; executed order lives in the ledger"
    )


def _command() -> str:
    return " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])


def _run_leg(arguments: argparse.Namespace) -> int:
    import numpy as np

    observed_omp = os.environ.get("OMP_NUM_THREADS")
    if observed_omp != str(arguments.omp):
        raise ProbeConventionError(
            f"leg requested OMP_NUM_THREADS={arguments.omp} but the child observed "
            f"{observed_omp!r}; the pin did not survive to the interpreter"
        )
    shared = build_shared_inputs(arguments.scale)
    started_ns = time.time_ns()
    if arguments.lane == "native":
        rows, leg = run_native_leg(
            shared,
            budget=arguments.budget,
            maxcor=arguments.maxcor,
            repeat=arguments.repeat,
        )
    else:
        rows, leg = run_jax_leg(
            shared,
            budget=arguments.budget,
            maxcor=arguments.maxcor,
            repeat=arguments.repeat,
            sample_tile=arguments.sample_tile,
        )
    finished_ns = time.time_ns()
    # Taken here rather than at entry: libgomp is mapped by the leg's own
    # imports, so this reads back the runtime the solves actually ran on.
    observed_threads = observed_openmp_threads()
    solves = [asdict(row) for row in rows]
    refused = refused_solve_rows(rows, budget=arguments.budget)
    # The ledger records what executed, including a leg the gate below refuses:
    # a broken leg still occupied the box and still belongs in the order.
    append_leg_ledger(
        arguments.ledger,
        {
            "schema": SCHEMA,
            "lane": arguments.lane,
            "scale": arguments.scale,
            "budget": arguments.budget,
            "maxcor": arguments.maxcor,
            "repeat": arguments.repeat,
            "omp_num_threads": arguments.omp,
            "observed_openmp_threads": observed_threads,
            "pid": os.getpid(),
            "started_ns": started_ns,
            "finished_ns": finished_ns,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "solves": solves,
            "publishable": not refused,
            "artifact": str(arguments.output),
            "command": _command(),
        },
    )
    if refused:
        raise ProbeConventionError(
            f"leg refused publication at --budget {arguments.budget}: {SOLVE_GATE}. "
            "These rows do not: "
            f"{json.dumps([asdict(row) for row in refused], sort_keys=True)}"
        )
    initial_parameters = np.ascontiguousarray(leg.pop("initial_parameters"))
    final_parameters = np.ascontiguousarray(leg.pop("final_parameters"))
    initial_sha256 = array_sha256(initial_parameters)
    samples = sample_identity(shared)
    identity = runtime_identity(arguments.lane)
    # What the OpenMP runtime reports, next to what the pin requested: the two
    # differ under a thread limit or a cgroup quota, and the denominator is the
    # observed one.
    identity["observed_openmp_threads"] = observed_threads
    # Levers that exist only on the JAX lane are stamped only on the JAX lane;
    # the native lane refuses them at parse time rather than recording a knob
    # that could not have moved its number.
    lane_options: dict[str, object] = (
        {
            "sample_tile": arguments.sample_tile,
            "compile_cache_dir": (
                None
                if arguments.compile_cache is None
                else str(arguments.compile_cache)
            ),
            "jax_platforms_requested": arguments.jax_platforms,
        }
        if arguments.lane == "jax-gpu"
        else {}
    )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "identity": identity,
        "plan": PLAN,
        "plan_tasks": ["P1.1", "P1.3"],
        "publication": PUBLICATION,
        "parity_case_cloned": PARITY_CASE,
        "disclosures": list(
            _SHARED_DISCLOSURES
            + (_JAX_LANE_DISCLOSURES if arguments.lane == "jax-gpu" else ())
        ),
        "lane": arguments.lane,
        "scale": arguments.scale,
        "budget": arguments.budget,
        "maxcor": arguments.maxcor,
        "repeat": arguments.repeat,
        "mpi_ranks": 1,
        "omp_num_threads": arguments.omp,
        "omp_sweep_contract": list(OMP_SWEEP),
        **lane_options,
        "environment": _attested_environment(),
        "environment_scrub_prefixes": list(SCRUBBED_ENVIRONMENT_PREFIXES),
        "configuration": asdict(shared.configuration),
        "surface_input_sha256": shared.configuration_mapping["surface_input_sha256"],
        "workload": workload_shape(shared),
        "samples": samples,
        "initial_parameters_sha256": initial_sha256,
        "shared_construction_seconds": shared.construction_seconds,
        "solves": solves,
        "solves_gate": SOLVE_GATE,
        "timed_window": "minimize call only; no Taylor test, no out-of-sample",
        "tolerance_bucket": TOLERANCE_BUCKET,
        # False is a scratch run outside the repository, not a defect: the
        # publication rule is enforced at parse time and its verdict is stamped
        # here so a reader never infers it from a path they cannot see.
        "published_under_evidence_root": published_under_evidence_root(
            arguments.output
        ),
        "ledger": str(arguments.ledger),
        "interleave_owner": "operator; executed order provable from the ledger",
        "command": _command(),
        **leg,
    }
    write_probe_artifact(arguments.output, payload)
    print(f"wrote {arguments.output}", flush=True)
    if arguments.endpoint_out is not None:
        write_endpoint(
            arguments.endpoint_out,
            metadata={
                "lane": arguments.lane,
                "scale": arguments.scale,
                "budget": arguments.budget,
                "maxcor": arguments.maxcor,
                # The endpoint is the *last* solve's; its iteration counts are
                # reported by --compare, never gated on.
                "nit": rows[-1].nit,
                "nfev": rows[-1].nfev,
                "training_sha256": samples["training_sha256"],
                "initial_parameters_sha256": initial_sha256,
                "final_objective": leg["final_objective"],
                "artifact": str(arguments.output),
                **lane_options,
            },
            leg={
                "final_parameters": final_parameters,
                "initial_parameters": initial_parameters,
                "final_objective": leg["final_objective"],
            },
        )
        print(f"wrote {arguments.endpoint_out}", flush=True)
    return 0


def main(argv: list[str]) -> int:
    arguments = _parse(argv)
    if arguments.compare is not None:
        report = compare_endpoints(*arguments.compare)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    if arguments.dry_run:
        _dry_run(arguments)
        return 0
    if arguments.lane is None:
        raise ProbeConventionError("--lane is required for a timed leg")
    if arguments.output is None:
        raise ProbeConventionError(
            "--output is required for a timed leg; an unpublished leg is not evidence"
        )
    validate_output_path(arguments.output)
    if arguments.endpoint_out is not None:
        validate_endpoint_path(arguments.endpoint_out)
    if arguments.omp is None:
        raise ProbeConventionError(
            "--omp is required for a timed leg on both lanes: an omitted pin is not "
            "a neutral default but the unset-OMP_NUM_THREADS collapse, and a leg "
            f"nobody pinned has no denominator; sweep {list(OMP_SWEEP)}"
        )
    if arguments.omp < 1:
        raise ProbeConventionError(f"--omp must be >= 1, got {arguments.omp}")
    if arguments.lane == "native":
        jax_only = [
            name
            for name, value in (
                ("--sample-tile", arguments.sample_tile),
                ("--compile-cache", arguments.compile_cache),
            )
            if value is not None
        ]
        if jax_only:
            raise ProbeConventionError(
                f"--lane native does not implement {', '.join(jax_only)}: the "
                "sample-axis tile is a factory lever of the JAX objective and the "
                "persistent compilation cache is an XLA lever, so neither can move "
                "a SciPy/simsopt number; stamping them into a native artifact "
                "would describe a knob that never applied"
            )
    if arguments.repeat < 1:
        raise ProbeConventionError(f"--repeat must be >= 1, got {arguments.repeat}")
    if arguments.budget < 1:
        raise ProbeConventionError(f"--budget must be >= 1, got {arguments.budget}")
    if arguments.child:
        return _run_leg(arguments)
    environment = leg_environment(
        arguments.lane,
        omp=arguments.omp,
        jax_platforms=arguments.jax_platforms,
        compile_cache=arguments.compile_cache,
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *argv, "--child"],
        cwd=str(REPO_ROOT),
        env=environment,
        check=False,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
