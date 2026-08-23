"""Kill-fast A/B probes for the marginal quartet (backlog plan Phase 4).

Four families sit just under the mechanism threshold of
``docs/jax_example_device_assignment.md`` and have never been timed:
``stage_two_optimization`` and ``stage_two_optimization_planar_coils`` (1.23M
Biot-Savart pairs per *native* evaluation against the mirror's 1.64M -- the two
lanes do not do the same per-evaluation work, and every dimension is therefore
published per lane), ``coil_forces`` (0.92M pairs on both lanes plus ~202.5k
force terms), and ``wireframe_rcls_with_ports`` (a null-space
equality-constrained least-squares solve).  Tasks P4.1-P4.3 of
``docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md`` ask one
question of each: does the shipped JAX lane beat the shipped native lane at
``native_default``?  This module is the instrument that asks it, and nothing
else -- it computes no ratio, sweeps nothing, and certifies nothing.  Every
artifact it writes is stamped ``diagnostic-not-certifying`` by
``benchmarks.probe_conventions``; the kill rule (``kill_threshold_warm``, 1.0x
warm) is recorded *in* the artifact as a field so the reader applies it, not
this script.

Interpreters (one lane per process, because the two lanes need different
environments)::

    native lane:   .venv/bin/python benchmarks/marginal_quartet_probes.py ...
    JAX GPU lane:  .venv-qn-gpu/bin/python benchmarks/marginal_quartet_probes.py ...

Usage::

    <interpreter> benchmarks/marginal_quartet_probes.py stage-two \\
        --lane jax-gpu --omp 8 --repeat 3 \\
        --compile-cache /path/to/cache \\
        --output docs/receipts/evidence/marginal_quartet_stage_two_jax.json

    <interpreter> benchmarks/marginal_quartet_probes.py rcls-ports \\
        --lane native --omp 8 --repeat 3 --currents-out /tmp/native.npy \\
        --output docs/receipts/evidence/marginal_quartet_rcls_native.json

    <interpreter> benchmarks/marginal_quartet_probes.py compare-artifacts \\
        native.json jax.json

``--dry-run`` resolves and prints the configuration and the problem dimensions
without running or publishing anything.  ``compare-artifacts`` reads two
already-published artifacts and prints the endpoint difference between the two
lanes; it is report-only and computes no speed ratio.

Five structural decisions are worth stating up front, because all five are load
bearing:

*Heavy imports are deferred into the lane functions on purpose.*  Thread
pinning (``OMP_NUM_THREADS`` and its BLAS siblings) and the JAX persistent
compilation-cache lever must be in the environment *before* NumPy, JAX, or the
compiled extension initialize, and ``probe_conventions.runtime_identity`` must
be able to report truthfully that a native leg never imported JAX.  A
module-level ``import jax`` would break both properties at once.  ``main``
therefore pins ``os.environ`` through
``probe_conventions.pinned_environment`` and refuses to continue if any
numerical module is already loaded, which is what makes the in-process pin a
pin rather than a wish.

*Environment construction is not re-implemented here.*  Both the native child's
environment and this process's own come from ``pinned_environment``: scrub the
whole ``OMP_``/``JAX_``/``XLA_``/``SIMSOPT_``/``CUDA_``/``MPI4PY_`` family out
of the inherited shell first, then pin what the leg needs.  Popping only the
variables one thinks to name leaves ``SIMSOPT_BACKEND_MODE``, ``XLA_FLAGS``,
``KMP_AFFINITY``, ``OMP_PROC_BIND``, ``CUDA_VISIBLE_DEVICES`` and
``JAX_COMPILATION_CACHE_DIR`` alive inside a "native" denominator.  The CI
family is the one thing ``pinned_environment`` does not cover -- ``CI``,
``GITHUB_ACTIONS`` and ``IN_GITHUB_ACTIONS`` match none of its prefixes -- so
this module pops those three on top and records that it did.

*The JAX lane executes the shipped mirror's own ``solve``*, loaded from the
example path, rather than a re-typed copy of its problem construction.  The
mirrors own the ``native_default`` scale, the weights, and the solver options
(``Driver.SIMSOPT_LBFGSB`` -> ``dispatch.minimize`` -> fused on-device L-BFGS);
re-typing them here would create a second source of truth that drifts silently
and turns a matched-budget probe into an unmatched one.  Loading the example by
path is the in-tree precedent (``benchmarks/flat675_promotion_robustness_child.py:97-104``).
The one family built here instead is ``rcls-ports``, where *both* lanes must
consume bit-identical geometry for the currents comparison to mean anything, so
a single JAX-free builder feeds both.

*The comparison is shipped-vs-shipped, not matched-work.*  The native scripts
minimize with ``maxcor=300, tol=1e-15``; the mirrors carry their own history
size and tolerances.  Both policies are published as data on both lanes and the
mismatch is disclosed in the artifact, because the resulting ratio answers
"where should this example be launched as shipped", not "which hardware is
faster at the same work".

*Dimensions are published per lane, never as one shared number.*  The policy
mismatch above is not the only one: the two stage-two native scripts leave
``numquadpoints`` at the ``create_equally_spaced_curves`` default of
``15 * order = 75`` while their mirrors pass ``numquadpoints=100``, so the JAX
lane evaluates 1.33x more Biot-Savart pairs per evaluation than the lane it is
being timed against.  A single ``declared_dimensions`` block cannot state that,
and stating only one of the two numbers publishes a per-evaluation work figure
that is wrong for one lane by that factor.  Each family therefore derives its
pair count per lane from ``coils * curve_quadrature_points * surface_points``
(:func:`_mirror_dimensions`) and discloses the resulting asymmetry against
``kill_threshold_warm``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

_MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULE_ROOT))

from benchmarks.probe_conventions import (
    OMP_SWEEP,
    PROBE_GRADE,
    REPO_ROOT,
    SCRUBBED_ENVIRONMENT_PREFIXES,
    ProbeConventionError,
    append_leg_ledger,
    array_sha256,
    compile_cache_env,
    observed_openmp_threads,
    pinned_environment,
    runtime_identity,
    ulp_distance,
    write_probe_artifact,
)

#: Schema of every artifact this module publishes.
SCHEMA = "marginal-quartet-probe.v1"

#: Schema of every line this module appends to the leg ledger.
LEDGER_SCHEMA = "marginal-quartet-leg.v1"

#: The plan's kill rule, carried in the artifact as data (plan Phase 4).  A
#: family whose warm JAX lane does not beat the swept-native lane by at least
#: this factor is closed, not charted.
KILL_THRESHOLD_WARM = 1.0

PLAN = "docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md"

#: Probe artifacts are published here and nowhere else.  A run whose ``--output``
#: lands entirely outside the repository is a scratch run: it is allowed, and it
#: is marked ``published_under_evidence_root: false`` in its own configuration.
EVIDENCE_ROOT = REPO_ROOT / "docs" / "receipts" / "evidence"

#: Executed leg order is evidence, and one artifact written at the end cannot
#: carry it: the two lanes run in different processes under different
#: interpreters.  Each leg appends its own line here as it finishes.
DEFAULT_LEDGER = EVIDENCE_ROOT / "probe_leg_ledger.jsonl"

NATIVE_INTERPRETER = REPO_ROOT / ".venv" / "bin" / "python"
GPU_INTERPRETER = REPO_ROOT / ".venv-qn-gpu" / "bin" / "python"
CHILD_PYTHONPATH = (
    REPO_ROOT / "src",
    REPO_ROOT / "build" / "cp311-cp311-linux_x86_64",
)

#: ``simsopt.util.in_github_actions`` reads ``CI`` (``src/simsopt/util/__init__.py:11``),
#: not ``IN_GITHUB_ACTIONS``; all three are scrubbed so the child takes the
#: non-CI branch (MAXITER 400, nphi/ntheta 32) whatever the parent inherited.
#: These are the *only* variables this module scrubs itself: none of them starts
#: with any prefix in :data:`SCRUBBED_ENVIRONMENT_PREFIXES`, so
#: ``pinned_environment`` leaves them inherited and the pop has to happen here.
_CI_VARIABLES = ("CI", "GITHUB_ACTIONS", "IN_GITHUB_ACTIONS")

#: Pinned by this module rather than by ``pinned_environment``, and therefore
#: not recoverable from the scrub prefixes alone.
_EXTRA_PINNED_KEYS = ("HWLOC_COMPONENTS", "PYTHONPATH", "PYTHONUNBUFFERED")

#: Modules whose presence at pin time would mean the pin came too late: every
#: one of them reads the threading or cache environment at initialization.
_PIN_BLOCKING_MODULES = ("numpy", "scipy", "jax", "jaxlib", "simsoptpp")

DEFAULT_REPEAT = 3

#: A JAX lane reports a warm number, and a warm number needs a warm sample.  One
#: repeat produces a cold leg and nothing else, so the kill rule would be read
#: off a compile.
MINIMUM_JAX_REPEAT = 2

#: The four leg kinds, and the single source :func:`_summarize` reads to decide
#: whether a series has a cold leg 0 to exclude.  ``COLD_IN_PROCESS`` is the
#: first leg of an in-process series -- on EITHER lane: a native in-process
#: repeat pays first touch (BLAS pool spin-up, page faults, lazy imports) just
#: as a JAX leg pays a compile, and excluding it from one lane's warm statistics
#: while folding it into the other's median is a bias in the ratio, not a
#: convention.  ``FRESH_PROCESS_COLD`` marks a series in which *every* leg is
#: cold, so there is nothing to exclude and no warm number exists at all.
COLD_IN_PROCESS = "cold_in_process"
WARM_IN_PROCESS = "warm_in_process"
REPEAT_PERSISTENT_CACHE = "repeat_persistent_cache"
FRESH_PROCESS_COLD = "fresh_process_cold"

#: Series labels, one per kind of repetition this probe can produce.
COLD_THEN_IN_PROCESS_WARM = "cold_then_in_process_warm"
COLD_THEN_PERSISTENT_CACHE = "cold_then_persistent_cache"
FRESH_PROCESS_COLD_ONLY = "fresh_process_cold_only"

#: Suffix of the string field that carries a non-finite value whose numeric
#: field had to be published as ``null``; see :func:`_endpoint_fields`.
SENTINEL_SUFFIX = "_sentinel"

#: The native scripts print one objective line per evaluation.  Their Taylor
#: test is one evaluation at the base point plus a (+eps, -eps) pair for each of
#: five epsilons, and one ``err`` line per epsilon.
TAYLOR_EVALUATIONS = 11
TAYLOR_ERROR_LINES = 5

#: What every native script hands ``scipy.optimize.minimize``.  Anchored per
#: family by ``MirrorFamily.minimize_call_sources``, which names both call
#: sites; the mirrors do not use these values, which is the point.
NATIVE_LBFGSB_POLICY: Mapping[str, object] = {
    "optimizer": "scipy.optimize.minimize(method='L-BFGS-B')",
    "maxcor": 300,
    "tol": 1.0e-15,
    "maxiter_symbol": "MAXITER",
}

#: RCLS constants shared by the native example and its mirror.
RCLS_REGULARIZATION_WEIGHT = 1.0e-10
RCLS_PORT_GAP = 0.04
RCLS_SURFACE_DISTANCE = 0.3
RCLS_WIREFRAME_PHI = 12
RCLS_WIREFRAME_THETA = 22
RCLS_FIELD_ON_AXIS = 1.0
RCLS_DEFAULT_RESOLUTION = 32

#: Accepted suffixes for ``--currents-out``.  The payload is always an ``npz``
#: archive (currents plus their configuration metadata); ``.npy`` stays accepted
#: because an operator's muscle memory writes it, and refusing the suffix
#: outright would be a worse answer than honouring it with the archive the
#: comparison step actually needs.
CURRENTS_SUFFIXES = (".npy", ".npz")


class ProbeConfigurationError(ProbeConventionError):
    """The requested probe configuration cannot produce a matched comparison."""


@dataclass(frozen=True)
class LaneQuadrature:
    """One lane's curve quadrature count and the code that sets it.

    ``points`` is a *factor* of the per-evaluation pair count, never the pair
    count itself: :func:`_mirror_dimensions` multiplies it out.  Storing the
    product beside the factors would give the artifact two sources for one
    number, and the one that drifts is always the product.
    """

    points: int
    source: str


def _mirror_dimensions(
    *,
    coils: int,
    base_curves: int,
    base_curve_order: int,
    surface_points: int,
    native_quadrature: LaneQuadrature,
    mirror_quadrature: LaneQuadrature,
    shared_source: str,
    **extra: object,
) -> dict[str, object]:
    """Per-lane problem dimensions, with every pair count derived here.

    The native scripts and their mirrors do not agree about curve quadrature:
    the stage-two pair leaves ``numquadpoints`` unset, taking
    ``create_equally_spaced_curves``' ``15 * order`` default, while the mirrors
    pass ``numquadpoints=100`` explicitly.  A single shared
    ``biot_savart_pairs_per_evaluation`` therefore cannot be true of both lanes,
    and publishing one of the two numbers as if it were shared overstates the
    native lane's work by the ratio between them.

    Both lanes' counts come out of one derivation --
    ``coils * curve_quadrature_points * surface_points`` -- so the published
    ratio and the published counts cannot disagree with each other.
    """
    native_pairs = coils * native_quadrature.points * surface_points
    mirror_pairs = coils * mirror_quadrature.points * surface_points
    return {
        "coils": coils,
        "base_curves": base_curves,
        "base_curve_order": base_curve_order,
        "surface_points": surface_points,
        "per_lane": {
            "native": {
                "curve_quadrature_points": native_quadrature.points,
                "curve_quadrature_source": native_quadrature.source,
                "biot_savart_pairs_per_evaluation": native_pairs,
            },
            "jax-gpu": {
                "curve_quadrature_points": mirror_quadrature.points,
                "curve_quadrature_source": mirror_quadrature.source,
                "biot_savart_pairs_per_evaluation": mirror_pairs,
            },
        },
        "biot_savart_pairs_derivation": (
            "coils * curve_quadrature_points * surface_points, evaluated per lane"
        ),
        "jax_over_native_pairs_per_evaluation": mirror_pairs / native_pairs,
        "source": shared_source,
        **extra,
    }


@dataclass(frozen=True)
class MirrorFamily:
    """A native example script paired with the JAX mirror that ports it.

    ``evaluation_marker`` selects the native script's per-evaluation objective
    lines out of its stdout.  It has to exclude any post-solve summary line that
    also begins with ``J=`` -- ``coil_forces`` prints one (its module-level
    post-solve summary, after ``calculate_modB_on_major_radius``) and the
    stage-two pair does not -- because that line lands after the last minimize
    call and would stretch the measured region past the solver.

    ``mirror_policy`` is the mirror's own solver policy, read out of the mirror
    and out of the shared solve helper it calls.  It is published beside
    :data:`NATIVE_LBFGSB_POLICY` so a reader can see, in the artifact, that the
    two lanes are not solving under the same stopping rule.

    ``declared_dimensions`` comes from :func:`_mirror_dimensions` and is
    per-lane wherever the lanes differ: quadrature and pair counts are not
    shared facts for this quartet, and a reader who is handed one number for
    both lanes is handed a wrong one for at least one of them.
    """

    name: str
    native_script: Path
    mirror_module: Path
    native_budget: int
    native_budget_source: str
    minimize_call_sources: tuple[str, ...]
    evaluation_marker: str
    mirror_policy: Mapping[str, object]
    declared_dimensions: Mapping[str, object]
    observable_names: tuple[str, ...]


_STAGE_TWO_OBSERVABLES = (
    "final_objective",
    "squared_flux",
    "solver_success",
    "solver_status",
    "solver_iterations",
)

#: ``_STAGE_TWO_LBFGS_HISTORY_SIZE`` is private to
#: ``src/simsopt_jax/examples/stage_two_standard.py``, so it is cited by symbol
#: and value rather than imported: importing a private name would make this
#: probe a consumer of an interface the module does not offer.
_STAGE_TWO_MIRROR_MAXCOR_SYMBOL = (
    "src/simsopt_jax/examples/stage_two_standard.py"
    "::_STAGE_TWO_LBFGS_HISTORY_SIZE (=10, consumed by "
    "::solve_standard_stage_two for both of its serial_solve_jax stages)"
)

#: How the native lane gets 75 quadrature points without ever naming a number:
#: neither stage-two script nor ``coil_forces`` passes ``numquadpoints``, so the
#: creator's ``15 * order`` fallback decides, and ``order`` is 5 in all three.
#: This is the citation the kill rule depends on, because it is the difference
#: between the native lane's per-evaluation work and the mirror's.
_NATIVE_QUADRATURE_DEFAULT = (
    "src/simsopt/geo/curve.py::create_equally_spaced_curves "
    "(numquadpoints=None falls back to 15 * order)"
)
_NATIVE_PLANAR_QUADRATURE_DEFAULT = (
    "src/simsopt/geo/curve.py::create_equally_spaced_planar_curves "
    "(numquadpoints=None falls back to 15 * order)"
)

MIRROR_FAMILIES: Mapping[str, MirrorFamily] = {
    "stage-two": MirrorFamily(
        name="stage-two",
        native_script=REPO_ROOT / "examples/2_Intermediate/stage_two_optimization.py",
        mirror_module=REPO_ROOT
        / "examples/jax/2_Intermediate/stage_two_optimization.py",
        native_budget=400,
        native_budget_source="examples/2_Intermediate/stage_two_optimization.py::MAXITER",
        minimize_call_sources=(
            "examples/2_Intermediate/stage_two_optimization.py "
            "(module-level minimize call, stage one: LENGTH_WEIGHT as declared)",
            "examples/2_Intermediate/stage_two_optimization.py "
            "(module-level minimize call, stage two: after LENGTH_WEIGHT *= 0.1)",
        ),
        evaluation_marker="║∇J║",
        mirror_policy={
            "optimizer": "Driver.SIMSOPT_LBFGSB via serial_solve_jax",
            "maxcor": 10,
            "maxcor_symbol": _STAGE_TWO_MIRROR_MAXCOR_SYMBOL,
            "rtol": 1.0e-12,
            "atol": 1.0e-10,
            "source": "examples/jax/2_Intermediate/stage_two_optimization.py::solve",
        },
        declared_dimensions=_mirror_dimensions(
            coils=16,
            base_curves=4,
            base_curve_order=5,
            surface_points=1024,
            native_quadrature=LaneQuadrature(
                points=75,
                source=(
                    "examples/2_Intermediate/stage_two_optimization.py "
                    "(module-level create_equally_spaced_curves call passes NO "
                    f"numquadpoints) -> {_NATIVE_QUADRATURE_DEFAULT} = 15 * 5"
                ),
            ),
            mirror_quadrature=LaneQuadrature(
                points=100,
                source=(
                    "examples/jax/2_Intermediate/stage_two_optimization.py"
                    "::_build_problem (numquadpoints=100 at native_default)"
                ),
            ),
            shared_source=(
                "plan Current Context (U5); coils = 4 base curves * nfp 2 * 2 "
                "(stellsym) via coils_via_symmetries; surface_points = nphi 32 * "
                "ntheta 32; both from examples/2_Intermediate/"
                "stage_two_optimization.py (module-level ncoils/order/nphi/ntheta) "
                "and examples/jax/2_Intermediate/stage_two_optimization.py"
                "::_build_problem"
            ),
        ),
        observable_names=_STAGE_TWO_OBSERVABLES,
    ),
    "planar": MirrorFamily(
        name="planar",
        native_script=REPO_ROOT
        / "examples/2_Intermediate/stage_two_optimization_planar_coils.py",
        mirror_module=REPO_ROOT
        / "examples/jax/2_Intermediate/stage_two_optimization_planar_coils.py",
        native_budget=400,
        native_budget_source=(
            "examples/2_Intermediate/stage_two_optimization_planar_coils.py::MAXITER"
        ),
        minimize_call_sources=(
            "examples/2_Intermediate/stage_two_optimization_planar_coils.py "
            "(module-level minimize call, stage one: LENGTH_WEIGHT as declared)",
            "examples/2_Intermediate/stage_two_optimization_planar_coils.py "
            "(module-level minimize call, stage two: after LENGTH_WEIGHT *= 0.1)",
        ),
        evaluation_marker="║∇J║",
        mirror_policy={
            "optimizer": "Driver.SIMSOPT_LBFGSB via serial_solve_jax",
            "maxcor": 10,
            "maxcor_symbol": _STAGE_TWO_MIRROR_MAXCOR_SYMBOL,
            "rtol": 1.0e-8,
            "atol": 1.0e-7,
            "source": (
                "examples/jax/2_Intermediate/"
                "stage_two_optimization_planar_coils.py::solve"
            ),
        },
        declared_dimensions=_mirror_dimensions(
            coils=16,
            base_curves=4,
            base_curve_order=5,
            surface_points=1024,
            native_quadrature=LaneQuadrature(
                points=75,
                source=(
                    "examples/2_Intermediate/stage_two_optimization_planar_coils.py "
                    "(module-level create_equally_spaced_planar_curves call passes "
                    f"NO numquadpoints) -> {_NATIVE_PLANAR_QUADRATURE_DEFAULT} "
                    "= 15 * 5"
                ),
            ),
            mirror_quadrature=LaneQuadrature(
                points=100,
                source=(
                    "examples/jax/2_Intermediate/"
                    "stage_two_optimization_planar_coils.py::_build_problem "
                    "(numquadpoints=100 at native_default)"
                ),
            ),
            shared_source=(
                "plan Current Context (U5); coils = 4 base curves * nfp 2 * 2 "
                "(stellsym) via coils_via_symmetries; surface_points = nphi 32 * "
                "ntheta 32; both from examples/2_Intermediate/"
                "stage_two_optimization_planar_coils.py (module-level "
                "ncoils/order/nphi/ntheta) and examples/jax/2_Intermediate/"
                "stage_two_optimization_planar_coils.py::_build_problem"
            ),
        ),
        observable_names=(
            "final_objective",
            "squared_flux",
            "planarity_penalty",
            "linking_number",
            "solver_success",
            "solver_status",
            "solver_iterations",
        ),
    ),
    "coil-forces": MirrorFamily(
        name="coil-forces",
        native_script=REPO_ROOT / "examples/3_Advanced/coil_forces.py",
        mirror_module=REPO_ROOT / "examples/jax/3_Advanced/coil_forces.py",
        native_budget=400,
        native_budget_source="examples/3_Advanced/coil_forces.py::MAXITER",
        minimize_call_sources=(
            "examples/3_Advanced/coil_forces.py "
            "(module-level minimize call, stage one: LENGTH_WEIGHT as declared)",
            "examples/3_Advanced/coil_forces.py "
            "(module-level minimize call, stage two: after LENGTH_WEIGHT *= 0.1)",
        ),
        evaluation_marker="B2Energy=",
        mirror_policy={
            "optimizer": "Driver.SIMSOPT_LBFGSB via serial_solve_jax",
            "maxcor": None,
            "maxcor_expression": "min(max_steps, 300)",
            "rtol": 1.0e-15,
            "atol": 1.0e-8,
            "source": "examples/jax/3_Advanced/coil_forces.py::_run_stage",
        },
        declared_dimensions=_mirror_dimensions(
            coils=12,
            base_curves=3,
            base_curve_order=5,
            surface_points=1024,
            native_quadrature=LaneQuadrature(
                points=75,
                source=(
                    "examples/3_Advanced/coil_forces.py "
                    "(module-level create_equally_spaced_curves call passes NO "
                    f"numquadpoints) -> {_NATIVE_QUADRATURE_DEFAULT} = 15 * 5"
                ),
            ),
            mirror_quadrature=LaneQuadrature(
                points=75,
                source=(
                    "examples/jax/3_Advanced/coil_forces.py::_build_problem "
                    "(numquadpoints=75 at native_default, which is what the "
                    "native default happens to evaluate to for this family)"
                ),
            ),
            shared_source=(
                "plan Current Context (U5); coils = 3 base curves * nfp 2 * 2 "
                "(stellsym) via coils_via_symmetries; surface_points = nphi 32 * "
                "ntheta 32; both from examples/3_Advanced/coil_forces.py "
                "(module-level ncoils/order/nphi/ntheta) and "
                "examples/jax/3_Advanced/coil_forces.py::_build_problem"
            ),
            force_coils=3,
            force_pair_terms_per_evaluation=202_500,
            force_source=(
                "src/simsopt_jax_adapters/field/force.py::curve_force_norms_pure; "
                "num_force_coils from examples/jax/3_Advanced/coil_forces.py"
                "::_force_config"
            ),
        ),
        observable_names=(
            "final_objective",
            "squared_flux",
            "force_objective",
            "maximum_force",
            "vacuum_energy",
            "solver_success",
            "solver_status",
            "solver_iterations",
        ),
    ),
}

RCLS_FAMILY = "rcls-ports"
COMPARE_COMMAND = "compare-artifacts"
FAMILIES = (*MIRROR_FAMILIES, RCLS_FAMILY)

#: What the plan predicts the RCLS device solve looks like
#: (``src/simsopt_jax_adapters/solve/wireframe.py``
#: ``::_regularized_constrained_least_squares_core`` -- the complete QR of
#: C-transpose and the lstsq on the augmented system).  Recorded beside the
#: measured shapes so a drift shows up as data, not as a surprise.
RCLS_PLAN_EXPECTATION = {
    "qr_of_constraint_transpose": [497, 254],
    "lstsq_augmented_system": [1521, 243],
    "source": f"{PLAN} P4.3 / Open Questions",
}

_SHARED_DISCLOSURES = (
    (
        "serial_solve_jax writes a bounded-objective log per solve: two extra host "
        "materializations per solve, not per iteration "
        "(src/simsopt_jax/solve/serial.py::serial_solve_jax, its "
        "_write_bounded_objective_log epilogue). Each mirror runs two stages, so "
        "the JAX leg pays it twice per solve call."
    ),
    (
        "The fused on-device L-BFGS driver is already the mirrors' default lane "
        "(Driver.SIMSOPT_LBFGSB -> dispatch.minimize -> fused_stepwise, "
        "src/simsopt_jax/solve/dispatch.py::_legacy_lbfgsb_options); this probe "
        "times what ships, it does not add a lever."
    ),
    (
        "Every leg's environment is built by probe_conventions.pinned_environment: "
        "the whole OMP_/GOMP_/KMP_/MKL_/OPENBLAS_/NUMEXPR_/VECLIB_/BLIS_/NUMBA_/"
        "JAX_/XLA_/SIMSOPT_/CUDA_/MPI4PY_ family is deleted from the inherited "
        "shell and only the pins below are put back, so SIMSOPT_BACKEND_MODE, "
        "XLA_FLAGS, KMP_AFFINITY, OMP_PROC_BIND, CUDA_VISIBLE_DEVICES and "
        "JAX_COMPILATION_CACHE_DIR cannot survive into a lane that did not ask "
        "for them. CI/GITHUB_ACTIONS/IN_GITHUB_ACTIONS match none of those "
        "prefixes and are popped by this probe on top, so the native script takes "
        "its non-CI branch (src/simsopt/util/__init__.py::in_github_actions reads "
        "CI)."
    ),
    (
        "The native lane is pinned to JAX_PLATFORMS=cpu rather than merely having "
        "the parent's selection scrubbed: simsopt's own guard "
        "(src/simsopt/geo/jit.py, its module-level JAX_PLATFORMS/JAX_PLATFORM_NAME "
        "check) only applies when nothing is set, and an explicit pin is what the "
        "artifact can show."
    ),
    "Probes measure; charters certify. Nothing in this artifact is a claim.",
)

_MIRROR_DISCLOSURES = (
    (
        "minimize_region_seconds runs from the last Taylor 'err' line to the last "
        "objective-evaluation line, so it contains BOTH minimize calls and the "
        "inter-stage VTK/surface writes between them. It is a region, not a pure "
        "solver timer; process_wall_seconds is the unambiguous native number."
    ),
    (
        "taylor_last_error_elapsed_seconds is elapsed time from process start to "
        "the last Taylor 'err' line. It is NOT the duration of the Taylor test: it "
        "also contains interpreter startup, imports, and the whole problem build."
    ),
    (
        "The JAX leg's solve_call_seconds is the mirror's whole solve() call, which "
        "includes host problem construction, the five-epsilon Taylor evaluation, and "
        "both stages -- the same content as the native leg's process wall minus "
        "interpreter startup and VTK output. Compared against the native minimize "
        "region alone it is conservative."
    ),
    (
        "Both lanes run the FULL two-stage length-weight ladder; neither the native "
        "script nor the mirror has a selector that stops after stage one, so probing "
        "stage one alone would mean editing both lanes."
    ),
    (
        "shipped-vs-shipped comparison: the ratio mixes optimizer policy (maxcor 300 "
        "vs 10; tol 1e-15 vs the mirror's) with hardware -- this matches the "
        "device-assignment row semantics (where to launch this example as shipped) "
        "and must never be quoted as a matched-work number. Both policies are "
        "published as native_policy and mirror_policy on both lanes."
    ),
    (
        "The native mirror lane runs the UNMODIFIED example script as a child "
        "process, so no identity can be captured inside the child: the script "
        "imports nothing from this probe and exits before the parent could ask it "
        "anything. Each leg therefore publishes parent_identity (the launching "
        "process), child_pid, and child_environment_pinned -- the exact pinned "
        "subset handed to Popen -- and nothing that claims to describe the child's "
        "own runtime state."
    ),
    (
        "The mirror's solve() builds a fresh jitted problem on every call, so an "
        f"in-memory warm leg does not exist for this family. Leg 0 is labelled "
        f"{COLD_IN_PROCESS}; every later leg is labelled "
        f"{REPEAT_PERSISTENT_CACHE} and is warm only through the XLA persistent "
        "compilation cache, which is why --compile-cache is mandatory whenever "
        "--repeat > 1. The JAX summary excludes leg 0 from its warm statistics."
    ),
    (
        "Every native mirror leg is a fresh process and is labelled "
        f"{FRESH_PROCESS_COLD}: the series is cold-only, there is no leg 0 to "
        "exclude, and there is no native warm counterpart to compare a warm JAX "
        "number against. The native summary therefore publishes minimum_seconds "
        "and median_seconds over ALL legs, and the JAX summary publishes "
        "warm_samples and warm_sample_count beside the median so a reader always "
        "sees n. Every summary also republishes the per-leg leg_kinds it split on."
    ),
    (
        "native_final_objective is parsed from the last per-evaluation stdout line, "
        "which the native scripts print with the '.1e' format: two significant "
        "digits. The endpoint comparison is a coarse diagnostic that can detect a "
        "lane solving a different problem, not a parity check."
    ),
)

_RCLS_DISCLOSURES = (
    (
        "RCLS is a direct equality-constrained least-squares solve (complete QR of "
        "C-transpose, then lstsq on the reduced system). Bit-identity between lanes "
        "is NOT expected: the JAX path reassociates the same algebra on device. The "
        "measured differences are reported as numbers and gated at the "
        "native_workflow tolerance bucket (src/simsopt_jax/parity_tolerances.py)."
    ),
    (
        "optimize_wireframe_seconds is the native example's own timed region "
        "(examples/2_Intermediate/wireframe_rcls_with_ports.py, its module-level "
        "t0..t1 optimize_wireframe call) and includes the normal-field matrix "
        "assembly plus the post-solve WireframeField rebuild. solve_only_seconds "
        "calls rcls_wireframe on the already-assembled matrices; that is the region "
        "matched against the JAX device solve."
    ),
    (
        "Both lanes consume one shared JAX-free problem build, so the geometry, the "
        "port collision constraints, and the poloidal-current constraint are "
        "bit-identical inputs on both sides of the comparison."
    ),
    (
        "rcls_wireframe_jax recomputes the host constraint matrices inside the timed "
        "device-solve region (src/simsopt_jax_adapters/solve/wireframe.py"
        "::rcls_wireframe_jax, its wframe.constraint_matrices/"
        "unconstrained_segments calls), exactly as native "
        "src/simsopt/solve/wireframe_optimization.py::rcls_wireframe does, so the "
        "two regions match."
    ),
    (
        "The JAX device-solve clock stops after block_until_ready on the FULL result "
        "pytree (x, f_B, f_R, f), not on x alone: the native lane's rcls_wireframe "
        "returns all four before its clock stops, so blocking on x only would have "
        "left the three objective reductions outside the timed window."
    ),
    (
        "The currents comparison is published twice: full_vector covers all "
        "n_segments entries including the constrained segments, whose currents are "
        "exactly zero in both lanes and therefore inflate the bitwise-identical "
        "count; free_segments covers only the n_free unconstrained segments, which "
        "are the unknowns the solver actually solved for."
    ),
    (
        "Both RCLS lanes run in one process and repeat the same call, so on BOTH "
        f"lanes leg 0 is labelled {COLD_IN_PROCESS} and every later leg is labelled "
        f"{WARM_IN_PROCESS}. The warm statistics exclude leg 0 symmetrically: the "
        "native lane pays first touch too (BLAS pool spin-up, page faults, lazy "
        "LAPACK binding), and folding its leg 0 into a native median while the JAX "
        "lane drops its own would bias the ratio toward the GPU. For this family "
        "the warm legs reuse a live compiled program, not a persistent cache."
    ),
    (
        "The currents comparison is gated on configuration metadata carried inside "
        "the saved archive (family, plasma resolution, wireframe phi/theta, "
        "n_segments, n_free, constraints_p, regularization weight, and a digest of "
        "the shared free-segment index vector). Shape agreement is NOT identity: "
        "--budget-override changes the plasma resolution without changing "
        "n_segments, so two dumps from different problems have identical shapes. "
        "The lane label rides along but is excluded from the gate, since a "
        "cross-lane comparison is exactly the case where it must differ."
    ),
)


def _json_ready(value: object) -> object:
    """Coerce one observable into something ``json.dumps`` accepts."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return _json_ready(value.item())
    raise ProbeConventionError(f"probe cannot serialize {type(value)!r}")


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _publishable_observables(
    observables: Mapping[str, object], names: Sequence[str]
) -> dict[str, object]:
    """The mirror's observables, in family order, made canonical-JSON-safe.

    A diverged mirror solve returns non-finite floats in exactly the same
    observables a converged one returns numbers in, so they go through
    :func:`_endpoint_fields` for the same reason the top-level endpoints do.
    Non-float observables (``solver_success``, ``solver_status``,
    ``solver_iterations``) pass through untouched: they cannot be non-finite.
    """
    published: dict[str, object] = {}
    for name in names:
        if name in observables:
            value = _json_ready(observables[name])
            published.update(
                _endpoint_fields(name, value)
                if isinstance(value, float)
                else {name: value}
            )
    return published


def _endpoint_fields(name: str, value: float) -> dict[str, object]:
    """One endpoint as publishable JSON: the number, or ``null`` plus a sentinel.

    ``write_probe_artifact`` refuses ``NaN``/``Infinity`` (``allow_nan=False``),
    and it refuses them *after* the legs have run: a diverged solve would take
    the whole artifact down with its endpoint -- timings, identity, disclosures
    and all -- and the run would have to be repeated to learn what it had
    already measured.  So the payload is made finite by construction here
    instead of being caught later: a non-finite value is published as ``null``
    beside a string ``<name>_sentinel`` carrying its ``repr``, which is both
    readable and canonical JSON.  This is the pattern
    ``benchmarks/pm_gpmo_probes.py`` landed for its own divergent endpoints.
    """
    if math.isfinite(value):
        return {name: value}
    return {name: None, f"{name}{SENTINEL_SUFFIX}": repr(value)}


def _nonfinite_disclosure(*published: Mapping[str, object]) -> list[str]:
    """The NON-FINITE note for whichever of ``published`` carry a sentinel.

    Derived from the published fields themselves rather than from a second
    record of what went wrong, so the note and the nulls cannot disagree.
    """
    names = sorted(
        {
            name.removesuffix(SENTINEL_SUFFIX)
            for mapping in published
            for name in mapping
            if name.endswith(SENTINEL_SUFFIX)
        }
    )
    if not names:
        return []
    return [
        "NON-FINITE ENDPOINT: "
        + ", ".join(names)
        + " could not be published as numbers and are null, with the offending "
        "value in the matching '_sentinel' field. The legs ran and their timings "
        "stand; the endpoint does not, and no kill verdict may be read off it."
    ]


def _work_asymmetry_disclosure(dimensions: Mapping[str, object]) -> str:
    """State, per family, whether the two lanes do the same per-evaluation work.

    The kill rule is a bare ratio against ``kill_threshold_warm`` (1.0x warm),
    and a ratio only means what its two lanes were doing.  Where the mirror
    evaluates more Biot-Savart pairs per evaluation than the native script it is
    timed against, the mismatch pushes the ratio *down*, so a sub-1.0x reading
    is exactly the reading it can manufacture -- which is why the direction is
    named here rather than left for the reader to work out.
    """
    per_lane = dimensions["per_lane"]
    native = per_lane["native"]
    mirror = per_lane["jax-gpu"]
    ratio = float(dimensions["jax_over_native_pairs_per_evaluation"])
    if ratio == 1.0:
        return (
            "Per-evaluation work IS matched for this family: both lanes evaluate "
            f"{native['curve_quadrature_points']} curve quadrature points per coil "
            f"and {native['biot_savart_pairs_per_evaluation']} Biot-Savart pairs "
            "per evaluation. The native lane reaches that number through the "
            "create_equally_spaced_curves 15*order default and the mirror through "
            "an explicit numquadpoints; they agree, and no work correction applies "
            "to the ratio."
        )
    return (
        "PER-EVALUATION WORK IS NOT MATCHED: the native script passes no "
        "numquadpoints and takes the create_equally_spaced_curves 15*order "
        f"default ({native['curve_quadrature_points']} points, "
        f"{native['biot_savart_pairs_per_evaluation']} Biot-Savart pairs per "
        f"evaluation), while the mirror passes numquadpoints="
        f"{mirror['curve_quadrature_points']} explicitly "
        f"({mirror['biot_savart_pairs_per_evaluation']} pairs). The JAX lane "
        f"therefore does {ratio:.3f}x MORE Biot-Savart work per evaluation than "
        "the lane it is timed against, and the asymmetry runs AGAINST the JAX "
        f"lane. A warm ratio below kill_threshold_warm ({KILL_THRESHOLD_WARM}) "
        "for this family may therefore be an artifact of the quadrature mismatch "
        "rather than a verdict on the hardware: a kill decision here is not "
        f"readable off the ratio alone, it must account for the {ratio:.3f}x work "
        "asymmetry, and only a matched-quadrature rerun settles it. A ratio ABOVE "
        "the threshold is unaffected in direction -- the JAX lane would have won "
        "while doing more work."
    )


def _mirror_disclosures(family: MirrorFamily) -> list[str]:
    """Every disclosure a mirror-family artifact carries, in publication order."""
    return [
        *_SHARED_DISCLOSURES,
        *_MIRROR_DISCLOSURES,
        _work_asymmetry_disclosure(family.declared_dimensions),
    ]


def _pinned_subset(environment: Mapping[str, str]) -> dict[str, str]:
    """The variables a leg's environment carries *because this probe pinned them*.

    ``pinned_environment`` deletes every :data:`SCRUBBED_ENVIRONMENT_PREFIXES`
    variable before pinning, so any prefix-matching name still present in the
    result was put there deliberately.  The handful this module pins itself is
    named in :data:`_EXTRA_PINNED_KEYS`.  Deriving the subset instead of listing
    it keeps the recorded pin honest when the convention adds a variable.
    """
    return {
        name: value
        for name, value in sorted(environment.items())
        if name.startswith(SCRUBBED_ENVIRONMENT_PREFIXES) or name in _EXTRA_PINNED_KEYS
    }


def _native_child_environment(omp: int, cache_dir: Path | None) -> dict[str, str]:
    """The environment one native child runs under.

    ``pinned_environment`` owns the scrub and the numerical pins; this function
    adds only what is specific to running an example *script* as a child: the CI
    pop (no prefix of the convention's scrub matches ``CI``), the documented
    import path, and unbuffered stdout, which is what makes the parent's arrival
    timestamps a timeline rather than a flush artifact.
    """
    environment = pinned_environment(
        lane="native",
        omp=omp,
        jax_platforms="cpu",
        compile_cache_dir=cache_dir,
    )
    for name in _CI_VARIABLES:
        environment.pop(name, None)
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in CHILD_PYTHONPATH)
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def _pin_process_environment(
    lane: str, omp: int, cache_dir: Path | None
) -> dict[str, str]:
    """Pin this process's own environment, before anything numerical is loaded.

    The deferred-import discipline at the top of this module is what makes this
    a real pin: ``OMP_NUM_THREADS`` and its BLAS siblings are read when the
    runtimes initialize, and the JAX persistent cache is read when JAX
    initializes.  If any of them is already imported the pin would be recorded
    in the artifact while the leg ran at whatever the shell had, so this refuses
    rather than pinning cosmetically.

    ``os.environ`` is replaced rather than updated: the scrub is the load-bearing
    half of ``pinned_environment`` and updating in place would leave every
    inherited selector alive in this process.
    """
    loaded = [name for name in _PIN_BLOCKING_MODULES if name in sys.modules]
    if loaded:
        raise ProbeConfigurationError(
            f"cannot pin the environment: {', '.join(loaded)} already imported; "
            "the threading and compilation-cache variables are read at module "
            "initialization, so this pin would be recorded but not applied"
        )
    environment = pinned_environment(
        lane=lane,
        omp=omp,
        jax_platforms="cpu" if lane == "native" else None,
        compile_cache_dir=cache_dir,
    )
    for name in _CI_VARIABLES:
        environment.pop(name, None)
    os.environ.clear()
    os.environ.update(environment)
    return environment


def _openmp_readback(requested_omp: int) -> int | None:
    """``omp_get_max_threads()`` for an in-process leg, gated against the request.

    ``None`` is "this process has no OpenMP runtime at all", a host fact that is
    recorded and is not a disagreement.  Any other value that differs from the
    request means the leg would run at a thread count the artifact does not
    record, and the probe stops.
    """
    observed = observed_openmp_threads()
    if observed is not None and observed != requested_omp:
        raise ProbeConfigurationError(
            f"OpenMP readback disagrees with the pin: omp_get_max_threads()="
            f"{observed} but --omp={requested_omp}; the leg would run at a thread "
            "count no field in this artifact records"
        )
    return observed


def _append_leg(
    ledger_path: Path,
    *,
    family: str,
    lane: str,
    leg: Mapping[str, object],
    timer: str,
) -> None:
    """Append one finished leg to the executed-order ledger."""
    append_leg_ledger(
        ledger_path,
        {
            "schema": LEDGER_SCHEMA,
            "probe": "marginal-quartet",
            "family": family,
            "lane": lane,
            "index": int(leg["index"]),
            "leg_kind": str(leg["leg_kind"]),
            "timer": timer,
            "seconds": float(leg[timer]),
            "parent_pid": os.getpid(),
            "child_pid": leg.get("child_pid"),
            "finished_timestamp_ns": time.time_ns(),
            "finished_wallclock_utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def _final_objective_from_line(family: MirrorFamily, line: str) -> float:
    """The ``J=`` value off one native per-evaluation stdout line.

    Every native script in this quartet opens its evaluation line with
    ``J={value:.1e},``, so the endpoint is the first comma-delimited field.  The
    format carries two significant digits, which is what
    ``native_final_objective_precision`` records.
    """
    head, separator, _ = line.partition(",")
    if not head.startswith("J=") or not separator:
        raise ProbeConventionError(
            f"{family.name}: cannot read a final objective from {line!r}; the "
            "native script's evaluation line no longer starts with 'J=<value>,'"
        )
    return float(head[2:])


def _timeline_regions(
    family: MirrorFamily,
    timeline: Sequence[tuple[float, str]],
) -> dict[str, object]:
    """Split one native child's stdout into its Taylor and minimize regions.

    The native scripts carry no timestamps, so the regions are derived from the
    arrival times of lines the parent reads off an unbuffered pipe.  Both
    boundaries are structural: the Taylor test ends at its last ``err`` line, and
    the solver ends at the last per-evaluation objective line.
    """
    evaluations = [
        (elapsed, line)
        for elapsed, line in timeline
        if line.startswith("J=") and family.evaluation_marker in line
    ]
    errors = [elapsed for elapsed, line in timeline if line.startswith("err ")]
    if len(errors) != TAYLOR_ERROR_LINES:
        raise ProbeConventionError(
            f"{family.name}: expected {TAYLOR_ERROR_LINES} Taylor 'err' lines, "
            f"read {len(errors)}; the native script's output shape changed"
        )
    if len(evaluations) <= TAYLOR_EVALUATIONS:
        raise ProbeConventionError(
            f"{family.name}: read {len(evaluations)} objective lines, which is not "
            f"more than the {TAYLOR_EVALUATIONS} the Taylor test alone emits; the "
            "solver produced nothing to time"
        )
    return {
        "taylor_last_error_elapsed_seconds": errors[-1],
        "taylor_last_error_elapsed_definition": (
            "process start to the last Taylor 'err' line; contains interpreter "
            "startup, imports and the problem build, so it is an elapsed-to-marker "
            "time and not the duration of the Taylor test"
        ),
        "minimize_region_seconds": evaluations[-1][0] - errors[-1],
        "minimize_region_definition": (
            "last Taylor 'err' line to last objective-evaluation line; contains "
            "both minimize calls and the inter-stage VTK writes"
        ),
        "evaluations_total": len(evaluations),
        "evaluations_in_minimize_region": len(evaluations) - TAYLOR_EVALUATIONS,
        "terminal_evaluation_lines": [line for _, line in evaluations[-2:]],
        **_endpoint_fields(
            "native_final_objective",
            _final_objective_from_line(family, evaluations[-1][1]),
        ),
        "native_final_objective_source": (
            "last per-evaluation stdout line matching "
            f"{family.evaluation_marker!r}, first comma-delimited field"
        ),
        "native_final_objective_precision": "printed at '.1e': two significant digits",
    }


def _run_native_child(
    family: MirrorFamily,
    omp: int,
    cache_dir: Path | None,
    workdir: Path,
) -> dict[str, object]:
    """Run one native example as a child process and time it."""
    command = [str(NATIVE_INTERPRETER), str(family.native_script)]
    environment = _native_child_environment(omp, cache_dir)
    started = perf_counter()
    child = subprocess.Popen(
        command,
        cwd=workdir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    timeline: list[tuple[float, str]] = []
    for line in child.stdout:
        timeline.append((perf_counter() - started, line.rstrip("\n")))
    returncode = child.wait()
    process_wall_seconds = perf_counter() - started
    if returncode != 0:
        tail = "\n".join(line for _, line in timeline[-20:])
        raise ProbeConventionError(
            f"{family.name} native child exited {returncode}\n{tail}"
        )
    leg: dict[str, object] = {
        "lane": "native",
        "command": command,
        "child_pid": child.pid,
        "child_environment_pinned": _pinned_subset(environment),
        "child_identity_available": False,
        "child_identity_unavailable_because": (
            "this leg runs the unmodified example script, which imports nothing "
            "from this probe and exits before it could be asked; the pinned "
            "environment handed to Popen and the child pid are what the parent can "
            "honestly attest to"
        ),
        "process_wall_seconds": process_wall_seconds,
        "returncode": returncode,
        "stdout_line_count": len(timeline),
        "requested_omp": omp,
    }
    leg.update(_timeline_regions(family, timeline))
    return leg


def _native_mirror_legs(
    family: MirrorFamily,
    repeat: int,
    omp: int,
    cache_dir: Path | None,
    ledger_path: Path,
) -> list[dict[str, object]]:
    legs: list[dict[str, object]] = []
    for index in range(repeat):
        parent_identity = runtime_identity("native")
        with tempfile.TemporaryDirectory(prefix="marginal-quartet-native-") as name:
            leg = _run_native_child(family, omp, cache_dir, Path(name))
        record: dict[str, object] = {
            "index": index,
            "leg_kind": FRESH_PROCESS_COLD,
            "parent_identity": parent_identity,
            **leg,
        }
        legs.append(record)
        _append_leg(
            ledger_path,
            family=family.name,
            lane="native",
            leg=record,
            timer="process_wall_seconds",
        )
    return legs


def _load_mirror_module(family: MirrorFamily):
    """Load the shipped JAX mirror from its example path.

    The mirror -- not this file -- owns the ``native_default`` scale, the stage
    weights, and the solver options, so the probe executes the mirror's own
    ``solve`` instead of a copy that could drift away from it.
    """
    import importlib.util

    module_name = f"marginal_quartet_mirror_{family.name.replace('-', '_')}"
    specification = importlib.util.spec_from_file_location(
        module_name, family.mirror_module
    )
    if specification is None or specification.loader is None:
        raise ProbeConventionError(f"cannot load the mirror at {family.mirror_module}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _jax_mirror_legs(
    family: MirrorFamily,
    repeat: int,
    budget: int,
    omp: int,
    ledger_path: Path,
) -> tuple[list[dict[str, object]], float]:
    """Run the mirror's ``native_default`` solve ``repeat`` times in one process.

    Leg 0 is ``cold_in_process``.  Every later leg is ``repeat_persistent_cache``
    and is *not* warm in memory: ``solve`` rebuilds its jitted problem on every
    call, so the only warmth available to a repeat is the XLA persistent
    compilation cache, which ``--compile-cache`` selects and which
    :func:`_validate` requires whenever ``--repeat > 1``.

    The mirror's ``solve`` returns host floats, so the timed region is already
    fully synchronized -- ``solve_standard_stage_two`` blocks at
    ``src/simsopt_jax/examples/stage_two_standard.py:336-338`` and the mirrors
    finish with ``jax.device_get``.
    """
    started = perf_counter()
    module = _load_mirror_module(family)
    mirror_import_seconds = perf_counter() - started

    import jax

    if not bool(jax.config.read("jax_enable_x64")):
        raise ProbeConfigurationError(
            "the JAX lane requires JAX_ENABLE_X64=1; this process is running fp32"
        )
    if int(module.NATIVE_ITERATIONS) != family.native_budget:
        raise ProbeConfigurationError(
            f"{family.name}: mirror NATIVE_ITERATIONS={module.NATIVE_ITERATIONS} no "
            f"longer matches the native budget {family.native_budget} at "
            f"{family.native_budget_source}; the budgets are unmatched"
        )

    legs: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="marginal-quartet-mirror-") as name:
        output_directory = Path(name)
        for index in range(repeat):
            identity = runtime_identity("jax-gpu")
            observed_omp_threads = _openmp_readback(omp)
            started = perf_counter()
            result = module.solve(output_directory, budget, "native_default")
            solve_call_seconds = perf_counter() - started
            record: dict[str, object] = {
                "index": index,
                "identity": identity,
                "lane": "jax-gpu",
                "leg_kind": (
                    COLD_IN_PROCESS if index == 0 else REPEAT_PERSISTENT_CACHE
                ),
                "requested_omp": omp,
                "observed_omp_threads": observed_omp_threads,
                "solve_call_seconds": solve_call_seconds,
                "example_id": result.example_id,
                "example_status": result.status,
                "observables": _publishable_observables(
                    result.observables, family.observable_names
                ),
            }
            legs.append(record)
            _append_leg(
                ledger_path,
                family=family.name,
                lane="jax-gpu",
                leg=record,
                timer="solve_call_seconds",
            )
    return legs, mirror_import_seconds


@dataclass(frozen=True)
class RclsProblem:
    """One built ``wireframe_rcls_with_ports`` problem, shared by both lanes."""

    plasma: object
    wireframe: object
    ports: object
    free_segments: object
    dimensions: dict[str, object]
    build_seconds: float


def _build_rcls_problem(resolution: int) -> RclsProblem:
    """Build the ports problem through public APIs, with no JAX in the path.

    This is the native example's construction
    (``examples/2_Intermediate/wireframe_rcls_with_ports.py:73-110``), which the
    mirror reproduces (``examples/jax/2_Intermediate/wireframe_rcls_with_ports.py:30-98``).
    One builder serves both lanes so the comparison of final currents is a
    comparison of solvers and not of geometry.
    """
    import numpy as np
    from simsopt.geo import CircularPort, PortSet, SurfaceRZFourier, ToroidalWireframe

    started = perf_counter()
    equilibrium = REPO_ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"
    plasma = SurfaceRZFourier.from_vmec_input(
        str(equilibrium),
        nphi=resolution,
        ntheta=resolution,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_vmec_input(str(equilibrium))
    wireframe_surface.extend_via_projected_normal(RCLS_SURFACE_DISTANCE)
    wireframe = ToroidalWireframe(
        wireframe_surface, RCLS_WIREFRAME_PHI, RCLS_WIREFRAME_THETA
    )

    ports = PortSet()
    gamma = wireframe_surface.gamma()
    normal = wireframe_surface.normal()
    for phi in (np.pi / 8.0, 3.0 * np.pi / 8.0):
        phi_index = int(
            np.argmin(np.abs((0.5 / np.pi) * phi - wireframe_surface.quadpoints_phi))
        )
        for theta in (np.pi / 4.0, 7.0 * np.pi / 4.0):
            theta_index = int(
                np.argmin(
                    np.abs((0.5 / np.pi) * theta - wireframe_surface.quadpoints_theta)
                )
            )
            origin = gamma[phi_index, theta_index]
            axis = normal[phi_index, theta_index]
            ports.add_ports(
                [
                    CircularPort(
                        ox=origin[0],
                        oy=origin[1],
                        oz=origin[2],
                        ax=axis[0],
                        ay=axis[1],
                        az=axis[2],
                        ir=0.1,
                        thick=0.005,
                        l0=-0.15,
                        l1=0.15,
                    )
                ]
            )
    ports = ports.repeat_via_symmetries(wireframe_surface.nfp, True)
    wireframe.constrain_colliding_segments(ports.collides, gap=RCLS_PORT_GAP)

    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = -2.0 * np.pi * plasma.get_rc(0, 0) * RCLS_FIELD_ON_AXIS / mu0
    wireframe.set_poloidal_current(poloidal_current)

    constraint, _ = wireframe.constraint_matrices(
        assume_no_crossings=False,
        remove_constrained_segments=True,
    )
    build_seconds = perf_counter() - started

    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    n_segments = int(wireframe.n_segments)
    n_free = int(free_segments.size)
    constraints = int(np.shape(constraint)[0])
    test_points = int(resolution * resolution)
    dimensions = {
        "plasma_resolution": resolution,
        "test_points_m": test_points,
        "n_segments": n_segments,
        "n_free": n_free,
        "constraints_p": constraints,
        "qr_of_constraint_transpose": [n_free, constraints],
        "lstsq_augmented_system": [test_points + n_free, n_free - constraints],
        "plan_expectation": RCLS_PLAN_EXPECTATION,
        "poloidal_current": float(poloidal_current),
    }
    return RclsProblem(
        plasma=plasma,
        wireframe=wireframe,
        ports=ports,
        free_segments=free_segments,
        dimensions=dimensions,
        build_seconds=build_seconds,
    )


def _native_rcls_legs(
    problem: RclsProblem, repeat: int, omp: int, ledger_path: Path
) -> tuple[list[dict[str, object]], object]:
    """Time the native RCLS: the example's own region, then the matched solve.

    These legs are in-process repeats, exactly like :func:`_jax_rcls_legs`, so
    they are labelled the same way: leg 0 is ``cold_in_process`` and every later
    leg is ``warm_in_process``.  The native lane has no compile to pay, but it
    does pay first touch -- the OpenMP/BLAS pool spins up, pages fault in, the
    LAPACK path binds lazily -- and :func:`_summarize` excludes leg 0 from the
    warm statistics on both lanes for that reason.
    """
    import numpy as np
    from simsopt.solve.wireframe_optimization import optimize_wireframe, rcls_wireframe

    parameters = {"reg_W": RCLS_REGULARIZATION_WEIGHT}
    legs: list[dict[str, object]] = []
    currents = None
    for index in range(repeat):
        identity = runtime_identity("native")
        observed_omp_threads = _openmp_readback(omp)
        started = perf_counter()
        result = optimize_wireframe(
            problem.wireframe,
            "rcls",
            parameters,
            surf_plas=problem.plasma,
            verbose=False,
        )
        optimize_wireframe_seconds = perf_counter() - started

        started = perf_counter()
        solution, f_b, f_r, total = rcls_wireframe(
            problem.wireframe,
            result["Amat"],
            result["bvec"],
            RCLS_REGULARIZATION_WEIGHT,
            False,
            False,
        )
        solve_only_seconds = perf_counter() - started
        currents = np.asarray(solution, dtype=np.float64).ravel()
        record: dict[str, object] = {
            "index": index,
            "identity": identity,
            "lane": "native",
            "leg_kind": COLD_IN_PROCESS if index == 0 else WARM_IN_PROCESS,
            "requested_omp": omp,
            "observed_omp_threads": observed_omp_threads,
            "optimize_wireframe_seconds": optimize_wireframe_seconds,
            "solve_only_seconds": solve_only_seconds,
            "f_B": float(f_b),
            "f_R": float(f_r),
            "f": float(total),
            "response_shape": [int(size) for size in np.shape(result["Amat"])],
        }
        legs.append(record)
        _append_leg(
            ledger_path,
            family=RCLS_FAMILY,
            lane="native",
            leg=record,
            timer="solve_only_seconds",
        )
    return legs, currents


def _jax_rcls_legs(
    problem: RclsProblem, repeat: int, omp: int, ledger_path: Path
) -> tuple[list[dict[str, object]], object]:
    """Time the JAX RCLS device solve: leg 0 cold, every later leg warm.

    The device-solve clock stops after ``block_until_ready`` on the whole result
    -- ``x``, ``f_B``, ``f_R`` and ``f`` -- because the native ``rcls_wireframe``
    call this region is matched against returns all four before its own clock
    stops.  Blocking on ``x`` alone would leave three reductions running past the
    end of the timed window.
    """
    import jax
    import numpy as np
    from simsopt_jax_adapters.solve.wireframe import (
        bnorm_obj_matrices_jax,
        rcls_wireframe_jax,
    )

    if not bool(jax.config.read("jax_enable_x64")):
        raise ProbeConfigurationError(
            "the JAX lane requires JAX_ENABLE_X64=1; this process is running fp32"
        )

    legs: list[dict[str, object]] = []
    currents = None
    for index in range(repeat):
        identity = runtime_identity("jax-gpu")
        observed_omp_threads = _openmp_readback(omp)
        started = perf_counter()
        response, target = bnorm_obj_matrices_jax(
            problem.wireframe,
            problem.plasma,
            area_weighted=True,
            verbose=False,
        )
        jax.block_until_ready((response, target))
        assembly_seconds = perf_counter() - started

        started = perf_counter()
        result = rcls_wireframe_jax(
            problem.wireframe,
            response,
            target,
            RCLS_REGULARIZATION_WEIGHT,
            assume_no_crossings=False,
        )
        jax.block_until_ready((result.x, result.f_B, result.f_R, result.f))
        device_solve_seconds = perf_counter() - started
        currents = np.asarray(jax.device_get(result.x), dtype=np.float64).ravel()
        record: dict[str, object] = {
            "index": index,
            "identity": identity,
            "lane": "jax-gpu",
            "leg_kind": COLD_IN_PROCESS if index == 0 else WARM_IN_PROCESS,
            "requested_omp": omp,
            "observed_omp_threads": observed_omp_threads,
            "matrix_assembly_seconds": assembly_seconds,
            "device_solve_seconds": device_solve_seconds,
            "f_B": float(jax.device_get(result.f_B)),
            "f_R": float(jax.device_get(result.f_R)),
            "f": float(jax.device_get(result.f)),
            "response_shape": [int(size) for size in np.shape(response)],
        }
        legs.append(record)
        _append_leg(
            ledger_path,
            family=RCLS_FAMILY,
            lane="jax-gpu",
            leg=record,
            timer="device_solve_seconds",
        )
    return legs, currents


def _difference_statistics(
    measured, reference, label: str, bucket
) -> dict[str, object]:
    """Elementwise agreement of two current vectors, labelled by what they cover.

    Every published statistic is taken over the elements that are finite in
    *both* vectors, and is ``null`` when there are none.  A single ``NaN`` in
    either lane's currents otherwise turns ``max_absolute_difference`` into
    ``NaN``, which ``write_probe_artifact`` refuses (``allow_nan=False``) after
    the run has finished -- so the diverged lane would destroy the artifact
    documenting its divergence.  ``comparable_elements`` and
    ``nonfinite_elements`` say how much of the vector the numbers cover, and
    when any element was dropped the unmasked maximum rides along as a string
    sentinel so the offending value is still on the record.

    ULP distances come from ``probe_conventions.ulp_distance``, which returns
    exact ``uint64`` distances; they are published as integers, never coerced
    through float64, which cannot hold a large one exactly.
    """
    import numpy as np

    difference = np.abs(measured - reference)
    scale = np.maximum(np.abs(measured), np.abs(reference))
    nonzero = scale > 0.0
    relative = np.zeros_like(difference)
    relative[nonzero] = difference[nonzero] / scale[nonzero]
    ulp = ulp_distance(measured, reference)
    finite = np.isfinite(measured) & np.isfinite(reference)
    any_finite = bool(finite.any())
    comparable = int(np.count_nonzero(finite))
    statistics_block: dict[str, object] = {
        "label": label,
        "elements": int(measured.size),
        "comparable_elements": comparable,
        "nonfinite_elements": int(measured.size) - comparable,
        "statistics_domain": (
            "the elements finite in BOTH vectors; every statistic is null when "
            "there are none"
        ),
        "bitwise_identical_elements": (
            int(np.count_nonzero(difference[finite] == 0.0)) if any_finite else 0
        ),
        "exactly_zero_elements": (
            int(np.count_nonzero(scale[finite] == 0.0)) if any_finite else 0
        ),
        "max_absolute_difference": (
            float(np.max(difference[finite])) if any_finite else None
        ),
        "max_relative_difference": (
            float(np.max(relative[finite])) if any_finite else None
        ),
        "max_ulp_distance": int(np.max(ulp[finite])) if any_finite else None,
        "reference_max_absolute_value": (
            float(np.max(np.abs(reference)[finite])) if any_finite else None
        ),
        "within_same_state_value_tolerance": bool(
            np.allclose(
                measured,
                reference,
                rtol=float(bucket["same_state_value_rtol"]),
                atol=float(bucket["same_state_value_atol"]),
            )
        ),
        "within_whole_solve_value_tolerance": bool(
            np.allclose(
                measured,
                reference,
                rtol=float(bucket["whole_solve_value_rtol"]),
                atol=float(bucket["whole_solve_value_atol"]),
            )
        ),
    }
    if comparable != int(measured.size):
        statistics_block[f"max_absolute_difference{SENTINEL_SUFFIX}"] = repr(
            float(np.max(difference))
        )
    return statistics_block


#: What two currents archives must agree on before their difference means
#: anything.  Shape is not on the list because shape is not identity here: the
#: wireframe is ``RCLS_WIREFRAME_PHI`` x ``RCLS_WIREFRAME_THETA`` whatever the
#: plasma resolution is, so ``--budget-override`` produces a *different problem*
#: with an *identical* ``n_segments`` -- two such dumps compared cleanly and
#: reported a difference that measured the resolution, not the lanes.
#: ``free_segments_sha256`` is the strongest member: it proves both lanes were
#: handed the same constraint structure, not merely the same nominal numbers.
#: ``lane`` is deliberately absent -- a cross-lane comparison is exactly the
#: case where it must differ -- and rides along reported instead.
_CURRENTS_IDENTITY_FIELDS = (
    "family",
    "plasma_resolution",
    "wireframe_phi",
    "wireframe_theta",
    "n_segments",
    "n_free",
    "constraints_p",
    "regularization_weight",
    "free_segments_sha256",
)


def _currents_metadata(problem: RclsProblem, lane: str) -> dict[str, object]:
    """The configuration one currents dump was produced under.

    Read out of the built problem rather than re-declared, so the metadata
    cannot describe a configuration the solve did not run at.
    """
    dimensions = problem.dimensions
    return {
        "family": RCLS_FAMILY,
        "lane": lane,
        "plasma_resolution": dimensions["plasma_resolution"],
        "wireframe_phi": RCLS_WIREFRAME_PHI,
        "wireframe_theta": RCLS_WIREFRAME_THETA,
        "n_segments": dimensions["n_segments"],
        "n_free": dimensions["n_free"],
        "constraints_p": dimensions["constraints_p"],
        "regularization_weight": RCLS_REGULARIZATION_WEIGHT,
        "free_segments_sha256": array_sha256(problem.free_segments),
        "schema": SCHEMA,
    }


def _write_currents(path: Path, currents, metadata: Mapping[str, object]) -> None:
    """Save one lane's currents as an ``npz`` archive carrying its own metadata.

    A bare ``.npy`` array cannot say which family, plasma resolution, wireframe
    or regularization weight produced it, so two files that disagree about all
    four compare cleanly and report a difference nobody can attribute.  The
    metadata rides inside the archive as canonical JSON, mirroring
    ``benchmarks/pm_gpmo_probes.py::write_moments`` and
    ``benchmarks/stochastic_stage_two_probe.py::write_endpoint``.

    ``np.savez`` is handed an open file object rather than a path: given a path
    it appends ``.npz`` itself, which would turn an operator's ``foo.npy`` into
    ``foo.npy.npz`` and lose the file the comparison step was told to read.
    ``"xb"`` is the overwrite refusal.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        np.savez(
            handle,
            currents=currents,
            metadata=np.asarray(
                json.dumps(dict(metadata), sort_keys=True, separators=(",", ":"))
            ),
        )


def _load_currents(path: Path) -> tuple[object, dict[str, object]]:
    """One saved currents archive, or a typed refusal naming what it is not."""
    import numpy as np

    archive = np.load(path, allow_pickle=False)
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ProbeConfigurationError(
            f"{path} is a bare array, not a {SCHEMA} currents archive: it carries "
            "no configuration metadata, so a comparison against it could not tell "
            "a cross-lane difference from a cross-configuration one"
        )
    with archive:
        return (
            np.asarray(archive["currents"], dtype=np.float64).ravel(),
            json.loads(str(archive["metadata"].item())),
        )


def _currents_field(metadata: Mapping[str, object], name: str, path: Path) -> object:
    """One metadata field, or a typed refusal naming the field and the file."""
    if name not in metadata:
        raise ProbeConfigurationError(
            f"{path} carries no {name!r} in its currents metadata: expected the "
            f"{SCHEMA} currents schema (identity fields "
            f"{list(_CURRENTS_IDENTITY_FIELDS)} plus 'lane'), got "
            f"{sorted(metadata)}; the file was written by a different or older "
            "probe schema"
        )
    return metadata[name]


def _compare_currents(
    measured, reference_path: Path, problem: RclsProblem, lane: str
) -> dict[str, object]:
    """Compare this lane's currents against another lane's archive, twice.

    Gated first on the configuration metadata both sides carry: two dumps that
    solved different plasma resolutions, different wireframes or different
    regularization weights are not comparable, and a difference between them
    measures the configuration rather than the lanes.  Shape agreement is not
    that gate and never was -- see :data:`_CURRENTS_IDENTITY_FIELDS`.

    ``full_vector`` covers all ``n_segments`` entries, which includes the
    constrained segments: those are exactly zero in both lanes, so they raise the
    bitwise-identical count without either solver having computed anything.
    ``free_segments`` covers only the unconstrained segments
    (``ToroidalWireframe.unconstrained_segments``), which are the unknowns the
    equality-constrained least-squares solve actually solved for.  Both are
    published because either one alone reads as the whole answer.
    """
    import numpy as np
    from simsopt_jax.parity_tolerances import PARITY_LADDER_TOLERANCES

    metadata = _currents_metadata(problem, lane)
    reference, reference_metadata = _load_currents(reference_path)
    mismatched = {
        name: (
            metadata[name],
            _currents_field(reference_metadata, name, reference_path),
        )
        for name in _CURRENTS_IDENTITY_FIELDS
        if metadata[name] != _currents_field(reference_metadata, name, reference_path)
    }
    if mismatched:
        raise ProbeConfigurationError(
            "currents are not comparable; these matched-configuration fields "
            f"differ (this lane vs {reference_path}): {mismatched}"
        )
    if reference.shape != measured.shape:
        raise ProbeConfigurationError(
            f"currents shape {measured.shape} does not match the reference "
            f"{reference.shape} in {reference_path}"
        )
    bucket = PARITY_LADDER_TOLERANCES["native_workflow"]
    indices = np.asarray(problem.free_segments, dtype=np.int64)
    return {
        "reference": str(reference_path),
        "reference_lane": _currents_field(reference_metadata, "lane", reference_path),
        "measured_lane": lane,
        "matched_fields": {name: metadata[name] for name in _CURRENTS_IDENTITY_FIELDS},
        "lane_excluded_from_gate": True,
        "gate_bucket": "native_workflow",
        "gate_source": (
            "src/simsopt_jax/parity_tolerances.py::PARITY_LADDER_TOLERANCES "
            "('native_workflow' bucket)"
        ),
        "same_state_value_rtol": float(bucket["same_state_value_rtol"]),
        "same_state_value_atol": float(bucket["same_state_value_atol"]),
        "whole_solve_value_rtol": float(bucket["whole_solve_value_rtol"]),
        "whole_solve_value_atol": float(bucket["whole_solve_value_atol"]),
        "full_vector": _difference_statistics(
            measured,
            reference,
            "all n_segments entries, constrained segments included (exact zeros)",
            bucket,
        ),
        "free_segments": _difference_statistics(
            measured[indices],
            reference[indices],
            "the n_free unconstrained segments only, the solved unknowns",
            bucket,
        ),
        "bit_identity_expected": False,
    }


def _summarize(
    legs: Sequence[Mapping[str, object]],
    key: str,
    *,
    series_kind: str,
) -> dict:
    """Cold/warm split of one timing series, with no ratio computed.

    The split is decided by the legs' own ``leg_kind`` and by nothing else --
    never by the lane.  A lane-keyed split is what produced the asymmetry this
    replaces: the JAX lane dropped its leg 0 while the native lane folded its
    own leg 0 into a minimum and a median, so for the in-process ``rcls-ports``
    family the native denominator carried a first-touch cost the numerator had
    been allowed to discard, and the resulting ratio was biased toward the GPU
    by exactly that cost.  A series whose leg 0 is ``cold_in_process`` therefore
    excludes it here on either lane; a series of ``fresh_process_cold`` legs has
    no leg to exclude and publishes minimum and median over all of them.

    ``leg_kinds`` is republished beside the numbers so the split is checkable
    from the artifact alone, and ``warm_samples``/``warm_sample_count`` sit
    beside the median because a median over one or two samples is a number a
    reader must be able to discount.  ``series_kind`` says what kind of
    repetition produced them.
    """
    values = [float(leg[key]) for leg in legs]
    kinds = [str(leg["leg_kind"]) for leg in legs]
    if kinds[0] != COLD_IN_PROCESS:
        return {
            "timer": key,
            "series_kind": series_kind,
            "leg_kinds": kinds,
            "cold_leg_excluded": False,
            "seconds": values,
            "sample_count": len(values),
            "minimum_seconds": min(values),
            "median_seconds": _median(values),
        }
    warm = values[1:]
    return {
        "timer": key,
        "series_kind": series_kind,
        "leg_kinds": kinds,
        "cold_leg_excluded": True,
        "seconds": values,
        "cold_seconds": values[0],
        "warm_samples": warm,
        "warm_sample_count": len(warm),
        "warm_min": min(warm) if warm else None,
        "warm_median": _median(warm),
    }


def _configuration(
    arguments: argparse.Namespace,
    budget: int,
    pinned: Mapping[str, str],
) -> dict[str, object]:
    interpreter = NATIVE_INTERPRETER if arguments.lane == "native" else GPU_INTERPRETER
    resolved_output = (
        arguments.output.resolve() if arguments.output is not None else None
    )
    return {
        "family": arguments.family,
        "lane": arguments.lane,
        "requested_omp": arguments.omp,
        "repeat": arguments.repeat,
        "budget": budget,
        "budget_overridden": arguments.budget_override is not None,
        "compile_cache_dir": (
            str(arguments.compile_cache.resolve())
            if arguments.compile_cache is not None
            else None
        ),
        "compile_cache_environment": compile_cache_env(arguments.compile_cache),
        "documented_interpreter": str(interpreter),
        "running_interpreter": sys.executable,
        "output": str(arguments.output) if arguments.output is not None else None,
        "published_under_evidence_root": (
            resolved_output is not None and EVIDENCE_ROOT in resolved_output.parents
        ),
        "leg_ledger": str(arguments.ledger),
        "environment_pin": {
            "source": "benchmarks/probe_conventions.py pinned_environment",
            "scrubbed_prefixes": list(SCRUBBED_ENVIRONMENT_PREFIXES),
            "additional_scrub": list(_CI_VARIABLES),
            "process_environment_pinned": _pinned_subset(pinned),
        },
        "declared_omp_sweep": list(OMP_SWEEP),
        "interleave_owner": (
            "operator; executed order provable from the ledger. Each process runs "
            "one lane, because the two lanes need different interpreters, so no "
            "single process can interleave them and none of them claims to."
        ),
    }


def _resolve_budget(arguments: argparse.Namespace) -> int:
    if arguments.family == RCLS_FAMILY:
        return (
            RCLS_DEFAULT_RESOLUTION
            if arguments.budget_override is None
            else arguments.budget_override
        )
    family = MIRROR_FAMILIES[arguments.family]
    if arguments.budget_override is None:
        return family.native_budget
    if arguments.lane == "native":
        raise ProbeConfigurationError(
            f"{family.name}: the native script hard-codes its budget at "
            f"{family.native_budget_source} and exposes no CLI or environment "
            "selector, so --budget-override cannot reach the native lane; drop it "
            "or probe the JAX lane"
        )
    return arguments.budget_override


def _validate(arguments: argparse.Namespace) -> None:
    if arguments.omp < 1:
        raise ProbeConfigurationError(f"--omp must be >= 1, got {arguments.omp}")
    if arguments.repeat < 1:
        raise ProbeConfigurationError(f"--repeat must be >= 1, got {arguments.repeat}")
    if arguments.lane == "jax-gpu" and arguments.repeat < MINIMUM_JAX_REPEAT:
        raise ProbeConfigurationError(
            f"--repeat must be >= {MINIMUM_JAX_REPEAT} for the JAX lane, got "
            f"{arguments.repeat}: leg 0 pays compilation and the kill rule "
            "(kill_threshold_warm) is read off the warm legs, so one leg publishes "
            "no warm sample at all"
        )
    if (
        arguments.family != RCLS_FAMILY
        and arguments.lane == "jax-gpu"
        and arguments.repeat > 1
        and arguments.compile_cache is None
    ):
        raise ProbeConfigurationError(
            f"{arguments.family}: --compile-cache is required with --repeat > 1. "
            "The mirror's solve() builds a fresh jitted problem per call, so a "
            "repeat is not warm in memory; without the XLA persistent cache every "
            "leg re-traces and the 'warm' series would be a series of cold legs"
        )
    if arguments.currents_out is not None:
        if arguments.currents_out.suffix not in CURRENTS_SUFFIXES:
            raise ProbeConfigurationError(
                f"--currents-out must end in one of {list(CURRENTS_SUFFIXES)}, got "
                f"{arguments.currents_out}: the payload is always an npz archive "
                "(currents plus their configuration metadata), and any other "
                "suffix names a file the comparison step will not recognise"
            )
        if arguments.currents_out.exists():
            raise ProbeConfigurationError(
                f"--currents-out {arguments.currents_out} already exists; a probe "
                "does not overwrite another lane's reference currents"
            )
    if arguments.dry_run:
        return
    if arguments.output is None:
        raise ProbeConfigurationError("--output is required unless --dry-run")
    resolved = arguments.output.resolve()
    if EVIDENCE_ROOT not in resolved.parents and REPO_ROOT in resolved.parents:
        raise ProbeConfigurationError(
            f"--output inside the repository must live under {EVIDENCE_ROOT}, got "
            f"{resolved}; a path outside the repository is accepted and is recorded "
            "as published_under_evidence_root: false"
        )


def _native_policy(family: MirrorFamily) -> dict[str, object]:
    """The native lane's optimizer policy, anchored to this family's call sites."""
    return {
        **NATIVE_LBFGSB_POLICY,
        "source": list(family.minimize_call_sources),
        "maxiter_source": family.native_budget_source,
    }


def _run_mirror_family(arguments: argparse.Namespace, budget: int) -> dict[str, object]:
    family = MIRROR_FAMILIES[arguments.family]
    payload: dict[str, object] = {
        "native_budget": family.native_budget,
        "native_budget_source": family.native_budget_source,
        "native_script": str(family.native_script.relative_to(REPO_ROOT)),
        "mirror_module": str(family.mirror_module.relative_to(REPO_ROOT)),
        "native_minimize_calls": list(family.minimize_call_sources),
        "native_policy": _native_policy(family),
        "mirror_policy": dict(family.mirror_policy),
        "policy_matched": False,
        "declared_dimensions": dict(family.declared_dimensions),
        "work_matched_per_evaluation": (
            float(family.declared_dimensions["jax_over_native_pairs_per_evaluation"])
            == 1.0
        ),
    }
    observable_maps: list[Mapping[str, object]] = []
    if arguments.lane == "native":
        legs = _native_mirror_legs(
            family,
            arguments.repeat,
            arguments.omp,
            arguments.compile_cache,
            arguments.ledger,
        )
        payload["legs"] = legs
        payload.update(
            {
                name: value
                for name, value in legs[-1].items()
                if name.startswith("native_final_objective")
            }
        )
        payload["summary"] = {
            "process_wall": _summarize(
                legs,
                "process_wall_seconds",
                series_kind=FRESH_PROCESS_COLD_ONLY,
            ),
            "minimize_region": _summarize(
                legs,
                "minimize_region_seconds",
                series_kind=FRESH_PROCESS_COLD_ONLY,
            ),
            "taylor_last_error_elapsed": _summarize(
                legs,
                "taylor_last_error_elapsed_seconds",
                series_kind=FRESH_PROCESS_COLD_ONLY,
            ),
        }
    else:
        legs, mirror_import_seconds = _jax_mirror_legs(
            family, arguments.repeat, budget, arguments.omp, arguments.ledger
        )
        payload["legs"] = legs
        payload["mirror_import_seconds"] = mirror_import_seconds
        observable_maps = [leg["observables"] for leg in legs]
        payload.update(
            {
                name: value
                for name, value in legs[-1]["observables"].items()
                if name.startswith("final_objective")
            }
        )
        payload.setdefault("final_objective", None)
        payload["summary"] = {
            "solve_call": _summarize(
                legs,
                "solve_call_seconds",
                series_kind=COLD_THEN_PERSISTENT_CACHE,
            ),
            "mirror_import_seconds": mirror_import_seconds,
        }
    payload["disclosures"] = [
        *_mirror_disclosures(family),
        *_nonfinite_disclosure(payload, *observable_maps),
    ]
    return payload


def _run_rcls_family(arguments: argparse.Namespace, budget: int) -> dict[str, object]:
    import numpy as np

    problem = _build_rcls_problem(budget)
    payload: dict[str, object] = {
        "native_script": "examples/2_Intermediate/wireframe_rcls_with_ports.py",
        "mirror_module": "examples/jax/2_Intermediate/wireframe_rcls_with_ports.py",
        "regularization_weight": RCLS_REGULARIZATION_WEIGHT,
        "measured_dimensions": problem.dimensions,
        "shared_build_seconds": problem.build_seconds,
    }
    if arguments.lane == "native":
        legs, currents = _native_rcls_legs(
            problem, arguments.repeat, arguments.omp, arguments.ledger
        )
        payload["legs"] = legs
        payload["summary"] = {
            "optimize_wireframe": _summarize(
                legs,
                "optimize_wireframe_seconds",
                series_kind=COLD_THEN_IN_PROCESS_WARM,
            ),
            "solve_only": _summarize(
                legs,
                "solve_only_seconds",
                series_kind=COLD_THEN_IN_PROCESS_WARM,
            ),
        }
    else:
        legs, currents = _jax_rcls_legs(
            problem, arguments.repeat, arguments.omp, arguments.ledger
        )
        payload["legs"] = legs
        payload["summary"] = {
            "device_solve": _summarize(
                legs,
                "device_solve_seconds",
                series_kind=COLD_THEN_IN_PROCESS_WARM,
            ),
            "matrix_assembly": _summarize(
                legs,
                "matrix_assembly_seconds",
                series_kind=COLD_THEN_IN_PROCESS_WARM,
            ),
        }
        problem.wireframe.currents[:] = currents
    payload["constraints_satisfied"] = bool(problem.wireframe.check_constraints())
    payload.update(_endpoint_fields("maximum_current", float(np.max(np.abs(currents)))))
    payload.update(_endpoint_fields("final_objective", float(legs[-1]["f"])))
    payload["currents_metadata"] = _currents_metadata(problem, arguments.lane)
    comparison_blocks: list[Mapping[str, object]] = []
    if arguments.currents_out is not None:
        _write_currents(arguments.currents_out, currents, payload["currents_metadata"])
        payload["currents_out"] = str(arguments.currents_out)
    if arguments.compare is not None:
        comparison = _compare_currents(
            currents, arguments.compare, problem, arguments.lane
        )
        payload["currents_comparison"] = comparison
        comparison_blocks = [comparison["full_vector"], comparison["free_segments"]]
    payload["disclosures"] = [
        *_SHARED_DISCLOSURES,
        *_RCLS_DISCLOSURES,
        *_nonfinite_disclosure(payload, *comparison_blocks),
    ]
    return payload


def _artifact_objective(
    document: Mapping[str, object], path: Path
) -> tuple[str, float]:
    """The endpoint one published artifact carries, by whichever field it uses."""
    for field in ("native_final_objective", "final_objective"):
        value = document.get(field)
        if isinstance(value, (int, float)):
            return field, float(value)
        sentinel = document.get(f"{field}{SENTINEL_SUFFIX}")
        if sentinel is not None:
            raise ProbeConfigurationError(
                f"{path} carries a non-finite {field} ({sentinel}); the solve ran "
                "but produced no endpoint, so there is nothing to compare"
            )
    raise ProbeConfigurationError(
        f"{path} carries neither 'native_final_objective' nor 'final_objective'; "
        "it is not a marginal-quartet artifact with an endpoint"
    )


def _compare_artifacts(first: Path, second: Path) -> dict[str, object]:
    """Report-only endpoint difference between two published lane artifacts.

    This computes no speed ratio and applies no gate.  It answers one question a
    pair of speed numbers cannot: did the two lanes finish anywhere near the same
    objective, or is one of them solving a different problem?  The native
    endpoint is parsed from stdout printed at ``.1e``, so the resolution of the
    relative difference is around a percent, and that is stated in the output.
    """
    documents = []
    for path in (first, second):
        with open(path, encoding="utf-8") as handle:
            documents.append(json.loads(handle.read()))
    families = {str(document.get("family")) for document in documents}
    if len(families) != 1:
        raise ProbeConfigurationError(
            f"cannot compare endpoints across families {sorted(families)}"
        )
    entries = []
    for path, document in zip((first, second), documents):
        field, value = _artifact_objective(document, path)
        entries.append(
            {
                "artifact": str(path),
                "lane": str(document.get("lane")),
                "field": field,
                "final_objective": value,
            }
        )
    left = entries[0]["final_objective"]
    right = entries[1]["final_objective"]
    scale = max(abs(left), abs(right))
    family = families.pop()
    disclosures = [
        "Report-only diagnostic. No gate is applied, no ratio is computed, and "
        "nothing here promotes either artifact beyond its own grade."
    ]
    if any(entry["field"] == "native_final_objective" for entry in entries):
        disclosures.append(
            "A native mirror endpoint is parsed from a '.1e' stdout line (two "
            "significant digits), so a relative difference below roughly 5e-2 is "
            "inside the printing resolution and means only 'not obviously "
            "different'."
        )
    if family in MIRROR_FAMILIES:
        disclosures.append(
            "The two lanes run under different optimizer policies (native_policy vs "
            "mirror_policy in each artifact); a nonzero difference is expected and "
            "is not by itself a defect."
        )
    return {
        "schema": "marginal-quartet-endpoint-comparison.v1",
        "grade": PROBE_GRADE,
        "plan": PLAN,
        "family": family,
        "artifacts": entries,
        "absolute_difference": abs(left - right),
        "relative_difference": (abs(left - right) / scale if scale > 0.0 else 0.0),
        "report_only": True,
        "disclosures": disclosures,
    }


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="family", required=True)
    for name in FAMILIES:
        if name == RCLS_FAMILY:
            budget_help = (
                "plasma surface resolution for both lanes "
                f"(default {RCLS_DEFAULT_RESOLUTION}, the shipped value)"
            )
        else:
            budget_help = (
                "iteration budget for the JAX lane only; the native script "
                "hard-codes 400 and has no selector"
            )
        family_parser = subparsers.add_parser(name)
        family_parser.add_argument(
            "--lane", choices=("native", "jax-gpu"), required=True
        )
        family_parser.add_argument("--omp", type=int, required=True)
        family_parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
        family_parser.add_argument(
            "--budget-override", type=int, default=None, help=budget_help
        )
        family_parser.add_argument("--compile-cache", type=Path, default=None)
        family_parser.add_argument("--output", type=Path, default=None)
        family_parser.add_argument(
            "--ledger",
            type=Path,
            default=DEFAULT_LEDGER,
            help="append-only JSONL record of the order the legs actually ran in",
        )
        family_parser.add_argument("--dry-run", action="store_true")
        if name == RCLS_FAMILY:
            family_parser.add_argument("--currents-out", type=Path, default=None)
            family_parser.add_argument("--compare", type=Path, default=None)
    compare_parser = subparsers.add_parser(
        COMPARE_COMMAND,
        help="print the endpoint difference between two published lane artifacts",
    )
    compare_parser.add_argument("artifacts", nargs=2, type=Path)
    arguments = parser.parse_args(argv)
    if arguments.family not in (RCLS_FAMILY, COMPARE_COMMAND):
        arguments.currents_out = None
        arguments.compare = None
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse(argv)
    if arguments.family == COMPARE_COMMAND:
        first, second = arguments.artifacts
        print(json.dumps(_compare_artifacts(first, second), indent=2, sort_keys=True))
        return 0
    _validate(arguments)
    # Before any numerical import: OpenMP, BLAS, and the JAX persistent cache
    # are all read at library initialization, and the deferred imports below are
    # what make this ordering possible.  The scrub is the load-bearing half, so
    # this replaces os.environ rather than updating it.
    pinned = _pin_process_environment(
        arguments.lane, arguments.omp, arguments.compile_cache
    )
    budget = _resolve_budget(arguments)
    configuration = _configuration(arguments, budget, pinned)

    if arguments.dry_run:
        report: dict[str, object] = {
            "dry_run": True,
            "schema": SCHEMA,
            "grade": PROBE_GRADE,
            "plan": PLAN,
            "kill_threshold_warm": KILL_THRESHOLD_WARM,
            "configuration": configuration,
        }
        if arguments.family == RCLS_FAMILY:
            problem = _build_rcls_problem(budget)
            report["measured_dimensions"] = problem.dimensions
            report["shared_build_seconds"] = problem.build_seconds
            report["currents_metadata"] = _currents_metadata(problem, arguments.lane)
            report["disclosures"] = list(_SHARED_DISCLOSURES + _RCLS_DISCLOSURES)
        else:
            family = MIRROR_FAMILIES[arguments.family]
            report["declared_dimensions"] = dict(family.declared_dimensions)
            report["native_script"] = str(family.native_script)
            report["mirror_module"] = str(family.mirror_module)
            report["native_budget_source"] = family.native_budget_source
            report["native_policy"] = _native_policy(family)
            report["mirror_policy"] = dict(family.mirror_policy)
            # The per-lane dimensions and the work-asymmetry disclosure are the
            # two things an operator needs *before* committing a lane to a
            # timed run, so the dry run prints exactly what the artifact will.
            report["disclosures"] = _mirror_disclosures(family)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if arguments.family == RCLS_FAMILY:
        measured = _run_rcls_family(arguments, budget)
    else:
        measured = _run_mirror_family(arguments, budget)

    payload = {
        "schema": SCHEMA,
        "probe": "marginal-quartet",
        "plan": PLAN,
        "plan_tasks": ["P4.1", "P4.2", "P4.3"],
        "family": arguments.family,
        "lane": arguments.lane,
        "kill_threshold_warm": KILL_THRESHOLD_WARM,
        "configuration": configuration,
        "identity": runtime_identity(arguments.lane),
        **measured,
    }
    write_probe_artifact(arguments.output, payload)
    print(
        json.dumps(
            {
                "family": arguments.family,
                "lane": arguments.lane,
                "output": str(arguments.output),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
