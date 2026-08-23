"""Two-lane GSCO-siblings probe at shipped and reference scale (plan P2.1).

The two GSCO sibling examples (``wireframe_gsco_modular``,
``wireframe_gsco_sector_saddle``) ship at 48x50 / 2,000 iterations and lose
0.79-0.89x against the best native lane there.  Both native sources name a
*reference* configuration in dead comments that no CLI or environment selector
can reach -- 96x100 / 20,000 iterations, and for the sector-saddle sibling also
``break_width`` 2->4, ``gsco_cur_frac`` 0.05->0.03, ``lambda_S`` 10**-6.5 ->
10**-7.5 (the ``to match reference`` comments on ``wf_n_phi``, ``wf_n_theta``
and ``max_iter`` in ``examples/2_Intermediate/wireframe_gsco_modular.py``, and
on ``wf_n_phi``, ``wf_n_theta``, ``max_iter``, ``break_width``,
``gsco_cur_frac`` and ``lambda_S`` in
``examples/2_Intermediate/wireframe_gsco_sector_saddle.py``).  At reference
scale the per-iteration reduction is 1,024 x 19,200 -- the same shape as the
certified ``wireframe_gsco_multistep`` 3.5x device-solve win -- but the
aggregate structure differs (one flat 20,000-iteration stage here versus
multistep's staged sweeps over shrinking masked subsets).  This script exists
so that difference is measured rather than argued.

It is a *probe*, not a campaign: it builds one sibling at one scale in one
lane, times it, dumps the final segment-currents vector at full precision, and
publishes one grade-stamped artifact through
:mod:`benchmarks.probe_conventions`.  It never sweeps, never gates, and never
computes a native/JAX ratio -- a certified claim needs its own preregistered
charter (plan section "Campaign protocol").

Problem construction is replicated from the native sources through public APIs
only; the example scripts are not imported, executed, or edited.  Both lanes
build the identical wireframe, plasma surface, TF seed and constraints, then
each splits its normal-field matrix build out of its solve window:

* native lane -- ``bnorm_obj_matrices`` timed as ``matrix_build_s``, then
  ``simsopt.solve.optimize_wireframe(wf, 'gsco', params, Amat=..., bvec=...)``
  timed as ``solve_call_s``.  That is ``optimize_wireframe``'s precomputed-
  matrix mode; the matrices handed to it are exactly the ones its ``surf_plas``
  mode (the mode both examples use) would have built for itself, from the same
  call with the same arguments;
* JAX lane -- ``bnorm_obj_matrices_jax`` timed as ``matrix_build_s``, then
  ``gsco_wireframe_jax`` (``gsco_wireframe_jax`` in
  ``src/simsopt_jax_adapters/solve/wireframe.py``) timed as the solve, with the
  same ``record_every``/``print_interval`` choice the two JAX mirrors make
  (``examples/jax/2_Intermediate/wireframe_gsco_modular.py``).

Both lanes publish sha256 digests of the ``A`` matrix and ``b`` vector they
actually solved against, so cross-lane problem identity is a digest comparison
rather than an assumption.

Every lane re-execs itself once, under an environment built by
``probe_conventions.pinned_environment``: OpenMP reads its thread count at
library load and JAX reads its persistent compilation cache at first import,
so both have to be pinned before the interpreter that measures them starts.

Run recipe (from the repository root)::

    # native lane -- --omp is mandatory (the fair-native denominator is chosen,
    # never inherited); the process re-execs itself with the pin in place
    PYTHONPATH=src:build/cp311-cp311-linux_x86_64 \
      .venv/bin/python benchmarks/wireframe_gsco_siblings_reference_scale.py \
      --sibling modular --scale reference --lane native --omp 16

    # GPU lane -- jax 0.10.0 + CUDA interpreter; --omp is the *host* thread pin
    # and defaults to 8; --compile-cache selects a warm persistent-cache leg
    SIMSOPT_BACKEND_MODE=jax_gpu_fast SIMSOPT_PRECISION=fp64 \
      PYTHONPATH=src:build/cp311-cp311-linux_x86_64 \
      .venv-qn-gpu/bin/python benchmarks/wireframe_gsco_siblings_reference_scale.py \
      --sibling modular --scale reference --lane jax-gpu --omp 32 \
      --compile-cache /tmp/gsco-cache

    # physics check between the two lanes' dumps
    .venv/bin/python benchmarks/wireframe_gsco_siblings_reference_scale.py \
      --compare native.currents.npy jax.currents.npy

The script never assumes which interpreter launched it: nothing outside the
standard library is imported at module scope, each lane imports what it needs
when it needs it, and it fails loud if those imports, the JAX device, the
OpenMP readback or the fp64 runtime policy are not what the requested lane
requires.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

_PROCESS_START = time.monotonic()

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.probe_conventions import (
    PROBE_GRADE,
    SCRUBBED_ENVIRONMENT_PREFIXES,
    ProbeConventionError,
    append_leg_ledger,
    array_sha256,
    observed_openmp_threads,
    pinned_environment,
    runtime_identity,
    sha256_file,
    ulp_distance,
    write_probe_artifact,
)

SCHEMA = "gsco-siblings-reference-probe.v1"
EVIDENCE_ROOT = REPO_ROOT / "docs" / "receipts" / "evidence"
DEFAULT_LEDGER = EVIDENCE_ROOT / "probe_leg_ledger.jsonl"
TEST_DATA = REPO_ROOT / "tests" / "test_files"
EQUILIBRIUM_FILE = TEST_DATA / "input.LandremanPaul2021_QA"
WIREFRAME_SURFACE_FILE = TEST_DATA / "nescin.LandremanPaul2021_QA"

#: Marker distinguishing the pinned child from the launching parent.  It is
#: itself a ``SIMSOPT_`` variable, so ``pinned_environment`` scrubs any stale
#: copy out of the inherited shell and only this module's own re-exec can put
#: it back.
CHILD_MARKER = "SIMSOPT_GSCO_SIBLINGS_PROBE_CHILD"

#: Host-thread pin for the JAX lane when ``--omp`` is not given.  8 is the
#: certified successor harness's own host pin for its device legs
#: (``GPU_HOST_OMP_THREADS`` in
#: ``benchmarks/stage_two_finitebuild_native_gpu.py``).  It is a default, not a
#: policy: the predecessor GSCO-siblings receipt
#: (``docs/receipts/wireframe_gsco_siblings_native_default.md``) pinned its JAX
#: GPU legs at 32, and reproducing it means passing ``--omp 32``.
JAX_LANE_DEFAULT_OMP_THREADS = 8

#: The JAX lane's backend selectors, carried across the scrub by name.
#: ``pinned_environment`` deletes the whole ``JAX_``/``XLA_``/``SIMSOPT_``
#: family because one stray variable reroutes a *native* leg onto the device or
#: into fp32.  The JAX lane, whose entire point is to be on the device, has to
#: put its own selectors back; doing it by an explicit name list is what lets
#: the artifact publish exactly which inherited variables survived.
#: ``JAX_COMPILATION_CACHE_DIR`` is deliberately absent -- the cache is owned by
#: ``--compile-cache`` and by nothing else.
JAX_LANE_CARRIED_VARIABLES = (
    "SIMSOPT_BACKEND_MODE",
    "SIMSOPT_BACKEND_STRICT",
    "SIMSOPT_PRECISION",
    "SIMSOPT_JAX_TRANSFER_GUARD",
    "JAX_TRANSFER_GUARD",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
)

#: What ``--compile-cache`` pins alongside the cache directory itself.  JAX
#: only persists a program whose compile took longer than
#: ``min_compile_time_secs`` (1 s by default) and whose entry exceeds
#: ``min_entry_size_bytes``, so without these two the entry count can sit at
#: zero through a genuinely cold leg and the cold/warm proof is vacuous.  The
#: predecessor receipt pinned both to zero for exactly that reason
#: (``docs/receipts/wireframe_gsco_siblings_native_default.md``, warm-cache
#: protocol); ``pinned_environment`` scrubs the whole ``JAX_`` family, so this
#: probe has to pin them itself.
PERSISTENT_CACHE_THRESHOLDS = {
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
}

MU0 = 4.0 * math.pi * 1.0e-7


# ---------------------------------------------------------------------------
# SSOT configuration table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GscoConfiguration:
    """One (sibling, scale) row.  Every number here is cited to its source."""

    sibling: str
    scale: str
    wireframe_nphi: int
    wireframe_ntheta: int
    max_iter: int
    plasma_resolution: int
    lambda_s: float
    coils_per_half_period: int
    field_on_axis: float
    #: Sector-saddle only; ``None`` marks the modular sibling, which places no
    #: toroidal breaks and lets GSCO reshape its seeded TF coils.
    break_width: int | None
    #: Sector-saddle only; ``None`` means the GSCO loop current is the seeded
    #: per-coil current itself, as the modular example sets it.
    gsco_current_fraction: float | None
    #: File plus symbol, never file plus line: this string is published into
    #: every artifact, and a line number in a published citation rots silently
    #: the first time the cited file is edited.
    source: str


#: The six rows this probe can build.  ``smoke`` is not reachable from
#: ``--scale``; it is selected by the hidden ``--smoke`` flag and exists only to
#: exercise both lanes' plumbing in seconds.  It is never a measurement.
CONFIGURATIONS: dict[tuple[str, str], GscoConfiguration] = {
    ("modular", "shipped"): GscoConfiguration(
        sibling="modular",
        scale="shipped",
        wireframe_nphi=48,
        wireframe_ntheta=50,
        max_iter=2_000,
        plasma_resolution=32,
        lambda_s=10.0**-6,
        coils_per_half_period=6,
        field_on_axis=1.0,
        break_width=None,
        gsco_current_fraction=None,
        source=(
            "examples/2_Intermediate/wireframe_gsco_modular.py "
            "(wf_n_phi, wf_n_theta, max_iter, plas_n, n_mod_coils_hp, "
            "field_on_axis, lambda_S; the non-CI branch)"
        ),
    ),
    ("modular", "reference"): GscoConfiguration(
        sibling="modular",
        scale="reference",
        wireframe_nphi=96,
        wireframe_ntheta=100,
        max_iter=20_000,
        plasma_resolution=32,
        lambda_s=10.0**-6,
        coils_per_half_period=6,
        field_on_axis=1.0,
        break_width=None,
        gsco_current_fraction=None,
        source=(
            "examples/2_Intermediate/wireframe_gsco_modular.py "
            "('to match reference' comments on wf_n_phi, wf_n_theta, max_iter; "
            "plas_n, n_mod_coils_hp, field_on_axis, lambda_S unchanged)"
        ),
    ),
    ("modular", "smoke"): GscoConfiguration(
        sibling="modular",
        scale="smoke",
        # 12, not 8: ``add_tfcoil_currents`` spaces 6 coils per half period over
        # n_phi toroidal slots (``add_tfcoil_currents``'s ``tf_inds``,
        # src/simsopt/geo/wireframe_toroidal.py:1210), and at n_phi=8 two of
        # them land on the same slot, so the seeded state already violates the
        # net-poloidal-current constraint before GSCO runs.
        wireframe_nphi=12,
        wireframe_ntheta=8,
        max_iter=20,
        plasma_resolution=8,
        lambda_s=10.0**-6,
        coils_per_half_period=6,
        field_on_axis=1.0,
        break_width=None,
        gsco_current_fraction=None,
        source="plumbing smoke only; physics parameters as shipped",
    ),
    ("sector_saddle", "shipped"): GscoConfiguration(
        sibling="sector_saddle",
        scale="shipped",
        wireframe_nphi=48,
        wireframe_ntheta=50,
        max_iter=2_000,
        plasma_resolution=32,
        lambda_s=10.0**-6.5,
        coils_per_half_period=3,
        field_on_axis=1.0,
        break_width=2,
        gsco_current_fraction=0.05,
        source=(
            "examples/2_Intermediate/wireframe_gsco_sector_saddle.py "
            "(wf_n_phi, wf_n_theta, max_iter, plas_n, n_tf_coils_hp, "
            "break_width, gsco_cur_frac, field_on_axis, lambda_S; the non-CI "
            "branch)"
        ),
    ),
    ("sector_saddle", "reference"): GscoConfiguration(
        sibling="sector_saddle",
        scale="reference",
        wireframe_nphi=96,
        wireframe_ntheta=100,
        max_iter=20_000,
        plasma_resolution=32,
        lambda_s=10.0**-7.5,
        coils_per_half_period=3,
        field_on_axis=1.0,
        break_width=4,
        gsco_current_fraction=0.03,
        source=(
            "examples/2_Intermediate/wireframe_gsco_sector_saddle.py "
            "('to match reference' comments on wf_n_phi, wf_n_theta, max_iter, "
            "break_width, gsco_cur_frac, lambda_S; plas_n, n_tf_coils_hp, "
            "field_on_axis unchanged)"
        ),
    ),
    ("sector_saddle", "smoke"): GscoConfiguration(
        sibling="sector_saddle",
        scale="smoke",
        # 12 for the same reason as the modular smoke row; it also satisfies
        # ``set_toroidal_breaks``'s n_breaks*width <= n_phi/2 guard
        # (src/simsopt/geo/wireframe_toroidal.py:657) at the shipped
        # break_width, so no physics parameter has to be bent for the smoke.
        wireframe_nphi=12,
        wireframe_ntheta=8,
        max_iter=20,
        plasma_resolution=8,
        lambda_s=10.0**-6.5,
        coils_per_half_period=3,
        field_on_axis=1.0,
        break_width=2,
        gsco_current_fraction=0.05,
        source="plumbing smoke only; physics parameters as shipped",
    ),
}

SIBLINGS = ("modular", "sector_saddle")
SELECTABLE_SCALES = ("shipped", "reference")
SMOKE_SCALE = "smoke"
NATIVE_LANE = "native"
JAX_LANE = "jax-gpu"
LANES = (NATIVE_LANE, JAX_LANE)

#: Call-order labels every repeat of every lane carries.  They say where in the
#: process a repeat sat, and nothing else: the native lane compiles nothing, so
#: ``first_in_process`` there is not a cache scope the way the JAX lane's
#: ``cold_solve_s`` is -- it is the repeat that pays this process's first-touch
#: costs (allocator growth, page faults, the first pass over freshly built
#: matrices).  Without them a ``--repeat 1`` native leg publishes one number
#: whose thermal status the artifact never states.
FIRST_IN_PROCESS = "first_in_process"
REPEAT_IN_PROCESS = "repeat_in_process"
SERIES_LABELS = (FIRST_IN_PROCESS, REPEAT_IN_PROCESS)

#: Raised into ``notes`` when a repeat's endpoint statistics are not finite.
#: The solve still ran and its timings still stand; what does not stand is its
#: endpoint, and saying so is the point of publishing the leg at all.
NONFINITE_NOTE = (
    "NON-FINITE ENDPOINT: the statistics named in nonfinite_statistics are "
    "published as null with the offending value beside them in "
    "<name>_sentinel. A null there is not a zero. The affected repeats' "
    "timings stand; their endpoints do not."
)

DISCLOSURES = (
    "grade is diagnostic-not-certifying: this probe measures, it never certifies.",
    "no ratio is computed here; the two lanes publish separate artifacts.",
    (
        "both lanes split the normal-field matrix build out of the solve "
        "window. matrix_build_s times bnorm_obj_matrices on the native lane "
        "and bnorm_obj_matrices_jax on the JAX lane; each solve timer starts "
        "after its own matrix build. Nothing is subtracted from any number."
    ),
    (
        "the native lane therefore calls optimize_wireframe in its "
        "precomputed-matrix mode (Amat/bvec), not the surf_plas mode the two "
        "examples use. The matrices handed in are the ones the surf_plas mode "
        "would have built for itself -- the same bnorm_obj_matrices call with "
        "the same arguments -- so the solve sees identical inputs and only the "
        "timing boundary moves "
        "(optimize_wireframe in src/simsopt/solve/wireframe_optimization.py)."
    ),
    (
        "the native solve window still contains one post-solve WireframeField "
        "construction over the solution currents, which optimize_wireframe "
        "performs in both of its modes; the JAX solve window contains no "
        "equivalent. That residual asymmetry is disclosed, not subtracted."
    ),
    (
        "response_matrix_sha256 and target_vector_sha256 digest the float64 "
        "bytes of the A matrix and b vector each lane actually solved against. "
        "Equal digests across the two lanes are the cross-lane "
        "problem-identity proof. Unequal digests mean the lanes did not solve "
        "the bit-identical problem -- bnorm_obj_matrices_jax reduces in a "
        "different order than bnorm_obj_matrices, so A can differ in its last "
        "bits while b matches -- and what such a comparison is then worth "
        "depends on the size of that difference, which this probe publishes "
        "nothing to bound. Adjudicating it is charter business, not a probe's."
    ),
    (
        "history bookkeeping is asymmetric exactly as the shipped lanes are: "
        "the native GSCO records every iteration, the JAX lane is called with "
        "record_every=max_iter as both mirrors call it, so it materializes one "
        "history sample."
    ),
    (
        "both lanes are passed print_interval=max_iter, a deviation from the "
        "print_interval=100 both examples set. On the JAX lane it is inert: "
        "gsco_wireframe_jax deletes the argument unread. On the native lane it "
        "is NOT inert under verbose=False -- verbose gates only the Python "
        "prints around the kernel, while the C++ GSCO's print_iter is an "
        "unguarded printf inside the timed loop (GSCO in "
        "src/simsoptpp/wireframe_optimization.cpp). At print_interval=max_iter "
        "the native kernel writes two progress lines instead of about "
        "max_iter/100, which is a small bias in the native lane's favour."
    ),
    (
        "every repeat of every lane carries series_label: first_in_process for "
        "repeat 0, repeat_in_process for each later one, counted at the top "
        "level in repeat_label_counts (repeat_count is the request; those "
        "counts are what actually ran)."
    ),
    (
        "JAX repeat 0 publishes cold_solve_s -- the first solve in this "
        "process, carrying its XLA compile -- and warm_solve_s, an identical "
        "second solve. Every later repeat is already warm and publishes "
        "warm_repeat_a_s and warm_repeat_b_s instead: no repeat past the first "
        "has a cold solve to report, and none claims one."
    ),
    (
        "the native series is thermally labelled but never cache-scoped: the "
        "native lane compiles nothing, so first_in_process is NOT a cold "
        "compile the way the JAX lane's cold_solve_s is. It is still the "
        "repeat that pays this process's first-touch costs -- allocator "
        "growth, page faults, the first pass over freshly built matrices -- "
        "which the repeat_in_process repeats after it do not. Each native "
        "repeat runs exactly one timed solve and publishes it as solve_call_s "
        "under its own label; at the default --repeat 1 the single native "
        "number is therefore a first_in_process number, and reading it as a "
        "warm one is a mistake the label exists to prevent. Comparing it "
        "against a JAX warm_solve_s compares two different thermal states."
    ),
    (
        "a repeat whose f_B, f_S, f or maximum_current_A came back non-finite "
        "publishes that statistic as null, names it in the repeat's "
        "nonfinite_statistics, and puts the offending value in "
        "<name>_sentinel as a string; the NON-FINITE ENDPOINT note then "
        "appears in notes. NaN and Infinity are not JSON, so the alternative "
        "is not a NaN in the artifact -- it is no artifact at all."
    ),
    (
        "cold and warm are cache scopes, not just call order. With no "
        "--compile-cache the JAX lane runs against no persistent cache at all "
        "(pinned_environment scrubs any inherited JAX_COMPILATION_CACHE_DIR "
        "before pinning, so the leg cannot be quietly warm). With "
        "--compile-cache DIR the leg's cache entry count is recorded before "
        "and after each timed solve; a cold leg is the one whose count grows. "
        "That count is only evidence because --compile-cache also pins "
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS and "
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES to 0, as the predecessor "
        "receipt's warm-cache protocol does; at JAX's 1-second default a cold "
        "leg can write nothing and still be cold."
    ),
    (
        "--omp pins both lanes through pinned_environment, before the "
        "interpreter that measures them starts. On the native lane it is the "
        "fair-native denominator and is mandatory. On the JAX lane it is the "
        "host thread pin only, defaulting to GPU_HOST_OMP_THREADS=8 from "
        "benchmarks/stage_two_finitebuild_native_gpu.py. Both lanes read the "
        "pin back out of libgomp and refuse to run on a mismatch."
    ),
)


def configuration(sibling: str, scale: str) -> GscoConfiguration:
    key = (sibling, scale)
    if key not in CONFIGURATIONS:
        raise ProbeConventionError(f"no configuration row for {key!r}")
    return CONFIGURATIONS[key]


# ---------------------------------------------------------------------------
# Problem construction (public APIs only, replicating the native sources)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltProblem:
    """Everything both lanes need, built identically for both."""

    wireframe: object
    plasma: object
    poloidal_current: float
    default_current: float
    max_current: float


@dataclass(frozen=True)
class LaneResult:
    """One lane's per-repeat rows, its currents, and the problem it solved.

    ``dimensions`` is produced once, by :func:`problem_dimensions`, from the
    problem built for repeat 0.  It is the artifact's only producer of every
    shape it reports -- in particular of ``n_segments``.
    """

    rows: list[dict[str, object]] = field(default_factory=list)
    currents: list[np.ndarray] = field(default_factory=list)
    dimensions: dict[str, object] = field(default_factory=dict)


def build_problem(config: GscoConfiguration) -> BuiltProblem:
    """Replicate the native example's construction for one configuration row.

    Modular: seed ``coils_per_half_period`` planar TF coils and let GSCO
    reshape them.  Sector-saddle: seed the TF coils, then constrain the
    toroidal breaks so new coils cannot form there and the TF coils cannot be
    reshaped.  Both then pin the net poloidal current.
    """
    import numpy as np

    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe

    plasma = SurfaceRZFourier.from_vmec_input(
        EQUILIBRIUM_FILE,
        nphi=config.plasma_resolution,
        ntheta=config.plasma_resolution,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_nescoil_input(
        WIREFRAME_SURFACE_FILE,
        "current",
    )
    wireframe = ToroidalWireframe(
        wireframe_surface,
        config.wireframe_nphi,
        config.wireframe_ntheta,
    )
    poloidal_current = -2.0 * np.pi * plasma.get_rc(0, 0) * config.field_on_axis / MU0
    coil_current = poloidal_current / (2 * wireframe.nfp * config.coils_per_half_period)
    wireframe.add_tfcoil_currents(config.coils_per_half_period, coil_current)
    if config.break_width is not None:
        wireframe.set_toroidal_breaks(
            config.coils_per_half_period,
            config.break_width,
            allow_pol_current=True,
        )
    wireframe.set_poloidal_current(poloidal_current)

    if config.gsco_current_fraction is None:
        default_current = abs(coil_current)
    else:
        default_current = abs(config.gsco_current_fraction * poloidal_current)
    return BuiltProblem(
        wireframe=wireframe,
        plasma=plasma,
        poloidal_current=float(poloidal_current),
        default_current=float(default_current),
        max_current=float(1.1 * default_current),
    )


def problem_dimensions(
    config: GscoConfiguration, problem: BuiltProblem
) -> dict[str, object]:
    """Shapes the two lanes will actually reduce over.  No matrix is built.

    Sole producer of every dimension this probe publishes.  ``n_segments``
    comes from the wireframe and from nothing else -- deriving it a second time
    from the length of a solution vector would put two producers in one
    artifact and make a disagreement between them unreportable.
    """
    import numpy as np

    wireframe = problem.wireframe
    free_cells = np.asarray(wireframe.get_free_cells(form="logical"))
    test_points = int(config.plasma_resolution) ** 2
    segments = int(wireframe.n_segments)
    return {
        "n_field_periods": int(wireframe.nfp),
        "n_test_points": test_points,
        "n_segments": segments,
        "response_matrix_shape": [test_points, segments],
        "response_matrix_elements": test_points * segments,
        "n_cells": int(free_cells.size),
        "n_free_cells": int(np.count_nonzero(free_cells)),
        "n_constrained_segments": int(
            np.asarray(wireframe.constrained_segments()).size
        ),
        "max_iter": int(config.max_iter),
        "poloidal_current_A": problem.poloidal_current,
        "default_current_A": problem.default_current,
        "max_current_A": problem.max_current,
    }


def series_label(index: int) -> str:
    """Call-order label for repeat ``index``.  Sole producer of both labels."""
    return FIRST_IN_PROCESS if index == 0 else REPEAT_IN_PROCESS


def repeat_label_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    """How many repeats actually carried each call-order label.

    Counted from the published rows, never from ``--repeat``: the request and
    the executed series are different facts, and a leg that stopped early must
    not be able to report the count it intended.
    """
    return {
        label: sum(1 for row in rows if row["series_label"] == label)
        for label in SERIES_LABELS
    }


def publishable_statistics(values: dict[str, float]) -> dict[str, object]:
    """A repeat's float statistics, canonical-JSON by construction.

    ``write_probe_artifact`` refuses a payload carrying ``NaN``/``Infinity``
    (``allow_nan=False``), and it refuses the *whole* payload -- so a single
    non-finite endpoint after 20,000 iterations would take the leg's timings,
    digests and identity down with it.  A GSCO endpoint going non-finite is a
    result this probe anticipates, so the payload is made finite here rather
    than being repaired after the fact: the statistic is published as ``null``,
    the offending value is preserved beside it as a string sentinel, and the
    names are listed in ``nonfinite_statistics`` so a reader never has to guess
    whether a ``null`` meant zero.
    """
    published: dict[str, object] = {}
    nonfinite: list[str] = []
    for name, value in values.items():
        if math.isfinite(value):
            published[name] = value
            continue
        published[name] = None
        published[f"{name}_sentinel"] = repr(value)
        nonfinite.append(name)
    published["nonfinite_statistics"] = nonfinite
    return published


def nonfinite_notes(rows: list[dict[str, object]]) -> list[str]:
    """The NON-FINITE note, once, if any repeat flagged a statistic."""
    flagged = any(row["nonfinite_statistics"] for row in rows)
    return [NONFINITE_NOTE] if flagged else []


# ---------------------------------------------------------------------------
# Native lane
# ---------------------------------------------------------------------------


def run_native_lane(config: GscoConfiguration, repeats: int) -> LaneResult:
    """Per repeat: fresh build, timed matrix build, timed mode-2 solve.

    ``optimize_wireframe``'s precomputed-matrix mode takes the matrices its
    ``surf_plas`` mode would otherwise have built itself, which is what puts
    the matrix build outside the solve window without changing what the solve
    is handed.

    One solve per repeat, each labelled by :func:`series_label`: repeat 0 is
    ``first_in_process`` and every later repeat is ``repeat_in_process``.  The
    native lane compiles nothing, so the label is call order rather than a
    cache scope -- but at ``--repeat 1`` it is the only thing in the artifact
    that says the single published ``solve_call_s`` is the first solve this
    process ran, and reading it as a warm number is exactly the mistake.
    """
    import numpy as np

    from simsopt.solve import bnorm_obj_matrices, optimize_wireframe

    result = LaneResult()
    for index in range(repeats):
        build_start = time.monotonic()
        problem = build_problem(config)
        build_seconds = time.monotonic() - build_start
        if index == 0:
            result.dimensions.update(problem_dimensions(config, problem))

        matrix_start = time.monotonic()
        response, target = bnorm_obj_matrices(
            problem.wireframe,
            problem.plasma,
            area_weighted=True,
            verbose=False,
        )
        matrix_seconds = time.monotonic() - matrix_start

        parameters = {
            "lambda_S": config.lambda_s,
            "max_iter": config.max_iter,
            "print_interval": config.max_iter,
            "no_crossing": True,
            "default_current": problem.default_current,
            "max_current": problem.max_current,
        }
        solve_start = time.monotonic()
        solved = optimize_wireframe(
            problem.wireframe,
            "gsco",
            parameters,
            Amat=response,
            bvec=target,
            verbose=False,
        )
        solve_seconds = time.monotonic() - solve_start

        solution = np.asarray(problem.wireframe.currents, dtype=np.float64).ravel()
        result.currents.append(solution)
        result.rows.append(
            {
                "repeat": index,
                "series_label": series_label(index),
                "build_s": build_seconds,
                "matrix_build_s": matrix_seconds,
                "solve_call_s": solve_seconds,
                "response_matrix_sha256": array_sha256(response),
                "target_vector_sha256": array_sha256(target),
                **publishable_statistics(
                    {
                        "f_B": float(np.asarray(solved["f_B"]).reshape(())),
                        "f_S": float(np.asarray(solved["f_S"]).reshape(())),
                        "f": float(np.asarray(solved["f"]).reshape(())),
                        "maximum_current_A": float(np.max(np.abs(solution))),
                    }
                ),
                "nonzero_segments": int(np.count_nonzero(solution)),
                "constraints_satisfied": bool(problem.wireframe.check_constraints()),
            }
        )
    return result


# ---------------------------------------------------------------------------
# JAX lane
# ---------------------------------------------------------------------------


def jax_runtime_facts(require_gpu: bool) -> dict[str, object]:
    """Bind the JAX leg to its device and precision policy, or fail loud."""
    import jax

    from simsopt_jax.backend.runtime import (
        get_backend_mode,
        get_resolved_precision,
        get_runtime_jax_device,
    )

    device = get_runtime_jax_device()
    platform = "cpu" if device is None else str(device.platform)
    facts = {
        "jax_version": str(jax.__version__),
        "jax_default_backend": str(jax.default_backend()),
        "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        "backend_mode": get_backend_mode(),
        "resolved_precision": get_resolved_precision(),
        "device_platform": platform,
        "device_kind": None if device is None else str(device.device_kind),
    }
    if not facts["jax_enable_x64"]:
        raise ProbeConventionError(
            "JAX lane requires jax_enable_x64; pinned_environment sets it, so "
            "seeing it off means the child was launched outside this probe"
        )
    if require_gpu and platform not in {"gpu", "cuda"}:
        raise ProbeConventionError(
            f"--lane jax-gpu resolved to device platform {platform!r}; launch "
            "under .venv-qn-gpu/bin/python on a CUDA host, or use --smoke for "
            "the CPU plumbing check"
        )
    return facts


def cache_entry_count(cache_dir: Path | None) -> int | None:
    """Files under the persistent compilation cache, or ``None`` without one.

    The predecessor receipt proved its cold leg cold by counting the entries it
    wrote (``docs/receipts/wireframe_gsco_siblings_native_default.md``); this
    is the same count, taken before and after each timed solve.  ``None`` means
    no cache was configured at all -- a different fact from an empty one, and
    the two must not encode the same.
    """
    if cache_dir is None:
        return None
    return sum(1 for entry in cache_dir.rglob("*") if entry.is_file())


def run_jax_lane(
    config: GscoConfiguration, repeats: int, cache_dir: Path | None
) -> LaneResult:
    """Per repeat: fresh build, timed matrix build, then two identical solves.

    ``block_until_ready`` on the whole result pytree is what makes the two
    solve walls device walls rather than dispatch walls; the second call is
    identical in every argument, so it reuses this process's compiled kernel.

    Only repeat 0's first solve can be cold, and only repeat 0 labels one that
    way: by repeat 1 this process is already warm, so its two solves are
    published as ``warm_repeat_a_s`` and ``warm_repeat_b_s``.
    """
    import numpy as np

    import jax

    from simsopt_jax_adapters.solve.wireframe import (
        bnorm_obj_matrices_jax,
        gsco_wireframe_jax,
    )

    result = LaneResult()
    for index in range(repeats):
        build_start = time.monotonic()
        problem = build_problem(config)
        build_seconds = time.monotonic() - build_start
        if index == 0:
            result.dimensions.update(problem_dimensions(config, problem))

        matrix_start = time.monotonic()
        response, target = bnorm_obj_matrices_jax(
            problem.wireframe,
            problem.plasma,
            area_weighted=True,
            verbose=False,
        )
        matrix_seconds = time.monotonic() - matrix_start

        arguments = {
            "A": response,
            "c": target,
            "lambda_S": config.lambda_s,
            "no_crossing": True,
            "match_current": False,
            "default_current": problem.default_current,
            "max_current": problem.max_current,
            "max_iter": config.max_iter,
            "print_interval": config.max_iter,
            "record_every": config.max_iter,
            "verbose": False,
        }
        entries_before = cache_entry_count(cache_dir)
        first_start = time.monotonic()
        first_result = jax.block_until_ready(
            gsco_wireframe_jax(problem.wireframe, **arguments)
        )
        first_seconds = time.monotonic() - first_start
        entries_after_first = cache_entry_count(cache_dir)

        second_start = time.monotonic()
        second_result = jax.block_until_ready(
            gsco_wireframe_jax(problem.wireframe, **arguments)
        )
        second_seconds = time.monotonic() - second_start
        entries_after_second = cache_entry_count(cache_dir)

        first_solution = np.asarray(
            jax.device_get(first_result.x), dtype=np.float64
        ).ravel()
        second_solution = np.asarray(
            jax.device_get(second_result.x), dtype=np.float64
        ).ravel()
        last = int(jax.device_get(second_result.history_length)) - 1
        f_b = float(np.asarray(jax.device_get(second_result.f_B_history))[last])
        f_s = float(np.asarray(jax.device_get(second_result.f_S_history))[last])
        f_total = float(np.asarray(jax.device_get(second_result.f_history))[last])
        iterations = int(
            np.asarray(jax.device_get(second_result.iter_history), dtype=np.int64)[last]
        )
        problem.wireframe.currents[:] = second_solution
        result.currents.append(second_solution)

        if index == 0:
            timings = {
                "cold_solve_s": first_seconds,
                "warm_solve_s": second_seconds,
            }
        else:
            timings = {
                "warm_repeat_a_s": first_seconds,
                "warm_repeat_b_s": second_seconds,
            }
        result.rows.append(
            {
                "repeat": index,
                "series_label": series_label(index),
                "build_s": build_seconds,
                "matrix_build_s": matrix_seconds,
                **timings,
                "solve_pair_bitwise_identical": bool(
                    np.array_equal(first_solution, second_solution)
                ),
                "cache_entries_before": entries_before,
                "cache_entries_after_first_solve": entries_after_first,
                "cache_entries_after_second_solve": entries_after_second,
                "response_matrix_sha256": array_sha256(response),
                "target_vector_sha256": array_sha256(target),
                "iterations": iterations,
                **publishable_statistics(
                    {
                        "f_B": f_b,
                        "f_S": f_s,
                        "f": f_total,
                        "maximum_current_A": float(np.max(np.abs(second_solution))),
                    }
                ),
                "nonzero_segments": int(np.count_nonzero(second_solution)),
                "constraints_satisfied": bool(problem.wireframe.check_constraints()),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Physics comparison
# ---------------------------------------------------------------------------


CURRENTS_SIDECAR_SUFFIX = ".meta.json"

#: Configuration fields that must agree before two dumps may be compared.
#: The lane is deliberately absent -- cross-lane comparison is the point.
CURRENTS_IDENTITY_FIELDS = (
    "sibling",
    "scale",
    "wireframe_nphi",
    "wireframe_ntheta",
    "max_iter",
    "plasma_resolution",
)


def _currents_sidecar_path(currents_path: Path) -> Path:
    return currents_path.with_name(currents_path.name + CURRENTS_SIDECAR_SUFFIX)


def _write_currents_sidecar(currents_path: Path, config: GscoConfiguration) -> None:
    """Bind a dump to the configuration that produced it, for the compare gate."""
    document = {field: getattr(config, field) for field in CURRENTS_IDENTITY_FIELDS}
    document["schema"] = SCHEMA
    sidecar = _currents_sidecar_path(currents_path)
    if sidecar.exists():
        raise ProbeConventionError(f"refusing to overwrite existing {sidecar}")
    sidecar.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _currents_identity(currents_path: Path) -> dict[str, object]:
    sidecar = _currents_sidecar_path(currents_path)
    if not sidecar.exists():
        raise ProbeConventionError(
            f"no configuration sidecar at {sidecar}; dumps written by this probe "
            "carry one, and a comparison without it cannot prove the two dumps "
            "came from the same (sibling, scale) problem"
        )
    return dict(json.loads(sidecar.read_text(encoding="utf-8")))


def compare_currents(first: Path, second: Path) -> dict[str, object]:
    """Bitwise / ULP / absolute comparison of two full-precision dumps.

    Gated on the configuration sidecars written beside every dump: the
    :data:`CURRENTS_IDENTITY_FIELDS` must agree (lane excluded), so two
    same-shaped vectors from different problems refuse instead of comparing.
    The ULP distance is :func:`probe_conventions.ulp_distance`, which returns
    the exact elementwise ``uint64`` distance between the two order-preserving
    integer keys; ``max_ulp`` is that array's maximum.
    """
    import numpy as np

    first_identity = _currents_identity(first)
    second_identity = _currents_identity(second)
    mismatched = {
        name: (first_identity.get(name), second_identity.get(name))
        for name in CURRENTS_IDENTITY_FIELDS
        if first_identity.get(name) != second_identity.get(name)
    }
    if mismatched:
        raise ProbeConventionError(
            f"configuration mismatch between dumps, refusing to compare: {mismatched}"
        )
    left = np.load(first)
    right = np.load(second)
    if left.shape != right.shape:
        raise ProbeConventionError(
            f"shape mismatch: {first} has {left.shape}, {second} has {right.shape}"
        )
    if left.dtype != np.float64 or right.dtype != np.float64:
        raise ProbeConventionError(
            f"both dumps must be float64, got {left.dtype} and {right.dtype}"
        )
    difference = np.abs(left - right)
    ulp = np.asarray(ulp_distance(left, right), dtype=np.uint64)
    return {
        "schema": SCHEMA,
        "grade": PROBE_GRADE,
        "mode": "compare",
        "first": str(first),
        "second": str(second),
        "first_sha256": sha256_file(first),
        "second_sha256": sha256_file(second),
        "matched_identity": {
            name: first_identity.get(name) for name in CURRENTS_IDENTITY_FIELDS
        },
        "shape": list(left.shape),
        "bitwise_equal": bool(np.array_equal(left, right)),
        "max_ulp": int(ulp.max()) if ulp.size else 0,
        **publishable_statistics(
            {"max_abs_diff_A": float(difference.max()) if difference.size else 0.0}
        ),
        "n_differing_entries": int(np.count_nonzero(left != right)),
        "support_identical": bool(np.array_equal(left != 0, right != 0)),
        "first_nonzero": int(np.count_nonzero(left)),
        "second_nonzero": int(np.count_nonzero(right)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-lane GSCO-siblings probe (plan P2.1).",
    )
    parser.add_argument("--sibling", choices=SIBLINGS)
    parser.add_argument("--scale", choices=SELECTABLE_SCALES)
    parser.add_argument("--lane", choices=LANES)
    parser.add_argument(
        "--omp",
        type=int,
        help=(
            "thread pin applied by re-exec before interpreter start; the "
            "fair-native denominator on --lane native (mandatory there) and "
            "the host thread pin on --lane jax-gpu (default "
            f"{JAX_LANE_DEFAULT_OMP_THREADS})"
        ),
    )
    parser.add_argument(
        "--compile-cache",
        type=Path,
        help=(
            "JAX-lane persistent compilation cache directory; omit for a leg "
            "with no persistent cache at all"
        ),
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "artifact path; a path inside the repository must live under "
            f"{EVIDENCE_ROOT.relative_to(REPO_ROOT)}, and a path outside the "
            "repository is accepted as a scratch run and stamped "
            "published_under_evidence_root: false"
        ),
    )
    parser.add_argument(
        "--currents-out",
        type=Path,
        help="full-precision currents dump; must end in .npy (np.save's suffix)",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="JSONL ledger the executed order of each leg is appended to",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("FIRST", "SECOND"))
    return parser.parse_args(argv)


def resolve_scale(options: argparse.Namespace) -> str:
    if options.smoke:
        if options.scale is not None:
            raise ProbeConventionError("--smoke selects its own scale; drop --scale")
        return SMOKE_SCALE
    if options.scale is None:
        raise ProbeConventionError("--scale is required (or --smoke)")
    return str(options.scale)


def resolve_omp(options: argparse.Namespace, lane: str) -> int:
    """The thread pin this leg runs under, per lane."""
    if lane == NATIVE_LANE:
        if options.omp is None:
            raise ProbeConventionError(
                "--lane native requires --omp N: the fair-native denominator "
                "is chosen and published, never inherited"
            )
        return int(options.omp)
    if options.omp is None:
        return JAX_LANE_DEFAULT_OMP_THREADS
    return int(options.omp)


def default_output_path(config: GscoConfiguration, lane: str) -> Path:
    return EVIDENCE_ROOT / (
        f"gsco_siblings_{config.sibling}_{config.scale}_{lane.replace('-', '_')}.json"
    )


def child_command(
    options: argparse.Namespace,
    config: GscoConfiguration,
    lane: str,
    omp: int,
    output_path: Path,
    currents_path: Path,
    ledger_path: Path,
    cache_dir: Path | None,
) -> list[str]:
    """The child's argv, rebuilt from the parsed namespace.

    Never ``sys.argv[1:]``: :func:`main` accepts an argument list, so the
    process's own argv and the options actually in force are two different
    things, and forwarding the former silently drops every option a caller
    passed programmatically.  Rebuilding from the namespace also forwards the
    *resolved* values -- the defaulted thread pin, the derived output paths --
    so parent and child cannot disagree about what the leg was.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--sibling",
        config.sibling,
        "--lane",
        lane,
        "--omp",
        str(omp),
        "--repeat",
        str(int(options.repeat)),
        "--output",
        str(output_path),
        "--currents-out",
        str(currents_path),
        "--ledger",
        str(ledger_path),
    ]
    if config.scale == SMOKE_SCALE:
        command.append("--smoke")
    else:
        command.extend(["--scale", config.scale])
    if cache_dir is not None:
        command.extend(["--compile-cache", str(cache_dir)])
    return command


def relaunch_pinned_child(
    options: argparse.Namespace,
    config: GscoConfiguration,
    lane: str,
    omp: int,
    output_path: Path,
    currents_path: Path,
    ledger_path: Path,
    cache_dir: Path | None,
) -> int:
    """Re-exec this script under a scrubbed, pinned environment.

    Both lanes go through here.  OpenMP reads its thread count once, at library
    load, and JAX reads its persistent compilation cache at first import;
    pinning either from inside a running interpreter is too late.
    ``pinned_environment`` scrubs the whole numerical / ``JAX_`` / ``XLA_`` /
    ``SIMSOPT_`` / ``CUDA_`` family first, so a variable reaches the child if
    and only if this function put it there.
    """
    environment = pinned_environment(
        lane=lane,
        omp=omp,
        jax_platforms="cpu" if lane == NATIVE_LANE else _jax_lane_platform(config),
        compile_cache_dir=cache_dir,
    )
    if lane != NATIVE_LANE:
        environment.update(carried_jax_environment())
    if cache_dir is not None:
        environment.update(PERSISTENT_CACHE_THRESHOLDS)
    environment[CHILD_MARKER] = "1"
    print(
        f"[{PROBE_GRADE}] re-exec {lane} child with OMP_NUM_THREADS={omp}",
        flush=True,
    )
    completed = subprocess.run(
        child_command(
            options,
            config,
            lane,
            omp,
            output_path,
            currents_path,
            ledger_path,
            cache_dir,
        ),
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def _jax_lane_platform(config: GscoConfiguration) -> str:
    """Which backend the JAX lane's child is pinned to.

    The smoke scale is the CPU plumbing check and says so; every measured
    scale is the device lane, and pinning ``cuda`` here is what makes a leg
    that silently fell back to CPU fail in :func:`jax_runtime_facts` instead of
    publishing a CPU number under a GPU lane label.
    """
    return "cpu" if config.scale == SMOKE_SCALE else "cuda"


def carried_jax_environment() -> dict[str, str]:
    """The named backend selectors the JAX lane carries across the scrub."""
    return {
        name: os.environ[name]
        for name in JAX_LANE_CARRIED_VARIABLES
        if name in os.environ
    }


def verify_thread_pin(omp: int) -> int | None:
    """Refuse to measure a leg the pin did not actually reach.

    Two checks, because they catch different failures: the environment says
    what was requested of this interpreter, and ``observed_openmp_threads``
    says what libgomp will hand out.  ``None`` from the readback means libgomp
    could not be loaded at all -- a host fact, recorded as itself and never
    folded into a thread count.
    """
    requested = str(omp)
    inherited = os.environ.get("OMP_NUM_THREADS")
    if inherited != requested:
        raise ProbeConventionError(
            f"child asked for OMP_NUM_THREADS={requested} but observed {inherited!r}"
        )
    observed = observed_openmp_threads()
    if observed is not None and observed != omp:
        raise ProbeConventionError(
            f"child pinned OMP_NUM_THREADS={requested} but libgomp reports "
            f"omp_get_max_threads()={observed}"
        )
    return observed


def publication_path(path: Path, option: str) -> Path:
    """Resolve one published path under the relaxed publication rule.

    The rule the sibling probes use (``benchmarks/marginal_quartet_probes.py``,
    ``_validate``): a path *inside* the repository has to live under
    :data:`EVIDENCE_ROOT`, because a probe that scatters artifacts through the
    source tree makes the evidence root stop meaning anything.  A path entirely
    *outside* the repository is a scratch run -- a smoke, a tmpfs rehearsal --
    and is allowed, marked in the artifact as
    ``published_under_evidence_root: false`` rather than refused.

    Resolved, so the boolean stamped in the payload and the string published
    beside it describe the same file, and so parent and child cannot disagree
    about where a relative path pointed.
    """
    resolved = path.resolve()
    if REPO_ROOT in resolved.parents and EVIDENCE_ROOT not in resolved.parents:
        raise ProbeConventionError(
            f"{option} inside the repository must live under {EVIDENCE_ROOT}, "
            f"got {resolved}; a path outside the repository is accepted and is "
            "recorded as published_under_evidence_root: false"
        )
    return resolved


def resolve_paths(
    options: argparse.Namespace, config: GscoConfiguration, lane: str
) -> tuple[Path, Path]:
    """Artifact and currents paths, validated and with the suffix enforced.

    ``np.save`` appends ``.npy`` to any path that does not already end in it,
    so a pre-flight "refusing to overwrite" check on the requested path would
    guard a file the run never writes.  Enforcing the suffix, rather than
    silently repairing it, keeps the checked path and the written path the
    same one.

    Both published paths go through :func:`publication_path`: the currents dump
    is evidence exactly as the artifact is, and defaults to a sibling of it.
    """
    output_path = publication_path(
        options.output or default_output_path(config, lane), "--output"
    )
    currents_path = options.currents_out or output_path.with_suffix(".currents.npy")
    if currents_path.suffix != ".npy":
        raise ProbeConventionError(
            f"--currents-out must end in .npy (np.save appends it), got {currents_path}"
        )
    return output_path, publication_path(currents_path, "--currents-out")


def run_probe(options: argparse.Namespace) -> int:
    scale = resolve_scale(options)
    if options.sibling is None:
        raise ProbeConventionError("--sibling is required")
    config = configuration(str(options.sibling), scale)

    if options.dry_run:
        problem = build_problem(config)
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "grade": PROBE_GRADE,
                    "mode": "dry-run",
                    "configuration": asdict(config),
                    "dimensions": problem_dimensions(config, problem),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if options.lane is None:
        raise ProbeConventionError("--lane is required")
    lane = str(options.lane)
    if options.repeat < 1:
        raise ProbeConventionError(f"--repeat must be >= 1, got {options.repeat}")
    omp = resolve_omp(options, lane)
    if lane == NATIVE_LANE and options.compile_cache is not None:
        raise ProbeConventionError(
            "--compile-cache selects a JAX persistent compilation cache; the "
            "native lane compiles nothing"
        )
    cache_dir = (
        None if options.compile_cache is None else options.compile_cache.resolve()
    )
    output_path, currents_path = resolve_paths(options, config, lane)
    for path in (output_path, currents_path):
        if path.exists():
            raise ProbeConventionError(f"refusing to overwrite existing {path}")
    ledger_path = Path(options.ledger)

    if os.environ.get(CHILD_MARKER) != "1":
        return relaunch_pinned_child(
            options,
            config,
            lane,
            omp,
            output_path,
            currents_path,
            ledger_path,
            cache_dir,
        )

    import numpy as np

    observed_threads = verify_thread_pin(omp)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{PROBE_GRADE}] {config.sibling} / {config.scale} / {lane} "
        f"repeat={options.repeat} omp={omp}",
        flush=True,
    )

    lane_facts: dict[str, object] = {}
    if lane == NATIVE_LANE:
        lane_result = run_native_lane(config, int(options.repeat))
    else:
        lane_facts = jax_runtime_facts(require_gpu=scale != SMOKE_SCALE)
        lane_result = run_jax_lane(config, int(options.repeat), cache_dir)

    for row in lane_result.rows:
        append_leg_ledger(
            ledger_path,
            {
                "schema": SCHEMA,
                "probe": "wireframe_gsco_siblings_reference_scale",
                "grade": PROBE_GRADE,
                "lane": lane,
                "sibling": config.sibling,
                "scale": config.scale,
                "omp_requested": omp,
                "omp_observed": observed_threads,
                "pid": os.getpid(),
                "wallclock_utc": datetime.now(timezone.utc).isoformat(),
                "output": str(output_path),
                "row": row,
            },
        )

    reference_currents = lane_result.currents[0]
    repeats_identical = all(
        bool(np.array_equal(reference_currents, other))
        for other in lane_result.currents[1:]
    )
    currents_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(currents_path, reference_currents)
    _write_currents_sidecar(currents_path, config)

    payload: dict[str, object] = {
        "schema": SCHEMA,
        "probe": "wireframe_gsco_siblings_reference_scale",
        "plan_task": "P2.1",
        "identity": runtime_identity(lane),
        "lane": lane,
        "sibling": config.sibling,
        "scale": config.scale,
        "omp_requested": omp,
        "omp_num_threads_environment": os.environ.get("OMP_NUM_THREADS"),
        "omp_observed_max_threads": observed_threads,
        "compile_cache_dir": None if cache_dir is None else str(cache_dir),
        "compile_cache_entries_final": cache_entry_count(cache_dir),
        "compile_cache_thresholds": (
            {} if cache_dir is None else dict(PERSISTENT_CACHE_THRESHOLDS)
        ),
        "jax_lane_carried_environment": (
            {} if lane == NATIVE_LANE else carried_jax_environment()
        ),
        "pinned_environment_scrub_prefixes": list(SCRUBBED_ENVIRONMENT_PREFIXES),
        "repeat_count": int(options.repeat),
        "repeat_label_counts": repeat_label_counts(lane_result.rows),
        "configuration": asdict(config),
        "dimensions": lane_result.dimensions,
        "lane_runtime": lane_facts,
        "repeats": lane_result.rows,
        "repeats_bitwise_identical": repeats_identical,
        "notes": nonfinite_notes(lane_result.rows),
        "output_path": str(output_path),
        "published_under_evidence_root": EVIDENCE_ROOT in output_path.parents,
        "currents_path": str(currents_path),
        "currents_sha256": sha256_file(currents_path),
        "ledger_path": str(ledger_path),
        "process_wall_s": time.monotonic() - _PROCESS_START,
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "disclosures": list(DISCLOSURES),
    }
    write_probe_artifact(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {output_path}", flush=True)
    print(f"wrote {currents_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    options = parse_arguments(argv)
    if options.compare is not None:
        print(
            json.dumps(
                compare_currents(options.compare[0], options.compare[1]),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    return run_probe(options)


if __name__ == "__main__":
    raise SystemExit(main())
