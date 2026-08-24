"""Diagnostic native-vs-JAX A/B probes for the permanent-magnet family (plan Phase 3).

One subcommand per rung of tasks P3.2-P3.5 of
``docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md``:

``pm-simple-16``
    GPMO baseline on the NCSX/FAMUS grid at the native non-CI scale
    (nphi=ntheta=16, downsample=4, K=500) -- the parity case's SSOT
    (``examples/jax/parity/cases/native_permanent_magnet_simple.py``,
    ``_scale_configuration``), not the mirror's bounded 2x2/100 branch.
    Supersedes the 2026-07-26 N=3 diagnostic under probe conventions (P3.2).
``muse-shipped``
    MUSE ``ArbVec_backtracking`` at the shipped scale.  Directly tests whether
    the device-assignment row's 4.05x loss is stale now that the frozen-step
    ``lax.cond`` skip has landed (``src/simsopt_jax/core/pm_optimization.py``,
    ``gpmo_arbvec_backtracking_step`` dispatching
    ``_gpmo_arbvec_backtracking_frozen_step``) (P3.3).
``muse-64`` / ``pm4stell-64``
    The same two configurations at the ">= 64 for a real run" resolution their
    native sources name, ``downsample`` held at 10 so the ``(N,N)``
    connectivity transient does not bind (P3.4).
``qa-64-memory``
    The relax-and-split QA grid at nphi=ntheta=64.  ``ndipoles`` there has
    never been measured (the grid is built per-phi-plane), so the default run
    reports the grid and its would-be device footprint and solves nothing;
    ``--solve`` arms the timed MwPGP leg for later (P3.5).

Grade
-----
Every artifact this script writes is stamped ``diagnostic-not-certifying`` by
``benchmarks/probe_conventions.py``.  A probe measures; only a preregistered
charter certifies (plan Section "Campaign protocol").

Lanes
-----
One invocation runs exactly one leg, because the two lanes need different
interpreters.  ``--dry-run`` prints the interleaved schedule the operator
should follow so neither lane systematically leads.

The native leg re-executes itself as a child with ``OMP_NUM_THREADS`` pinned
*before* libgomp initializes; the JAX leg resolves its backend environment
before ``jax`` is imported, which is why this module imports nothing heavier
than the standard library at top level.

``--omp`` is mandatory for any timed leg, on both lanes.  The fair-native
denominator is the swept optimum over
``benchmarks.probe_conventions.OMP_SWEEP``, never a shipped default, and a
number quoted without a pinned thread count is not evidence.

Cold and warm
-------------
The convention is symmetric and applies to *both* lanes: solve index 0 is
``cold_in_process`` and is excluded from ``warm_seconds`` everywhere.  The
native lane's first call pays first-touch page faults, the C++ extension's
lazy relocation and the OpenMP team's first fork; the JAX lane's pays XLA
compilation.  They are not the same cost, but counting one and discarding the
other is a lane-shaped bias in the warm mean, so neither is counted.
``warm_sample_count`` is published per leg, and a timed leg therefore needs
``--repeat >= 2`` on both lanes.

Rows 1.. are *warm* only where the leg has an executable to reuse.  The QA
relax-and-split JAX leg does not: ``relax_and_split_jax`` and
``_relax_and_split_scan`` are unjitted and rebuild their scan closure per call,
so every repeat re-traces.  Those rows are labelled ``repeat_retrace``, they
are excluded from ``warm_seconds``, and ``--cache-dir`` is mandatory there with
``--repeat > 1`` so at least the compile half is served from the persistent
cache.  ``timings.row_labels`` names every row.

History
-------
The two lanes' history bookkeeping is *variant-dependent* and cannot be
equalized for the ArbVec variants at all -- ``gpmo_arbvec_backtracking_solve``
has no ``retain_history`` parameter, and ``GPMO_ArbVec_backtracking`` writes
its ``m_history`` twice from outside the ``verbose`` predicate.  So the probe
publishes the true per-lane write counts and byte formulas instead of a claimed
equality; see the History bookkeeping (SSOT) block below.

Executed order
--------------
Nothing here publishes an *intended* interleave into an artifact.  Each
executed leg appends one line to the shared append-only ledger (``--ledger``),
so the order the legs actually ran in is provable from the file rather than
asserted by a schedule this process never followed.  ``--dry-run`` still
*prints* a suggested alternation for the operator; that is a plan, not a
measurement, and it never reaches an artifact.

Citations
---------
Every ``source``/``provenance``/``notes`` string published into an artifact
cites FILE plus SYMBOL (a function or module-level constant name).  Line
numbers drift silently and were measured wrong here on 2026-08-23; they are
allowed only in transient in-file comments, never in a published string.

Interpreters
------------
Native lane::

    OMP_NUM_THREADS=<n> JAX_ENABLE_X64=1 \
    PYTHONPATH=src:build/cp311-cp311-linux_x86_64 \
    .venv/bin/python benchmarks/pm_gpmo_probes.py pm-simple-16 \
        --lane native --omp 16 --repeat 2 \
        --output docs/receipts/evidence/pm_gpmo_probe_pm_simple_16_native.json

JAX GPU lane::

    JAX_ENABLE_X64=1 PYTHONPATH=src:build/cp311-cp311-linux_x86_64 \
    .venv-qn-gpu/bin/python benchmarks/pm_gpmo_probes.py pm-simple-16 \
        --lane jax-gpu --omp 16 --repeat 2 --backend-mode jax_gpu_fast \
        --output docs/receipts/evidence/pm_gpmo_probe_pm_simple_16_jax.json

The plan requires the first 64-rung run to use ``--preallocate-off`` together
with a small ``--k-override`` so peak device memory is observed before the
full iteration budget is committed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.probe_conventions import (
    COMPILATION_CACHE_VARIABLE,
    OMP_SWEEP,
    PROBE_GRADE,
    SCRUBBED_ENVIRONMENT_PREFIXES,
    THREAD_COUNT_VARIABLES,
    ProbeConventionError,
    append_leg_ledger,
    array_sha256,
    interleave_schedule,
    observed_openmp_threads,
    pinned_environment,
    runtime_identity,
    sha256_file,
    ulp_distance,
    write_probe_artifact,
)

SCHEMA = "pm-gpmo-probe.v1"

PUBLICATION = (
    "Permanent-magnet family native-vs-JAX A/B probe (plan Phase 3, tasks "
    "P3.2-P3.5). Diagnostic timing and grid-footprint measurement only: no "
    "gate, no win rule, no certified speed claim."
)

TEST_DATA = REPO_ROOT / "tests" / "test_files"

#: Probe artifacts are published here and nowhere else *inside the repository*.
#: A run whose ``--output`` lands entirely outside the repository is a scratch
#: run: it is allowed, and it is stamped ``published_under_evidence_root:
#: false`` in its own configuration block.  This is the rule
#: ``benchmarks/marginal_quartet_probes.py`` (``EVIDENCE_ROOT``, ``_validate``)
#: already applies; the stricter in-repo-only variant this file used to enforce
#: made an operator's scratch run impossible to write at all rather than
#: impossible to mistake for evidence.
EVIDENCE_DIRECTORY = REPO_ROOT / "docs" / "receipts" / "evidence"

#: Append-only record of the legs that actually executed, shared by every probe
#: invocation.  This replaces the fabricated ``interleave_schedule`` field the
#: artifacts used to carry: a schedule is an intention, a ledger line is a fact.
DEFAULT_LEDGER = EVIDENCE_DIRECTORY / "probe_leg_ledger.jsonl"

NATIVE_LANE = "native"
JAX_LANE = "jax-gpu"
LANES = (NATIVE_LANE, JAX_LANE)

#: Set in the re-executed native child so it does not re-exec forever.  It
#: survives the scrub in :func:`pinned_environment` because it is not in
#: :data:`SCRUBBED_ENVIRONMENT_PREFIXES` -- deliberately, since scrubbing it
#: would make the child re-exec forever.
CHILD_MARKER = "SIMSOPT_PM_GPMO_PROBE_CHILD"

#: Resolved before ``import jax``; ``false`` lets the 64-rung probes watch peak
#: device memory grow instead of seeing one pre-grabbed slab.
PREALLOCATE_VARIABLE = "XLA_PYTHON_CLIENT_PREALLOCATE"

#: ``simsopt_jax`` resolves platform, x64 and matmul precision from this.
BACKEND_MODE_VARIABLE = "SIMSOPT_BACKEND_MODE"

#: Pinned alongside ``--cache-dir``, and only there.  JAX's shipped defaults
#: refuse to persist an entry that compiled in under a second or is under a
#: size floor, so a leg can be genuinely cold, write nothing, and still look
#: like a cache hit to the next leg.  Both are pinned to ``"0"`` -- the value
#: ``benchmarks/wireframe_gsco_siblings_reference_scale.py``'s
#: ``PERSISTENT_CACHE_THRESHOLDS`` uses -- rather than the ``"-1"`` some older
#: harnesses carry, so that the two probe families' warm lanes mean the same
#: thing.  :func:`pinned_environment` scrubs the whole ``JAX_`` family, so this
#: probe has to pin them itself, after the scrub.
PERSISTENT_CACHE_THRESHOLDS = {
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
}

#: Exactly the variables a leg *pins* -- what the artifact's
#: ``applied_environment`` reports.  Publishing the whole post-scrub
#: environment instead would bury the pins under the caller's inherited
#: ``PATH``/``HOME``, and publishing nothing would leave the pin unprovable.
#: The scrub itself is disclosed separately, by prefix, in the same block.
APPLIED_ENVIRONMENT_NAMES = (
    *THREAD_COUNT_VARIABLES,
    "OMP_DYNAMIC",
    "OMP_SCHEDULE",
    "JAX_ENABLE_X64",
    "JAX_PLATFORMS",
    COMPILATION_CACHE_VARIABLE,
    *PERSISTENT_CACHE_THRESHOLDS,
    "MPI4PY_RC_INITIALIZE",
    "HWLOC_COMPONENTS",
    BACKEND_MODE_VARIABLE,
    PREALLOCATE_VARIABLE,
    CHILD_MARKER,
)

#: Accepted suffixes for ``--moments-out``.  The payload is always an ``npz``
#: archive (moments plus their metadata); ``.npy`` is accepted because the
#: cross-lane comparison used to be handed bare arrays, and refusing the
#: suffix outright would silently invalidate an operator's muscle memory
#: instead of loudly writing the archive they asked for.
MOMENTS_SUFFIXES = (".npy", ".npz")

#: The fields two moment files must agree on before ``--compare`` will form a
#: verdict, mirroring ``benchmarks/stochastic_stage_two_probe.py``'s
#: ``_ENDPOINT_IDENTITY_FIELDS`` gate.  ``lane`` is deliberately absent: a
#: cross-lane comparison is exactly the case where it must differ.  The two
#: grid digests are the strongest members -- they prove both lanes were handed
#: the same host problem, not merely the same nominal configuration.
MOMENTS_IDENTITY_FIELDS = (
    "case",
    "nphi",
    "ntheta",
    "downsample",
    "iterations",
    "history_policy",
    "A_obj_sha256",
    "b_obj_sha256",
)

#: Labels for the rows of ``timings.solve_seconds``.  Row 0 is cold on both
#: lanes by convention; what rows 1.. are is a property of the leg, not of the
#: lane, so it is named per leg instead of assumed to be ``warm``.
COLD_ROW = "cold_in_process"
WARM_ROW = "warm"
RETRACE_ROW = "repeat_retrace"

#: Why the QA relax-and-split JAX rows are all cold, and what the probe does
#: about it.  Measured against the shipped library, not a jitted counterfactual.
JAX_RELAX_SPLIT_RETRACE = (
    "relax_and_split_jax and _relax_and_split_scan carry no @jax.jit "
    "(src/simsopt_jax/solve/permanent_magnet.py), and with reg_l0 != 0 the "
    "_scan_body closure is rebuilt on every call, so each solve re-traces AND "
    "re-compiles rather than reusing an executable. The two continuation "
    "stages cannot share one either: reg_l0 * (stage + 1) / stages is baked in "
    "as a constant, so each stage is its own program. Rows 1.. are therefore "
    "repeat_retrace, NOT warm -- they measure trace plus compile plus execute "
    "-- and this leg publishes warm_sample_count 0 rather than a mean of cold "
    "solves. --cache-dir is required with --repeat > 1 here so that at least "
    "the compile half is served from the persistent cache; the trace half is "
    "paid on every call. Jitting the library would remove the retrace and is "
    "deliberately out of this probe's scope: a probe measures the code that "
    "ships, and this is a disclosure, not a defect report against src/."
)

#: What the shipped examples set, quoted so the override is legible.  The
#: examples do not agree with each other: ``permanent_magnet_MUSE.py`` sets
#: ``nHistory = 20`` while ``permanent_magnet_PM4Stell.py`` sets
#: ``nHistory = 10`` and ``permanent_magnet_simple.py`` sets
#: ``kwargs['nhistory'] = 10``.
SHIPPED_NHISTORY_NOTE = (
    "History equalization OVERRIDES the shipped configuration, and the shipped "
    "values are not one number: examples/1_Simple/permanent_magnet_simple.py "
    "sets kwargs['nhistory'] = 10, "
    "examples/2_Intermediate/permanent_magnet_MUSE.py sets nHistory = 20, and "
    "examples/2_Intermediate/permanent_magnet_PM4Stell.py sets nHistory = 10. "
    "The timed configuration is the equalized one, so a number from here is "
    "not a shipped-example runtime."
)

#: The iteration semantics the two lanes do NOT share.
MATCHED_WORK_ITERATION_NOTE = (
    "configuration.iterations is a BUDGET (K), not an executed count, and the "
    "two lanes spend it differently. Native GPMO_ArbVec_backtracking breaks "
    "out of its k loop at 'if ((num_nonzero >= N) || (num_nonzero >= "
    "max_nMagnets))' (src/simsoptpp/permanent_magnet_optimization.cpp), so at "
    "the shipped MUSE settings (K=10000, max_nMagnets=5000) it executes only "
    "the prefix up to that break; the prefix is published as "
    "iteration_report.last_reported_iteration whenever the variant prints it. "
    "The JAX lane's jax.lax.scan has no early exit and executes all K: past "
    "the stopping point gpmo_arbvec_backtracking_step's lax.cond dispatches "
    "_gpmo_arbvec_backtracking_frozen_step, which skips the candidate scan and "
    "the dewyrming pass -- cheap, but nonzero. The lanes therefore do NOT do "
    "matched work at equal K, and nothing here divides that out."
)

MU0 = 4.0e-7 * 3.141592653589793
PI = 3.141592653589793


class ProbeError(RuntimeError):
    """The probe cannot produce a trustworthy measurement as configured."""


# --------------------------------------------------------------------------
# History bookkeeping (SSOT)
# --------------------------------------------------------------------------
#
# The two lanes do not record history the same way, the difference is NOT the
# same for every GPMO variant, and it must be equalized as far as each variant
# allows before timing rather than asserted afterwards.  Everything below is
# quoted from the two implementations; line numbers are transient and appear
# only in this comment, never in a published string.
#
# Native (``src/simsoptpp/permanent_magnet_optimization.cpp``).  The claim that
# "every GPMO variant writes history only inside the verbose predicate" is
# FALSE, and it is false for the three ArbVec_backtracking cases this probe
# runs.  ``print_GPMO`` is the only writer (it stores ``x`` into
# ``m_history(:, :, print_iter)`` and advances ``print_iter``), and its call
# sites are:
#
#   * ``GPMO_baseline`` (:1237), ``GPMO_multi`` (:583), ``GPMO_ArbVec``
#     (:1123): one call site each, at :1327/:708/:1228, inside
#     ``if (verbose && (((k % int(K / nhistory)) == 0) || k == 0 || k == K-1))``.
#     Buffer ``xt::zeros<double>({N, 3, nhistory + 1})``.
#   * ``GPMO_backtracking`` (:380): the verbose-gated call at :541 PLUS an
#     UNCONDITIONAL call at :560, inside the magnet-limit exit
#     ``if ((num_nonzero >= N) || (num_nonzero >= max_nMagnets))`` (:558),
#     which then ``break``s.  Buffer ``nhistory + 1``.
#   * ``GPMO_ArbVec_backtracking`` (:729): an UNCONDITIONAL pre-loop call at
#     :791 ("Save a record of the magnet array as initialized"), the
#     verbose-gated call at :942, and the UNCONDITIONAL magnet-limit-exit call
#     at :966 (:964 predicate, then ``break``).  Buffer ``nhistory + 2``, which
#     is exactly the three slots those three sites need.
#
# The buffer is ``xt::zeros`` -- allocated whatever ``verbose`` says -- so the
# native history *bytes* are ``ndipoles * 3 * (nhistory + slack) * 8`` even at
# ``verbose=False``, and the write count is a separate fact from the footprint.
# Because the magnet-limit exit ``break``s, it is mutually exclusive with the
# ``k == K-1`` verbose write: the two termination routes cost the same three
# writes at ``nhistory=1``, which is why both totals are published.
#
# JAX (``src/simsopt_jax/core/pm_optimization.py``,
# ``src/simsopt_jax/solve/permanent_magnet.py``).  Only ``gpmo_baseline_solve``
# takes ``retain_history``; ``gpmo_arbvec_backtracking_solve`` does not.  So
# for the ArbVec variants ``record_every=None`` does NOT mean "record nothing",
# it selects the FULL-TRACE ``jax.lax.scan`` whose trace element 4 is ``x``,
# materializing ``x_history`` of shape ``(K, ndipoles, 3)``; then
# ``_gpmo_public_result`` eagerly computes
# ``m_history = x_history * m_maxima[None, :, None]``, a second array of the
# same size.  At MUSE scale (K=10000, ndipoles=7530) that is ~1.68 GiB each,
# ~3.4 GiB live, per call -- i.e. ``record_every=None`` is the MAXIMAL-memory
# setting for those variants, not the minimal one.  ``record_every=K`` selects
# ``_gpmo_recording_scan`` and exactly one recorded row (``_record_rows``).
#
# Hence the policy is VARIANT-AWARE.  ``off`` means "the least history this
# variant can be made to keep": ``retain_history=False`` (zero rows) where the
# solver offers it, ``record_every=K`` (one row) where it does not.  That is
# not an equal count across lanes -- for ArbVec_backtracking no setting is --
# so the per-lane counts and the byte asymmetry are published as a disclosure
# instead of a claimed equality.  ``final`` records one row on the JAX lane and
# three native writes into three slots.


@dataclass(frozen=True)
class VariantHistoryModel:
    """How one GPMO variant records history, per lane.

    ``native_buffer_slack`` is the ``+1``/``+2`` in the C++
    ``xt::zeros<double>({N, 3, nhistory + slack})`` allocation.  The two
    unconditional-write flags are the call sites that sit *outside* the
    ``verbose`` predicate.  ``jax_retain_history`` says whether the JAX solver
    exposes a ``retain_history`` parameter at all -- where it does not,
    ``record_every=None`` selects the full-trace branch rather than a silent
    lane.
    """

    native_symbol: str
    native_buffer_slack: int
    native_preloop_write: bool
    native_magnet_limit_exit_write: bool
    jax_retain_history: bool
    jax_symbol: str


#: Keyed by :attr:`GpmoSpec.algorithm`; only the variants this probe runs.
GPMO_HISTORY_MODELS: dict[str, VariantHistoryModel] = {
    "baseline": VariantHistoryModel(
        native_symbol="GPMO_baseline",
        native_buffer_slack=1,
        native_preloop_write=False,
        native_magnet_limit_exit_write=False,
        jax_retain_history=True,
        jax_symbol="gpmo_baseline_solve",
    ),
    "ArbVec_backtracking": VariantHistoryModel(
        native_symbol="GPMO_ArbVec_backtracking",
        native_buffer_slack=2,
        native_preloop_write=True,
        native_magnet_limit_exit_write=True,
        jax_retain_history=False,
        jax_symbol="gpmo_arbvec_backtracking_solve",
    ),
}


@dataclass(frozen=True)
class HistoryPolicy:
    """How much optimizer history each lane is asked to keep."""

    name: str
    native_verbose: bool
    native_nhistory: int
    jax_records_final: bool
    rationale: str


HISTORY_POLICIES: dict[str, HistoryPolicy] = {
    "off": HistoryPolicy(
        name="off",
        native_verbose=False,
        native_nhistory=1,
        jax_records_final=False,
        rationale=(
            "Least history each variant can be made to keep, which is NOT the "
            "same count on the two lanes and cannot be. On the JAX lane the "
            "baseline solver takes retain_history=False (zero rows), while "
            "gpmo_arbvec_backtracking_solve has no such parameter, so its "
            "minimum is record_every=K (one row): record_every=None would "
            "select the full-trace scan and materialize x_history plus the "
            "eager m_history at (K, ndipoles, 3) each. On the native lane "
            "verbose=False silences the periodic predicate, but "
            "GPMO_ArbVec_backtracking still writes unconditionally before the "
            "loop and again on the magnet-limit exit. The per-lane write "
            "counts and history bytes are published side by side rather than "
            "claimed equal."
        ),
    ),
    "final": HistoryPolicy(
        name="final",
        native_verbose=True,
        native_nhistory=1,
        jax_records_final=True,
        rationale=(
            "Final-state snapshot: JAX records exactly one row (k=K-1) via "
            "_gpmo_recording_scan. Native writes more, and how many depends on "
            "the variant: GPMO_baseline writes two (k=0 and k=K-1, both "
            "hard-coded in its predicate), GPMO_ArbVec_backtracking writes "
            "three into its three slots (the unconditional pre-loop record, "
            "k=0, and then either the unconditional magnet-limit-exit record "
            "or k=K-1 -- the exit breaks, so never both). The asymmetry is "
            "reported per lane, not hidden."
        ),
    ),
}


@dataclass(frozen=True)
class NativeHistoryWrites:
    """Truthful ``m_history`` accounting for one C++ GPMO variant under a policy.

    ``buffer_slots`` is allocated unconditionally, so ``buffer_bytes`` is the
    native footprint even when no write happens.  The two totals are the two
    termination routes, which are mutually exclusive because the magnet-limit
    exit ``break``s out of the ``k`` loop:
    ``writes_if_budget_exhausted`` runs all ``K`` iterations and takes the
    ``k == K-1`` verbose write; ``writes_if_magnet_limit_reached`` stops early
    and takes the unconditional exit write instead, keeping only the periodic
    writes guaranteed for any prefix (the hard-coded ``k == 0``).  It is
    ``None`` for a variant that has no magnet-limit exit at all.
    """

    native_symbol: str
    verbose: bool
    nhistory: int
    buffer_slots: int
    unconditional_preloop: int
    verbose_periodic_over_budget: int
    verbose_periodic_guaranteed: int
    unconditional_magnet_limit_exit: int
    writes_if_budget_exhausted: int
    writes_if_magnet_limit_reached: int | None


def native_history_writes(
    iterations: int, policy: HistoryPolicy, model: VariantHistoryModel
) -> NativeHistoryWrites:
    """Where ``print_GPMO`` will write, for one variant under one policy."""
    if not 1 <= policy.native_nhistory <= iterations:
        raise ProbeError(
            f"nhistory must satisfy 1 <= nhistory <= K; got "
            f"{policy.native_nhistory} with K={iterations}"
        )
    stride = iterations // policy.native_nhistory
    periodic_over_budget = (
        sum(
            1
            for step in range(iterations)
            if step % stride == 0 or step == 0 or step == iterations - 1
        )
        if policy.native_verbose
        else 0
    )
    # The predicate hard-codes ``k == 0``, so a verbose run writes there on any
    # prefix; every later periodic write depends on where the run stops.
    periodic_guaranteed = 1 if policy.native_verbose else 0
    preloop = 1 if model.native_preloop_write else 0
    exit_write = 1 if model.native_magnet_limit_exit_write else 0
    return NativeHistoryWrites(
        native_symbol=model.native_symbol,
        verbose=policy.native_verbose,
        nhistory=policy.native_nhistory,
        buffer_slots=policy.native_nhistory + model.native_buffer_slack,
        unconditional_preloop=preloop,
        verbose_periodic_over_budget=periodic_over_budget,
        verbose_periodic_guaranteed=periodic_guaranteed,
        unconditional_magnet_limit_exit=exit_write,
        writes_if_budget_exhausted=preloop + periodic_over_budget,
        writes_if_magnet_limit_reached=(
            preloop + periodic_guaranteed + exit_write
            if model.native_magnet_limit_exit_write
            else None
        ),
    )


@dataclass(frozen=True)
class JaxHistoryPlan:
    """The ``record_every``/``retain_history`` pair one leg will actually pass.

    ``full_trace_branch`` records whether this pair would select the
    unrecorded ``jax.lax.scan`` that stacks ``x`` every iteration.  It is
    ``False`` for every pair :func:`jax_history_plan` returns -- the function
    refuses to build any other -- and is published so the refusal is legible in
    the artifact instead of only in this file.
    """

    record_every: int | None
    retain_history: bool
    recorded_rows: int
    full_trace_branch: bool
    full_trace_rows_avoided: int
    #: ``x_history`` and the eager ``m_history`` in ``_gpmo_public_result``.
    materialized_history_arrays: int


def _jax_recorded_rows(iterations: int, record_every: int) -> int:
    """Rows ``_record_rows`` yields, reimplemented without importing JAX."""
    rows = list(range(record_every - 1, iterations, record_every))
    if not rows or rows[-1] != iterations - 1:
        rows.append(iterations - 1)
    return len(rows)


def jax_history_plan(
    iterations: int, policy: HistoryPolicy, model: VariantHistoryModel
) -> JaxHistoryPlan:
    """Resolve one variant's JAX history levers, refusing the full-trace branch.

    The refusal is the assertion the history model rests on: for a solver
    without ``retain_history``, ``record_every=None`` is the maximal-memory
    setting, not the minimal one, so no policy this probe offers is allowed to
    reach it.  By construction none does; the raise makes that checkable rather
    than assumed.
    """
    if policy.jax_records_final:
        record_every: int | None = iterations
        retain_history = True
    elif model.jax_retain_history:
        record_every, retain_history = None, False
    else:
        record_every, retain_history = iterations, True
    full_trace_branch = record_every is None and not model.jax_retain_history
    if full_trace_branch:
        raise ProbeError(
            f"history policy {policy.name!r} would call {model.jax_symbol} with "
            "record_every=None, which selects the full-trace scan: x_history "
            f"and the eager m_history at (K={iterations}, ndipoles, 3) each. "
            "That branch is unreachable from this probe by construction"
        )
    rows = 0 if record_every is None else _jax_recorded_rows(iterations, record_every)
    return JaxHistoryPlan(
        record_every=record_every,
        retain_history=retain_history,
        recorded_rows=rows,
        full_trace_branch=full_trace_branch,
        full_trace_rows_avoided=(0 if model.jax_retain_history else iterations),
        materialized_history_arrays=2,
    )


# --------------------------------------------------------------------------
# Case tables (SSOT)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    """Everything the host needs to rebuild one permanent-magnet grid."""

    nphi: int
    ntheta: int
    downsample: int | None
    coordinate_flag: str
    dr: float | None
    inner_offset: float | None
    outer_offset: float | None
    source: str


@dataclass(frozen=True)
class GpmoSpec:
    """One GPMO variant's hyperparameters, identical on both lanes."""

    algorithm: str
    iterations: int
    reg_l2: float
    single_direction: int
    adjacent: int | None
    backtracking: int | None
    max_magnets: int | None
    threshold_angle: float | None


@dataclass(frozen=True)
class RelaxSplitSpec:
    """The QA relax-and-split continuation, as the native example runs it."""

    stages: int
    reg_l0: float
    nu: float
    max_iter: int
    max_iter_rs: int


@dataclass(frozen=True)
class GridBuild:
    """A built CPU grid plus the facts a probe reports about it."""

    grid: object
    seconds: float
    input_sha256: dict[str, str]
    polarization_count: int | None


@dataclass(frozen=True)
class ProbeCase:
    """One subcommand: a grid, a solver configuration, and its provenance."""

    name: str
    summary: str
    grid: GridSpec
    builder: Callable[[GridSpec], GridBuild]
    gpmo: GpmoSpec | None
    relax_split: RelaxSplitSpec | None
    solve_by_default: bool
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class SolveResult:
    """What one timed leg produced: endpoints, per-solve times, box readings.

    ``device_memory_mib`` is empty on the native lane -- not zero-filled.  A
    native leg holds no device memory, and a reading of ``0`` would claim a
    measurement nobody took; an absent key says no measurement applies.
    """

    moments: object
    seconds: tuple[float, ...]
    device_memory_mib: dict[str, object]
    #: What the lane can prove about how many iterations it actually ran, as
    #: opposed to the budget it was given.  Empty where the lane exposes
    #: nothing (the relax-and-split legs), never zero-filled.
    iteration_report: dict[str, object]


def device_memory_used_mib() -> list[dict[str, int]] | None:
    """Per-device ``memory.used``, or ``None`` off GPU hosts.

    Same query shape as ``benchmarks/probe_conventions.py``'s device
    inventory (``_gpu_identity``/:func:`gpu_compute_processes`): resolve
    ``nvidia-smi`` on ``PATH``, ask for ``nounits`` CSV, fail loudly on a bad
    query rather than reporting an empty reading.  ``None`` means the host has
    no ``nvidia-smi`` at all -- distinct from "zero bytes in use", and the two
    must not encode the same, which is why the memory rung's fields are
    nullable rather than defaulted to 0.

    This is a whole-device reading, not this process's allocation: it includes
    every other context on the card.  ``gpu_compute_processes`` in the identity
    block is what says whether there was one.
    """
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, int]] = []
    for line in completed.stdout.strip().splitlines():
        index_text, used_text = (field.strip() for field in line.split(","))
        rows.append({"index": int(index_text), "used_mib": int(used_text)})
    return rows


def _positive_polarization_family(name: str, family: int):
    """The positive half of one polarization family, offset by its type index."""
    from simsopt.util import polarization_axes

    axes, types = polarization_axes([name])
    positive = len(types) // 2
    return axes[:positive], types[:positive] + family


#: ``PermanentMagnetGrid.geo_setup_from_famus`` pops exactly these four names
#: out of its ``**kwargs`` and never reads the mapping again
#: (``src/simsopt/geo/permanent_magnet_grid.py``,
#: ``PermanentMagnetGrid.geo_setup_from_famus``), so every *other* keyword --
#: ``dr`` among them -- is accepted and silently discarded.  The shipped
#: ``examples/2_Intermediate/permanent_magnet_MUSE.py`` passes the same inert
#: ``dr`` in its ``kwargs`` dict, so this probe keeps passing it (the mirror
#: stays faithful to the example) and publishes it as a declared-inert kwarg
#: rather than as an applied grid parameter.
FAMUS_CONSUMED_KWARGS = (
    "coordinate_flag",
    "downsample",
    "pol_vectors",
    "m_maxima",
)

#: Why a famus-grid case's declared ``dr`` reaches no geometry.
DR_INERT_MECHANISM = (
    "geo_setup_from_famus pops only "
    + "/".join(FAMUS_CONSUMED_KWARGS)
    + " from **kwargs and discards the rest, so dr never reaches the grid; the "
    "radial extent of a FAMUS brick comes from the .focus inventory. The "
    "shipped examples/2_Intermediate/permanent_magnet_MUSE.py passes the same "
    "inert kwarg. Only geo_setup_between_toroidal_surfaces (the QA case) takes "
    "dr as a real argument."
)


def _build_simple_grid(spec: GridSpec) -> GridBuild:
    """NCSX boundary + purely toroidal background field + FAMUS inventory.

    Mirrors ``examples/1_Simple/permanent_magnet_simple.py`` (constants
    ``nphi``/``downsample``, then ``kwargs_geo``) at its non-CI branch, which
    is the scale the parity case already froze.
    """
    import numpy as np
    from simsopt.field import ToroidalField
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier

    surface_path = TEST_DATA / "wout_c09r00_fixedBoundary_0.5T_vacuum_ns201.nc"
    magnet_path = TEST_DATA / "init_orient_pm_nonorm_5E4_q4_dp.focus"
    start = time.perf_counter()
    surface = SurfaceRZFourier.from_wout(
        str(surface_path),
        range="half period",
        nphi=spec.nphi,
        ntheta=spec.ntheta,
    )
    background = ToroidalField(R0=1.0, B0=MU0 * 3.7713e6 / (2.0 * PI))
    background.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        background.B().reshape((spec.nphi, spec.ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            magnet_path,
            coordinate_flag=spec.coordinate_flag,
            downsample=spec.downsample,
        )
    seconds = time.perf_counter() - start
    return GridBuild(
        grid=grid,
        seconds=seconds,
        input_sha256={
            surface_path.name: sha256_file(surface_path),
            magnet_path.name: sha256_file(magnet_path),
        },
        polarization_count=None,
    )


def _build_muse_grid(spec: GridSpec) -> GridBuild:
    """MUSE boundary + pre-optimized TF coils + face-aligned FAMUS magnets.

    Mirrors ``examples/2_Intermediate/permanent_magnet_MUSE.py`` (constants
    ``surface_filename``/``famus_filename``/``downsample``/``dr``, the
    ``pol_vectors`` block, then ``kwargs``).
    """
    import numpy as np
    from simsopt.field import BiotSavart
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.util import FocusData, discretize_polarizations
    from simsopt.util.permanent_magnet_helper_functions import (
        initialize_coils_for_pm_optimization,
    )

    surface_path = TEST_DATA / "input.muse"
    magnet_path = TEST_DATA / "zot80.focus"
    coil_path = TEST_DATA / "muse_tf_coils.focus"
    start = time.perf_counter()
    surface = SurfaceRZFourier.from_focus(
        surface_path,
        range="half period",
        nphi=spec.nphi,
        ntheta=spec.ntheta,
    )
    with tempfile.TemporaryDirectory() as scratch, redirect_stdout(io.StringIO()):
        # The helper writes ``curves_init.vtu`` next to ``out_dir``
        # unconditionally (initialize_coils_for_pm_optimization's curves_to_vtk
        # call, permanent_magnet_helper_functions.py:93); an empty
        # default would drop it in the caller's working directory.
        _, _, coils = initialize_coils_for_pm_optimization(
            "muse_famus", TEST_DATA, surface, scratch
        )
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((spec.nphi, spec.ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(magnet_path, downsample=spec.downsample)
    axes, types = _positive_polarization_family("face", 0)
    discretize_polarizations(
        magnet_data,
        np.arctan2(magnet_data.oy, magnet_data.ox),
        axes,
        types,
    )
    polarizations = np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z), axis=-1
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            normal_field,
            magnet_path,
            pol_vectors=polarizations,
            # The CASES row declares the coordinate system, so the row is what
            # must reach the constructor.  The example leans on the
            # ``geo_setup_from_famus`` default instead ("which is the default,
            # so no need to specify it here"), which made this probe's
            # declared ``coordinate_flag`` inert: it was published in the
            # artifact and never applied.  Passing it makes the row load-bearing
            # -- and it agrees with the default here, so nothing measured moves.
            coordinate_flag=spec.coordinate_flag,
            downsample=spec.downsample,
            # Inert here, and passed anyway because the shipped example passes
            # it: see DR_INERT_MECHANISM. It is published as a declared-inert
            # kwarg, never as an applied grid parameter.
            dr=spec.dr,
        )
    seconds = time.perf_counter() - start
    return GridBuild(
        grid=grid,
        seconds=seconds,
        input_sha256={
            path.name: sha256_file(path)
            for path in (surface_path, magnet_path, coil_path)
        },
        polarization_count=int(polarizations.shape[1]),
    )


def _build_pm4stell_grid(spec: GridSpec) -> GridBuild:
    """NCSX plasma + TF coils + MAGPIE arrangement with face-triplet axes.

    Mirrors ``examples/2_Intermediate/permanent_magnet_PM4Stell.py`` (constants
    ``N``/``downsample``, the three ``polarization_axes`` families, then
    ``kwargs_geo``).
    """
    import numpy as np
    from simsopt.field import BiotSavart, Coil
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.util import (
        FocusData,
        FocusPlasmaBnormal,
        discretize_polarizations,
        orientation_phi,
        read_focus_coils,
    )

    plasma_path = TEST_DATA / "c09r00_B_axis_half_tesla_PM4Stell.plasma"
    coil_path = TEST_DATA / "tf_only_half_tesla_symmetry_baxis_PM4Stell.focus"
    magnet_path = TEST_DATA / "magpie_trial104b_PM4Stell.focus"
    corner_path = TEST_DATA / "magpie_trial104b_corners_PM4Stell.csv"
    start = time.perf_counter()
    surface = SurfaceRZFourier.from_focus(
        plasma_path,
        range="half period",
        nphi=spec.nphi,
        ntheta=spec.ntheta,
    )
    plasma_normal = FocusPlasmaBnormal(plasma_path).bnormal_grid(
        spec.nphi, spec.ntheta, "half period"
    )
    base_curves, base_currents, coil_count = read_focus_coils(coil_path)
    field = BiotSavart(
        [Coil(base_curves[i], base_currents[i]) for i in range(coil_count)]
    )
    field.set_points(surface.gamma().reshape((-1, 3)))
    coil_normal = np.sum(
        field.B().reshape((spec.nphi, spec.ntheta, 3)) * surface.unitnormal(),
        axis=2,
    )
    magnet_data = FocusData(magnet_path, downsample=spec.downsample)
    families = (
        _positive_polarization_family("face", 0),
        _positive_polarization_family("fe_ftri", 1),
        _positive_polarization_family("fc_ftri", 2),
    )
    axes = np.concatenate([family[0] for family in families], axis=0)
    types = np.concatenate([family[1] for family in families])
    discretize_polarizations(
        magnet_data,
        orientation_phi(corner_path)[: magnet_data.nMagnets],
        axes,
        types,
    )
    polarizations = np.stack(
        (magnet_data.pol_x, magnet_data.pol_y, magnet_data.pol_z), axis=-1
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_from_famus(
            surface,
            plasma_normal + coil_normal,
            magnet_path,
            pol_vectors=polarizations,
            m_maxima=5.0 / MU0,
            # See ``_build_muse_grid``: the declared row, not the constructor
            # default, is the SSOT for the coordinate system.
            coordinate_flag=spec.coordinate_flag,
            downsample=spec.downsample,
        )
    seconds = time.perf_counter() - start
    return GridBuild(
        grid=grid,
        seconds=seconds,
        input_sha256={
            path.name: sha256_file(path)
            for path in (plasma_path, coil_path, magnet_path, corner_path)
        },
        polarization_count=int(polarizations.shape[1]),
    )


def _build_qa_grid(spec: GridSpec) -> GridBuild:
    """Landreman/Paul QA grid between two offset toroidal surfaces.

    Mirrors ``examples/2_Intermediate/permanent_magnet_QA.py`` (constants
    ``nphi``/``ntheta``/``dr``/``coff``/``poff``, then ``kwargs_geo``) except
    that its ``coil_optimization`` call is not run: it changes
    only ``Bnormal``, hence ``b_obj``, and this probe's question -- ``ndipoles``
    at nphi=64 and the device footprint that follows from it -- is a property
    of the two extended surfaces and ``dr`` alone.  Both lanes then consume the
    identical host grid, so the omission cannot tilt a ratio either.
    """
    import numpy as np
    from simsopt.field import BiotSavart
    from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
    from simsopt.util.permanent_magnet_helper_functions import (
        initialize_coils_for_pm_optimization,
    )

    if spec.dr is None or spec.inner_offset is None or spec.outer_offset is None:
        raise ProbeError("the QA grid needs dr and both surface offsets")
    if spec.downsample is not None:
        raise ProbeError(
            "geo_setup_between_toroidal_surfaces has no downsample argument; "
            f"the QA grid spec must leave it unset, got {spec.downsample}"
        )
    surface_path = TEST_DATA / "input.LandremanPaul2021_QA_lowres"
    start = time.perf_counter()
    surfaces = [
        SurfaceRZFourier.from_vmec_input(
            surface_path,
            range="half period",
            nphi=spec.nphi,
            ntheta=spec.ntheta,
        )
        for _ in range(3)
    ]
    boundary, inner, outer = surfaces
    inner.extend_via_projected_normal(spec.inner_offset)
    outer.extend_via_projected_normal(spec.outer_offset)
    with tempfile.TemporaryDirectory() as scratch, redirect_stdout(io.StringIO()):
        _, _, coils = initialize_coils_for_pm_optimization(
            "qa", TEST_DATA, boundary, scratch
        )
    field = BiotSavart(coils)
    field.set_points(boundary.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((spec.nphi, spec.ntheta, 3)) * boundary.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
            boundary,
            normal_field,
            inner,
            outer,
            dr=spec.dr,
            coordinate_flag=spec.coordinate_flag,
        )
    seconds = time.perf_counter() - start
    return GridBuild(
        grid=grid,
        seconds=seconds,
        input_sha256={surface_path.name: sha256_file(surface_path)},
        polarization_count=None,
    )


#: The only builder whose constructor consumes ``dr``.  Every other builder
#: reaches :func:`PermanentMagnetGrid.geo_setup_from_famus`, which discards it
#: (:data:`DR_INERT_MECHANISM`).  Deriving "is dr applied" from the builder --
#: rather than storing a second flag next to ``GridSpec.dr`` -- keeps one
#: source: a case cannot declare an applied ``dr`` its builder never applies.
DR_CONSUMING_BUILDERS = (_build_qa_grid,)


def dr_is_applied(case: ProbeCase) -> bool:
    """Whether this case's declared ``dr`` reaches the grid constructor."""
    return case.builder in DR_CONSUMING_BUILDERS


_MUSE_GPMO = GpmoSpec(
    algorithm="ArbVec_backtracking",
    iterations=10_000,
    reg_l2=0.0,
    single_direction=-1,
    adjacent=1,
    backtracking=200,
    max_magnets=5_000,
    threshold_angle=PI,
)

_PM4STELL_GPMO = GpmoSpec(
    algorithm="ArbVec_backtracking",
    iterations=2_000,
    reg_l2=0.0,
    single_direction=-1,
    adjacent=10,
    backtracking=200,
    max_magnets=1_000,
    threshold_angle=PI,
)

CASES: dict[str, ProbeCase] = {
    "pm-simple-16": ProbeCase(
        name="pm-simple-16",
        summary="GPMO baseline, NCSX/FAMUS, native non-CI scale (plan P3.2)",
        grid=GridSpec(
            nphi=16,
            ntheta=16,
            downsample=4,
            coordinate_flag="cylindrical",
            dr=None,
            inner_offset=None,
            outer_offset=None,
            source=(
                "examples/1_Simple/permanent_magnet_simple.py "
                "(nphi/ntheta/downsample non-CI branch; kwargs_geo)"
            ),
        ),
        builder=_build_simple_grid,
        gpmo=GpmoSpec(
            algorithm="baseline",
            iterations=500,
            reg_l2=0.0,
            single_direction=-1,
            adjacent=None,
            backtracking=None,
            max_magnets=None,
            threshold_angle=None,
        ),
        relax_split=None,
        solve_by_default=True,
        provenance=(
            "examples/jax/parity/cases/native_permanent_magnet_simple.py"
            "::_scale_configuration",
            "src/simsopt/solve/permanent_magnet_optimization.py::GPMO",
            "src/simsopt_jax/solve/permanent_magnet.py::GPMO_baseline_jax",
        ),
    ),
    "muse-shipped": ProbeCase(
        name="muse-shipped",
        summary="MUSE ArbVec_backtracking at shipped scale (plan P3.3)",
        grid=GridSpec(
            nphi=16,
            ntheta=16,
            downsample=10,
            coordinate_flag="cartesian",
            dr=0.01,
            inner_offset=None,
            outer_offset=None,
            source=(
                "examples/2_Intermediate/permanent_magnet_MUSE.py "
                "(nphi/nIter_max/nBacktracking/max_nMagnets/downsample non-CI "
                "branch; dr, declared and inert on a famus grid; nAdjacent; "
                "thresh_angle; kwargs)"
            ),
        ),
        builder=_build_muse_grid,
        gpmo=_MUSE_GPMO,
        relax_split=None,
        solve_by_default=True,
        provenance=(
            "examples/jax/2_Intermediate/permanent_magnet_MUSE.py::_build_grid",
            "src/simsopt_jax/core/pm_optimization.py"
            "::gpmo_arbvec_backtracking_step (the lax.cond dispatching "
            "_gpmo_arbvec_backtracking_frozen_step)",
            "src/simsopt_jax/solve/permanent_magnet.py::GPMO_ArbVec_backtracking_jax",
        ),
    ),
    "muse-64": ProbeCase(
        name="muse-64",
        summary="MUSE ArbVec_backtracking at the real-run 64x64 grid (plan P3.4)",
        grid=GridSpec(
            nphi=64,
            ntheta=64,
            downsample=10,
            coordinate_flag="cartesian",
            dr=0.01,
            inner_offset=None,
            outer_offset=None,
            source=(
                "examples/2_Intermediate/permanent_magnet_MUSE.py "
                "(nphi, commented '>= 64 for high-resolution runs')"
            ),
        ),
        builder=_build_muse_grid,
        gpmo=_MUSE_GPMO,
        relax_split=None,
        solve_by_default=True,
        provenance=(
            "docs/jax_gpu_examples_backlog_native_speed_implementation_plan.md P3.4",
            "src/simsopt_jax/solve/permanent_magnet.py::GPMO_ArbVec_backtracking_jax",
        ),
    ),
    "pm4stell-64": ProbeCase(
        name="pm4stell-64",
        summary="PM4Stell ArbVec_backtracking at the real-run 64x64 grid (plan P3.4)",
        grid=GridSpec(
            nphi=64,
            ntheta=64,
            downsample=10,
            coordinate_flag="cartesian",
            dr=None,
            inner_offset=None,
            outer_offset=None,
            source=(
                "examples/2_Intermediate/permanent_magnet_PM4Stell.py "
                "(N, commented '>= 64 for high-resolution runs' on both the CI "
                "and the non-CI branch; this probe takes the non-CI one)"
            ),
        ),
        builder=_build_pm4stell_grid,
        gpmo=_PM4STELL_GPMO,
        relax_split=None,
        solve_by_default=True,
        provenance=(
            "examples/2_Intermediate/permanent_magnet_PM4Stell.py "
            "(the three polarization_axes families; kwargs_geo)",
            "src/simsopt_jax/solve/permanent_magnet.py::GPMO_ArbVec_backtracking_jax",
        ),
    ),
    "qa-64-memory": ProbeCase(
        name="qa-64-memory",
        summary="QA relax-and-split grid footprint at 64x64; solves only with --solve (plan P3.5)",
        grid=GridSpec(
            nphi=64,
            ntheta=64,
            downsample=None,
            coordinate_flag="cylindrical",
            dr=0.02,
            inner_offset=0.05,
            outer_offset=0.15,
            source=(
                "examples/2_Intermediate/permanent_magnet_QA.py "
                "(nphi/ntheta/dr non-CI branch; coff; poff; "
                "kwargs_geo['coordinate_flag'])"
            ),
        ),
        builder=_build_qa_grid,
        gpmo=None,
        relax_split=RelaxSplitSpec(
            stages=2,
            reg_l0=0.05,
            nu=1.0e10,
            max_iter=10,
            max_iter_rs=10,
        ),
        solve_by_default=False,
        provenance=(
            "examples/2_Intermediate/permanent_magnet_QA.py "
            "(kwargs from initialize_default_kwargs; the two-stage "
            "relax_and_split loop)",
            "src/simsopt_jax/solve/permanent_magnet.py::relax_and_split_jax",
        ),
    ),
}


# --------------------------------------------------------------------------
# Grid reporting
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryReport:
    """One case's per-lane history accounting: the two byte totals, plus prose.

    The two byte fields are what :func:`describe_grid` publishes as derived
    transients; ``document`` is the full per-lane block.  They come from one
    computation so the artifact cannot carry a footprint that disagrees with
    its own explanation.
    """

    native_buffer_bytes: int
    jax_history_bytes: int
    document: dict[str, object]


def history_report(
    case: ProbeCase,
    iterations: int,
    policy: HistoryPolicy,
    ndipoles: int,
    itemsize: int,
) -> HistoryReport | None:
    """Per-lane history writes and bytes, or ``None`` for a case with no GPMO.

    Nothing here is a single cross-lane number.  The two lanes keep different
    amounts of history under every policy this probe offers, so a count that
    did not name its lane would be a claim of matched work that the
    implementations do not support.  Bytes and writes are also separate facts
    on the native lane: its ``m_history`` is ``xt::zeros``-allocated whether or
    not ``verbose`` lets anything reach it.
    """
    if case.gpmo is None:
        return None
    model = GPMO_HISTORY_MODELS[case.gpmo.algorithm]
    native = native_history_writes(iterations, policy, model)
    plan = jax_history_plan(iterations, policy, model)
    row_bytes = ndipoles * 3 * itemsize
    native_buffer_bytes = native.buffer_slots * row_bytes
    jax_bytes = plan.materialized_history_arrays * plan.recorded_rows * row_bytes
    document: dict[str, object] = {
        "policy": policy.name,
        "rationale": policy.rationale,
        "algorithm": case.gpmo.algorithm,
        "native": {
            "symbol": (
                f"src/simsoptpp/permanent_magnet_optimization.cpp::"
                f"{native.native_symbol}"
            ),
            "verbose": native.verbose,
            "nhistory": native.nhistory,
            "m_history_shape": [ndipoles, 3, native.buffer_slots],
            "buffer_slots": native.buffer_slots,
            "buffer_bytes": native_buffer_bytes,
            "buffer_bytes_formula": (
                "ndipoles * 3 * (nhistory + buffer_slack) * itemsize; the "
                "xt::zeros allocation happens whatever verbose says"
            ),
            "unconditional_preloop_writes": native.unconditional_preloop,
            "verbose_periodic_writes_over_budget": (
                native.verbose_periodic_over_budget
            ),
            "verbose_periodic_writes_guaranteed": (native.verbose_periodic_guaranteed),
            "unconditional_magnet_limit_exit_writes": (
                native.unconditional_magnet_limit_exit
            ),
            "writes_if_budget_exhausted": native.writes_if_budget_exhausted,
            "writes_if_magnet_limit_reached": (native.writes_if_magnet_limit_reached),
            "write_sites": (
                "print_GPMO is the only writer. For "
                f"{native.native_symbol} the call sites outside the "
                "'if (verbose && ((k % int(K/nhistory)) == 0 || k == 0 || "
                "k == K-1))' predicate are: "
                + (
                    "the pre-loop 'Save a record of the magnet array as "
                    "initialized' call and the call inside "
                    "'if ((num_nonzero >= N) || (num_nonzero >= max_nMagnets))' "
                    "which then breaks"
                    if native.unconditional_preloop
                    else (
                        "the call inside 'if ((num_nonzero >= N) || "
                        "(num_nonzero >= max_nMagnets))' which then breaks"
                        if native.unconditional_magnet_limit_exit
                        else "none"
                    )
                )
            ),
        },
        "jax": {
            "symbol": (f"src/simsopt_jax/core/pm_optimization.py::{model.jax_symbol}"),
            "solver_has_retain_history": model.jax_retain_history,
            "record_every": plan.record_every,
            "retain_history": plan.retain_history,
            "recorded_rows": plan.recorded_rows,
            "materialized_history_arrays": plan.materialized_history_arrays,
            "history_bytes": jax_bytes,
            "history_bytes_formula": (
                "2 * recorded_rows * ndipoles * 3 * itemsize -- x_history from "
                "the solver plus the eager m_history = x_history * "
                "m_maxima[None, :, None] that "
                "src/simsopt_jax/solve/permanent_magnet.py::_gpmo_public_result "
                "computes on every call"
            ),
            "full_trace_branch": plan.full_trace_branch,
            "full_trace_bytes_avoided": (
                plan.materialized_history_arrays
                * plan.full_trace_rows_avoided
                * row_bytes
            ),
            "full_trace_note": (
                f"{model.jax_symbol} takes no retain_history parameter, so "
                "record_every=None would select the full-trace scan and "
                "materialize both history arrays at (K, ndipoles, 3); "
                "full_trace_bytes_avoided is what this leg does not pay. "
                "jax_history_plan refuses to build that pair"
                if not model.jax_retain_history
                else (
                    f"{model.jax_symbol} takes retain_history, so "
                    "retain_history=False returns both history arrays with a "
                    "leading axis of length 0 and the full-trace scan is never "
                    "reached"
                )
            ),
        },
        "lane_asymmetry": (
            "Native and JAX history footprints are NOT equal under any policy "
            "here and the probe does not claim they are: compare "
            "native.buffer_bytes with jax.history_bytes above."
        ),
    }
    return HistoryReport(
        native_buffer_bytes=native_buffer_bytes,
        jax_history_bytes=jax_bytes,
        document=document,
    )


def describe_grid(
    case: ProbeCase, build: GridBuild, iterations: int, policy: HistoryPolicy
) -> dict[str, object]:
    """The measured shape of one built grid, plus its would-be device footprint.

    Every ``staged_bytes`` count is derived from a shape that was actually
    built.  ``derived_transient_bytes`` is the other kind: buffers no leg has
    allocated yet, whose sizes are *computed* from the built shapes and the
    configuration.  How many appear depends on the case -- the two lanes'
    history where a GPMO spec exists, plus the GPMO scaled matrix, plus the
    pairwise connectivity where that spec sets ``Nadjacent``, plus the MwPGP
    residual trace where a relax-split spec exists.  Each formula is stated in
    the key itself so a reader can check it against the kernels.

    History is reported per lane, never once, and :func:`history_report`
    carries the full accounting -- the two entries here are the two bottom-line
    byte figures, which are not equal and are not meant to be.
    """
    import numpy as np

    grid = build.grid
    response = np.asarray(grid.A_obj)
    ndipoles = int(grid.ndipoles)
    rows = int(response.shape[0])
    itemsize = int(response.dtype.itemsize)
    staged = {
        "A_obj": response.nbytes,
        "b_obj": int(np.asarray(grid.b_obj).nbytes),
        "ATb": ndipoles * 3 * itemsize,
        "m0_m_mproxy": 3 * ndipoles * 3 * itemsize,
        "m_maxima": ndipoles * itemsize,
        "dipole_grid_xyz": ndipoles * 3 * itemsize,
    }
    if build.polarization_count is not None:
        staged["pol_vectors"] = ndipoles * build.polarization_count * 3 * itemsize
    history = history_report(case, iterations, policy, ndipoles, itemsize)
    derived: dict[str, int] = {}
    if history is not None:
        derived["native_history_buffer_bytes"] = history.native_buffer_bytes
        derived["jax_history_bytes"] = history.jax_history_bytes
    if case.gpmo is not None:
        # ``A_obj * mmax_vec`` is materialized once per solve on both lanes
        # (``simsopt.solve.permanent_magnet_optimization.GPMO``'s ``A_obj``
        # assignment; ``simsopt_jax.solve.permanent_magnet``'s ``A_scaled``
        # assignment in each ``GPMO_*_jax`` wrapper).
        derived["gpmo_scaled_matrix_bytes"] = response.nbytes
        if case.gpmo.adjacent is not None:
            derived["pairwise_connectivity_bytes"] = ndipoles * ndipoles * itemsize
    if case.relax_split is not None:
        derived["mwpgp_residual_history_bytes"] = case.relax_split.max_iter * itemsize
    applied_dr = dr_is_applied(case)
    return {
        "source": case.grid.source,
        "nphi": case.grid.nphi,
        "ntheta": case.grid.ntheta,
        "downsample": case.grid.downsample,
        "coordinate_flag": str(grid.coordinate_flag),
        # Applied only where the constructor consumes it. A famus-grid case
        # publishes null here and carries its declared value under
        # declared_inert_kwargs, because publishing an unapplied number as a
        # grid parameter claims geometry nobody built.
        "dr": case.grid.dr if applied_dr else None,
        "declared_inert_kwargs": (
            {}
            if applied_dr or case.grid.dr is None
            else {"dr": {"value": case.grid.dr, "mechanism": DR_INERT_MECHANISM}}
        ),
        "inner_offset": case.grid.inner_offset,
        "outer_offset": case.grid.outer_offset,
        "rows": rows,
        "ndipoles": ndipoles,
        "polarization_count": build.polarization_count,
        "dtype": str(response.dtype),
        "A_obj_shape": [int(size) for size in response.shape],
        "staged_bytes": {name: int(value) for name, value in staged.items()},
        "staged_bytes_total": int(sum(staged.values())),
        "derived_transient_bytes": {
            name: int(value) for name, value in derived.items()
        },
        "history": None if history is None else history.document,
        "A_obj_sha256": array_sha256(response),
        "b_obj_sha256": array_sha256(np.asarray(grid.b_obj)),
        "build_seconds": build.seconds,
        "input_sha256": build.input_sha256,
    }


# --------------------------------------------------------------------------
# Native lane
# --------------------------------------------------------------------------


def _native_gpmo_kwargs(
    spec: GpmoSpec,
    iterations: int,
    policy: HistoryPolicy,
    dipole_grid_xyz,
) -> dict[str, object]:
    """The kwargs the native GPMO wrapper consumes, built once per solve.

    ``GPMO`` pops from the mapping it is handed, so each solve gets its own.
    """
    import numpy as np

    kwargs: dict[str, object] = {
        "K": iterations,
        "reg_l2": spec.reg_l2,
        "nhistory": policy.native_nhistory,
        "verbose": policy.native_verbose,
    }
    if spec.algorithm == "baseline":
        kwargs["single_direction"] = spec.single_direction
        return kwargs
    if spec.adjacent is None or spec.backtracking is None or spec.max_magnets is None:
        raise ProbeError(f"{spec.algorithm} needs adjacent/backtracking/max_magnets")
    kwargs["Nadjacent"] = spec.adjacent
    kwargs["backtracking"] = spec.backtracking
    kwargs["max_nMagnets"] = spec.max_magnets
    kwargs["dipole_grid_xyz"] = np.ascontiguousarray(dipole_grid_xyz)
    if spec.threshold_angle is not None:
        kwargs["thresh_angle"] = spec.threshold_angle
    return kwargs


def _timed_with_captured_stdout(call: Callable[[], None]) -> tuple[float, str]:
    """Run ``call`` with file descriptor 1 redirected; return seconds and text.

    ``contextlib.redirect_stdout`` rebinds ``sys.stdout`` and nothing else, so
    the compiled kernel's ``printf`` -- which writes straight to descriptor 1
    -- walks past it onto the operator's terminal, *inside* the timed window.
    That costs two things at once: a tty write nobody accounted for in the
    native denominator, and the only output that says which iteration the C++
    loop broke at.  Redirecting the descriptor captures the Python-level and
    the C-level writes together, which fixes both.

    The restore is in a ``finally`` because it is not error handling: a process
    that leaves descriptor 1 pointing at a deleted temporary file has no way to
    report anything ever again, including the exception that put it there.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    with tempfile.TemporaryFile() as sink:
        os.dup2(sink.fileno(), 1)
        try:
            start = time.perf_counter()
            call()
            seconds = time.perf_counter() - start
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
        sink.seek(0)
        transcript = sink.read().decode("utf-8", errors="replace")
    return seconds, transcript


def _last_reported_native_iteration(transcript: str) -> tuple[int | None, int | None]:
    """The last ``Iteration = k, Number of nonzero dipoles = n`` line, if any.

    Both backtracking variants print that line from inside their magnet-limit
    exit, which is *outside* the ``verbose`` predicate
    (``src/simsoptpp/permanent_magnet_optimization.cpp``,
    ``GPMO_backtracking`` and ``GPMO_ArbVec_backtracking``), so a run that
    stopped early names the iteration it stopped at even at
    ``verbose=False``.  The variants with no magnet-limit exit
    (``GPMO_baseline``, ``GPMO_multi``, ``GPMO_ArbVec``) print it only under
    ``verbose``; there the answer is ``(None, None)``, which is a missing
    measurement and is published as ``null`` rather than folded into ``K``.
    """
    marker = "Iteration = "
    separator = ", Number of nonzero dipoles = "
    for line in reversed(transcript.splitlines()):
        head, found, tail = line.partition(separator)
        if found and head.startswith(marker):
            return int(head[len(marker) :]), int(tail)
    return None, None


def run_native_gpmo(
    build: GridBuild,
    spec: GpmoSpec,
    iterations: int,
    policy: HistoryPolicy,
    repeat: int,
) -> SolveResult:
    """Time ``simsopt.solve.GPMO`` ``repeat`` times on an already-built grid.

    The timed window is the public wrapper call, which is what a user pays:
    the ``A_obj * mmax_vec`` scaling, the C++ kernel, the ``m_history``
    rescale loop and the wrapper's own prints
    (``src/simsopt/solve/permanent_magnet_optimization.py``, ``GPMO``).
    Repeats are independent -- no variant reads ``pm_opt.m`` as an initial
    guess unless ``m_init`` is passed, and it is not.

    The wrapper's return value and its transcript are kept, not discarded:
    ``K`` is a *budget*, the backtracking variants ``break`` out of the loop
    when ``num_nonzero >= max_nMagnets``, and the iteration they broke at is
    the only thing that says how much of the budget the native lane actually
    executed.  Both are read from the last repeat, which is the one whose
    endpoint is published.
    """
    import numpy as np
    from simsopt.solve import GPMO

    grid = build.grid
    seconds: list[float] = []
    moments = np.zeros((int(grid.ndipoles), 3), dtype=np.float64)
    history_slots = 0
    objective_entries = 0
    transcript = ""
    for _ in range(repeat):
        kwargs = _native_gpmo_kwargs(spec, iterations, policy, grid.dipole_grid_xyz)
        sink = io.StringIO()
        start = time.perf_counter()
        with redirect_stdout(sink):
            errors, _, m_history = GPMO(grid, spec.algorithm, **kwargs)
        seconds.append(time.perf_counter() - start)
        transcript = sink.getvalue()
        history_slots = int(np.asarray(m_history).shape[-1])
        objective_entries = int(np.asarray(errors).size)
        moments = np.array(grid.m, dtype=np.float64, copy=True).reshape(
            (int(grid.ndipoles), 3)
        )
    last_iteration, last_nonzero = _last_reported_native_iteration(transcript)
    can_exit_early = GPMO_HISTORY_MODELS[spec.algorithm].native_magnet_limit_exit_write
    if last_iteration is not None:
        executed: int | None = last_iteration + 1
    elif can_exit_early:
        # The variant can break and did not say where; a number here would be
        # a guess, and ``K`` in particular would be the wrong one.
        executed = None
    else:
        executed = iterations
    return SolveResult(
        moments=moments,
        seconds=tuple(seconds),
        device_memory_mib={},
        iteration_report={
            "lane": NATIVE_LANE,
            "budget_K": iterations,
            "can_exit_early": can_exit_early,
            "executed_iterations": executed,
            "max_nMagnets": spec.max_magnets,
            "last_reported_iteration": last_iteration,
            "last_reported_nonzero_dipoles": last_nonzero,
            "m_history_slots_returned": history_slots,
            "nonzero_objective_history_entries": objective_entries,
            "source": (
                "src/simsopt/solve/permanent_magnet_optimization.py::GPMO "
                "returns (errors, Bn_errors, m_history); the 'Iteration = k, "
                "Number of nonzero dipoles = n' line is printed by the "
                "magnet-limit exit in "
                "src/simsoptpp/permanent_magnet_optimization.cpp::"
                "GPMO_ArbVec_backtracking, outside its verbose predicate"
            ),
            "semantics": (
                "The C++ loop breaks at 'if ((num_nonzero >= N) || "
                "(num_nonzero >= max_nMagnets))', so a backtracking variant "
                "executes last_reported_iteration + 1 of the K iterations, not "
                "K. errors = algorithm_history[algorithm_history != 0] counts "
                "history slots that received a nonzero objective, which is a "
                "lower bound on print_GPMO calls, not the iteration count."
            ),
        },
    )


def run_native_relax_split(
    build: GridBuild, spec: RelaxSplitSpec, repeat: int
) -> SolveResult:
    """Time the native relax-and-split continuation exactly as QA runs it.

    ``verbose`` stays true because the wrapper indexes the recorded history
    (``src/simsopt/solve/permanent_magnet_optimization.py``,
    ``relax_and_split``): a silent
    native MwPGP would raise, not run faster.
    """
    import numpy as np
    from simsopt.solve import relax_and_split
    from simsopt.util import initialize_default_kwargs

    grid = build.grid
    ndipoles = int(grid.ndipoles)
    # ``rescale_for_opt`` adds 1/nu into ``ATA_scale`` in place
    # (``simsopt.geo.permanent_magnet_grid.PermanentMagnetGrid.rescale_for_opt``),
    # so it is called exactly
    # once per process -- calling it per repeat would shrink the step size.
    reg_l0, _, _, nu = grid.rescale_for_opt(spec.reg_l0, 0.0, 0.0, spec.nu)
    seconds: list[float] = []
    moments = np.zeros((ndipoles, 3), dtype=np.float64)
    for _ in range(repeat):
        initial = np.zeros(ndipoles * 3)
        sink = io.StringIO()
        start = time.perf_counter()
        with redirect_stdout(sink):
            for stage in range(spec.stages):
                kwargs = initialize_default_kwargs()
                kwargs["nu"] = nu
                kwargs["max_iter"] = spec.max_iter
                kwargs["max_iter_RS"] = spec.max_iter_rs
                kwargs["reg_l0"] = reg_l0 * (stage + 1) / spec.stages
                relax_and_split(grid, m0=initial, **kwargs)
                initial = grid.m
        seconds.append(time.perf_counter() - start)
        moments = np.array(grid.m, dtype=np.float64, copy=True).reshape((ndipoles, 3))
    return SolveResult(
        moments=moments,
        seconds=tuple(seconds),
        device_memory_mib={},
        iteration_report={},
    )


# --------------------------------------------------------------------------
# JAX lane
# --------------------------------------------------------------------------


def _stage_jax_grid(build: GridBuild):
    """Move the host grid onto the device and wait for the transfer to land.

    Staging is deliberately outside every timed window: both lanes are handed
    the same host grid, and the probe times solves, not ``device_put``.
    """
    import jax
    from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX

    staged = PermanentMagnetGridJAX.from_cpu(build.grid)
    jax.block_until_ready(
        (
            staged.A_obj,
            staged.b_obj,
            staged.ATb,
            staged.m0,
            staged.m_maxima,
            staged.dipole_grid_xyz,
        )
    )
    return staged


def run_jax_gpmo(
    build: GridBuild,
    spec: GpmoSpec,
    iterations: int,
    policy: HistoryPolicy,
    repeat: int,
) -> SolveResult:
    """Time the JAX GPMO variant; call 0 is cold-in-process, the rest warm.

    Device memory is read three times -- before staging, after staging, after
    the last solve -- because the memory rung's question is where the footprint
    lands, and one reading at the end cannot separate the staged grid from the
    solve's transients.  The readings are whole-device, so they are only as
    clean as the identity block's ``gpu_compute_processes`` says the box was.

    The history levers come from :func:`jax_history_plan`, which is
    variant-aware: ``GPMO_baseline_jax`` takes ``retain_history`` and
    ``GPMO_ArbVec_backtracking_jax`` does not, so "record nothing" means two
    different call shapes and ``record_every=None`` means two different things.
    """
    import jax
    import numpy as np
    from simsopt_jax.solve.permanent_magnet import (
        GPMO_ArbVec_backtracking_jax,
        GPMO_baseline_jax,
    )

    plan = jax_history_plan(iterations, policy, GPMO_HISTORY_MODELS[spec.algorithm])
    before_staging = device_memory_used_mib()
    staged = _stage_jax_grid(build)
    after_staging = device_memory_used_mib()
    record_every = plan.record_every
    if spec.algorithm == "baseline":
        retain_history = plan.retain_history

        def solve():
            return GPMO_baseline_jax(
                staged,
                K=iterations,
                reg_l2=spec.reg_l2,
                single_direction=spec.single_direction,
                record_every=record_every,
                retain_history=retain_history,
            )

    elif spec.algorithm == "ArbVec_backtracking":
        if (
            spec.adjacent is None
            or spec.backtracking is None
            or spec.max_magnets is None
        ):
            raise ProbeError(
                "ArbVec_backtracking needs adjacent/backtracking/max_magnets"
            )
        adjacent = spec.adjacent
        backtracking = spec.backtracking
        max_magnets = spec.max_magnets
        threshold = spec.threshold_angle if spec.threshold_angle is not None else PI

        def solve():
            return GPMO_ArbVec_backtracking_jax(
                staged,
                K=iterations,
                reg_l2=spec.reg_l2,
                Nadjacent=adjacent,
                backtracking=backtracking,
                thresh_angle=threshold,
                max_nMagnets=max_magnets,
                record_every=record_every,
            )

    else:
        raise ProbeError(f"unsupported GPMO variant {spec.algorithm!r}")

    seconds: list[float] = []
    moments = np.zeros((int(build.grid.ndipoles), 3), dtype=np.float64)
    for _ in range(repeat):
        start = time.perf_counter()
        result = solve()
        jax.block_until_ready((result.m, result.residual))
        seconds.append(time.perf_counter() - start)
        moments = np.asarray(jax.device_get(result.m), dtype=np.float64)
    return SolveResult(
        moments=moments,
        seconds=tuple(seconds),
        device_memory_mib={
            "before_staging": before_staging,
            "after_staging": after_staging,
            "after_solve": device_memory_used_mib(),
        },
        iteration_report={
            "lane": JAX_LANE,
            "budget_K": iterations,
            "can_exit_early": False,
            "executed_iterations": iterations,
            "max_nMagnets": spec.max_magnets,
            "source": (
                "src/simsopt_jax/core/pm_optimization.py::"
                "gpmo_arbvec_backtracking_solve -- the scan length is fixed by "
                "K and jax.lax.scan has no early exit"
            ),
            "semantics": (
                "The JAX lane executes all K scan iterations. Once the C++ "
                "stopping condition would have fired, done is monotone and "
                "gpmo_arbvec_backtracking_step's lax.cond dispatches "
                "_gpmo_arbvec_backtracking_frozen_step, which skips the "
                "candidate scan and the dewyrming pass -- cheap, but not free, "
                "and definitely not skipped."
            ),
        },
    )


def run_jax_relax_split(
    build: GridBuild, spec: RelaxSplitSpec, repeat: int
) -> SolveResult:
    """Time ``relax_and_split_jax`` on the same continuation the native lane runs.

    Staging happens BEFORE ``rescale_for_opt``, and the ordering is the
    contract (double-shift defect adjudicated 2026-08-23, backlog plan P3.5;
    fixed 2026-08-24): ``rescale_for_opt`` folds ``2 reg_l2 + 1/nu`` into
    ``grid.ATA_scale`` **in place**, while ``PermanentMagnetGridJAX.from_cpu``'s
    ``_mwpgp_spec`` validator takes ``ATA_scale`` as the *raw* spectral scale
    and applies that same shift itself. Staging the already-shifted grid made
    the validator re-shift and reject a legitimate step (the refusal is
    archived: ``docs/receipts/evidence/qa64_jaxgpu_solve_refusal_20260824.log``).
    Staged raw, the explicit ``alpha`` below — still computed from the
    post-rescale host scale, exactly as the native lane's step rule derives
    it — sits inside the validator's bound by its own 1e-5 margin and equals
    the solver's default-step formula on the same operands. Staging is a
    device copy, so the later in-place host rescale cannot reach the staged
    arrays.

    Every row this returns is cold.  See :data:`JAX_RELAX_SPLIT_RETRACE`: the
    solve path is unjitted and rebuilds its scan closure per call, so repeat 1
    onward re-trace rather than reusing an executable, and they are labelled
    ``repeat_retrace``, never ``warm``.
    """
    import jax
    import numpy as np
    from simsopt_jax.solve.permanent_magnet import relax_and_split_jax

    grid = build.grid
    ndipoles = int(grid.ndipoles)
    before_staging = device_memory_used_mib()
    staged = _stage_jax_grid(build)
    after_staging = device_memory_used_mib()
    reg_l0, _, _, nu = grid.rescale_for_opt(spec.reg_l0, 0.0, 0.0, spec.nu)
    alpha = 2.0 * (1.0 - 1.0e-5) / float(grid.ATA_scale)

    seconds: list[float] = []
    moments = np.zeros((ndipoles, 3), dtype=np.float64)
    for _ in range(repeat):
        initial = np.zeros(ndipoles * 3)
        start = time.perf_counter()
        for stage in range(spec.stages):
            result = relax_and_split_jax(
                staged,
                initial,
                alpha=alpha,
                max_iter=spec.max_iter,
                max_iter_RS=spec.max_iter_rs,
                nu=nu,
                reg_l0=reg_l0 * (stage + 1) / spec.stages,
            )
            jax.block_until_ready((result.m, result.m_proxy))
            initial = result.m
        seconds.append(time.perf_counter() - start)
        moments = np.asarray(jax.device_get(initial), dtype=np.float64).reshape(
            (ndipoles, 3)
        )
    return SolveResult(
        moments=moments,
        seconds=tuple(seconds),
        device_memory_mib={
            "before_staging": before_staging,
            "after_staging": after_staging,
            "after_solve": device_memory_used_mib(),
        },
        iteration_report={},
    )


# --------------------------------------------------------------------------
# Moment comparison
# --------------------------------------------------------------------------


def _validate_moments_path(path: Path) -> Path:
    """Refuse a moments endpoint that is misnamed or would clobber an earlier one.

    Both refusals are about evidence, not tidiness.  A suffix outside
    :data:`MOMENTS_SUFFIXES` is usually a mistyped flag and would leave the
    archive somewhere the comparison step will not look for it; an existing
    file is a previous leg's endpoint, and silently overwriting it destroys
    exactly the half of a cross-lane comparison that already ran.  The write
    itself opens ``"xb"``, so the refusal is also enforced by the kernel and
    not only by this check -- two legs racing the same path cannot both win.
    """
    if path.suffix not in MOMENTS_SUFFIXES:
        raise ProbeError(
            f"--moments-out must end in one of {list(MOMENTS_SUFFIXES)}; "
            f"got {path.name!r}"
        )
    if path.exists():
        raise ProbeError(
            f"--moments-out refuses to overwrite an existing endpoint: {path}"
        )
    return path


def write_moments(path: Path, moments, metadata: dict[str, object]) -> None:
    """Save one leg's endpoint as an ``npz`` archive carrying its own metadata.

    A bare ``.npy`` array cannot say which case, scale, budget, history policy
    or lane produced it, so two files that disagree about all five compare
    cleanly and report a difference nobody can attribute.  The metadata rides
    in the archive as canonical JSON, mirroring
    ``benchmarks/stochastic_stage_two_probe.py``'s ``write_endpoint``.

    ``np.savez`` is handed an open file object rather than a path: given a path
    it appends ``.npz`` itself, which would turn an operator's ``foo.npy`` into
    ``foo.npy.npz`` and lose the file the comparison step was told to read.
    The ``"xb"`` mode is the overwrite refusal.
    """
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "xb") as handle:
        np.savez(
            handle,
            moments=moments,
            metadata=np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        )


def _load_moments(path: Path) -> tuple[object, dict[str, object]]:
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["moments"], dtype=np.float64),
            json.loads(str(archive["metadata"].item())),
        )


def compare_moments(left_path: Path, right_path: Path) -> dict[str, object]:
    """Bitwise verdict, max ULP distance and max absolute difference.

    Gated first on the metadata the two archives carry: two endpoints that
    solved different cases, scales, budgets or history policies -- or were
    handed different host grids, which the two grid digests catch -- are not
    comparable, and a difference between them measures the configuration, not
    the lanes.  ``lane`` is excluded from the gate on purpose: a cross-lane
    comparison is the case where it must differ.

    Every published number is finite by construction.  ``None`` is what an
    all-nonfinite pair reports, never ``nan``, so the verdict survives being
    written down.
    """
    import numpy as np

    left, left_metadata = _load_moments(left_path)
    right, right_metadata = _load_moments(right_path)
    mismatched = {
        name: (left_metadata[name], right_metadata[name])
        for name in MOMENTS_IDENTITY_FIELDS
        if left_metadata[name] != right_metadata[name]
    }
    if mismatched:
        raise ProbeError(
            "moment endpoints are not comparable; these matched-configuration "
            f"fields differ (left vs right): {mismatched}"
        )
    if left.shape != right.shape:
        raise ProbeError(f"shape mismatch: {left.shape} vs {right.shape}")
    if left.dtype != np.float64 or right.dtype != np.float64:
        raise ProbeError(f"moments must be float64; got {left.dtype}/{right.dtype}")
    left_digest = array_sha256(left)
    right_digest = array_sha256(right)
    finite = np.isfinite(left) & np.isfinite(right)
    any_finite = bool(finite.any())
    difference = np.abs(left - right)
    # ``benchmarks/probe_conventions.py``'s ``ulp_distance`` -- uint64-exact,
    # shared with the other probes, so two families cannot report ULP
    # distances measured by two slightly different kernels.
    ulp = ulp_distance(left, right)
    return {
        "left": str(left_path),
        "right": str(right_path),
        "lane_left": left_metadata["lane"],
        "lane_right": right_metadata["lane"],
        "matched_fields": {
            name: left_metadata[name] for name in MOMENTS_IDENTITY_FIELDS
        },
        "shape": [int(size) for size in left.shape],
        "left_sha256": left_digest,
        "right_sha256": right_digest,
        "bitwise_identical": left_digest == right_digest,
        "comparable_elements": int(finite.sum()),
        "max_ulp": int(ulp[finite].max()) if any_finite else None,
        "max_absolute_difference": (
            float(difference[finite].max()) if any_finite else None
        ),
        "nonfinite_elements": int(left.size - int(finite.sum())),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pm_gpmo_probes.py",
        description=(
            "Diagnostic permanent-magnet native-vs-JAX probes "
            f"({PROBE_GRADE}); see the module docstring for interpreters."
        ),
    )
    subparsers = parser.add_subparsers(dest="case", required=True)
    for case in CASES.values():
        child = subparsers.add_parser(case.name, help=case.summary)
        child.add_argument("--lane", choices=LANES, default=NATIVE_LANE)
        child.add_argument(
            "--omp",
            type=int,
            default=None,
            help=(
                "OMP_NUM_THREADS for this leg. Mandatory for a timed run: the "
                f"fair-native denominator is swept over {list(OMP_SWEEP)}, "
                "never defaulted."
            ),
        )
        child.add_argument(
            "--repeat", type=int, default=2, help="timed solves in this leg"
        )
        child.add_argument(
            "--k-override",
            type=int,
            default=None,
            help="replace the case iteration budget (small-K smoke / memory rung)",
        )
        child.add_argument("--output", type=Path, default=None)
        child.add_argument(
            "--moments-out",
            type=Path,
            default=None,
            metavar="ENDPOINT.npz",
            help=(
                "save this leg's moments plus their configuration metadata; "
                "refuses a foreign suffix and refuses to overwrite"
            ),
        )
        child.add_argument(
            "--compare",
            nargs=2,
            type=Path,
            default=None,
            metavar=("A.npz", "B.npz"),
            help=(
                "compare two saved moment archives and exit; runs no solve. "
                "Refuses two archives whose configuration metadata disagrees."
            ),
        )
        child.add_argument(
            "--ledger",
            type=Path,
            default=DEFAULT_LEDGER,
            help=(
                "append-only JSONL recording every executed leg, in the order "
                "it ran. Executed order is evidence; a schedule is not."
            ),
        )
        child.add_argument("--dry-run", action="store_true")
        child.add_argument(
            "--history",
            choices=sorted(HISTORY_POLICIES),
            default="off",
            help=(
                "snapshot bookkeeping, resolved per GPMO variant (default: "
                "off = the least history that variant can be made to keep, "
                "which is NOT an equal count across lanes)"
            ),
        )
        child.add_argument("--backend-mode", default="jax_gpu_fast")
        child.add_argument(
            "--preallocate-off",
            action="store_true",
            help=f"set {PREALLOCATE_VARIABLE}=false before JAX initializes",
        )
        child.add_argument(
            "--cache-dir",
            type=Path,
            default=None,
            help="persistent JAX compilation cache (absent = cold lane)",
        )
        if case.relax_split is not None:
            child.add_argument(
                "--solve",
                action="store_true",
                help="run the timed relax-and-split legs instead of reporting the grid only",
            )
    return parser


def _default_output(case: ProbeCase, lane: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = case.name.replace("-", "_")
    return (
        EVIDENCE_DIRECTORY
        / f"pm_gpmo_probe_{slug}_{lane.replace('-', '_')}_{stamp}.json"
    )


def _validate_output(path: Path) -> Path:
    """Refuse an in-repo artifact outside the evidence root; allow a scratch run.

    The rule ``benchmarks/marginal_quartet_probes.py`` (``_validate``) already
    applies: a path *inside* the repository must live under the evidence root,
    because an artifact loose in the tree gets read as evidence sooner or
    later; a path entirely outside the repository is a scratch run, is
    accepted, and is stamped ``published_under_evidence_root: false`` in its
    own configuration block so nobody has to infer it from the filename.
    """
    resolved = path.resolve()
    if (
        EVIDENCE_DIRECTORY.resolve() not in resolved.parents
        and REPO_ROOT.resolve() in resolved.parents
    ):
        raise ProbeError(
            f"--output inside the repository must live under "
            f"{EVIDENCE_DIRECTORY}; got {resolved}. A path outside the "
            "repository is accepted and is recorded as "
            "published_under_evidence_root: false"
        )
    return resolved


def _published_under_evidence_root(path: Path) -> bool:
    return EVIDENCE_DIRECTORY.resolve() in path.resolve().parents


def _applied_environment(environment: dict[str, str]) -> dict[str, str]:
    """The pins, out of a whole leg environment, in publishable form."""
    return {
        name: environment[name]
        for name in APPLIED_ENVIRONMENT_NAMES
        if name in environment
    }


def _child_argv(arguments: argparse.Namespace) -> list[str]:
    """Rebuild this invocation's argv from the *parsed* arguments.

    Not ``sys.argv``.  ``main([...])`` is a supported entry point -- the tests
    and any programmatic driver use it -- and in that call ``sys.argv`` still
    holds whatever launched the interpreter, so re-executing it would relaunch
    something nobody validated (in the worst case a different subcommand
    entirely, or pytest).  Rebuilding from the namespace guarantees the child
    runs exactly the configuration the parser accepted, and only that: flags
    the operator did not pass stay unpassed, so the child re-derives the same
    defaults rather than being handed a parent's resolved value as if it had
    been requested.
    """
    argv = [
        str(Path(__file__).resolve()),
        arguments.case,
        "--lane",
        arguments.lane,
        "--repeat",
        str(arguments.repeat),
        "--history",
        arguments.history,
        "--backend-mode",
        str(arguments.backend_mode),
        "--ledger",
        str(arguments.ledger),
    ]
    if arguments.omp is not None:
        argv += ["--omp", str(arguments.omp)]
    if arguments.k_override is not None:
        argv += ["--k-override", str(arguments.k_override)]
    if arguments.output is not None:
        argv += ["--output", str(arguments.output)]
    if arguments.moments_out is not None:
        argv += ["--moments-out", str(arguments.moments_out)]
    if arguments.cache_dir is not None:
        argv += ["--cache-dir", str(arguments.cache_dir)]
    if arguments.preallocate_off:
        argv.append("--preallocate-off")
    if getattr(arguments, "solve", False):
        argv.append("--solve")
    return argv


def _reexec_native_child(arguments: argparse.Namespace) -> None:
    """Replace this process with one whose OpenMP thread count is already pinned.

    libgomp reads ``OMP_NUM_THREADS`` when it initializes, which happens the
    first time the compiled extension is loaded.  Mutating ``os.environ`` in a
    process that may already have touched it is not a pin; re-execution is.

    The child's environment is built by :func:`pinned_environment`, not copied
    from this one.  A verbatim copy carries every selector the operator's shell
    happened to export -- ``KMP_AFFINITY`` re-pinning the threads underneath
    the ``OMP_NUM_THREADS`` this function just set, ``SIMSOPT_BACKEND_MODE``
    rerouting the "native" leg onto the GPU backend,
    ``JAX_COMPILATION_CACHE_DIR`` making a cold leg warm -- and a native
    denominator measured under an unknown pin is not a denominator.
    ``JAX_PLATFORMS=cpu`` is passed because ``simsopt.geo`` objectives are
    jax-jitted: without it the native leg's transitive JAX import can take a
    CUDA context on the card a GPU leg is timed on.
    """
    child_environment = pinned_environment(
        lane=NATIVE_LANE, omp=arguments.omp, jax_platforms="cpu"
    )
    child_environment[CHILD_MARKER] = "1"
    os.execve(
        sys.executable, [sys.executable, *_child_argv(arguments)], child_environment
    )


def verify_native_thread_pin(requested_omp: int) -> dict[str, object]:
    """Prove the native leg is running at the thread count it was pinned to.

    Called *after* the heavy imports, because that is when the pin becomes
    checkable: libgomp initializes with the compiled extension, and until it is
    loaded ``omp_get_max_threads`` is a prediction rather than a readback.
    ``OMP_NUM_THREADS`` is only a request -- ``OMP_THREAD_LIMIT``, a library
    calling ``omp_set_num_threads``, or a cgroup CPU quota can all silently
    reduce it -- and a fair-native denominator quoted at the wrong thread count
    is a wrong denominator, so a mismatch fails the leg instead of being
    recorded next to a number nobody can use.

    A ``None`` readback means libgomp could not be loaded at all.  That is a
    host fact, not a thread count, and it is published as ``null`` rather than
    folded into a number: "no OpenMP here" and "one thread" are different
    measurements.  It is not a mismatch, so it does not fail the leg; the
    artifact says the pin was unverifiable and a reader can act on that.
    """
    observed = observed_openmp_threads()
    if observed is not None and observed != requested_omp:
        raise ProbeError(
            f"native leg requested OMP_NUM_THREADS={requested_omp} but libgomp "
            f"reports omp_get_max_threads()={observed}; the fair-native "
            "denominator would be quoted at a thread count the run never used"
        )
    return {
        "requested_omp": requested_omp,
        "observed_openmp_threads": observed,
        "readback_available": observed is not None,
        "environment_echo": {
            name: os.environ.get(name) for name in THREAD_COUNT_VARIABLES
        },
    }


def _prepare_jax_environment(arguments: argparse.Namespace) -> dict[str, str]:
    """Scrub, pin, and resolve every variable JAX reads only at initialization.

    Applied to *every* ``jax-gpu`` invocation, timed or not.  Gating this on
    ``--omp`` (which only a timed leg must supply) made ``--preallocate-off``
    and ``--cache-dir`` silently inert on the untimed memory rung while the
    artifact went on reporting them as applied -- a flag that is published but
    never reaches the runtime is worse than a missing flag.  ``--omp`` stays
    mandatory for a timed leg only; ``omp=None`` here is
    :func:`pinned_environment`'s documented shipped-default lane, where the
    inherited threading configuration is still scrubbed but nothing is pinned
    in its place.

    The scrub is real because ``os.environ`` is *replaced*, not updated: every
    name outside the pinned environment is deleted first, so an inherited
    ``JAX_COMPILATION_CACHE_DIR`` cannot quietly make a cold leg warm and an
    inherited ``CUDA_VISIBLE_DEVICES`` cannot move the leg to another card.
    The two probe-owned selectors are re-added afterwards precisely because the
    scrub would otherwise remove them: they live under the ``SIMSOPT_`` and
    ``XLA_`` prefixes it deletes.

    The import guard covers numpy and simsoptpp as well as jax.  All three read
    thread and device settings once, at load: numpy's BLAS opens its pool when
    the extension loads, simsoptpp brings libgomp with it, and jax resolves its
    platform on first import.  A pin applied after any of them is decoration.
    """
    already_imported = [
        name for name in ("jax", "numpy", "simsoptpp") if name in sys.modules
    ]
    if already_imported:
        raise ProbeError(
            f"{already_imported} were imported before the leg environment was "
            f"resolved; the thread pins, {PREALLOCATE_VARIABLE} and "
            "backend-mode settings would be ignored by an already-loaded runtime"
        )
    environment = pinned_environment(
        lane=JAX_LANE, omp=arguments.omp, compile_cache_dir=arguments.cache_dir
    )
    environment[BACKEND_MODE_VARIABLE] = str(arguments.backend_mode)
    if arguments.cache_dir is not None:
        # Pinned only alongside the cache, and only to 0: at JAX's shipped
        # thresholds a genuinely cold leg can compile fast enough or small
        # enough to persist nothing, and the next leg is then cold while
        # reporting a warm cache. Same values as the wireframe probe's
        # PERSISTENT_CACHE_THRESHOLDS, so the two families' warm lanes mean the
        # same thing.
        environment.update(PERSISTENT_CACHE_THRESHOLDS)
    if arguments.preallocate_off:
        environment[PREALLOCATE_VARIABLE] = "false"
    for name in [name for name in os.environ if name not in environment]:
        del os.environ[name]
    os.environ.update(environment)

    from simsopt_jax.config import apply_jax_runtime_config

    apply_jax_runtime_config()
    import jax

    if not bool(jax.config.read("jax_enable_x64")):
        raise ProbeError(
            f"{BACKEND_MODE_VARIABLE}={arguments.backend_mode} left x64 disabled; "
            "a permanent-magnet probe in float32 is not a measurement"
        )
    return _applied_environment(environment)


def jax_relax_split_retraces(case: ProbeCase, lane: str, solving: bool) -> bool:
    """Whether this leg's repeats re-trace instead of reusing an executable.

    True exactly for a solving JAX relax-and-split leg; see
    :data:`JAX_RELAX_SPLIT_RETRACE` for the mechanism.  The native lane has no
    executable to reuse in the first place, so the question does not arise
    there, and the GPMO wrappers are jitted.
    """
    return case.relax_split is not None and lane == JAX_LANE and solving


def _print_dry_run(
    case: ProbeCase,
    build: GridBuild,
    grid_report: dict[str, object],
    iterations: int,
    policy: HistoryPolicy,
    arguments: argparse.Namespace,
) -> None:
    solving = case.solve_by_default or bool(getattr(arguments, "solve", False))
    print(f"case                {case.name}: {case.summary}")
    print(f"grid source         {case.grid.source}")
    print(
        f"grid                nphi={case.grid.nphi} ntheta={case.grid.ntheta} "
        f"downsample={case.grid.downsample} coordinate_flag={grid_report['coordinate_flag']}"
    )
    print(f"rows                {grid_report['rows']}")
    print(f"ndipoles            {grid_report['ndipoles']}")
    print(f"pol vectors         {grid_report['polarization_count']}")
    print(f"A_obj               {grid_report['A_obj_shape']} {grid_report['dtype']}")
    print(f"staged bytes        {grid_report['staged_bytes_total']}")
    print(f"  per array         {grid_report['staged_bytes']}")
    print(f"derived transients  {grid_report['derived_transient_bytes']}")
    print(f"grid build seconds  {build.seconds:.3f}")
    if case.gpmo is not None:
        print(
            f"solver              GPMO {case.gpmo.algorithm} K={iterations}"
            f" reg_l2={case.gpmo.reg_l2}"
            + (
                ""
                if case.gpmo.adjacent is None
                else (
                    f" Nadjacent={case.gpmo.adjacent}"
                    f" backtracking={case.gpmo.backtracking}"
                    f" max_nMagnets={case.gpmo.max_magnets}"
                    f" thresh_angle={case.gpmo.threshold_angle}"
                )
            )
        )
    if case.relax_split is not None:
        print(
            f"solver              relax-and-split stages={case.relax_split.stages}"
            f" max_iter={case.relax_split.max_iter}"
            f" max_iter_RS={case.relax_split.max_iter_rs}"
            f" reg_l0={case.relax_split.reg_l0} nu={case.relax_split.nu}"
            f"  (armed={solving})"
        )
    if case.grid.dr is not None and not dr_is_applied(case):
        print(f"declared inert      dr={case.grid.dr}: {DR_INERT_MECHANISM}")
    history = grid_report["history"]
    if isinstance(history, dict):
        native = history["native"]
        jax_side = history["jax"]
        if isinstance(native, dict) and isinstance(jax_side, dict):
            print(f"history policy      {policy.name} ({history['algorithm']})")
            print(
                f"history native      {native['symbol'].rsplit('::', 1)[-1]}: "
                f"verbose={native['verbose']} nhistory={native['nhistory']} -> "
                f"{native['unconditional_preloop_writes']} unconditional "
                f"pre-loop + {native['verbose_periodic_writes_over_budget']} "
                f"verbose-periodic + "
                f"{native['unconditional_magnet_limit_exit_writes']} "
                f"magnet-limit-exit; totals "
                f"{native['writes_if_budget_exhausted']} (budget exhausted) / "
                f"{native['writes_if_magnet_limit_reached']} (magnet limit) "
                f"into {native['buffer_slots']} slots = "
                f"{native['buffer_bytes']} bytes allocated"
            )
            print(
                f"history jax         {jax_side['symbol'].rsplit('::', 1)[-1]}: "
                f"record_every={jax_side['record_every']} "
                f"retain_history={jax_side['retain_history']} -> "
                f"{jax_side['recorded_rows']} row(s) x "
                f"{jax_side['materialized_history_arrays']} arrays = "
                f"{jax_side['history_bytes']} bytes; full_trace_branch="
                f"{jax_side['full_trace_branch']} "
                f"(avoided {jax_side['full_trace_bytes_avoided']} bytes)"
            )
    if case.gpmo is not None:
        # Only narrated where it binds: a relax-and-split case runs no GPMO
        # variant, so the history policy applies to nothing there and saying
        # otherwise would read as a configuration that was applied.
        print(f"history rationale   {policy.rationale}")
        print(f"history override    {SHIPPED_NHISTORY_NOTE}")
    # A suggested alternation for the operator, printed and never published:
    # nothing here observed it being followed. What ran is the ledger's.
    print(
        f"interleave (plan)   {interleave_schedule(3, NATIVE_LANE, JAX_LANE)}"
        "  -- suggestion only; executed order is the ledger's"
    )
    print(f"ledger              {arguments.ledger}")
    print(
        "cold/warm           solve 0 is cold_in_process on BOTH lanes and is "
        "excluded from warm_seconds on both"
    )
    if jax_relax_split_retraces(case, arguments.lane, solving):
        print(f"retrace             {JAX_RELAX_SPLIT_RETRACE}")
    if case.gpmo is not None:
        print(f"iteration semantics {MATCHED_WORK_ITERATION_NOTE}")
    print(f"grade               {PROBE_GRADE}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    case = CASES[arguments.case]
    policy = HISTORY_POLICIES[arguments.history]

    if arguments.compare is not None:
        report = compare_moments(arguments.compare[0], arguments.compare[1])
        for key, value in report.items():
            print(f"{key:26s} {value}")
        return 0

    iterations = arguments.k_override
    if iterations is None and case.gpmo is not None:
        iterations = case.gpmo.iterations
    if iterations is None:
        iterations = case.relax_split.max_iter if case.relax_split else 1
    if iterations < 1:
        raise ProbeError(f"iteration budget must be positive; got {iterations}")

    solving = case.solve_by_default or bool(getattr(arguments, "solve", False))
    timed = solving and not arguments.dry_run

    if timed and arguments.omp is None:
        raise ProbeError(
            "a timed leg needs an explicit --omp; the fair-native denominator "
            f"is the swept optimum over {list(OMP_SWEEP)}, never a default"
        )
    if timed and arguments.repeat < 2:
        # Symmetric with the cold/warm convention below: solve 0 is
        # cold-in-process on BOTH lanes, so a leg that only runs solve 0
        # publishes an empty warm list on either lane.  The native first call
        # pays first-touch faults and the OpenMP team's first fork; the JAX one
        # pays compilation.  Different costs, same accounting.
        raise ProbeError(
            f"a timed leg needs --repeat >= 2 on both lanes; got {arguments.repeat}. "
            "Solve 0 is cold-in-process and the warm number is solve 1 onward"
        )
    retracing = jax_relax_split_retraces(case, arguments.lane, solving)
    if retracing and timed and arguments.repeat > 1 and arguments.cache_dir is None:
        raise ProbeError(
            f"{case.name}: --cache-dir is required with --repeat > 1 on the "
            f"{JAX_LANE} solve path. {JAX_RELAX_SPLIT_RETRACE}"
        )
    # Pre-flight, before the re-exec and before any solve: an endpoint path
    # that is misnamed or already taken must fail while the only thing lost is
    # a parse, not after a full solve has been paid for and has nowhere to land.
    moments_out = (
        None
        if arguments.moments_out is None
        else _validate_moments_path(arguments.moments_out)
    )
    # Same reason, for the artifact path: resolved and refused up front rather
    # than after the timed window.
    output = _validate_output(
        arguments.output
        if arguments.output is not None
        else _default_output(case, arguments.lane)
    )

    if arguments.lane == NATIVE_LANE and timed and os.environ.get(CHILD_MARKER) != "1":
        _reexec_native_child(arguments)

    applied_environment: dict[str, str] = {}
    if arguments.lane == JAX_LANE:
        applied_environment = _prepare_jax_environment(arguments)
    elif os.environ.get(CHILD_MARKER) == "1":
        applied_environment = _applied_environment(dict(os.environ))

    build = case.builder(case.grid)
    grid_report = describe_grid(case, build, iterations, policy)

    if arguments.dry_run:
        _print_dry_run(case, build, grid_report, iterations, policy, arguments)
        return 0

    identity = runtime_identity(arguments.lane)
    notes = [
        (
            "Timed window is the solve only; grid construction, device staging "
            "and moment retrieval are outside it on both lanes."
        ),
        (
            "Native timed window includes the GPMO wrapper's A_obj*mmax "
            "scaling, m_history rescale loop and prints "
            "(src/simsopt/solve/permanent_magnet_optimization.py::GPMO)."
        ),
        (
            "UNDISCLOSED-NO-LONGER: the same native timed window also builds "
            "two host copies with no JAX counterpart -- contig(A_obj.T), a full "
            "transposed copy of the scaled response matrix, and the Nnorms "
            "ravel of the boundary normals "
            "(src/simsopt/solve/permanent_magnet_optimization.py::GPMO). The "
            "JAX lane stages A_obj once, outside every timed window. This runs "
            "in the anti-GPU direction -- it inflates the native denominator -- "
            "and is disclosed, NOT subtracted: the probe times the wrapper a "
            "user actually calls."
        ),
        (
            f"{SHIPPED_NHISTORY_NOTE} This probe pins nhistory="
            f"{policy.native_nhistory} with verbose={policy.native_verbose}."
        ),
        f"History policy: {policy.rationale}",
        (
            "History is NOT equalized across lanes and this probe does not "
            "claim it is -- for the ArbVec_backtracking variants no setting "
            "can be, because gpmo_arbvec_backtracking_solve has no "
            "retain_history parameter and GPMO_ArbVec_backtracking writes "
            "twice outside its verbose predicate. grid.history publishes the "
            "per-lane write counts and byte formulas; the two numbers to "
            "compare are grid.history.native.buffer_bytes and "
            "grid.history.jax.history_bytes."
        ),
        (
            "Cold/warm is symmetric across lanes: solve 0 is cold_in_process on "
            "BOTH lanes and is excluded from warm_seconds on both. The native "
            "first call pays first-touch faults, extension relocation and the "
            "first OpenMP fork; the JAX first call pays XLA compilation. "
            "warm_sample_count states how many samples the warm list holds, "
            "and timings.row_labels names what each row actually is."
        ),
    ]
    if case.gpmo is not None:
        notes.append(MATCHED_WORK_ITERATION_NOTE)
    if retracing:
        notes.append(JAX_RELAX_SPLIT_RETRACE)
    if case.grid.dr is not None and not dr_is_applied(case):
        notes.append(
            f"grid.dr is published as null and grid.declared_inert_kwargs "
            f"carries the declared {case.grid.dr}: {DR_INERT_MECHANISM}"
        )
    moments = None
    seconds: tuple[float, ...] = ()
    device_memory: dict[str, object] = {}
    iteration_report: dict[str, object] = {}
    thread_pin: dict[str, object] | None = None
    if arguments.lane == NATIVE_LANE and timed and arguments.omp is not None:
        thread_pin = verify_native_thread_pin(arguments.omp)
    if not solving:
        notes.append(
            "Grid-footprint report only: no solve was run. Pass --solve to arm "
            "the timed relax-and-split legs (plan P3.5, second half)."
        )
        if arguments.lane == JAX_LANE:
            # The rung's whole question is the footprint, so it has to be read
            # rather than computed. With no solve the only landmark is the
            # build, so that is the point sampled -- and it is labelled as such
            # instead of being passed off as a post-solve reading.
            device_memory = {"after_build": device_memory_used_mib()}
    elif case.gpmo is not None:
        if arguments.lane == NATIVE_LANE:
            result = run_native_gpmo(
                build, case.gpmo, iterations, policy, arguments.repeat
            )
        else:
            result = run_jax_gpmo(
                build, case.gpmo, iterations, policy, arguments.repeat
            )
        moments, seconds, device_memory, iteration_report = (
            result.moments,
            result.seconds,
            result.device_memory_mib,
            result.iteration_report,
        )
    else:
        if case.relax_split is None:
            raise ProbeError(f"{case.name} has neither a GPMO nor a relax-split spec")
        notes.append(
            "Relax-and-split bookkeeping is NOT symmetric: native MwPGP records "
            "its objective only every int(max_iter/5) iterations and only when "
            "verbose (src/simsoptpp/permanent_magnet_optimization.cpp"
            "::MwPGP_algorithm, its print_MwPGP predicate), while the JAX lane "
            "traces a residual proxy every iteration by default "
            "(src/simsopt_jax/core/pm_optimization.py::mwpgp_solve)."
        )
        if arguments.lane == NATIVE_LANE:
            result = run_native_relax_split(build, case.relax_split, arguments.repeat)
        else:
            result = run_jax_relax_split(build, case.relax_split, arguments.repeat)
        moments, seconds, device_memory, iteration_report = (
            result.moments,
            result.seconds,
            result.device_memory_mib,
            result.iteration_report,
        )

    moment_report: dict[str, object] | None = None
    if moments is not None:
        import numpy as np

        # A non-finite endpoint is a result this probe explicitly anticipates
        # (the ArbVec variants can diverge), and publishing the diagnosis IS
        # the contract -- so the payload is made finite by construction rather
        # than being allowed to carry a NaN that write_probe_artifact would
        # (correctly) refuse to serialize, taking the whole diagnosis with it.
        # The offending statistic is published as a string sentinel next to a
        # null, which is readable and canonical; NaN is neither.
        magnitude = np.linalg.norm(moments, axis=1)
        finite_elements = np.isfinite(moments)
        finite_magnitudes = np.isfinite(magnitude)
        all_finite = bool(finite_elements.all())
        moment_report = {
            "sha256": array_sha256(moments),
            "shape": [int(size) for size in moments.shape],
            "nonzero_dipoles": int(np.count_nonzero(magnitude)),
            "finite": all_finite,
            "nonfinite_elements": int(moments.size - int(finite_elements.sum())),
            "max_magnitude": float(magnitude.max()) if all_finite else None,
        }
        if not all_finite:
            moment_report["max_magnitude_sentinel"] = repr(float(magnitude.max()))
            moment_report["max_finite_magnitude"] = (
                float(magnitude[finite_magnitudes].max())
                if bool(finite_magnitudes.any())
                else None
            )
            notes.append(
                "NON-FINITE ENDPOINT: max_magnitude is published as null with "
                "the offending value in max_magnitude_sentinel. The solve ran "
                "and its timings stand; its endpoint does not."
            )
        if moments_out is not None:
            write_moments(
                moments_out,
                moments,
                {
                    "case": case.name,
                    "lane": arguments.lane,
                    "nphi": case.grid.nphi,
                    "ntheta": case.grid.ntheta,
                    "downsample": case.grid.downsample,
                    "iterations": iterations,
                    "history_policy": policy.name,
                    "A_obj_sha256": grid_report["A_obj_sha256"],
                    "b_obj_sha256": grid_report["b_obj_sha256"],
                },
            )
            moment_report["path"] = str(moments_out)

    # Solve 0 is cold-in-process on BOTH lanes; see the module docstring.
    # Rows 1.. are warm only where an executable exists to reuse: a solving JAX
    # relax-and-split leg re-traces every call, so its rows are repeat_retrace
    # and its warm list is empty rather than a mean of cold solves.
    cold = seconds[0] if seconds else None
    repeated = seconds[1:]
    warm = () if retracing else repeated
    row_labels = [
        COLD_ROW if index == 0 else (RETRACE_ROW if retracing else WARM_ROW)
        for index in range(len(seconds))
    ]
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "publication": PUBLICATION,
        "probe": case.name,
        "summary": case.summary,
        "provenance": list(case.provenance),
        "lane": arguments.lane,
        "configuration": {
            "iterations": iterations,
            "iterations_are_overridden": arguments.k_override is not None,
            "repeat": arguments.repeat,
            "omp_num_threads": arguments.omp,
            "omp_sweep": list(OMP_SWEEP),
            "history_policy": policy.name,
            "native_verbose": policy.native_verbose,
            "native_nhistory": policy.native_nhistory,
            # The per-lane history accounting lives in grid.history, computed
            # once from the built shapes so a count here cannot disagree with
            # the bytes there.
            "history": grid_report["history"],
            # JAX-lane levers are reported as null on the native lane rather
            # than echoing a default the leg never applied.
            "backend_mode": (
                arguments.backend_mode if arguments.lane == JAX_LANE else None
            ),
            "preallocate_off": (
                bool(arguments.preallocate_off) if arguments.lane == JAX_LANE else None
            ),
            "compilation_cache_dir": (
                str(arguments.cache_dir)
                if (arguments.lane == JAX_LANE and arguments.cache_dir is not None)
                else None
            ),
            "applied_environment": applied_environment,
            "scrubbed_environment_prefixes": list(SCRUBBED_ENVIRONMENT_PREFIXES),
            "persistent_cache_thresholds": (
                dict(PERSISTENT_CACHE_THRESHOLDS)
                if (arguments.lane == JAX_LANE and arguments.cache_dir is not None)
                else None
            ),
            "native_thread_pin": thread_pin,
            "solved": solving,
            "output": str(output),
            "published_under_evidence_root": _published_under_evidence_root(output),
        },
        "grid": grid_report,
        "timings": {
            "solve_seconds": list(seconds),
            "row_labels": row_labels,
            "cold_in_process_seconds": cold,
            "warm_seconds": list(warm),
            "warm_sample_count": len(warm),
            "repeat_retrace_seconds": list(repeated) if retracing else [],
            "cold_warm_convention": (
                "solve index 0 is cold_in_process on BOTH lanes and is excluded "
                "from warm_seconds on both. Rows 1.. are warm only where the "
                "leg has an executable to reuse; row_labels says which they "
                "are, and a repeat_retrace row is not a warm sample"
            ),
            "repeat_retrace_mechanism": (
                JAX_RELAX_SPLIT_RETRACE if retracing else None
            ),
        },
        "device_memory_mib": device_memory,
        "iteration_report": iteration_report,
        "moments": moment_report,
        # No interleave *schedule* is published. This process ran one leg; the
        # order the legs actually executed in is the ledger's, and only the
        # ledger can carry it, because the legs are separate processes and a
        # schedule written here would be an intention this process never
        # verified. See the module docstring, "Executed order".
        "interleave_owner": "operator; executed order provable from the ledger",
        "ledger": str(arguments.ledger),
        "notes": notes,
        "identity": identity,
    }
    write_probe_artifact(output, payload)
    # Appended only after the artifact lands: a ledger line claims a leg
    # produced evidence, so it must not outlive a failed publication.
    append_leg_ledger(
        arguments.ledger,
        {
            "utc": datetime.now(timezone.utc).isoformat(),
            "family": case.name,
            "subcommand": arguments.case,
            "lane": arguments.lane,
            "requested_omp": arguments.omp,
            "artifact": str(output),
            "schema": SCHEMA,
        },
    )
    print(f"wrote {output}")
    print(f"ledger {arguments.ledger}")
    for index, value in enumerate(seconds):
        print(f"solve[{index}] {row_labels[index]} {value:.6f} s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProbeError, ProbeConventionError) as failure:
        print(f"pm_gpmo_probes: {failure}", file=sys.stderr)
        raise SystemExit(2) from failure
