"""Bounded CPU rehearsal of the projected route's certification chain.

The certified claim for the projected route is a GPU measurement taken once,
against a one-shot campaign root: a launch spends the root whatever it reaches.
Four such roots were spent by the predecessor route on failures that had
nothing to do with its numerics -- a name the bootstrap phase could not
resolve, an editable install that redirected production imports away from the
sealed tree, a data file the membership rule never sealed, and a receipt gate
that demanded bitwise equality between two independently compiled executables.
Every one of them was reachable in seconds and cost hours.

This module is the gate that makes those failures cheap.  It runs the SAME
chain a certified run runs -- environment, execution-source binding, problem
bootstrap, observable-bound problem identity, lowering pre-gate, engine
compile, solve loop, evidence build, sealed publication, independent artifact
re-validation -- on CPU at a three-attempt budget, in about two minutes.  What
it deliberately does NOT do is decide anything scientific: three attempts
cannot reach the quality target, and the receipt says so in as many words.

Two properties make the substitution sound.  The loop runs on the host, so no
kernel carries the attempt budget and the rehearsal compiles exactly the
program a full-length run compiles -- asserted, not assumed, by the lowering
pre-gate, which lowers both budgets and compares.  And every budget comes from
``CERTIFIED_ROUTE_OPTIONS`` through ``rehearsal_options``, which replaces one
field, so a rehearsal cannot drift into a different route than the one being
certified.

Problem identity is bound by REPRODUCED OBSERVABLES, never by the exact-numeric
problem sha.  That sha is not stable: two runs of the same commit on the same
GPU produced 3ca54b8b and 21b0efc3, and the CPU build of the same problem gives
e6df89b6.  The bootstrap observables agree to ~1e-14 relative across all of
them, so they carry the identity and the sha carries only provenance.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import stat
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Final

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    KernelLowering,
    ProjectedLbfgsIteration,
    ProjectedLbfgsOptions,
    ProjectedLbfgsRun,
    ProjectedLbfgsStatus,
    evaluate_projected_point,
    lower_projected_lbfgs_kernels,
    run_projected_lbfgs,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceEvaluation,
    evaluate_fullspace,
    flatten_fullspace_constraints,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.solve.fullspace import (
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)

from benchmarks.single_stage_fullspace_snapshot import (
    JsonValue,
    canonical_json_bytes,
    load_canonical_json_bytes,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
    DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
    certify_agreement,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG4_EXECUTION_SOURCE_BROAD_ROOTS,
    DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
    DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION,
    DIAG5_EXECUTION_SOURCE_ENTRY_COUNT,
)

REHEARSAL_SCHEMA_VERSION: Final = "single-stage-projected-route-cpu-rehearsal-v1"
REHEARSAL_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-projected-route-cpu-rehearsal-manifest-v1"
)
REHEARSAL_ROUTE: Final = "projected-lagrangian-newton-cg"

EVIDENCE_FILENAME: Final = "rehearsal-evidence.json"
MANIFEST_FILENAME: Final = "artifact-manifest.json"
TERMINAL_COORDINATES_FILENAME: Final = "terminal-coordinates.npy"

# The configuration the two RTX 5090 quality latches were measured at (commit
# 5ab98d15b; Q1 168.02 s at Phi 4.4441e-8, Q2 190.95 s at Phi 4.4783e-8).
# Every budget this module runs is this object with ``maximum_iterations``
# replaced, so a rehearsal cannot rehearse a different route.
CERTIFIED_MAXIMUM_ITERATIONS: Final = 700
CERTIFIED_ROUTE_OPTIONS: Final = ProjectedLbfgsOptions(
    maximum_iterations=CERTIFIED_MAXIMUM_ITERATIONS,
    feasibility_tolerance=1.0e-10,
    objective_target=4.4822246533126125e-08,
    lagrangian_newton=True,
    frozen_projector_line_search=True,
    newton_tangent_fraction_threshold=0.25,
    projector_refresh_period=4,
)
REHEARSAL_MAXIMUM_ITERATIONS: Final = 3

# The native C++ reference sealed with the campaign's native-reference
# artifact: the objective its 1000 BFGS iterations reached, and the wall it
# spent reaching it.  The certified claim is the first inside the second.
NATIVE_TARGET_OBJECTIVE: Final = 4.4822246533126125e-08
NATIVE_WALL_SECONDS_BAR: Final = 287.30421751597896

# The bootstrap point's observables, measured on CPU at the sha-gated build and
# reproduced on both GPUs the route has run on.  These, not the exact-numeric
# sha, bind an artifact to this problem.
CPU_BOOTSTRAP_OBSERVABLES: Final = {
    "objective": 8.44421289101312e-05,
    "feasibility_inf": 1.8708818992040455e-14,
    "gradient_norm": 0.011322200934491376,
    "projected_gradient_norm": 0.0006791192049597319,
}
# Relative bands for the binding gate.  The objective is the tightest quantity
# available (measured agreement ~1e-14 across backends).  The gradient norm
# sums 716 contributions and the projected gradient additionally passes through
# a Gram matrix of condition 3.6e6, so each loosens by the conditioning it
# carries.  Feasibility is NOT compared relatively: both sides sit at 1e-14,
# four decades below the tolerance the route enforces, so comparing them
# relatively would measure rounding and nothing else.
BOOTSTRAP_BINDING_RELATIVE_TOLERANCES: Final = {
    "objective": 1.0e-10,
    "gradient_norm": 1.0e-8,
    "projected_gradient_norm": 1.0e-6,
}
BOOTSTRAP_FEASIBILITY_ABSOLUTE_TOLERANCE: Final = 1.0e-10

# The native C++ endpoint, in PHYSICAL coordinates, sealed with the campaign's
# native-reference artifact.  The endpoint ledger evaluates it through this
# repository's objective so that both sides of every per-term comparison come
# out of one executable.
#
# It is the ONE input this chain reads from outside the repository, so it is
# outside the execution-source manifest and outside the GPU root's sealed source
# snapshot, and it sits in a `.partial-` staging directory of a publication that
# never completed -- exactly the shape an ordinary cleanup pass removes.  Both
# digests the producing artifact records are therefore pinned HERE and verified
# on every load.  They are DIFFERENT quantities and the campaign's naming
# convention uses the second: `single_stage_native_equivalent_reference.py:418`
# writes `arrays/{content_sha256}.npy`, where the content digest is taken over
# the C-order float64 buffer, while the file digest covers the `.npy` container
# around it.  Reading the basename as the file digest is the mistake this
# comment exists to prevent.
NATIVE_ENDPOINT_STATE_FILE_SHA256: Final = (
    "2ec9a9e38e9e4262c4b5dac49f418d1396572ddb22472f9ada979582fe6bf070"
)
NATIVE_ENDPOINT_STATE_CONTENT_SHA256: Final = (
    "2639a955ede349edfdd7f5083776ae3ed0151627f3468d3430ca46029d63a912"
)
NATIVE_ENDPOINT_STATE_PATH: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr1-diag3-cb0-20260811T150010Z.partial-"
    "56a1ec6d730cc005db84f99e9965b868/native-reference/arrays/"
    f"{NATIVE_ENDPOINT_STATE_CONTENT_SHA256}.npy"
)
# Quality parity is DEFINED on these: the objective's pinned physics terms and
# the equality constraints.  See ``PINNED_ENDPOINT_QUALITY_GATES`` for the
# measured agreement between the banked 5090 endpoints and native.
PINNED_ENDPOINT_QUALITY_TERMS: Final = (
    "constraint.boozer|inf",
    "constraint.volume",
    "observable.iota",
    "observable.major_radius",
    "observable.non_qs_ratio",
    "observable.total_length",
    "observable.volume",
    "raw.non_qs",
    "raw.residual",
    "weighted_total",
)
# Reported, never gated.  See ``build_endpoint_ledger`` for why.
INFORMATIONAL_ENDPOINT_OBSERVABLES: Final = ("observable.G", "state.G")

# The equal-minima ceiling.  All five objective weights are 1.0 and the geometry
# penalties are ``0.5 * (x - target)**2``, summed into a total of nonnegative
# terms, so at the certified quality every penalized observable satisfies
# ``|x - target| <= sqrt(2 * Phi)`` and two LEGITIMATE endpoints of equal
# certified quality may differ by up to ``2 * sqrt(2 * Phi)`` in that
# coordinate.  A band below the spread the objective leaves free manufactures a
# false reject; a band above this ceiling refuses nothing the objective permits.
EQUAL_MINIMA_PENALTY_SPREAD: Final = 2.0 * math.sqrt(2.0 * NATIVE_TARGET_OBJECTIVE)


def equal_minima_raw_term_ceiling(native_value: float) -> float:
    """The largest ``not_worse`` excess a LATCH can measure on a raw term.

    The companion of ``EQUAL_MINIMA_PENALTY_SPREAD`` for the other kind of
    pinned term.  A penalized *observable* is bounded through its penalty, which
    is where the square root comes from; a term that IS one of the objective's
    raw summands is bounded directly, because all five weights are 1.0 and every
    raw term is nonnegative, so a latch's ``Phi <= NATIVE_TARGET_OBJECTIVE``
    bounds each summand by that same number.  Dividing by the native value gives
    the ceiling: a band at or above it can never refuse a latch, and a band far
    below it refuses latches the objective permits.

    Both bands the QS leg carries are drawn from this, at the decade below the
    ceiling -- the same placement the geometry observables take against
    ``EQUAL_MINIMA_PENALTY_SPREAD``.  Deriving it for one term and asserting it
    for another is how the predecessor bands came to refuse the campaign's own
    banked evidence.
    """

    return (NATIVE_TARGET_OBJECTIVE - native_value) / abs(native_value)

# How each pinned term is judged, on the attempt that discharges the claim.  The
# comparison class is per term because the terms are not the same KIND of
# quantity, and a single band across all of them manufactures a false reject --
# the failure class this campaign has already hit three times (the V260 shell
# gate, the SQP rho floor, and the fourth predecessor root's bitwise receipt).
#
# * ``absolute`` -- raw equality residuals and the Boozer residual term.  Both
#   sides sit at machine zero, so a relative comparison of them measures
#   rounding.  The band is the route's own raw-equality feasibility tolerance.
#   Measured on the banked latches: 5.5e-13 (Q1) and 3.8e-12 (Q2) worst.
# * ``relative`` -- the two-sided geometry observables.  Each band is the next
#   decade at or above ten times the worst deviation the two banked 5090
#   latches show against native, and stays under the equal-minima ceiling:
#
#       term                    native      worst banked   ceiling    band
#       observable.iota         0.40620273  3.9247e-6      1.474e-3   1e-4
#       observable.major_radius 1.46744380  5.6022e-6      4.081e-4   1e-4
#       observable.volume       -0.29044576 5.0e-15        (equality) 1e-6
#
#   The predecessor bands were 1e-6 on all three; measured against the sealed
#   endpoints that refuses Q2 on iota (3.92e-6) AND BOTH ARMS on major radius
#   (5.60e-6, 1.03e-6) -- the campaign's own banked evidence, refused by the
#   gate meant to certify it.  Volume keeps 1e-6 because it is an equality
#   constraint, machine-identical on both sides, not a slack direction.
# * ``not_worse`` -- one-sided terms, judged only on the worse side.  Two kinds
#   qualify.  The QUALITY quantities: the claim is "REACH native's endpoint
#   objective", never "equal it", and non-QS came out 0.885% BETTER than native
#   on Q1 and 0.087% better on Q2, so a two-sided band would refuse the very
#   evidence the campaign banked.  And the HINGED geometry term: the length
#   penalty is ``0.5 * max(L - target, 0)**2``, exactly flat below the target,
#   so nothing in the shared objective pins a SHORTER coil set -- Q1's terminal
#   sits below the hinge with ``raw.length`` exactly 0.0.  Gating it two-sidedly
#   would gate a free direction, which is why G is informational.  Gating it on
#   the longer side alone is a real gate: 1e-4 relative on L is a 2.1e-3 excess
#   whose penalty alone is 49x the certified Phi, so no endpoint at the
#   certified quality can fail it for a legitimate reason.
#
#   The one-sided bands are derived the SAME way the two-sided ones are, through
#   ``equal_minima_raw_term_ceiling`` rather than ``EQUAL_MINIMA_PENALTY_SPREAD``
#   because these terms are raw objective summands, not penalized coordinates:
#
#       term                     native        ceiling    band     under ceiling
#       raw.non_qs               4.4809e-08    2.961e-4   1e-4     2.96x
#       observable.non_qs_ratio  4.4809e-08    2.961e-4   1e-4     2.96x
#       weighted_total           4.4822e-08    2.061e-13  1e-6     INERT
#
#   ``raw.non_qs`` and ``observable.non_qs_ratio`` are the same variable and
#   carry 99.93% of the objective, so they are where a false reject costs the
#   most.  The predecessor band of 1e-6 sat 296x under their ceiling -- two
#   decades tighter than the placement the geometry terms received, on the
#   dominant term -- and left the nearer banked latch (Q2) clearing refusal by
#   only 3.95x while the two banked arms differ FROM EACH OTHER in this very
#   quantity by 7.98e-03, i.e. 9.2x that surviving margin.  A gate whose margin
#   is smaller than the observed run-to-run spread of the quantity it gates is
#   not calibrated.  1e-4 is the decade below the ceiling, the same placement
#   ``observable.major_radius`` takes (4.08x under its own), and it moves Q2's
#   margin to 5.94x.
#
#   ``weighted_total`` is the opposite case and is kept at 1e-6 deliberately:
#   its ceiling is 2.061e-13, so ANY band a latch could fail would have to be
#   tighter than the cross-executable ULP between this repository's re-evaluated
#   native total and the frozen literal.  1e-6 sits 4.85e6x ABOVE the ceiling,
#   which makes the term INERT -- it cannot refuse a latch and it cannot false
#   reject one either.  Its substantive gate is the latch number itself,
#   ``terminal_objective <= NATIVE_TARGET_OBJECTIVE``, re-checked from the
#   published bytes.  ``observable.total_length`` is inert for the same reason
#   (band 3.51x above its 2.853e-05 ceiling) and is documented as such above.
PINNED_ENDPOINT_QUALITY_GATES: Final = {
    "constraint.boozer|inf": ("absolute", 1.0e-10),
    "constraint.volume": ("absolute", 1.0e-10),
    "observable.iota": ("relative", 1.0e-4),
    "observable.major_radius": ("relative", 1.0e-4),
    "observable.non_qs_ratio": ("not_worse", 1.0e-4),
    "observable.total_length": ("not_worse", 1.0e-4),
    "observable.volume": ("relative", 1.0e-6),
    "raw.non_qs": ("not_worse", 1.0e-4),
    "raw.residual": ("absolute", 1.0e-10),
    "weighted_total": ("not_worse", 1.0e-6),
}

REQUIRED_ENVIRONMENT: Final = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "true",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}

REPOSITORY_ROOT: Final = Path(__file__).resolve(strict=True).parents[1]


class RehearsalError(RuntimeError):
    """A rehearsal gate refused; the chain stops and keeps its partial root."""


def rehearsal_options(iterations: int) -> ProjectedLbfgsOptions:
    """The certified configuration at a chosen attempt budget.

    Exactly one field differs from ``CERTIFIED_ROUTE_OPTIONS``, so a change to
    the certified configuration reaches every rehearsal without an edit here.
    """

    return replace(CERTIFIED_ROUTE_OPTIONS, maximum_iterations=iterations)


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_scalar(value: object) -> JsonValue:
    """Encode one host scalar, mapping nonfinite floats to null.

    ``canonical_json_bytes`` refuses ``NaN`` and the infinities, and the
    campaign's evidence protocol writes an unavailable numeric field as JSON
    null.  ``bool`` is tested first because it is a subclass of ``int``.
    """

    if value is None or type(value) is bool or type(value) is int:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, tuple):
        return [json_scalar(item) for item in value]
    raise RehearsalError(f"unencodable evidence scalar: {value!r}")


def validate_environment(
    environment: Mapping[str, str],
    *,
    required: Mapping[str, str] = REQUIRED_ENVIRONMENT,
) -> dict[str, JsonValue]:
    """Refuse any launch whose backend or precision differs from the contract.

    JAX resolves its platform and its x64 flag lazily, so an absent variable
    does not fail loudly: it silently produces a different run.  The gate is
    therefore an equality check against every variable the contract names.
    ``required`` is a parameter because the GPU lane pins a different platform
    and must not re-spell the check to say so.
    """

    for name, expected in sorted(required.items()):
        observed = environment.get(name)
        if observed != expected:
            raise RehearsalError(f"{name} must equal {expected!r}, observed {observed!r}")
    return {name: environment[name] for name in sorted(required)}


def load_execution_source_manifest(
    repository: Path,
) -> tuple[dict[str, JsonValue], Mapping[str, Mapping[str, JsonValue]]]:
    """Load the frozen execution-source manifest, recomputing what it asserts.

    The manifest is the repository's single statement of which bytes may
    execute in a certified run.  Its self-describing digest is recomputed here
    rather than trusted, and its entry count is checked against the authority
    module's frozen constant so that a manifest and an authority which disagree
    cannot both be believed.
    """

    manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    payload = manifest_path.read_bytes()
    document = load_canonical_json_bytes(payload)
    if not isinstance(document, dict) or frozenset(document) != frozenset(
        {"entries", "entries_sha256", "schema_version"}
    ):
        raise RehearsalError("execution-source manifest fields differ")
    if document["schema_version"] != DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION:
        raise RehearsalError("execution-source manifest schema differs")
    entries = document["entries"]
    if not isinstance(entries, dict):
        raise RehearsalError("execution-source manifest entries are not an object")
    if len(entries) != DIAG5_EXECUTION_SOURCE_ENTRY_COUNT:
        raise RehearsalError(
            f"execution-source manifest holds {len(entries)} entries, "
            f"authority freezes {DIAG5_EXECUTION_SOURCE_ENTRY_COUNT}"
        )
    recomputed = sha256_hex(canonical_json_bytes(entries))
    if recomputed != document["entries_sha256"]:
        raise RehearsalError("execution-source manifest entries digest differs")
    evidence: dict[str, JsonValue] = {
        "relative_path": DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
        "schema_version": DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION,
        "manifest_sha256": sha256_hex(payload),
        "entries_sha256": recomputed,
        "entry_count": len(entries),
    }
    return evidence, entries


# The distribution's own top-level packages.  A module of one of these that
# resolved OUTSIDE the checkout is the root-2 failure class itself -- the
# scikit-build-core editable finder outranking PYTHONPATH -- and the hash gate
# is structurally blind to it, because a module outside the tree has no entry to
# be compared against.  Naming the packages is what turns that silence into a
# refusal.
DISTRIBUTION_PACKAGE_ROOTS: Final = (
    "simsopt",
    "simsopt_jax",
    "simsopt_jax_adapters",
    "simsoptpp",
)


def _imported_modules() -> list[tuple[str, Path]]:
    """Every currently imported module that has a source file, resolved."""

    resolved: list[tuple[str, Path]] = []
    for name, module in sorted(sys.modules.items()):
        if not isinstance(module, ModuleType):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        resolved.append((name, Path(origin).resolve()))
    return resolved


def refuse_redirected_distribution_modules() -> None:
    """Refuse any of this distribution's modules that came from another tree.

    The hash gate can only compare what it can see, and a module resolved
    OUTSIDE the checkout has no manifest entry to be compared against: it is not
    refused, it is invisible, and the gate's liveness check is satisfied by
    ``benchmarks/*`` alone.  That silence is the root-2 failure class itself.

    The comparison is against ``REPOSITORY_ROOT``, the tree THIS module lives
    in, not against ``bind_execution_sources``' parameter: the invariant is
    about where the distribution's own packages came from, while the parameter
    exists so the manifest machinery can be pointed at a synthetic tree.
    """

    for name, path in _imported_modules():
        if path.is_relative_to(REPOSITORY_ROOT):
            continue
        if name.split(".")[0] in DISTRIBUTION_PACKAGE_ROOTS:
            raise RehearsalError(
                f"module {name} executes {path}, which is outside "
                f"{REPOSITORY_ROOT}"
            )


def bind_execution_sources(repository: Path) -> dict[str, JsonValue]:
    """Prove every executing repository module is the manifest's byte-for-byte.

    The predecessor route lost a one-shot root to a scikit-build-core editable
    install whose meta-path finder outranks ``PYTHONPATH``, so production
    modules resolved outside the tree the run believed it was executing.  What
    catches that class is not "is the manifest well formed" but "does every
    module this process actually imported hash to its manifest entry", asked
    after the imports have happened.

    A module of this distribution that resolved outside the checkout entirely is
    refused first -- see ``refuse_redirected_distribution_modules`` -- because
    the hash gate is structurally blind to it.

    Modules under the three manifest roots are gated.  Repository modules
    outside those roots -- the test bootstrap, for instance -- are recorded as
    evidence and not gated: those roots are what the campaign's execution-source
    contract is written against, and a certified launch imports nothing else
    from the tree.  Modules under a hidden top-level directory are the
    interpreter's own installation, which lives inside this checkout
    (``.venv-qn-cpu``): they are counted and their roots named, but a
    virtualenv that happens to sit in the tree is not repository source and
    listing two thousand of its files would bury the evidence that matters.
    """

    refuse_redirected_distribution_modules()
    manifest_evidence, entries = load_execution_source_manifest(repository)
    bound: list[JsonValue] = []
    unmanifested: list[JsonValue] = []
    environment_roots: set[str] = set()
    environment_count = 0
    for name, path in _imported_modules():
        if not path.is_relative_to(repository):
            continue
        relative = path.relative_to(repository).as_posix()
        leading = PurePosixPath(relative).parts[0]
        if leading.startswith("."):
            environment_roots.add(leading)
            environment_count += 1
            continue
        if leading not in DIAG4_EXECUTION_SOURCE_BROAD_ROOTS:
            unmanifested.append({"module": name, "relative_path": relative})
            continue
        entry = entries.get(relative)
        if entry is None:
            raise RehearsalError(
                f"module {name} executes {relative}, which the manifest omits"
            )
        payload = path.read_bytes()
        digest = sha256_hex(payload)
        if digest != entry["sha256"] or len(payload) != entry["size_bytes"]:
            raise RehearsalError(
                f"module {name} executes {relative} with bytes the manifest "
                "does not describe"
            )
        bound.append(
            {
                "module": name,
                "relative_path": relative,
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )
    if not bound:
        raise RehearsalError("no manifest-bound module reached this process")
    return {
        "manifest": manifest_evidence,
        "bound_modules": sorted(bound, key=lambda item: item["module"]),
        "unmanifested_repository_modules": sorted(
            unmanifested, key=lambda item: item["module"]
        ),
        "interpreter_installation_modules": {
            "count": environment_count,
            "roots": sorted(environment_roots),
        },
    }


@dataclass(frozen=True, slots=True)
class NativeEndpointState:
    """The sealed native endpoint and the two digests that identify it."""

    values: np.ndarray
    file_sha256: str
    content_sha256: str


def load_native_endpoint_state() -> NativeEndpointState:
    """Read the sealed native endpoint, refusing any bytes but the pinned ones.

    This array is the reference side of every pinned-term comparison, i.e. the
    definition of quality parity, and it is the one input the chain reads from
    outside the repository: no execution-source entry covers it, no source
    snapshot seals it, and the directory it sits in is owner-writable.  Three
    facts are therefore checked -- the file digest, the array's own content
    digest, and the campaign's basename-IS-the-content-digest convention -- and
    the bytes that were hashed are the bytes that get decoded.
    """

    payload = NATIVE_ENDPOINT_STATE_PATH.read_bytes()
    file_sha256 = sha256_hex(payload)
    if file_sha256 != NATIVE_ENDPOINT_STATE_FILE_SHA256:
        raise RehearsalError(
            f"native endpoint state {NATIVE_ENDPOINT_STATE_PATH} hashes to "
            f"{file_sha256}, not the pinned {NATIVE_ENDPOINT_STATE_FILE_SHA256}"
        )
    values = np.load(io.BytesIO(payload), allow_pickle=False)
    content_sha256 = sha256_hex(
        np.ascontiguousarray(values, dtype=np.float64).tobytes(order="C")
    )
    if content_sha256 != NATIVE_ENDPOINT_STATE_CONTENT_SHA256:
        raise RehearsalError(
            f"native endpoint state values hash to {content_sha256}, not the "
            f"pinned {NATIVE_ENDPOINT_STATE_CONTENT_SHA256}"
        )
    if NATIVE_ENDPOINT_STATE_PATH.stem != content_sha256:
        raise RehearsalError(
            "native endpoint state basename is not its content digest: "
            f"{NATIVE_ENDPOINT_STATE_PATH.stem} != {content_sha256}"
        )
    return NativeEndpointState(
        values=values, file_sha256=file_sha256, content_sha256=content_sha256
    )


def run_latched(run: ProjectedLbfgsRun) -> bool:
    """Whether the engine reported reaching its configured objective target."""

    return int(run.status) == int(ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED)


def endpoint_ledger_is_gated(*, iterations: int, latched: bool) -> bool:
    """Whether one attempt's ledger carries per-term verdicts, not just terms.

    The pinned-term gate is a claim about the LATCH: it decides whether the
    endpoint that reached native's objective reached native's physics.  It
    therefore binds the attempt that discharges the claim and no other.  A
    non-latching attempt at any budget publishes its ledger ungated and the
    protocol continues to the next draw -- gating it would convert a stochastic
    miss (the campaign's own measured rate is one in five) into
    ``GATE_REFUSED:endpoint_ledger``, break the attempt loop, and make
    ``NO_LATCH_IN_PROTOCOL`` unreachable at a root, dissolving exactly the
    insurance the pre-registered N-attempt protocol was written to buy.  A
    bounded budget is likewise ungated: three attempts sit four orders of
    magnitude from the endpoint.
    """

    return latched and iterations == CERTIFIED_MAXIMUM_ITERATIONS


class BoundCase:
    """The coupled full-space problem, ready to optimize and to bind.

    ``raw_joint`` returns the true weighted objective and the RAW equality
    residuals, so the optimizer's own feasibility tolerance IS the campaign's
    1e-10 raw-equality gate rather than a scaled proxy for it.
    ``standalone_evaluation`` reaches the same objective -- and every physics
    term behind it -- through a separate executable, which is what the endpoint
    agreement certifies against and what the endpoint ledger is built from.
    """

    def __init__(self) -> None:
        bootstrap = build_single_stage_fullspace_bootstrap()
        problem = bootstrap.problem
        scaling = fullspace_scaling_from_bootstrap(bootstrap.z0, problem)

        def raw_joint(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
            physical = fullspace_physical_coordinates(coordinates, scaling)
            evaluation = evaluate_fullspace(physical, problem)
            return (
                evaluation.weighted_total,
                flatten_fullspace_constraints(evaluation.constraints),
            )

        def standalone_evaluation(
            coordinates: jax.Array,
        ) -> tuple[FullSpaceEvaluation, jax.Array]:
            physical = fullspace_physical_coordinates(coordinates, scaling)
            return evaluate_fullspace(physical, problem), physical

        self.raw_joint = raw_joint
        self.standalone_evaluation = jax.jit(standalone_evaluation)
        self.evaluate_point = jax.jit(
            lambda coordinates: evaluate_projected_point(raw_joint, coordinates)
        )
        native_state = load_native_endpoint_state()
        self.start = fullspace_optimizer_coordinates(bootstrap.z0, scaling)
        self.native_endpoint = fullspace_optimizer_coordinates(
            jnp.asarray(native_state.values, dtype=jnp.float64),
            scaling,
        )
        self.native_state = native_state
        self.problem_sha256 = exact_numeric_tree_sha256(problem)
        self.bootstrap_sha256 = exact_numeric_tree_sha256(bootstrap.z0)


def bind_problem_identity(case: BoundCase) -> dict[str, JsonValue]:
    """Bind this process to the campaign's problem by reproducing observables.

    The exact-numeric problem sha is recorded and NOT gated: it is stable only
    for a fixed backend and build order, and two runs of the same commit on the
    same GPU have produced different values.  The four bootstrap observables
    reproduce across every backend the route has run on, so they carry the
    identity.
    """

    point = jax.block_until_ready(case.evaluate_point(case.start))
    measured = {
        "objective": float(point.objective),
        "feasibility_inf": float(point.feasibility_inf),
        "gradient_norm": float(point.gradient_norm),
        "projected_gradient_norm": float(point.projected_gradient_norm),
    }
    relative = {
        name: abs(measured[name] - reference) / abs(reference)
        for name, reference in CPU_BOOTSTRAP_OBSERVABLES.items()
    }
    checks = {
        name: relative[name] <= tolerance
        for name, tolerance in BOOTSTRAP_BINDING_RELATIVE_TOLERANCES.items()
    }
    checks["feasibility_inf_absolute"] = (
        measured["feasibility_inf"] <= BOOTSTRAP_FEASIBILITY_ABSOLUTE_TOLERANCE
        and CPU_BOOTSTRAP_OBSERVABLES["feasibility_inf"]
        <= BOOTSTRAP_FEASIBILITY_ABSOLUTE_TOLERANCE
    )
    evidence: dict[str, JsonValue] = {
        "reference_observables": dict(CPU_BOOTSTRAP_OBSERVABLES),
        "measured_observables": measured,
        "relative_difference": relative,
        "relative_tolerances": dict(BOOTSTRAP_BINDING_RELATIVE_TOLERANCES),
        "feasibility_absolute_tolerance": BOOTSTRAP_FEASIBILITY_ABSOLUTE_TOLERANCE,
        "checks": dict(sorted(checks.items())),
        "bound": all(checks.values()),
        "recorded_problem_sha256": case.problem_sha256,
        "recorded_bootstrap_sha256": case.bootstrap_sha256,
        "sha_is_binding": False,
    }
    if not evidence["bound"]:
        raise RehearsalError(
            "bootstrap observables do not reproduce the campaign problem: "
            f"{json.dumps(evidence['checks'], sort_keys=True)}"
        )
    return evidence


def measure_lowering_pre_gate(case: BoundCase, iterations: int) -> dict[str, JsonValue]:
    """Lower the route at the rehearsal budget and at the certified budget.

    Lowering costs one trace per kernel and never reaches a backend compiler,
    so this runs BEFORE the compile it protects.  The gate is self-referential
    and needs no frozen size constant that could drift out of date: the two
    budgets must lower to identical IR.  Where they do not, the loop has taken
    the attempt budget into a kernel and no bounded rehearsal can stand in for
    a certified run's compile -- which is exactly the regression the
    predecessor route shipped when its safeguard unrolled the attempt body
    three times and its CPU compile grew to seventy minutes.
    """

    rehearsal = lower_projected_lbfgs_kernels(
        case.raw_joint, case.start, options=rehearsal_options(iterations)
    )
    certified = lower_projected_lbfgs_kernels(
        case.raw_joint,
        case.start,
        options=rehearsal_options(CERTIFIED_MAXIMUM_ITERATIONS),
    )
    if rehearsal != certified:
        raise RehearsalError(
            "lowered kernels depend on the attempt budget; a bounded rehearsal "
            "cannot stand in for the certified run's compile"
        )
    return {
        "rehearsal_iterations": iterations,
        "certified_iterations": CERTIFIED_MAXIMUM_ITERATIONS,
        "budget_independent": True,
        "kernels": [lowering_payload(record) for record in rehearsal],
        "total_ir_bytes": sum(record.ir_bytes for record in rehearsal),
    }


def lowering_payload(record: KernelLowering) -> dict[str, JsonValue]:
    return {
        "name": record.name,
        "ir_bytes": record.ir_bytes,
        "while_operations": record.while_operations,
    }


def iteration_payload(record: ProjectedLbfgsIteration) -> dict[str, JsonValue]:
    return {
        field: json_scalar(value)
        for field, value in record._asdict().items()
        if field != "coordinates"
    }


def collapse_proximity_margin(
    run: ProjectedLbfgsRun, options: ProjectedLbfgsOptions
) -> float:
    """The run's smallest recorded step scale, in units of the line search floor.

    Backtracking on a secant direction stops below ``minimum_step_scale``, so
    the smallest scale a run ever recorded says how close it came to that end.
    One caveat bounds the reading: a MODEL direction backtracks only to
    ``model_step_rescue_scale`` before the secant rescue takes the iteration
    over, and when that rescue also places nothing the row keeps the model
    arm's scale, four orders above the floor.  Measured on the banked arms:
    10.8x and 2.70x on the two 5090 latches, 67x and 4.75x on the two A100
    latches, and 0.95x on the A100 arm that collapsed -- the only one below
    one, and the collapse itself.  Those four latches span 2.70x to 67x, so any
    threshold drawn above unity already excludes one of them; unity is the only
    separation the evidence carries.  On a collapsed run the terminal row
    carries the scale the search gave up at rather than one it placed, which is
    exactly why that run's margin falls below one.

    This is REPORTED and never acted on.  Every input is already in the
    iteration rows at zero marginal cost, so emitting it changes no iterate; a
    refresh or floor policy driven by it would re-open certification, and the
    measured evidence says such a policy spends wall without buying latch
    probability.  A run that recorded no iteration has no margin and says so.
    """

    scales = [record.step_scale for record in run.iterations]
    if not scales:
        return float("nan")
    return min(scales) / options.minimum_step_scale


def _physics_leaves(record: object, prefix: str) -> dict[str, float]:
    """Flatten one registered evaluation dataclass into named host scalars."""

    flattened: dict[str, float] = {}
    for field in dataclass_fields(record):
        value = np.asarray(jax.device_get(getattr(record, field.name)))
        name = f"{prefix}.{field.name}"
        flattened[name if value.ndim == 0 else f"{name}|inf"] = float(
            value if value.ndim == 0 else np.max(np.abs(value))
        )
    return flattened


def _pinned_term_verdict(
    name: str, terminal: float, native: float
) -> dict[str, JsonValue]:
    """Judge one pinned term by the comparison class its kind requires."""

    comparison, band = PINNED_ENDPOINT_QUALITY_GATES[name]
    difference = terminal - native
    if comparison == "absolute":
        measured = abs(difference)
        passed = measured <= band
    elif comparison == "relative":
        measured = abs(difference) / abs(native)
        passed = measured <= band
    else:
        # Lower is better on every ``not_worse`` term, so an endpoint that beat
        # native measures a negative excess and passes with room to spare.
        measured = difference / abs(native)
        passed = measured <= band
    return {
        "comparison": comparison,
        "band": band,
        "measured": json_scalar(measured),
        "passed": bool(passed),
    }


def gate_endpoint_ledger(ledger: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Judge the pinned terms of one ledger; G is absent by construction.

    The pinned set is read from this module, never from the ledger, so an
    artifact cannot narrow the terms it is then judged on.  The informational
    observables are not consulted at all: gating a direction the shared
    objective deliberately leaves free is how a false reject gets manufactured.
    """

    terminal = ledger["terminal"]
    native = ledger["native"]
    if not isinstance(terminal, dict) or not isinstance(native, dict):
        raise RehearsalError("endpoint ledger sides are not term maps")
    if frozenset(PINNED_ENDPOINT_QUALITY_GATES) != frozenset(
        PINNED_ENDPOINT_QUALITY_TERMS
    ):
        raise RehearsalError("pinned quality terms and their gates disagree")
    if frozenset(PINNED_ENDPOINT_QUALITY_GATES) & frozenset(
        INFORMATIONAL_ENDPOINT_OBSERVABLES
    ):
        raise RehearsalError("an informational observable is being gated")
    verdicts = {
        name: _pinned_term_verdict(
            name, float(terminal[name]), float(native[name])
        )
        for name in sorted(PINNED_ENDPOINT_QUALITY_TERMS)
    }
    return {
        "terms": verdicts,
        "failed_terms": sorted(
            name for name, verdict in verdicts.items() if not verdict["passed"]
        ),
        "passed": all(verdict["passed"] for verdict in verdicts.values()),
    }


def build_endpoint_ledger(
    case: BoundCase, run: ProjectedLbfgsRun, *, gated: bool = False
) -> dict[str, JsonValue]:
    """Report every physics term at the endpoint beside the native reference.

    A total objective and a feasibility number cannot distinguish "reached the
    same physics" from "reached the same scalar", so the receipt carries the
    per-term ledger: the five raw objective terms, the eight observables, both
    constraint blocks, and the two state components that are not coil geometry.

    ``PINNED_ENDPOINT_QUALITY_TERMS`` names what quality parity is DEFINED on.
    ``INFORMATIONAL_ENDPOINT_OBSERVABLES`` names what it is not: the non-QS
    term is a field-scale-invariant ratio and nothing in the shared objective
    pins the net poloidal current, so G is a flat valley direction along which
    distinct equal-quality minima exist -- the measured endpoints sit 0.785%
    (Q1) and 0.934% (Q2) below native in G at a scaled distance of ~2.3, while
    every pinned geometry term agrees to 5.7e-6 relative or better.  Gating G
    would manufacture a false reject on a direction the objective deliberately
    leaves free, and so would gating the length below its hinge; see
    ``PINNED_ENDPOINT_QUALITY_GATES`` for how each band is derived.

    ``gated`` adds the per-term verdicts and says so; the caller decides what a
    failed verdict costs.  Whether it is set is ``endpoint_ledger_is_gated``'s
    decision, not this function's, so both lanes ask one owner.
    """

    terminal_evaluation, terminal_physical = jax.block_until_ready(
        case.standalone_evaluation(run.coordinates)
    )
    native_evaluation, native_physical = jax.block_until_ready(
        case.standalone_evaluation(case.native_endpoint)
    )
    rows: dict[str, dict[str, JsonValue]] = {}
    for label, evaluation, physical in (
        ("terminal", terminal_evaluation, terminal_physical),
        ("native", native_evaluation, native_physical),
    ):
        state = np.asarray(jax.device_get(physical))
        rows[label] = {
            **_physics_leaves(evaluation.raw_terms, "raw"),
            **_physics_leaves(evaluation.observables, "observable"),
            **_physics_leaves(evaluation.constraints, "constraint"),
            "weighted_total": float(evaluation.weighted_total),
            # The last two physical components are the rotational transform and
            # the net poloidal current; every earlier one is coil geometry.
            "state.iota": float(state[-2]),
            "state.G": float(state[-1]),
        }
    ledger: dict[str, JsonValue] = {
        "terminal": dict(sorted(rows["terminal"].items())),
        "native": dict(sorted(rows["native"].items())),
        "relative_difference": {
            name: json_scalar(
                abs(rows["terminal"][name] - rows["native"][name])
                / abs(rows["native"][name])
                if rows["native"][name] != 0.0
                else None
            )
            for name in sorted(rows["terminal"])
        },
        "native_state_relative_path": NATIVE_ENDPOINT_STATE_PATH.name,
        "native_state_sha256": case.native_state.file_sha256,
        "native_state_content_sha256": case.native_state.content_sha256,
        "pinned_quality_terms": list(PINNED_ENDPOINT_QUALITY_TERMS),
        "informational_observables": list(INFORMATIONAL_ENDPOINT_OBSERVABLES),
        "gated_at_this_budget": bool(gated),
    }
    if gated:
        ledger["pinned_term_gate"] = gate_endpoint_ledger(ledger)
    return ledger


def certify_endpoint_agreement(
    case: BoundCase,
    run: ProjectedLbfgsRun,
) -> dict[str, JsonValue]:
    """Certify the loop's endpoint against a standalone re-evaluation.

    The loop's terminal objective leaves the fused point-evaluation kernel;
    ``standalone_evaluation`` reaches the same number through the objective
    alone.  Two independently compiled executables agree to a few ULP and never
    bitwise -- demanding bitwise here is what refused the predecessor route's
    fourth root after a complete solve -- so the agreement is certified against
    the campaign's frozen relative band with its absolute floor.  Feasibility
    is gated absolutely for the same reason the bootstrap binding does not
    compare it relatively, and the terminal state is hashed exactly because a
    published copy of one array is not a cross-executable quantity.
    """

    evaluation, _ = jax.block_until_ready(case.standalone_evaluation(run.coordinates))
    standalone = float(evaluation.weighted_total)
    certify_agreement(
        standalone,
        run.objective,
        "projected route terminal objective",
        relative_tolerance=DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
        absolute_floor=DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
    )
    # Spelled as the negation of the contract rather than as ``> tolerance``:
    # every comparison against a NaN is false, so the ``>`` form passes a
    # nonfinite feasibility through the gate that exists to refuse it.
    if not run.feasibility_inf <= CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance:
        raise RehearsalError(
            f"terminal feasibility {run.feasibility_inf!r} is not within "
            f"{CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance!r}"
        )
    return {
        "loop_terminal_objective": run.objective,
        "standalone_terminal_objective": standalone,
        "relative_tolerance": DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
        "absolute_floor": DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
        "terminal_feasibility_inf": run.feasibility_inf,
        "feasibility_absolute_tolerance": CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance,
        "terminal_state_sha256": exact_numeric_tree_sha256(run.coordinates),
    }


def build_rehearsal_evidence(
    *,
    environment: Mapping[str, JsonValue],
    execution_sources: Mapping[str, JsonValue],
    problem_identity: Mapping[str, JsonValue],
    lowering: Mapping[str, JsonValue],
    run: ProjectedLbfgsRun,
    endpoint: Mapping[str, JsonValue],
    endpoint_ledger: Mapping[str, JsonValue],
    iterations: int,
    timing_seconds: Mapping[str, float],
) -> dict[str, JsonValue]:
    """Assemble the rehearsal receipt in the shape a certified receipt takes."""

    options = rehearsal_options(iterations)
    objectives = [record.objective for record in run.iterations]
    return {
        "schema_version": REHEARSAL_SCHEMA_VERSION,
        "route": REHEARSAL_ROUTE,
        "environment": dict(environment),
        "execution_sources": dict(execution_sources),
        "problem_identity": dict(problem_identity),
        "lowering_pre_gate": dict(lowering),
        "options": {
            field: json_scalar(getattr(options, field))
            for field in sorted(options.__dataclass_fields__)
        },
        "certified_options_delta": {
            field: json_scalar(getattr(options, field))
            for field in sorted(options.__dataclass_fields__)
            if getattr(options, field) != getattr(CERTIFIED_ROUTE_OPTIONS, field)
        },
        "native_reference": {
            "target_objective": NATIVE_TARGET_OBJECTIVE,
            "wall_seconds_bar": NATIVE_WALL_SECONDS_BAR,
        },
        "solve": {
            "status": int(run.status),
            "status_name": ProjectedLbfgsStatus(int(run.status)).name,
            "iterations_run": len(run.iterations),
            "terminal_objective": run.objective,
            "terminal_projected_gradient_inf": run.projected_gradient_inf,
            "stored_pairs": run.stored_pairs,
            "projector_materializations": run.projector_materializations,
            "tangency_forced_refreshes": run.tangency_forced_refreshes,
            "line_search_forced_refreshes": run.line_search_forced_refreshes,
            "monotone_descent": all(
                later <= earlier
                for earlier, later in zip(objectives, objectives[1:], strict=False)
            ),
            "maximum_feasibility_inf": json_scalar(
                max(
                    (record.feasibility_inf for record in run.iterations),
                    default=float("nan"),
                )
            ),
            "collapse_proximity_margin": json_scalar(
                collapse_proximity_margin(run, options)
            ),
            "rows": [iteration_payload(record) for record in run.iterations],
        },
        "endpoint_agreement": dict(endpoint),
        "endpoint_ledger": dict(endpoint_ledger),
        # The rehearsal exercises the chain and decides nothing scientific.
        # Three attempts cannot reach the target, and the receipt says so
        # rather than leaving a reader to infer it from a status code.
        "quality_claim": "NOT_CLAIMED_AT_REHEARSAL_BUDGET",
        "timing_seconds": dict(sorted(timing_seconds.items())),
        # The certified wall is the engine's compile plus its solve.  Building
        # the problem and binding it are setup the native reference's 287.30 s
        # does not contain either; stating the boundary in the artifact is what
        # makes the comparison checkable instead of assumed.
        "timing_boundary": "engine_compile_plus_solve",
    }


def seal_and_sync(root: Path) -> None:
    """Make the artifact read-only and durable, refusing anything irregular."""

    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RehearsalError("rehearsal artifact contains a symlink")
        observed = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            directories.append(path)
            continue
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
            raise RehearsalError("rehearsal artifact contains an invalid file")
        path.chmod(0o444)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        directory.chmod(0o555)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def validate_sealed_modes(root: Path) -> None:
    """Refuse a published tree that is not 0555/0444 all the way down.

    Sealing and checking the seal are two different facts: ``seal_and_sync``
    says what this process did, and this says what the bytes a later reader
    finds actually are.  Both lanes publish under the same rule, so the check
    lives once here rather than twice.
    """

    for path in (root, *root.rglob("*")):
        observed = path.stat(follow_symlinks=False)
        expected_mode = 0o555 if stat.S_ISDIR(observed.st_mode) else 0o444
        if stat.S_IMODE(observed.st_mode) != expected_mode:
            raise RehearsalError(f"sealed artifact mode differs: {path}")


def artifact_manifest_payload(
    root: Path, *, schema_version: str = REHEARSAL_MANIFEST_SCHEMA_VERSION
) -> dict[str, JsonValue]:
    """Describe every artifact file except the manifest that will carry this.

    ``schema_version`` is a parameter so the GPU root lane publishes its own
    schema through THIS enumeration rather than a second spelling of it.
    """

    manifest_path = root / MANIFEST_FILENAME
    files: list[JsonValue] = []
    directories: list[JsonValue] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise RehearsalError("rehearsal artifact contains a symlink")
        observed = path.stat(follow_symlinks=False)
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(observed.st_mode):
            directories.append({"mode": "0555", "relative_path": relative})
        elif stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1:
            payload = path.read_bytes()
            files.append(
                {
                    "mode": "0444",
                    "relative_path": relative,
                    "sha256": sha256_hex(payload),
                    "size_bytes": len(payload),
                }
            )
        else:
            raise RehearsalError("rehearsal artifact contains an invalid file")
    return {
        "directories": directories,
        "files": files,
        "schema_version": schema_version,
    }


def rename_noreplace(source: Path, destination: Path) -> None:
    """Publish by rename, refusing to replace an existing final root."""

    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), str(destination))
    raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_rehearsal(
    staging_root: Path,
    output_root: Path,
    evidence: Mapping[str, JsonValue],
    terminal_coordinates: np.ndarray,
) -> Path:
    """Write, seal and atomically publish one rehearsal artifact."""

    (staging_root / EVIDENCE_FILENAME).write_bytes(canonical_json_bytes(evidence))
    with (staging_root / TERMINAL_COORDINATES_FILENAME).open("wb") as stream:
        np.save(stream, terminal_coordinates, allow_pickle=False)
    (staging_root / MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(artifact_manifest_payload(staging_root))
    )
    seal_and_sync(staging_root)
    rename_noreplace(staging_root, output_root)
    descriptor = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return output_root


def validate_rehearsal_artifact(artifact_root: Path) -> dict[str, JsonValue]:
    """Re-derive every claim a sealed rehearsal artifact makes about itself.

    Run against the artifact this process just published, and runnable later by
    anyone: the manifest is recomputed from the tree, the sealed modes are
    checked, the recorded tolerances are compared against the frozen campaign
    constants BEFORE they are used, the endpoint agreement is re-certified with
    those constants, and the terminal state is re-hashed from the published
    array.  The state hash is compared exactly -- it is a same-source copy, not
    a cross-executable value -- while the objective pair is toleranced.
    """

    manifest = load_canonical_json_bytes(
        (artifact_root / MANIFEST_FILENAME).read_bytes()
    )
    if manifest != artifact_manifest_payload(artifact_root):
        raise RehearsalError("rehearsal manifest differs from the artifact tree")
    validate_sealed_modes(artifact_root)

    evidence = load_canonical_json_bytes(
        (artifact_root / EVIDENCE_FILENAME).read_bytes()
    )
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != REHEARSAL_SCHEMA_VERSION
        or evidence.get("route") != REHEARSAL_ROUTE
    ):
        raise RehearsalError("rehearsal evidence schema differs")
    if not evidence["problem_identity"]["bound"]:
        raise RehearsalError("rehearsal evidence claims an unbound problem")
    if evidence["problem_identity"]["sha_is_binding"]:
        raise RehearsalError("rehearsal evidence binds identity to an unstable sha")
    if not evidence["lowering_pre_gate"]["budget_independent"]:
        raise RehearsalError("rehearsal evidence claims budget-dependent lowering")
    # An artifact may not restate what quality parity is measured on: the
    # pinned set and the informational set are the campaign's, not the run's.
    ledger = evidence["endpoint_ledger"]
    if ledger["pinned_quality_terms"] != list(PINNED_ENDPOINT_QUALITY_TERMS) or (
        ledger["informational_observables"]
        != list(INFORMATIONAL_ENDPOINT_OBSERVABLES)
    ):
        raise RehearsalError("rehearsal endpoint ledger scope differs from the campaign's")

    endpoint = evidence["endpoint_agreement"]
    if (
        endpoint["relative_tolerance"] != DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE
        or endpoint["absolute_floor"] != DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR
    ):
        raise RehearsalError("rehearsal endpoint tolerances differ from the campaign's")
    certify_agreement(
        endpoint["standalone_terminal_objective"],
        endpoint["loop_terminal_objective"],
        "published projected route terminal objective",
        relative_tolerance=DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
        absolute_floor=DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
    )
    with (artifact_root / TERMINAL_COORDINATES_FILENAME).open("rb") as stream:
        coordinates = np.load(stream, allow_pickle=False)
    republished = exact_numeric_tree_sha256(jnp.asarray(coordinates, dtype=jnp.float64))
    if republished != endpoint["terminal_state_sha256"]:
        raise RehearsalError("published terminal state differs from its recorded hash")
    return evidence


def run_bounded_rehearsal(
    output_root: Path,
    *,
    iterations: int = REHEARSAL_MAXIMUM_ITERATIONS,
    repository: Path = REPOSITORY_ROOT,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Run the whole certification chain at a bounded budget and publish it.

    The phases execute in the order a certified run executes them and each one
    fails closed.  A failure keeps the staging root: it is the evidence of what
    the gate saw.
    """

    chain_started = time.perf_counter()
    environment_evidence = validate_environment(
        os.environ if environment is None else environment
    )
    if iterations < 1:
        raise RehearsalError("rehearsal budget must be positive")
    if output_root.exists():
        raise RehearsalError(f"rehearsal output root already exists: {output_root}")
    staging_root = (
        output_root.parent / f"{output_root.name}.partial-{os.urandom(8).hex()}"
    )
    os.mkdir(staging_root, 0o700)

    execution_sources = bind_execution_sources(repository)

    bootstrap_started = time.perf_counter()
    case = BoundCase()
    bootstrap_seconds = time.perf_counter() - bootstrap_started

    identity_started = time.perf_counter()
    problem_identity = bind_problem_identity(case)
    identity_seconds = time.perf_counter() - identity_started

    lowering_started = time.perf_counter()
    lowering = measure_lowering_pre_gate(case, iterations)
    lowering_seconds = time.perf_counter() - lowering_started

    run = run_projected_lbfgs(
        case.raw_joint, case.start, options=rehearsal_options(iterations)
    )
    endpoint = certify_endpoint_agreement(case, run)
    endpoint_ledger = build_endpoint_ledger(
        case,
        run,
        gated=endpoint_ledger_is_gated(
            iterations=iterations, latched=run_latched(run)
        ),
    )
    evidence = build_rehearsal_evidence(
        environment=environment_evidence,
        execution_sources=execution_sources,
        problem_identity=problem_identity,
        lowering=lowering,
        run=run,
        endpoint=endpoint,
        endpoint_ledger=endpoint_ledger,
        iterations=iterations,
        timing_seconds={
            "bootstrap": bootstrap_seconds,
            "problem_identity": identity_seconds,
            "lowering_pre_gate": lowering_seconds,
            "engine_compile": run.compile_seconds,
            "engine_solve": run.solve_seconds,
            "engine_wall": run.compile_seconds + run.solve_seconds,
            "chain_wall": time.perf_counter() - chain_started,
        },
    )
    published = publish_rehearsal(
        staging_root,
        output_root,
        evidence,
        np.asarray(jax.device_get(run.coordinates), dtype=np.float64),
    )
    validate_rehearsal_artifact(published)
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--iterations",
        type=int,
        default=REHEARSAL_MAXIMUM_ITERATIONS,
        help="attempt budget; the rest of the configuration is the certified one",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    published = run_bounded_rehearsal(
        arguments.output_root.resolve(), iterations=arguments.iterations
    )
    evidence = load_canonical_json_bytes((published / EVIDENCE_FILENAME).read_bytes())
    print(
        json.dumps(
            {
                "published_root": str(published),
                "timing_seconds": evidence["timing_seconds"],
                "solve_status": evidence["solve"]["status_name"],
                "iterations_run": evidence["solve"]["iterations_run"],
                "terminal_objective": evidence["solve"]["terminal_objective"],
                "lowering_total_ir_bytes": evidence["lowering_pre_gate"][
                    "total_ir_bytes"
                ],
            },
            indent=1,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = (
    "BOOTSTRAP_BINDING_RELATIVE_TOLERANCES",
    "BOOTSTRAP_FEASIBILITY_ABSOLUTE_TOLERANCE",
    "CERTIFIED_MAXIMUM_ITERATIONS",
    "CERTIFIED_ROUTE_OPTIONS",
    "CPU_BOOTSTRAP_OBSERVABLES",
    "DISTRIBUTION_PACKAGE_ROOTS",
    "EQUAL_MINIMA_PENALTY_SPREAD",
    "EVIDENCE_FILENAME",
    "INFORMATIONAL_ENDPOINT_OBSERVABLES",
    "MANIFEST_FILENAME",
    "NATIVE_ENDPOINT_STATE_CONTENT_SHA256",
    "NATIVE_ENDPOINT_STATE_FILE_SHA256",
    "NATIVE_ENDPOINT_STATE_PATH",
    "NATIVE_TARGET_OBJECTIVE",
    "NATIVE_WALL_SECONDS_BAR",
    "PINNED_ENDPOINT_QUALITY_GATES",
    "PINNED_ENDPOINT_QUALITY_TERMS",
    "REHEARSAL_MANIFEST_SCHEMA_VERSION",
    "REHEARSAL_MAXIMUM_ITERATIONS",
    "REHEARSAL_ROUTE",
    "REHEARSAL_SCHEMA_VERSION",
    "REQUIRED_ENVIRONMENT",
    "TERMINAL_COORDINATES_FILENAME",
    "BoundCase",
    "NativeEndpointState",
    "RehearsalError",
    "artifact_manifest_payload",
    "bind_execution_sources",
    "bind_problem_identity",
    "build_endpoint_ledger",
    "build_rehearsal_evidence",
    "certify_endpoint_agreement",
    "collapse_proximity_margin",
    "endpoint_ledger_is_gated",
    "equal_minima_raw_term_ceiling",
    "gate_endpoint_ledger",
    "iteration_payload",
    "json_scalar",
    "load_execution_source_manifest",
    "load_native_endpoint_state",
    "lowering_payload",
    "main",
    "measure_lowering_pre_gate",
    "publish_rehearsal",
    "refuse_redirected_distribution_modules",
    "rehearsal_options",
    "rename_noreplace",
    "run_bounded_rehearsal",
    "run_latched",
    "seal_and_sync",
    "sha256_hex",
    "validate_environment",
    "validate_rehearsal_artifact",
    "validate_sealed_modes",
)


if __name__ == "__main__":
    raise SystemExit(main())
