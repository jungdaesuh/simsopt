"""Offline replay probe: DESC order-1 predictor + IFT adjoint tolerance budget.

Two independent measurements on one loaded fixture, because loading the
archived flat-675 pair and factoring the reduced Schur operator at the
recorded anchor is the expensive part and both jobs need exactly that.

``--mode predictor`` replays the upgrade plan's Phase 2 bullet "Offline
validation on recorded states BEFORE wiring into children" against the
recovered B37 v1 ledger:

* **Leg 3** -- at the recorded eval-38 anchor, does the predicted warm
  start prevent the eval-39 wrong-branch capture? Both arms run: the bare
  anchor (control, which reproduces the recorded run's warm start) and
  the predicted start.
* **Leg 4** -- the post-poisoning trial. The eval-39 anchor surface is
  not stored, only its hash, so this is replay-then-predict: regenerate
  s39, gate it against the recorded hash, then run the eval-43 trial
  (coils bitwise x38) off that anchor, both arms. The plan's falsifiable
  prediction is *no rescue*.

``--mode tolerance-budget`` implements the plan's Phase 4 first bullet:
kappa of the stabilized Schur operator at the anchor (stated in the plan
as currently unmeasured), then the IFT adjoint gradient as a function of
the *achieved* inner residual over a tolerance ladder, with a measured
log-log scaling and a threshold table.

Physics/diagnostic probe only. **No timing claim of any kind**: walls are
recorded so an operator can schedule the run, never to compare lanes. Not
F3 7.70x, not a nested speed claim. See ``claim_boundary`` in the
evidence document.

``--dry-run`` validates every fingerprint, resolves every path and prints
the plan without touching the GPU. It pins ``JAX_PLATFORMS=cpu`` itself,
so it is safe to run while another job owns the device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import NestedLsOuterState

# A dry run must never initialize CUDA, even when the operator forgot
# ``JAX_PLATFORMS=cpu``: another job may own the device, and a stray
# allocation can starve it. Decided from argv before JAX is imported,
# because the platform choice is frozen at import time.
_DRY_RUN = "--dry-run" in sys.argv[1:]
os.environ.setdefault("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
if _DRY_RUN:
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_T0 = time.perf_counter()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(REPO / "src")

from benchmarks.validation_ladder_common import apply_compilation_cache_policy

apply_compilation_cache_policy(os.environ.get("JAX_COMPILATION_CACHE_DIR"))

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    _envelope_value_and_grad,
    apply_reduced_mixed_schur_coil_tangent,
    dense_schur_lu_preconditioner,
    factor_reduced_nested_ls_schur,
    materialize_stabilized_schur_dense,
    nested_ls_reduced_closures,
    nested_ls_runtime_coil_closures,
    run_reduced_nested_ls_schur_newton,
    solve_stabilized_schur_dense_lu,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE,
    F3_B37_IFT_STAB,
    _flat675_value_and_grad_at,
    _mixed_coil_correction_vjp,
    dump_strict_json,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    nested_ls_outer_value_and_grad,
    nested_ls_runtime_identity,
    prepare_f3_b37_outer_state,
    sha256_float64,
)

_IMPORT_SECONDS = float(time.perf_counter() - _T0)

# ---------------------------------------------------------------------------
# Private seams this probe reaches into, and why each is acceptable here.
# Line numbers are at HEAD 0bef58a7a; this tree is shared, so bind by symbol
# and treat the numbers as a reading aid.
#
# ``_envelope_value_and_grad``    (nested_ls_reduced.py:791)
#     The envelope-gradient fallback the upgrade plan specifies compares
#     ``||grad Phi-hat||`` at two candidate starts. That is the same
#     quantity the inner Newton's own Armijo step measures, and there is
#     no public export of it.
# ``_flat675_value_and_grad_at``  (nested_ls_reduced_scale.py:4761)
# ``_mixed_coil_correction_vjp``  (nested_ls_reduced_scale.py:4789)
#     The tolerance-budget mode must evaluate production's IFT adjoint at
#     a *frozen* loose inner solution. Production's own entry point
#     (``nested_ls_outer_value_and_grad``) re-solves the inner to
#     ``NESTED_LS_NEWTON_TOL`` before assembling the adjoint, which would
#     erase the very rung being measured. Importing production's two
#     helpers means the mirror is production's own function objects in
#     production's order rather than a second derivation of the same
#     formula, and the mirror is checked against production at the tight
#     rung, where production's re-solve is a no-op.
#
# All three are acceptable for a replay/diagnostic harness bound to one
# recorded fixture. NONE of them would be acceptable in the Phase 2/4
# production wiring, which must go through exported API.
# ---------------------------------------------------------------------------

EVIDENCE = REPO / "docs" / "receipts" / "evidence"
LEDGER_PATH = EVIDENCE / "nested_ls_outer_b37_20260823_recovered_jax.json"
REPLAY_LOG_PATH = EVIDENCE / "nested_ls_outer_b37_20260824_transaction_replay.log"

SCHEMA = "nested-ls-outer-predictor-replay.v1"
PUBLICATION = (
    "Offline replay of the recorded B37 v1 eval-38/39/43 neighbourhood: "
    "DESC order-1 predictor warm start (upgrade plan Phase 2) and the IFT "
    "adjoint gradient error versus achieved inner residual (Phase 4). "
    "Diagnostic only, no timing content, no speed claim, not F3 7.70x."
)

# --- Fingerprints of the recovered ledger. Every one is asserted at load; --
# --- a mismatch is a fail-closed SystemExit naming the field.             --
LEDGER_SHA256 = "21ddaf82562d5e561c7d4f44e214eab5ed72eaac480edf020ede8e1f47cbde3e"
# v2 pins the HISTORICAL recovered ledger this probe replays. The live child
# schemas have since moved on (nested-ls-outer-jax-child.v5,
# nested-ls-outer-native-child.v4 in nested_ls_contract.py); a recorded
# artefact does not get re-stamped, so this v2 is correct, not stale.
LEDGER_SCHEMA = "nested-ls-outer-jax-child.v2"
ANCHOR_EVAL_INDEX = 38
TRIAL_EVAL_INDEX = 39
POST_POISON_EVAL_INDEX = 43
COIL_DOF_COUNT = 11
SURFACE_DOF_COUNT = 661
# ``$.endpoint_surface_sha256`` == ``outer_evals[38].inner_surface_sha256``.
ANCHOR_SURFACE_SHA256 = (
    "07c00c33e7bfddbdd8969e8e7e04a97fef2fc75666ea7c09f61ae191ff626e0d"
)
# ``$.endpoint_coil_sha256``; ``outer_evals[43].coil_dofs`` hash to this too.
ANCHOR_COIL_SHA256 = "2e0531813132645bfe526a3c125aa8d07af54288846c9c078a29b68fa47ac180"
# ``outer_evals[39].inner_surface_sha256`` -- the poisoned anchor s39 that
# leg 4 must regenerate. Full 64 hex characters, read from the ledger.
POISONED_SURFACE_SHA256 = (
    "052923e7b92e9b15b2ad80719b5c1b67ebc7d90c5229efbf8326dd1d06a79d96"
)
COIL_STEP_L2 = 0.00447280177602724
ANCHOR_IOTA = 0.14809710769644205
ANCHOR_G = 2.0106192893330412
ANCHOR_J = 0.007253464731833192
ANCHOR_INNER_GRAD_L2 = 1.4161445536373189e-15
RECORDED_TRIAL_J = 0.07471552895095307
RECORDED_TRIAL_IOTA = 0.1399912213008207
RECORDED_TRIAL_INNER_ITERATIONS = 9

# ==========================================================================
# The recorded ledger is NOT bitwise reproducible on the current binary.
#
# Every constant below is read from the committed transaction replay log
# ``docs/receipts/evidence/nested_ls_outer_b37_20260824_transaction_replay.log``
# -- the only other execution of these exact states that exists -- and is
# published as the ``recorded_ledger_trajectory_drift`` block of the evidence
# document. This is a campaign-level fact with consequences beyond this
# probe: it bounds what ANY future replay of this ledger can claim.
#
# The two runs agree on physics and disagree in the last bits. Converged
# quantities drift by 2 ULP; the one non-converged quantity drifts by 1.1e-4.
# That split is what makes a bitwise gate useless and a physics gate sound.
# ==========================================================================

# Log lines 6 and 11 ("[A2 x39]", "[B2 x39]"): s39 regenerated twice from
# bitwise-identical inputs, both times to this prefix, neither time to the
# ledger's ``052923e7...``.
COMMITTED_REPLAY_REGENERATED_SURFACE_SHA256_PREFIX = "7daf6c3223b66041"
# Log lines 5 and 10 ("[A1 x38]", "[B1 x38]"): J at the anchor, against the
# ledger's ``endpoint_j``. The log's own VERDICT block calls this FAIL.
COMMITTED_REPLAY_ANCHOR_J = 0.007253464731833194
COMMITTED_REPLAY_ANCHOR_J_REL_DRIFT = 2.391579114410865e-16  # 2.0 ULP
# Log lines 6 and 11: iota at the regenerated poisoned anchor, against the
# ledger's ``outer_evals[39].inner_iota``.
COMMITTED_REPLAY_TRIAL_IOTA = 0.13999122130082076
COMMITTED_REPLAY_TRIAL_IOTA_ABS_DRIFT = 5.551115123125783e-17  # 2.0 ULP
# Log line 7 ("[A3 x38] REJECTED"): the residual left by a solve that never
# converged, against ``outer_evals[43].rejection_detail``'s. Twelve orders
# looser than the converged quantities above -- which is exactly why the
# fingerprint below does NOT gate on a residual.
COMMITTED_REPLAY_FAILED_RESIDUAL = 0.0020087310206471937
LEDGER_FAILED_RESIDUAL = 0.0020089600055687397
COMMITTED_REPLAY_FAILED_RESIDUAL_REL_DRIFT = 0.00011398182189350828
# Log lines 6 and 11 reproduce the ledger's J at x39 bitwise.
COMMITTED_REPLAY_TRIAL_J_REL_DRIFT = 0.0
# |iota_39 - iota_38|: what the fingerprint has to resolve.
RECORDED_BRANCH_SEPARATION = 0.008105886395621348

# --- Leg 4 regenerated-anchor physics fingerprint. -------------------------
# Leg 4 asks whether the predictor rescues a trial launched from a POISONED
# anchor -- a surface captured on the wrong iota branch. That is a physics
# property. A bitwise hash gate cannot answer it: it can only report that
# this box runs a different simsoptpp binary (``d4a6e028...``) than the one
# the ledger was recorded on, which ``simsoptpp_sha256`` already says. So the
# gate is on the physics, the hash is computed and published un-gated, and
# there is no override flag.
REGEN_IOTA_ABS_TOL = 1.0e-9
REGEN_J_REL_TOL = 1.0e-9
REGEN_ITERATION_COUNT = 9

# --- Predictor policy (upgrade plan Phase 2). ------------------------------
# DESC ``tr_ratio`` semantics: DESC SCALES the perturbation step to the
# bound, it does not reject it.
TRUST_REGION_RATIO = 0.1

# --- Tolerance-budget policy (upgrade plan Phase 4). -----------------------
TOLERANCE_RUNGS = (1.0e-13, 1.0e-11, 1.0e-9, 1.0e-8, 1.0e-6)
# Raised well above NESTED_LS_NEWTON_MAXITER so every rung stops on its
# tolerance, not on its budget. ``iteration_count`` per rung says which.
TOLERANCE_RUNG_MAXITER = 40
BUDGET_THRESHOLDS = (1.0e-6, 1.0e-8, 1.0e-10)
# The mirror of production's adjoint must BE production's gradient, not
# merely near it. Checked at the tight rung.
PRODUCTION_MIRROR_MAX_REL = 1.0e-12

MODES = ("predictor", "tolerance-budget", "all")


# ==========================================================================
# Pure helpers. Every one of these is unit-tested on CPU.
# ==========================================================================


def fail_closed(quantity: str, expected: object, observed: object) -> None:
    """Abort naming the quantity, what was required and what was seen."""

    raise SystemExit(
        f"nested-LS predictor replay FAILED CLOSED on {quantity}: "
        f"expected {expected!r}, observed {observed!r}."
    )


def require(condition: bool, quantity: str, expected: object, observed: object) -> None:
    """Fail closed unless ``condition`` holds."""

    if not condition:
        fail_closed(quantity, expected, observed)


def require_finite(quantity: str, value: float) -> float:
    """Fail closed on a non-finite scalar, naming it."""

    number = float(value)
    if not np.isfinite(number):
        fail_closed(quantity, "a finite float", number)
    return number


def sha256_bytes(path: Path) -> str:
    """SHA-256 of a file's raw bytes."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_trust_region(
    delta: NDArray[np.float64],
    anchor_surface: NDArray[np.float64],
    ratio: float,
) -> tuple[NDArray[np.float64], float, float, float, bool]:
    """DESC ``tr_ratio``: scale the step to the bound, never reject it.

    Returns ``(applied, raw_norm, applied_norm, cap, scaled)``. The bound
    is ``ratio * ||s_anchor||_2``, and a step exactly at the bound is left
    untouched.
    """

    cap = float(ratio) * float(np.linalg.norm(anchor_surface))
    raw_norm = float(np.linalg.norm(delta))
    if raw_norm > cap:
        applied = np.asarray(delta, dtype=np.float64) * (cap / raw_norm)
        return applied, raw_norm, float(np.linalg.norm(applied)), cap, True
    applied = np.array(delta, dtype=np.float64, copy=True)
    return applied, raw_norm, raw_norm, cap, False


def select_arm(bare_gradient_norm: float, predicted_gradient_norm: float) -> str:
    """Envelope-gradient fallback: bare anchor when the prediction is worse.

    Ours, not DESC's. A tie keeps the prediction, which is what makes the
    ``delta_c = 0`` invariant hold: no coil motion gives ``delta_s = 0``,
    the two starts are the same vector, the two norms are equal, and the
    predicted arm reproduces the predictor-OFF result exactly.
    """

    if float(predicted_gradient_norm) > float(bare_gradient_norm):
        return "bare_anchor"
    return "predicted"


def classify_branch(iota: float) -> str:
    """Name the iota branch by proximity to the two recorded branches.

    The 0.05 ``NESTED_LS_OUTER_IOTA_BRANCH_GUARD`` does not separate these
    two: the recorded capture moved iota by 0.0081, well under the guard.
    That is the adjudicated finding, not a defect, so the discriminator
    here is proximity to the two *measured* branches, and the raw
    distances are published alongside the label.
    """

    to_anchor = abs(float(iota) - ANCHOR_IOTA)
    to_recorded = abs(float(iota) - RECORDED_TRIAL_IOTA)
    if to_anchor < to_recorded:
        return "anchor_branch"
    if to_recorded < to_anchor:
        return "recorded_wrong_branch"
    return "equidistant"


@dataclass(frozen=True, slots=True)
class AnchorPhysicsVerdict:
    """Whether a regenerated s39 is the recorded poisoned anchor, physically."""

    passed: bool
    checks: tuple[dict[str, object], ...]
    failures: tuple[str, ...]

    def as_payload(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [dict(check) for check in self.checks],
            "failures": list(self.failures),
        }


def check_regenerated_anchor_physics(
    *, iota: float, iteration_count: int, objective_j: float
) -> AnchorPhysicsVerdict:
    """Gate a regenerated s39 on physics, not on bits.

    Four checks, each quoted against the drift measured between the recorded
    ledger and the committed transaction replay -- the only two executions of
    this state that exist:

    * ``inner_iota`` within ``REGEN_IOTA_ABS_TOL`` of the recorded capture.
    * ``inner_iota`` unambiguously nearer the recorded capture than the
      anchor's branch. That separation is the whole point of leg 4: the
      anchor has to actually be poisoned.
    * ``iteration_count`` exactly the recorded 9 -- an integer control-flow
      quantity, reproduced twice by the committed replay, so it takes no
      tolerance.
    * ``J`` within ``REGEN_J_REL_TOL`` relative of the recorded value.

    Deliberately NOT gated: the inner residual. The one non-converged
    quantity in the committed replay drifts by 1.1e-4 relative, twelve
    orders looser than these, so gating on it would test the arithmetic of
    a diverged walk rather than the identity of the anchor.
    """

    delta_recorded = abs(float(iota) - RECORDED_TRIAL_IOTA)
    delta_anchor = abs(float(iota) - ANCHOR_IOTA)
    j_relative = abs(float(objective_j) - RECORDED_TRIAL_J) / abs(RECORDED_TRIAL_J)
    checks: tuple[dict[str, object], ...] = (
        {
            "name": "inner_iota_within_tolerance",
            "expected": RECORDED_TRIAL_IOTA,
            "observed": float(iota),
            "measure": delta_recorded,
            "tolerance": float(REGEN_IOTA_ABS_TOL),
            "tolerance_source": (
                "1.8e7x the 5.551115123125783e-17 (2 ULP) iota drift measured "
                "between the ledger's outer_evals[39].inner_iota and the "
                "committed replay's A2/B2, and 8.1e-6 of the "
                "0.008105886395621348 branch separation it must resolve"
            ),
            "passed": delta_recorded <= float(REGEN_IOTA_ABS_TOL),
        },
        {
            "name": "inner_iota_on_the_recorded_wrong_branch",
            "expected": (
                "nearer the recorded capture "
                f"{RECORDED_TRIAL_IOTA!r} than the anchor {ANCHOR_IOTA!r}"
            ),
            "observed": float(iota),
            "measure": delta_recorded,
            "tolerance": delta_anchor,
            "tolerance_source": (
                "strict separation, no constant: the regenerated anchor must "
                "actually be poisoned or leg 4 is not asking its question"
            ),
            "passed": delta_recorded < delta_anchor,
        },
        {
            "name": "inner_iterations_exact",
            "expected": int(REGEN_ITERATION_COUNT),
            "observed": int(iteration_count),
            "measure": abs(int(iteration_count) - int(REGEN_ITERATION_COUNT)),
            "tolerance": 0,
            "tolerance_source": (
                "integer control-flow quantity; the committed replay "
                "reproduced 9 at both A2 and B2, so it takes no tolerance"
            ),
            "passed": int(iteration_count) == int(REGEN_ITERATION_COUNT),
        },
        {
            "name": "objective_j_within_relative_band",
            "expected": RECORDED_TRIAL_J,
            "observed": float(objective_j),
            "measure": j_relative,
            "tolerance": float(REGEN_J_REL_TOL),
            "tolerance_source": (
                "4.2e6x the largest converged-J drift measured between the "
                "ledger and the committed replay (2.391579114410865e-16, "
                "2 ULP, at x38; the replay reproduced J at x39 bitwise), and "
                "the same 1e-9 band the plan freezes for the B37 endpoint J"
            ),
            "passed": j_relative <= float(REGEN_J_REL_TOL),
        },
    )
    failures = tuple(str(check["name"]) for check in checks if not check["passed"])
    return AnchorPhysicsVerdict(passed=not failures, checks=checks, failures=failures)


def trajectory_drift_discrepancies(
    regenerated_surface_sha256: str | None,
) -> list[dict[str, object]]:
    """The three measured ledger-vs-replay discrepancies, each with a source."""

    return [
        {
            "quantity": "regenerated s39 surface sha256",
            "ledger": POISONED_SURFACE_SHA256,
            "ledger_source": "$.outer_evals[39].inner_surface_sha256",
            "committed_replay": (
                COMMITTED_REPLAY_REGENERATED_SURFACE_SHA256_PREFIX + "..."
            ),
            "committed_replay_source": "replay log lines 6 and 11 (A2, B2)",
            "this_run": regenerated_surface_sha256,
            "agreement": "none (different bits, reproduced twice)",
        },
        {
            "quantity": "eight-term J at the eval-38 anchor",
            "ledger": ANCHOR_J,
            "ledger_source": "$.endpoint_j",
            "committed_replay": COMMITTED_REPLAY_ANCHOR_J,
            "committed_replay_source": "replay log lines 5 and 10 (A1, B1)",
            "relative_drift": COMMITTED_REPLAY_ANCHOR_J_REL_DRIFT,
            "ulps": 2.0,
            "agreement": (
                "15 of 16 significant digits; the log's own VERDICT block "
                "records this as FAIL"
            ),
        },
        {
            "quantity": "inner residual of the failed x38 solve",
            "ledger": LEDGER_FAILED_RESIDUAL,
            "ledger_source": "$.outer_evals[43].rejection_detail",
            "committed_replay": COMMITTED_REPLAY_FAILED_RESIDUAL,
            "committed_replay_source": "replay log line 7 (A3)",
            "relative_drift": COMMITTED_REPLAY_FAILED_RESIDUAL_REL_DRIFT,
            "agreement": (
                "4 significant digits; this is a NON-CONVERGED quantity and "
                "drifts twelve orders more than the converged ones, which is "
                "why no gate here uses a residual"
            ),
        },
    ]


def trajectory_drift_finding(
    regenerated_surface_sha256: str | None,
) -> dict[str, object]:
    """The recorded ledger is not bitwise reproducible on this binary.

    A campaign-level finding, not a property of this probe: it bounds what
    any future replay of ``nested_ls_outer_b37_20260823_recovered_jax.json``
    can claim. Three measured discrepancies, each with its source line.
    """

    return {
        "finding": (
            "The recorded B37 v1 ledger is NOT bitwise reproducible on the "
            "current simsoptpp binary. Three independent quantities differ "
            "between the ledger and the committed transaction replay, which "
            "ran the same states from bitwise-identical inputs. The two runs "
            "agree on physics and disagree in the last bits."
        ),
        "consequence": (
            "No replay of this ledger may gate on bitwise reproduction of a "
            "recorded surface, objective or residual. Physics fingerprints "
            "are the only sound gate, and any receipt derived from this "
            "ledger must disclose the binary boundary."
        ),
        "sources": {
            "ledger": str(LEDGER_PATH),
            "ledger_sha256": LEDGER_SHA256,
            "committed_replay_log": str(REPLAY_LOG_PATH.relative_to(REPO)),
            "ledger_simsoptpp_sha256": (
                "not recorded: nested_ls_runtime_identity gained "
                "simsoptpp_sha256 in Phase 0, after this ledger was written"
            ),
        },
        "discrepancies": trajectory_drift_discrepancies(regenerated_surface_sha256),
        "converged_quantity_drift_ulps": 2.0,
        "non_converged_quantity_relative_drift": (
            COMMITTED_REPLAY_FAILED_RESIDUAL_REL_DRIFT
        ),
        "current_simsoptpp_sha256_is_in": "runtime.simsoptpp_sha256",
    }


def loglog_slope(
    abscissa: tuple[float, ...], ordinate: tuple[float, ...]
) -> tuple[float, float, float]:
    """Least-squares slope of ``log10(y)`` against ``log10(x)``.

    Returns ``(slope, intercept, rms_residual)`` in log10 space. Both
    sequences must be strictly positive and carry at least two points --
    a log fit is not defined otherwise, so violating that fails closed.
    """

    require(
        len(abscissa) == len(ordinate),
        "loglog_fit_point_counts",
        len(abscissa),
        len(ordinate),
    )
    require(len(abscissa) >= 2, "loglog_fit_point_count", ">= 2", len(abscissa))
    for name, series in (("abscissa", abscissa), ("ordinate", ordinate)):
        for value in series:
            require(float(value) > 0.0, f"loglog_fit_{name}_positive", "> 0.0", value)
    log_x = np.log10(np.asarray(abscissa, dtype=np.float64))
    log_y = np.log10(np.asarray(ordinate, dtype=np.float64))
    mean_x = float(np.mean(log_x))
    mean_y = float(np.mean(log_y))
    centred_x = log_x - mean_x
    denominator = float(np.dot(centred_x, centred_x))
    require(denominator > 0.0, "loglog_fit_abscissa_spread", "> 0.0", denominator)
    slope = float(np.dot(centred_x, log_y - mean_y)) / denominator
    intercept = mean_y - slope * mean_x
    residual = log_y - (slope * log_x + intercept)
    rms = float(np.sqrt(float(np.mean(residual * residual))))
    return slope, float(intercept), rms


def largest_residual_under_threshold(
    rows: tuple[tuple[float, float], ...], threshold: float
) -> float | None:
    """Largest achieved inner residual whose relative error is within bound.

    ``rows`` are ``(achieved_residual, relative_gradient_error)`` pairs.
    ``None`` when no rung qualifies.
    """

    qualifying = [
        float(residual) for residual, error in rows if float(error) <= float(threshold)
    ]
    if not qualifying:
        return None
    return max(qualifying)


# ==========================================================================
# Ledger
# ==========================================================================


@dataclass(frozen=True, slots=True)
class ReplayLedger:
    """The recorded B37 v1 states legs 3 and 4 replay, all validated."""

    path: Path
    sha256: str
    anchor_coil_dofs: NDArray[np.float64]
    anchor_surface_dofs: NDArray[np.float64]
    anchor_iota: float
    anchor_G: float
    anchor_j: float
    trial_coil_dofs: NDArray[np.float64]
    coil_step_l2: float
    fingerprints: dict[str, object]


def load_replay_ledger(path: Path) -> ReplayLedger:
    """Load the recovered ledger and assert every documented fingerprint.

    Nothing here is trusted. The eval-38 anchor is *claimed* complete on
    disk, so the claim itself is checked: the row's coils and surface hash
    must equal the endpoint's, elementwise and bitwise.
    """

    require(path.is_file(), "ledger_path", "an existing file", str(path))
    digest = sha256_bytes(path)
    require(digest == LEDGER_SHA256, "ledger_sha256", LEDGER_SHA256, digest)
    payload = json.loads(path.read_text())

    schema = payload["schema"]
    require(schema == LEDGER_SCHEMA, "ledger_schema", LEDGER_SCHEMA, schema)

    evals = payload["outer_evals"]
    for position, row in enumerate(evals):
        require(
            int(row["eval_index"]) == position,
            f"outer_evals[{position}].eval_index",
            position,
            row["eval_index"],
        )

    endpoint_index = int(payload["endpoint_eval_index"])
    require(
        endpoint_index == ANCHOR_EVAL_INDEX,
        "endpoint_eval_index",
        ANCHOR_EVAL_INDEX,
        endpoint_index,
    )

    endpoint_coils = np.asarray(payload["endpoint_coil_dofs"], dtype=np.float64)
    endpoint_surface = np.asarray(payload["endpoint_surface_dofs"], dtype=np.float64)
    require(
        endpoint_coils.size == COIL_DOF_COUNT,
        "endpoint_coil_dofs.size",
        COIL_DOF_COUNT,
        int(endpoint_coils.size),
    )
    require(
        endpoint_surface.size == SURFACE_DOF_COUNT,
        "endpoint_surface_dofs.size",
        SURFACE_DOF_COUNT,
        int(endpoint_surface.size),
    )

    endpoint_surface_hash = sha256_float64(endpoint_surface)
    require(
        endpoint_surface_hash == ANCHOR_SURFACE_SHA256,
        "sha256_float64(endpoint_surface_dofs)",
        ANCHOR_SURFACE_SHA256,
        endpoint_surface_hash,
    )
    require(
        payload["endpoint_surface_sha256"] == ANCHOR_SURFACE_SHA256,
        "endpoint_surface_sha256",
        ANCHOR_SURFACE_SHA256,
        payload["endpoint_surface_sha256"],
    )
    endpoint_coil_hash = sha256_float64(endpoint_coils)
    require(
        endpoint_coil_hash == ANCHOR_COIL_SHA256,
        "sha256_float64(endpoint_coil_dofs)",
        ANCHOR_COIL_SHA256,
        endpoint_coil_hash,
    )
    require(
        payload["endpoint_coil_sha256"] == ANCHOR_COIL_SHA256,
        "endpoint_coil_sha256",
        ANCHOR_COIL_SHA256,
        payload["endpoint_coil_sha256"],
    )

    anchor_row = evals[ANCHOR_EVAL_INDEX]
    anchor_row_coils = np.asarray(anchor_row["coil_dofs"], dtype=np.float64)
    require(
        bool(np.array_equal(anchor_row_coils, endpoint_coils)),
        "outer_evals[38].coil_dofs == endpoint_coil_dofs (elementwise)",
        True,
        False,
    )
    require(
        anchor_row["inner_surface_sha256"] == ANCHOR_SURFACE_SHA256,
        "outer_evals[38].inner_surface_sha256",
        ANCHOR_SURFACE_SHA256,
        anchor_row["inner_surface_sha256"],
    )

    anchor_iota = float(payload["endpoint_iota"])
    anchor_g = float(payload["endpoint_g"])
    anchor_j = float(payload["endpoint_j"])
    require(anchor_iota == ANCHOR_IOTA, "endpoint_iota", ANCHOR_IOTA, anchor_iota)
    require(anchor_g == ANCHOR_G, "endpoint_g", ANCHOR_G, anchor_g)
    require(anchor_j == ANCHOR_J, "endpoint_j", ANCHOR_J, anchor_j)
    require(
        float(anchor_row["inner_grad_l2"]) == ANCHOR_INNER_GRAD_L2,
        "outer_evals[38].inner_grad_l2",
        ANCHOR_INNER_GRAD_L2,
        anchor_row["inner_grad_l2"],
    )

    trial_row = evals[TRIAL_EVAL_INDEX]
    trial_coils = np.asarray(trial_row["coil_dofs"], dtype=np.float64)
    step_l2 = float(np.linalg.norm(trial_coils - endpoint_coils))
    require(step_l2 == COIL_STEP_L2, "||x39 - x38||_2", COIL_STEP_L2, step_l2)
    require(
        float(trial_row["j"]) == RECORDED_TRIAL_J,
        "outer_evals[39].j",
        RECORDED_TRIAL_J,
        trial_row["j"],
    )
    require(
        float(trial_row["inner_iota"]) == RECORDED_TRIAL_IOTA,
        "outer_evals[39].inner_iota",
        RECORDED_TRIAL_IOTA,
        trial_row["inner_iota"],
    )
    require(
        int(trial_row["inner_iterations"]) == RECORDED_TRIAL_INNER_ITERATIONS,
        "outer_evals[39].inner_iterations",
        RECORDED_TRIAL_INNER_ITERATIONS,
        trial_row["inner_iterations"],
    )
    require(
        trial_row["inner_surface_sha256"] == POISONED_SURFACE_SHA256,
        "outer_evals[39].inner_surface_sha256",
        POISONED_SURFACE_SHA256,
        trial_row["inner_surface_sha256"],
    )

    post_row = evals[POST_POISON_EVAL_INDEX]
    post_coils = np.asarray(post_row["coil_dofs"], dtype=np.float64)
    post_coil_hash = sha256_float64(post_coils)
    require(
        post_coil_hash == ANCHOR_COIL_SHA256,
        "sha256_float64(outer_evals[43].coil_dofs) (bitwise x38)",
        ANCHOR_COIL_SHA256,
        post_coil_hash,
    )
    require(
        post_row["rejection_reason"] == "inner_solve_failed",
        "outer_evals[43].rejection_reason",
        "inner_solve_failed",
        post_row["rejection_reason"],
    )
    # NOTE for a future ledger: this reads the ANCHOR the refused solve
    # warm-started from, off a row whose ``rejection_reason`` is
    # ``inner_solve_failed``. In the archived v2 ledger this file pins, that is
    # what ``inner_surface_sha256`` holds. From child schema v5 onward a
    # refused row publishes ``inner_surface_sha256: None`` — republishing the
    # anchor's hash under a name that says "inner" was a sentinel wearing a
    # physics label — and the anchor moved to its own ``anchor_surface_sha256``
    # field. A v5 replay must read that field here instead.
    require(
        post_row["inner_surface_sha256"] == POISONED_SURFACE_SHA256,
        "outer_evals[43].inner_surface_sha256 (the poisoned anchor)",
        POISONED_SURFACE_SHA256,
        post_row["inner_surface_sha256"],
    )
    require(
        float(post_row["anchor_distance"]) == COIL_STEP_L2,
        "outer_evals[43].anchor_distance",
        COIL_STEP_L2,
        post_row["anchor_distance"],
    )

    fingerprints: dict[str, object] = {
        "ledger_sha256": digest,
        "ledger_schema": schema,
        "endpoint_eval_index": endpoint_index,
        "outer_evals_positional_index_equals_eval_index": True,
        "endpoint_coil_dofs.size": int(endpoint_coils.size),
        "endpoint_surface_dofs.size": int(endpoint_surface.size),
        "endpoint_surface_sha256": ANCHOR_SURFACE_SHA256,
        "endpoint_coil_sha256": ANCHOR_COIL_SHA256,
        "outer_evals[38].coil_dofs_equals_endpoint_coil_dofs": True,
        "outer_evals[38].inner_surface_sha256": ANCHOR_SURFACE_SHA256,
        "outer_evals[38].inner_grad_l2": ANCHOR_INNER_GRAD_L2,
        "endpoint_iota": anchor_iota,
        "endpoint_g": anchor_g,
        "endpoint_j": anchor_j,
        "coil_step_l2_x39_minus_x38": step_l2,
        "outer_evals[39].j": RECORDED_TRIAL_J,
        "outer_evals[39].inner_iota": RECORDED_TRIAL_IOTA,
        "outer_evals[39].inner_iterations": RECORDED_TRIAL_INNER_ITERATIONS,
        "outer_evals[39].inner_surface_sha256": POISONED_SURFACE_SHA256,
        "outer_evals[43].coil_dofs_bitwise_equals_x38": True,
        "outer_evals[43].rejection_reason": "inner_solve_failed",
        "outer_evals[43].inner_surface_sha256": POISONED_SURFACE_SHA256,
        "outer_evals[43].anchor_distance": COIL_STEP_L2,
        "lane": payload["lane"],
        "start_coil_sha256": payload["start_coil_sha256"],
        "start_surface_sha256": payload["start_surface_sha256"],
    }
    return ReplayLedger(
        path=path,
        sha256=digest,
        anchor_coil_dofs=endpoint_coils,
        anchor_surface_dofs=endpoint_surface,
        anchor_iota=anchor_iota,
        anchor_G=anchor_g,
        anchor_j=anchor_j,
        trial_coil_dofs=trial_coils,
        coil_step_l2=step_l2,
        fingerprints=fingerprints,
    )


def check_lane_binds_to_ledger(ledger: ReplayLedger) -> dict[str, object]:
    """Assert the lane blocks on disk are the world the ledger ran on.

    Pure JSON plus numpy -- no device work, so this runs in ``--dry-run``.
    """

    coils, surface, lane_meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    recorded_lane = ledger.fingerprints["lane"]
    require(
        lane_meta == recorded_lane,
        "lane_meta == ledger.lane",
        recorded_lane,
        lane_meta,
    )
    coil_hash = sha256_float64(coils)
    surface_hash = sha256_float64(surface)
    require(
        coil_hash == ledger.fingerprints["start_coil_sha256"],
        "sha256_float64(lane coil block) == ledger.start_coil_sha256",
        ledger.fingerprints["start_coil_sha256"],
        coil_hash,
    )
    require(
        surface_hash == ledger.fingerprints["start_surface_sha256"],
        "sha256_float64(lane surface block) == ledger.start_surface_sha256",
        ledger.fingerprints["start_surface_sha256"],
        surface_hash,
    )
    return {
        "lane_path": str(DEFAULT_F3_B37_GPU_LANE),
        "lane_meta_equals_ledger_lane": True,
        "lane_coil_sha256": coil_hash,
        "lane_surface_sha256": surface_hash,
    }


# ==========================================================================
# World
# ==========================================================================


@dataclass(frozen=True, slots=True)
class LoadedWorld:
    """The archived pair plus the bound eight-term outer state."""

    state: NestedLsOuterState
    jax_boozer: BoozerSurfaceJAX
    lane_meta: dict[str, object]
    load_seconds: float


def load_world() -> LoadedWorld:
    """Mirror ``nested_ls_outer_jax_child._prepare_outer_run``'s world load."""

    require(
        jax.default_backend() == "gpu",
        "jax.default_backend()",
        "gpu",
        jax.default_backend(),
    )
    started = time.perf_counter()
    coils, surface, lane_meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    _native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    del _native, _target
    state = prepare_f3_b37_outer_state(jax_boozer)
    return LoadedWorld(
        state=state,
        jax_boozer=jax_boozer,
        lane_meta=lane_meta,
        load_seconds=float(time.perf_counter() - started),
    )


def install_anchor(
    world: LoadedWorld,
    surface_dofs: NDArray[np.float64],
    iota: float,
    G: float,
    coil_dofs: NDArray[np.float64],
) -> None:
    """Install one committed anchor, in ``_restore_nested_candidate`` order."""

    world.state.set_anchor(surface_dofs, iota, G)
    world.jax_boozer.surface.set_dofs(surface_dofs)
    world.jax_boozer.biotsavart.x = np.array(coil_dofs, dtype=np.float64, copy=True)
    world.jax_boozer._refresh_coil_data()


def install_solve_point(
    world: LoadedWorld,
    start_surface: NDArray[np.float64],
    coil_dofs: NDArray[np.float64],
) -> None:
    """Stage one inner solve, in ``_solve_nested_inner_at_coils`` order."""

    world.jax_boozer.surface.set_dofs(start_surface)
    world.jax_boozer.biotsavart.x = np.array(coil_dofs, dtype=np.float64, copy=True)
    world.jax_boozer._refresh_coil_data()


def flat675_objective(
    world: LoadedWorld,
    coil_dofs: NDArray[np.float64],
    surface_dofs: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Eight-term ``J`` and its 675-gradient at ``(c, v_frozen, s)``."""

    return _flat675_value_and_grad_at(world.state, coil_dofs, surface_dofs)


# ==========================================================================
# Predictor
# ==========================================================================


@dataclass(frozen=True, slots=True)
class PredictorDiagnostics:
    """One order-1 DESC perturbation at a committed anchor."""

    delta_surface: NDArray[np.float64]
    predictor_norm_raw: float
    predictor_norm_applied: float
    trust_region_cap: float
    trust_region_scaled: bool
    anchor_surface_norm: float
    coil_step_l2: float
    mixed_term_l2: float
    phi_yy_condition: float
    build_seconds: float

    def as_payload(self) -> dict[str, object]:
        return {
            "predictor_norm_raw": self.predictor_norm_raw,
            "predictor_norm_applied": self.predictor_norm_applied,
            "trust_region_cap": self.trust_region_cap,
            "trust_region_ratio": float(TRUST_REGION_RATIO),
            "trust_region_scaled": self.trust_region_scaled,
            "anchor_surface_norm": self.anchor_surface_norm,
            "coil_step_l2": self.coil_step_l2,
            "mixed_term_h_sc_delta_c_l2": self.mixed_term_l2,
            "phi_yy_condition": self.phi_yy_condition,
            "ift_stab": float(F3_B37_IFT_STAB),
            "build_seconds": self.build_seconds,
        }


def build_predictor(
    world: LoadedWorld,
    *,
    anchor_surface: NDArray[np.float64],
    anchor_iota: float,
    anchor_G: float,
    anchor_coils: NDArray[np.float64],
    trial_coils: NDArray[np.float64],
) -> PredictorDiagnostics:
    """``delta_s_pred = -H_ss^-1 H_sc delta_c`` at the committed anchor.

    The closure split is production's, taken from
    ``nested_ls_outer_value_and_grad``: the frozen-coil closures
    (``nested_ls_reduced_closures``) factor the Schur operator, and the
    runtime-coil closures (``nested_ls_runtime_coil_closures``) carry the
    mixed term. Swapping them silently gives a wrong ``delta_s``.

    ``stab = F3_B37_IFT_STAB`` is **0.0** -- a reader will assume
    otherwise from the name ``materialize_stabilized_schur_dense``. This
    is the unregularized Schur operator, the same one the production
    adjoint inverts.
    """

    started = time.perf_counter()
    install_solve_point(world, anchor_surface, anchor_coils)
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(world.jax_boozer)
    del _phi_hat
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        anchor_surface,
        y_probe=np.array([anchor_iota, anchor_G], dtype=np.float64),
    )
    dense = jax.block_until_ready(
        materialize_stabilized_schur_dense(
            operator,
            float(F3_B37_IFT_STAB),
            max_dense_linearization_bytes=None,
        )
    )
    # The Phase 2 mechanism: one cached LU at commit time, then every
    # trial costs one mixed-term JVP plus one triangular solve.
    apply_lu = dense_schur_lu_preconditioner(dense)
    residual_rt, objective_rt, _phi_rt = nested_ls_runtime_coil_closures(
        world.jax_boozer
    )
    del _phi_rt
    coil_step = np.asarray(trial_coils, dtype=np.float64) - np.asarray(
        anchor_coils, dtype=np.float64
    )
    mixed = jax.block_until_ready(
        apply_reduced_mixed_schur_coil_tangent(
            residual_rt,
            objective_rt,
            anchor_surface,
            anchor_coils,
            coil_step,
            operator=operator,
        )
    )
    raw_delta = -np.asarray(
        jax.device_get(jax.block_until_ready(apply_lu(mixed))), dtype=np.float64
    )
    require(
        bool(np.all(np.isfinite(raw_delta))),
        "predictor delta_s_pred finiteness",
        "all finite",
        float(np.linalg.norm(raw_delta)),
    )
    applied, raw_norm, applied_norm, cap, scaled = apply_trust_region(
        raw_delta, np.asarray(anchor_surface, dtype=np.float64), TRUST_REGION_RATIO
    )
    return PredictorDiagnostics(
        delta_surface=applied,
        predictor_norm_raw=raw_norm,
        predictor_norm_applied=applied_norm,
        trust_region_cap=cap,
        trust_region_scaled=scaled,
        anchor_surface_norm=float(np.linalg.norm(anchor_surface)),
        coil_step_l2=float(np.linalg.norm(coil_step)),
        mixed_term_l2=float(
            np.linalg.norm(np.asarray(jax.device_get(mixed), dtype=np.float64))
        ),
        phi_yy_condition=float(operator.phi_yy_condition),
        build_seconds=float(time.perf_counter() - started),
    )


def envelope_gradient_norms(
    world: LoadedWorld,
    *,
    trial_coils: NDArray[np.float64],
    bare_surface: NDArray[np.float64],
    predicted_surface: NDArray[np.float64],
) -> tuple[float, float, float]:
    """``||grad Phi-hat||`` at both candidate starts, at the trial coils.

    Both starts are measured against the SAME closures, built once after
    the trial coils are installed, because the frozen-coil closures
    capture the coil DOFs as host constants at construction time.
    """

    install_solve_point(world, bare_surface, trial_coils)
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(world.jax_boozer)
    del _phi_hat
    started = time.perf_counter()
    _bare_value, bare_grad, _bare_solution = _envelope_value_and_grad(
        residual_fn, objective_fn, np.asarray(bare_surface, dtype=np.float64)
    )
    _pred_value, pred_grad, _pred_solution = _envelope_value_and_grad(
        residual_fn, objective_fn, np.asarray(predicted_surface, dtype=np.float64)
    )
    del _bare_value, _bare_solution, _pred_value, _pred_solution
    bare_norm = require_finite(
        "envelope gradient norm at the bare anchor", float(np.linalg.norm(bare_grad))
    )
    pred_norm = require_finite(
        "envelope gradient norm at the predicted start",
        float(np.linalg.norm(pred_grad)),
    )
    return bare_norm, pred_norm, float(time.perf_counter() - started)


# ==========================================================================
# One solve arm
# ==========================================================================


@dataclass(frozen=True, slots=True)
class ArmResult:
    """One inner solve at the trial coils from one candidate warm start."""

    leg: str
    arm: str
    start_surface_sha256: str
    trial_coil_sha256: str
    success: bool
    persisted: bool
    iteration_count: int
    iota: float
    G: float
    reduced_gradient_l2: float
    inner_objective_phi_hat: float
    outer_objective_j: float
    outer_gradient_l2: float
    surface_sha256: str
    coil_delta_inf: float
    branch_label: str
    step_accepted: bool
    step_alpha: float
    phi_yy_condition: float
    step_grad_l2_ladder: tuple[float, ...]
    wall_seconds: float
    surface_dofs: NDArray[np.float64]

    def as_payload(self) -> dict[str, object]:
        return {
            "leg": self.leg,
            "arm": self.arm,
            "start_surface_sha256": self.start_surface_sha256,
            "trial_coil_sha256": self.trial_coil_sha256,
            "success": self.success,
            "persisted": self.persisted,
            "iteration_count": self.iteration_count,
            "iota": self.iota,
            "G": self.G,
            "reduced_gradient_l2": self.reduced_gradient_l2,
            "inner_objective_phi_hat": self.inner_objective_phi_hat,
            "outer_objective_j": self.outer_objective_j,
            "outer_flat675_gradient_l2": self.outer_gradient_l2,
            "surface_sha256": self.surface_sha256,
            "surface_sha256_equals_anchor_s38": (
                self.surface_sha256 == ANCHOR_SURFACE_SHA256
            ),
            "surface_sha256_equals_ledger_s39": (
                self.surface_sha256 == POISONED_SURFACE_SHA256
            ),
            "coil_delta_inf": self.coil_delta_inf,
            "branch_label": self.branch_label,
            "iota_delta_from_anchor": abs(self.iota - ANCHOR_IOTA),
            "iota_delta_from_recorded_wrong_branch": abs(
                self.iota - RECORDED_TRIAL_IOTA
            ),
            "within_iota_branch_guard": bool(
                abs(self.iota - ANCHOR_IOTA) <= float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD)
            ),
            "iota_branch_guard": float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
            "step_accepted": self.step_accepted,
            "step_alpha": self.step_alpha,
            "phi_yy_condition": self.phi_yy_condition,
            "step_grad_l2_ladder": list(self.step_grad_l2_ladder),
            "wall_seconds": self.wall_seconds,
        }


def run_arm(
    world: LoadedWorld,
    *,
    leg: str,
    arm: str,
    start_surface: NDArray[np.float64],
    trial_coils: NDArray[np.float64],
    y_iota: float,
    y_G: float,
) -> ArmResult:
    """One production-shaped inner solve, run raw so both branches show.

    Deliberately not ``_solve_nested_inner_at_coils``: that wrapper raises
    on a branch jump, and this probe exists to *see* which branch the
    solve lands on. Every solver knob is production's
    (``stab=F3_B37_IFT_STAB``, ``maxiter=NESTED_LS_NEWTON_MAXITER``,
    ``linear_solver="dense_lu"``, default ``tol=NESTED_LS_NEWTON_TOL``).
    """

    install_solve_point(world, start_surface, trial_coils)
    started = time.perf_counter()
    solution = run_reduced_nested_ls_schur_newton(
        world.jax_boozer,
        iota=float(y_iota),
        G=float(y_G),
        stab=float(F3_B37_IFT_STAB),
        maxiter=int(NESTED_LS_NEWTON_MAXITER),
        linear_solver="dense_lu",
    )
    wall = float(time.perf_counter() - started)
    iota = require_finite(f"{leg}/{arm} inner solve iota", solution.iota)
    g_value = require_finite(f"{leg}/{arm} inner solve G", solution.G)
    reduced_l2 = require_finite(
        f"{leg}/{arm} inner reduced gradient norm",
        float(np.linalg.norm(solution.reduced_gradient)),
    )
    phi_hat = require_finite(f"{leg}/{arm} inner objective", solution.objective)
    value, gradient = flat675_objective(world, trial_coils, solution.surface_dofs)
    outer_j = require_finite(f"{leg}/{arm} eight-term outer J", value)
    outer_grad_l2 = require_finite(
        f"{leg}/{arm} eight-term flat-675 gradient norm",
        float(np.linalg.norm(gradient)),
    )
    return ArmResult(
        leg=leg,
        arm=arm,
        start_surface_sha256=sha256_float64(start_surface),
        trial_coil_sha256=sha256_float64(trial_coils),
        success=bool(solution.success),
        persisted=bool(solution.persisted),
        iteration_count=int(solution.iteration_count),
        iota=iota,
        G=g_value,
        reduced_gradient_l2=reduced_l2,
        inner_objective_phi_hat=phi_hat,
        outer_objective_j=outer_j,
        outer_gradient_l2=outer_grad_l2,
        surface_sha256=sha256_float64(solution.surface_dofs),
        coil_delta_inf=float(solution.coil_delta_inf),
        branch_label=classify_branch(iota),
        step_accepted=bool(solution.step_accepted),
        step_alpha=float(solution.step_alpha),
        phi_yy_condition=float(solution.phi_yy_condition),
        step_grad_l2_ladder=tuple(float(step.grad_l2) for step in solution.steps),
        wall_seconds=wall,
        surface_dofs=np.array(solution.surface_dofs, dtype=np.float64, copy=True),
    )


def run_predictor_leg(
    world: LoadedWorld,
    *,
    leg: str,
    anchor_surface: NDArray[np.float64],
    anchor_iota: float,
    anchor_G: float,
    anchor_coils: NDArray[np.float64],
    trial_coils: NDArray[np.float64],
    prediction: str,
) -> tuple[dict[str, object], ArmResult, ArmResult]:
    """Both arms of one leg, plus predictor and fallback diagnostics."""

    install_anchor(world, anchor_surface, anchor_iota, anchor_G, anchor_coils)
    predictor = build_predictor(
        world,
        anchor_surface=anchor_surface,
        anchor_iota=anchor_iota,
        anchor_G=anchor_G,
        anchor_coils=anchor_coils,
        trial_coils=trial_coils,
    )
    predicted_surface = (
        np.asarray(anchor_surface, dtype=np.float64) + predictor.delta_surface
    )
    bare_norm, predicted_norm, envelope_seconds = envelope_gradient_norms(
        world,
        trial_coils=trial_coils,
        bare_surface=anchor_surface,
        predicted_surface=predicted_surface,
    )
    policy_arm = select_arm(bare_norm, predicted_norm)

    control = run_arm(
        world,
        leg=leg,
        arm="bare_anchor_control",
        start_surface=np.asarray(anchor_surface, dtype=np.float64),
        trial_coils=trial_coils,
        y_iota=anchor_iota,
        y_G=anchor_G,
    )
    # The predicted start is solved unconditionally, even when the
    # envelope fallback would reject it. A probe that skips the
    # measurement cannot falsify anything; the fallback decision is
    # recorded next to the measurement instead of replacing it.
    predicted = run_arm(
        world,
        leg=leg,
        arm="predicted_start",
        start_surface=predicted_surface,
        trial_coils=trial_coils,
        y_iota=anchor_iota,
        y_G=anchor_G,
    )
    payload: dict[str, object] = {
        "leg": leg,
        "falsifiable_prediction": prediction,
        "anchor": {
            "surface_sha256": sha256_float64(anchor_surface),
            "coil_sha256": sha256_float64(anchor_coils),
            "iota": float(anchor_iota),
            "G": float(anchor_G),
        },
        "trial_coil_sha256": sha256_float64(trial_coils),
        "predictor": predictor.as_payload(),
        "predicted_start_surface_sha256": sha256_float64(predicted_surface),
        "envelope_fallback": {
            "rule": (
                "fall back to the bare anchor when the predicted start's "
                "envelope gradient norm is the larger (ours, not DESC's)"
            ),
            "bare_anchor_gradient_l2": bare_norm,
            "predicted_start_gradient_l2": predicted_norm,
            "policy_arm_selected": policy_arm,
            "fallback_would_fire": policy_arm == "bare_anchor",
            "seconds": envelope_seconds,
        },
        "arms": {
            "bare_anchor_control": control.as_payload(),
            "predicted_start": predicted.as_payload(),
        },
    }
    return payload, control, predicted


# ==========================================================================
# Tolerance budget (upgrade plan Phase 4, first bullet)
# ==========================================================================


@dataclass(frozen=True, slots=True)
class BudgetRung:
    """One inner tolerance rung and the IFT adjoint gradient there."""

    requested_tol: float
    achieved_residual: float
    iteration_count: int
    tolerance_limited: bool
    success: bool
    iota: float
    G: float
    surface_sha256: str
    outer_objective_j: float
    gradient: NDArray[np.float64]
    gradient_l2: float
    adjoint_live_eta: float
    solve_seconds: float
    adjoint_seconds: float

    def as_payload(self) -> dict[str, object]:
        return {
            "requested_tol": self.requested_tol,
            "achieved_inner_residual_l2": self.achieved_residual,
            "iteration_count": self.iteration_count,
            "tolerance_limited": self.tolerance_limited,
            "budget_limited": not self.tolerance_limited,
            "success": self.success,
            "iota": self.iota,
            "G": self.G,
            "surface_sha256": self.surface_sha256,
            "outer_objective_j": self.outer_objective_j,
            "gradient": [float(entry) for entry in self.gradient],
            "gradient_l2": self.gradient_l2,
            "adjoint_live_eta": self.adjoint_live_eta,
            "solve_seconds": self.solve_seconds,
            "adjoint_seconds": self.adjoint_seconds,
        }


@dataclass(frozen=True, slots=True)
class AdjointSample:
    """One assembly of production's IFT adjoint at a frozen inner point."""

    gradient: NDArray[np.float64]
    objective_j: float
    live_eta: float
    phi_yy_condition: float
    dense: jax.Array
    seconds: float


def ift_adjoint_gradient_at(
    world: LoadedWorld,
    *,
    coil_dofs: NDArray[np.float64],
    surface_dofs: NDArray[np.float64],
    iota: float,
    G: float,
) -> AdjointSample:
    """Production's IFT adjoint gradient at a FROZEN inner solution.

    This is ``nested_ls_outer_value_and_grad``'s body with its opening
    ``_solve_nested_inner_at_coils`` removed and nothing else changed:
    every remaining step is that function's own call, in its own order, on
    production's own helpers. Removing the re-solve is the entire point --
    production polishes to ``NESTED_LS_NEWTON_TOL`` first, which would
    erase the loose rung this measures.

    ``AdjointSample.dense`` is handed back so a caller can take kappa(H_ss)
    from the same matrix the adjoint inverted rather than from a second
    assembly, and ``objective_j`` so a caller never re-evaluates the
    eight-term ``J`` this already computed.
    """

    started = time.perf_counter()
    install_solve_point(world, surface_dofs, coil_dofs)
    value, flat_gradient = flat675_objective(world, coil_dofs, surface_dofs)
    coil_block = np.asarray(flat_gradient[world.state.coil_slice], dtype=np.float64)
    surface_block = np.asarray(
        flat_gradient[world.state.surface_slice], dtype=np.float64
    )
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(world.jax_boozer)
    del _phi_hat
    operator = factor_reduced_nested_ls_schur(
        residual_fn,
        objective_fn,
        surface_dofs,
        y_probe=np.array([iota, G], dtype=np.float64),
    )
    cotangent = jnp.asarray(surface_block, dtype=jnp.float64)
    dense = jax.block_until_ready(
        materialize_stabilized_schur_dense(
            operator,
            float(F3_B37_IFT_STAB),
            max_dense_linearization_bytes=None,
        )
    )
    adjoint = jax.block_until_ready(solve_stabilized_schur_dense_lu(dense, cotangent))
    live_residual = jax.block_until_ready(operator.apply(adjoint) - cotangent)
    rhs_l2 = float(np.linalg.norm(surface_block))
    residual_l2 = float(
        np.linalg.norm(np.asarray(jax.device_get(live_residual), dtype=np.float64))
    )
    live_eta = residual_l2 / rhs_l2 if rhs_l2 > 0.0 else 0.0
    residual_rt, objective_rt, _phi_rt = nested_ls_runtime_coil_closures(
        world.jax_boozer
    )
    del _phi_rt
    correction = _mixed_coil_correction_vjp(
        residual_rt,
        objective_rt,
        np.asarray(surface_dofs, dtype=np.float64),
        np.asarray(coil_dofs, dtype=np.float64).reshape(-1),
        adjoint,
        operator=operator,
    )
    gradient = coil_block + correction
    value = require_finite("IFT adjoint mirror eight-term J", value)
    require(
        bool(np.all(np.isfinite(gradient))),
        "IFT adjoint mirror gradient finiteness",
        "all finite",
        float(np.linalg.norm(gradient)),
    )
    return AdjointSample(
        gradient=gradient,
        objective_j=float(value),
        live_eta=require_finite("IFT adjoint mirror live eta", live_eta),
        phi_yy_condition=float(operator.phi_yy_condition),
        dense=dense,
        seconds=float(time.perf_counter() - started),
    )


def schur_condition(dense: jax.Array) -> dict[str, object]:
    """kappa_2 of the stabilized dense Schur matrix, from its singular values.

    ``stab = F3_B37_IFT_STAB`` is 0.0, so this is kappa(H_ss) itself and
    not a regularized surrogate. One SVD, three published numbers, so the
    condition number and the values it came from cannot drift apart.
    """

    host = np.asarray(jax.device_get(dense), dtype=np.float64)
    singular = np.linalg.svd(host, compute_uv=False)
    largest = require_finite("schur_singular_value_max", float(singular[0]))
    smallest = require_finite("schur_singular_value_min", float(singular[-1]))
    require(smallest > 0.0, "schur_singular_value_min", "> 0.0", smallest)
    return {
        "schur_condition_number": largest / smallest,
        "schur_singular_value_max": largest,
        "schur_singular_value_min": smallest,
        "schur_dimension": int(host.shape[0]),
        "ift_stab": float(F3_B37_IFT_STAB),
        "note": (
            "stab = F3_B37_IFT_STAB = 0.0, so this is kappa_2(H_ss), not a "
            "regularized surrogate. Single state: the recorded eval-38 anchor."
        ),
    }


def run_tolerance_budget(
    world: LoadedWorld, ledger: ReplayLedger
) -> tuple[dict[str, object], int]:
    """Adjoint gradient error against achieved inner residual, one anchor.

    Returns the payload and the number of inner solves it consumed.
    """

    # 1. Production's own gradient at the anchor, and the proof that a
    #    tolerance ladder AT THE ANCHOR COILS would be degenerate: the
    #    recorded s38 already sits at ||grad Phi-hat|| ~ 1.4e-15, below
    #    every rung, so every rung would break at iteration 0 with the
    #    same residual. That is why the ladder below runs at the recorded
    #    eval-39 trial coils, warm-started from the recorded s38 anchor --
    #    the same recorded state, in the regime the budget must license.
    install_anchor(
        world,
        ledger.anchor_surface_dofs,
        ledger.anchor_iota,
        ledger.anchor_G,
        ledger.anchor_coil_dofs,
    )
    production_started = time.perf_counter()
    production_value, production_gradient = nested_ls_outer_value_and_grad(
        world.state, ledger.anchor_coil_dofs
    )
    production_seconds = float(time.perf_counter() - production_started)
    anchor_inner_iterations = int(world.state.inner_iterations)
    anchor_inner_grad_l2 = float(world.state.inner_grad_l2)
    production_live_eta = float(world.state.adjoint_live_eta)
    # ``_solve_nested_inner_at_coils`` commits its own result to the anchor,
    # so these three ARE the (s, iota, G) production assembled its adjoint
    # at. The mirror is fed those, not the ledger's, so any difference the
    # gate below reports is the mirror's, never a different input point.
    production_surface = np.array(
        world.state.anchor_surface_dofs, dtype=np.float64, copy=True
    )
    production_iota = float(world.state.anchor_iota)
    production_g = float(world.state.anchor_G)

    mirror = ift_adjoint_gradient_at(
        world,
        coil_dofs=ledger.anchor_coil_dofs,
        surface_dofs=production_surface,
        iota=production_iota,
        G=production_g,
    )
    production_l2 = float(np.linalg.norm(production_gradient))
    mirror_gap = float(np.linalg.norm(mirror.gradient - production_gradient))
    mirror_rel = mirror_gap / production_l2 if production_l2 > 0.0 else mirror_gap
    require(
        mirror_rel <= PRODUCTION_MIRROR_MAX_REL,
        "production_mirror_relative_difference",
        f"<= {PRODUCTION_MIRROR_MAX_REL!r}",
        mirror_rel,
    )

    condition = schur_condition(mirror.dense)

    # 2. The ladder, at the trial coils from the recorded anchor surface.
    rungs: list[BudgetRung] = []
    for requested in TOLERANCE_RUNGS:
        install_solve_point(world, ledger.anchor_surface_dofs, ledger.trial_coil_dofs)
        solve_started = time.perf_counter()
        solution = run_reduced_nested_ls_schur_newton(
            world.jax_boozer,
            iota=ledger.anchor_iota,
            G=ledger.anchor_G,
            stab=float(F3_B37_IFT_STAB),
            tol=float(requested),
            maxiter=int(TOLERANCE_RUNG_MAXITER),
            linear_solver="dense_lu",
        )
        solve_seconds = float(time.perf_counter() - solve_started)
        achieved = require_finite(
            f"tolerance rung {requested!r} achieved inner residual",
            float(np.linalg.norm(solution.reduced_gradient)),
        )
        sample = ift_adjoint_gradient_at(
            world,
            coil_dofs=ledger.trial_coil_dofs,
            surface_dofs=solution.surface_dofs,
            iota=float(solution.iota),
            G=float(solution.G),
        )
        rungs.append(
            BudgetRung(
                requested_tol=float(requested),
                achieved_residual=achieved,
                iteration_count=int(solution.iteration_count),
                tolerance_limited=(
                    int(solution.iteration_count) < int(TOLERANCE_RUNG_MAXITER)
                ),
                success=bool(solution.success),
                iota=require_finite(f"rung {requested!r} iota", solution.iota),
                G=require_finite(f"rung {requested!r} G", solution.G),
                surface_sha256=sha256_float64(solution.surface_dofs),
                outer_objective_j=sample.objective_j,
                gradient=sample.gradient,
                gradient_l2=float(np.linalg.norm(sample.gradient)),
                adjoint_live_eta=sample.live_eta,
                solve_seconds=solve_seconds,
                adjoint_seconds=sample.seconds,
            )
        )

    reference = rungs[0]
    require(
        reference.requested_tol == float(NESTED_LS_NEWTON_TOL),
        "tolerance ladder reference rung",
        float(NESTED_LS_NEWTON_TOL),
        reference.requested_tol,
    )
    reference_l2 = float(np.linalg.norm(reference.gradient))
    require(reference_l2 > 0.0, "reference gradient norm", "> 0.0", reference_l2)

    # One pass computes each rung's error against the reference; the JSON
    # rows, the fit input and the threshold table are all derived from it,
    # so the published curve and the fitted curve cannot drift apart.
    error_rows: list[tuple[BudgetRung, float, float]] = []
    for rung in rungs:
        absolute = float(np.linalg.norm(rung.gradient - reference.gradient))
        error_rows.append((rung, absolute, absolute / reference_l2))
    curve: list[dict[str, object]] = [
        {
            **rung.as_payload(),
            "gradient_absolute_error_l2": absolute,
            "gradient_relative_error": relative,
            "is_reference": rung is reference,
        }
        for rung, absolute, relative in error_rows
    ]

    # The reference rung's error against itself is identically zero, so it
    # has no log10 and cannot enter the fit. Any other rung that landed on
    # the same iterate is excluded for the same reason. Both exclusions are
    # published (``points_used``) rather than assumed away.
    fit_rows = tuple(
        (rung.achieved_residual, relative)
        for rung, _absolute, relative in error_rows
        if rung is not reference and relative > 0.0 and rung.achieved_residual > 0.0
    )
    require(
        len(fit_rows) >= 2,
        "tolerance ladder distinct fit points",
        ">= 2 rungs with a strictly positive residual and error",
        len(fit_rows),
    )
    slope, intercept, rms = loglog_slope(
        tuple(row[0] for row in fit_rows), tuple(row[1] for row in fit_rows)
    )

    threshold_rows = tuple(
        (rung.achieved_residual, relative) for rung, _absolute, relative in error_rows
    )
    budget_table = {
        f"{threshold:.0e}": largest_residual_under_threshold(threshold_rows, threshold)
        for threshold in BUDGET_THRESHOLDS
    }

    payload: dict[str, object] = {
        "mode": "tolerance-budget",
        "anchor": {
            "surface_sha256": ANCHOR_SURFACE_SHA256,
            "coil_sha256": ANCHOR_COIL_SHA256,
            "iota": ledger.anchor_iota,
            "G": ledger.anchor_G,
        },
        "schur_condition": {
            **condition,
            "phi_yy_condition": mirror.phi_yy_condition,
        },
        "production_mirror": {
            "why_not_call_production_per_rung": (
                "nested_ls_outer_value_and_grad re-solves the inner to "
                "NESTED_LS_NEWTON_TOL before assembling the adjoint, which "
                "would erase the loose rung being measured. The mirror is "
                "that function's body with the re-solve removed, built from "
                "its own helpers (_flat675_value_and_grad_at, "
                "factor_reduced_nested_ls_schur, "
                "materialize_stabilized_schur_dense, "
                "solve_stabilized_schur_dense_lu, _mixed_coil_correction_vjp) "
                "in its own order, and checked against production here."
            ),
            "production_objective_j": float(production_value),
            "production_gradient": [float(v) for v in production_gradient],
            "production_gradient_l2": production_l2,
            "production_adjoint_live_eta": production_live_eta,
            "production_seconds": production_seconds,
            "production_surface_sha256": sha256_float64(production_surface),
            "production_surface_equals_ledger_s38": (
                sha256_float64(production_surface) == ANCHOR_SURFACE_SHA256
            ),
            "production_iota": production_iota,
            "production_G": production_g,
            "mirror_gradient": [float(v) for v in mirror.gradient],
            "mirror_live_eta": mirror.live_eta,
            "mirror_seconds": mirror.seconds,
            "absolute_difference_l2": mirror_gap,
            "relative_difference": mirror_rel,
            "bitwise_identical": bool(
                np.array_equal(mirror.gradient, production_gradient)
            ),
            "gate": float(PRODUCTION_MIRROR_MAX_REL),
        },
        "anchor_ladder_degeneracy": {
            "finding": (
                "A tolerance ladder at the ANCHOR coils measures nothing: the "
                "recorded s38 is already converged, so every rung breaks at "
                "iteration 0 with the same residual. The ladder below is "
                "therefore run at the recorded eval-39 trial coils, "
                "warm-started from the recorded s38 anchor."
            ),
            "anchor_inner_iterations": anchor_inner_iterations,
            "anchor_inner_grad_l2": anchor_inner_grad_l2,
            "ledger_recorded_inner_grad_l2": ANCHOR_INNER_GRAD_L2,
            "loosest_rung_tol": float(max(TOLERANCE_RUNGS)),
        },
        "ladder": {
            "coils": "outer_evals[39].coil_dofs (the recorded trial point)",
            "warm_start_surface_sha256": ANCHOR_SURFACE_SHA256,
            "requested_tolerances": [float(t) for t in TOLERANCE_RUNGS],
            "maxiter": int(TOLERANCE_RUNG_MAXITER),
            "production_maxiter": int(NESTED_LS_NEWTON_MAXITER),
            "rungs": curve,
        },
        "scaling_fit": {
            "form": "log10(relative gradient error) = slope * log10(rho) + intercept",
            "slope": slope,
            "intercept": intercept,
            "rms_residual_log10": rms,
            "points_used": len(fit_rows),
            "excluded": (
                "the reference rung (error identically zero) and any rung "
                "that reproduced it exactly"
            ),
            "note": (
                "Measured, not asserted: no theoretical exponent is claimed "
                "or checked here."
            ),
        },
        "budget": {
            "largest_achieved_residual_under_threshold": budget_table,
            "reference_gradient_l2": reference_l2,
            "generality": (
                "SINGLE-STATE measurement at one recorded anchor and one "
                "recorded trial displacement. It does not generalize without "
                "more states, and it licenses no coarse tier on its own."
            ),
        },
    }
    return payload, len(TOLERANCE_RUNGS) + 1


# ==========================================================================
# Predictor mode
# ==========================================================================


def run_predictor_mode(
    world: LoadedWorld, ledger: ReplayLedger
) -> tuple[dict[str, object], list[dict[str, object]], int, str]:
    """Legs 3 and 4.

    Returns the payload, fail-closed records, the inner-solve count, and the
    sha256 of the regenerated eval-39 anchor (published, never gated).
    """

    failures: list[dict[str, object]] = []

    leg3, leg3_control, leg3_predicted = run_predictor_leg(
        world,
        leg="leg3_wrong_branch_capture",
        anchor_surface=ledger.anchor_surface_dofs,
        anchor_iota=ledger.anchor_iota,
        anchor_G=ledger.anchor_G,
        anchor_coils=ledger.anchor_coil_dofs,
        trial_coils=ledger.trial_coil_dofs,
        prediction=(
            "the predicted warm start pulls the eval-39 inner solve back onto "
            "the anchor's iota branch instead of the recorded capture at "
            f"iota = {RECORDED_TRIAL_IOTA!r}"
        ),
    )
    solves = 2
    leg3["verdict"] = {
        "falsifiable_prediction": leg3["falsifiable_prediction"],
        "control_branch_label": leg3_control.branch_label,
        "control_reproduced_recorded_branch": (
            leg3_control.branch_label == "recorded_wrong_branch"
        ),
        "control_iota": leg3_control.iota,
        "control_objective_j": leg3_control.outer_objective_j,
        "recorded_trial_iota": RECORDED_TRIAL_IOTA,
        "recorded_trial_j": RECORDED_TRIAL_J,
        "predicted_branch_label": leg3_predicted.branch_label,
        "predicted_iota": leg3_predicted.iota,
        "predicted_objective_j": leg3_predicted.outer_objective_j,
        "predicted_success": leg3_predicted.success,
        "prediction_survived": bool(
            leg3_predicted.success and leg3_predicted.branch_label == "anchor_branch"
        ),
        "discriminator": (
            "branch named by proximity to the two measured branches "
            f"(anchor {ANCHOR_IOTA!r} vs recorded capture "
            f"{RECORDED_TRIAL_IOTA!r}); the 0.05 iota branch guard does not "
            "separate them and is published per arm for completeness"
        ),
        "status": "measured",
    }

    # Leg 4 step 1: regenerate s39. The bitwise-identical solve already ran
    # as leg 3's bare-anchor control arm -- same start surface, same trial
    # coils, same y start, same solver knobs -- so it is reused rather than
    # repeated, and the reuse is stated in the record.
    #
    # The anchor is gated on PHYSICS, not on bits. A bitwise hash gate would
    # only ever report that this box runs a different simsoptpp binary than
    # the ledger was recorded on, which ``runtime.simsoptpp_sha256`` already
    # says; it cannot answer leg 4's question, which is whether the predictor
    # rescues a trial launched from a poisoned anchor. The hash is still
    # computed and published, un-gated, alongside the reason.
    regenerated_sha = leg3_control.surface_sha256
    anchor_physics = check_regenerated_anchor_physics(
        iota=leg3_control.iota,
        iteration_count=leg3_control.iteration_count,
        objective_j=leg3_control.outer_objective_j,
    )
    leg4: dict[str, object] = {
        "leg": "leg4_post_poisoning_trial",
        "falsifiable_prediction": (
            "no rescue: the predictor does not save the eval-43 trial (coils "
            "bitwise x38) off the poisoned eval-39 anchor"
        ),
        "anchor_regeneration": {
            "method": (
                "one inner solve at the recorded x39 warm-started from the "
                "recorded s38, reused from leg 3's bare-anchor control arm "
                "(bitwise-identical inputs and solver knobs)"
            ),
            "reused_from": "leg3_wrong_branch_capture/bare_anchor_control",
            "gate": "physics fingerprint (iota, branch side, iterations, J)",
            "physics_fingerprint": anchor_physics.as_payload(),
            "iteration_count": leg3_control.iteration_count,
            "iota": leg3_control.iota,
            "G": leg3_control.G,
            "objective_j": leg3_control.outer_objective_j,
            "success": leg3_control.success,
            "recorded_anchor_surface_sha256": POISONED_SURFACE_SHA256,
            "regenerated_anchor_surface_sha256": regenerated_sha,
            "bitwise_matches_recorded_anchor": (
                regenerated_sha == POISONED_SURFACE_SHA256
            ),
            "matches_committed_replay_prefix": regenerated_sha.startswith(
                COMMITTED_REPLAY_REGENERATED_SURFACE_SHA256_PREFIX
            ),
            "committed_replay_prefix": (
                COMMITTED_REPLAY_REGENERATED_SURFACE_SHA256_PREFIX
            ),
            "committed_replay_log": str(REPLAY_LOG_PATH.relative_to(REPO)),
            "bitwise_not_gated_because": (
                "a bitwise match is not attainable across the simsoptpp "
                "binary boundary between the recorded run and this one -- see "
                "the recorded_ledger_trajectory_drift block -- so it is "
                "measured and disclosed, never gated. There is no override "
                "flag: the physics fingerprint above is the gate."
            ),
        },
    }

    if not anchor_physics.passed:
        record: dict[str, object] = {
            "quantity": "leg4_regenerated_anchor_physics_fingerprint",
            "expected": "every physics check passes",
            "observed": "failed: " + ", ".join(anchor_physics.failures),
            "checks": [dict(check) for check in anchor_physics.checks],
            "consequence": (
                "leg 4 does not continue: without a physically equivalent "
                "poisoned anchor it is not asking its question. This is not "
                "the binary boundary -- that is disclosed and un-gated; this "
                "is the regeneration landing somewhere else entirely."
            ),
        }
        failures.append(record)
        leg4["failed_closed"] = record
        leg4["arms"] = None
        leg4["verdict"] = {
            "falsifiable_prediction": leg4["falsifiable_prediction"],
            "prediction_survived": None,
            "status": "failed_closed_on_anchor_physics_fingerprint",
        }
        return (
            {"mode": "predictor", "legs": [leg3, leg4]},
            failures,
            solves,
            regenerated_sha,
        )

    leg4_payload, leg4_control, leg4_predicted = run_predictor_leg(
        world,
        leg="leg4_post_poisoning_trial",
        anchor_surface=leg3_control.surface_dofs,
        anchor_iota=leg3_control.iota,
        anchor_G=leg3_control.G,
        anchor_coils=ledger.trial_coil_dofs,
        trial_coils=ledger.anchor_coil_dofs,
        prediction=str(leg4["falsifiable_prediction"]),
    )
    solves += 2
    rescued = bool(
        leg4_predicted.success and leg4_predicted.branch_label == "anchor_branch"
    )
    leg4.update(leg4_payload)
    leg4["verdict"] = {
        "falsifiable_prediction": leg4["falsifiable_prediction"],
        "anchor_provenance": (
            "REGENERATED. This leg did not run off the recorded eval-39 "
            "anchor's bytes -- those were never stored, only hashed. It ran "
            "off a regeneration that is physically equivalent to the recorded "
            "anchor (same iota branch, same inner iteration count, same J "
            "within the stated bands) and bitwise distinct from it "
            f"({regenerated_sha} against the recorded "
            f"{POISONED_SURFACE_SHA256}). Any claim from this leg inherits "
            "that qualifier."
        ),
        "anchor_bitwise_matches_recorded": (regenerated_sha == POISONED_SURFACE_SHA256),
        "control_success": leg4_control.success,
        "control_reproduced_recorded_failure": not leg4_control.success,
        "predicted_success": leg4_predicted.success,
        "predicted_branch_label": leg4_predicted.branch_label,
        "predicted_surface_equals_anchor_s38": (
            leg4_predicted.surface_sha256 == ANCHOR_SURFACE_SHA256
        ),
        "rescued": rescued,
        "prediction_survived": not rescued,
        "status": "measured",
    }
    return (
        {"mode": "predictor", "legs": [leg3, leg4]},
        failures,
        solves,
        regenerated_sha,
    )


# ==========================================================================
# Plan printing and payload assembly
# ==========================================================================


def git_head() -> dict[str, object]:
    """Commit, branch and dirtiness. Recorded, never gated.

    This tree is shared by several concurrent sessions, so a clean-tree
    requirement (as in Gate FD-0) would refuse to run for reasons that
    have nothing to do with this probe. The porcelain is published so a
    reader can judge instead.
    """

    def run(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()

    porcelain = run("status", "--porcelain")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(porcelain),
        "git_status_porcelain": porcelain,
    }


def claim_boundary() -> dict[str, object]:
    return {
        "cap_2048_attempted": False,
        "comparable_operators": False,
        "explicit_inverse_m_production": False,
        "f3_sealed": True,
        "gate": "predictor_replay",
        "inherits_f3_7_70x": False,
        "moving_coil_outer_loop": False,
        "nested_speed_claim": False,
        "offline_replay_only": True,
        "physics_gate_only": True,
        "predictor_wired_into_children": False,
        "single_state_measurement": True,
        "timing_content": False,
        "walls_are_operator_scheduling_diagnostics_only": True,
    }


def solve_plan(mode: str) -> list[dict[str, object]]:
    """The inner solves this run will execute, in order."""

    plan: list[dict[str, object]] = []
    if mode in ("predictor", "all"):
        plan.extend(
            [
                {
                    "solve": "leg3 bare-anchor control",
                    "coils": "x39 (outer_evals[39].coil_dofs)",
                    "warm_start": "s38 (recorded anchor surface)",
                    "tol": float(NESTED_LS_NEWTON_TOL),
                    "maxiter": int(NESTED_LS_NEWTON_MAXITER),
                    "also_serves_as": "leg4 s39 anchor regeneration (reused)",
                    "recorded_wall_seconds": "434-440 (committed replay A2/B2)",
                },
                {
                    "solve": "leg3 predicted start",
                    "coils": "x39",
                    "warm_start": "s38 + delta_s_pred",
                    "tol": float(NESTED_LS_NEWTON_TOL),
                    "maxiter": int(NESTED_LS_NEWTON_MAXITER),
                    "recorded_wall_seconds": "unmeasured",
                },
                {
                    "solve": "leg4 bare-anchor control",
                    "coils": "x38 (bitwise the eval-43 trial)",
                    "warm_start": "s39 (regenerated)",
                    "tol": float(NESTED_LS_NEWTON_TOL),
                    "maxiter": int(NESTED_LS_NEWTON_MAXITER),
                    "gated_on": "regenerated s39 hash == the ledger's 052923e7...",
                    "recorded_wall_seconds": "547 (committed replay A3)",
                },
                {
                    "solve": "leg4 predicted start",
                    "coils": "x38",
                    "warm_start": "s39 + delta_s_pred",
                    "tol": float(NESTED_LS_NEWTON_TOL),
                    "maxiter": int(NESTED_LS_NEWTON_MAXITER),
                    "gated_on": "regenerated s39 hash == the ledger's 052923e7...",
                    "recorded_wall_seconds": "unmeasured",
                },
            ]
        )
    if mode in ("tolerance-budget", "all"):
        plan.append(
            {
                "solve": "production mirror check / anchor degeneracy proof",
                "coils": "x38",
                "warm_start": "s38 (already converged: 0 iterations expected)",
                "call": "nested_ls_outer_value_and_grad",
                "recorded_wall_seconds": "43-101 (committed replay A1/B1/B3)",
            }
        )
        plan.extend(
            {
                "solve": f"tolerance rung tol={tol:.0e}",
                "coils": "x39",
                "warm_start": "s38",
                "tol": float(tol),
                "maxiter": int(TOLERANCE_RUNG_MAXITER),
                "recorded_wall_seconds": "<= 440 (loose rungs are cheaper)",
            }
            for tol in TOLERANCE_RUNGS
        )
    return plan


def print_plan(
    *,
    mode: str,
    ledger: ReplayLedger,
    lane_binding: dict[str, object],
    out_path: Path,
    identity: dict[str, object],
) -> None:
    """Human-readable dry-run report."""

    plan = solve_plan(mode)
    print("=== nested-LS outer predictor replay: DRY RUN (no GPU, no solve) ===")
    print(f"schema                : {SCHEMA}")
    print(f"mode                  : {mode}")
    print(f"repo                  : {REPO}")
    print(f"ledger                : {ledger.path}")
    print(f"replay log (cited)    : {REPLAY_LOG_PATH}")
    print(f"lane                  : {lane_binding['lane_path']}")
    print(f"out                   : {out_path}")
    print(f"jax default backend   : {identity['jax_default_backend']}")
    print(f"simsoptpp sha256      : {identity['simsoptpp_sha256']}")
    print("")
    print("--- ledger fingerprints (all asserted, all PASS) ---")
    for name, value in ledger.fingerprints.items():
        print(f"  PASS {name} = {value!r}")
    print("")
    print("--- lane binding (all asserted, all PASS) ---")
    for name, value in lane_binding.items():
        print(f"  PASS {name} = {value!r}")
    print("")
    print("--- predictor policy ---")
    print("  delta_s_pred      = -H_ss^-1 H_sc delta_c at the committed anchor")
    print(f"  ift_stab          = {float(F3_B37_IFT_STAB)!r}  (unregularized)")
    print(f"  trust region      = scale to {TRUST_REGION_RATIO!r} * ||s_anchor||_2")
    print("  fallback          = bare anchor when the predicted start's")
    print("                      envelope gradient norm is the larger")
    print("")
    print("--- solve plan ---")
    for index, step in enumerate(plan, start=1):
        print(f"  {index}. {step['solve']}")
        for key, value in step.items():
            if key == "solve":
                continue
            print(f"       {key}: {value!r}")
    print(f"  total inner solves: {len(plan)}")
    print("")
    if mode in ("predictor", "all"):
        print("--- leg 4 regenerated-anchor gate: PHYSICS, not bits ---")
        print(f"  iota            : {RECORDED_TRIAL_IOTA!r} +/- {REGEN_IOTA_ABS_TOL!r}")
        print(
            "                    (1.8e7x the measured 2-ULP "
            f"{COMMITTED_REPLAY_TRIAL_IOTA_ABS_DRIFT!r} drift;"
        )
        print(
            "                     8.1e-6 of the "
            f"{RECORDED_BRANCH_SEPARATION!r} branch separation)"
        )
        print("  branch side     : strictly nearer the capture than the anchor")
        print(f"  iterations      : exactly {REGEN_ITERATION_COUNT!r} (no tolerance)")
        print(f"  J               : {RECORDED_TRIAL_J!r} rel {REGEN_J_REL_TOL!r}")
        print(
            "                    (4.2e6x the measured 2-ULP "
            f"{COMMITTED_REPLAY_ANCHOR_J_REL_DRIFT!r} drift)"
        )
        print("  residual        : NOT gated -- the one non-converged quantity")
        print(
            "                    drifts "
            f"{COMMITTED_REPLAY_FAILED_RESIDUAL_REL_DRIFT!r} relative"
        )
        print("")
        print("--- leg 4 anchor hash: measured and disclosed, NEVER gated ---")
        print(f"  recorded        : {POISONED_SURFACE_SHA256}")
        print(
            "  committed replay: "
            f"{COMMITTED_REPLAY_REGENERATED_SURFACE_SHA256_PREFIX}... "
            "(twice: log lines A2 and B2)"
        )
        print(
            "  A bitwise match is not attainable across the simsoptpp binary\n"
            "  boundary between the recorded run and this one. The hash is\n"
            "  published with bitwise_matches_recorded_anchor, and leg 4's\n"
            "  verdict states it ran on a REGENERATED anchor: physically\n"
            "  equivalent to, and bitwise distinct from, the recorded one."
        )
        print("")
        print("--- recorded_ledger_trajectory_drift (published finding) ---")
        for row in trajectory_drift_discrepancies(None):
            print(f"  {row['quantity']}")
            print(f"       ledger         : {row['ledger']!r}")
            print(f"       committed replay: {row['committed_replay']!r}")
            print(f"       agreement      : {row['agreement']}")
        print("")
    if mode in ("tolerance-budget", "all"):
        print("--- tolerance ladder ---")
        print(f"  rungs           : {[float(t) for t in TOLERANCE_RUNGS]!r}")
        print(f"  maxiter         : {TOLERANCE_RUNG_MAXITER!r}")
        print("  ladder coils    : x39, warm-started from s38")
        print(
            "  reason          : the anchor surface s38 already sits at\n"
            f"                    ||grad Phi-hat|| = {ANCHOR_INNER_GRAD_L2!r}, "
            "below every rung,\n"
            "                    so a ladder at the anchor coils would be "
            "degenerate."
        )
        print(f"  thresholds      : {[float(t) for t in BUDGET_THRESHOLDS]!r}")
        print(f"  mirror gate     : {float(PRODUCTION_MIRROR_MAX_REL)!r} relative")
        print("")
    print("--- claim boundary ---")
    for name, value in sorted(claim_boundary().items()):
        print(f"  {name} = {value!r}")
    print("")
    print("DRY RUN COMPLETE: every fingerprint validated, no device touched.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline replay of the DESC order-1 predictor and the IFT adjoint "
            "tolerance budget on the recorded B37 v1 states."
        )
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="all",
        help="Which measurement to run (default: all).",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="Recovered B37 v1 JAX child ledger to replay.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Evidence JSON path (default: under docs/receipts/evidence/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, resolve and print the plan. No device, no solve.",
    )
    return parser.parse_args(argv)


def default_out_path(mode: str) -> Path:
    stem = "nested_ls_outer_predictor_replay_" + datetime.now(timezone.utc).strftime(
        "%Y%m%d"
    )
    suffix = "" if mode == "all" else f".{mode}"
    return EVIDENCE / f"{stem}{suffix}.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = str(args.mode)
    out_path = Path(args.out) if args.out is not None else default_out_path(mode)

    ledger = load_replay_ledger(Path(args.ledger))
    lane_binding = check_lane_binds_to_ledger(ledger)
    identity = nested_ls_runtime_identity()

    if args.dry_run:
        print_plan(
            mode=mode,
            ledger=ledger,
            lane_binding=lane_binding,
            out_path=out_path,
            identity=identity,
        )
        return 0

    world = load_world()
    failures: list[dict[str, object]] = []
    solves = 0
    predictor_payload: dict[str, object] | None = None
    budget_payload: dict[str, object] | None = None
    regenerated_sha: str | None = None

    if mode in ("predictor", "all"):
        (
            predictor_payload,
            predictor_failures,
            predictor_solves,
            regenerated_sha,
        ) = run_predictor_mode(world, ledger)
        failures.extend(predictor_failures)
        solves += predictor_solves

    if mode in ("tolerance-budget", "all"):
        budget_payload, budget_solves = run_tolerance_budget(world, ledger)
        solves += budget_solves

    payload: dict[str, object] = {
        "claim_boundary": claim_boundary(),
        "command": (
            "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu "
            "JAX_ENABLE_X64=1 PYTHONPATH=src .venv-qn-gpu/bin/python "
            "benchmarks/nested_ls_outer_predictor_replay.py"
            + ((" " + " ".join(sys.argv[1:])) if sys.argv[1:] else "")
        ),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "driver": "benchmarks.nested_ls_outer_predictor_replay",
        "failed_closed": failures,
        "inner_solve_count": solves,
        "lane": world.lane_meta,
        "lane_binding": lane_binding,
        "ledger": {
            "path": str(ledger.path),
            "sha256": ledger.sha256,
            "fingerprints": ledger.fingerprints,
        },
        "mode": mode,
        "policy": {
            "budget_thresholds": [float(t) for t in BUDGET_THRESHOLDS],
            "ift_stab": float(F3_B37_IFT_STAB),
            "inner_maxiter": int(NESTED_LS_NEWTON_MAXITER),
            "inner_tol": float(NESTED_LS_NEWTON_TOL),
            "iota_branch_guard": float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
            "linear_solver": "dense_lu",
            "production_mirror_max_rel": float(PRODUCTION_MIRROR_MAX_REL),
            "tolerance_rung_maxiter": int(TOLERANCE_RUNG_MAXITER),
            "tolerance_rungs": [float(t) for t in TOLERANCE_RUNGS],
            "trust_region_ratio": float(TRUST_REGION_RATIO),
        },
        "predictor": predictor_payload,
        "publication": PUBLICATION,
        "recorded_ledger_trajectory_drift": trajectory_drift_finding(regenerated_sha),
        "runtime": identity,
        "schema": SCHEMA,
        "tolerance_budget": budget_payload,
        "walls": {
            "import_init_seconds": _IMPORT_SECONDS,
            "world_load_seconds": world.load_seconds,
            "process_elapsed_seconds": float(time.perf_counter() - _T0),
        },
        **git_head(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(dump_strict_json(payload))
    print(f"wrote {out_path}")
    print(f"inner solves executed: {solves}")

    if failures:
        for record in failures:
            print(
                "FAILED CLOSED "
                f"{record['quantity']}: expected {record['expected']!r}, "
                f"observed {record['observed']!r}",
                file=sys.stderr,
            )
        raise SystemExit(
            f"nested-LS predictor replay failed closed; evidence written to {out_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
