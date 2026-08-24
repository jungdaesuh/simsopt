"""Parent-side consumption of both outer children, driven end to end.

Phase 1 of ``docs/nested_ls_upgrade_implementation_plan.md``: one
child -> parent -> dual-rejudge B3 integration test proving the driver in
``benchmarks/nested_ls_outer_claim.py`` consumes both children's current
schemas, binds each timed endpoint to its untimed rejudge, and gates the
physics. Until now the only way to execute that code was a ~3.3-hour GPU
claim run, so none of it was covered.

The seam these tests drive is the one the driver already has: it reaches
both lanes only by ``subprocess.run([PYTHON, script, out_path, ...])``
against the module-level ``NATIVE_CHILD`` / ``JAX_CHILD`` paths. Those two
constants are repointed at stub scripts written into ``tmp_path``. The
stubs import nothing but the standard library -- no JAX, no simsoptpp, no
device -- and emit complete, schema-valid payloads whose numbers are
self-consistent: the rejudge stub *reads the endpoint file the parent
handed it* and reports that endpoint, so the source binding the driver
checks is a real derivation and not a hand-written match. Everything
below the subprocess boundary is the production driver.

Three module-level names are also repointed, none of them the logic under
test:

* ``_require_clean_tree`` -- the only gate deliberately disabled. It
  guards the provenance of a real claim run (a dirty implementation tree
  must not mint a receipt); this worktree is permanently dirty and shared
  with other agents, and the gate has nothing to do with how the driver
  consumes child payloads. It is replaced by a function returning a fixed
  fake HEAD, and every artifact in these tests is built against that same
  value so the git_head interlocks stay real.
* ``EVIDENCE`` -- output location only. It must stay under the repo root
  because the receipt publishes ``execution_log`` as a repo-relative
  path, so it is repointed into the git-ignored ``.artifacts`` tree and
  deleted afterwards, rather than writing receipts into
  ``docs/receipts/evidence/`` beside real ones.
* ``CACHE_OUTER`` -- the XLA persistent-cache directory the JAX child env
  points at, and another path the receipt publishes repo-relative. No JAX
  runs here; repointing it beside the test's evidence directory keeps the
  run from touching the shared cache a live certification run owns.

The synthetic OMP sweep artifact is assembled by :func:`_sweep_payload`
from the contract constants themselves -- ``omp_set`` from
``F3_B37_BANANA_OMP_CONTRACT_THREADS``, ``repeats`` from
``NESTED_LS_OUTER_OMP_SWEEP_REPEATS``, ``aggregation`` from
``NESTED_LS_GATE6_AGGREGATION``, the row ``child_schema`` from
``NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA``, budget/maxcor/schema from the
driver's own constants -- so it cannot drift from the contract the way a
copied literal would. Its walls are shaped to put the unique minimum at
``NESTED_LS_GATE6_NATIVE_OMP_THREADS``; the artifact declares that value
as ``best_omp_num_threads`` and the driver independently recomputes it
from the rows, which is the property the interlock exists for.

"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest
from benchmarks import nested_ls_outer_claim as claim
from simsopt_jax_adapters.geo.nested_ls_contract import (
    F3_B37_BANANA_OMP_CONTRACT_THREADS,
    NESTED_LS_GATE6_AGGREGATION,
    NESTED_LS_GATE6_NATIVE_OMP_THREADS,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
    NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    NESTED_LS_OUTER_OMP_SWEEP_REPEATS,
    NESTED_LS_OUTER_REJUDGE_SCHEMA,
)

# The fake clean-tree HEAD every artifact in this file is minted against.
# It is not a real revision on purpose: a test that happened to agree with
# the live HEAD would stop proving the git_head interlocks the moment
# somebody committed.
_RUN_HEAD = "0" * 39 + "1"
_OTHER_HEAD = "0" * 39 + "2"

_ARTIFACT_ROOT = claim.REPO / ".artifacts" / "nested-ls-outer-claim-tests"
# Every stub-written child payload path, so the session teardown can unlink
# exactly the files this file caused. The driver deliberately puts child
# payloads in ``.artifacts/nested-ls-outer-tmp`` and unlinks them only on a
# completed run, so the refusal tests leave them behind; that directory is
# shared with real runs, which is why nothing here deletes by glob.
_STUB_LEDGERS: list[Path] = []

# The 11 moving-coil outer DOFs. Both lanes must open on the same start
# point (the driver refuses "start_coils_differ") and fork at the endpoint,
# which is exactly what a budget-truncated B3 pair does.
_START_COILS = [0.31, -0.72, 1.05, 0.0, 2.5, -0.4, 0.17, 0.93, -1.2, 0.6, 0.08]
_NATIVE_ENDPOINT_COILS = [value + 1.0e-3 for value in _START_COILS]
_JAX_ENDPOINT_COILS = [value + 1.0e-3 + 4.0e-9 for value in _START_COILS]
_NATIVE_ENDPOINT_SURFACE = [0.9, -0.11, 0.023, 0.0041, -0.00052]
_JAX_ENDPOINT_SURFACE = [0.9, -0.11, 0.023, 0.0041, -0.00053]
_ENDPOINT_IOTA = 0.15432109876
_ENDPOINT_G = 2.7182818284
_NATIVE_J = 0.5
# The JAX lane lands worse by a fork-sized relative gap. B3 only measures
# this; B37 gates on a band frozen from it, so it has to be nonzero for
# either assertion to mean anything.
_J_REL_GAP = 2.5e-9
_JAX_J = _NATIVE_J * (1.0 + _J_REL_GAP)
_FROZEN_J_BAND = 1.0e-8

_OUTER_POLICY: dict[str, object] = {
    "method": "L-BFGS-B",
    "ftol": 0.0,
    "gtol": 0.0,
    "maxls": 100,
    # Overwritten by both stubs from their own --budget/--maxcor argv, so
    # the driver's per-pair "same policy on both lanes" gate is comparing
    # what the driver actually handed each child.
    "maxiter": 0,
    "maxcor": 0,
}

_STUB_PREAMBLE = '''\
"""Stand-in outer child. Standard library only: no JAX, no simsoptpp."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PLAN = json.loads(Path(os.environ["NESTED_LS_STUB_PLAN"]).read_text(encoding="utf-8"))


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_json")
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--maxcor", type=int, required=True)
    parser.add_argument("--rejudge-endpoint", default=None)
    return parser.parse_args()


def rung(payload, args):
    """Take the rung from argv, so the payload reports what the parent asked."""

    payload["budget"] = args.budget
    payload["maxcor"] = args.maxcor
    payload["outer_policy"] = {
        **payload["outer_policy"],
        "maxiter": args.budget,
        "maxcor": args.maxcor,
    }
    return payload


def record(lane, out_json):
    """Log this payload's path so the test can unlink what the driver leaks."""

    with Path(PLAN["ledger"]).open("a", encoding="utf-8") as handle:
        handle.write(lane + " " + out_json + "\\n")


def recorded(lane):
    lines = Path(PLAN["ledger"]).read_text(encoding="utf-8").splitlines()
    return [line.split(" ", 1)[1] for line in lines if line.startswith(lane + " ")]


def emit(out_json, payload, style):
    """Write the payload, then fail closed only after it is on disk."""

    encoded = (
        json.dumps(payload, allow_nan=False, indent=2)
        if style == "indent2"
        else json.dumps(payload, allow_nan=False)
    )
    Path(out_json).write_text(encoded + "\\n", encoding="utf-8")
    return 0 if payload.get("child_fault_reason") is None else 1
'''

_STUB_NATIVE = (
    _STUB_PREAMBLE
    + """

ARGS = parse()
PAYLOAD = rung(dict(PLAN["native"]), ARGS)
THREADING = {
    name: os.environ.get(name)
    for name in ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")
}
OBSERVED = THREADING["OMP_NUM_THREADS"]
PAYLOAD["threading"] = THREADING
PAYLOAD["omp_num_threads"] = OBSERVED
PAYLOAD["omp_pinned"] = bool(OBSERVED) and OBSERVED.strip().isdigit()
record("native", ARGS.out_json)
sys.exit(emit(ARGS.out_json, PAYLOAD, "indent2"))
"""
)

_STUB_JAX = (
    _STUB_PREAMBLE
    + """

ARGS = parse()
if ARGS.rejudge_endpoint is None:
    PAYLOAD = rung(dict(PLAN["jax"]), ARGS)
    record("jax", ARGS.out_json)
    if PLAN["mutate_recorded_native_endpoint"]:
        NATIVE_ENDPOINTS = recorded("native")
        if NATIVE_ENDPOINTS:
            with Path(NATIVE_ENDPOINTS[-1]).open("ab") as handle:
                handle.write(b" ")
    sys.exit(emit(ARGS.out_json, PAYLOAD, "compact"))

SOURCE = Path(ARGS.rejudge_endpoint).read_bytes()
CHILD = json.loads(SOURCE)
SCHEMA = str(CHILD["schema"])
if SCHEMA == PLAN["native_schema"]:
    LANE = "native"
    ENDPOINT = CHILD["endpoint"]
    JUDGED = {
        "endpoint_j": ENDPOINT["objective"],
        "endpoint_iota": ENDPOINT["iota"],
        "endpoint_g": ENDPOINT["G"],
        "endpoint_coil_sha256": ENDPOINT["coil_sha256"],
        "endpoint_surface_sha256": ENDPOINT["surface_sha256"],
    }
else:
    LANE = "jax"
    JUDGED = {
        name: CHILD[name]
        for name in (
            "endpoint_j",
            "endpoint_iota",
            "endpoint_g",
            "endpoint_coil_sha256",
            "endpoint_surface_sha256",
        )
    }
PAYLOAD = {
    **PLAN["rejudge"],
    **JUDGED,
    "budget": ARGS.budget,
    "maxcor": ARGS.maxcor,
    "judged_lane": LANE,
    "judged_schema": SCHEMA,
    "source_child_schema": SCHEMA,
    "source_child_payload_sha256": hashlib.sha256(SOURCE).hexdigest(),
}
PAYLOAD.update(PLAN["rejudge_overrides"].get(LANE, {}))
sys.exit(emit(ARGS.out_json, PAYLOAD, PLAN["rejudge_encoding"]))
"""
)


def _sha_dofs(values: list[float]) -> str:
    """A declared endpoint sha the parent treats as an opaque witness."""

    return hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest()


def _evaluation_rows(objective: float) -> list[dict[str, object]]:
    """One objective row and one containment-barrier row, Phase-0 shaped."""

    return [
        {
            "eval_index": 0,
            "inner_feasible": True,
            "value_is_valid": True,
            "rejection_reason": None,
            "rejection_detail": None,
            "j": objective,
        },
        {
            "eval_index": 1,
            "inner_feasible": False,
            "value_is_valid": False,
            "rejection_reason": "inner_solve_failed",
            "rejection_detail": "stub inner refusal",
            "j": objective + 0.5,
        },
    ]


def _restart_attempts(objective: float) -> list[dict[str, object]]:
    return [
        {
            "attempt": 0,
            "status": 1,
            "message": "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT",
            "nit": 3,
            "result_fun": objective,
            "value_is_valid": True,
        }
    ]


def _native_child_template() -> dict[str, object]:
    """A complete ``nested-ls-outer-native-child.v3`` payload.

    ``threading`` / ``omp_num_threads`` / ``omp_pinned`` are deliberately
    absent: the stub fills them from its own process environment, which is
    what makes the driver's OMP gates a measurement of the env the driver
    itself pinned.
    """

    return {
        "schema": NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
        "publication": "stub native nested twin",
        "success": True,
        "child_fault_reason": None,
        "optimizer_success": False,
        "ftol_zero_stop": False,
        "feasible_evaluations": 1,
        "rejected_evaluations": 1,
        "rejection_reasons": ["inner_solve_failed"],
        "budget": 0,
        "maxcor": 0,
        "nit": 3,
        "nfev": 4,
        "njev": 4,
        "restart_count": 0,
        "restart_nits": [3],
        "restart_attempts": _restart_attempts(_NATIVE_J),
        "status": 1,
        "message": "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT",
        "endpoint_is_optimizer_x": True,
        "optimizer_x": _NATIVE_ENDPOINT_COILS,
        "outer_policy": dict(_OUTER_POLICY),
        "start": {"coil_dofs": _START_COILS, "evaluation": 0},
        "endpoint": {
            "coil_dofs": _NATIVE_ENDPOINT_COILS,
            "coil_sha256": _sha_dofs(_NATIVE_ENDPOINT_COILS),
            "objective": _NATIVE_J,
            "iota": _ENDPOINT_IOTA,
            "G": _ENDPOINT_G,
            "gradient_l2": 4.2e-4,
            "surface_dofs": _NATIVE_ENDPOINT_SURFACE,
            "surface_sha256": _sha_dofs(_NATIVE_ENDPOINT_SURFACE),
        },
        "evaluations": _evaluation_rows(_NATIVE_J),
    }


def _jax_child_template() -> dict[str, object]:
    """A complete ``nested-ls-outer-jax-child.v4`` payload."""

    return {
        "schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
        "mode": "outer",
        "budget": 0,
        "maxcor": 0,
        "success": True,
        "child_fault_reason": None,
        "optimizer_success": False,
        "status": 1,
        "message": "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT",
        "ftol_zero_stop": False,
        "nit": 3,
        "nfev": 4,
        "njev": 4,
        "restart_count": 0,
        "restart_nits": [3],
        "restart_attempts": _restart_attempts(_JAX_J),
        "result_fun": _JAX_J,
        "result_fun_is_valid": True,
        "endpoint_is_optimizer_x": True,
        "optimizer_x": _JAX_ENDPOINT_COILS,
        "outer_policy": dict(_OUTER_POLICY),
        "iota_branch_guard": NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
        "feasible_evaluations": 1,
        "rejected_evaluations": 1,
        "start_policy": "raw_lane_surface_unnest",
        "start_coil_dofs": _START_COILS,
        "evaluations": _evaluation_rows(_JAX_J),
        "endpoint_coil_dofs": _JAX_ENDPOINT_COILS,
        "endpoint_coil_sha256": _sha_dofs(_JAX_ENDPOINT_COILS),
        "endpoint_surface_dofs": _JAX_ENDPOINT_SURFACE,
        "endpoint_surface_sha256": _sha_dofs(_JAX_ENDPOINT_SURFACE),
        "endpoint_j": _JAX_J,
        "endpoint_grad_l2": 3.1e-4,
        "endpoint_grad_inf": 1.7e-4,
        "endpoint_iota": _ENDPOINT_IOTA,
        "endpoint_g": _ENDPOINT_G,
        "endpoint_adjoint_live_eta": 0.0,
    }


def _rejudge_template() -> dict[str, object]:
    """The lane-independent half of ``nested-ls-outer-rejudge.v1``.

    The lane-bound half -- judged lane, source schema, source payload sha
    and every endpoint field -- is derived by the stub from the endpoint
    file the parent handed it, never from this template.
    """

    return {
        "schema": NESTED_LS_OUTER_REJUDGE_SCHEMA,
        "mode": "rejudge",
        "grad_tol": NESTED_LS_NEWTON_TOL,
        "y_star_iota": _ENDPOINT_IOTA,
        "y_star_g": _ENDPOINT_G,
        "y_star_vs_endpoint_iota": 0.0,
        "y_star_vs_endpoint_g": 0.0,
        "y_rank": 2,
        "reduced_grad_l2": 0.0,
        "reduced_grad_ok": True,
        "native_rejudge_success": True,
        "native_rejudge_iter": 0,
        "native_rejudge_iota": _ENDPOINT_IOTA,
        "native_rejudge_g": _ENDPOINT_G,
        "native_rejudge_grad_l2": 0.0,
        "native_rejudge_grad_inf": 0.0,
        "native_rejudge_coil_delta_inf": 0.0,
        "native_rejudge_surface_delta_inf": 0.0,
        "native_rejudge_seconds": 0.0,
        "rejudge_noop": True,
        "fail_closed_reason": None,
    }


def _plan(ledger: Path) -> dict[str, object]:
    """The transcript both stubs replay. Every test mutates one field of it."""

    return {
        "ledger": str(ledger),
        "native_schema": NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
        "native": _native_child_template(),
        "jax": _jax_child_template(),
        "rejudge": _rejudge_template(),
        "rejudge_overrides": {},
        # Default to the encoding the PRODUCTION child writes: compact,
        # insertion-ordered, one trailing newline. The parent binds the
        # child's actual bytes, so the binding is encoding-agnostic by
        # design — which is the point. It was not always: the parent used to
        # re-encode the parsed payload with ``indent=2`` and compare that to
        # the sha of the bytes the child wrote, so no real pair could pass.
        "rejudge_encoding": "compact",
        "mutate_recorded_native_endpoint": False,
    }


def _sweep_wall(omp_num_threads: int, repeat: int, *, best: int) -> float:
    """A wall shape whose unique minimum sits at ``best``."""

    return 100.0 + float(abs(omp_num_threads - best)) + 0.25 * float(repeat)


def _sweep_payload(
    *,
    git_head: str = _RUN_HEAD,
    best: int = NESTED_LS_GATE6_NATIVE_OMP_THREADS,
    maxcor: int = claim.DEFAULT_MAXCOR,
) -> dict[str, object]:
    """Build a swept-OMP artifact out of the contract constants."""

    omp_set = F3_B37_BANANA_OMP_CONTRACT_THREADS
    rows = [
        {
            "omp_num_threads": int(omp_num_threads),
            "observed_omp_num_threads": int(omp_num_threads),
            "omp_pinned": True,
            "repeat": int(repeat),
            "success": True,
            "child_schema": NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
            "nit": 3,
            "nfev": 4,
            "endpoint_j": _NATIVE_J,
            "endpoint_iota": _ENDPOINT_IOTA,
            "process_wall_seconds": _sweep_wall(omp_num_threads, repeat, best=best),
        }
        for repeat in range(NESTED_LS_OUTER_OMP_SWEEP_REPEATS)
        for omp_num_threads in omp_set
    ]
    per_omp_min = {
        str(omp_num_threads): min(
            float(row["process_wall_seconds"])
            for row in rows
            if int(row["omp_num_threads"]) == omp_num_threads
        )
        for omp_num_threads in omp_set
    }
    return {
        "aggregation": NESTED_LS_GATE6_AGGREGATION,
        "best_omp_num_threads": int(best),
        "budget": claim.B3_BUDGET,
        "claim": None,
        "date": "2026-08-24",
        "driver": "benchmarks.nested_ls_outer_claim",
        "git_head": git_head,
        "interleaved_repeats": True,
        "jax_lane_run": False,
        "maxcor": int(maxcor),
        "omp_set": [int(value) for value in omp_set],
        "per_omp_min_process_wall_seconds": per_omp_min,
        "repeats": NESTED_LS_OUTER_OMP_SWEEP_REPEATS,
        "rows": rows,
        "schema": claim.SWEEP_SCHEMA,
        "tag": None,
        "written_by_pytest": True,
    }


@dataclass(frozen=True)
class _Harness:
    """One installed driver run: stub children, evidence dir, sweep artifact."""

    workdir: Path
    evidence: Path
    ledger: Path
    sweep: Path

    def receipt(self, budget: int) -> Path:
        return self.evidence / f"nested_ls_outer_b{budget}_{claim.EVIDENCE_DATE}.json"

    def read_receipt(self, budget: int) -> dict[str, object]:
        return json.loads(self.receipt(budget).read_text(encoding="utf-8"))

    def b3_argv(self, *extra: str) -> list[str]:
        """The one chartered B3 command line every B3 test here runs.

        Identical in every test, so ``green_b3`` is a true control: the only
        difference between it and a refusal case is the single mutated field
        that case puts in the plan or the sweep artifact.
        """

        return [
            "--budget",
            str(claim.B3_BUDGET),
            "--omp",
            str(NESTED_LS_GATE6_NATIVE_OMP_THREADS),
            "--maxcor",
            str(claim.DEFAULT_MAXCOR),
            "--pairs",
            str(claim.REPEATS),
            "--omp-evidence",
            str(self.sweep),
            *extra,
        ]


def _install(
    monkeypatch: pytest.MonkeyPatch,
    workdir: Path,
    *,
    plan: dict[str, object] | None = None,
    sweep: dict[str, object] | None = None,
) -> _Harness:
    """Repoint the four module-level seams and write the run's fixtures."""

    native_script = workdir / "stub_native_child.py"
    jax_script = workdir / "stub_jax_child.py"
    native_script.write_text(_STUB_NATIVE, encoding="utf-8")
    jax_script.write_text(_STUB_JAX, encoding="utf-8")
    ledger = workdir / "child_payload_paths.txt"
    ledger.write_text("", encoding="utf-8")
    _STUB_LEDGERS.append(ledger)
    plan_path = workdir / "plan.json"
    plan_path.write_text(
        json.dumps(_plan(ledger) if plan is None else plan), encoding="utf-8"
    )
    sweep_path = workdir / "omp_sweep.json"
    sweep_path.write_text(
        json.dumps(_sweep_payload() if sweep is None else sweep, indent=2) + "\n",
        encoding="utf-8",
    )
    evidence = _ARTIFACT_ROOT / workdir.name
    evidence.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(claim, "NATIVE_CHILD", native_script)
    monkeypatch.setattr(claim, "JAX_CHILD", jax_script)
    monkeypatch.setattr(claim, "EVIDENCE", evidence)
    monkeypatch.setattr(claim, "CACHE_OUTER", evidence / "xla-cache")
    monkeypatch.setattr(claim, "_require_clean_tree", lambda: _RUN_HEAD)
    monkeypatch.setenv("NESTED_LS_STUB_PLAN", str(plan_path))
    return _Harness(workdir=workdir, evidence=evidence, ledger=ledger, sweep=sweep_path)


@pytest.fixture(scope="session", autouse=True)
def _artifact_root() -> object:
    """Keep receipts out of ``docs/receipts/evidence`` and off the tree."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    yield _ARTIFACT_ROOT
    for ledger in _STUB_LEDGERS:
        for line in ledger.read_text(encoding="utf-8").splitlines():
            Path(line.split(" ", 1)[1]).unlink(missing_ok=True)
    shutil.rmtree(_ARTIFACT_ROOT)


@pytest.fixture(scope="module")
def green_b3(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """One physics-green B3 run over the full chartered pair count.

    Module-scoped because it is both the happy-path subject and the
    control the B37 interlock tests measure their mutations against.
    """

    workdir = tmp_path_factory.mktemp("greenb3")
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, workdir)
        claim.main(harness.b3_argv())
        receipt_path = harness.receipt(claim.B3_BUDGET)
        return {
            "path": receipt_path,
            "receipt": json.loads(receipt_path.read_text(encoding="utf-8")),
        }


def test_b3_run_consumes_both_child_schemas_into_a_green_v2_receipt(
    green_b3: dict[str, object],
) -> None:
    """A full B3 run publishes a green claim.v2 receipt over both lanes."""

    receipt = green_b3["receipt"]
    assert receipt["schema"] == "nested-ls-outer-claim.v2"
    assert receipt["fail_closed_reason"] is None
    assert receipt["git_head"] == _RUN_HEAD
    boundary = receipt["claim_boundary"]
    assert boundary["budget"] == claim.B3_BUDGET
    assert boundary["repeats"] == claim.REPEATS
    assert boundary["rejudged_lanes"] == ["native", "jax"]
    assert boundary["native_omp_num_threads"] == NESTED_LS_GATE6_NATIVE_OMP_THREADS
    assert boundary["omp_provenance"] == claim.OMP_PROVENANCE_SWEPT
    assert boundary["omp_evidence"]["best_omp_num_threads"] == (
        NESTED_LS_GATE6_NATIVE_OMP_THREADS
    )
    pairs = receipt["pairs"]
    assert [pair["repeat"] for pair in pairs] == list(range(claim.REPEATS))
    assert [pair["fail_closed_reason"] for pair in pairs] == [None] * claim.REPEATS
    assert [pair["native"]["child_payload"]["schema"] for pair in pairs] == (
        [NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA] * claim.REPEATS
    )
    assert [pair["jax"]["child_payload"]["schema"] for pair in pairs] == (
        [NESTED_LS_OUTER_JAX_CHILD_SCHEMA] * claim.REPEATS
    )
    assert [pair["native"]["observed_omp_num_threads"] for pair in pairs] == (
        [NESTED_LS_GATE6_NATIVE_OMP_THREADS] * claim.REPEATS
    )


def test_b3_receipt_carries_an_untimed_rejudge_of_each_lane_per_pair(
    green_b3: dict[str, object],
) -> None:
    """Every pair gets two rejudges, each bound to its own lane's endpoint."""

    for pair in green_b3["receipt"]["pairs"]:
        for lane in ("native", "jax"):
            row = pair[lane]
            envelope = pair[f"{lane}_rejudge"]
            payload = envelope["payload"]
            assert envelope["timed"] is False
            assert envelope["repeat"] == pair["repeat"]
            assert payload["schema"] == NESTED_LS_OUTER_REJUDGE_SCHEMA
            assert payload["judged_lane"] == lane
            assert payload["budget"] == claim.B3_BUDGET
            assert payload["source_child_payload_sha256"] == row["child_payload_sha256"]
            assert payload["endpoint_j"] == row["endpoint_j"]


def test_b3_measures_the_j_fork_without_gating_on_it(
    green_b3: dict[str, object],
) -> None:
    """B3 publishes measured_j_rel_gap_max and leaves the band unjudged."""

    receipt = green_b3["receipt"]
    pairs = receipt["pairs"]
    assert [pair["endpoint_j_within_frozen_band"] for pair in pairs] == (
        [None] * claim.REPEATS
    )
    assert receipt["claim_boundary"]["j_parity_rtol"] is None
    assert receipt["claim_boundary"]["j_parity_mode"] == "observational_b3"
    gaps = [pair["endpoint_j_rel_gap_worse_direction"] for pair in pairs]
    assert gaps == pytest.approx([_J_REL_GAP] * claim.REPEATS, rel=1e-9)
    assert receipt["claim_boundary"]["measured_j_rel_gap_max"] == max(gaps)


def test_phase0_child_evidence_survives_into_the_receipt(
    green_b3: dict[str, object],
) -> None:
    """value_is_valid, restart_attempts and child_fault_reason arrive intact."""

    pair = green_b3["receipt"]["pairs"][0]
    for lane in ("native", "jax"):
        row = pair[lane]
        raw = row["child_payload_raw"]
        embedded_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert embedded_sha == row["child_payload_sha256"]
        embedded = row["child_payload"]
        assert json.loads(raw) == embedded
        assert embedded["child_fault_reason"] is None
        assert [entry["value_is_valid"] for entry in embedded["evaluations"]] == [
            True,
            False,
        ]
        assert [entry["value_is_valid"] for entry in embedded["restart_attempts"]] == [
            True
        ]


def test_j_parity_rtol_is_refused_at_b3(tmp_path: Path) -> None:
    """B3 measures the fork band, so it may not be handed one to gate on."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path)
        with pytest.raises(SystemExit) as refusal:
            claim.main(harness.b3_argv("--j-parity-rtol", repr(_FROZEN_J_BAND)))
    assert "--j-parity-rtol is forbidden for --budget 3" in str(refusal.value)
    assert not harness.receipt(claim.B3_BUDGET).exists()


def test_native_lane_rejudge_of_a_different_endpoint_fails_the_run(
    tmp_path: Path,
) -> None:
    """A rejudge whose native endpoint J is not the timed one fails closed."""

    plan = _plan(tmp_path / "child_payload_paths.txt")
    plan["rejudge_overrides"] = {"native": {"endpoint_j": _NATIVE_J * (1.0 + 1e-12)}}
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path, plan=plan)
        with pytest.raises(SystemExit) as refusal:
            claim.main(harness.b3_argv())
    assert "native_rejudge_endpoint_j_mismatch" in str(refusal.value)
    receipt = harness.read_receipt(claim.B3_BUDGET)
    assert receipt["fail_closed_reason"] == "native_rejudge_endpoint_j_mismatch"
    assert receipt["pairs"][0]["physics_ok"] is False


def test_jax_lane_rejudge_of_a_different_endpoint_fails_the_run(
    tmp_path: Path,
) -> None:
    """A rejudge whose JAX endpoint J is not the timed one fails closed."""

    plan = _plan(tmp_path / "child_payload_paths.txt")
    plan["rejudge_overrides"] = {"jax": {"endpoint_j": _JAX_J * (1.0 + 1e-12)}}
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path, plan=plan)
        with pytest.raises(SystemExit) as refusal:
            claim.main(harness.b3_argv())
    assert "jax_rejudge_endpoint_j_mismatch" in str(refusal.value)
    receipt = harness.read_receipt(claim.B3_BUDGET)
    assert receipt["fail_closed_reason"] == "jax_rejudge_endpoint_j_mismatch"
    assert receipt["pairs"][0]["physics_ok"] is False


def test_rejudge_naming_other_source_bytes_is_refused(tmp_path: Path) -> None:
    """The rejudge must cite the sha of the exact bytes the parent embedded."""

    plan = _plan(tmp_path / "child_payload_paths.txt")
    plan["rejudge_overrides"] = {
        "native": {"source_child_payload_sha256": hashlib.sha256(b"").hexdigest()}
    }
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path, plan=plan)
        with pytest.raises(SystemExit) as refusal:
            claim.main(harness.b3_argv())
    assert "native_rejudge_source_child_payload_sha256_mismatch" in str(refusal.value)
    receipt = harness.read_receipt(claim.B3_BUDGET)
    assert receipt["fail_closed_reason"] == (
        "native_rejudge_source_child_payload_sha256_mismatch"
    )


def test_endpoint_file_touched_after_the_parent_embedded_it_is_refused(
    tmp_path: Path,
) -> None:
    """One byte appended to a timed endpoint stops its rejudge from launching."""

    plan = _plan(tmp_path / "child_payload_paths.txt")
    plan["mutate_recorded_native_endpoint"] = True
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path, plan=plan)
        with pytest.raises(RuntimeError) as refusal:
            claim.main(harness.b3_argv())
    assert "rejudge native endpoint payload changed after the timed child" in str(
        refusal.value
    )
    assert not harness.receipt(claim.B3_BUDGET).exists()


def test_omp_that_is_not_the_swept_best_is_refused(tmp_path: Path) -> None:
    """A claim may only run at the OMP the sweep artifact's rows recompute to."""

    unswept = F3_B37_BANANA_OMP_CONTRACT_THREADS[0]
    assert unswept != NESTED_LS_GATE6_NATIVE_OMP_THREADS
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path)
        argv = harness.b3_argv()
        argv[argv.index("--omp") + 1] = str(unswept)
        with pytest.raises(SystemExit) as refusal:
            claim.main(argv)
    assert (
        f"--omp {unswept} is not the swept best-of-contract "
        f"{NESTED_LS_GATE6_NATIVE_OMP_THREADS}" in str(refusal.value)
    )


def test_sweep_artifact_from_another_head_is_refused(tmp_path: Path) -> None:
    """Denominator evidence must come from the implementation being claimed."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(
            monkeypatch, tmp_path, sweep=_sweep_payload(git_head=_OTHER_HEAD)
        )
        with pytest.raises(SystemExit) as refusal:
            claim.main(harness.b3_argv())
    message = str(refusal.value)
    assert f"--omp-evidence git_head is {_OTHER_HEAD!r}" in message
    assert f"expected the claim implementation {_RUN_HEAD!r}" in message


def _honest_child_failure_message(tmp_path: Path) -> str:
    """Run a B3 whose native child publishes a fault payload and exits 1."""

    plan = _plan(tmp_path / "child_payload_paths.txt")
    native = plan["native"]
    native["success"] = False
    native["child_fault_reason"] = NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path, plan=plan)
        with pytest.raises(RuntimeError) as refusal:
            claim.main(harness.b3_argv())
        assert not harness.receipt(claim.B3_BUDGET).exists()
    return str(refusal.value)


def test_child_failing_after_writing_its_receipt_keeps_that_receipt(
    tmp_path: Path,
) -> None:
    """The parent cites the fault payload a nonzero child preserved on disk."""

    message = _honest_child_failure_message(tmp_path)
    assert "outer native child failed rc=1" in message
    preserved = re.search(r"child_payload=(\S+\.json)", message)
    assert preserved is not None, message
    payload = json.loads(Path(preserved.group(1)).read_text(encoding="utf-8"))
    assert payload["schema"] == NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA
    assert (
        payload["child_fault_reason"] == NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON
    )
    assert payload["success"] is False


def test_child_failure_reason_reaches_the_parents_error_message(
    tmp_path: Path,
) -> None:
    """The parent's error says WHY the child failed, not just that it did.

    A child that fails closed goes out of its way to publish its ledger before
    exiting nonzero. That is wasted unless the reason reaches the operator: the
    parent used to report only a return code and a stderr tail. The assertion
    is deliberately key-agnostic — it asks that the reason VALUE appear in the
    message — so it stays honest if producer and consumer ever rename the key
    again, which they did once already (``failure_reason`` →
    ``child_fault_reason``) and which is exactly how this test earned a strict
    xfail for part of its life.
    """

    message = _honest_child_failure_message(tmp_path)
    assert NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON in message


def test_b37_accepts_a_green_b3_receipt_and_gates_the_frozen_band(
    tmp_path: Path, green_b3: dict[str, object]
) -> None:
    """B37 inherits B3's swept bar and judges the endpoint-J band it froze."""

    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path)
        claim.main(
            [
                "--budget",
                str(claim.B37_BUDGET),
                "--omp",
                str(NESTED_LS_GATE6_NATIVE_OMP_THREADS),
                "--maxcor",
                str(claim.DEFAULT_MAXCOR),
                "--pairs",
                "1",
                "--b3-receipt",
                str(green_b3["path"]),
                "--j-parity-rtol",
                repr(_FROZEN_J_BAND),
            ]
        )
        receipt = harness.read_receipt(claim.B37_BUDGET)
    assert receipt["fail_closed_reason"] is None
    boundary = receipt["claim_boundary"]
    assert boundary["budget"] == claim.B37_BUDGET
    assert boundary["j_parity_mode"] == "frozen_from_b3"
    assert boundary["j_parity_rtol"] == _FROZEN_J_BAND
    measured = green_b3["receipt"]["claim_boundary"]["measured_j_rel_gap_max"]
    assert boundary["b3_measured_j_rel_gap_max"] == measured
    assert receipt["pairs"][0]["endpoint_j_within_frozen_band"] is True
    assert receipt["pairs"][0]["native"]["child_payload"]["budget"] == claim.B37_BUDGET


def test_b37_refuses_a_b3_receipt_whose_embedded_child_bytes_were_edited(
    tmp_path: Path, green_b3: dict[str, object]
) -> None:
    """The receipt's endpoint fields are recomputed from the embedded bytes."""

    receipt = json.loads(json.dumps(green_b3["receipt"]))
    row = receipt["pairs"][0]["native"]
    row["child_payload_raw"] = row["child_payload_raw"].replace(
        '"nfev": 4', '"nfev": 5', 1
    )
    edited = tmp_path / "edited_b3_receipt.json"
    edited.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path)
        with pytest.raises(SystemExit) as refusal:
            claim.main(
                [
                    "--budget",
                    str(claim.B37_BUDGET),
                    "--omp",
                    str(NESTED_LS_GATE6_NATIVE_OMP_THREADS),
                    "--maxcor",
                    str(claim.DEFAULT_MAXCOR),
                    "--pairs",
                    "1",
                    "--b3-receipt",
                    str(edited),
                    "--j-parity-rtol",
                    repr(_FROZEN_J_BAND),
                ]
            )
    assert "native_child_payload_sha256_mismatch" in str(refusal.value)
    assert not harness.receipt(claim.B37_BUDGET).exists()


@pytest.mark.parametrize("rejudge_encoding", ("compact", "indent2"))
def test_rejudge_binding_accepts_the_bytes_the_child_actually_wrote(
    tmp_path: Path,
    rejudge_encoding: str,
) -> None:
    """The sha binding closes on whatever encoding the child wrote.

    The parent hashes the child's bytes and re-hashes the stored copy of those
    same bytes, so it never has to guess the producer's encoding. Both cells
    matter: ``compact`` is what the production child writes, and ``indent2``
    proves the binding is not merely re-tuned to a second fixed convention.

    Fails against the predecessor, which re-encoded the PARSED payload with
    ``indent=2`` and compared that to the sha of the bytes on disk. That
    could never match a compact producer, so every claim.v2 pair failed
    closed on ``<lane>_rejudge_payload_sha256_mismatch`` — the whole dual
    rejudge path was unrunnable, and untriggered only because no claim.v2
    receipt had ever been minted end to end.
    """

    plan = _plan(tmp_path / "child_payload_paths.txt")
    plan["rejudge_encoding"] = rejudge_encoding
    with pytest.MonkeyPatch.context() as monkeypatch:
        harness = _install(monkeypatch, tmp_path, plan=plan)
        claim.main(harness.b3_argv())
        receipt = harness.read_receipt(claim.B3_BUDGET)
    assert receipt["fail_closed_reason"] is None
