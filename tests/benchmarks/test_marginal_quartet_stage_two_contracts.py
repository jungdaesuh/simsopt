"""Contracts the marginal-quartet probe must hold for its mirror families.

Everything here runs without JAX, without a GPU, and without executing a leg:
the probe's own rule is that a numerical import before the environment pin makes
the pin cosmetic, and a test that imported JAX to check a JSON shape would both
break that rule and put a context on a box that may be running someone's timed
leg.  What is checked is therefore the probe's bookkeeping -- how it reads the
native policy, how it compares the two lanes, what it attests about a device,
and what it keeps instead of deleting -- against stand-ins and against the live
repository sources it claims to be reading.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from scipy.optimize import fmin_l_bfgs_b

from benchmarks.marginal_quartet_probes import (
    COMPARED_LBFGSB_OPTIONS,
    ENDPOINT_VECTOR_OBSERVABLES,
    MIRROR_FAMILIES,
    SCIPY_LBFGSB_DEFAULTS,
    ProbeConventionError,
    _endpoint_root,
    _execution_device_attestation,
    _minimize_region,
    _mirror_binding,
    _mirror_native_matched_options,
    _mirror_optimizer_options,
    _module_level_constants,
    _native_optimizer_options,
    _policy_comparison,
    _retain_native_outputs,
    _smoke_attestation,
    _write_endpoint_archive,
)

PROBE = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "marginal_quartet_probes.py"
)
STAGE_TWO = MIRROR_FAMILIES["stage-two"]
COIL_FORCES = MIRROR_FAMILIES["coil-forces"]

#: What the standard Stage-II mirror's SciPy route builds, one stage's worth.
#: Kept here as a literal on purpose: the point of the comparison is that two
#: independently derived descriptions of the same rule agree.
MIRROR_STAGE_OPTIONS = {
    "type": "ScipyLBFGSBOptions",
    "maxiter": 400,
    "maxcor": 300,
    "ftol": 1.0e-15,
    "gtol": 1.0e-15,
    "maxfun": 15000,
    "maxls": 20,
}


class _FakeDevice:
    def __init__(self, label: str, platform: str) -> None:
        self._label = label
        self.platform = platform

    def __str__(self) -> str:
        return self._label


class _FakeJax:
    """Enough of the JAX module surface for the attestation, and nothing more."""

    @staticmethod
    def default_backend() -> str:
        return "gpu"

    @staticmethod
    def devices() -> list[_FakeDevice]:
        return [_FakeDevice("cuda:0", "cuda")]


def test_scipy_defaults_match_scipys_own_signature() -> None:
    """The fallback for unnamed options must be SciPy's, not a remembered number."""
    signature = inspect.signature(fmin_l_bfgs_b)

    assert SCIPY_LBFGSB_DEFAULTS == {
        "maxfun": signature.parameters["maxfun"].default,
        "maxls": signature.parameters["maxls"].default,
    }


@pytest.mark.parametrize("family_name", sorted(MIRROR_FAMILIES))
def test_native_policy_is_parsed_from_the_native_script(family_name: str) -> None:
    """Every mirror family's native lane stops under one rule, read from its file."""
    family = MIRROR_FAMILIES[family_name]
    options = _native_optimizer_options(family)
    constants = _module_level_constants(
        ast.parse(family.native_script.read_text(encoding="utf-8"))
    )
    expected_maxls = constants.get("MAXLS", SCIPY_LBFGSB_DEFAULTS["maxls"])

    assert options["maxiter"] == family.native_budget
    assert options["maxcor"] == 300
    assert options["ftol"] == 1.0e-15
    assert options["gtol"] == 1.0e-15
    assert options["maxfun"] == SCIPY_LBFGSB_DEFAULTS["maxfun"]
    assert options["maxls"] == expected_maxls
    assert not isinstance(options["maxls"], str)
    assert options["tol_argument"] == 1.0e-15


def test_coil_forces_native_maxls_is_resolved_from_the_named_constant() -> None:
    """Commit 2b14e7d02 named MAXLS=32; the receipt must publish 32, not a symbol."""
    options = _native_optimizer_options(COIL_FORCES)
    constants = _module_level_constants(
        ast.parse(COIL_FORCES.native_script.read_text(encoding="utf-8"))
    )

    assert options["maxls"] == constants["MAXLS"]
    assert options["maxls"] != SCIPY_LBFGSB_DEFAULTS["maxls"]
    assert options["maxls_symbol"] == "<symbol MAXLS>"
    assert not str(options["maxls"]).startswith("<symbol")


def test_coil_forces_native_and_jax_native_matched_attestations_agree() -> None:
    """Native constants and ScipyLBFGSBOptions.native_matched are one stopping rule."""
    native = _native_optimizer_options(COIL_FORCES)
    declared = _mirror_native_matched_options(COIL_FORCES)

    assert declared is not None
    assert declared["type"] == "ScipyLBFGSBOptions"
    for name in COMPARED_LBFGSB_OPTIONS:
        assert native[name] == declared[name], name
    assert native["maxls"] == declared["maxls"] != SCIPY_LBFGSB_DEFAULTS["maxls"]


def test_stage_two_mirror_does_not_declare_native_matched_at_the_example() -> None:
    assert _mirror_native_matched_options(STAGE_TWO) is None


def test_smoke_attestation_matches_for_coil_forces() -> None:
    report = _smoke_attestation(COIL_FORCES)

    assert report["smoke_policy_matched"] is True
    assert report["policy_differences"] == {}
    assert (
        report["native_optimizer_options"]["maxls"]
        == report["declared_mirror_optimizer_options"]["maxls"]
    )


def test_smoke_attestation_refuses_a_mirror_without_solve_scalar_stage() -> None:
    with pytest.raises(ProbeConventionError, match="solve_scalar_stage"):
        _smoke_attestation(STAGE_TWO)


def test_policy_matched_is_true_when_both_lanes_carry_the_same_rule() -> None:
    comparison = _policy_comparison(
        STAGE_TWO, [dict(MIRROR_STAGE_OPTIONS), dict(MIRROR_STAGE_OPTIONS)]
    )

    assert comparison["policy_matched"] is True
    assert comparison["policy_differences"] == {}


def test_policy_matched_is_false_against_the_old_evaluation_cap() -> None:
    """The mirror's former ``maxfun = 20 * maxiter`` is a policy difference.

    This is the finding the probe used to hard-code away: with ``policy_matched``
    pinned to ``False`` the artifact said the same thing whether the lanes agreed
    or not, so neither this case nor the matched one above was distinguishable.
    """
    capped = {**MIRROR_STAGE_OPTIONS, "maxfun": 8000}
    comparison = _policy_comparison(STAGE_TWO, [capped, capped])

    assert comparison["policy_matched"] is False
    assert comparison["policy_differences"] == {
        "maxfun": {"native": 15000, "mirror": 8000}
    }


def test_policy_matched_is_null_when_a_lane_was_not_observed() -> None:
    """A lane that published no options has not been shown to differ."""
    comparison = _policy_comparison(STAGE_TWO, None)

    assert comparison["policy_matched"] is None
    assert "policy_matched_unavailable_because" in comparison
    assert comparison["mirror_optimizer_options"] is None


def test_policy_matched_is_never_a_literal_in_the_probe() -> None:
    assert '"policy_matched": False' not in PROBE.read_text(encoding="utf-8")


def test_stages_that_ran_under_different_rules_do_not_match() -> None:
    comparison = _policy_comparison(
        STAGE_TWO,
        [dict(MIRROR_STAGE_OPTIONS), {**MIRROR_STAGE_OPTIONS, "maxcor": 10}],
    )

    assert comparison["policy_matched"] is False
    assert "stages_disagree" in comparison["policy_differences"]


def test_mirror_options_come_from_the_leg_or_not_at_all() -> None:
    assert _mirror_optimizer_options({}) is None
    assert _mirror_optimizer_options({"solver_options": ()}) is None
    assert _mirror_optimizer_options({"solver_options": ["not a mapping"]}) is None
    assert _mirror_optimizer_options(
        {"solver_options": (MIRROR_STAGE_OPTIONS, MIRROR_STAGE_OPTIONS)}
    ) == [MIRROR_STAGE_OPTIONS, MIRROR_STAGE_OPTIONS]


def test_compared_options_cover_the_whole_stopping_rule() -> None:
    """``tol`` is deliberately absent: the solver reads ``ftol`` and ``gtol``."""
    assert set(COMPARED_LBFGSB_OPTIONS) == {
        "maxiter",
        "maxcor",
        "ftol",
        "gtol",
        "maxfun",
        "maxls",
    }


@pytest.mark.parametrize("family_name", sorted(MIRROR_FAMILIES))
def test_mirror_binding_is_derived_from_the_mirror_not_declared(
    family_name: str,
) -> None:
    """Thin-wrapper status is read off the mirror, so a promotion cannot go unnoticed.

    While a mirror owns its own problem construction there is no library callable
    to bind to, and the probe says so in the artifact.  The day a mirror becomes
    a thin wrapper this assertion fails and the family should be rebound to that
    callable instead of being loaded from a path.
    """
    family = MIRROR_FAMILIES[family_name]
    binding = _mirror_binding(family)

    assert binding["mechanism"] == "path_load"
    assert binding["mirror_is_thin_wrapper"] is False
    assert "_build_problem" in binding["mirror_module_level_functions"]
    assert "not a Python identifier" in binding["static_import_blocked_because"]
    assert binding["loaded_from"] == str(
        family.mirror_module.relative_to(PROBE.parents[1])
    )


def test_minimize_region_uses_the_mirrors_own_clock() -> None:
    region = _minimize_region(
        {
            "two_stage_minimize_seconds": 310.0,
            "first_stage_minimize_seconds": 200.0,
            "second_stage_minimize_seconds": 100.0,
        }
    )

    assert region["minimize_region_seconds"] == 310.0
    assert region["first_stage_minimize_seconds"] == 200.0
    assert (
        "inter-stage state evaluation included" in region["minimize_region_definition"]
    )


def test_minimize_region_is_null_rather_than_substituted() -> None:
    """A mirror without the clock must not be given ``solve_call_seconds`` instead."""
    region = _minimize_region({"solve_call_seconds": 91.0})

    assert region["minimize_region_seconds"] is None
    assert (
        "publishes no two_stage_minimize_seconds"
        in region["minimize_region_unavailable_because"]
    )


def test_execution_device_attestation_names_the_result_array_device() -> None:
    attestation = _execution_device_attestation(
        _FakeJax, {"execution_device": "cuda:0"}
    )

    assert attestation["result_array_device"] == "cuda:0"
    assert attestation["result_array_device_unavailable_because"] is None
    assert attestation["process_default_backend"] == "gpu"
    assert attestation["process_devices"] == ["cuda:0"]
    assert attestation["process_device_platforms"] == ["cuda"]


def test_process_backend_is_never_promoted_into_an_attestation() -> None:
    """Without a mirror attestation the result-array device stays unknown."""
    attestation = _execution_device_attestation(_FakeJax, {})

    assert attestation["result_array_device"] is None
    assert (
        "publishes no execution_device"
        in attestation["result_array_device_unavailable_because"]
    )
    assert attestation["process_default_backend"] == "gpu"


def test_endpoint_archive_keeps_every_vector_at_full_precision(tmp_path: Path) -> None:
    import numpy as np

    solution = np.asarray([1.0 / 3.0, np.nextafter(1.0, 2.0)], dtype=np.float64)
    published = _write_endpoint_archive(
        tmp_path / "endpoints", "jax-gpu", {"leg0:solution": solution}
    )
    archive = Path(published["endpoint_archive"])

    assert published["endpoint_archive_keys"] == ["leg0:solution"]
    assert len(published["endpoint_archive_sha256"]) == 64
    with np.load(archive) as loaded:
        np.testing.assert_array_equal(loaded["leg0:solution"], solution)


def test_endpoint_archive_says_so_when_a_mirror_published_no_vector() -> None:
    published = _write_endpoint_archive(None, "jax-gpu", {})

    assert published["endpoint_archive"] is None
    assert "no vector observable" in published["endpoint_archive_unavailable_because"]


def test_endpoint_vector_names_cover_both_ends_of_a_run() -> None:
    assert "initial_parameters" in ENDPOINT_VECTOR_OBSERVABLES
    assert "solution" in ENDPOINT_VECTOR_OBSERVABLES


def test_endpoint_root_is_named_after_the_artifact(tmp_path: Path) -> None:
    root = _endpoint_root(tmp_path / "stage_two_jax.json")

    assert root == (tmp_path / "stage_two_jax-endpoints").resolve()
    assert _endpoint_root(None) is None


def test_native_outputs_are_retained_instead_of_deleted(tmp_path: Path) -> None:
    """The child's saved field survives its temporary workdir, with a digest."""
    workdir = tmp_path / "child"
    (workdir / "output").mkdir(parents=True)
    (workdir / "output" / "biot_savart_opt.json").write_text(
        json.dumps({"@class": "SIMSON"}), encoding="utf-8"
    )
    (workdir / "output" / "curves_opt_long.vts").write_text("vtk", encoding="utf-8")

    retained: dict[str, Any] = _retain_native_outputs(
        STAGE_TWO, workdir, tmp_path / "retained", 0
    )

    assert retained["workdir_file_count"] == 2
    assert {entry["path"] for entry in retained["workdir_file_manifest"]} == {
        "output/biot_savart_opt.json",
        "output/curves_opt_long.vts",
    }
    assert all(
        len(entry["sha256"]) == 64 for entry in retained["workdir_file_manifest"]
    )
    copied = Path(retained["saved_field_retained"])
    assert copied.name == "leg0-biot_savart_opt.json"
    assert copied.exists()
    assert (
        "does not promise equals res.x"
        in retained["saved_field_is_not_the_optimizer_endpoint"]
    )


def test_a_run_that_saved_nothing_says_so(tmp_path: Path) -> None:
    workdir = tmp_path / "child"
    workdir.mkdir()

    retained = _retain_native_outputs(STAGE_TWO, workdir, tmp_path / "retained", 1)

    assert retained["saved_field_retained"] is None
    assert retained["workdir_file_count"] == 0
    assert (
        "wrote no biot_savart_opt.json" in retained["saved_field_unavailable_because"]
    )


def test_jax_gpu_lane_pins_its_platform_before_the_mirror_imports() -> None:
    """``src/simsopt/geo/jit.py`` pins JAX to the CPU when neither ``JAX_PLATFORMS``
    nor ``JAX_PLATFORM_NAME`` is set and a mirror imports native simsopt before
    ``simsopt_jax``, so a jax-gpu leg that left the platform unset ran on the CPU
    for every mirror that places on the process default (coil-forces, 2026-09-13).
    The lane must pin the platform itself, as a literal, for both lanes."""
    tree = ast.parse(PROBE.read_text(encoding="utf-8"), filename=str(PROBE))
    pin = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_pin_process_environment"
    )
    keyword = next(
        keyword
        for node in ast.walk(pin)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "pinned_environment"
        for keyword in node.keywords
        if keyword.arg == "jax_platforms"
    )
    assert isinstance(keyword.value, ast.IfExp)
    native, mirror = keyword.value.body, keyword.value.orelse
    assert isinstance(native, ast.Constant) and native.value == "cpu"
    assert isinstance(mirror, ast.Constant) and isinstance(mirror.value, str)
    assert mirror.value.split(",")[0] == "cuda", mirror.value
