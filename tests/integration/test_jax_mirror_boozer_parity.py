"""Exact parity for the ``2_Intermediate/boozer.py`` mirror."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle

# venv site-packages/tests shadows the repo tests package, so the helper
# is imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from parity_native_cpu import run_native_cpu_child

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Child source is a string so OpenMP is in the environment before the
# extension is imported. ``run_native_cpu_child`` applies
# ``build_parity_lane_environment``, the SSOT that sets
# ``OMP_NUM_THREADS=1`` for native-cpu. An in-process env pin cannot undo
# the pytest process team; on kernel B that team-dependent L-BFGS warm
# start walks the area-constrained residual onto a second Boozer root
# (iota collapsing toward 0) while JAX and the OMP=1 native lane stay on
# the NCSX-like root near the initial iota -0.4.
_NATIVE_CHILD = """\
import pickle
import sys
from dataclasses import fields
from pathlib import Path

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle

bundle_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
observation = get_case("native-boozer").execute(
    "native-cpu", *read_input_bundle(bundle_root)
)
payload = {field.name: getattr(observation, field.name) for field in fields(observation)}
payload["values"] = dict(observation.values)
payload["applicability"] = dict(observation.applicability)
out_path.write_bytes(pickle.dumps(payload))
"""


def _native_observation(input_root: Path, out_path: Path) -> LaneObservation:
    completed = run_native_cpu_child(
        _NATIVE_CHILD,
        str(input_root),
        str(out_path),
        repo_root=_REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    return LaneObservation(**pickle.loads(out_path.read_bytes()))


def test_exact_boozer_surface_workflow_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native lane is a one-thread child; kernel-B L-BFGS is not unique under OMP>1.

    ``OMP_NUM_THREADS`` is read when libgomp starts, so an in-process env
    pin cannot undo the pytest process team. The native lane therefore
    runs in a subprocess whose environment comes from
    ``run_native_cpu_child`` / ``build_parity_lane_environment`` (the
    SSOT that sets ``OMP_NUM_THREADS=1`` before the child imports the
    extension).
    """
    case = get_case("native-boozer")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = _native_observation(input_root, tmp_path / "native-observation.pkl")

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:axis_dofs",
        "construction:field_dofs",
        "initial:surface_dofs",
        "initial:residual",
        "initial:jacobian",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-11,
            atol=1.0e-13,
        )

    for observable in (
        "area:iota",
        "area:G",
        "area:label",
        "area:residual_norm",
        "flux:target",
        "flux:iota",
        "flux:G",
        "flux:label",
        "flux:residual_norm",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-3,
            atol=1.0e-8,
        )

    np.testing.assert_allclose(
        jax.values["flux:surface_dofs"],
        native.values["flux:surface_dofs"],
        rtol=0.0,
        atol=2.0e-3,
    )
    assert float(native.values["flux:residual_norm"]) < float(
        native.values["initial:residual_norm"]
    )
    assert float(jax.values["flux:residual_norm"]) < float(
        jax.values["initial:residual_norm"]
    )
