"""Exact parity for the ``2_Intermediate/permanent_magnet_QA.py`` mirror."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
from benchmarks.validation_ladder_contract import OPTIMIZER_DRIFT_TOLERANCES
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
# ``OMP_NUM_THREADS=1`` for native-cpu. Receipts cannot carry the
# observation: ``ascontiguousarray`` turns 0-d scalars into shape
# ``(1,)`` and the in-process ``int(...)`` assertions would fail.
_NATIVE_CHILD = """\
import pickle
import sys
from dataclasses import fields
from pathlib import Path

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import read_input_bundle

bundle_root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
observation = get_case("native-permanent-magnet-qa").execute(
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


def test_exact_permanent_magnet_qa_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native lane is a one-thread child; OpenMP scatter is rel ~2e-2.

    Native MwPGP OpenMP reductions at nphi=4 are not unique under OMP>1
    (relative scatter ~2e-2 on this geometry). ``OMP_NUM_THREADS`` is read
    when libgomp starts, so an in-process env pin cannot undo the pytest
    process team. The native lane therefore runs in a subprocess whose
    environment comes from ``run_native_cpu_child`` /
    ``build_parity_lane_environment`` (the SSOT that sets
    ``OMP_NUM_THREADS=1`` before the child imports the extension).
    """
    case = get_case("native-permanent-magnet-qa")
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
    assert native.nit == jax.nit == 2
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:response_matrix",
        "construction:target",
        "construction:moment_maxima",
        "construction:dipole_grid_xyz",
        "initial:moments",
        "initial:residual",
        "initial:objective_sum_squares",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    assert int(native.values["final:nonzero_count"]) > 0
    assert int(jax.values["final:nonzero_count"]) > 0
    final_relative_tolerance = OPTIMIZER_DRIFT_TOLERANCES["tier2_stage2_e2e"][
        "final_objective_rel_tol_20_iter"
    ]
    assert final_relative_tolerance is not None
    for observable in (
        "final:objective_sum_squares",
        "final:residual_norm",
        "final:moment_l2_norm",
        "final:proxy_moment_l2_norm",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=final_relative_tolerance,
            atol=0.0,
        )

    assert np.all(np.isfinite(jax.values["final:moments"]))
    assert np.all(np.isfinite(jax.values["final:proxy_moments"]))
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )
