from pathlib import Path
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_config import DEFAULT_CONFIGS, resolve_configs
from benchmarks.benchmark_problem import (
    build_ls_parity_problem,
    build_synthetic_boozer_problem,
    clone_tensor_surface,
)
from benchmarks.run_code_benchmark_common import summarize_result_fun


def test_resolve_configs_defaults_to_all_configs():
    assert resolve_configs(None) == DEFAULT_CONFIGS


def test_resolve_configs_preserves_requested_order():
    labels = [
        "Columbia (12 coils, 128x64)",
        "Small (4 coils, 15x15)",
    ]
    configs = resolve_configs(labels)
    assert [config.label for config in configs] == labels


def test_resolve_configs_rejects_unknown_labels():
    with pytest.raises(ValueError, match="Unknown benchmark config"):
        resolve_configs(["does-not-exist"])


def test_summarize_result_fun_prefers_fun():
    assert summarize_result_fun({"fun": np.float64(1.25)}) == 1.25


def test_summarize_result_fun_falls_back_to_residual_norm():
    residual = np.array([1.0, 2.0, 3.0])
    expected = 0.5 * float(np.mean(np.square(residual)))
    assert summarize_result_fun({"residual": residual}) == expected


def test_summarize_result_fun_returns_nan_without_fun_or_residual():
    assert math.isnan(summarize_result_fun({}))


def test_build_synthetic_boozer_problem_uses_requested_grid():
    config = DEFAULT_CONFIGS[0]
    problem = build_synthetic_boozer_problem(config)

    assert len(problem.surface.quadpoints_phi) == config.nphi
    assert len(problem.surface.quadpoints_theta) == config.ntheta
    assert problem.surface.stellsym is False
    assert problem.surface.nfp == config.nfp
    assert problem.iota0 == pytest.approx(0.3)
    assert problem.G0 > 0.0


def test_build_ls_parity_problem_matches_known_good_fixture_shape():
    problem = build_ls_parity_problem()

    assert problem.surface.stellsym is True
    assert problem.surface.nfp == 2
    assert len(problem.surface.quadpoints_phi) == 5
    assert len(problem.surface.quadpoints_theta) == 5
    assert problem.iota0 == pytest.approx(0.3)
    assert problem.G0 > 0.0


def test_clone_tensor_surface_is_independent():
    problem = build_ls_parity_problem()
    surface_copy = clone_tensor_surface(problem.surface)

    np.testing.assert_allclose(surface_copy.get_dofs(), problem.surface.get_dofs())
    new_dofs = surface_copy.get_dofs().copy()
    new_dofs[0] += 1.0
    surface_copy.set_dofs(new_dofs)

    assert surface_copy is not problem.surface
    assert surface_copy.get_dofs()[0] != pytest.approx(problem.surface.get_dofs()[0])
