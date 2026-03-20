from pathlib import Path
import sys
import math

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.benchmark_config import DEFAULT_CONFIGS, resolve_configs
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
