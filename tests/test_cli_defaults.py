import logging
import sys
from unittest.mock import patch

import pytest

from simsopt.backend.runtime import BackendConfig, invalidate_backend_cache
from examples.single_stage_optimization.STAGE_2.banana_coil_solver import (
    resolve_stage2_default_optimizer_backend,
    parse_args as stage2_parse_args,
)
from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
    resolve_single_stage_default_optimizer_backend,
    parse_args as single_stage_parse_args,
)


def backend_config(platform: str) -> BackendConfig:
    mode = {"cpu": "jax_cpu_fast", "cuda": "jax_gpu_fast"}[platform]
    return BackendConfig(mode=mode, backend="jax", jax_platform=platform)


@pytest.fixture(autouse=True)
def isolate_backend_runtime(monkeypatch):
    monkeypatch.delenv("STAGE2_OPTIMIZER_BACKEND", raising=False)
    monkeypatch.delenv("OPTIMIZER_BACKEND", raising=False)
    invalidate_backend_cache()
    yield
    invalidate_backend_cache()


@patch("simsopt.backend.runtime.get_backend_config")
def test_resolve_stage2_default_optimizer_backend(mock_get_config):
    mock_get_config.return_value = backend_config("cpu")
    invalidate_backend_cache()
    assert resolve_stage2_default_optimizer_backend("jax") == "scipy-jax-fullgraph"
    assert resolve_stage2_default_optimizer_backend("cpu") == "scipy"

    mock_get_config.return_value = backend_config("cuda")
    invalidate_backend_cache()
    assert resolve_stage2_default_optimizer_backend("jax") == "ondevice"
    assert resolve_stage2_default_optimizer_backend("cpu") == "scipy"

    assert resolve_stage2_default_optimizer_backend("jax", "scipy-jax") == "scipy-jax"


@patch("simsopt.backend.runtime.get_backend_config")
def test_resolve_single_stage_default_optimizer_backend(mock_get_config):
    mock_get_config.return_value = backend_config("cpu")
    invalidate_backend_cache()
    assert resolve_single_stage_default_optimizer_backend("jax") == "scipy-jax-fullgraph"
    assert resolve_single_stage_default_optimizer_backend("cpu") == "scipy"

    mock_get_config.return_value = backend_config("cuda")
    invalidate_backend_cache()
    assert resolve_single_stage_default_optimizer_backend("jax") == "scipy-jax"
    assert resolve_single_stage_default_optimizer_backend("cpu") == "scipy"

    assert resolve_single_stage_default_optimizer_backend("jax", "scipy-jax") == "scipy-jax"


@patch("simsopt.backend.runtime.get_backend_config")
def test_stage2_cpu_ondevice_warning(mock_get_config, caplog):
    mock_get_config.return_value = backend_config("cpu")
    invalidate_backend_cache()

    test_args = [
        "banana_coil_solver.py",
        "--backend", "jax",
        "--optimizer-backend", "ondevice",
    ]

    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            stage2_parse_args()

    assert any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)


@patch("simsopt.backend.runtime.get_backend_config")
def test_stage2_gpu_ondevice_no_warning(mock_get_config, caplog):
    mock_get_config.return_value = backend_config("cuda")
    invalidate_backend_cache()

    test_args = [
        "banana_coil_solver.py",
        "--backend", "jax",
        "--optimizer-backend", "ondevice",
    ]

    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            stage2_parse_args()

    assert not any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)


@patch("simsopt.backend.runtime.get_backend_config")
def test_stage2_parse_args_uses_platform_default(mock_get_config, caplog):
    test_args = ["banana_coil_solver.py", "--backend", "jax"]

    mock_get_config.return_value = backend_config("cpu")
    invalidate_backend_cache()
    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            cpu_args = stage2_parse_args()
    assert cpu_args.optimizer_backend == "scipy-jax-fullgraph"
    assert not any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)

    mock_get_config.return_value = backend_config("cuda")
    invalidate_backend_cache()
    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            gpu_args = stage2_parse_args()
    assert gpu_args.optimizer_backend == "ondevice"
    assert not any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)


@patch("simsopt.backend.runtime.get_backend_config")
def test_single_stage_cpu_ondevice_warning(mock_get_config, caplog):
    mock_get_config.return_value = backend_config("cpu")
    invalidate_backend_cache()

    test_args = [
        "single_stage_banana_example.py",
        "--backend", "jax",
        "--optimizer-backend", "ondevice",
    ]

    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            single_stage_parse_args()

    assert any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)


@patch("simsopt.backend.runtime.get_backend_config")
def test_single_stage_gpu_ondevice_no_warning(mock_get_config, caplog):
    mock_get_config.return_value = backend_config("cuda")
    invalidate_backend_cache()

    test_args = [
        "single_stage_banana_example.py",
        "--backend", "jax",
        "--optimizer-backend", "ondevice",
    ]

    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            single_stage_parse_args()

    assert not any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)


@patch("simsopt.backend.runtime.get_backend_config")
def test_single_stage_parse_args_uses_platform_default(mock_get_config, caplog):
    test_args = ["single_stage_banana_example.py", "--backend", "jax"]

    mock_get_config.return_value = backend_config("cpu")
    invalidate_backend_cache()
    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            cpu_args = single_stage_parse_args()
    assert cpu_args.optimizer_backend == "scipy-jax-fullgraph"
    assert not any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)

    mock_get_config.return_value = backend_config("cuda")
    invalidate_backend_cache()
    caplog.clear()
    with patch.object(sys, "argv", test_args):
        with caplog.at_level(logging.WARNING):
            gpu_args = single_stage_parse_args()
    assert gpu_args.optimizer_backend == "scipy-jax"
    assert not any("WARNING: Running JAX 'ondevice' optimizer on CPU" in record.message for record in caplog.records)
