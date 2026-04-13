import importlib.util
from pathlib import Path
import sys

from simsopt.backend import get_backend_config, invalidate_backend_cache, set_backend


def _load_benchmark_module(name: str, relpath: str):
    module_path = Path(__file__).resolve().parents[1] / relpath
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


biotsavart_donation_probe = _load_benchmark_module(
    "biotsavart_donation_probe",
    "benchmarks/biotsavart_donation_probe.py",
)

TEST_PROBE_SHAPE = biotsavart_donation_probe.DonationProbeShape(
    ncoils=2,
    nquad=8,
    npoints=16,
)


def _build_test_payload(title: str):
    return biotsavart_donation_probe.build_biotsavart_donation_probe_payload(
        title=title,
        mode="jax_cpu_parity",
        shape=TEST_PROBE_SHAPE,
        warmup=0,
        repeat=1,
        seed=0,
    )


def _build_real_stage2_test_payload(title: str):
    return biotsavart_donation_probe.build_biotsavart_donation_probe_payload(
        title=title,
        mode="jax_cpu_parity",
        shape=TEST_PROBE_SHAPE,
        warmup=0,
        repeat=1,
        seed=0,
        fixture="real-stage2",
        stage2_nphi=5,
        stage2_ntheta=4,
    )


def _restore_backend_config(config) -> None:
    invalidate_backend_cache()
    set_backend(
        config.mode,
        strict=config.strict,
        debug_nans=config.debug_nans,
        transfer_guard=config.transfer_guard,
        compilation_cache_dir=config.compilation_cache_dir,
        configure_runtime=False,
    )


def _assert_backend_config_matches(restored, expected) -> None:
    assert restored.mode == expected.mode
    assert restored.strict is expected.strict
    assert restored.debug_nans is expected.debug_nans
    assert restored.transfer_guard == expected.transfer_guard
    assert restored.compilation_cache_dir == expected.compilation_cache_dir


def test_biotsavart_donation_probe_matches_baseline():
    payload = _build_test_payload("Unit test probe")

    assert payload["cases"]["baseline"]["donate_argnums"] == []
    assert payload["cases"]["donate_points"]["donate_argnums"] == [0]
    assert payload["cases"]["baseline"]["public_api_safe"] is True
    assert payload["cases"]["donate_points"]["public_api_safe"] is False
    assert payload["comparison"]["output_shape"] == [16, 3]
    assert payload["comparison"]["max_abs_diff"] == 0.0
    assert payload["comparison"]["max_rel_diff"] == 0.0


def test_biotsavart_donation_probe_supports_real_stage2_fixture():
    payload = _build_real_stage2_test_payload("Real Stage 2 probe")

    assert payload["fixture"]["kind"] == "real-stage2"
    assert payload["fixture"]["point_count"] == 20
    assert payload["cases"]["baseline"]["donate_argnums"] == []
    assert payload["cases"]["donate_points"]["donate_argnums"] == [0]
    assert payload["comparison"]["output_shape"] == [20, 3]
    assert payload["comparison"]["max_abs_diff"] == 0.0
    assert payload["comparison"]["max_rel_diff"] == 0.0


def test_biotsavart_donation_probe_restores_backend_config():
    invalidate_backend_cache()
    original = get_backend_config()
    set_backend(
        "jax_gpu_fast",
        strict=True,
        debug_nans=True,
        transfer_guard="log",
        compilation_cache_dir="/tmp/biotsavart-donation-probe-cache",
        configure_runtime=False,
    )
    previous = get_backend_config()

    try:
        _build_test_payload("Backend restore probe")

        restored = get_backend_config()
        _assert_backend_config_matches(restored, previous)
    finally:
        _restore_backend_config(original)
