from __future__ import annotations

from pathlib import Path

from examples.jax.parity.runtime import build_parity_lane_environment


def test_parity_runtime_uses_fail_closed_preimport_lane_policy() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    poisoned = {
        "JAX_PLATFORMS": "poison",
        "JAX_ENABLE_X64": "0",
        "SIMSOPT_BACKEND_STRICT": "0",
        "PRESERVED": "yes",
    }

    native = build_parity_lane_environment("native-cpu", poisoned, repo_root=repo_root)
    cpu = build_parity_lane_environment("jax-cpu", poisoned, repo_root=repo_root)
    gpu = build_parity_lane_environment("jax-gpu", poisoned, repo_root=repo_root)

    assert native["SIMSOPT_BACKEND_MODE"] == "native_cpu"
    assert native["SIMSOPT_PRECISION"] == "fp64"
    assert native["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu["JAX_PLATFORMS"] == "cpu"
    assert cpu["SIMSOPT_BACKEND_MODE"] == "jax_cpu_parity"
    assert cpu["JAX_TRANSFER_GUARD"] == "allow"
    assert gpu["JAX_PLATFORMS"] == "cuda"
    assert gpu["SIMSOPT_BACKEND_MODE"] == "jax_gpu_parity"
    assert gpu["SIMSOPT_JAX_TRANSFER_GUARD"] == "disallow"
    assert gpu["JAX_TRANSFER_GUARD"] == "disallow"
    assert gpu["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert cpu["JAX_ENABLE_X64"] == gpu["JAX_ENABLE_X64"] == "1"
    assert native["PRESERVED"] == cpu["PRESERVED"] == gpu["PRESERVED"] == "yes"
    assert str(repo_root / "src") == cpu["PYTHONPATH"].split(":", maxsplit=1)[0]
