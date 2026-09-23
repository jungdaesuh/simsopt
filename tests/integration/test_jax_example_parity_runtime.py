from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import simsoptpp
from examples.jax.parity.provenance import normalize_snapshot_lane_environment
from examples.jax.parity.runtime import build_parity_lane_environment
from simsopt_jax_adapters.isolated_kernel import (
    isolated_child_command,
    loaded_kernel_directory,
    pythonpath_with_loaded_kernel,
)


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
    assert native["OMP_NUM_THREADS"] == "1"
    assert native["OPENBLAS_NUM_THREADS"] == "1"
    assert native["MKL_NUM_THREADS"] == "1"
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
    kernel = str(loaded_kernel_directory())
    for environment in (native, cpu, gpu):
        environment["PYTHONPATH"] = pythonpath_with_loaded_kernel(
            *environment["PYTHONPATH"].split(os.pathsep)
        )
        pythonpath = environment["PYTHONPATH"].split(os.pathsep)
        assert pythonpath[0] == kernel
        assert str(repo_root / "src") in pythonpath


HOST_THREAD_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@pytest.mark.parametrize(
    "inherited",
    (
        pytest.param(
            {name: "64" for name in HOST_THREAD_NAMES}, id="poisoned-parent-values"
        ),
        pytest.param({}, id="absent-parent-values"),
    ),
)
def test_every_parity_lane_pins_the_same_host_threading_policy(
    inherited: dict[str, str],
) -> None:
    """Every compared lane pins host OpenMP/BLAS parallelism to the same value.

    The three lanes are compared against each other numerically.  If one lane
    runs single-threaded and the others inherit the parent shell's thread count,
    the comparison varies with how the run was launched.  Pinning bounds that
    launch-dependent variability; it does not by itself fix every backend's
    reduction order.
    """
    repo_root = Path(__file__).resolve().parents[2]
    original = dict(inherited)
    lanes = ("native-cpu", "jax-cpu", "jax-gpu")
    environments = {
        lane: build_parity_lane_environment(lane, inherited, repo_root=repo_root)
        for lane in lanes
    }
    for lane in lanes:
        environment = environments[lane]
        for name in HOST_THREAD_NAMES:
            assert environment.get(name) == "1", (
                f"{lane} does not pin {name}: {environment.get(name)!r}"
            )
    assert inherited == original, "the caller's environment mapping was mutated"


def test_lane_environment_policy_retains_host_threading() -> None:
    """A retained receipt must say what host threading the lane actually ran under."""
    observed = {
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    retained = normalize_snapshot_lane_environment(observed)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert retained.get(name) == "1", f"{name} is not retained in lane provenance"


def test_loaded_kernel_directory_is_the_parent_of_the_extension() -> None:
    kernel_file = Path(simsoptpp.__file__).resolve()
    assert loaded_kernel_directory() == kernel_file.parent
    assert kernel_file.name.startswith("simsoptpp")


def test_isolated_child_imports_curve_from_the_parent_kernel() -> None:
    completed = subprocess.run(
        isolated_child_command(
            ("-c", "from simsoptpp import Curve; print(Curve.__name__)")
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Curve"


def test_installed_layout_kernel_directory_beats_src_namespace(
    tmp_path: Path,
) -> None:
    """A wheel puts ``simsoptpp*.so`` in site-packages; ``src/simsoptpp/`` must not win."""
    extension = next(loaded_kernel_directory().glob("simsoptpp*.so"))
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / extension.name).symlink_to(extension)
    src = Path(__file__).resolve().parents[2] / "src"
    probe = "from simsoptpp import Curve; print(Curve.__name__)"
    installed = subprocess.run(
        (sys.executable, "-S", "-c", probe),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(site_packages), str(src))),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    shadowed = subprocess.run(
        (sys.executable, "-S", "-c", probe),
        env={**os.environ, "PYTHONPATH": str(src)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    assert installed.stdout.strip() == "Curve"
    assert shadowed.returncode != 0
    assert "Curve" in shadowed.stderr
