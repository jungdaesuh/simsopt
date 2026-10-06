"""``set_backend`` makes its platform JAX's default after a native import.

Without JAX platform variables, ``simsopt.geo`` sets the legacy
``jax_platform_name="cpu"``, which JAX consults before ``jax_platforms``; each
case runs in a fresh process so that import happens first.
"""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import os
import subprocess
import sys
from pathlib import Path

import jax
import pytest

_SRC = str(Path(__file__).resolve().parents[2] / "src")
_GPU_DETERMINISM_FLAG = "--xla_gpu_exclude_nondeterministic_ops=true"


def _fresh_environment() -> dict[str, str]:
    """The parent environment without JAX or SIMSOPT selectors.

    GPU allocator variables are pinned to the GPU modes' default (no
    preallocation) because the child imports JAX before ``set_backend``.
    """
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("JAX_", "SIMSOPT_", "XLA_", "TF_GPU_"))
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (_SRC, os.environ.get("PYTHONPATH")) if path
    )
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    environment["XLA_FLAGS"] = _GPU_DETERMINISM_FLAG
    return environment


_STUBBED_CUDA_CHILD = """
import os
import sys

import jax
from jax._src import xla_bridge
from simsopt import geo  # noqa: F401
from simsopt_jax.backend import set_backend

assert jax.config.values["jax_platform_name"] == "cpu"
platforms_after_import = sys.argv[1]
if platforms_after_import:
    os.environ["JAX_PLATFORMS"] = platforms_after_import


class _Backend:
    def __init__(self, platform):
        self.platform = platform


cuda = _Backend("gpu")
backends = {"cuda": cuda}
if platforms_after_import:
    backends["cpu"] = _Backend("cpu")
xla_bridge.backends = lambda: backends
xla_bridge._default_backend = cuda

set_backend("jax", device="gpu", intent="parity")
assert jax.config.jax_platforms == (platforms_after_import or "cuda")
assert jax.config.values["jax_platform_name"] == ""
assert jax.default_backend() == "gpu"
"""


@pytest.mark.parametrize("platforms_after_import", [None, "cuda,cpu"])
def test_set_backend_default_platform_after_native_import_in_fresh_process(
    platforms_after_import,
):
    """A stubbed CUDA backend table resolves the default as a CUDA host would.

    The child replaces JAX's backend table with a CUDA backend (plus CPU when
    listed), so a CPU-only host checks the resolution without a GPU. Before the
    fix it was ``Unknown backend cpu`` or, with ``cuda,cpu``, CPU while CUDA
    was requested.
    """
    subprocess.run(
        [sys.executable, "-c", _STUBBED_CUDA_CHILD, platforms_after_import or ""],
        env=_fresh_environment(),
        check=True,
        timeout=300,
    )


_NATIVE_STAGE_TWO_CHILD = """
import sys

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveLength, SurfaceRZFourier, create_equally_spaced_curves
from simsopt.geo.jit import native_jax_device
from simsopt.objectives import SquaredFlux
from simsopt_jax.backend import get_runtime_jax_device, set_backend
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

device, native_platform, platform_env = sys.argv[1:]
assert jax.config.values["jax_platform_name"] == ("" if platform_env else "cpu")
set_backend("jax", device=device, intent="parity")
assert jax.config.values["jax_platform_name"] == ""
assert jax.default_backend() == device
assert jnp.ones(2).device.platform == device
runtime_device = get_runtime_jax_device()
assert runtime_device.platform == device
assert native_jax_device().platform == native_platform

curve = create_equally_spaced_curves(
    1, 1, False, R0=1.0, R1=0.5, order=2, numquadpoints=32
)[0]
length = CurveLength(curve)
with jax.transfer_guard("disallow"):
    length_value, length_gradient = length.J(), length.dJ()
assert isinstance(length_value, np.float64)
assert isinstance(length_gradient, np.ndarray)
np.testing.assert_allclose(
    length_value,
    np.mean(np.linalg.norm(curve.gammadash(), axis=1)),
    rtol=1e-14,
    atol=1e-14,
)

surface = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=8)
surface.set_rc(0, 0, 1.0)
surface.set_rc(1, 0, 0.3)
surface.set_zs(1, 0, 0.3)
coils = [Coil(curve, Current(1e5))]
native, adapter = BiotSavart(coils), BiotSavartJAX(coils)
native_flux, adapter_flux = SquaredFlux(surface, native), SquaredFlux(surface, adapter)
adapter.set_points(surface.gamma().reshape((-1, 3)))
value = adapter.B()
assert value.devices() == {runtime_device}
native.set_points(surface.gamma().reshape((-1, 3)))
np.testing.assert_allclose(jax.device_get(value), native.B(), rtol=1e-12, atol=1e-14)
np.testing.assert_allclose(adapter_flux.J(), native_flux.J(), rtol=1e-12, atol=1e-14)
np.testing.assert_allclose(adapter_flux.dJ(), native_flux.dJ(), rtol=1e-12, atol=1e-14)
assert jax.default_backend() == device
assert jnp.ones(2).device.platform == device
"""


@pytest.mark.parametrize(
    ("device", "jax_platforms"),
    [("cpu", "cpu"), ("gpu", "cuda,cpu"), ("gpu", None)],
    ids=["cpu", "gpu-cuda-cpu-platforms", "gpu-no-platform-env"],
)
def test_native_stage_two_after_set_backend_in_fresh_process(
    tmp_path, device, jax_platforms
):
    """Native stage II after ``set_backend``.

    ``gpu-no-platform-env`` exports no JAX platform variable, so
    ``simsopt.geo`` sets ``cpu`` before ``set_backend`` selects the GPU, and
    the native length kernels run on CUDA alone.
    """
    if device == "gpu" and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA JAX device required")
    environment = _fresh_environment()
    environment["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = str(
        tmp_path / "compilation-cache"
    )
    if jax_platforms is not None:
        environment["JAX_PLATFORMS"] = jax_platforms
    native_platform = "gpu" if jax_platforms is None else "cpu"
    subprocess.run(
        [
            sys.executable,
            "-c",
            _NATIVE_STAGE_TWO_CHILD,
            device,
            native_platform,
            jax_platforms or "",
        ],
        env=environment,
        check=True,
        timeout=600,
    )


_GUARDED_PLATFORM_NAME_MODULE = """
from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
from simsopt_jax.backend import set_backend


def test_a_set_backend_clears_the_platform_name():
    assert jax.config.values["jax_platform_name"] == "cpu"
    set_backend("jax_cpu_parity")
    assert jax.config.values["jax_platform_name"] == ""


def test_b_the_guard_restored_the_platform_name():
    assert jax.config.values["jax_platform_name"] == "cpu"
"""


def test_runtime_guard_restores_the_platform_name(tmp_path):
    """``jax_runtime_guard`` undoes the ``jax_platform_name`` that ``set_backend`` clears.

    The two tests run in order in a fresh pytest session whose JAX starts with
    ``jax_platform_name="cpu"`` (as after ``simsopt.geo`` without platform
    variables); without the restore the second test sees ``""``.
    """
    module = tmp_path / "test_guarded_platform_name.py"
    module.write_text(_GUARDED_PLATFORM_NAME_MODULE)
    environment = _fresh_environment()
    environment.update(JAX_PLATFORMS="cpu", JAX_PLATFORM_NAME="cpu")
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(Path(__file__).resolve().parents[1]), environment["PYTHONPATH"])
    )
    completed = subprocess.run(
        (sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(module)),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
