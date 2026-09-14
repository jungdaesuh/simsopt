"""The Boozer-residual kernel reduces over the OpenMP team it actually got.

``src/simsoptpp/boozerresidual_impl.h`` allocates one partial-sum slot per
``omp_get_max_threads()`` and then reduces them in thread order. The team the
runtime actually grants can be smaller than that request (``OMP_THREAD_LIMIT``,
dynamic teams, nesting), so the reduction is bounded by ``omp_get_num_threads()``
read inside the parallel region. Summing the requested count instead is
harmless-looking -- the extra slots are zero -- but the kernel is one edit away
from indexing or accumulating past the granted team, and the OMP-1 case
(``OMP_NUM_THREADS=8`` with ``OMP_THREAD_LIMIT=1``) is the configuration where
requested and granted disagree maximally.

Both legs run in child processes because OpenMP reads its environment when the
extension module is loaded, not per call.

NOTE: the ``.so`` shipped in this tree today is single-threaded in this kernel,
so both legs already take the one-thread path and the test is trivially green.
It is a tracked regression for the OpenMP kernel and starts discriminating the
moment a multi-threaded ``boozerresidual_impl.h`` ships. The test deliberately
uses the installed extension as-is and never swaps a ``.so``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Child program: build a small NCSX-fit surface, call the three kernel entry
# points, and save every returned array. Kept as source text so the OpenMP
# environment is in place before the extension is imported.
_CHILD_PROGRAM = """
import sys
from pathlib import Path

repo_root, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, repo_root)
from repo_bootstrap import bootstrap_local_simsopt

# Same shim tests/conftest.py installs: it is what makes ``simsoptpp``
# importable from this tree, and it does not touch the OpenMP environment.
bootstrap_local_simsopt(Path(repo_root) / "src")

import numpy as np
import simsoptpp as sopp
from simsopt.configs.zoo import get_data
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier

base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
mpol = ntor = 2
surface = SurfaceXYZTensorFourier(
    mpol=mpol,
    ntor=ntor,
    stellsym=True,
    nfp=nfp,
    quadpoints_phi=np.linspace(0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, 2 * mpol + 1, endpoint=False),
)
surface.fit_to_curve(ma, 0.1, flip_theta=True)

iota = -0.406
G = (
    2.0
    * np.pi
    * nfp
    * sum(abs(current.get_value()) for current in base_currents)
    * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
)

field = BiotSavart(bs.coils)
x = surface.gamma()
nphi, ntheta = x.shape[0], x.shape[1]
field.set_points(x.reshape((x.size // 3, 3)).copy())
field.compute(2)
B = field.B().reshape((nphi, ntheta, 3))
dB_dx = field.dB_by_dX().reshape((nphi, ntheta, 3, 3))
d2B_dx2 = field.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
xphi = surface.gammadash1()
xtheta = surface.gammadash2()
dx_dc = surface.dgamma_by_dcoeff()
dxphi_dc = surface.dgammadash1_by_dcoeff()
dxtheta_dc = surface.dgammadash2_by_dcoeff()

arrays = {}
for weight_inv_modB in (False, True):
    tag = "winv" if weight_inv_modB else "plain"
    arrays[tag + "_res"] = np.asarray(
        sopp.boozer_residual(G, iota, xphi, xtheta, B, weight_inv_modB)
    )
    value, gradient = sopp.boozer_residual_ds(
        G, iota, B, dB_dx, xphi, xtheta, dx_dc, dxphi_dc, dxtheta_dc, weight_inv_modB
    )
    arrays[tag + "_ds_value"] = np.asarray(value)
    arrays[tag + "_ds_gradient"] = np.asarray(gradient)
    value2, gradient2, hessian2 = sopp.boozer_residual_ds2(
        G, iota, B, dB_dx, d2B_dx2, xphi, xtheta, dx_dc, dxphi_dc, dxtheta_dc, weight_inv_modB
    )
    arrays[tag + "_ds2_value"] = np.asarray(value2)
    arrays[tag + "_ds2_gradient"] = np.asarray(gradient2)
    arrays[tag + "_ds2_hessian"] = np.asarray(hessian2)

np.savez(out_path, **arrays)
"""

# The kernel's own reduction contract: reordering finite sums only, so the two
# legs must agree to round-off of the reduction, not to the solve tolerance.
_RELATIVE_L2_TOLERANCE = 1.0e-12


def _run_leg(tmp_path: Path, name: str, omp_env: dict[str, str]) -> dict[str, np.ndarray]:
    out_path = tmp_path / f"{name}.npz"
    env = dict(os.environ)
    env.update(omp_env)
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, str(_REPO_ROOT), str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        f"{name} leg failed (env {omp_env}):\n{completed.stdout}\n{completed.stderr}"
    )
    with np.load(out_path) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def _relative_l2(actual: np.ndarray, reference: np.ndarray) -> float:
    reference_norm = float(np.linalg.norm(np.atleast_1d(reference).ravel()))
    difference_norm = float(
        np.linalg.norm(np.atleast_1d(actual).ravel() - np.atleast_1d(reference).ravel())
    )
    if reference_norm == 0.0:
        return difference_norm
    return difference_norm / reference_norm


@pytest.mark.slow
def test_boozer_residual_matches_across_requested_and_granted_team(tmp_path):
    # OMP_NUM_THREADS=8 with OMP_THREAD_LIMIT=1 requests eight partial-sum slots
    # and is granted one; the single-thread leg requests and is granted one.
    # Every returned array must agree.
    granted_one = _run_leg(
        tmp_path, "thread_limit_one", {"OMP_NUM_THREADS": "8", "OMP_THREAD_LIMIT": "1"}
    )
    reference = _run_leg(
        tmp_path, "single_thread", {"OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1"}
    )
    assert set(granted_one) == set(reference)
    for key in sorted(reference):
        error = _relative_l2(granted_one[key], reference[key])
        assert error <= _RELATIVE_L2_TOLERANCE, (
            f"{key}: relative-L2 {error:.3e} between the OMP_THREAD_LIMIT=1 team "
            f"(requested 8) and the single-thread team exceeds "
            f"{_RELATIVE_L2_TOLERANCE:.0e}; the kernel reduction depends on the "
            f"requested thread count rather than the granted team."
        )
