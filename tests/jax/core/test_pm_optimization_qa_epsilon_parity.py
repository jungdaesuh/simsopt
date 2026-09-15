"""QA nphi=4 MwPGP epsilon x_sum stop matches native relax-and-split."""

from __future__ import annotations

import io
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

import jax
import numpy as np

from benchmarks.pm_gpmo_probes import GridSpec, _build_qa_grid
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import relax_and_split_jax

# venv site-packages/tests shadows the repo tests package, so the helper
# is imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[2])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from parity_native_cpu import run_native_cpu_child

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Child source is a string so OpenMP is in the environment before the
# extension is imported. ``run_native_cpu_child`` is the SSOT that
# applies ``build_parity_lane_environment`` (``OMP_NUM_THREADS=1``)
# for native-cpu.
_NATIVE_CHILD = """\
import io
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from benchmarks.pm_gpmo_probes import _build_qa_grid
from simsopt.solve.permanent_magnet_optimization import relax_and_split
from simsopt.util.permanent_magnet_helper_functions import initialize_default_kwargs

spec = pickle.loads(Path(sys.argv[1]).read_bytes())
out_path = Path(sys.argv[2])
with redirect_stdout(io.StringIO()):
    grid = _build_qa_grid(spec).grid
    reg_l0, _, _, nu = grid.rescale_for_opt(0.05, 0.0, 0.0, 1.0e10)
ndipoles = int(grid.ndipoles)
initial = np.zeros(ndipoles * 3, dtype=np.float64)
with redirect_stdout(io.StringIO()):
    for stage in range(2):
        kwargs = initialize_default_kwargs()
        kwargs["nu"] = nu
        kwargs["max_iter"] = 10
        kwargs["max_iter_RS"] = 10
        kwargs["reg_l0"] = reg_l0 * (stage + 1) / 2
        kwargs["epsilon"] = 1.0e-3
        kwargs["epsilon_RS"] = 1.0e-3
        kwargs["min_fb"] = 0.0
        kwargs["verbose"] = True
        relax_and_split(grid, m0=initial, **kwargs)
        initial = np.asarray(grid.m, dtype=np.float64)
native_m = np.asarray(grid.m, dtype=np.float64).reshape((ndipoles, 3))
out_path.write_bytes(pickle.dumps(native_m))
"""


def _qa_spec(nphi: int) -> GridSpec:
    return GridSpec(
        nphi=nphi,
        ntheta=nphi,
        downsample=None,
        coordinate_flag="cylindrical",
        dr=0.02,
        inner_offset=0.05,
        outer_offset=0.15,
        source=f"qa epsilon parity nphi={nphi}",
    )


def _qa_pack(nphi: int):
    spec = _qa_spec(nphi)
    with redirect_stdout(io.StringIO()):
        grid = _build_qa_grid(spec).grid
    staged = PermanentMagnetGridJAX.from_cpu(grid)
    jax.block_until_ready((staged.A_obj, staged.b_obj, staged.ATb))
    with redirect_stdout(io.StringIO()):
        reg_l0, _, _, nu = grid.rescale_for_opt(0.05, 0.0, 0.0, 1.0e10)
    alpha = 2.0 * (1.0 - 1.0e-5) / float(grid.ATA_scale)
    return grid, staged, float(alpha), float(nu), float(reg_l0)


def test_qa_nphi4_epsilon_1e_3_matches_native(tmp_path: Path) -> None:
    """Default native epsilon=1e-3 L1 stop at nphi=4, relative error ≤ 1e-6.

    Native MwPGP OpenMP reductions at nphi=4 are not unique under OMP>1
    (relative scatter ~2e-2 on this geometry). ``OMP_NUM_THREADS`` is read
    when libgomp starts, so an in-process env pin cannot undo the pytest
    process team. The native lane therefore runs in a subprocess whose
    environment comes from ``run_native_cpu_child`` /
    ``build_parity_lane_environment`` (the SSOT that sets
    ``OMP_NUM_THREADS=1`` before the child imports the extension).
    """
    spec = _qa_spec(4)
    spec_path = tmp_path / "qa-spec.pkl"
    out_path = tmp_path / "native-m.pkl"
    spec_path.write_bytes(pickle.dumps(spec))
    completed = run_native_cpu_child(
        _NATIVE_CHILD,
        str(spec_path),
        str(out_path),
        repo_root=_REPO_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    native_m = pickle.loads(out_path.read_bytes())

    _, staged, alpha, nu, reg_l0 = _qa_pack(4)
    ndipoles = int(staged.ndipoles)
    jax_initial: object = np.zeros((ndipoles, 3), dtype=np.float64)
    jax_m = None
    for stage in range(2):
        result = relax_and_split_jax(
            staged,
            jax_initial,
            alpha=alpha,
            max_iter=10,
            max_iter_RS=10,
            nu=nu,
            reg_l0=reg_l0 * (stage + 1) / 2,
            epsilon=1.0e-3,
            epsilon_RS=1.0e-3,
        )
        jax.block_until_ready(result.m)
        jax_m = np.asarray(result.m, dtype=np.float64)
        jax_initial = result.m
    assert jax_m is not None
    scale = max(float(np.max(np.abs(native_m))), 1.0e-30)
    rel = float(np.max(np.abs(jax_m - native_m)) / scale)
    assert rel <= 1.0e-6, rel
