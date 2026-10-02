"""Native-vs-JAX parity for the ``2_Intermediate/permanent_magnet_QA.py`` mirror.

The problem is upstream's: the QA boundary with its TF coils, a few coil
pre-optimization iterations (with the boundary fixed, see the last test), the
cylindrical grid between two offset surfaces, then a two-stage relax-and-split
continuation. The native lane is that continuation over
``simsoptpp.MwPGP_algorithm`` and ``prox_l0``; the JAX lane is
``relax_and_split_jax`` on the same frozen arrays.

The native construction and solve run in a child process at
``OMP_NUM_THREADS=1``: native MwPGP OpenMP reductions at this scale are not
unique under OMP>1, and ``OMP_NUM_THREADS`` is read when libgomp starts, so an
in-process pin cannot undo the pytest process team. Measured 2026-09-20 at
``OMP_NUM_THREADS=4``: the SAME bounded input solved twice in ONE process moves
``final:objective_sum_squares`` by 2.672e-04 relative with no perturbation at
all -- 0.53 x the ``mirror_pmqa_final`` rtol the endpoint comparison declares.
The JAX lane runs in the pytest process: it never enters the
``simsoptpp.MwPGP_algorithm`` OpenMP reductions this pin exists for.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import pickle
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from parity_native_cpu import run_native_cpu_child
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax.solve.permanent_magnet import relax_and_split_jax

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NATIVE_EXAMPLE_SOURCE = (
    REPOSITORY_ROOT / "examples" / "2_Intermediate" / "permanent_magnet_QA.py"
)
#: The JAX lane's runtime: the parity backend in FP64.
JAX_LANE_ENVIRONMENT = {
    "SIMSOPT_BACKEND_MODE": "jax_cpu_parity",
    "SIMSOPT_PRECISION": "fp64",
    "JAX_ENABLE_X64": "1",
}
#: The bounded ``final`` endpoint values compared at ``mirror_pmqa_final``.
BOUNDED_FINAL_VALUES = (
    "final:objective_sum_squares",
    "final:residual_norm",
    "final:moment_l2_norm",
    "final:proxy_moment_l2_norm",
)


def _scale_configuration(native_scale: bool) -> dict[str, object]:
    """Upstream's shipped configuration, or its reduced bounded counterpart."""
    return {
        "nphi": 16 if native_scale else 4,
        "ntheta": 16 if native_scale else 4,
        "radial_extent": 0.02 if native_scale else 0.10,
        "coil_iterations": 500 if native_scale else 3,
        "inner_iterations": 10,
        "outer_iterations": 10 if native_scale else 1,
        "continuation_stages": 2,
        "nu": 1.0e10,
        "unscaled_reg_l0": 0.05,
        "coordinate_flag": "cylindrical",
    }


#: Child source is a string so OpenMP is in the environment before the compiled
#: extension is imported. It builds the problem and runs the native lane, and
#: returns the frozen arrays both lanes consume.
_NATIVE_CHILD = """\
import io
import os
import pickle
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import simsoptpp
from simsopt.field import BiotSavart
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.solve.permanent_magnet_optimization import prox_l0
from simsopt.util.coil_optimization_helper_functions import coil_optimization
from simsopt.util.permanent_magnet_helper_functions import (
    initialize_coils_for_pm_optimization,
)

configuration = pickle.loads(Path(sys.argv[1]).read_bytes())
output_root = Path(sys.argv[2])
out_path = Path(sys.argv[3])
output_root.mkdir(parents=True, exist_ok=True)
test_data = Path.cwd() / "tests" / "test_files"
surface_input = test_data / "input.LandremanPaul2021_QA_lowres"
nphi = configuration["nphi"]
ntheta = configuration["ntheta"]


def boundary():
    return SurfaceRZFourier.from_vmec_input(
        surface_input, range="half period", nphi=nphi, ntheta=ntheta
    )


surface = boundary()
inner_surface = boundary()
outer_surface = boundary()
inner_surface.extend_via_projected_normal(0.05)
outer_surface.extend_via_projected_normal(0.15)
# The boundary is this problem's prescribed target, never a variable.
surface.fix_all()
with redirect_stdout(io.StringIO()):
    base_curves, curves, coils = initialize_coils_for_pm_optimization(
        "qa", test_data, surface, output_root
    )
    field = coil_optimization(
        surface,
        BiotSavart(coils),
        base_curves,
        curves,
        MAXITER=configuration["coil_iterations"],
    )
field.set_points(surface.gamma().reshape((-1, 3)))
normal_field = np.sum(
    field.B().reshape((nphi, ntheta, 3)) * surface.unitnormal(), axis=2
)
with redirect_stdout(io.StringIO()):
    grid = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
        surface,
        normal_field,
        inner_surface,
        outer_surface,
        dr=configuration["radial_extent"],
        coordinate_flag=configuration["coordinate_flag"],
    )
ndipoles = int(grid.ndipoles)
arrays = {
    "response_matrix": np.array(grid.A_obj, dtype=np.float64, copy=True),
    "target": np.array(grid.b_obj, dtype=np.float64, copy=True),
    "atb": np.array(grid.ATb, dtype=np.float64, copy=True).reshape((ndipoles, 3)),
    "ata_scale_unregularized": np.array(grid.ATA_scale, dtype=np.float64, copy=True),
    "initial_moments": np.array(grid.m0, dtype=np.float64, copy=True).reshape(
        (ndipoles, 3)
    ),
    "moments": np.array(grid.m, dtype=np.float64, copy=True).reshape((ndipoles, 3)),
    "moment_maxima": np.array(grid.m_maxima, dtype=np.float64, copy=True).reshape(
        (ndipoles,)
    ),
    "dipole_grid_xyz": np.array(grid.dipole_grid_xyz, dtype=np.float64, copy=True),
}

# The two-stage continuation: each stage runs ``outer_iterations`` MwPGP calls
# with ``epsilon = 0`` and ``verbose = False``, so no stage stops early.
nu = configuration["nu"]
base_reg_l0 = configuration["unscaled_reg_l0"] / (2.0 * nu)
ata_scale = arrays["ata_scale_unregularized"].item() + 1.0 / nu
alpha = 2.0 * (1.0 - 1.0e-5) / ata_scale
atb = np.ascontiguousarray(arrays["atb"])
maxima = arrays["moment_maxima"]
initial = arrays["initial_moments"].reshape(-1)
for stage in range(configuration["continuation_stages"]):
    reg_l0 = base_reg_l0 * (stage + 1) / 2.0
    moments = np.array(initial, copy=True)
    proxy = prox_l0(moments, maxima, reg_l0, nu)
    for _outer in range(configuration["outer_iterations"]):
        _, _, _, moment_matrix = simsoptpp.MwPGP_algorithm(
            arrays["response_matrix"],
            arrays["target"],
            atb,
            np.ascontiguousarray(proxy.reshape((ndipoles, 3))),
            np.ascontiguousarray(moments.reshape((ndipoles, 3))),
            maxima,
            alpha,
            nu,
            0.0,
            reg_l0,
            0.0,
            0.0,
            configuration["inner_iterations"],
            0.0,
            False,
        )
        moments = np.asarray(moment_matrix, dtype=np.float64).reshape(-1)
        proxy = prox_l0(moments, maxima, reg_l0, nu)
    initial = moments

out_path.write_bytes(
    pickle.dumps(
        {
            "arrays": arrays,
            "R0": float(grid.R0),
            "nfp": int(grid.plasma_boundary.nfp),
            "stellsym": bool(grid.plasma_boundary.stellsym),
            "ndipoles": ndipoles,
            "native_moments": moments.reshape((ndipoles, 3)),
            "native_proxy_moments": proxy.reshape((ndipoles, 3)),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        }
    )
)
"""


@dataclass(frozen=True)
class _NativeLane:
    """What the one-thread child built and solved."""

    configuration: dict[str, object]
    arrays: dict[str, np.ndarray]
    R0: float
    nfp: int
    stellsym: bool
    ndipoles: int
    values: dict[str, np.ndarray]


def _values(
    arrays: dict[str, np.ndarray],
    final_moments: np.ndarray,
    final_proxy: np.ndarray,
) -> dict[str, np.ndarray]:
    """The endpoint, naming which set of moments each observable uses.

    Upstream carries two moment vectors: ``pm_opt.m`` (the MwPGP iterate) and
    ``pm_opt.m_proxy`` (its sparse prox-l0 companion). ``final:residual``,
    ``final:objective_sum_squares``, ``final:residual_norm`` and the nonzero
    mask/count are computed from ``m_proxy``; ``final:moment_residual`` and
    ``final:moment_objective_sum_squares`` are the same quantities on ``m``.
    """
    response = arrays["response_matrix"]
    target = arrays["target"]
    initial_residual = response @ arrays["initial_moments"].reshape(-1) - target
    final_residual = response @ final_proxy.reshape(-1) - target
    moment_residual = response @ final_moments.reshape(-1) - target
    nonzero_mask = np.linalg.norm(final_proxy, axis=1) != 0.0
    return {
        "initial:objective_sum_squares": np.asarray(
            np.vdot(initial_residual, initial_residual), dtype=np.float64
        ),
        "final:moments": final_moments,
        "final:proxy_moments": final_proxy,
        "final:moment_residual": moment_residual,
        "final:moment_objective_sum_squares": np.asarray(
            np.vdot(moment_residual, moment_residual), dtype=np.float64
        ),
        "final:residual": final_residual,
        "final:objective_sum_squares": np.asarray(
            np.vdot(final_residual, final_residual), dtype=np.float64
        ),
        "final:residual_norm": np.asarray(
            np.linalg.norm(final_residual), dtype=np.float64
        ),
        "final:moment_l2_norm": np.asarray(
            np.linalg.norm(final_moments), dtype=np.float64
        ),
        "final:proxy_moment_l2_norm": np.asarray(
            np.linalg.norm(final_proxy), dtype=np.float64
        ),
        "final:nonzero_mask": nonzero_mask,
        "final:nonzero_count": np.asarray(
            np.count_nonzero(nonzero_mask), dtype=np.int64
        ),
    }


def _native_lane(
    tmp_path_factory: pytest.TempPathFactory, native_scale: bool
) -> _NativeLane:
    root = tmp_path_factory.mktemp(
        "permanent-magnet-qa-native-default" if native_scale else "pm-qa-bounded"
    )
    configuration = _scale_configuration(native_scale)
    configuration_path = root / "configuration.pkl"
    configuration_path.write_bytes(pickle.dumps(configuration))
    payload_path = root / "native.pkl"
    completed = run_native_cpu_child(
        _NATIVE_CHILD,
        str(configuration_path),
        str(root / "inputs"),
        str(payload_path),
        repo_root=REPOSITORY_ROOT,
    )
    assert completed.returncode == 0, completed.stderr
    payload = pickle.loads(payload_path.read_bytes())
    assert payload["omp_num_threads"] == "1", (
        "the native reference lane must run with OpenMP pinned before libgomp "
        f"starts; the child saw OMP_NUM_THREADS={payload['omp_num_threads']!r}"
    )
    arrays = payload["arrays"]
    return _NativeLane(
        configuration=configuration,
        arrays=arrays,
        R0=payload["R0"],
        nfp=payload["nfp"],
        stellsym=payload["stellsym"],
        ndipoles=payload["ndipoles"],
        values=_values(
            arrays, payload["native_moments"], payload["native_proxy_moments"]
        ),
    )


def _jax_lane(
    native: _NativeLane, monkeypatch: pytest.MonkeyPatch
) -> dict[str, np.ndarray]:
    """``relax_and_split_jax`` over the child's frozen arrays, same continuation."""
    for name, value in JAX_LANE_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    arrays = native.arrays
    configuration = native.configuration
    grid = PermanentMagnetGridJAX(
        A_obj=jax.device_put(arrays["response_matrix"]),
        b_obj=jax.device_put(arrays["target"]),
        ATb=jax.device_put(arrays["atb"]),
        ATA_scale=jax.device_put(
            np.asarray(arrays["ata_scale_unregularized"].item(), dtype=np.float64)
        ),
        m0=jax.device_put(arrays["initial_moments"]),
        m=jax.device_put(arrays["moments"]),
        m_proxy=jax.device_put(arrays["moments"]),
        m_maxima=jax.device_put(arrays["moment_maxima"]),
        dipole_grid_xyz=jax.device_put(arrays["dipole_grid_xyz"]),
        coordinate_flag=str(configuration["coordinate_flag"]),
        R0=native.R0,
        nfp=native.nfp,
        stellsym=native.stellsym,
        nphi=int(configuration["nphi"]),
        ntheta=int(configuration["ntheta"]),
        ndipoles=native.ndipoles,
    )
    nu = float(configuration["nu"])
    base_reg_l0 = float(configuration["unscaled_reg_l0"]) / (2.0 * nu)
    alpha = 2.0 * (1.0 - 1.0e-5) / (arrays["ata_scale_unregularized"].item() + 1.0 / nu)
    initial = grid.m0
    result = None
    for stage in range(int(configuration["continuation_stages"])):
        result = relax_and_split_jax(
            grid,
            m0=initial,
            alpha=alpha,
            nu=nu,
            max_iter=int(configuration["inner_iterations"]),
            max_iter_RS=int(configuration["outer_iterations"]),
            reg_l0=base_reg_l0 * (stage + 1) / 2.0,
        )
        initial = result.m
    assert result is not None
    final_moments, final_proxy = jax.device_get((result.m, result.m_proxy))
    return _values(
        arrays,
        np.asarray(final_moments, dtype=np.float64),
        np.asarray(final_proxy, dtype=np.float64),
    )


@pytest.fixture(scope="module")
def bounded_native(tmp_path_factory: pytest.TempPathFactory) -> _NativeLane:
    return _native_lane(tmp_path_factory, native_scale=False)


@pytest.fixture(scope="module")
def shipped_scale_native(tmp_path_factory: pytest.TempPathFactory) -> _NativeLane:
    """The official scale; the whole construction costs about half a minute."""
    return _native_lane(tmp_path_factory, native_scale=True)


def test_exact_permanent_magnet_qa_matches_native_and_jax_cpu(
    bounded_native: _NativeLane,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = bounded_native.values
    jax_values = _jax_lane(bounded_native, monkeypatch)

    assert set(native) == set(jax_values)
    assert int(native["final:nonzero_count"]) > 0
    assert int(jax_values["final:nonzero_count"]) > 0
    # What the two lanes must agree on at this scale is the SET of magnets the
    # relax-and-split continuation kept; the endpoint VALUES are compared at
    # their own bucket by the next test.
    assert int(native["final:nonzero_count"]) == int(jax_values["final:nonzero_count"])
    assert np.array_equal(
        np.asarray(native["final:nonzero_mask"]),
        np.asarray(jax_values["final:nonzero_mask"]),
    )

    assert np.all(np.isfinite(native["final:moments"]))
    assert np.all(np.isfinite(native["final:proxy_moments"]))
    assert np.all(np.isfinite(jax_values["final:moments"]))
    assert np.all(np.isfinite(jax_values["final:proxy_moments"]))
    assert float(native["final:objective_sum_squares"]) < float(
        native["initial:objective_sum_squares"]
    )


def test_the_bounded_endpoint_values_compare_inside_their_bucket(
    bounded_native: _NativeLane,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded endpoint values, native one-thread lane against JAX.

    Every value is determined to this bucket by its own input: over eight
    one-ulp perturbations of ``ATb`` the worst native drift was 7.928e-11
    (objective), 3.964e-11 (residual norm), 5.818e-12 (moment norm) and
    1.250e-11 (proxy moment norm) against rtol 5e-4 (measured 2026-09-29).
    Only agreement is asserted -- a required disagreement would fail the day
    two lanes agree better.
    """
    jax_values = _jax_lane(bounded_native, monkeypatch)
    bucket = parity_ladder_tolerances("mirror_pmqa_final")

    for observable in BOUNDED_FINAL_VALUES:
        np.testing.assert_allclose(
            float(np.asarray(jax_values[observable])),
            float(np.asarray(bounded_native.values[observable])),
            rtol=bucket["rtol"],
            atol=bucket["atol"],
            err_msg=observable,
        )


def test_permanent_magnet_qa_shipped_scale_jax_matches_native(
    shipped_scale_native: _NativeLane,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ``m``-based and proxy endpoint at upstream's shipped scale.

    At ``nphi=ntheta=16`` with upstream's own coil ``MAXITER=500`` the problem
    is the official run's. The JAX lane is judged against the one-thread native
    lane, at the tolerance the official comparison used for both lanes.
    """
    native = shipped_scale_native.values
    jax_values = _jax_lane(shipped_scale_native, monkeypatch)

    np.testing.assert_allclose(
        0.5 * float(jax_values["final:moment_objective_sum_squares"]),
        0.5 * float(native["final:moment_objective_sum_squares"]),
        rtol=1.0e-8,
        atol=0.0,
    )
    np.testing.assert_allclose(
        0.5 * float(jax_values["final:objective_sum_squares"]),
        0.5 * float(native["final:objective_sum_squares"]),
        rtol=1.0e-8,
        atol=0.0,
    )
    np.testing.assert_allclose(
        float(jax_values["final:moment_l2_norm"]),
        float(native["final:moment_l2_norm"]),
        rtol=1.0e-8,
        atol=0.0,
    )
    assert int(jax_values["final:nonzero_count"]) == int(native["final:nonzero_count"])
    # The two ``m``-based comparisons are applicable at this scale; the residual
    # is the array they compare, at the whole-solve value bucket.
    native_workflow = parity_ladder_tolerances("native_workflow")
    np.testing.assert_allclose(
        np.asarray(jax_values["final:moment_residual"], dtype=np.float64),
        np.asarray(native["final:moment_residual"], dtype=np.float64),
        rtol=native_workflow["whole_solve_value_rtol"],
        atol=native_workflow["whole_solve_value_atol"],
    )


def test_the_native_example_fixes_the_boundary_before_the_coil_optimization() -> None:
    """The shipped native script carries the same adaptation as its siblings.

    Branch commit ``28e470f43`` gave ``CurveSurfaceDistance`` ownership of the
    surface, so the plasma boundary's free ``SurfaceRZFourier`` dofs enter
    ``coil_optimization``'s ``JF.x`` unless the caller fixes the boundary.
    ``examples/2_Intermediate/permanent_magnet_QA.py`` is a module-level
    program, so the order of the two statements is read from its bytes.
    """

    source = NATIVE_EXAMPLE_SOURCE.read_text(encoding="utf-8")
    fix_index = source.index("s.fix_all()")
    call_index = source.index("coil_optimization(s, bs, base_curves, curves)")
    assert fix_index < call_index
