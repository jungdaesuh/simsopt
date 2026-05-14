import numpy as np
import jax

from simsopt._core.optimizable import Optimizable
from simsopt.jax_core.mhd_reductions import (
    boozer_quasisymmetry_mode_indices,
    boozer_quasisymmetry_residuals,
)
from simsopt.mhd.boozer import Quasisymmetry


class _FrozenBoozXform:
    def __init__(self, mpol: int, ntor: int, nfp: int):
        self.bmnc_b, self.xm_b, self.xn_b = _mock_boozer_spectrum(mpol, ntor, nfp)
        self.nfp = nfp


class _FrozenBoozer(Optimizable):
    def __init__(self, mpol: int, ntor: int, nfp: int):
        self.bx = _FrozenBoozXform(mpol, ntor, nfp)
        self.s_to_index = {}
        self.s_used = {}
        self.mpi = None
        super().__init__()

    def register(self, surfaces):
        for surface in np.atleast_1d(surfaces):
            key = float(surface)
            self.s_to_index.setdefault(key, len(self.s_to_index))
            self.s_used.setdefault(key, key)

    def run(self):
        pass


def _mock_boozer_spectrum(mpol: int, ntor: int, nfp: int):
    mnmax = (ntor * 2 + 1) * mpol + ntor + 1
    xm = np.zeros(mnmax)
    xn = np.zeros(mnmax)
    xn[: ntor + 1] = np.arange(ntor + 1)
    for m in range(1, mpol + 1):
        index = ntor + 1 + (ntor * 2 + 1) * (m - 1)
        xm[index : index + (ntor * 2 + 1)] = m
        xn[index : index + (ntor * 2 + 1)] = np.arange(-ntor, ntor + 1)

    arr1 = np.arange(1.0, mnmax + 1) * 10
    arr2 = arr1 + 1
    arr2[0] = 100
    return np.stack((arr1, arr2)).transpose(), xm, xn * nfp


def _cpu_quasisymmetry_residuals(
    boozer: _FrozenBoozer,
    surfaces,
    *,
    helicity_m: int,
    helicity_n: int,
    normalization: str = "B00",
    weight: str = "even",
):
    return Quasisymmetry(
        boozer,
        surfaces,
        helicity_m,
        helicity_n,
        normalization,
        weight,
    ).J()


def _mode_indices(boozer: _FrozenBoozer, helicity_m: int, helicity_n: int):
    return boozer_quasisymmetry_mode_indices(
        boozer.bx.xm_b,
        boozer.bx.xn_b,
        boozer.bx.nfp,
        helicity_m,
        helicity_n,
    )


def test_boozer_quasisymmetry_reducer_matches_legacy_mock_modes():
    boozer = _FrozenBoozer(mpol=3, ntor=2, nfp=4)
    surfaces = (0.0, 1.0)
    symmetric_indices, nonsymmetric_indices = _mode_indices(
        boozer,
        helicity_m=1,
        helicity_n=0,
    )

    actual = boozer_quasisymmetry_residuals(
        boozer.bx.bmnc_b,
        symmetric_indices,
        nonsymmetric_indices,
        np.array([0, 1]),
        np.array(surfaces),
    )
    expected = _cpu_quasisymmetry_residuals(
        boozer,
        surfaces,
        helicity_m=1,
        helicity_n=0,
    )

    np.testing.assert_allclose(np.asarray(actual), expected)


def test_boozer_quasisymmetry_reducer_matches_cpu_oracle_for_weights():
    boozer = _FrozenBoozer(mpol=3, ntor=2, nfp=4)
    surfaces = (0.5, 0.8)
    surface_indices = np.array([0, 1])
    s_used = np.array(surfaces)

    for normalization, weight in [
        ("symmetric", "even"),
        ("B00", "stellopt"),
        ("symmetric", "stellopt_ornl"),
    ]:
        symmetric_indices, nonsymmetric_indices = _mode_indices(
            boozer,
            helicity_m=1,
            helicity_n=1,
        )
        actual = boozer_quasisymmetry_residuals(
            boozer.bx.bmnc_b,
            symmetric_indices,
            nonsymmetric_indices,
            surface_indices,
            s_used,
            normalization=normalization,
            weight=weight,
        )
        expected = _cpu_quasisymmetry_residuals(
            boozer,
            surfaces,
            helicity_m=1,
            helicity_n=1,
            normalization=normalization,
            weight=weight,
        )
        np.testing.assert_allclose(np.asarray(actual), expected)


def test_boozer_quasisymmetry_reducer_traces_with_frozen_mode_metadata():
    boozer = _FrozenBoozer(mpol=3, ntor=2, nfp=4)
    surfaces = (0.5, 0.8)
    surface_indices = np.array([0, 1])
    s_used = np.array(surfaces)
    symmetric_indices, nonsymmetric_indices = _mode_indices(
        boozer,
        helicity_m=0,
        helicity_n=1,
    )

    compiled = jax.jit(
        lambda bmnc: boozer_quasisymmetry_residuals(
            bmnc,
            symmetric_indices,
            nonsymmetric_indices,
            surface_indices,
            s_used,
        )
    )

    expected = _cpu_quasisymmetry_residuals(
        boozer,
        surfaces,
        helicity_m=0,
        helicity_n=1,
        normalization="B00",
        weight="even",
    )
    np.testing.assert_allclose(np.asarray(compiled(boozer.bx.bmnc_b)), expected)


def test_boozer_quasisymmetry_reducer_traces_with_index_metadata_inputs():
    boozer = _FrozenBoozer(mpol=3, ntor=2, nfp=4)
    surfaces = (0.5, 0.8)
    surface_indices = np.array([0, 1])
    s_used = np.array(surfaces)
    symmetric_indices, nonsymmetric_indices = _mode_indices(
        boozer,
        helicity_m=0,
        helicity_n=1,
    )

    compiled = jax.jit(
        lambda bmnc, symmetric, nonsymmetric, indices, radial_labels: (
            boozer_quasisymmetry_residuals(
                bmnc,
                symmetric,
                nonsymmetric,
                indices,
                radial_labels,
            )
        )
    )

    expected = _cpu_quasisymmetry_residuals(
        boozer,
        surfaces,
        helicity_m=0,
        helicity_n=1,
        normalization="B00",
        weight="even",
    )
    actual = compiled(
        boozer.bx.bmnc_b,
        symmetric_indices,
        nonsymmetric_indices,
        surface_indices,
        s_used,
    )
    np.testing.assert_allclose(np.asarray(actual), expected)
