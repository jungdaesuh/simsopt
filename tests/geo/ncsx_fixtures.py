"""NCSX Boozer-surface fixtures shared by the Boozer least-squares tests.

``NCSX_EXAMPLE_JSON`` is the NCSX start state that the official
``examples/2_Intermediate/boozerQA_ls_mpi.py`` loads; the surface helpers pad a
TensorFourier surface to a larger resolution without changing its geometry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from simsopt.geo import SurfaceXYZTensorFourier

NCSX_EXAMPLE_JSON: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "2_Intermediate"
    / "inputs"
    / "input_ncsx"
    / "ncsx_init.json"
)


def remap_tensor_fourier_index(index: int, m_old: int, m_new: int) -> int:
    """Map a stellsym-aware TensorFourier cosine/sine slot onto a padded grid."""

    if index <= m_old:
        return int(index)
    return int(m_new + (index - m_old))


def upsample_surface_xyz_tensor_fourier(
    old: SurfaceXYZTensorFourier,
    *,
    mpol: int,
    ntor: int,
    nphi: int,
    ntheta: int,
) -> SurfaceXYZTensorFourier:
    """Pad TensorFourier modes without changing the realized geometry."""

    phi = np.linspace(0.0, 1.0 / old.nfp, nphi, endpoint=False)
    theta = np.linspace(0.0, 1.0, ntheta, endpoint=False)
    new = SurfaceXYZTensorFourier(
        nfp=old.nfp,
        stellsym=old.stellsym,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=phi,
        quadpoints_theta=theta,
        clamped_dims=list(old.clamped_dims),
    )
    for arr in (new.xcs, new.ycs, new.zcs):
        arr[:, :] = 0.0
    for i in range(2 * old.mpol + 1):
        for j in range(2 * old.ntor + 1):
            ii = remap_tensor_fourier_index(i, old.mpol, mpol)
            jj = remap_tensor_fourier_index(j, old.ntor, ntor)
            new.xcs[ii, jj] = old.xcs[i, j]
            new.ycs[ii, jj] = old.ycs[i, j]
            new.zcs[ii, jj] = old.zcs[i, j]
    new.local_full_x = new.get_dofs()
    new.invalidate_cache()
    return new
