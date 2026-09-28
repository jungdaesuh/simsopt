"""Landreman/Paul QA permanent-magnet grid shared by the JAX PM parity tests.

``_build_qa_grid`` rebuilds the grid of
``examples/2_Intermediate/permanent_magnet_QA.py`` between its two offset
toroidal surfaces, without the example's coil optimization.
"""

from __future__ import annotations

import hashlib
import io
import tempfile
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simsopt.field import BiotSavart
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.util.permanent_magnet_helper_functions import (
    initialize_coils_for_pm_optimization,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_DATA = REPO_ROOT / "tests" / "test_files"


class ProbeError(RuntimeError):
    """The probe cannot produce a trustworthy measurement as configured."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class GridSpec:
    """Everything the host needs to rebuild one permanent-magnet grid."""

    nphi: int
    ntheta: int
    downsample: int | None
    coordinate_flag: str
    dr: float | None
    inner_offset: float | None
    outer_offset: float | None
    source: str


@dataclass(frozen=True)
class GridBuild:
    """A built CPU grid plus the facts a probe reports about it."""

    grid: object
    seconds: float
    input_sha256: dict[str, str]
    polarization_count: int | None


def _build_qa_grid(spec: GridSpec) -> GridBuild:
    """Landreman/Paul QA grid between two offset toroidal surfaces.

    Mirrors ``examples/2_Intermediate/permanent_magnet_QA.py`` (constants
    ``nphi``/``ntheta``/``dr``/``coff``/``poff``, then ``kwargs_geo``) except
    that its ``coil_optimization`` call is not run: it changes only
    ``Bnormal``, hence ``b_obj``.  Both lanes of a parity test consume the
    identical host grid, so the omission cannot tilt a comparison.
    """

    if spec.dr is None or spec.inner_offset is None or spec.outer_offset is None:
        raise ProbeError("the QA grid needs dr and both surface offsets")
    if spec.downsample is not None:
        raise ProbeError(
            "geo_setup_between_toroidal_surfaces has no downsample argument; "
            f"the QA grid spec must leave it unset, got {spec.downsample}"
        )
    surface_path = TEST_DATA / "input.LandremanPaul2021_QA_lowres"
    start = time.perf_counter()
    surfaces = [
        SurfaceRZFourier.from_vmec_input(
            surface_path,
            range="half period",
            nphi=spec.nphi,
            ntheta=spec.ntheta,
        )
        for _ in range(3)
    ]
    boundary, inner, outer = surfaces
    inner.extend_via_projected_normal(spec.inner_offset)
    outer.extend_via_projected_normal(spec.outer_offset)
    with tempfile.TemporaryDirectory() as scratch, redirect_stdout(io.StringIO()):
        _, _, coils = initialize_coils_for_pm_optimization(
            "qa", TEST_DATA, boundary, scratch
        )
    field = BiotSavart(coils)
    field.set_points(boundary.gamma().reshape((-1, 3)))
    normal_field = np.sum(
        field.B().reshape((spec.nphi, spec.ntheta, 3)) * boundary.unitnormal(),
        axis=2,
    )
    with redirect_stdout(io.StringIO()):
        grid = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
            boundary,
            normal_field,
            inner,
            outer,
            dr=spec.dr,
            coordinate_flag=spec.coordinate_flag,
        )
    seconds = time.perf_counter() - start
    return GridBuild(
        grid=grid,
        seconds=seconds,
        input_sha256={surface_path.name: sha256_file(surface_path)},
        polarization_count=None,
    )
