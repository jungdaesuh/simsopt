from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("simsoptpp")

from simsopt._core.optimizable import load  # noqa: E402
from simsopt.field.biotsavart import BiotSavart  # noqa: E402
from simsopt.field.coil import Current, coils_via_symmetries  # noqa: E402
from simsopt.geo.curvexyzfourier import CurveXYZFourier  # noqa: E402
from simsopt_jax.core import make_biot_savart_spec  # noqa: E402
from simsopt_jax_adapters.field.biotsavart_backend import (  # noqa: E402
    BiotSavartJAX,
    SpecBackedBiotSavartJAX,
)


def _write_legacy_biot_savart(root: Path) -> Path:
    curve = CurveXYZFourier(8, 1)
    coeffs = curve.dofs_matrix
    coeffs[1][0] = 1.0
    coeffs[2][1] = 1.0
    curve.set_dofs(np.concatenate(coeffs))
    coils = coils_via_symmetries([curve], [Current(1.0)], nfp=2, stellsym=True)
    biot_savart = BiotSavart(coils)
    biot_savart.set_points(np.asarray([[0.2, 0.1, 0.3], [0.3, 0.2, 0.4]]))

    biot_savart_path = root / "biot_savart_opt.json"
    biot_savart.save(filename=biot_savart_path)
    return biot_savart_path


def test_spec_backed_biot_savart_restart_dofs_are_mutable(tmp_path):
    biot_savart_path = _write_legacy_biot_savart(tmp_path)
    legacy_bs = load(biot_savart_path)
    legacy_bs_jax = BiotSavartJAX(legacy_bs.coils)
    restart_spec = make_biot_savart_spec(
        coil_dof_extraction=legacy_bs_jax.coil_dof_extraction_spec(),
        coil_dofs=legacy_bs_jax.x,
    )

    spec_backed_bs = SpecBackedBiotSavartJAX(restart_spec)
    spec_backed_bs.local_x = spec_backed_bs.local_x.copy()

    np.testing.assert_allclose(spec_backed_bs.local_x, legacy_bs_jax.x)
