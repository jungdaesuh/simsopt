"""Host NumPy bake of single-stage construction constants.

Construction of the exact analytic route used to dispatch one-off JAX primitives
(``moveaxis``, ``mean``, ``Lp_curvature_pure``, Fourier ``sin``/``cos``, coil
``slice``/``matmul``, …) while assembling the seed surface basis and the
cached coil groups. Those arrays are computed once from host data, so this
module evaluates them with NumPy and leaves device placement to
``explicit_device_array`` / ``make_coil_group_spec``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray
from simsopt_jax.core.specs import make_coil_group_spec, make_grouped_coil_set_spec
from simsopt_jax.runtime.host_boundary import host_array
from simsopt_jax_adapters.field._coil_graph import (
    _unwrap_coil_curve_and_current_objects,
)
from simsopt_jax_adapters.geo.boozer_surface import _BoozerPenaltyGeometry

# Same bits as ``simsopt_jax.core._device_scalars.two_pi`` on float64.
_TWO_PI = np.float64(2.0) * np.arccos(np.float64(-1.0))


def host_grouped_coil_set_spec(biotsavart):
    """Grouped coil spec from native curve geometry, without JAX primitives."""
    by_nquad: dict[
        int, list[tuple[int, NDArray[np.float64], NDArray[np.float64], float]]
    ] = defaultdict(list)
    for coil_index, coil in enumerate(biotsavart.coils):
        curve, rotmat, current, scale = _unwrap_coil_curve_and_current_objects(
            coil.curve, coil.current
        )
        gamma = np.asarray(curve.gamma(), dtype=np.float64)
        gammadash = np.asarray(curve.gammadash(), dtype=np.float64)
        if rotmat is not None:
            rotation = np.asarray(rotmat, dtype=np.float64)
            gamma = gamma @ rotation
            gammadash = gammadash @ rotation
        current_value = float(current.get_value()) * float(scale)
        by_nquad[int(gamma.shape[0])].append(
            (coil_index, gamma, gammadash, current_value)
        )
    groups = []
    for members in sorted(by_nquad.values(), key=lambda group: group[0][0]):
        indices = tuple(item[0] for item in members)
        gammas = np.stack([item[1] for item in members], axis=0)
        gammadashs = np.stack([item[2] for item in members], axis=0)
        currents = np.asarray([item[3] for item in members], dtype=np.float64)
        groups.append(make_coil_group_spec(gammas, gammadashs, currents, indices))
    return make_grouped_coil_set_spec(groups)


def host_coil_currents(coil_set_spec) -> NDArray[np.float64]:
    """Dense current vector in coil-index order, assembled on the host."""
    coil_count = (
        max((max(group.coil_indices) for group in coil_set_spec.groups), default=-1) + 1
    )
    currents = np.zeros((coil_count,), dtype=np.float64)
    for group in coil_set_spec.groups:
        currents[np.asarray(group.coil_indices, dtype=np.int64)] = host_array(
            group.currents, dtype=np.float64
        )
    return currents


def host_analytic_geometry_origin_and_basis(
    *,
    n_dofs: int,
    quadpoints_phi,
    quadpoints_theta,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    scatter_indices,
    surface_kind: str,
    clamped_dims,
) -> tuple[_BoozerPenaltyGeometry, _BoozerPenaltyGeometry]:
    """Affine surface origin and Jacobian, matching the JAX jacfwd bake.

    Origin is geometry at zero coefficients. The Jacobian columns are the
    same Fourier evaluation as ``surface_gamma`` / ``surface_gammadash1`` /
    ``surface_gammadash2`` applied to the identity, which is the host form of
    ``jax.vmap(jax.jvp)`` for this linear map.
    """
    if surface_kind != "xyztensorfourier":
        raise ValueError(
            "Host analytic geometry bake supports surface_kind='xyztensorfourier' "
            f"only; got {surface_kind!r}."
        )
    if any(bool(flag) for flag in clamped_dims):
        raise ValueError("Host analytic geometry bake requires clamped_dims all False.")
    nphi = int(np.asarray(quadpoints_phi).shape[0])
    ntheta = int(np.asarray(quadpoints_theta).shape[0])
    origin = _BoozerPenaltyGeometry(
        gamma=np.zeros((nphi, ntheta, 3), dtype=np.float64),
        xphi=np.zeros((nphi, ntheta, 3), dtype=np.float64),
        xtheta=np.zeros((nphi, ntheta, 3), dtype=np.float64),
    )
    xc, yc, zc = _host_identity_xyzc(
        n_dofs=int(n_dofs),
        mpol=int(mpol),
        ntor=int(ntor),
        stellsym=bool(stellsym),
        scatter_indices=scatter_indices,
    )
    phi = np.asarray(quadpoints_phi, dtype=np.float64)
    theta = np.asarray(quadpoints_theta, dtype=np.float64)
    w_theta, dw_theta = _host_theta_basis(theta, int(mpol))
    v_phi, dv_phi = _host_phi_basis(phi, int(ntor), int(nfp))
    basis = _BoozerPenaltyGeometry(
        gamma=_host_move_dof_axis(_host_gamma(v_phi, w_theta, xc, yc, zc, phi)),
        xphi=_host_move_dof_axis(
            _host_gammadash1(v_phi, dv_phi, w_theta, xc, yc, zc, phi)
        ),
        xtheta=_host_move_dof_axis(
            _host_gammadash2(v_phi, w_theta, dw_theta, xc, yc, zc, phi)
        ),
    )
    return origin, basis


def _host_mode_range(start: int, stop: int) -> NDArray[np.float64]:
    return np.arange(start, stop, dtype=np.float64)


def _host_theta_basis(quadpoints_theta: NDArray[np.float64], mpol: int):
    theta = _TWO_PI * quadpoints_theta
    m_cos = _host_mode_range(0, mpol + 1)
    m_sin = _host_mode_range(1, mpol + 1)
    arg_cos = m_cos[None, :] * theta[:, None]
    arg_sin = m_sin[None, :] * theta[:, None]
    basis = np.concatenate([np.cos(arg_cos), np.sin(arg_sin)], axis=1)
    derivative = np.concatenate(
        [
            -m_cos[None, :] * _TWO_PI * np.sin(arg_cos),
            m_sin[None, :] * _TWO_PI * np.cos(arg_sin),
        ],
        axis=1,
    )
    return basis, derivative


def _host_phi_basis(quadpoints_phi: NDArray[np.float64], ntor: int, nfp: int):
    phi = _TWO_PI * quadpoints_phi
    nfp_scale = np.float64(nfp)
    n_cos = _host_mode_range(0, ntor + 1) * nfp_scale
    n_sin = _host_mode_range(1, ntor + 1) * nfp_scale
    arg_cos = n_cos[None, :] * phi[:, None]
    arg_sin = n_sin[None, :] * phi[:, None]
    basis = np.concatenate([np.cos(arg_cos), np.sin(arg_sin)], axis=1)
    derivative = np.concatenate(
        [
            -n_cos[None, :] * _TWO_PI * np.sin(arg_cos),
            n_sin[None, :] * _TWO_PI * np.cos(arg_sin),
        ],
        axis=1,
    )
    return basis, derivative


def _host_eval_hat(v_phi, w_theta, coeffs):
    return np.matmul(np.matmul(v_phi, np.swapaxes(coeffs, -1, -2)), w_theta.T)


def _host_rotate(quadpoints_phi, radial, toroidal):
    phi_angle = _TWO_PI * quadpoints_phi
    cosine = np.cos(phi_angle)[None, :, None]
    sine = np.sin(phi_angle)[None, :, None]
    return radial * cosine - toroidal * sine, radial * sine + toroidal * cosine


def _host_gamma(v_phi, w_theta, xc, yc, zc, quadpoints_phi):
    xhat = _host_eval_hat(v_phi, w_theta, xc)
    yhat = _host_eval_hat(v_phi, w_theta, yc)
    zhat = _host_eval_hat(v_phi, w_theta, zc)
    x_coord, y_coord = _host_rotate(quadpoints_phi, xhat, yhat)
    return np.stack([x_coord, y_coord, zhat], axis=-1)


def _host_gammadash1(v_phi, dv_phi, w_theta, xc, yc, zc, quadpoints_phi):
    xhat = _host_eval_hat(v_phi, w_theta, xc)
    yhat = _host_eval_hat(v_phi, w_theta, yc)
    dxhat_dphi = _host_eval_hat(dv_phi, w_theta, xc)
    dyhat_dphi = _host_eval_hat(dv_phi, w_theta, yc)
    dz_dphi = _host_eval_hat(dv_phi, w_theta, zc)
    radial = dxhat_dphi - _TWO_PI * yhat
    toroidal = dyhat_dphi + _TWO_PI * xhat
    dx_coord, dy_coord = _host_rotate(quadpoints_phi, radial, toroidal)
    return np.stack([dx_coord, dy_coord, dz_dphi], axis=-1)


def _host_gammadash2(v_phi, w_theta, dw_theta, xc, yc, zc, quadpoints_phi):
    dxhat_dtheta = _host_eval_hat(v_phi, dw_theta, xc)
    dyhat_dtheta = _host_eval_hat(v_phi, dw_theta, yc)
    dz_dtheta = _host_eval_hat(v_phi, dw_theta, zc)
    dx_coord, dy_coord = _host_rotate(quadpoints_phi, dxhat_dtheta, dyhat_dtheta)
    return np.stack([dx_coord, dy_coord, dz_dtheta], axis=-1)


def _host_identity_xyzc(*, n_dofs, mpol, ntor, stellsym, scatter_indices):
    n_per = (2 * mpol + 1) * (2 * ntor + 1)
    flat = np.zeros((n_dofs, 3 * n_per), dtype=np.float64)
    if stellsym:
        indices = np.asarray(scatter_indices, dtype=np.int32).reshape(-1)
        flat[np.arange(n_dofs), indices] = 1.0
    else:
        flat[np.arange(n_dofs), np.arange(n_dofs)] = 1.0
    shape = (n_dofs, 2 * mpol + 1, 2 * ntor + 1)
    return (
        flat[:, :n_per].reshape(shape),
        flat[:, n_per : 2 * n_per].reshape(shape),
        flat[:, 2 * n_per :].reshape(shape),
    )


def _host_move_dof_axis(batched_xyz):
    return np.moveaxis(batched_xyz, 0, -1)
