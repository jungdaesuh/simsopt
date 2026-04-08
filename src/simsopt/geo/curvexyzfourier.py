from itertools import chain

import numpy as np
from scipy.fft import rfft

from ._simsoptpp import sopp_namespace
from .curve import Curve, JaxCurve, _HAS_JAX, _as_jax_float64, jax, jnp

sopp = sopp_namespace("CurveXYZFourier")


__all__ = [
    "CurveXYZFourier",
    "JaxCurveXYZFourier",
    "jaxfouriercurve_pure",
    "jaxfouriercurve_geometry_pure",
]

_TWO_PI = 2.0 * np.pi


def _contains_jax_leaves(value):
    if not _HAS_JAX:
        return False
    return any(
        isinstance(leaf, jax.Array) or hasattr(leaf, "aval")
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _is_tracer(value):
    return _HAS_JAX and hasattr(value, "aval") and not isinstance(value, jax.Array)


def _as_runtime_float64(value, *, reference):
    if not _HAS_JAX:
        return np.asarray(value, dtype=np.float64)
    if _is_tracer(reference) and not _contains_jax_leaves(value):
        return np.asarray(value, dtype=np.float64)
    return _as_jax_float64(value)


def _mode_numbers(order, *, reference):
    return _as_runtime_float64(
        np.arange(1, order + 1, dtype=np.float64),
        reference=reference,
    )


def _constant_row(length, value, *, reference):
    return _as_runtime_float64(
        np.full((1, int(length)), value, dtype=np.float64),
        reference=reference,
    )


def _interleave_harmonics(first, second):
    return jnp.reshape(jnp.stack((first, second), axis=1), (-1, first.shape[1]))


def _fourier_basis_terms(quadpoints, order):
    quadpoints = _as_jax_float64(quadpoints)
    two_pi = _as_runtime_float64(_TWO_PI, reference=quadpoints)
    points = two_pi * quadpoints
    mode_numbers = _mode_numbers(order, reference=points)
    phase = jnp.expand_dims(mode_numbers, axis=1) * jnp.expand_dims(points, axis=0)
    sin_phase = jnp.sin(phase)
    cos_phase = jnp.cos(phase)
    mode_scale = two_pi * mode_numbers
    mode_scale_sq = mode_scale * mode_scale
    mode_scale_cu = mode_scale_sq * mode_scale
    zero_row = _constant_row(points.shape[0], 0.0, reference=points)

    basis = jnp.concatenate(
        (
            _constant_row(points.shape[0], 1.0, reference=points),
            _interleave_harmonics(sin_phase, cos_phase),
        ),
        axis=0,
    )
    dash_basis = jnp.concatenate(
        (
            zero_row,
            _interleave_harmonics(
                jnp.expand_dims(mode_scale, axis=1) * cos_phase,
                -jnp.expand_dims(mode_scale, axis=1) * sin_phase,
            ),
        ),
        axis=0,
    )
    dashdash_basis = jnp.concatenate(
        (
            zero_row,
            _interleave_harmonics(
                -jnp.expand_dims(mode_scale_sq, axis=1) * sin_phase,
                -jnp.expand_dims(mode_scale_sq, axis=1) * cos_phase,
            ),
        ),
        axis=0,
    )
    dashdashdash_basis = jnp.concatenate(
        (
            zero_row,
            _interleave_harmonics(
                -jnp.expand_dims(mode_scale_cu, axis=1) * cos_phase,
                jnp.expand_dims(mode_scale_cu, axis=1) * sin_phase,
            ),
        ),
        axis=0,
    )
    return basis, dash_basis, dashdash_basis, dashdashdash_basis


class CurveXYZFourier(sopp.CurveXYZFourier, Curve):
    r"""
       ``CurveXYZFourier`` is a curve that is represented in Cartesian
       coordinates using the following Fourier series:

        .. math::
           x(\theta) &= \sum_{m=0}^{\text{order}} x_{c,m}\cos(m\theta) + \sum_{m=1}^{\text{order}} x_{s,m}\sin(m\theta) \\
           y(\theta) &= \sum_{m=0}^{\text{order}} y_{c,m}\cos(m\theta) + \sum_{m=1}^{\text{order}} y_{s,m}\sin(m\theta) \\
           z(\theta) &= \sum_{m=0}^{\text{order}} z_{c,m}\cos(m\theta) + \sum_{m=1}^{\text{order}} z_{s,m}\sin(m\theta)

       The dofs are stored in the order

        .. math::
           [x_{c,0}, x_{s,1}, x_{c,1},\cdots x_{s,\text{order}}, x_{c,\text{order}},y_{c,0},y_{s,1},y_{c,1},\cdots]

    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurveXYZFourier.__init__(self, quadpoints, order)
        if dofs is None:
            Curve.__init__(
                self,
                x0=self.get_dofs(),
                names=self._make_names(order),
                external_dof_setter=CurveXYZFourier.set_dofs_impl,
            )
        else:
            Curve.__init__(
                self, dofs=dofs, external_dof_setter=CurveXYZFourier.set_dofs_impl
            )

    def _make_names(self, order):
        """
        This function returns the names of the dofs associated to this object.
        Args:
            order (int): Order of the Fourier series.

        Returns:
            List of dof names.
        """
        x_names = ["xc(0)"]
        x_cos_names = [f"xc({i})" for i in range(1, order + 1)]
        x_sin_names = [f"xs({i})" for i in range(1, order + 1)]
        x_names += list(chain.from_iterable(zip(x_sin_names, x_cos_names)))
        y_names = ["yc(0)"]
        y_cos_names = [f"yc({i})" for i in range(1, order + 1)]
        y_sin_names = [f"ys({i})" for i in range(1, order + 1)]
        y_names += list(chain.from_iterable(zip(y_sin_names, y_cos_names)))
        z_names = ["zc(0)"]
        z_cos_names = [f"zc({i})" for i in range(1, order + 1)]
        z_sin_names = [f"zs({i})" for i in range(1, order + 1)]
        z_names += list(chain.from_iterable(zip(z_sin_names, z_cos_names)))
        return x_names + y_names + z_names

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurveXYZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurveXYZFourier.set_dofs(self, dofs)

    def to_spec(self):
        """Build an immutable JAX geometry spec from the current curve state."""
        from ..jax_core import make_curve_xyzfourier_spec

        return make_curve_xyzfourier_spec(
            dofs=self.get_dofs(),
            quadpoints=self.quadpoints,
            order=self.order,
        )

    @staticmethod
    def load_curves_from_file(filename, order=None, ppp=20, delimiter=","):
        """
        This function loads a file containing Fourier coefficients for several coils.
        The file is expected to have :mod:`6*num_coils` many columns, and :mod:`order+1` many rows.
        The columns are in the following order,

            sin_x_coil1, cos_x_coil1, sin_y_coil1, cos_y_coil1, sin_z_coil1, cos_z_coil1, sin_x_coil2, cos_x_coil2, sin_y_coil2, cos_y_coil2, sin_z_coil2, cos_z_coil2,  ...

        """
        coil_data = np.loadtxt(filename, delimiter=delimiter)

        assert coil_data.shape[1] % 6 == 0
        assert order <= coil_data.shape[0] - 1

        num_coils = coil_data.shape[1] // 6
        coils = [CurveXYZFourier(order * ppp, order) for i in range(num_coils)]
        for ic in range(num_coils):
            dofs = coils[ic].dofs_matrix
            dofs[0][0] = coil_data[0, 6 * ic + 1]
            dofs[1][0] = coil_data[0, 6 * ic + 3]
            dofs[2][0] = coil_data[0, 6 * ic + 5]
            for io in range(0, min(order, coil_data.shape[0] - 1)):
                dofs[0][2 * io + 1] = coil_data[io + 1, 6 * ic + 0]
                dofs[0][2 * io + 2] = coil_data[io + 1, 6 * ic + 1]
                dofs[1][2 * io + 1] = coil_data[io + 1, 6 * ic + 2]
                dofs[1][2 * io + 2] = coil_data[io + 1, 6 * ic + 3]
                dofs[2][2 * io + 1] = coil_data[io + 1, 6 * ic + 4]
                dofs[2][2 * io + 2] = coil_data[io + 1, 6 * ic + 5]
            coils[ic].local_x = np.concatenate(dofs)
        return coils

    @staticmethod
    def load_curves_from_makegrid_file(
        filename: str, order: int, ppp=20, group_names=None
    ):
        """
        This function loads a Makegrid input file containing the Cartesian
        coordinates for several coils and finds the corresponding Fourier
        coefficients through an fft. The format is described at
        https://princetonuniversity.github.io/STELLOPT/MAKEGRID

        Args:
            filename: file to load.
            order: maximum mode number in the Fourier series.
            ppp: points-per-period: number of quadrature points per period.
            group_names: List of coil group names (str). If not 'None', only get coils in coil groups that are in the list.

        Returns:
            A list of ``CurveXYZFourier`` objects.
        """

        with open(filename, "r") as f:
            file_lines = f.read().splitlines()[3:]

        curve_data = []
        single_curve_data = []
        for j_line in range(len(file_lines)):
            vals = file_lines[j_line].split()
            n_vals = len(vals)
            if n_vals == 4:
                float_vals = [float(val) for val in vals[:3]]
                single_curve_data.append(float_vals)
            elif n_vals == 6:
                # This must be the last line of the coil
                if group_names is None:
                    curve_data.append(single_curve_data)
                else:
                    this_group_name = vals[5]
                    if this_group_name in group_names:
                        curve_data.append(single_curve_data)
                single_curve_data = []
            elif n_vals == 1:
                # Presumably the line that is just "end"
                break
            else:
                raise RuntimeError("Should not get here")

        coil_data = []

        # Compute the Fourier coefficients for each coil
        for curve in curve_data:
            xArr, yArr, zArr = np.transpose(curve)

            curves_Fourier = []

            # Compute the Fourier coefficients
            for x in [xArr, yArr, zArr]:
                assert (
                    len(x) >= 2 * order
                )  # the order of the fft is limited by the number of samples
                xf = rfft(x) / len(x)

                fft_0 = [xf[0].real]  # find the 0 order coefficient
                fft_cos = 2 * xf[1 : order + 1].real  # find the cosine coefficients
                fft_sin = -2 * xf[: order + 1].imag  # find the sine coefficients

                combined_fft = np.concatenate([fft_sin, fft_0, fft_cos])
                curves_Fourier.append(combined_fft)

            coil_data.append(np.concatenate(curves_Fourier))

        coil_data = np.asarray(coil_data)
        coil_data = coil_data.reshape(
            6 * len(curve_data), order + 1
        )  # There are 6 * order coefficients per coil
        coil_data = np.transpose(coil_data)

        assert coil_data.shape[1] % 6 == 0
        assert order <= coil_data.shape[0] - 1

        num_coils = coil_data.shape[1] // 6
        coils = [CurveXYZFourier(order * ppp, order) for i in range(num_coils)]
        for ic in range(num_coils):
            dofs = coils[ic].dofs_matrix
            dofs[0][0] = coil_data[0, 6 * ic + 1]
            dofs[1][0] = coil_data[0, 6 * ic + 3]
            dofs[2][0] = coil_data[0, 6 * ic + 5]
            for io in range(0, min(order, coil_data.shape[0] - 1)):
                dofs[0][2 * io + 1] = coil_data[io + 1, 6 * ic + 0]
                dofs[0][2 * io + 2] = coil_data[io + 1, 6 * ic + 1]
                dofs[1][2 * io + 1] = coil_data[io + 1, 6 * ic + 2]
                dofs[1][2 * io + 2] = coil_data[io + 1, 6 * ic + 3]
                dofs[2][2 * io + 1] = coil_data[io + 1, 6 * ic + 4]
                dofs[2][2 * io + 2] = coil_data[io + 1, 6 * ic + 5]
            coils[ic].local_x = np.concatenate(dofs)
        return coils


def jaxfouriercurve_pure(dofs, quadpoints, order):
    """
    This pure function returns the curve position vector in XYZ coordinates..

    Args:
        dofs (array, shape (ndofs,)): Array of dofs.
        quadpoints (array, shape (N, 3)): Array of quadrature points.
        order (int): Order of the Fourier series.

    Returns:
        Array of curve points, shape (N, 3)
    """
    dofs = _as_jax_float64(dofs)
    coeffs = jnp.reshape(dofs, (3, dofs.shape[0] // 3))
    basis, _, _, _ = _fourier_basis_terms(quadpoints, order)
    gamma = coeffs @ basis
    return jnp.moveaxis(gamma, 0, -1)


def jaxfouriercurve_geometry_pure(dofs, quadpoints, order):
    """Return XYZ Fourier geometry and its first three quadpoint derivatives."""
    dofs = _as_jax_float64(dofs)
    coeffs = jnp.reshape(dofs, (3, dofs.shape[0] // 3))
    basis, dash_basis, dashdash_basis, dashdashdash_basis = _fourier_basis_terms(
        quadpoints,
        order,
    )
    gamma = coeffs @ basis
    gammadash = coeffs @ dash_basis
    gammadashdash = coeffs @ dashdash_basis
    gammadashdashdash = coeffs @ dashdashdash_basis
    return tuple(
        jnp.moveaxis(component, 0, -1)
        for component in (gamma, gammadash, gammadashdash, gammadashdashdash)
    )


class JaxCurveXYZFourier(JaxCurve):
    """
    A Python+Jax implementation of the CurveXYZFourier class. This is an autodiff
    compatible version of the same CurveXYZFourier class in the C++ implementation in
    :mod:`simsoptpp`. The point of this class is to illustrate how jax can be used
    to define a geometric object class and calculate all the derivatives (both
    with respect to dofs and with respect to the angle :math:`\theta`) automatically.

    Args:
        quadpoints (array): Array of quadrature points.
        order (int): Order of the Fourier series.
        dofs (array): Array of dofs.
    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        def pure(dofs, points):
            return jaxfouriercurve_pure(dofs, points, order)

        self.order = order
        self.coefficients = [
            np.zeros((2 * order + 1,)),
            np.zeros((2 * order + 1,)),
            np.zeros((2 * order + 1,)),
        ]
        if dofs is None:
            super().__init__(
                quadpoints,
                pure,
                x0=np.concatenate(self.coefficients),
                names=self._make_names(order),
                external_dof_setter=JaxCurveXYZFourier.set_dofs_impl,
            )
        else:
            super().__init__(
                quadpoints,
                pure,
                dofs=dofs,
                names=self._make_names(order),
                external_dof_setter=JaxCurveXYZFourier.set_dofs_impl,
            )

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 * (2 * self.order + 1)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        counter = 0
        for i in range(3):
            self.coefficients[i][0] = dofs[counter]
            counter += 1
            for j in range(1, self.order + 1):
                self.coefficients[i][2 * j - 1] = dofs[counter]
                counter += 1
                self.coefficients[i][2 * j] = dofs[counter]
                counter += 1

    def _make_names(self, order):
        """
        This function returns the names of the dofs associated to this object.

        Args:
            order (int): Order of the Fourier series.

        Returns:
            List of dof names.
        """
        x_names = ["xc(0)"]
        x_cos_names = [f"xc({i})" for i in range(1, order + 1)]
        x_sin_names = [f"xs({i})" for i in range(1, order + 1)]
        x_names += list(chain.from_iterable(zip(x_sin_names, x_cos_names)))
        y_names = ["yc(0)"]
        y_cos_names = [f"yc({i})" for i in range(1, order + 1)]
        y_sin_names = [f"ys({i})" for i in range(1, order + 1)]
        y_names += list(chain.from_iterable(zip(y_sin_names, y_cos_names)))
        z_names = ["zc(0)"]
        z_cos_names = [f"zc({i})" for i in range(1, order + 1)]
        z_sin_names = [f"zs({i})" for i in range(1, order + 1)]
        z_names += list(chain.from_iterable(zip(z_sin_names, z_cos_names)))
        return x_names + y_names + z_names
