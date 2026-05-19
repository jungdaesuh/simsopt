import numpy as np
from .curve import JaxCurve
from ..jax_core.oriented_curve import centercurve_pure, rotate_pure, shift_pure

__all__ = [
    "OrientedCurveXYZFourier",
    "centercurve_pure",
    "rotate_pure",
    "shift_pure",
]


class OrientedCurveXYZFourier(JaxCurve):
    """
    OrientedCurveXYZFourier is a translated and rotated Curve.

    Args:
     - quadpoints: Integer (number of quadrature points), or array of size N, with float values between 0 and 1.
     - order: Maximum mode order
     - dofs (optionnal): Degrees of freedom
    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        self.order = order
        pure = lambda dofs, points: centercurve_pure(dofs, points, self.order)

        self.coefficients = [
            np.zeros((3,)),
            np.zeros((3,)),
            np.zeros((2 * order,)),
            np.zeros((2 * order,)),
            np.zeros((2 * order,)),
        ]
        if dofs is None:
            super().__init__(
                quadpoints,
                pure,
                x0=np.concatenate(self.coefficients),
                external_dof_setter=OrientedCurveXYZFourier.set_dofs_impl,
                names=self._make_names(),
            )
        else:
            super().__init__(
                quadpoints,
                pure,
                dofs=dofs,
                external_dof_setter=OrientedCurveXYZFourier.set_dofs_impl,
                names=self._make_names(),
            )

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3 + 3 + 3 * (2 * self.order)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        self.coefficients[0][:] = dofs[0:3]
        self.coefficients[1][:] = dofs[3:6]

        counter = 6
        for i in range(0, 3):
            for j in range(0, self.order):
                self.coefficients[i + 2][2 * j] = dofs[counter]
                counter += 1
                self.coefficients[i + 2][2 * j + 1] = dofs[counter]
                counter += 1

    def _make_names(self):
        xyc_name = ["x0", "y0", "z0"]
        ypr_name = ["yaw", "pitch", "roll"]
        dofs_name = []
        for c in ["x", "y", "z"]:
            for j in range(0, self.order):
                dofs_name += [f"{c}s({j + 1})", f"{c}c({j + 1})"]
        return xyc_name + ypr_name + dofs_name

    def to_spec(self):
        """Return an immutable JAX spec for this oriented curve."""
        from simsopt.jax_core.specs import make_oriented_curve_xyzfourier_spec

        return make_oriented_curve_xyzfourier_spec(
            dofs=self.get_dofs(),
            quadpoints=self.quadpoints,
            order=self.order,
        )

    @classmethod
    def from_curvexyzfourier(cls, xyzcurve):
        oriented_curve = cls(xyzcurve.quadpoints, xyzcurve.order)

        for dname in xyzcurve.local_full_dof_names:
            if dname in ["xc(0)", "yc(0)", "zc(0)"]:
                continue
            oriented_curve.set(dname, xyzcurve.get(dname))

        oriented_curve.set("x0", xyzcurve.get("xc(0)"))
        oriented_curve.set("y0", xyzcurve.get("yc(0)"))
        oriented_curve.set("z0", xyzcurve.get("zc(0)"))

        return oriented_curve
