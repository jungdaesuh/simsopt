import numpy as np
from scipy.linalg import lu
from scipy.optimize import minimize, least_squares, root
import simsoptpp as sopp

from .surfaceobjectives import (
    boozer_surface_residual,
    boozer_surface_dexactresidual_dcoils_dcurrents_vjp,
    boozer_surface_dlsqgrad_dcoils_vjp,
)
from ._boozersurface_current_guard import (
    guard_none_G_coil_gradient_callback as _guard_none_G_coil_gradient_callback,
    require_fixed_currents_for_none_G as _require_fixed_currents_for_none_G,
)
from .surfacexyzfourier import SurfaceXYZFourier
from .surfacexyztensorfourier import SurfaceXYZTensorFourier
from ._simsoptpp_boozer_compat import (
    KEY_BOOZER_RESIDUAL,
    KEY_BOOZER_RESIDUAL_DS,
    KEY_BOOZER_RESIDUAL_DS2,
    _call_with_abi_fallback,
)
from .._core.optimizable import Optimizable
from ..backend import is_parity_mode
from ..objectives.utilities import forward_backward
from functools import partial

__all__ = ["BoozerSurface"]

_BOOZER_SURFACE_REQUIRED_ATTRIBUTES = (
    "quadpoints_phi",
    "quadpoints_theta",
    "set_dofs",
    "gamma",
    "gammadash1",
    "gammadash2",
    "dgamma_by_dcoeff",
    "dgammadash1_by_dcoeff",
    "dgammadash2_by_dcoeff",
)
_BOOZER_EXACT_SURFACE_REQUIRED_ATTRIBUTES = (
    *_BOOZER_SURFACE_REQUIRED_ATTRIBUTES,
    "get_stellsym_mask",
)
_BOOZER_SURFACE_SPECS = (
    (
        SurfaceXYZTensorFourier,
        "simsopt.geo.surfacexyztensorfourier",
        "SurfaceXYZTensorFourier",
        _BOOZER_SURFACE_REQUIRED_ATTRIBUTES,
    ),
    (
        SurfaceXYZFourier,
        "simsopt.geo.surfacexyzfourier",
        "SurfaceXYZFourier",
        _BOOZER_SURFACE_REQUIRED_ATTRIBUTES,
    ),
)
_BOOZER_EXACT_SURFACE_SPEC = (
    SurfaceXYZTensorFourier,
    "simsopt.geo.surfacexyztensorfourier",
    "SurfaceXYZTensorFourier",
    _BOOZER_EXACT_SURFACE_REQUIRED_ATTRIBUTES,
)


class _LegacyBoozerAdjointRuntimeState:
    """Compatibility adjoint runtime wrapper for the legacy CPU Boozer solver."""

    def __init__(
        self,
        *,
        linearization_kind,
        decision_size,
        solve_transpose,
        project_coil_adjoint_derivative,
    ):
        self.linearization_kind = linearization_kind
        self.decision_size = int(decision_size)
        self.solve_transpose = solve_transpose
        self.project_coil_adjoint_derivative = project_coil_adjoint_derivative


def _matches_supported_surface_spec(surface, surface_spec):
    supported_type, module_name, class_name, required_attributes = surface_spec
    if isinstance(surface, supported_type):
        return True
    surface_cls = type(surface)
    if surface_cls.__module__ != module_name or surface_cls.__name__ != class_name:
        return False
    return all(hasattr(surface, attr) for attr in required_attributes)


def _is_supported_boozer_surface(surface):
    return any(
        _matches_supported_surface_spec(surface, surface_spec)
        for surface_spec in _BOOZER_SURFACE_SPECS
    )


def _is_supported_boozer_exact_surface(surface):
    return _matches_supported_surface_spec(surface, _BOOZER_EXACT_SURFACE_SPEC)


def _call_boozer_residual(G, iota, xphi, xtheta, B, weight_inv_modB):
    """Dispatch across the supported simsoptpp residual signatures.

    See ``_simsoptpp_boozer_compat`` for the ``I=0.0`` vacuum-field rationale.
    """
    return _call_with_abi_fallback(
        KEY_BOOZER_RESIDUAL,
        sopp.boozer_residual,
        (G, 0.0, iota, xphi, xtheta, B, weight_inv_modB),
        (G, iota, xphi, xtheta, B, weight_inv_modB),
    )


def _call_boozer_residual_ds(
    G,
    iota,
    B,
    dB_dx,
    xphi,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    weight_inv_modB,
):
    """Dispatch across the supported simsoptpp first-derivative signatures.

    See ``_simsoptpp_boozer_compat`` for the ``I=0.0`` vacuum-field rationale.
    """
    return _call_with_abi_fallback(
        KEY_BOOZER_RESIDUAL_DS,
        sopp.boozer_residual_ds,
        (
            G,
            0.0,
            iota,
            B,
            dB_dx,
            xphi,
            xtheta,
            dx_ds,
            dxphi_ds,
            dxtheta_ds,
            weight_inv_modB,
        ),
        (
            G,
            iota,
            B,
            dB_dx,
            xphi,
            xtheta,
            dx_ds,
            dxphi_ds,
            dxtheta_ds,
            weight_inv_modB,
        ),
    )


def _call_boozer_residual_ds2(
    G,
    iota,
    B,
    dB_dx,
    d2B_dx2,
    xphi,
    xtheta,
    dx_ds,
    dxphi_ds,
    dxtheta_ds,
    weight_inv_modB,
):
    """Dispatch across the supported simsoptpp second-derivative signatures.

    See ``_simsoptpp_boozer_compat`` for the ``I=0.0`` vacuum-field rationale.
    """
    return _call_with_abi_fallback(
        KEY_BOOZER_RESIDUAL_DS2,
        sopp.boozer_residual_ds2,
        (
            G,
            0.0,
            iota,
            B,
            dB_dx,
            d2B_dx2,
            xphi,
            xtheta,
            dx_ds,
            dxphi_ds,
            dxtheta_ds,
            weight_inv_modB,
        ),
        (
            G,
            iota,
            B,
            dB_dx,
            d2B_dx2,
            xphi,
            xtheta,
            dx_ds,
            dxphi_ds,
            dxtheta_ds,
            weight_inv_modB,
        ),
    )


class BoozerSurface(Optimizable):
    r"""
    The BoozerSurface class computes a flux surface of a BiotSavart magnetic field where the angles
    of the surface are Boozer angles [1,2]. The class takes as input a Surface representation
    (:obj:`~simsopt.geo.SurfaceXYZFourier` or :obj:`~simsopt.geo.SurfaceXYZTensorFourier`),
    a BiotSavart magnetic field, a flux surface label evaluator, and a target value of the label.

    The Boozer angles are computed by solving a constrained least squares problem,

        .. math::

            \min_x J(x) = \frac{1}{2} \mathbf r^T(x) \mathbf r(x)

    subject to

        .. math::

            l(x) = l_0

            z(\varphi=0,\theta=0) = 0

    where :math:`\mathbf r` is a vector of residuals computed by :obj:`~simsopt.geo.boozer_surface_residual`,
    :math:`l` is a surface label function with target value :math:`l_0`. The degrees of freedom are the
    surface coefficients, the rotational transform, :math:`\iota`, and the value of Boozer's :math:`G` on the surface.
    This objective is zero when the surface corresponds to a magnetic surface of the field, :math:`(\phi,\theta)`
    that parametrize the surface correspond to Boozer angles, and the constraints are satisfied.

    The recommended approach to finding the Boozer angles is to use the :mod:`run_code` method,

        :obj:`~simsopt.geo.BoozerSurface.run_code(iota_guess, G=G_guess)`.

    Depending on how the class is initialized, :mod:`run_code`, will use either the BoozerLS [2] or BoozerExact [1] approach
    to finding the flux surface. The BoozerLS approach finds the flux surface by solving the constrained least squares
    problem mentioned above. The methods

        #. :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_penalty_constraints_LBFGS`
        #. :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_penalty_constraints_newton`
        #. :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_penalty_constraints_ls`

    scalarize the constrained problem using a quadratic penalty method,
    and apply L-BFGS, Newton, or :mod:`scipy.optimize.least_squares` to solve the penalty problem.
    Alternatively, the constraints can be enforced exactly (not with a penalty) using,

        :obj:`~simsopt.geo.BoozerSurface.minimize_boozer_exact_constraints_newton`

    In this approach, Newton's method is used to solve the first order necessary conditions for optimality. Note
    that this differs from the BoozerExact approach.The BoozerExact approach solves the residual equations directly
    at a specific set of colocation points on the surface,

        .. math::

            \mathbf r(x) = 0

            l(x) = l_0

            z(\varphi=0,\theta=0) = 0

    The colocation points are chosen such that the number of colocation points is equal to the number of unknowns
    in on the surface, so that the resulting nonlinear system of equations can be solved using
    Newton's method. The BoozerExact approach is implemented in

         :obj:`~simsopt.geo.BoozerSurface.solve_residual_equation_exactly_newton`

    Generally, the BoozerExact approach is faster than the BoozerLS approach, but it is less robust. Note that there
    are specific requirements on the set of colocation points, i.e. :mod:`surface.quadpoints_phi` and
    :mod:`surface.quadpoints_theta`, for stellarator symmetric BoozerExact surfaces. See the class method
    :obj:`~simsopt.geo.BoozerSurface.solve_residual_equation_exactly_newton` and :obj:`~simsopt.geo.SurfaceXYZTensorFourier.get_stellsym_mask()`
    for more information.

    *[1]: Giuliani A, Wechsung F, Stadler G, Cerfon A, Landreman M. Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasisymmetry. Journal of Plasma Physics. 2022;88(4):905880401. doi:10.1017/S0022377822000563*

    *[2]: Giuliani, A., Wechsung, F., Cerfon, A., Landreman, M., & Stadler, G. (2023). Direct stellarator coil optimization for nested magnetic surfaces with precise quasi-symmetry. Physics of Plasmas, 30(4).*
    """

    supports_explicit_surface_warm_start = False

    def __init__(
        self,
        biotsavart,
        surface,
        label,
        targetlabel,
        constraint_weight=None,
        options=None,
    ):
        """
        Args:
            biotsavart (:obj:`~simsopt.field.BiotSavart`): BiotSavart object.
            surface (:obj:`~simsopt.geo.SurfaceXYZFourier`, :obj:`~simsopt.geo.SurfaceXYZTensorFourier`): Surface object.
            label (:obj:`~simsopt._core.optimizable.Optimizable`): A method that computes a flux surface label for the surface, such as
                :obj:`~simsopt.geo.Volume`, :obj:`~simsopt.geo.Area`, or :obj:`~simsopt.geo.ToroidalFlux`.
            targetlabel (float): The target value of the label on the surface.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
                If None, then Boozer Exact is used in the :mod:`run_code` method.
            options (dict, Optional): A dictionary of solver options. If a keyword is not specified, then a default
                value is used. Possible keywords are:

                - `verbose` (bool): display convergence information. Defaults to True.
                - `newton_tol` (float): tolerance for newton solver. Defaults to 1e-13 for BoozerExact and 1e-11 for BoozerLS.
                - `bfgs_tol` (float): tolerance for bfgs solver. Defaults to 1e-10.
                - `newton_maxiter` (int): maximum number of iterations for Newton solver. Defaults to 40.
                - `bfgs_maxiter` (int): maximum number of iterations for BFGS solver. Defaults to 1500.
                - `limited_memory` (bool): True if L-BFGS solver is desired, False if the BFGS solver otherwise. Defaults to False.
                - `weight_inv_modB` (float): for BoozerLS surfaces, weight the residual by modB so that it does not scale with coil currents.  Defaults to True.
                - `newton_stab` (float): for BoozerLS surfaces, Tikhonov damping applied to the
                  Newton-polish Hessian as ``(H + stab * I)``. Defaults to 0.0 (no damping).
                  Mirrors the JAX-side default in :class:`BoozerSurfaceJAX` so both backends
                  honour the same regularization knob.
                - `record_scipy_callback_trace` (bool): for BoozerLS surfaces, record every SciPy objective callback evaluation. Defaults to False.
        """
        super().__init__(depends_on=[biotsavart])

        if not _is_supported_boozer_surface(surface):
            raise Exception(
                "The input surface must be a SurfaceXYZTensorFourier or SurfaceXYZFourier."
            )

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.boozer_type = "ls" if constraint_weight is not None else "exact"
        self.need_to_run_code = True

        if options is None:
            options = {}

        # set the default options now
        if "verbose" not in options:
            options["verbose"] = True

        # default solver options for the BoozerExact and BoozerLS solvers
        if self.boozer_type == "exact":
            if "newton_tol" not in options:
                options["newton_tol"] = 1e-13
            if "newton_maxiter" not in options:
                options["newton_maxiter"] = 40
        elif self.boozer_type == "ls":
            if "bfgs_tol" not in options:
                options["bfgs_tol"] = 1e-10
            if "newton_tol" not in options:
                options["newton_tol"] = 1e-11
            if "newton_maxiter" not in options:
                options["newton_maxiter"] = 40
            if "bfgs_maxiter" not in options:
                options["bfgs_maxiter"] = 1500
            if "limited_memory" not in options:
                options["limited_memory"] = False
            if "weight_inv_modB" not in options:
                options["weight_inv_modB"] = True
            if "newton_stab" not in options:
                options["newton_stab"] = 0.0
            if "record_scipy_callback_trace" not in options:
                options["record_scipy_callback_trace"] = False
        # Mirror BoozerSurfaceJAX._normalize_solver_options: in parity mode
        # the linear residual gate measures the *undamped* operator, so any
        # non-zero ``newton_stab`` would silently shift the CPU answer
        # relative to the JAX reference. Apply for all boozer_types so the
        # gate stays symmetric across CPU/JAX even if the option is ignored
        # for ``exact`` (matches BoozerSurfaceJAX which raises uniformly).
        if is_parity_mode() and float(options.get("newton_stab", 0.0)) != 0.0:
            raise ValueError(
                "BoozerSurface parity mode requires newton_stab=0.0 so "
                "linear residuals are checked against the undamped operator."
            )
        self.options = options

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def get_adjoint_runtime_state(self):
        """Expose the legacy dense adjoint contract through the runtime seam."""
        if self.need_to_run_code:
            raise RuntimeError(
                "BoozerSurface has no valid adjoint state. "
                "Call boozer_surface.run_code(...) before requesting adjoints."
            )
        linear_solve_factors = self.res.get("PLU")
        legacy_vjp = self.res.get("vjp")
        if linear_solve_factors is None or legacy_vjp is None:
            raise RuntimeError(
                "BoozerSurface has no valid adjoint state. "
                "Call boozer_surface.run_code(...) before requesting adjoints."
            )
        P, L, U = linear_solve_factors
        iota = self.res["iota"]
        G = self.res["G"]
        linearization_kind = (
            "exact_jacobian" if self.res.get("type") == "exact" else "hessian"
        )

        def solve_transpose(rhs):
            return forward_backward(P, L, U, rhs)

        def project_coil_adjoint_derivative(adjoint):
            return legacy_vjp(adjoint, self, iota, G)

        return _LegacyBoozerAdjointRuntimeState(
            linearization_kind=linearization_kind,
            decision_size=L.shape[0],
            solve_transpose=solve_transpose,
            project_coil_adjoint_derivative=project_coil_adjoint_derivative,
        )

    def _validate_none_G_precondition(self, G):
        if G is not None:
            return
        _require_fixed_currents_for_none_G(
            self.biotsavart,
            component="BoozerSurface",
            coil_attrs=("coils", "_coils"),
        )

    def run_code(self, iota, G=None):
        """
        Run the default solvers, i.e., run Newton's method directly if you are computing a BoozerExact surface,
        and run BFGS followed by Newton if you are computing a BoozerLS surface.

        Args:
            iota (float): Guess for value of rotational transform on the surface.
            G (float, Optional): Guess for value of G on surface, defaults to None. Note that if None is used, then the coil currents must be fixed.

        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - `"residual"`: the residual of the optimization problem
                - `"iter"`: the number of iterations taken to converge
                - `"success"`: True if the optimization converged, False otherwise
                - `"G"`: the value of G on the surface
                - `"s"`: the surface object
                - `"iota"`: the value of iota on the surface
                - `"PLU"`: the LU decomposition of the hessian

        """
        if not self.need_to_run_code:
            return

        # for coil optimizations, the gradient calculations of the objective assume
        # that the coil currents are fixed when G is None.
        self._validate_none_G_precondition(G)

        # BoozerExact default solver
        if self.boozer_type == "exact":
            res = self.solve_residual_equation_exactly_newton(
                iota=iota,
                G=G,
                tol=self.options["newton_tol"],
                maxiter=self.options["newton_maxiter"],
                verbose=self.options["verbose"],
            )
            return res

        # BoozerLS default solver
        elif self.boozer_type == "ls":
            # you need a label constraint for a BoozerLS surface
            assert self.constraint_weight is not None

            # first try BFGS.  You could also try L-BFGS by setting limited_memory=True in the options dictionary, which might be faster.  However, BFGS appears
            # to generally result in solutions closer to optimality.
            res = self.minimize_boozer_penalty_constraints_LBFGS(
                constraint_weight=self.constraint_weight,
                iota=iota,
                G=G,
                tol=self.options["bfgs_tol"],
                maxiter=self.options["bfgs_maxiter"],
                verbose=self.options["verbose"],
                limited_memory=self.options["limited_memory"],
                weight_inv_modB=self.options["weight_inv_modB"],
            )
            pre_newton_surface_dofs = np.asarray(
                res["s"].get_dofs(), dtype=float
            ).copy()
            pre_newton_decision_pieces = [
                pre_newton_surface_dofs,
                np.asarray([float(res["iota"])], dtype=float),
            ]
            if res["G"] is not None:
                pre_newton_decision_pieces.append(
                    np.asarray([float(res["G"])], dtype=float)
                )
            pre_newton = {
                "optimizer_method": res.get("optimizer_method"),
                "success": bool(res["success"]),
                "iter": int(res["iter"]),
                "fun": float(res["fun"]),
                "iota": float(res["iota"]),
                "G": None if res["G"] is None else float(res["G"]),
                "surface_dofs": pre_newton_surface_dofs,
                "decision_vector": np.concatenate(pre_newton_decision_pieces),
                "gradient": np.asarray(res["gradient"], dtype=float).copy(),
                "scipy_call_contract": res.get("scipy_call_contract"),
                "scipy_initial_call": res.get("scipy_initial_call"),
                "scipy_callback_trace": res.get("scipy_callback_trace"),
            }
            iota, G = res["iota"], res["G"]

            ## polish off using Newton's method
            self.need_to_run_code = True
            res = self.minimize_boozer_penalty_constraints_newton(
                constraint_weight=self.constraint_weight,
                iota=iota,
                G=G,
                verbose=self.options["verbose"],
                tol=self.options["newton_tol"],
                maxiter=self.options["newton_maxiter"],
                stab=self.options["newton_stab"],
                weight_inv_modB=self.options["weight_inv_modB"],
            )
            res["optimizer_method"] = pre_newton["optimizer_method"]
            res["pre_newton"] = pre_newton
            return res

    def boozer_penalty_constraints(
        self,
        x,
        derivatives=0,
        constraint_weight=1.0,
        scalarize=True,
        optimize_G=False,
        weight_inv_modB=True,
    ):
        r"""
        Define the residual

        .. math::
            \mathbf r(x) = [r_1(x),...,r_n(x), \sqrt{w_c}  (l-l_0), \sqrt{w_c}  (z(\varphi=0, \theta=0) - 0)]

        where :math:`w_c` is the constraint weight, :math:`r_i` are the Boozer residuals
        at quadrature points :math:`1,\dots,n`, :math:`l` is the surface label, and :math:`l_0` is
        the target surface label.

        For ``scalarized=False``, this function returns :math:`\mathbf r(x)` and optionally the Jacobian
        of :math:`\mathbf r(x)`.

        for ``scalarized=True``, this function returns

        .. math::
            J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x),

        i.e. the least squares residual and optionally the gradient and the Hessian of :math:`J(x)`.

        If ``weight_inv_modB=True``, the Boozer residuals are weighted by the inverse of the field strength
        (i.e. multiplied by :math:`1/\|\mathbf B \|`), otherwise, they are unweighted (multiplied by 1).  Setting
        this to True is useful to prevent the least squares residual from scaling with the coil currents.

        Args:
            x (ndarray): The degrees of freedom of the Surface object, followed by the value of iota and G.
                e.g. ``[surface.x, iota, G]`` or ``[surface.x, iota]`` if ``optimize_G=False``.
            derivatives (int, Optional): 0 if no derivatives are requested, 1 if first derivatives are requested.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            scalarize (bool, Optional): If True, return the least squares residual, otherwise return the residual vector.
            optimize_G (bool, Optional): True if G is a variable in the optimization problem, False otherwise.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.

        Returns:
            tuple: If ``scalarize=True``, return ``(val, dval, d2val)`` the values of the least squares residual, its gradient and Hessian.
            If ``derivatives=0``, then ``dval`` and ``d2val`` are None. If ``scalarize=False``, return ``(r, J)`` the residual vector and the Jacobian
            of the optimization problem. If ``derivatives=0``, then ``J`` is None.
        """

        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = x[:-2]
            iota = x[-2]
            G = x[-1]
        else:
            sdofs = x[:-1]
            iota = x[-1]
            G = None
        nsurfdofs = sdofs.size
        s = self.surface
        num_res = 3 * s.quadpoints_phi.size * s.quadpoints_theta.size
        biotsavart = self.biotsavart

        s.set_dofs(sdofs)

        boozer = boozer_surface_residual(
            s,
            iota,
            G,
            biotsavart,
            derivatives=derivatives,
            weight_inv_modB=weight_inv_modB,
        )
        # normalizing the residuals here
        boozer = tuple([b / np.sqrt(num_res) for b in boozer])

        r = boozer[0]
        l = self.label.J()
        rl = l - self.targetlabel
        rz = s.gamma()[0, 0, 2] - 0.0
        r = np.concatenate(
            (r, [np.sqrt(constraint_weight) * rl, np.sqrt(constraint_weight) * rz])
        )

        val = 0.5 * np.sum(r**2)
        if derivatives == 0:
            if scalarize:
                return val
            else:
                return r

        J = boozer[1]

        dl = np.zeros(x.shape)
        drz = np.zeros(x.shape)

        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)

        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]
        J = np.concatenate(
            (
                J,
                np.sqrt(constraint_weight) * dl[None, :],
                np.sqrt(constraint_weight) * drz[None, :],
            ),
            axis=0,
        )
        dval = np.sum(r[:, None] * J, axis=0)
        if derivatives == 1:
            if scalarize:
                return val, dval
            else:
                return r, J
        if not scalarize:
            raise NotImplementedError("Can only return Hessian for scalarized version.")

        H = boozer[2]

        d2l = np.zeros((x.shape[0], x.shape[0]))
        d2l[:nsurfdofs, :nsurfdofs] = (
            self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        )

        H = np.concatenate(
            (
                H,
                np.sqrt(constraint_weight) * d2l[None, :, :],
                np.zeros(d2l[None, :, :].shape),
            ),
            axis=0,
        )
        d2val = J.T @ J + np.sum(r[:, None, None] * H, axis=0)
        return val, dval, d2val

    def _boozer_penalty_vectorized_inputs(self, sdofs, derivatives):
        """Materialize the arrays consumed by ``_call_boozer_residual_ds*``.

        This is the CPU-side boundary helper for the bit-identity census
        (``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` Phase
        1). The same materialized arrays feed ``_call_boozer_residual``,
        ``_call_boozer_residual_ds``, and ``_call_boozer_residual_ds2`` inside
        ``boozer_penalty_constraints_vectorized``. Splitting the materialization
        out lets the parity census assert byte-identity on these arrays without
        re-running the residual contraction.

        Args:
            sdofs (ndarray): Surface DOFs only (without iota/G appended).
            derivatives (int): 0, 1, or 2; controls whether Jacobian and
                Hessian inputs are also returned.

        Side effects:
            Calls ``self.surface.set_dofs(sdofs)`` and
            ``self.biotsavart.set_points`` / ``self.biotsavart.compute`` at the
            resulting surface quadrature points.

        Returns:
            dict: keys ``'gamma'``, ``'xphi'``, ``'xtheta'``, ``'B'``,
            ``'nphi'``, ``'ntheta'``. When ``derivatives >= 1`` it additionally
            contains ``'dB_dx'``, ``'dx_dc'``, ``'dxphi_dc'``, and
            ``'dxtheta_dc'``. When ``derivatives == 2`` it also contains
            ``'d2B_by_dXdX'``.
        """
        assert derivatives in (0, 1, 2)
        s = self.surface
        s.set_dofs(sdofs)
        biotsavart = self.biotsavart
        gamma = s.gamma()
        xphi = s.gammadash1()
        xtheta = s.gammadash2()
        nphi = gamma.shape[0]
        ntheta = gamma.shape[1]
        gamma_flat = gamma.reshape((gamma.size // 3, 3)).copy()
        biotsavart.set_points(gamma_flat)
        biotsavart.compute(derivatives)
        inputs = {
            "gamma": gamma,
            "xphi": xphi,
            "xtheta": xtheta,
            "B": biotsavart.B().reshape((nphi, ntheta, 3)),
            "nphi": nphi,
            "ntheta": ntheta,
        }
        if derivatives >= 1:
            inputs["dx_dc"] = s.dgamma_by_dcoeff()
            inputs["dxphi_dc"] = s.dgammadash1_by_dcoeff()
            inputs["dxtheta_dc"] = s.dgammadash2_by_dcoeff()
            inputs["dB_dx"] = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        if derivatives == 2:
            inputs["d2B_by_dXdX"] = biotsavart.d2B_by_dXdX().reshape(
                (nphi, ntheta, 3, 3, 3)
            )
        return inputs

    def boozer_penalty_constraints_vectorized(
        self,
        dofs,
        derivatives=0,
        constraint_weight=1.0,
        optimize_G=False,
        weight_inv_modB=True,
    ):
        """
        This function returns the same thing as `boozer_penalty_constraints` when `scalarized=True`.  It
        is much faster and uses less memory since it calls a vectorized implementation in cpp. This is
        especially true when `derivatives=2`, i.e., when the Hessian is requested.

        Args:
            dofs (ndarray): The degrees of freedom of the Surface object, followed by the value of iota and G.
                e.g. ``[surface.x, iota, G]`` or ``[surface.x, iota]`` if ``optimize_G=False``.
            derivatives (int, Optional): 0 if no derivatives are requested, 1 if first derivatives are requested.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            optimize_G (bool, Optional): True if G is a variable in the optimization problem, False otherwise.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.

        Returns:
            tuple: ``(r, J, H)`` The residual vector, the Jacobian of the optimization problem, and the Hessian of the optimization problem.
            If ``derivatives=0``, then ``J`` and ``H`` are None. If ``derivatives=1``, then ``H`` is None.
        """

        assert derivatives in [0, 1, 2]
        if optimize_G:
            sdofs = dofs[:-2]
            iota = dofs[-2]
            G = dofs[-1]
        else:
            sdofs = dofs[:-1]
            iota = dofs[-1]
            G = (
                2.0
                * np.pi
                * np.sum(
                    np.abs(
                        [coil.current.get_value() for coil in self.biotsavart._coils]
                    )
                )
                * (4 * np.pi * 10 ** (-7) / (2 * np.pi))
            )

        s = self.surface
        nsurfdofs = sdofs.size

        inputs = self._boozer_penalty_vectorized_inputs(sdofs, derivatives)
        nphi = inputs["nphi"]
        ntheta = inputs["ntheta"]
        xphi = inputs["xphi"]
        xtheta = inputs["xtheta"]
        B = inputs["B"]
        if derivatives >= 1:
            dB_dx = inputs["dB_dx"]
            dx_dc = inputs["dx_dc"]
            dxphi_dc = inputs["dxphi_dc"]
            dxtheta_dc = inputs["dxtheta_dc"]
        if derivatives == 2:
            d2B_by_dXdX = inputs["d2B_by_dXdX"]

        num_res = 3 * nphi * ntheta
        if derivatives == 0:
            val = _call_boozer_residual(G, iota, xphi, xtheta, B, weight_inv_modB)
            boozer = (val,)
        elif derivatives == 1:
            val, dval = _call_boozer_residual_ds(
                G,
                iota,
                B,
                dB_dx,
                xphi,
                xtheta,
                dx_dc,
                dxphi_dc,
                dxtheta_dc,
                weight_inv_modB,
            )
            boozer = val, dval
        elif derivatives == 2:
            val, dval, d2val = _call_boozer_residual_ds2(
                G,
                iota,
                B,
                dB_dx,
                d2B_by_dXdX,
                xphi,
                xtheta,
                dx_dc,
                dxphi_dc,
                dxtheta_dc,
                weight_inv_modB,
            )
            boozer = val, dval, d2val

        # normalizing the residuals here
        boozer = tuple([b / num_res for b in boozer])

        lab = self.label.J()

        rnl = boozer[0]
        rl = np.sqrt(constraint_weight) * (lab - self.targetlabel)
        rz = np.sqrt(constraint_weight) * (s.gamma()[0, 0, 2] - 0.0)
        r = rnl + 0.5 * rl**2 + 0.5 * rz**2

        if derivatives == 0:
            return r

        dl = np.zeros(dofs.shape)
        drz = np.zeros(dofs.shape)
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        Jnl = boozer[1]
        if not optimize_G:
            Jnl = Jnl[:-1]

        drl = np.sqrt(constraint_weight) * dl
        drz = np.sqrt(constraint_weight) * drz
        J = Jnl + rl * drl + rz * drz

        if derivatives == 1:
            return r, J

        Hnl = boozer[2]
        if not optimize_G:
            Hnl = Hnl[:-1, :-1]

        d2rl = np.zeros((dofs.shape[0], dofs.shape[0]))
        d2rl[:nsurfdofs, :nsurfdofs] = (
            np.sqrt(constraint_weight)
            * self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        )
        H = Hnl + drl[:, None] @ drl[None, :] + drz[:, None] @ drz[None, :] + rl * d2rl

        return r, J, H

    def boozer_exact_constraints(self, xl, derivatives=0, optimize_G=True):
        r"""
        This function returns the optimality conditions corresponding to the minimization problem

        .. math::
            \text{min}_x ~J(x)

        subject to 

        .. math::
            l - l_0 &= 0 \\
            z(\varphi=0,\theta=0) - 0 &= 0

        The function can additionally return the first derivatives of these optimality conditions.

        Args:
            xl (ndarray): The degrees of freedom of the Surface object, followed by the value of iota and G.
                e.g. ``[surface.x, iota, G]`` or ``[surface.x, iota]`` if ``optimize_G=False``.
            derivatives (int, Optional): 0 if no derivatives are requested, 1 if first derivatives are requested.
            optimize_G (bool, Optional): True if G is a variable in the optimization problem, False otherwise.

        Returns:
            If ``derivatives=0``, return ``res`` the residual of the optimization problem.
            If ``derivatives=1``, return ``(res, dres)`` the residual and the Jacobian of the optimization problem.
        """
        assert derivatives in [0, 1]
        if optimize_G:
            sdofs = xl[:-4]
            iota = xl[-4]
            G = xl[-3]
        else:
            sdofs = xl[:-3]
            iota = xl[-3]
            G = None
        lm = xl[-2:]
        s = self.surface
        biotsavart = self.biotsavart
        s.set_dofs(sdofs)
        nsurfdofs = sdofs.size

        boozer = boozer_surface_residual(
            s, iota, G, biotsavart, derivatives=derivatives + 1
        )
        r, J = boozer[0:2]

        dl = np.zeros((xl.shape[0] - 2,))

        l = self.label.J()
        dl[:nsurfdofs] = self.label.dJ(partials=True)(s)
        drz = np.zeros((xl.shape[0] - 2,))
        g = [l - self.targetlabel]
        rz = s.gamma()[0, 0, 2] - 0.0
        drz[:nsurfdofs] = s.dgamma_by_dcoeff()[0, 0, 2, :]

        res = np.zeros(xl.shape)
        res[:-2] = np.sum(r[:, None] * J, axis=0) - lm[-2] * dl - lm[-1] * drz
        res[-2] = g[0]
        res[-1] = rz
        if derivatives == 0:
            return res

        H = boozer[2]

        d2l = np.zeros((xl.shape[0] - 2, xl.shape[0] - 2))
        d2l[:nsurfdofs, :nsurfdofs] = (
            self.label.d2J_by_dsurfacecoefficientsdsurfacecoefficients()
        )

        dres = np.zeros((xl.shape[0], xl.shape[0]))
        dres[:-2, :-2] = J.T @ J + np.sum(r[:, None, None] * H, axis=0) - lm[-2] * d2l
        dres[:-2, -2] = -dl
        dres[:-2, -1] = -drz

        dres[-2, :-2] = dl
        dres[-1, :-2] = drz
        return res, dres

    def minimize_boozer_penalty_constraints_LBFGS(
        self,
        tol=1e-3,
        maxiter=1000,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        limited_memory=True,
        weight_inv_modB=True,
        verbose=False,
    ):
        r"""
        This function uses L-BFGS to find the surface that approximately solves

        .. math::
            \text{min}_x ~J(x) + \frac{1}{2} w_c (l - l_0)^2
                                 + \frac{1}{2} w_c (z(\varphi=0, \theta=0) - 0)^2

        where :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-3.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 1000.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            limited_memory (bool, Optional): If True, use the limited memory version of L-BFGS. Defaults to True.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.
            verbose (bool, Optional): If True, print the optimization progress. Defaults to False.

        Returns:
            res (dict): A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'fun': the value of the objective function at the solution
                - 'gradient': the gradient of the objective function at the solution
                - 'iter': the number of iterations taken to converge
                - 'info': the optimization result
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 's': the surface object
                - 'iota': the value of iota on the surface
                - 'weight_inv_modB': the value of weight_inv_modB used in the optimization
                - 'type': the type of optimization used

        """

        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))

        def fun(x):
            return self.boozer_penalty_constraints_vectorized(
                x,
                derivatives=1,
                constraint_weight=constraint_weight,
                optimize_G=G is not None,
                weight_inv_modB=weight_inv_modB,
            )

        method = "L-BFGS-B" if limited_memory else "BFGS"
        options = {"maxiter": maxiter, "gtol": tol}
        if limited_memory:
            options["maxcor"] = 200
            options["ftol"] = tol

        scipy_initial_call = {}
        scipy_callback_trace = (
            [] if self.options["record_scipy_callback_trace"] else None
        )

        def scipy_fun(x_np):
            value, gradient = fun(x_np)
            if "payload" not in scipy_initial_call:
                scipy_initial_call["payload"] = {
                    "decision_vector": np.asarray(x_np, dtype=float).copy(),
                    "fun": np.asarray(value, dtype=float)[()],
                    "gradient": np.asarray(gradient, dtype=float).copy(),
                }
            if scipy_callback_trace is not None:
                scipy_callback_trace.append(
                    {
                        "decision_vector": np.asarray(x_np, dtype=float).copy(),
                        "fun": np.asarray(value, dtype=float)[()],
                        "gradient": np.asarray(gradient, dtype=float).copy(),
                    }
                )
            return value, gradient

        res = minimize(scipy_fun, x, jac=True, method=method, options=options)
        scipy_call_contract = {
            "semantic_method": "lbfgs" if limited_memory else "bfgs",
            "scipy_method": method,
            "scipy_options": dict(options),
            "callback": None,
            "success": bool(res.success),
            "status": int(getattr(res, "status", 0)),
            "message": str(getattr(res, "message", "")),
            "nit": int(getattr(res, "nit", 0)),
            "nfev": int(getattr(res, "nfev", 0)),
            "njev": int(getattr(res, "njev", 0)),
        }
        res.scipy_call_contract = scipy_call_contract
        res.scipy_initial_call = scipy_initial_call["payload"]
        res.scipy_callback_trace = scipy_callback_trace

        resdict = {
            "fun": res.fun,
            "gradient": res.jac,
            "iter": res.nit,
            "info": res,
            "success": res.success,
            "G": None,
            "optimizer_method": method,
            "scipy_call_contract": scipy_call_contract,
            "scipy_initial_call": scipy_initial_call["payload"],
            "scipy_callback_trace": scipy_callback_trace,
            "weight_inv_modB": weight_inv_modB,
            "type": "ls",
        }
        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]
            resdict["G"] = G
        resdict["s"] = s
        resdict["iota"] = iota

        self.res = resdict
        self.need_to_run_code = False

        if verbose:
            print(
                f"{method} solve - {resdict['success']}  iter={resdict['iter']}, iota={resdict['iota']:.16f}, ||grad||_inf = {np.linalg.norm(resdict['gradient'], ord=np.inf):.3e}",
                flush=True,
            )

        return resdict

    def minimize_boozer_penalty_constraints_newton(
        self,
        tol=1e-12,
        maxiter=10,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        stab=0.0,
        weight_inv_modB=True,
        verbose=False,
    ):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it uses
        Newton's method.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-12.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            stab (float, Optional): The stabilization parameter for the Newton method. Defaults to 0.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.
            verbose (bool, Optional): If True, print the optimization progress. Defaults to False.

        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'jacobian': the value of the Jacobian at the solution
                - 'hessian': the value of the Hessian at the solution
                - 'iter': the number of iterations taken to converge
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 'iota': the value of iota on the surface
                - 'PLU': the LU decomposition of the hessian
                - 'type': 'ls'.
                - 'weight_inv_modB': the value of weight_inv_modB used in the optimization
        """
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        G_provided = G is not None
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0

        val, dval, d2val = self.boozer_penalty_constraints_vectorized(
            x,
            derivatives=2,
            constraint_weight=constraint_weight,
            optimize_G=G is not None,
            weight_inv_modB=weight_inv_modB,
        )

        norm = np.linalg.norm(dval)
        final_step_dense_refinement_ran = False
        dense_refinement_ran = False
        while i < maxiter and norm > tol:
            d2val += stab * np.identity(d2val.shape[0])
            dx = np.linalg.solve(d2val, dval)
            refine_step = norm < 1e-9
            if refine_step:
                dx += np.linalg.solve(d2val, dval - d2val @ dx)
                dense_refinement_ran = True
            x = x - dx
            val, dval, d2val = self.boozer_penalty_constraints_vectorized(
                x,
                derivatives=2,
                constraint_weight=constraint_weight,
                optimize_G=G is not None,
                weight_inv_modB=weight_inv_modB,
            )
            norm = np.linalg.norm(dval)
            i = i + 1
            final_step_dense_refinement_ran = bool(refine_step)

        r = self.boozer_penalty_constraints(
            x,
            derivatives=0,
            constraint_weight=constraint_weight,
            scalarize=False,
            optimize_G=G is not None,
            weight_inv_modB=weight_inv_modB,
        )

        # Match the JAX backend's contract (``optimizer_jax.newton_polish``):
        # the reported Hessian and its PLU factors are the *stabilized*
        # operator ``H + stab * I`` so adjoint/PLU consumers see the same
        # linear system on both backends. With ``stab=0`` (the default) the
        # reported Hessian is unchanged.
        if stab:
            np.fill_diagonal(d2val, d2val.diagonal() + stab)

        P, L, U = lu(d2val)
        hessian_shape = tuple(int(dim) for dim in d2val.shape)
        res = {
            "fun": val,
            "residual": r,
            "jacobian": dval,
            "hessian": d2val,
            "iter": i,
            "newton_iter": i,
            "success": norm <= tol,
            "G": None,
            "PLU": (P, L, U),
            "vjp": _guard_none_G_coil_gradient_callback(
                partial(
                    boozer_surface_dlsqgrad_dcoils_vjp,
                    weight_inv_modB=weight_inv_modB,
                ),
                biotsavart=self.biotsavart,
                component="BoozerSurface",
                coil_attrs=("coils", "_coils"),
                G_provided=G_provided,
            ),
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
            "linearization_kind": "hessian",
            "linear_solve_backend": "dense-plu",
            "dense_linear_solve_factors_available": True,
            "dense_newton_steps_materialized": True,
            "hessian_materialized": True,
            "dense_hessian_shape": hessian_shape,
            "dense_hessian_bytes": int(d2val.nbytes),
            "max_dense_hessian_bytes": None,
            "final_gradient_norm": float(norm),
            "final_gradient_inf_norm": float(np.linalg.norm(dval, ord=np.inf)),
            "iterative_refinement_ran": bool(dense_refinement_ran),
            "final_step_iterative_refinement_ran": bool(
                final_step_dense_refinement_ran
            ),
            "dense_refinement_ran": bool(dense_refinement_ran),
            "final_step_dense_refinement_ran": bool(final_step_dense_refinement_ran),
        }
        if G is None:
            s.set_dofs(x[:-1])
            iota = x[-1]
        else:
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            res["G"] = G
        res["iota"] = iota

        self.res = res
        self.need_to_run_code = False

        if verbose:
            print(
                f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, ||grad||_inf = {np.linalg.norm(res['jacobian'], ord=np.inf):.3e}",
                flush=True,
            )

        return res

    def minimize_boozer_penalty_constraints_ls(
        self,
        tol=1e-12,
        maxiter=10,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        method="lm",
        weight_inv_modB=True,
    ):
        """
        This function does the same as :mod:`minimize_boozer_penalty_constraints_LBFGS`, but instead of LBFGS it
        uses a nonlinear least squares algorithm when ``method='lm'``.  Options for the method
        are the same as for :mod:`scipy.optimize.least_squares`. If ``method='manual'``, then a
        damped Gauss-Newton method is used.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-12.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            constraint_weight (float, Optional): The weight of the label constraint used when solving Boozer least squares.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            method (str, Optional): The method to use for the optimization. Defaults to 'lm'.
            weight_inv_modB (bool, Optional): If True, weight the residual by modB so that it does not scale with coil currents. Defaults to True.

        Returns:
            res (dict): A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'gradient': the value of the gradient at the solution
                - 'jacobian': the value of the jacobian at the solution
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 's': the surface object
                - 'iota': the value of iota on the surface
        """

        if not self.need_to_run_code:
            return self.res

        s = self.surface
        if G is None:
            x = np.concatenate((s.get_dofs(), [iota]))
        else:
            x = np.concatenate((s.get_dofs(), [iota, G]))
        norm = 1e10
        if method == "manual":
            i = 0
            lam = 1.0
            r, J = self._get_residual_vector_and_jacobian(
                x, constraint_weight, G is not None, weight_inv_modB
            )
            b = J.T @ r
            JTJ = J.T @ J
            while i < maxiter and norm > tol:
                dx = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ)), b)
                x -= dx
                r, J = self._get_residual_vector_and_jacobian(
                    x, constraint_weight, G is not None, weight_inv_modB
                )
                b = J.T @ r
                JTJ = J.T @ J
                norm = np.linalg.norm(b)
                lam *= 1 / 3
                i += 1
            resdict = {
                "residual": r,
                "gradient": b,
                "jacobian": JTJ,
                "success": norm <= tol,
            }
            if G is None:
                s.set_dofs(x[:-1])
                iota = x[-1]
            else:
                s.set_dofs(x[:-2])
                iota = x[-2]
                G = x[-1]
                resdict["G"] = G
            resdict["s"] = s
            resdict["iota"] = iota
            return resdict

        def fun(x):
            return self._get_residual_vector_and_jacobian(
                x, constraint_weight, G is not None, weight_inv_modB
            )[0]

        def jac(x):
            return self._get_residual_vector_and_jacobian(
                x, constraint_weight, G is not None, weight_inv_modB
            )[1]

        res = least_squares(
            fun,
            x,
            jac=jac,
            method=method,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            x_scale=1.0,
            max_nfev=maxiter,
        )
        resdict = {
            "info": res,
            "residual": res.fun,
            "gradient": res.grad,
            "jacobian": res.jac,
            "success": res.status > 0,
            "G": None,
        }
        if G is None:
            s.set_dofs(res.x[:-1])
            iota = res.x[-1]
        else:
            s.set_dofs(res.x[:-2])
            iota = res.x[-2]
            G = res.x[-1]
            resdict["G"] = G
        resdict["s"] = s
        resdict["iota"] = iota

        self.res = resdict
        self.need_to_run_code = False
        return resdict

    def _get_residual_vector_and_jacobian(
        self, x, constraint_weight, optimize_G, weight_inv_modB
    ):
        return self.boozer_penalty_constraints(
            x,
            derivatives=1,
            constraint_weight=constraint_weight,
            scalarize=False,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
        )

    def minimize_boozer_exact_constraints_newton(
        self, tol=1e-12, maxiter=10, iota=0.0, G=None, lm=[0.0, 0.0]
    ):
        r"""
        This function solves the constrained optimization problem

        .. math::
            \text{min}_x ~ J(x)

        subject to

        .. math::
            l - l_0 &= 0 \\
            z(\varphi=0,\theta=0) - 0 &= 0

        using the method of Lagrange multipliers and applying Newton's method. In the above,
        :math:`J(x) = \frac{1}{2}\mathbf r(x)^T \mathbf r(x)`, and :math:`\mathbf r(x)` contains
        the Boozer residuals at quadrature points :math:`1,\dots,n`.

        The final constraint is not necessary for stellarator symmetric surfaces as it is automatically
        satisfied by the stellarator symmetric surface parametrization.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-12.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            lm (list, Optional): The initial guesses for the Lagrange multipliers. Defaults to [0., 0.].

        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'jacobian': the value of the jacobian at the solution
                - 'iter': the number of iterations taken to converge
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 'lm': the value of the Lagrange multipliers
        """

        if not self.need_to_run_code:
            return self.res

        s = self.surface
        optimize_G = G is not None
        if optimize_G:
            xl = np.concatenate((s.get_dofs(), [iota, G], lm))
        else:
            xl = np.concatenate((s.get_dofs(), [iota], lm))
        val, dval = self.boozer_exact_constraints(
            xl, derivatives=1, optimize_G=optimize_G
        )
        norm = np.linalg.norm(val)
        if s.stellsym:
            i = 0
            while i < maxiter and norm > tol:
                A = dval[:-1, :-1]
                b = val[:-1]
                dx = np.linalg.solve(A, b)
                if (
                    norm < 1e-9
                ):  # iterative refinement for higher accuracy. TODO: cache LU factorisation
                    dx += np.linalg.solve(A, b - A @ dx)
                xl[:-1] = xl[:-1] - dx
                val, dval = self.boozer_exact_constraints(
                    xl, derivatives=1, optimize_G=optimize_G
                )
                norm = np.linalg.norm(val)
                i = i + 1
        else:

            def exact_residual_and_jacobian(x):
                return self.boozer_exact_constraints(
                    x, derivatives=1, optimize_G=optimize_G
                )

            def solve_exact_fallback(x0, *, xtol, maxfev):
                return root(
                    exact_residual_and_jacobian,
                    x0,
                    jac=True,
                    method="hybr",
                    options={
                        "xtol": xtol,
                        "maxfev": maxfev,
                    },
                )

            fallback_xtol = max(1e-12, tol * 1e-2)
            fallback = solve_exact_fallback(
                xl, xtol=fallback_xtol, maxfev=max(400, 4 * maxiter)
            )
            fallback_xl = np.asarray(fallback.x)
            fallback_val, fallback_dval = self.boozer_exact_constraints(
                fallback_xl, derivatives=1, optimize_G=optimize_G
            )
            fallback_norm = np.linalg.norm(fallback_val)
            fallback_nfev = getattr(fallback, "nfev", 0)
            polish_xtol = min(1e-10, fallback_xtol)
            if (
                np.isfinite(fallback_norm)
                and fallback_norm > tol
                and polish_xtol < fallback_xtol
            ):
                polished = solve_exact_fallback(
                    fallback_xl,
                    xtol=polish_xtol,
                    maxfev=max(200, 2 * maxiter),
                )
                polished_xl = np.asarray(polished.x)
                polished_val, polished_dval = self.boozer_exact_constraints(
                    polished_xl, derivatives=1, optimize_G=optimize_G
                )
                polished_norm = np.linalg.norm(polished_val)
                if np.isfinite(polished_norm) and polished_norm <= fallback_norm:
                    fallback_xl = polished_xl
                    fallback_val = polished_val
                    fallback_dval = polished_dval
                    fallback_norm = polished_norm
                fallback_nfev += getattr(polished, "nfev", 0)
            if np.isfinite(fallback_norm) and fallback_norm <= norm:
                xl = fallback_xl
                val = fallback_val
                dval = fallback_dval
                norm = fallback_norm
            # nfev: root() does not report iteration count for hybr
            i = fallback_nfev

        if s.stellsym:
            lm = xl[-2]
        else:
            lm = xl[-2:]

        res = {
            "residual": val,
            "jacobian": dval,
            "iter": i,
            "success": norm <= tol,
            "lm": lm,
            "G": None,
        }
        if optimize_G:
            s.set_dofs(xl[:-4])
            iota = xl[-4]
            G = xl[-3]
            res["G"] = G
        else:
            s.set_dofs(xl[:-3])
            iota = xl[-3]
        res["s"] = s
        res["iota"] = iota

        self.res = res
        self.need_to_run_code = False
        return res

    def solve_residual_equation_exactly_newton(
        self, tol=1e-10, maxiter=10, iota=0.0, G=None, verbose=False
    ):
        """
        The function implements the BoozerExact approach by solving residual equation exactly using Newtons
        method.

        For Newton's method to be applied, we need the right balance of quadrature points, degrees
        of freedom and constraints.  For this reason, this function is only implemented for
        surfaces of type :obj:`~simsopt.geo.SurfaceXYZTensorFourier` right now.

        Given ``ntor``, ``mpol``, ``nfp`` and ``stellsym``, the surface is expected to be
        created in the following way::

            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            s = SurfaceXYZTensorFourier(
                mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                quadpoints_phi=phis, quadpoints_theta=thetas)

        Or the following two are also possible in the stellsym case::

            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 0.5, mpol+1, endpoint=False)

        or::

            phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)

        and then::

            s = SurfaceXYZTensorFourier(
                mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                quadpoints_phi=phis, quadpoints_theta=thetas)

        For the stellarator symmetric case, there is some redundancy between DOFs.  This is
        taken care of inside this function.

        In the non-stellarator-symmetric case, the surface has
        ``(2*ntor+1)*(2*mpol+1)`` many quadrature points and
        ``3*(2*ntor+1)*(2*mpol+1)`` many dofs.

        Equations:
            - Boozer residual in x, y, and z at all quadrature points
            - z(0, 0) = 0
            - label constraint (e.g. volume or flux)

        Unknowns:
            - Surface dofs
            - iota
            - G

        So we end up having ``3*(2*ntor+1)*(2*mpol+1) + 2`` equations and the
        same number of unknowns.

        In the stellarator-symmetric case, we have
        ``D = (ntor+1)*(mpol+1)+ ntor*mpol + 2*(ntor+1)*mpol + 2*ntor*(mpol+1)
        = 6*ntor*mpol + 3*ntor + 3*mpol + 1``
        many dofs in the surface. After calling ``surface.get_stellsym_mask()`` we have kicked out
        ``2*ntor*mpol + ntor + mpol``
        quadrature points, i.e. we have
        ``2*ntor*mpol + ntor + mpol + 1``
        quadrature points remaining. In addition we know that the x coordinate of the
        residual at phi=0=theta is also always satisfied. In total this
        leaves us with
        ``3*(2*ntor*mpol + ntor + mpol) + 2`` equations for the boozer residual, plus
        1 equation for the label,
        which is the same as the number of surface dofs + 2 extra unknowns
        given by iota and G.

        Args:
            tol (float, Optional): The tolerance for the optimization. Defaults to 1e-10.
            maxiter (int, Optional): The maximum number of iterations for the optimization. Defaults to 10.
            iota (float, Optional): The initial guess for the value of the rotational transform on the surface. Defaults to 0.
            G (float, Optional): The initial guess for the value of G on the surface. Defaults to None.
            verbose (bool, Optional): If True, print the optimization progress. Defaults to False.

        Returns:
            dict: A dictionary containing the results of the optimization. The dictionary contains the following keys in addition
            to others:

                - 'residual': the value of the residual at the solution
                - 'jacobian': the value of the jacobian at the solution
                - 'iter': the number of iterations taken to converge
                - 'success': True if the optimization converged, False otherwise
                - 'G': the value of G on the surface
                - 's': the surface object
                - 'iota': the value of iota on the surface
                - 'PLU': the LU decomposition of the jacobian
                - 'mask': a mask for the residuals that are not used in the optimization
                - 'type': 'exact'.
                - 'vjp': the vector-Jacobian product for the optimization
        """
        if not self.need_to_run_code:
            return self.res

        s = self.surface
        G_provided = G is not None
        if not _is_supported_boozer_exact_surface(s):
            raise RuntimeError(
                "Exact solution of Boozer Surfaces only supported for SurfaceXYZTensorFourier"
            )

        # In the case of stellarator symmetry, some of the information is
        # redundant, since the coordinates at (-phi, -theta) are the same (up
        # to sign changes) to those at (phi, theta). In addition, for stellsym
        # surfaces and stellsym magnetic fields, the residual in the x
        # component is always satisfied at phi=theta=0, so we ignore that one
        # too. The mask object below is True for those parts of the residual
        # that we need to keep, and False for those that we ignore.
        m = s.get_stellsym_mask()
        mask = np.concatenate((m[..., None], m[..., None], m[..., None]), axis=2)
        if s.stellsym:
            mask[0, 0, 0] = False
        mask = mask.flatten()

        label = self.label
        if G is None:
            G = (
                2.0
                * np.pi
                * np.sum(np.abs([c.current.get_value() for c in self.biotsavart.coils]))
                * (4 * np.pi * 10 ** (-7) / (2 * np.pi))
            )
        x = np.concatenate((s.get_dofs(), [iota, G]))
        i = 0
        r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1)
        norm = 1e6
        while i < maxiter:
            if s.stellsym:
                b = np.concatenate((r[mask], [(label.J() - self.targetlabel)]))
            else:
                b = np.concatenate(
                    (r[mask], [(label.J() - self.targetlabel), s.gamma()[0, 0, 2]])
                )
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            if s.stellsym:
                J = np.vstack(
                    (
                        J[mask, :],
                        np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                    )
                )
            else:
                J = np.vstack(
                    (
                        J[mask, :],
                        np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                        np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0.0, 0.0])),
                    )
                )
            dx = np.linalg.solve(J, b)
            dx += np.linalg.solve(J, b - J @ dx)
            x -= dx
            s.set_dofs(x[:-2])
            iota = x[-2]
            G = x[-1]
            i += 1
            r, J = boozer_surface_residual(s, iota, G, self.biotsavart, derivatives=1)

        if s.stellsym:
            J = np.vstack(
                (
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                )
            )
        else:
            J = np.vstack(
                (
                    J[mask, :],
                    np.concatenate((label.dJ(partials=True)(s), [0.0, 0.0])),
                    np.concatenate((s.dgamma_by_dcoeff()[0, 0, 2, :], [0.0, 0.0])),
                )
            )

        P, L, U = lu(J)
        res = {
            "residual": r,
            "jacobian": J,
            "iter": i,
            "success": norm <= tol,
            "G": G,
            "s": s,
            "iota": iota,
            "PLU": (P, L, U),
            "mask": mask,
            "type": "exact",
            "vjp": _guard_none_G_coil_gradient_callback(
                boozer_surface_dexactresidual_dcoils_dcurrents_vjp,
                biotsavart=self.biotsavart,
                component="BoozerSurface",
                coil_attrs=("coils", "_coils"),
                G_provided=G_provided,
            ),
        }

        if verbose:
            print(
                f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, ||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}",
                flush=True,
            )

        self.res = res
        self.need_to_run_code = False
        return res
