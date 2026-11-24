import numpy as np

from jax import grad, vmap
from jax import jit as jaxjit
from jax import numpy as jnp
from scipy.optimize import NonlinearConstraint, minimize

from simsopt._core import Derivative, Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.objectives import forward_backward
from simsopt.util import proc0_print

__all__ = ["OmnigenousField", "OmnigenousResidual", "OmnigenousSlopePenalty"]

class OmnigenousField(Optimizable):
    @staticmethod
    def Dfunc_pure(x, Dmodes): # Dmodes is shape (DN,)
        first_term = jnp.pi - x
        second_term = sum([k*jnp.sin((i+1)*x) for i, k in enumerate(Dmodes)])
        return first_term + second_term

    @staticmethod
    def sfunc_pure(x, y, smodes): # smodes is shape (sL, sN)
        _s_x_basis = lambda x, l: jnp.sin((l+1./2.)*x)
        _s_y_basis = lambda y, n: jnp.sin((n+1)*y)
        out = 0
        sL, sN = smodes.shape
        for l in range(sL):
            for n in range(sN):
                out += smodes[l, n] * _s_x_basis(x, l) * _s_y_basis(y, n)
        return out

    def __init__(self, Bmax, Bmin, nfp, iota, Dparams=None, sparams=None, phi=128, theta=64, neta=181, helicity=(0, 1), dofs=None):
        if not(Bmax > Bmin > 0):
            raise ValueError("Require Bmax > Bmin > 0")

        self._nfp = nfp
        self._iota = iota

        Dparams = np.zeros(3) if (Dparams is None) else np.asarray(Dparams, dtype=float)
        sparams = np.zeros((3, 3)) if (sparams is None) else np.asarray(sparams, dtype=float)
        self._DN = Dparams.size
        self._sL, self._sN = sparams.shape
        self._ndofs = 2 + Dparams.size + sparams.size
        self._Dfunc_jax         = jaxjit(self.Dfunc_pure)
        self._sfunc_jax         = jaxjit(self.sfunc_pure)
        self._dD_by_dx          = jaxjit(vmap(grad(self._Dfunc_jax, argnums=0), in_axes=(0, None)))
        self._dD_by_dDparams    = jaxjit(vmap(grad(self._Dfunc_jax, argnums=1), in_axes=(0, None)))
        self._ds_by_dx          = jaxjit(vmap(grad(self._sfunc_jax, argnums=0), in_axes=(0, 0, None)))
        self._ds_by_dy          = jaxjit(vmap(grad(self._sfunc_jax, argnums=1), in_axes=(0, 0, None)))
        self._ds_by_dsparams    = jaxjit(vmap(grad(self._sfunc_jax, argnums=2), in_axes=(0, 0, None)))
        self._d2s_by_dxdy       = jaxjit(vmap(grad(grad(self._sfunc_jax, argnums=0), argnums=1), in_axes=(0, 0, None)))
        self._d2s_by_dydy       = jaxjit(vmap(grad(grad(self._sfunc_jax, argnums=1), argnums=1), in_axes=(0, 0, None)))
        self._d2s_by_dxdsparams = jaxjit(vmap(grad(grad(self._sfunc_jax, argnums=0), argnums=2), in_axes=(0, 0, None)))
        self._d2s_by_dydsparams = jaxjit(vmap(grad(grad(self._sfunc_jax, argnums=1), argnums=2), in_axes=(0, 0, None)))
        self._d2D_by_dxdDparams = jaxjit(vmap(grad(grad(self._Dfunc_jax, argnums=0), argnums=1), in_axes=(0, None)))

        rng = np.random.default_rng(0)
        Dp = rng.standard_normal(self.DN)
        sp = rng.standard_normal((self.sL, self.sN))
        y = np.linspace(0, 2*np.pi, 17)
        if not (np.isclose(self.Dfunc_pure(0., Dp), np.pi)
                and np.isclose(self.Dfunc_pure(np.pi, Dp), 0.)):
            raise ValueError("Dfunc_pure must satisfy D(0)=pi, D(pi)=0")
        if not (np.allclose(self.sfunc_pure(0., y, sp), 0.)
                and np.allclose(self.sfunc_pure(1., y, sp), -self.sfunc_pure(1., -y, sp))):
            raise ValueError("sfunc_pure must satisfy s(0,y)=0 and be odd in y")

        if dofs is None:
            x0 = np.r_[Bmax, Bmin, Dparams.ravel(), sparams.ravel()]
            Optimizable.__init__(self, x0=x0, names=self.local_full_dof_names)
        else:
            Optimizable.__init__(self, dofs=dofs)

        # Default helicity = (0, 1) for QI fields
        self._helicity = tuple(map(int, helicity))
        M, N = self._helicity
        if not N and abs(M) != 1:
            raise ValueError(f"helicity {self._helicity} is not supported: with N = 0 the angles "
                             "assume M = ±1. A nonzero N gives |N| identical wells per field "
                             "period.")
        self.tilde_mat = np.array([[-M*nfp, N*nfp], [1, 0]]) if N else \
                         np.array([[1, 0], [0, nfp]])

        if isinstance(phi, (int, np.integer)):
            self._nphi = phi
            self._phi = np.linspace(0, 2*np.pi/nfp, phi, endpoint=False)
        else:
            phi = np.asarray(phi)
            self._nphi = phi.size
            self._phi = phi

        if isinstance(theta, (int, np.integer)):
            self._ntheta = theta
            self._theta = np.linspace(0, 2*np.pi, theta, endpoint=False)
        else:
            theta = np.asarray(theta)
            self._ntheta = theta.size
            self._theta = theta

        self._neta = neta

        self.need_to_update = True

    def dD_by_dx(self, x):
        x = np.asarray(x)
        return np.asarray(self._dD_by_dx(x.ravel(), self.Dparams)).reshape(x.shape)
    def dD_by_dDparams(self, x):
        x = np.asarray(x)
        return np.asarray(self._dD_by_dDparams(x.ravel(), self.Dparams)).reshape(*x.shape, -1)
    def ds_by_dx(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.asarray(self._ds_by_dx(x.ravel(), y.ravel(), self.sparams)).reshape(x.shape)
    def ds_by_dy(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.asarray(self._ds_by_dy(x.ravel(), y.ravel(), self.sparams)).reshape(x.shape)
    def ds_by_dsparams(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        return np.asarray(self._ds_by_dsparams(x.ravel(), y.ravel(), self.sparams)).reshape(*x.shape, -1)

    def Dfunc(self, x):
        return self._Dfunc_jax(x, self.Dparams)

    def sfunc(self, x, y):
        return self._sfunc_jax(x, y, self.sparams)

    @property
    def Bmax(self):
        return float(self.local_full_x[0])

    @property
    def Bmin(self):
        return float(self.local_full_x[1])

    @property
    def nfp(self):
        return self._nfp

    @property
    def iota(self):
        return self._iota

    @property
    def DN(self):
        return self._DN

    @property
    def sL(self):
        return self._sL

    @property
    def sN(self):
        return self._sN

    @property
    def Dparams(self):
        return np.asarray(self.local_full_x[2:2+self.DN], dtype=float)

    @property
    def sparams(self):
        return np.asarray(self.local_full_x[2+self.DN:], dtype=float).reshape(
            self.sL, self.sN)

    @property
    def ndofs(self):
        return self._ndofs

    @property
    def helicity(self):
        return self._helicity

    @property
    def epsilon(self):
        return (self.Bmax - self.Bmin) / (2 * self.Bmin)

    @property
    def nphi(self):
        return self._nphi

    @property
    def ntheta(self):
        return self._ntheta

    @property
    def neta(self):
        return self._neta

    @property
    def phi(self):
        return self._phi

    @property
    def theta(self):
        return self._theta

    @property
    def phi_grid(self):
        if self.need_to_update:
            self.update()
        return self._phi_grid

    @property
    def theta_grid(self):
        if self.need_to_update:
            self.update()
        return self._theta_grid

    @property
    def tilde_phi_grid(self):
        if self.need_to_update:
            self.update()
        return self._tilde_phi_grid

    @property
    def tilde_theta_grid(self):
        if self.need_to_update:
            self.update()
        return self._tilde_theta_grid

    @property
    def tilde_iota(self):
        if self.need_to_update:
            self.update()
        return self._tilde_iota

    @property
    def modB(self):
        if self.need_to_update:
            self.update()
        return self._modB

    @property
    def local_full_dof_names(self):
        names = ["Bmax", "Bmin"]
        for n in range(self.DN):
            names.append(f"Ds({n+1})")
        for l in range(self.sL):
            for n in range(self.sN):
                names.append(f"s({l},{n+1})")
        return names

    @property
    def eta_grid(self):
        if self.need_to_update:
            self.update()
        return self._eta_grid

    def set_dofs(self, new_values):
        if np.array_equal(np.array(self.local_full_x, dtype=float), np.array(new_values, dtype=float)):
            return
        self.local_full_x = np.array(new_values, dtype=float)
        self.need_to_update = True

    def set_Bmax(self, Bmax):
        x = np.array(self.local_full_x, dtype=float)
        x[0] = Bmax
        self.local_full_x = x
        self.need_to_update = True

    def set_Bmin(self, Bmin):
        x = np.array(self.local_full_x, dtype=float)
        x[1] = Bmin
        self.local_full_x = x
        self.need_to_update = True

    def set_iota(self, iota):
        if iota == self._iota:
            return
        self._iota = iota
        self.need_to_update = True

    def set_neta(self, neta):
        self._neta = neta
        self.need_to_update = True

    def set_Dparams(self, Dparams):
        x = np.array(self.local_full_x, dtype=float)
        x[2:2+self.DN] = np.asarray(Dparams, dtype=float).ravel()
        self.local_full_x = x
        self.need_to_update = True

    def set_sparams(self, sparams):
        x = np.array(self.local_full_x, dtype=float)
        x[2+self.DN:] = np.asarray(sparams, dtype=float).ravel()
        self.local_full_x = x
        self.need_to_update = True

    def get_dofs(self):
        return np.r_[self.Bmax, self.Bmin, self.Dparams.ravel(), self.sparams.ravel()]

    def get_tilde_coordinates(self, phi, theta):
        (a, b), (c, d) = self.tilde_mat
        return a*theta + b*phi, c*theta + d*phi

    def get_phi_tilde_from_eta(self, eta, tilde_theta):
        # Called by update()
        eta = np.asarray(eta)
        tilde_theta = np.asarray(tilde_theta)

        D1 = self.Dfunc(eta)
        term1 = np.pi - self.sfunc(eta, tilde_theta + self._tilde_iota * D1) - D1

        eta2 = 2*np.pi - eta
        D2 = self.Dfunc(eta2)
        term2 = np.pi + self.sfunc(eta2, -tilde_theta + self._tilde_iota * D2) + D2

        return np.where(eta <= np.pi, term1, term2)

    def get_eta_from_phi_tilde(self, nbisect=50):
        tilde_phi_grid = np.mod(self._tilde_phi_grid, 2*np.pi)
        tilde_theta_grid = self._tilde_theta_grid
        eta_scan = np.linspace(0, 2*np.pi, self.neta)
        residuals = np.asarray(self.get_phi_tilde_from_eta(
            eta_scan[None, None, :], tilde_theta_grid[..., None]
        )) - tilde_phi_grid[..., None]

        brackets = residuals[..., :-1] * residuals[..., 1:] <= 0
        if not brackets.any(axis=-1).all():
            raise RuntimeError("no bracket for eta on part of the grid")

        icell = np.argmin(np.where(brackets,
                                   np.abs(eta_scan[:-1] - tilde_phi_grid[..., None]),
                                   np.inf), axis=-1)
        eta_lo = eta_scan[icell]
        eta_hi = eta_scan[icell + 1]
        r_lo = np.take_along_axis(residuals, icell[..., None], axis=-1)[..., 0]

        for _ in range(nbisect):
            eta_mid = 0.5*(eta_lo + eta_hi)
            resid_mid = np.asarray(self.get_phi_tilde_from_eta(eta_mid, tilde_theta_grid)) - tilde_phi_grid
            keep_lower = (resid_mid * r_lo) > 0
            eta_lo = np.where(keep_lower, eta_mid, eta_lo)
            r_lo   = np.where(keep_lower, resid_mid, r_lo)
            eta_hi = np.where(keep_lower, eta_hi, eta_mid)

        return 0.5*(eta_lo + eta_hi)

    def get_args(self):
        eta_grid = self.eta_grid
        tilde_theta_grid = self.tilde_theta_grid
        cond = eta_grid <= np.pi
        sign = np.where(cond, -1.0, 1.0)
        Darg = np.where(cond, eta_grid, 2*np.pi - eta_grid)
        sargx = Darg.copy()
        sargy = -sign*tilde_theta_grid + self.tilde_iota*self.Dfunc(Darg)
        return sign, Darg, sargx, sargy

    def update_coords(self):
        self._theta_grid, self._phi_grid = np.meshgrid(self.theta, self.phi)
        self._tilde_phi_grid, self._tilde_theta_grid = \
            self.get_tilde_coordinates(self._phi_grid, self._theta_grid)

        M, N = self.helicity
        den = (N - self.iota*M)*self.nfp
        if not N:
            if np.isclose(self.iota, 0.):
                raise ValueError(f"helicity {self.helicity} is degenerate at iota={self.iota}")
            self._tilde_iota = self.nfp / self.iota
        elif np.isclose(den, 0.):
            raise ValueError(f"helicity {self.helicity} is degenerate at iota={self.iota}")
        else:
            self._tilde_iota = self.iota / den

    def update(self):
        self.update_coords()

        self._eta_grid = self.get_eta_from_phi_tilde()

        self._modB = self.Bmin * (
                1 + self.epsilon + self.epsilon * np.cos(self._eta_grid)
            )
        self.need_to_update = False

    def _slope_args(self):
        r"""
        Arguments of D and s on the eta scan at each tilde theta of the grid, flattened.
        """
        self.update_coords()
        eta = np.linspace(0, 2*np.pi, self.neta)[None, :]
        tilde_theta = np.unique(self._tilde_theta_grid)[:, None]
        sign = np.where(eta <= np.pi, 1.0, -1.0)
        Darg = np.where(eta <= np.pi, eta, 2*np.pi - eta)
        sargy = sign*tilde_theta + self._tilde_iota*self.Dfunc(Darg)
        Darg, sargy = np.broadcast_arrays(Darg, sargy)
        return Darg.ravel(), sargy.ravel()

    def dzeta_by_deta(self):
        r"""
        Slope of tilde phi with respect to eta, on an eta scan at each tilde theta of the grid.
        The eta-to-tilde-phi map is invertible only while every slope remains positive.
        """
        x, y = self._slope_args()
        slope = -(self.ds_by_dx(x, y) + (1 + self._tilde_iota*self.ds_by_dy(x, y))*self.dD_by_dx(x))
        return slope

    def dslope_by_dparams(self):
        r"""
        Derivative of `dzeta_by_deta` with respect to the dofs (Bmax, Bmin, Dparams, sparams),
        shaped (number of slopes, number of dofs). Bmax and Bmin do not enter the slopes.
        """
        x, y = self._slope_args()
        tilde_iota = self._tilde_iota
        D_x = self.dD_by_dx(x)
        D_p = self.dD_by_dDparams(x)
        D_xp = np.asarray(self._d2D_by_dxdDparams(x, self.Dparams))
        s_y = self.ds_by_dy(x, y)
        s_xy = np.asarray(self._d2s_by_dxdy(x, y, self.sparams))
        s_yy = np.asarray(self._d2s_by_dydy(x, y, self.sparams))
        s_xp = np.asarray(self._d2s_by_dxdsparams(x, y, self.sparams)).reshape(x.size, -1)
        s_yp = np.asarray(self._d2s_by_dydsparams(x, y, self.sparams)).reshape(x.size, -1)
        dslope = np.zeros((x.size, self.ndofs))
        dslope[:, 2:2+self.DN] = -(
            tilde_iota*(s_xy + tilde_iota*s_yy*D_x)[:, None]*D_p
            + (1 + tilde_iota*s_y)[:, None]*D_xp)
        dslope[:, 2+self.DN:] = -(s_xp + tilde_iota*D_x[:, None]*s_yp)
        return dslope

    def dslope_by_diota(self):
        r"""
        Derivative of `dzeta_by_deta` with respect to iota.
        """
        x, y = self._slope_args()
        tilde_iota = self._tilde_iota
        D = np.asarray(self.Dfunc(x))
        s_y = self.ds_by_dy(x, y)
        s_xy = np.asarray(self._d2s_by_dxdy(x, y, self.sparams))
        s_yy = np.asarray(self._d2s_by_dydy(x, y, self.sparams))
        return (-(s_xy*D + (s_y + tilde_iota*s_yy*D)*self.dD_by_dx(x))
                * self.dtilde_iota_by_diota())

    def dmodB_by_dparams(self):
        eta_grid = self.eta_grid
        dBdp = np.zeros((self.nphi, self.ntheta, self.ndofs))

        deps_by_dBmax = 1./(2*self.Bmin)
        dmodB_by_dBmax = \
            self.Bmin * deps_by_dBmax * (1 + np.cos(eta_grid))
        dBdp[..., 0] = dmodB_by_dBmax

        deps_by_dBmin = -self.Bmax / (2 * self.Bmin**2)
        dmodB_by_dBmin = \
            (1 + self.epsilon * (1 + np.cos(eta_grid))) + \
            self.Bmin * deps_by_dBmin * (1 + np.cos(eta_grid))
        dBdp[..., 1] = dmodB_by_dBmin

        sign, Darg, sargx, sargy = self.get_args()

        s_x = self.ds_by_dx(sargx, sargy)
        s_y = self.ds_by_dy(sargx, sargy)
        a = 1 + s_y * self.tilde_iota
        den = s_x + a * self.dD_by_dx(Darg)
        deta_by_dDparams = sign[..., None]*a[..., None]*self.dD_by_dDparams(Darg)/den[..., None]
        deta_by_dsparams = sign[..., None]*self.ds_by_dsparams(sargx, sargy)/den[..., None]
        dBdp[..., 2:2+self.DN] = \
            -self.Bmin * self.epsilon * np.sin(eta_grid[..., None]) * deta_by_dDparams
        dBdp[..., 2+self.DN:] = \
            -self.Bmin * self.epsilon * np.sin(eta_grid[..., None]) * deta_by_dsparams

        return dBdp

    def dtilde_iota_by_diota(self):
        r"""
        Derivative of tilde iota with respect to iota. Necessary for optimizing Boozer surfaces
        where iota changes over the optimization.
        """
        M, N = self.helicity
        return -self.nfp/self.iota**2 if not N else N/((N - self.iota*M)**2 * self.nfp)

    def dmodB_by_diota(self):
        eta_grid = self.eta_grid
        sign, Darg, _, sargy = self.get_args()
        D    = self.Dfunc(Darg)
        s_y  = self.ds_by_dy(Darg, sargy)
        den  = self.ds_by_dx(Darg, sargy) + (1 + s_y*self.tilde_iota)*self.dD_by_dx(Darg)
        return -(self.Bmin*self.epsilon*np.sin(eta_grid) * (sign*s_y*D/den)
                 * self.dtilde_iota_by_diota())

    def fit_to(self, coil_modB, **kwargs):
        self.set_Bmax(float(coil_modB.max()))
        self.set_Bmin(float(coil_modB.min()))
        rms_modB = np.sqrt(np.mean(coil_modB**2))

        cache = dict(x=None, r=None, J=None)
        tracker = dict(nfev=-1, njac=-1)

        def residual(x):
            if (cache["x"] is None) or (not np.array_equal(cache["x"], x)):
                self.set_dofs(x)
                tracker["nfev"] += 1
                cache["x"] = np.asarray(x).copy() # pyright: ignore[reportArgumentType]
                cache["r"] = (coil_modB - self.modB).ravel() / rms_modB
                cache["J"] = None
                proc0_print(
                    f"nfev={tracker['nfev']}, "
                    f"avg(r)={np.mean(cache['r']**2):.6e}")
            return cache["r"]

        def jac(x):
            r = residual(x)
            if cache["J"] is None:
                tracker["njac"] += 1
                self.set_dofs(x)
                cache["J"] = -self.dmodB_by_dparams().reshape(
                    -1, np.size(x)
                ) / rms_modB
            return cache["J"].T @ r

        def fun(x):
            r = residual(x)
            return 0.5 * float(r @ r) # pyright: ignore[reportOptionalOperand]

        def slopes(x):
            self.set_dofs(x)
            return self.dzeta_by_deta()

        def slopes_jac(x):
            self.set_dofs(x)
            return self.dslope_by_dparams()

        x0 = self.get_dofs()
        slope_lb = kwargs.pop("slope_lb", 1e-2)
        if slopes(x0).min() <= slope_lb:
            is_ok = lambda a: slopes(np.r_[x0[:2], a*x0[2:]]).min() > slope_lb
            lo, hi = 0.0, 1.0
            for _ in range(40):
                m = 0.5 * (lo + hi)
                lo, hi = (m, hi) if is_ok(m) else (lo, m)
            x0 = np.r_[x0[:2], lo * x0[2:]]

        if "maxiter" not in kwargs: kwargs["maxiter"] = 300
        if "ftol" not in kwargs: kwargs["ftol"] = 1e-12
        slopes_constraint = NonlinearConstraint(slopes, slope_lb, np.inf, jac=slopes_jac)
        well_constraint = NonlinearConstraint(lambda x: np.array([x[0] - x[1], x[1]]), 1e-12, np.inf)
        bounds = [
            (1e-12, None), # Bmax
            (1e-12, None), # Bmin
        ] + [(None, None)]*(x0.size-2)
        res = minimize( # pyright: ignore[reportCallIssue]
            fun,
            x0,
            jac=jac,
            method="SLSQP",
            constraints=[slopes_constraint, well_constraint],
            bounds=bounds,
            options=kwargs, # pyright: ignore[reportArgumentType]
        )
        self.fit_res = res
        self.set_dofs(res.x)
        proc0_print(
            f"fit_to: nfev={tracker['nfev']}, "
            f"min_slope={slopes(res.x).min():.4e} success={res.success}"
        )
        proc0_print(res.message)
        return res

    def recompute_bell(self, parent=None):
        self.need_to_update = True

class OmnigenousResidual(Optimizable):
    def __init__(self, boozersurface, target):
        s = boozersurface.surface
        if not (np.allclose(target.phi, s.quadpoints_phi*2*np.pi)
                and np.allclose(target.theta, s.quadpoints_theta*2*np.pi)):
            raise ValueError("target grid must match the surface quadpoints")

        Optimizable.__init__(self, depends_on=[boozersurface, target])
        self.boozersurface = boozersurface
        self.target = target

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute_objective_and_derivatives()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute_objective_and_derivatives()
        return self._dJ

    def fit_target_field(self, **kwargs):
        if self.boozersurface.need_to_run_code:
            raise RuntimeError("Need to run boozer surface first")
        self.boozersurface.biotsavart.set_points(
            self.boozersurface.surface.gamma().reshape(-1, 3))
        modB = np.linalg.norm(self.boozersurface.biotsavart.B().reshape(
            self.boozersurface.surface.gamma().shape), axis=-1)
        return self.target.fit_to(modB, **kwargs)

    def compute_objective_and_derivatives(self):
        if self.boozersurface.need_to_run_code:
            raise RuntimeError("Need to run boozer surface first")
        iota = self.boozersurface.res["iota"]
        self.target.set_iota(iota)

        surf_gamma = self.boozersurface.surface.gamma()
        surf_shape = surf_gamma.shape
        self.boozersurface.biotsavart.set_points(surf_gamma.reshape(-1, 3))
        B = self.boozersurface.biotsavart.B().reshape(surf_shape)
        modB = np.linalg.norm(B, axis=-1)

        num = np.sum((modB - self.target.modB)**2)
        den = np.sum(modB**2)
        J = np.sqrt(num / den)
        self._J = J

        dnum_by_dmodB = 2*(modB - self.target.modB)
        dden_by_dmodB = 2*modB
        dJ_by_dmodB = \
            1./(2*J) * ( (dnum_by_dmodB * den - num * dden_by_dmodB) / den**2 )
        dJ_by_dselfB = -1./J * (modB - self.target.modB) / den
        dmodB_by_dB = B / modB[..., None]
        dJ_by_dB = dJ_by_dmodB[..., None] * dmodB_by_dB

        dB_by_dX = \
            self.boozersurface.biotsavart.dB_by_dX().reshape(*surf_shape, 3)
        dgamma_by_dsurf = \
            self.boozersurface.surface.dgamma_by_dcoeff()
        dB_by_dsurf = np.einsum(
            "ijkl,ijkm->ijlm", dB_by_dX, dgamma_by_dsurf, optimize=True)

        dmodB_by_dsurf = (
            B[..., 0, None] * dB_by_dsurf[..., 0, :] + \
            B[..., 1, None] * dB_by_dsurf[..., 1, :] + \
            B[..., 2, None] * dB_by_dsurf[..., 2, :]
        ) / modB[..., None]

        dj_by_dsurf = np.einsum("ij,ijs->s", dJ_by_dmodB, dmodB_by_dsurf)

        G = self.boozersurface.res["G"]
        P, L, U = self.boozersurface.res["PLU"]
        vjp = self.boozersurface.res["vjp"]

        dJ_by_dsurf = np.zeros(L.shape[0], dtype=float)
        dJ_by_dsurf[:dj_by_dsurf.size] = dj_by_dsurf
        dJ_by_dsurf[dj_by_dsurf.size] = np.sum(dJ_by_dselfB * self.target.dmodB_by_diota())
        adj = forward_backward(P, L, U, dJ_by_dsurf)
        adj_times_dg_dcoil = vjp(adj, self.boozersurface, iota, G)

        dJ_by_coils = \
            self.boozersurface.biotsavart.B_vjp(dJ_by_dB.reshape(-1, 3)) - \
            adj_times_dg_dcoil
        dJ_by_local_x = Derivative({
            self.target: np.einsum("ij,ijl->l", dJ_by_dselfB, self.target.dmodB_by_dparams())
        }) # pyright: ignore[reportArgumentType]

        self._dJ = dJ_by_coils + dJ_by_local_x

class OmnigenousSlopePenalty(Optimizable):
    r"""
    Penalty that keeps the eta-to-tilde-phi map of an omnigenous target invertible,

    .. math::
        J = \frac{1}{N} \sum_k \max(0, \sigma_{\min} - \sigma_k)^2,

    where :math:`\sigma_k` are the slopes from `OmnigenousField.dzeta_by_deta`. The target's
    iota is the Boozer surface's, so the gradient reaches the coils through the Boozer solve.

    Args:
        boozersurface: Boozer surface that sets the target's iota.
        target: OmnigenousField whose map is kept invertible.
        slope_lb: Smallest allowed slope.
    """

    def __init__(self, boozersurface, target, slope_lb=0.1):
        Optimizable.__init__(self, depends_on=[boozersurface, target])
        self.boozersurface = boozersurface
        self.target = target
        self.slope_lb = slope_lb

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute_objective_and_derivatives()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute_objective_and_derivatives()
        return self._dJ

    def compute_objective_and_derivatives(self):
        if self.boozersurface.need_to_run_code:
            raise RuntimeError("Need to run boozer surface first")
        iota = self.boozersurface.res["iota"]
        G = self.boozersurface.res["G"]
        self.target.set_iota(iota)

        shortfall = np.maximum(0., self.slope_lb - self.target.dzeta_by_deta())
        self._J = float(np.mean(shortfall**2))
        if not shortfall.any():
            self._dJ = Derivative()
            return

        dJ_by_dslope = -2*shortfall/shortfall.size
        P, L, U = self.boozersurface.res["PLU"]
        dJ_by_dsurf = np.zeros(L.shape[0])
        dJ_by_dsurf[-2 if G is not None else -1] = dJ_by_dslope @ self.target.dslope_by_diota()
        adj = forward_backward(P, L, U, dJ_by_dsurf)
        adj_times_dg_dcoil = self.boozersurface.res["vjp"](adj, self.boozersurface, iota, G)

        self._dJ = (
            Derivative({self.target: dJ_by_dslope @ self.target.dslope_by_dparams()})
            - adj_times_dg_dcoil) # pyright: ignore[reportOperatorIssue]
