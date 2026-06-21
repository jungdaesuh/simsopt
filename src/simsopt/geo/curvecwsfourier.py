import numpy as np
from jax import vjp, jacfwd, jvp
import jax.numpy as jnp

import simsoptpp as sopp
from .._core.derivative import Derivative
from .curve import Curve

from .jit import jit

__all__ = ['CurveCWSFourierCPP']

def gamma_2d(cdofs, qpts, order, G:int=0, H:int=0):
    """Given some dofs, return curve position in 2D cartesian coordinate
    
    Args:
     - cdofs: Input dofs. Array of size 2*(2*order+1)
     - qpts: quadrature points. Array of floats from 0 to 1, of size N.
     - order: Maximum Fourier series order.

    Returns:
     - phi: Array of size N x 1.
     - theta: Array of size N x 1.
    """
    # Unpack dofs
    phic = cdofs[:order+1]
    phis = cdofs[order+1:2*order+1]
    thetac   = cdofs[2*order+1:3*order+2]
    thetas   = cdofs[3*order+2:]

    # Construct theta and phi arrays
    theta = jnp.zeros((qpts.size,))
    phi = jnp.zeros((qpts.size,))

    ll = qpts*2.0*jnp.pi
    for ii in range(order+1):
        theta = theta + thetac[ii] * jnp.cos(ii*ll)
        phi   = phi   + phic[ii]   * jnp.cos(ii*ll)

    for ii in range(order):
        theta = theta + thetas[ii] * jnp.sin((ii+1)*ll)
        phi   = phi   + phis[ii]   * jnp.sin((ii+1)*ll)

    # Add secular terms
    theta = theta + G * qpts
    phi = phi + H * qpts

    # Prepare output
    out = jnp.zeros((qpts.size, 2))
    out = out.at[:,0].set(phi)
    out = out.at[:,1].set(theta)

    return out

def vjp_contraction_1d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum('ij,i->j',mat,v)

def vjp_contraction_2d(mat, v):
    # contract matrix of size ijk times vector of size jk into array of size i
    return np.einsum('ijk,ij->k',mat,v)


def surfrz_gamma_lin_jax(quadpoints_phi, quadpoints_theta, mpol, ntor, surf_dofs, nfp, stellsym):
    """Evaluate ``SurfaceRZFourier.gamma_lin`` at paired normalized angles."""
    npts = quadpoints_phi.size
    theta = quadpoints_theta * 2.0 * jnp.pi
    phi = quadpoints_phi * 2.0 * jnp.pi
    num_cos_modes = ntor + 1 + mpol * (2 * ntor + 1)
    num_sin_modes = num_cos_modes - 1
    rs_offset = num_cos_modes
    zc_offset = num_cos_modes + num_sin_modes
    zs_offset = num_cos_modes if stellsym else zc_offset + num_cos_modes

    r = jnp.zeros((npts,))
    z = jnp.zeros((npts,))
    cos_counter = 0
    sin_counter = 0
    for mm in range(mpol + 1):
        for nn in range(-ntor, ntor + 1):
            if mm == 0 and nn < 0:
                continue
            angle = mm * theta - nn * nfp * phi
            sin_angle = jnp.sin(angle)
            cos_angle = jnp.cos(angle)
            r = r + surf_dofs[cos_counter] * cos_angle
            if not stellsym:
                z = z + surf_dofs[zc_offset + cos_counter] * cos_angle
            if not (mm == 0 and nn <= 0):
                if not stellsym:
                    r = r + surf_dofs[rs_offset + sin_counter] * sin_angle
                z = z + surf_dofs[zs_offset + sin_counter] * sin_angle
                sin_counter += 1
            cos_counter += 1

    gamma = jnp.zeros((npts, 3))
    gamma = gamma.at[:, 0].set(r * jnp.cos(phi))
    gamma = gamma.at[:, 1].set(r * jnp.sin(phi))
    gamma = gamma.at[:, 2].set(z)
    return gamma


def gamma_curve_on_surfrz_surface(curve_dofs, qpts, order, G, H, surf_dofs, mpol, ntor, nfp, stellsym):
    gamma2d = gamma_2d(curve_dofs, qpts, order, G, H)
    return surfrz_gamma_lin_jax(
        gamma2d[:, 0], gamma2d[:, 1], mpol, ntor, surf_dofs, nfp, stellsym
    )


def unit_normal_curve_on_surfrz_surface(curve_dofs, qpts, order, G, H, surf_dofs, mpol, ntor, nfp, stellsym):
    gamma2d = gamma_2d(curve_dofs, qpts, order, G, H)
    phi = gamma2d[:, 0]
    theta = gamma2d[:, 1]
    dxdphi = jvp(
        lambda p: surfrz_gamma_lin_jax(p, theta, mpol, ntor, surf_dofs, nfp, stellsym),
        (phi,),
        (jnp.ones_like(phi),),
    )[1]
    dxdtheta = jvp(
        lambda t: surfrz_gamma_lin_jax(phi, t, mpol, ntor, surf_dofs, nfp, stellsym),
        (theta,),
        (jnp.ones_like(theta),),
    )[1]
    normal = jnp.cross(dxdphi, dxdtheta)
    return normal / jnp.linalg.norm(normal, axis=1)[:, None]


def zfactor_curve_on_surfrz_surface(curve_dofs, qpts, order, G, H, surf_dofs, mpol, ntor, nfp, stellsym, sgn_z):
    return sgn_z * unit_normal_curve_on_surfrz_surface(
        curve_dofs, qpts, order, G, H, surf_dofs, mpol, ntor, nfp, stellsym
    )[:, 2]


def rfactor_curve_on_surfrz_surface(curve_dofs, qpts, order, G, H, surf_dofs, mpol, ntor, nfp, stellsym, sgn_r):
    gamma2d = gamma_2d(curve_dofs, qpts, order, G, H)
    unit_normal = unit_normal_curve_on_surfrz_surface(
        curve_dofs, qpts, order, G, H, surf_dofs, mpol, ntor, nfp, stellsym
    )
    return sgn_r * (
        unit_normal[:, 0] * jnp.cos(gamma2d[:, 0])
        + unit_normal[:, 1] * jnp.sin(gamma2d[:, 0])
    )


class CurveCWSFourierCPP( Curve, sopp.Curve ):
    def __init__(self, quadpoints, order, surf, G=0, H=0, **kwargs):
        # Curve order. Number of Fourier harmonics for phi and theta
        self.order = order
        self.G = G
        self.H = H
        
        # Modes are order as phic, phis, thetac, thetas
        self.modes = [np.zeros((order+1,)), np.zeros((order,)), np.zeros((order+1,)), np.zeros((order,))]

        #self.quadpoints = quadpoints
        self.surf = surf

        # Initialize C++ class and Curve class
        sopp.Curve.__init__(self, quadpoints)
        Curve.__init__(self, x0=self.get_dofs(), depends_on=[surf], names=self._make_names(), external_dof_setter=CurveCWSFourierCPP.set_dofs_impl, **kwargs)

        self.numquadpoints = self.quadpoints.size

        points = jnp.asarray(self.quadpoints)

        ## gamma
        self.gamma_2d_pure = jit(lambda cdofs, qpts: gamma_2d(cdofs, qpts, self.order, self.G, self.H))
        self.gamma_2d_jax = jit(lambda cdofs: self.gamma_2d_pure(cdofs, self.quadpoints))
        self.dgamma_2d_by_dcoeff_jax = jit(lambda cdofs: jacfwd(self.gamma_2d_jax)(cdofs))
        self.dgamma_2d_by_dcoeff_vjp = jit(lambda cdofs, v: vjp(self.gamma_2d_jax, cdofs)[1](v)[0])

        self.gamma_pure = jit(
            lambda cdofs, sdofs, qpts: gamma_curve_on_surfrz_surface(
                cdofs,
                qpts,
                self.order,
                self.G,
                self.H,
                sdofs,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
            )
        )
        self.gamma_jax = jit(lambda cdofs, sdofs: self.gamma_pure(cdofs, sdofs, points))
        self.gamma_impl_jax = jit(lambda cdofs, sdofs, qpts: self.gamma_pure(cdofs, sdofs, qpts))
        self.dgamma_by_dcoeff_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.gamma_jax(dofs, sdofs))(cdofs)
        )
        self.dgamma_by_dsurf_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.gamma_jax(cdofs, dofs))(sdofs)
        )
        self.dgamma_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.gamma_jax(dofs, sdofs), cdofs)[1](v)[0]
        )
        self.dgamma_by_dsurf_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.gamma_jax(cdofs, dofs), sdofs)[1](v)[0]
        )

        ## gammadash
        self.gammadash_2d_pure = jit(lambda cdofs, q: jvp(lambda qpts: self.gamma_2d_pure(cdofs, qpts), (q,), (jnp.ones_like(q),))[1])
        self.gammadash_2d_jax = jit(lambda cdofs: self.gammadash_2d_pure(cdofs, self.quadpoints))
        self.dgammadash_2d_by_dcoeff_jax = jit(lambda cdofs: jacfwd(self.gammadash_2d_jax)(cdofs))
        self.dgammadash_2d_by_dcoeff_vjp = jit(lambda cdofs, v: vjp(self.gammadash_2d_jax, cdofs)[1](v)[0])

        self.gammadash_pure = jit(
            lambda cdofs, sdofs, qpts: jvp(
                lambda points: self.gamma_pure(cdofs, sdofs, points),
                (qpts,),
                (jnp.ones_like(qpts),),
            )[1]
        )
        self.gammadash_jax = jit(lambda cdofs, sdofs: self.gammadash_pure(cdofs, sdofs, points))
        self.dgammadash_by_dcoeff_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.gammadash_jax(dofs, sdofs))(cdofs)
        )
        self.dgammadash_by_dsurf_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.gammadash_jax(cdofs, dofs))(sdofs)
        )
        self.dgammadash_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.gammadash_jax(dofs, sdofs), cdofs)[1](v)[0]
        )
        self.dgammadash_by_dsurf_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.gammadash_jax(cdofs, dofs), sdofs)[1](v)[0]
        )

        ## gammadashdash
        self.gammadashdash_2d_pure = jit(lambda cdofs, q: jvp(lambda qpts: self.gammadash_2d_pure(cdofs, qpts), (q,), (jnp.ones_like(q),))[1])
        self.gammadashdash_2d_jax = jit(lambda cdofs: self.gammadashdash_2d_pure(cdofs, self.quadpoints))
        self.dgammadashdash_2d_by_dcoeff_jax = jit(lambda cdofs: jacfwd(self.gammadashdash_2d_jax)(cdofs))
        self.dgammadashdash_2d_by_dcoeff_vjp = jit(lambda cdofs, v: vjp(self.gammadashdash_2d_jax, cdofs)[1](v)[0])
        self.gammadashdash_pure = jit(
            lambda cdofs, sdofs, qpts: jvp(
                lambda points: self.gammadash_pure(cdofs, sdofs, points),
                (qpts,),
                (jnp.ones_like(qpts),),
            )[1]
        )
        self.gammadashdash_jax = jit(lambda cdofs, sdofs: self.gammadashdash_pure(cdofs, sdofs, points))
        self.dgammadashdash_by_dcoeff_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.gammadashdash_jax(dofs, sdofs))(cdofs)
        )
        self.dgammadashdash_by_dsurf_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.gammadashdash_jax(cdofs, dofs))(sdofs)
        )
        self.dgammadashdash_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.gammadashdash_jax(dofs, sdofs), cdofs)[1](v)[0]
        )
        self.dgammadashdash_by_dsurf_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.gammadashdash_jax(cdofs, dofs), sdofs)[1](v)[0]
        )

        ## gammadashdashdash
        # Third derivative w.r.t. the curve parameter, following the same
        # nested-jvp pattern as gammadash/gammadashdash. Needed to evaluate the
        # curvature of finite-build filaments offset from this centerline: a
        # filament's second derivative depends on the centerline's third (via the
        # rotated frame's second derivative). Forward-evaluation only -- no
        # optimizer path differentiates it with respect to the dofs.
        self.gammadashdashdash_pure = jit(
            lambda cdofs, sdofs, qpts: jvp(
                lambda pts: self.gammadashdash_pure(cdofs, sdofs, pts),
                (qpts,),
                (jnp.ones_like(qpts),),
            )[1]
        )
        self.gammadashdashdash_jax = jit(lambda cdofs, sdofs: self.gammadashdashdash_pure(cdofs, sdofs, points))


        # determine sign for normal
        nr = self.unit_normal_impl(np.array([0]), np.array([0])) #theta=phi=0
        if nr[0,0]>0:
            self.sgn_r = 1
            nz = self.unit_normal_impl(np.array([0]), np.array([0.25])) #this is on top of the device
            if nz[0,2]>0:
                self.sgn_z = 1
            else:
                self.sgn_z = -1
        else:
            self.sgn_r = -1
            nz = self.unit_normal_impl(np.array([0]), np.array([-0.25])) #this is on top of the device
            if nz[0,2]>0:
                self.sgn_z = 1
            else:
                self.sgn_z = -1

        self.zfactor_pure = jit(
            lambda cdofs, sdofs, qpts: zfactor_curve_on_surfrz_surface(
                cdofs,
                qpts,
                self.order,
                self.G,
                self.H,
                sdofs,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
                self.sgn_z,
            )
        )
        self.zfactor_jax = jit(lambda cdofs, sdofs: self.zfactor_pure(cdofs, sdofs, points))
        self.dzfactor_by_dcoeff_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.zfactor_jax(dofs, sdofs))(cdofs)
        )
        self.dzfactor_by_dsurf_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.zfactor_jax(cdofs, dofs))(sdofs)
        )
        self.dzfactor_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.zfactor_jax(dofs, sdofs), cdofs)[1](v)[0]
        )
        self.dzfactor_by_dsurf_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.zfactor_jax(cdofs, dofs), sdofs)[1](v)[0]
        )

        self.rfactor_pure = jit(
            lambda cdofs, sdofs, qpts: rfactor_curve_on_surfrz_surface(
                cdofs,
                qpts,
                self.order,
                self.G,
                self.H,
                sdofs,
                self.surf.mpol,
                self.surf.ntor,
                self.surf.nfp,
                self.surf.stellsym,
                self.sgn_r,
            )
        )
        self.rfactor_jax = jit(lambda cdofs, sdofs: self.rfactor_pure(cdofs, sdofs, points))
        self.drfactor_by_dcoeff_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.rfactor_jax(dofs, sdofs))(cdofs)
        )
        self.drfactor_by_dsurf_jax = jit(
            lambda cdofs, sdofs: jacfwd(lambda dofs: self.rfactor_jax(cdofs, dofs))(sdofs)
        )
        self.drfactor_by_dcoeff_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.rfactor_jax(dofs, sdofs), cdofs)[1](v)[0]
        )
        self.drfactor_by_dsurf_vjp_jax = jit(
            lambda cdofs, sdofs, v: vjp(lambda dofs: self.rfactor_jax(cdofs, dofs), sdofs)[1](v)[0]
        )


    def set_dofs(self, dofs):
        self.local_x = dofs
        sopp.Curve.set_dofs(self, dofs)

    def num_dofs(self):
        return 2*(self.order+1) + 2*self.order
    
    def get_dofs(self):
        return np.concatenate(self.modes)

    def set_dofs_impl(self, dofs):
        self.modes[0] = dofs[0:self.order+1]
        self.modes[1] = dofs[self.order+1:2*self.order+1]
        self.modes[2] = dofs[2*self.order+1:3*self.order+2]
        self.modes[3] = dofs[3*self.order+2:4*self.order+2]

    def _make_names(self):
        dofs_name = []
        for mode in ['phic', 'phis', 'thetac', 'thetas']:
            for ii in range(self.order+1):
                if mode=='phis' and ii==0:
                    continue

                if mode=='thetas' and ii==0:
                    continue

                dofs_name.append(f'{mode}({ii})')

        return dofs_name

    # =========================================================================
    # GAMMA
    # -----
    def gamma_2d(self):
        cdofs = self.get_dofs()
        return self.gamma_2d_jax(cdofs)
    
    def gamma_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:,:] =  self.gamma_2d_pure(cdofs, quadpoints)
    
    def gamma(self):
        return np.asarray(self.gamma_jax(self.get_dofs(), self.surf.get_dofs()))
    
    def gamma_impl(self, gamma, quadpoints):
        gamma[:, :] = np.asarray(
            self.gamma_impl_jax(self.get_dofs(), self.surf.get_dofs(), quadpoints)
        )

    def dgamma_2d_by_dcoeff(self):
        cdofs = self.get_dofs()
        return self.dgamma_2d_by_dcoeff_jax(cdofs)

    def dgamma_by_dcoeff(self):
        return np.asarray(self.dgamma_by_dcoeff_jax(self.get_dofs(), self.surf.get_dofs()))

    def dgamma_by_dsurf(self):
        return np.asarray(self.dgamma_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs()))

    def dgamma_by_dcoeff_impl(self, v):
        v[:,:,:] = self.dgamma_by_dcoeff()

    def dgamma_by_dcoeff_vjp(self, v):
        return Derivative({
            self: self.dgamma_by_dcoeff_vjp_impl(v),
            self.surf: self.dgamma_by_dsurf_vjp_impl(v),
        })
        
    def dgamma_by_dcoeff_vjp_impl(self, v):
        return np.asarray(
            self.dgamma_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    def dgamma_by_dsurf_vjp_impl(self, v):
        return np.asarray(
            self.dgamma_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    #=========================================================================
    # GAMMADASH
    # ---------
    def gammadash_2d(self):
        cdofs = self.get_dofs()
        return self.gammadash_2d_jax(cdofs)
    
    def gammadash_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:,:] =  self.gammadash_2d_pure(cdofs, quadpoints)

    def dgammadash_2d_by_dcoeff(self):
        cdofs = self.get_dofs()
        return self.dgammadash_2d_by_dcoeff_jax(cdofs)
    
    def gammadash(self):
        return np.asarray(self.gammadash_jax(self.get_dofs(), self.surf.get_dofs()))
    
    def gammadash_impl(self, gammadash):
        gammadash[:,:] = self.gammadash()

    def dgammadash_by_dcoeff(self):# dgammadash by dcoeff
        return np.asarray(self.dgammadash_by_dcoeff_jax(self.get_dofs(), self.surf.get_dofs()))

    def dgammadash_by_dsurf(self):
        return np.asarray(self.dgammadash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs()))

    def dgammadash_by_dcoeff_impl(self, v):
        v[:,:,:] = self.dgammadash_by_dcoeff()

    def dgammadash_by_dcoeff_vjp(self, v):
        return Derivative({
            self: self.dgammadash_by_dcoeff_vjp_impl(v),
            self.surf: self.dgammadash_by_dsurf_vjp_impl(v),
        })
        
    def dgammadash_by_dcoeff_vjp_impl(self, v):
        return np.asarray(
            self.dgammadash_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    def dgammadash_by_dsurf_vjp_impl(self, v):
        return np.asarray(
            self.dgammadash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

 
    #=========================================================================
    # GAMMADASHDASH
    # -------------
    def gammadashdash_2d(self):
        cdofs = self.get_dofs()
        return self.gammadashdash_2d_jax(cdofs)
    
    def gammadashdash_2d_impl(self, g2, quadpoints):
        cdofs = self.get_dofs()
        g2[:,:] =  self.gammadashdash_2d_pure(cdofs, quadpoints)
    
    def gammadashdash(self):
        return np.asarray(self.gammadashdash_jax(self.get_dofs(), self.surf.get_dofs()))
        
    
    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:,:] = self.gammadashdash()

    def dgammadashdash_by_dcoeff(self):
        return np.asarray(self.dgammadashdash_by_dcoeff_jax(self.get_dofs(), self.surf.get_dofs()))

    def dgammadashdash_by_dsurf(self):
        return np.asarray(self.dgammadashdash_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs()))

    def dgammadashdash_by_dcoeff_impl(self, v):
        v[:,:,:] = self.dgammadashdash_by_dcoeff()

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return Derivative({
            self: self.dgammadashdash_by_dcoeff_vjp_impl(v),
            self.surf: self.dgammadashdash_by_dsurf_vjp_impl(v),
        })
        
    def dgammadashdash_by_dcoeff_vjp_impl(self, v):
        return np.asarray(
            self.dgammadashdash_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    def dgammadashdash_by_dsurf_vjp_impl(self, v):
        return np.asarray(
            self.dgammadashdash_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    #=========================================================================
    # GAMMADASHDASHDASH
    # -----------------
    def gammadashdashdash(self):
        return np.asarray(self.gammadashdashdash_jax(self.get_dofs(), self.surf.get_dofs()))

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:, :] = self.gammadashdashdash()

    #=========================================================================
    # NORMAL
    # ------
    def unit_normal(self):
        g2 = self.gamma_2d()
        return self.unit_normal_impl(g2[:,0], g2[:,1])
    
    def unit_normal_impl(self, phi, theta):
        npts = phi.size
        dxdtheta = np.zeros((npts,3))
        dxdphi = np.zeros((npts,3))
        self.surf.gammadash1_lin(dxdphi,  phi, theta)
        self.surf.gammadash2_lin(dxdtheta,  phi, theta)

        normal = np.cross(dxdphi, dxdtheta)
        unit_normal = normal / np.linalg.norm(normal, axis=1 )[:,None]
        return unit_normal


    def dunit_normal_by_dcoeff(self):
        g2 = self.gamma_2d()
        dxdtheta = np.zeros((self.numquadpoints,3))
        dxdphi = np.zeros((self.numquadpoints,3))
        self.surf.gammadash1_lin(dxdphi, g2[:,0], g2[:,1])
        self.surf.gammadash2_lin(dxdtheta, g2[:,0], g2[:,1])

        normal = np.cross(dxdphi, dxdtheta)
        normal_norm = np.linalg.norm( normal, axis=1 )

        dg2_by_dcoeff = self.dgamma_2d_by_dcoeff()
        dxdthetadtheta = np.zeros((self.numquadpoints,3))
        dxdphidtheta = np.zeros((self.numquadpoints,3))
        dxdphidphi = np.zeros((self.numquadpoints,3))
        self.surf.gammadash1dash1_lin(dxdphidphi, g2[:,0], g2[:,1])
        self.surf.gammadash1dash2_lin(dxdphidtheta, g2[:,0], g2[:,1])
        self.surf.gammadash2dash2_lin(dxdthetadtheta, g2[:,0], g2[:,1])

        p0 = np.cross(dxdphi, dxdphidtheta) + np.cross(dxdphidphi, dxdtheta)
        p1 = np.cross(dxdphi, dxdthetadtheta) + np.cross(dxdphidtheta, dxdtheta)
        dnormal_by_dcoeff = np.einsum('ik,ij->ijk', dg2_by_dcoeff[:,0,:], p0) \
                          + np.einsum('ik,ij->ijk', dg2_by_dcoeff[:,1,:], p1)


        t1 = np.einsum('ijk,i->ijk', dnormal_by_dcoeff, 1./normal_norm) # this has shape (nqpts,3,ndofs)
        prod = np.einsum('ij,ijk->ik', normal, dnormal_by_dcoeff)
        t2 = np.einsum('ik,ij,i->ijk', prod, normal, normal_norm**(-3))
        dunit_normal_by_dcoef =  t1 - t2
        return dunit_normal_by_dcoef

    def zfactor(self):
        return np.asarray(self.zfactor_jax(self.get_dofs(), self.surf.get_dofs()))

    def dzfactor_by_dcoeff(self):
        return np.asarray(
            self.dzfactor_by_dcoeff_jax(self.get_dofs(), self.surf.get_dofs())
        )

    def dzfactor_by_dsurf(self):
        return np.asarray(
            self.dzfactor_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())
        )
        
    def dzfactor_by_dcoeff_vjp(self,v):
        return Derivative({
            self: self.dzfactor_by_dcoeff_vjp_impl(v),
            self.surf: self.dzfactor_by_dsurf_vjp_impl(v),
        })

    def dzfactor_by_dcoeff_vjp_impl(self, v):
        return np.asarray(
            self.dzfactor_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    def dzfactor_by_dsurf_vjp_impl(self, v):
        return np.asarray(
            self.dzfactor_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    def rfactor(self):
        return np.asarray(self.rfactor_jax(self.get_dofs(), self.surf.get_dofs()))
    
    def drfactor_by_dcoeff(self):
        return np.asarray(
            self.drfactor_by_dcoeff_jax(self.get_dofs(), self.surf.get_dofs())
        )

    def drfactor_by_dsurf(self):
        return np.asarray(
            self.drfactor_by_dsurf_jax(self.get_dofs(), self.surf.get_dofs())
        )

    def drfactor_by_dcoeff_vjp(self, v):
        return Derivative({
            self: self.drfactor_by_dcoeff_vjp_impl(v),
            self.surf: self.drfactor_by_dsurf_vjp_impl(v),
        })

    def drfactor_by_dcoeff_vjp_impl(self, v):
        return np.asarray(
            self.drfactor_by_dcoeff_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )

    def drfactor_by_dsurf_vjp_impl(self, v):
        return np.asarray(
            self.drfactor_by_dsurf_vjp_jax(self.get_dofs(), self.surf.get_dofs(), v)
        )
    
