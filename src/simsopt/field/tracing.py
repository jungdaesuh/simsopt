import logging
from collections.abc import Callable
from math import sqrt

import numpy as np

import simsoptpp as sopp
from .._core.util import parallel_loop_bounds
from ..backend.runtime import is_jax_backend
from ..field.magneticfield import MagneticField
from ..field.boozermagneticfield import BoozerMagneticField
from ..field.sampling import draw_uniform_on_curve, draw_uniform_on_surface
from ..geo.surface import SurfaceClassifier, _surface_classifier_from_interpolant
from ..util.constants import (
    ALPHA_PARTICLE_MASS,
    ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from .._core.types import RealArray


logger = logging.getLogger(__name__)


def _allgather_flat(comm, values):
    if comm is None:
        return values
    return [item for rank_values in comm.allgather(values) for item in rank_values]


def _event_hits_prefix(phi_hits, phi_hits_count, *, context: str) -> np.ndarray:
    hits = np.asarray(phi_hits, dtype=np.float64)
    count = int(phi_hits_count)
    max_phi_hits = hits.shape[0]
    if count > max_phi_hits:
        raise RuntimeError(
            f"{context} recorded {count} event rows but max_phi_hits={max_phi_hits}; "
            "increase the JAX tracing event buffer before using these results."
        )
    return hits[:count]


__all__ = [
    "SurfaceClassifier",
    "LevelsetStoppingCriterion",
    "MinToroidalFluxStoppingCriterion",
    "MaxToroidalFluxStoppingCriterion",
    "MinRStoppingCriterion",
    "MinZStoppingCriterion",
    "MaxRStoppingCriterion",
    "MaxZStoppingCriterion",
    "IterationStoppingCriterion",
    "ToroidalTransitStoppingCriterion",
    "compute_fieldlines",
    "compute_resonances",
    "compute_poloidal_transits",
    "compute_toroidal_transits",
    "trace_particles",
    "trace_particles_boozer",
    "trace_particles_starting_on_curve",
    "trace_particles_starting_on_surface",
    "particles_to_vtk",
    "plot_poincare_data",
]


def _normalize_parallel_speeds(
    parallel_speeds: RealArray, nparticles: int
) -> np.ndarray:
    speed_par = np.asarray(parallel_speeds).reshape(-1)
    if speed_par.shape != (nparticles,):
        raise ValueError(
            f"Expected {nparticles} parallel speeds, got shape {speed_par.shape}"
        )
    return speed_par


def _require_jax_field_B(field: MagneticField) -> Callable[[object], object]:
    field_fn = getattr(field, "jax_B_at", None)
    if not callable(field_fn):
        raise TypeError(
            "JAX tracing requires a JAX-native MagneticField wrapper exposing "
            "`jax_B_at(point)`. Use a *JAX field class such as "
            "ToroidalFieldJAX, or run this tracing call on the CPU backend."
        )
    return field_fn


def _require_jax_field_B_dB(
    field: MagneticField,
) -> Callable[[object], tuple[object, object]]:
    field_fn = getattr(field, "jax_B_dB_at", None)
    if callable(field_fn):
        return field_fn
    grad_abs_fn = getattr(field, "jax_B_GradAbsB_at", None)
    if callable(grad_abs_fn):
        import jax.numpy as jnp

        def _field_fn(point):
            B_raw, grad_abs_raw = grad_abs_fn(point)
            B = jnp.asarray(B_raw, dtype=jnp.float64).reshape((3,))
            grad_abs_B = jnp.asarray(grad_abs_raw, dtype=jnp.float64).reshape((3,))
            abs_B = jnp.linalg.norm(B)
            dB_by_dX = grad_abs_B[:, None] * B[None, :] / abs_B
            return B, dB_by_dX

        return _field_fn
    raise TypeError(
        "JAX particle tracing requires a JAX-native MagneticField wrapper "
        "exposing `jax_B_dB_at(point)` or `jax_B_GradAbsB_at(point)`. Use a "
        "*JAX field class such as ToroidalFieldJAX or InterpolatedFieldJAX, "
        "or run this tracing call on the CPU backend."
    )


def compute_gc_radius(m, vperp, q, absb):
    """
    Computes the gyro radius of a particle in a field with strenght ``absb```,
    that is ``r=m*vperp/(abs(q)*absb)``.
    """

    return m * vperp / (abs(q) * absb)


def gc_to_fullorbit_initial_guesses(
    field, xyz_inits, speed_pars, speed_total, m, q, eta=0
):
    """
    Takes in guiding center positions ``xyz_inits`` as well as a parallel
    speeds ``speed_pars`` and total velocities ``speed_total`` to compute orbit
    positions for a full orbit calculation that matches the given guiding
    center. The phase angle can be controll via the `eta` parameter
    """

    nparticles = xyz_inits.shape[0]
    xyz_inits_full = np.zeros_like(xyz_inits)
    v_inits = np.zeros((nparticles, 3))
    rgs = np.zeros((nparticles,))
    field.set_points(xyz_inits)
    Bs = field.B()
    AbsBs = field.AbsB()
    eB = Bs / AbsBs
    for i in range(nparticles):
        p1 = eB[i, :]
        p2 = np.asarray([0, 0, 1])
        p3 = -np.cross(p1, p2)
        p3 *= 1.0 / np.linalg.norm(p3)
        q1 = p1
        q2 = p2 - np.sum(q1 * p2) * q1
        q2 = q2 / np.sum(q2**2) ** 0.5
        q3 = p3 - np.sum(q1 * p3) * q1 - np.sum(q2 * p3) * q2
        q3 = q3 / np.sum(q3**2) ** 0.5
        speed_perp = np.sqrt(speed_total**2 - speed_pars[i] ** 2)
        rgs[i] = compute_gc_radius(m, speed_perp, q, AbsBs[i, 0])

        xyz_inits_full[i, :] = (
            xyz_inits[i, :] + rgs[i] * np.sin(eta) * q2 + rgs[i] * np.cos(eta) * q3
        )
        vperp = -speed_perp * np.cos(eta) * q2 + speed_perp * np.sin(eta) * q3
        v_inits[i, :] = speed_pars[i] * q1 + vperp
    return xyz_inits_full, v_inits, rgs


def trace_particles_boozer(
    field: BoozerMagneticField,
    stz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    zetas=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
):
    r"""
    Follow particles in a :class:`BoozerMagneticField`. This is modeled after
    :func:`trace_particles`.


    In the case of ``mod='gc_vac'`` we solve the guiding center equations under
    the vacuum assumption, i.e :math:`G =` const. and :math:`I = 0`:

    .. math::

        \dot s = -|B|_{,\theta} m(v_{||}^2/|B| + \mu)/(q \psi_0)

        \dot \theta = |B|_{,s} m(v_{||}^2/|B| + \mu)/(q \psi_0) + \iota v_{||} |B|/G

        \dot \zeta = v_{||}|B|/G

        \dot v_{||} = -(\iota |B|_{,\theta} + |B|_{,\zeta})\mu |B|/G,

    where :math:`q` is the charge, :math:`m` is the mass, and :math:`v_\perp^2 = 2\mu|B|`.

    In the case of ``mode='gc'`` we solve the general guiding center equations
    for an MHD equilibrium:

    .. math::

        \dot s = (I |B|_{,\zeta} - G |B|_{,\theta})m(v_{||}^2/|B| + \mu)/(\iota D \psi_0)

        \dot \theta = ((G |B|_{,\psi} - K |B|_{,\zeta}) m(v_{||}^2/|B| + \mu) - C v_{||} |B|)/(\iota D)

        \dot \zeta = (F v_{||} |B| - (|B|_{,\psi} I - |B|_{,\theta} K) m(\rho_{||}^2 |B| + \mu) )/(\iota D)

        \dot v_{||} = (C|B|_{,\theta} - F|B|_{,\zeta})\mu |B|/(\iota D)

        C = - m v_{||} K_{,\zeta}/|B|  - q \iota + m v_{||}G'/|B|

        F = - m v_{||} K_{,\theta}/|B| + q + m v_{||}I'/|B|

        D = (F G - C I))/\iota

    where primes indicate differentiation wrt :math:`\psi`. In the case ``mod='gc_noK'``,
    the above equations are used with :math:`K=0`.

    Args:
        field: The :class:`BoozerMagneticField` instance
        stz_inits: A ``(nparticles, 3)`` array with the initial positions of
            the particles in Boozer coordinates :math:`(s,\theta,\zeta)`.
        parallel_speeds: A ``(nparticles, )`` array containing the speed in
            direction of the B field for each particle.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        zetas: list of angles in [0, 2pi] for which intersection with the plane
            corresponding to that zeta should be computed
        stopping_criteria: list of stopping criteria, mostly used in
            combination with the ``LevelsetStoppingCriterion``
            accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. Options are
            `gc`: general guiding center equations.
            `gc_vac`: simplified guiding center equations for the case :math:`G` = const.,
            :math:`I = 0`, and :math:`K = 0`.
            `gc_noK`: simplified guiding center equations for the case :math:`K = 0`.
        forget_exact_path: return only the first and last position of each
            particle for the ``res_tys``. To be used when only res_zeta_hits is of
            interest or one wants to reduce memory usage.

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, M)
            with M depending on the ``mode``.  Each row contains the time and
            the state.  So for `mode='gc'` and `mode='gc_vac'` the state
            consists of the :math:`(s,\theta,\zeta)` position and the parallel speed, hence
            each row contains `[t, s, t, z, v_par]`.

        - ``res_zeta_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the zeta planes or
            one of the stopping criteria. Each row of the array contains
            `[time] + [idx] + state`, where `idx` tells us which of the `zetas`
            or `stopping_criteria` was hit.  If `idx>=0`, then `zetas[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.
    """

    nparticles = stz_inits.shape[0]
    speed_par = _normalize_parallel_speeds(parallel_speeds, nparticles)
    m = mass
    speed_total = sqrt(2 * Ekin / m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    mode = mode.lower()
    assert mode in ["gc", "gc_vac", "gc_nok"]

    if is_jax_backend():
        return _trace_particles_boozer_jax(
            field,
            stz_inits,
            speed_par,
            speed_total,
            tmax=tmax,
            mass=m,
            charge=charge,
            tol=tol,
            comm=comm,
            zetas=zetas,
            stopping_criteria=stopping_criteria,
            mode=mode,
            forget_exact_path=forget_exact_path,
        )

    res_tys = []
    res_zeta_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        res_ty, res_zeta_hit = sopp.particle_guiding_center_boozer_tracing(
            field,
            stz_inits[i, :],
            m,
            charge,
            speed_total,
            speed_par[i],
            tmax,
            tol,
            vacuum=(mode == "gc_vac"),
            noK=(mode == "gc_nok"),
            zetas=zetas,
            stopping_criteria=stopping_criteria,
        )
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_zeta_hits.append(np.asarray(res_zeta_hit))
        dtavg = res_ty[-1][0] / len(res_ty)
        logger.debug(
            f"{i + 1:3d}/{nparticles}, t_final={res_ty[-1][0]}, average timestep {1000 * dtavg:.10f}ms"
        )
        if res_ty[-1][0] < tmax - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_zeta_hits = [i for o in comm.allgather(res_zeta_hits) for i in o]
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}={(100 * loss_ctr) // nparticles:d}%"
    )
    return res_tys, res_zeta_hits


def _trace_particles_boozer_jax(
    field,
    stz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    zetas,
    stopping_criteria,
    mode,
    forget_exact_path,
):
    """JAX backend for :func:`trace_particles_boozer` (4-state Boozer GC).

    Routes each initial particle through
    :func:`simsopt.jax_core.tracing.trace_guiding_center_boozer`. The
    driver switches between the upstream
    ``GuidingCenterVacuumBoozerRHS`` / ``GuidingCenterNoKBoozerRHS`` /
    ``GuidingCenterBoozerRHS`` equations via the ``mode`` argument and
    reproduces the upstream return shape
    ``(res_tys, res_zeta_hits)`` (each row of the trajectory being
    ``[t, s, theta, zeta, v_par]``).

    Zeta-plane crossings (``zetas`` argument) and stopping criteria
    are wired through the JAX driver's fixed-shape ``phi_hits`` buffer
    (named ``phi_hits`` on the dataclass for layout-compatibility with
    the Cartesian route; on the Boozer route it stores zeta-plane
    crossings with per-row layout ``[t_hit, idx, s, theta, zeta,
    v_par]``). Stopping-criterion fires populate the same buffer with
    ``idx < 0``; flux-coordinate criteria
    (``Min/MaxToroidalFluxStoppingCriterion``) fire on ``s = y[0]``
    and the iteration criterion fires on the accepted-step counter.
    Unsupported criterion types raise :class:`NotImplementedError`
    from :func:`_translate_stopping_criteria_to_jax`.

    Rejected argument shapes (explicit :class:`NotImplementedError`):

    - ``field`` not a :class:`BoozerRadialInterpolantJAX` or
      :class:`InterpolatedBoozerFieldJAX` instance — the JAX path
      requires a frozen-state JAX wrapper to evaluate ``modB``,
      ``modB_derivs``, ``K``, ``K_derivs``, ``iota``, ``G``, ``I``,
      ``dGds``, ``dIds`` on-device. CPU ``BoozerMagneticField``
      instances are rejected under the JAX backend instead of falling
      through to the C++ oracle.
    - ``stopping_criteria`` containing an unsupported criterion type
      — raises :class:`NotImplementedError` from
      :func:`_translate_stopping_criteria_to_jax`.
    MPI ``comm`` uses the same host-level contiguous split/gather as
    the CPU wrapper. Each rank runs the JAX driver on its assigned
    particles; no compiled cross-rank collective is introduced.
    """

    from .boozermagneticfield_jax import (
        BoozerRadialInterpolantJAX,
        InterpolatedBoozerFieldJAX,
    )

    if not isinstance(field, (BoozerRadialInterpolantJAX, InterpolatedBoozerFieldJAX)):
        raise NotImplementedError(
            "trace_particles_boozer JAX backend requires a "
            "BoozerRadialInterpolantJAX or InterpolatedBoozerFieldJAX field "
            f"instance; got {type(field).__name__}. Wrap the upstream "
            "BoozerRadialInterpolant via BoozerRadialInterpolantJAX(upstream), "
            "wrap the upstream InterpolatedBoozerField via "
            "InterpolatedBoozerFieldJAX(...), or switch to the CPU backend."
        )
    rhs_mode_map = {
        "gc_vac": "vacuum",
        "gc_nok": "no_k",
        "gc": "full",
    }
    if mode not in rhs_mode_map:
        raise NotImplementedError(
            "trace_particles_boozer JAX backend supports mode in "
            f"{set(rhs_mode_map)}; got mode={mode!r}. Full-orbit "
            "Boozer tracing is not implemented."
        )
    rhs_mode = rhs_mode_map[mode]

    import jax.numpy as jnp

    from ..jax_core.tracing import (
        GuidingCenterTracingSpec,
        trace_guiding_center_boozer,
    )

    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(zetas) > 0:
        zetas_arr = jnp.asarray(list(zetas), dtype=jnp.float64)
    else:
        zetas_arr = None

    nparticles = stz_inits.shape[0]
    # Static step budget mirrors `_trace_particles_jax_guiding_center_vacuum`.
    max_steps = 4000
    max_phi_hits = 4096
    res_tys = []
    res_zeta_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        point = np.asarray(stz_inits[i, :], dtype=np.float64).reshape((1, 3))
        field.set_points(point)
        abs_B_initial = float(np.asarray(field.modB(), dtype=np.float64).reshape(-1)[0])
        vperp2 = max(speed_total * speed_total - speed_par[i] * speed_par[i], 0.0)
        mu_i = vperp2 / (2.0 * abs_B_initial)
        spec = GuidingCenterTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        y0 = jnp.asarray(
            [
                float(stz_inits[i, 0]),
                float(stz_inits[i, 1]),
                float(stz_inits[i, 2]),
                float(speed_par[i]),
            ],
            dtype=jnp.float64,
        )
        result = trace_guiding_center_boozer(
            spec,
            y0,
            field,
            m=float(mass),
            q=float(charge),
            mu=float(mu_i),
            mode=rhs_mode,
            zetas=zetas_arr,
            stopping_criteria=jax_stopping_criteria,
        )
        traj = np.asarray(result.trajectory, dtype=np.float64)
        traj_mask = np.asarray(result.mask, dtype=bool)
        live = traj[traj_mask]
        if forget_exact_path and live.shape[0] >= 2:
            res_tys.append(np.stack([live[0], live[-1]], axis=0))
        else:
            res_tys.append(live)
        res_zeta_hits.append(
            _event_hits_prefix(
                result.phi_hits,
                result.phi_hits_count,
                context="JAX Boozer guiding-centre tracing",
            )
        )
        status = int(result.status)
        t_final = float(result.t_final)
        if status > 0:
            logger.debug(
                f"{i + 1:3d}/{nparticles}, JAX Boozer guiding-centre "
                f"status={status} t_final={t_final}, "
                f"steps_taken={int(result.steps_taken)}"
            )
        elif status < 0:
            logger.debug(
                f"{i + 1:3d}/{nparticles}, JAX Boozer guiding-centre stopped by "
                f"criterion index {-1 - status}, t_final={t_final}"
            )
        if t_final < float(tmax) - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_zeta_hits = _allgather_flat(comm, res_zeta_hits)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX Boozer backend)"
    )
    return res_tys, res_zeta_hits


def trace_particles(
    field: MagneticField,
    xyz_inits: RealArray,
    parallel_speeds: RealArray,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    phis=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
    phase_angle=0,
):
    r"""
    Follow particles in a magnetic field.

    In the case of ``mod='full'`` we solve

    .. math::

        [\ddot x, \ddot y, \ddot z] = \frac{q}{m}  [\dot x, \dot y, \dot z] \times B

    in the case of ``mod='gc_vac'`` we solve the guiding center equations under
    the assumption :math:`\nabla p=0`, that is

    .. math::

        [\dot x, \dot y, \dot z] &= v_{||}\frac{B}{|B|} + \frac{m}{q|B|^3}  (0.5v_\perp^2 + v_{||}^2)  B\times \nabla(|B|)\\
        \dot v_{||}    &= -\mu  (B \cdot \nabla(|B|))

    where :math:`v_\perp = 2\mu|B|`. See equations (12) and (13) of
    [Guiding Center Motion, H.J. de Blank, https://doi.org/10.13182/FST04-A468].

    Args:
        field: The magnetic field :math:`B`.
        xyz_inits: A (nparticles, 3) array with the initial positions of the particles.
        parallel_speeds: A (nparticles, ) array containing the speed in direction of the B field
                         for each particle.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, M)
            with M depending on the ``mode``.  Each row contains the time and
            the state.  So for `mode='gc'` and `mode='gc_vac'` the state
            consists of the xyz position and the parallel speed, hence
            each row contains `[t, x, y, z, v_par]`.  For `mode='full'`, the
            state consists of position and velocity vector, i.e. each row
            contains `[t, x, y, z, vx, vy, vz]`.

        - ``res_phi_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the phi planes or
            one of the stopping criteria. Each row of the array contains
            `[time] + [idx] + state`, where `idx` tells us which of the `phis`
            or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.

    Backend routing
    ---------------
    When ``simsopt.backend.is_jax_backend()`` returns ``True`` the public
    wrapper routes JAX-native field wrappers to one of two in-repo JAX
    Dormand-Prince drivers depending on the requested ``mode``:

    - ``mode in {'gc', 'gc_vac'}`` routes to
      :func:`_trace_particles_jax_guiding_center_vacuum` (the
      4-state Cartesian vacuum guiding-centre driver shipped under
      item 14). Only ``mode='gc_vac'`` is implemented today; ``mode='gc'``
      (non-vacuum guiding-centre) raises explicit
      :class:`NotImplementedError`. The JAX route requires the field
      object to expose ``jax_B_dB_at(point)`` or
      ``jax_B_GradAbsB_at(point)``.
    - ``mode='full'`` routes to
      :func:`_trace_particles_jax_fullorbit_vacuum` (the 6-state
      Cartesian full-orbit Lorentz driver). The JAX route requires the
      field object to expose ``jax_B_at(point)``; the
      guiding-centre-to-full-orbit seeding consults the CPU
      ``MagneticField`` interface on the host.

    CPU ``MagneticField`` instances without the required ``jax_*_at``
    hooks are rejected instead of being bridged through host
    callbacks. The Cartesian guiding-centre route supports phi-plane
    crossings and the Python stopping-criterion wrappers translated by
    :func:`_translate_stopping_criteria_to_jax`; the full-orbit route
    supports the same translated stopping criteria and fixed-shape
    event buffer. MPI ``comm`` uses the same host-level split/gather as
    the CPU wrapper; each rank runs its assigned JAX traces locally. No
    silent fallback to the C++ path is allowed.
    """

    nparticles = xyz_inits.shape[0]
    speed_par = _normalize_parallel_speeds(parallel_speeds, nparticles)
    mode = mode.lower()
    assert mode in ["gc", "gc_vac", "full"]
    m = mass
    speed_total = sqrt(2 * Ekin / m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)

    if is_jax_backend():
        if mode == "full":
            return _trace_particles_jax_fullorbit_vacuum(
                field,
                xyz_inits,
                speed_par,
                speed_total,
                tmax=tmax,
                mass=m,
                charge=charge,
                tol=tol,
                comm=comm,
                phis=phis,
                stopping_criteria=stopping_criteria,
                forget_exact_path=forget_exact_path,
                phase_angle=phase_angle,
            )
        return _trace_particles_jax_guiding_center_vacuum(
            field,
            xyz_inits,
            speed_par,
            speed_total,
            tmax=tmax,
            mass=m,
            charge=charge,
            tol=tol,
            comm=comm,
            phis=phis,
            stopping_criteria=stopping_criteria,
            mode=mode,
            forget_exact_path=forget_exact_path,
        )

    if mode == "full":
        xyz_inits, v_inits, _ = gc_to_fullorbit_initial_guesses(
            field, xyz_inits, speed_par, speed_total, m, charge, eta=phase_angle
        )
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        if "gc" in mode:
            res_ty, res_phi_hit = sopp.particle_guiding_center_tracing(
                field,
                xyz_inits[i, :],
                m,
                charge,
                speed_total,
                speed_par[i],
                tmax,
                tol,
                vacuum=(mode == "gc_vac"),
                phis=phis,
                stopping_criteria=stopping_criteria,
            )
        else:
            res_ty, res_phi_hit = sopp.particle_fullorbit_tracing(
                field,
                xyz_inits[i, :],
                v_inits[i, :],
                m,
                charge,
                tmax,
                tol,
                phis=phis,
                stopping_criteria=stopping_criteria,
            )
        if not forget_exact_path:
            res_tys.append(np.asarray(res_ty))
        else:
            res_tys.append(np.asarray([res_ty[0], res_ty[-1]]))
        res_phi_hits.append(np.asarray(res_phi_hit))
        dtavg = res_ty[-1][0] / len(res_ty)
        logger.debug(
            f"{i + 1:3d}/{nparticles}, t_final={res_ty[-1][0]}, average timestep {1000 * dtavg:.10f}ms"
        )
        if res_ty[-1][0] < tmax - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_phi_hits = [i for o in comm.allgather(res_phi_hits) for i in o]
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}={(100 * loss_ctr) // nparticles:d}%"
    )
    return res_tys, res_phi_hits


def _trace_particles_jax_guiding_center_vacuum(
    field,
    xyz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    phis,
    stopping_criteria,
    mode,
    forget_exact_path,
):
    """JAX backend for :func:`trace_particles` (4-state Cartesian vacuum GC).

    Routes each initial particle through
    :func:`simsopt.jax_core.tracing.trace_guiding_center`. The driver
    follows the upstream ``GuidingCenterVacuumRHS`` equations of motion
    in Cartesian coordinates. The wrapper reproduces the upstream
    return shape ``(res_tys, res_phi_hits)`` (with each row of the
    trajectory being ``[t, x, y, z, v_par]``) so call sites are
    agnostic to the backend choice.

    Rejected argument shapes (explicit :class:`NotImplementedError`):

    - ``mode != 'gc_vac'`` — only the vacuum guiding-centre Cartesian
      RHS is implemented in this helper. ``mode='gc'`` (non-vacuum
      guiding-centre) is a deferred follow-up. ``mode='full'``
      (``FullorbitRHS``) is now routed by
      :func:`_trace_particles_jax_fullorbit_vacuum` from the public
      :func:`trace_particles` wrapper and is never seen here.
    MPI ``comm`` uses the same host-level contiguous split/gather as
    the CPU wrapper. Each rank runs the JAX driver on its assigned
    particles; no compiled cross-rank collective is introduced.

    Phi-plane crossings and translated stopping criteria are wired
    through the JAX driver's fixed-shape ``phi_hits`` buffer. Levelset
    criteria built from ``SurfaceClassifier`` are threaded into the
    integration loop as a JAX classifier closure.
    """
    if mode != "gc_vac":
        raise NotImplementedError(
            "trace_particles JAX guiding-centre helper currently only "
            f"supports mode='gc_vac' (got mode={mode!r}). The non-vacuum "
            "guiding-centre mode ('gc') is a deferred follow-up. The "
            "'full' mode is routed separately to "
            "_trace_particles_jax_fullorbit_vacuum from the public "
            "trace_particles wrapper. Drop the mode override or switch "
            "to the CPU backend."
        )
    import jax.numpy as jnp

    from ..jax_core.tracing import (
        GuidingCenterTracingSpec,
        trace_guiding_center,
    )

    field_fn = _require_jax_field_B_dB(field)

    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(phis) > 0:
        phis_arr = jnp.asarray(list(phis), dtype=jnp.float64)
    else:
        phis_arr = None

    # Compute mu (magnetic moment) per particle from the initial state.
    # mu = v_perp^2 / (2 |B|) where v_perp^2 = v_total^2 - v_par^2 (the
    # upstream definition; see particle_guiding_center_tracing in
    # simsoptpp/tracing.cpp). |B| is evaluated at the initial position
    # using the CPU field (no JAX needed: mu is a fixed parameter once
    # the orbit starts).
    nparticles = xyz_inits.shape[0]
    # Static step budget mirroring _compute_fieldlines_jax. The lane
    # bounds the JAX vs CPU accepted step count to within 25%, so 4000
    # is generous for the lane fixtures and leaves headroom for stiff
    # fields.
    max_steps = 4000
    max_phi_hits = 4096
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0
    first, last = parallel_loop_bounds(comm, nparticles)
    for i in range(first, last):
        B_initial, _ = field_fn(jnp.asarray(xyz_inits[i], dtype=jnp.float64))
        abs_B_initial = np.linalg.norm(np.asarray(B_initial, dtype=np.float64))
        vperp2 = max(speed_total * speed_total - speed_par[i] * speed_par[i], 0.0)
        mu_i = vperp2 / (2.0 * float(abs_B_initial))
        spec = GuidingCenterTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        y0 = jnp.asarray(
            [
                float(xyz_inits[i, 0]),
                float(xyz_inits[i, 1]),
                float(xyz_inits[i, 2]),
                float(speed_par[i]),
            ],
            dtype=jnp.float64,
        )
        result = trace_guiding_center(
            spec,
            y0,
            field_fn,
            m=float(mass),
            q=float(charge),
            mu=float(mu_i),
            phis=phis_arr,
            stopping_criteria=jax_stopping_criteria,
        )
        traj = np.asarray(result.trajectory, dtype=np.float64)
        mask = np.asarray(result.mask, dtype=bool)
        live = traj[mask]
        if forget_exact_path and live.shape[0] >= 2:
            res_tys.append(np.stack([live[0], live[-1]], axis=0))
        else:
            res_tys.append(live)
        res_phi_hits.append(
            _event_hits_prefix(
                result.phi_hits,
                result.phi_hits_count,
                context="JAX guiding-centre tracing",
            )
        )
        status = int(result.status)
        t_final = float(result.t_final)
        if status > 0:
            logger.debug(
                f"{i + 1:3d}/{nparticles}, JAX guiding-centre status={status} "
                f"t_final={t_final}, steps_taken={int(result.steps_taken)}"
            )
        elif status < 0:
            logger.debug(
                f"{i + 1:3d}/{nparticles}, JAX guiding-centre stopped by "
                f"criterion index {-1 - status}, t_final={t_final}"
            )
        if t_final < float(tmax) - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX backend)"
    )
    return res_tys, res_phi_hits


def _trace_particles_jax_fullorbit_vacuum(
    field,
    xyz_inits,
    speed_par,
    speed_total,
    *,
    tmax,
    mass,
    charge,
    tol,
    comm,
    phis,
    stopping_criteria,
    forget_exact_path,
    phase_angle,
):
    """JAX backend for :func:`trace_particles` (6-state Cartesian full orbit).

    Routes each initial particle through
    :func:`simsopt.jax_core.tracing.trace_fullorbit`. The driver follows
    the upstream ``FullorbitRHS`` Lorentz equations of motion in
    Cartesian coordinates (vacuum branch: ``dv/dt = (q/m) v x B``, no E
    field). The wrapper reproduces the upstream return shape
    ``(res_tys, res_phi_hits)`` (with each row of the trajectory being
    ``[t, x, y, z, vx, vy, vz]``) so call sites are agnostic to the
    backend choice.

    The guiding-centre-to-full-orbit transformation reuses the upstream
    :func:`gc_to_fullorbit_initial_guesses` helper to seed the
    full-orbit initial conditions from the supplied guiding-centre
    positions, parallel speeds, and phase angle. The helper requires a
    CPU ``MagneticField`` interface (``field.B()`` / ``field.AbsB()``)
    so the initial-guess construction happens on the host even when the
    JAX backend is active; only the inner ODE integration runs on the
    JAX device.

    Phi-plane crossings and non-Levelset stopping criteria are wired
    through the JAX driver's fixed-shape ``phi_hits`` buffer; the
    rows are ``[t_hit, idx, x, y, z, vx, vy, vz]`` matching the
    upstream ``sopp.particle_fullorbit_tracing`` row layout (8 columns).

    Rejected argument shapes (explicit :class:`NotImplementedError`):

    - ``stopping_criteria`` containing an unsupported criterion type
      — raises :class:`NotImplementedError` from
      :func:`_translate_stopping_criteria_to_jax`.
    MPI ``comm`` uses the same host-level contiguous split/gather as
    the CPU wrapper. Each rank runs the JAX driver on its assigned
    particles; no compiled cross-rank collective is introduced.
    """
    import jax.numpy as jnp

    from ..jax_core.tracing import (
        FullorbitTracingSpec,
        trace_fullorbit,
    )

    field_fn = _require_jax_field_B(field)
    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(phis) > 0:
        phis_arr = jnp.asarray(list(phis), dtype=jnp.float64)
    else:
        phis_arr = None

    # Seed full-orbit initial conditions from guiding-centre inputs via
    # the upstream helper. The helper consults the CPU ``MagneticField``
    # interface (``set_points`` / ``B`` / ``AbsB``) to compute the local
    # ``b_hat`` and gyroradius; this is host work, identical to the C++
    # branch in ``trace_particles``.
    # Static step budget. Full-orbit gyromotion has a much shorter
    # characteristic time than the guiding-centre drift, so we lift the
    # cap to 20000 — generous enough for typical fusion-particle
    # gyroperiods at the lane fixtures while still bounding compile
    # carry size.
    max_steps = 20000
    max_phi_hits = 4096
    nparticles = xyz_inits.shape[0]
    first, last = parallel_loop_bounds(comm, nparticles)
    local_xyz = np.asarray(xyz_inits[first:last], dtype=np.float64)
    local_speed_par = np.asarray(speed_par[first:last], dtype=np.float64)
    if local_xyz.shape[0] > 0:
        xyz_inits_full, v_inits, _ = gc_to_fullorbit_initial_guesses(
            field,
            local_xyz,
            local_speed_par,
            float(speed_total),
            float(mass),
            float(charge),
            eta=float(phase_angle),
        )
    else:
        xyz_inits_full = np.zeros((0, 3), dtype=np.float64)
        v_inits = np.zeros((0, 3), dtype=np.float64)
    res_tys = []
    res_phi_hits = []
    loss_ctr = 0
    for local_i, i in enumerate(range(first, last)):
        spec = FullorbitTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        y0 = jnp.asarray(
            [
                float(xyz_inits_full[local_i, 0]),
                float(xyz_inits_full[local_i, 1]),
                float(xyz_inits_full[local_i, 2]),
                float(v_inits[local_i, 0]),
                float(v_inits[local_i, 1]),
                float(v_inits[local_i, 2]),
            ],
            dtype=jnp.float64,
        )
        result = trace_fullorbit(
            spec,
            y0,
            field_fn,
            m=float(mass),
            q=float(charge),
            phis=phis_arr,
            stopping_criteria=jax_stopping_criteria,
        )
        traj = np.asarray(result.trajectory, dtype=np.float64)
        mask = np.asarray(result.mask, dtype=bool)
        live = traj[mask]
        if forget_exact_path and live.shape[0] >= 2:
            res_tys.append(np.stack([live[0], live[-1]], axis=0))
        else:
            res_tys.append(live)
        res_phi_hits.append(
            _event_hits_prefix(
                result.phi_hits,
                result.phi_hits_count,
                context="JAX full-orbit tracing",
            )
        )
        status = int(result.status)
        t_final = float(result.t_final)
        if status > 0:
            logger.debug(
                f"{i + 1:3d}/{nparticles}, JAX full-orbit status={status} "
                f"t_final={t_final}, steps_taken={int(result.steps_taken)}"
            )
        elif status < 0:
            logger.debug(
                f"{i + 1:3d}/{nparticles}, JAX full-orbit stopped by "
                f"criterion index {-1 - status}, t_final={t_final}"
            )
        if t_final < float(tmax) - 1e-15:
            loss_ctr += 1
    if comm is not None:
        loss_ctr = comm.allreduce(loss_ctr)
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    logger.debug(
        f"Particles lost {loss_ctr}/{nparticles}="
        f"{(100 * loss_ctr) // max(nparticles, 1):d}% (JAX full-orbit backend)"
    )
    return res_tys, res_phi_hits


def trace_particles_starting_on_curve(
    curve,
    field,
    nparticles,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    seed=1,
    umin=-1,
    umax=+1,
    phis=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
    phase_angle=0,
):
    r"""
    Follows particles spawned at random locations on the magnetic axis with random pitch angle.
    See :mod:`simsopt.field.tracing.trace_particles` for the governing equations.

    Args:
        curve: The :mod:`simsopt.geo.curve.Curve` to spawn the particles on. Uses rejection sampling
               to sample points on the curve. *Warning*: assumes that the underlying
               quadrature points on the Curve are uniformly distributed.
        field: The magnetic field :math:`B`.
        nparticles: number of particles to follow.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        seed: random seed
        umin: the parallel speed is defined as  ``v_par = u * speed_total``
              where  ``u`` is drawn uniformly in ``[umin, umax]``
        umax: see ``umin``
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: see :mod:`simsopt.field.tracing.trace_particles`
    """
    m = mass
    speed_total = sqrt(2 * Ekin / m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    np.random.seed(seed)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles,))
    speed_par = us * speed_total
    xyz, _ = draw_uniform_on_curve(curve, nparticles, safetyfactor=10)
    return trace_particles(
        field,
        xyz,
        speed_par,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        comm=comm,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
        phase_angle=phase_angle,
    )


def trace_particles_starting_on_surface(
    surface,
    field,
    nparticles,
    tmax=1e-4,
    mass=ALPHA_PARTICLE_MASS,
    charge=ALPHA_PARTICLE_CHARGE,
    Ekin=FUSION_ALPHA_PARTICLE_ENERGY,
    tol=1e-9,
    comm=None,
    seed=1,
    umin=-1,
    umax=+1,
    phis=[],
    stopping_criteria=[],
    mode="gc_vac",
    forget_exact_path=False,
    phase_angle=0,
):
    r"""
    Follows particles spawned at random locations on the magnetic axis with random pitch angle.
    See :mod:`simsopt.field.tracing.trace_particles` for the governing equations.

    Args:
        surface: The :mod:`simsopt.geo.surface.Surface` to spawn the particles
                 on. Uses rejection sampling to sample points on the curve. *Warning*:
                 assumes that the underlying quadrature points on the Curve as uniformly
                 distributed.
        field: The magnetic field :math:`B`.
        nparticles: number of particles to follow.
        tmax: integration time
        mass: particle mass in kg, defaults to the mass of an alpha particle
        charge: charge in Coulomb, defaults to the charge of an alpha particle
        Ekin: kinetic energy in Joule, defaults to 3.52MeV
        tol: tolerance for the adaptive ode solver
        comm: MPI communicator to parallelize over
        seed: random seed
        umin: the parallel speed is defined as  ``v_par = u * speed_total``
              where  ``u`` is drawn uniformly in ``[umin, umax]``.
        umax: see ``umin``
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.
        mode: how to trace the particles. options are
            `gc`: general guiding center equations,
            `gc_vac`: simplified guiding center equations for the case :math:`\nabla p=0`,
            `full`: full orbit calculation (slow!)
        forget_exact_path: return only the first and last position of each
                           particle for the ``res_tys``. To be used when only res_phi_hits is of
                           interest or one wants to reduce memory usage.
        phase_angle: the phase angle to use in the case of full orbit calculations

    Returns: see :mod:`simsopt.field.tracing.trace_particles`
    """
    m = mass
    speed_total = sqrt(2 * Ekin / m)  # Ekin = 0.5 * m * v^2 <=> v = sqrt(2*Ekin/m)
    np.random.seed(seed)
    us = np.random.uniform(low=umin, high=umax, size=(nparticles,))
    speed_par = us * speed_total
    xyz, _ = draw_uniform_on_surface(surface, nparticles, safetyfactor=10)
    return trace_particles(
        field,
        xyz,
        speed_par,
        tmax=tmax,
        mass=mass,
        charge=charge,
        Ekin=Ekin,
        tol=tol,
        comm=comm,
        phis=phis,
        stopping_criteria=stopping_criteria,
        mode=mode,
        forget_exact_path=forget_exact_path,
        phase_angle=phase_angle,
    )


def compute_resonances(res_tys, res_phi_hits, ma=None, delta=1e-2):
    r"""
    Computes resonant particle orbits given the output of either
    :func:`trace_particles` or :func:`trace_particles_boozer`, ``res_tys`` and
    ``res_phi_hits``/``res_zeta_hits``, with ``forget_exact_path=False``.
    Resonance indicates a trajectory which returns to the same position
    at the :math:`\zeta = 0` plane after ``mpol`` poloidal turns and
    ``ntor`` toroidal turns. For the case of particles traced in a
    :class:`MagneticField` (not a :class:`BoozerMagneticField`), the poloidal
    angle is computed using the arctangent angle in the poloidal plane with
    respect to the coordinate axis, ``ma``,

    .. math::
        \theta = \tan^{-1} \left( \frac{R(\phi)-R_{\mathrm{ma}}(\phi)}{Z(\phi)-Z_{\mathrm{ma}}(\phi)} \right),

    where :math:`(R,\phi,Z)` are the cylindrical coordinates of the trajectory
    and :math:`(R_{\mathrm{ma}}(\phi),Z_{\mathrm{ma}(\phi)})` is the position
    of the coordinate axis.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``
        res_phi_hits: output of :func:`trace_particles` or
                :func:`trace_particles_boozer` with `phis/zetas = [0]`
        ma: an instance of :class:`Curve` representing the coordinate axis with
                respect to which the poloidal angle is computed. If the orbit is
                computed in flux coordinates, ``ma`` should be ``None``. (defaults to None)
        delta: the distance tolerance in the poloidal plane used to compute
                a resonant orbit. (defaults to 1e-2)

    Returns:
        resonances: list of 7d arrays containing resonant particle orbits. The
                elements of each array is ``[s0, theta0, zeta0, vpar0, t, mpol, ntor]``
                if ``ma=None``, and ``[R0, Z0, phi0, vpar0, t, mpol, ntor]`` otherwise.
                Here ``(s0, theta0, zeta0, vpar0)/(R0, Z0, phi0, vpar0)`` indicates the
                initial position and parallel velocity of the particle, ``t``
                indicates the time of the  resonance, ``mpol`` is the number of
                poloidal turns of the orbit, and ``ntor`` is the number of toroidal turns.
    """
    flux = False
    if ma is None:
        flux = True
    nparticles = len(res_tys)
    resonances = []
    gamma = np.zeros((1, 3))
    # Iterate over particles
    for ip in range(nparticles):
        nhits = len(res_phi_hits[ip])
        if flux:
            s0 = res_tys[ip][0, 1]
            theta0 = res_tys[ip][0, 2]
            zeta0 = res_tys[ip][0, 3]
            theta0_mod = theta0 % (2 * np.pi)
            x0 = s0 * np.cos(theta0)
            y0 = s0 * np.sin(theta0)
        else:
            X0 = res_tys[ip][0, 1]
            Y0 = res_tys[ip][0, 2]
            Z0 = res_tys[ip][0, 3]
            R0 = np.sqrt(X0**2 + Y0**2)
            phi0 = np.arctan2(Y0, X0)
            ma.gamma_impl(gamma, phi0 / (2 * np.pi))
            R_ma0 = np.sqrt(gamma[0, 0] ** 2 + gamma[0, 1] ** 2)
            Z_ma0 = gamma[0, 2]
            theta0 = np.arctan2(Z0 - Z_ma0, R0 - R_ma0)
        vpar0 = res_tys[ip][0, 4]
        for it in range(1, nhits):
            # Check whether phi hit or stopping criteria achieved
            if int(res_phi_hits[ip][it, 1]) >= 0:
                if flux:
                    s = res_phi_hits[ip][it, 2]
                    theta = res_phi_hits[ip][it, 3]
                    zeta = res_phi_hits[ip][it, 4]
                    theta_mod = theta % 2 * np.pi
                    x = s * np.cos(theta)
                    y = s * np.sin(theta)
                    dist = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
                else:
                    # Check that distance is less than delta
                    X = res_phi_hits[ip][it, 2]
                    Y = res_phi_hits[ip][it, 3]
                    R = np.sqrt(X**2 + Y**2)
                    Z = res_phi_hits[ip][it, 4]
                    dist = np.sqrt((R - R0) ** 2 + (Z - Z0) ** 2)
                t = res_phi_hits[ip][it, 0]
                if dist < delta:
                    logger.debug("Resonance found.")
                    if flux:
                        logger.debug(
                            f"theta = {theta_mod}, theta0 = {theta0_mod}, s = {s}, s0 = {s0}"
                        )
                        mpol = np.rint((theta - theta0) / (2 * np.pi))
                        ntor = np.rint((zeta - zeta0) / (2 * np.pi))
                        resonances.append(
                            np.asarray([s0, theta0, zeta0, vpar0, t, mpol, ntor])
                        )
                    else:
                        # Find index of closest point along trajectory
                        indexm = np.argmin(np.abs(res_tys[ip][:, 0] - t))
                        # Compute mpol and ntor for neighboring points as well
                        indexl = indexm - 1
                        indexr = indexm + 1
                        dtl = np.abs(res_tys[ip][indexl, 0] - t)
                        trajlistl = []
                        trajlistl.append(res_tys[ip][0 : indexl + 1, :])
                        mpoll = np.abs(compute_poloidal_transits(trajlistl, ma, flux))
                        ntorl = np.abs(compute_toroidal_transits(trajlistl, flux))
                        logger.debug(
                            f"dtl ={dtl}, mpoll = {mpoll}, ntorl = {ntorl}, tl={res_tys[ip][indexl, 0]}"
                        )
                        logger.debug(
                            f"(R,Z)l = {np.sqrt(res_tys[ip][indexl, 1] ** 2 + res_tys[ip][indexl, 2] ** 2), res_tys[ip][indexl, 3]}"
                        )

                        trajlistm = []
                        dtm = np.abs(res_tys[ip][indexm, 0] - t)
                        trajlistm.append(res_tys[ip][0 : indexm + 1, :])
                        mpolm = np.abs(compute_poloidal_transits(trajlistm, ma, flux))
                        ntorm = np.abs(compute_toroidal_transits(trajlistm, flux))
                        logger.debug(
                            f"dtm ={dtm}, mpolm = {mpolm}, ntorm = {ntorm}, tm={res_tys[ip][indexm, 0]}"
                        )
                        logger.debug(
                            f"(R,Z)m = {np.sqrt(res_tys[ip][indexm, 1] ** 2 + res_tys[ip][indexm, 2] ** 2), res_tys[ip][indexm, 3]}"
                        )

                        mpolr = 0
                        ntorr = 0
                        if indexr < len(res_tys[ip][:, 0]):
                            trajlistr = []
                            dtr = np.abs(res_tys[ip][indexr, 0] - t)
                            trajlistr.append(res_tys[ip][0 : indexr + 1, :])
                            # Take maximum over neighboring points to catch near resonances
                            mpolr = np.abs(
                                compute_poloidal_transits(trajlistr, ma, flux)
                            )
                            ntorr = np.abs(compute_toroidal_transits(trajlistr, flux))
                            logger.debug(
                                f"dtr ={dtr}, mpolr = {mpolr}, ntorr = {ntorr}, tr={res_tys[ip][indexr, 0]}"
                            )
                            logger.debug(
                                f"(R,Z)r = {np.sqrt(res_tys[ip][indexr, 1] ** 2 + res_tys[ip][indexr, 2] ** 2), res_tys[ip][indexr, 3]}"
                            )

                        mpol = np.amax([mpoll, mpolm, mpolr])
                        # index_mpol = np.argmax([mpoll, mpolm, mpolr])
                        ntor = np.amax([ntorl, ntorm, ntorr])
                        # index_ntor = np.argmax([ntorl, ntorm, ntorr])
                        # index = np.amax([index_mpol, index_ntor])
                        resonances.append(
                            np.asarray([R0, Z0, phi0, vpar0, t, mpol, ntor])
                        )
    return resonances


def compute_toroidal_transits(res_tys, flux=True):
    r"""
    Computes the number of toroidal transits of an orbit.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``.
        flux: if ``True``, ``res_tys`` represents the position in flux coordinates
                (should be ``True`` if computed from :func:`trace_particles_boozer`)
    Returns:
        ntransits: array with length ``len(res_tys)``. Each element contains the
                number of toroidal transits of the orbit.
    """
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        if flux:
            phi_init = res_tys[ip][0, 3]
        else:
            phi_init = sopp.get_phi(res_tys[ip][0, 1], res_tys[ip][0, 2], np.pi)
        phi_prev = phi_init
        for it in range(1, ntraj):
            if flux:
                phi = res_tys[ip][it, 3]
            else:
                phi = sopp.get_phi(res_tys[ip][it, 1], res_tys[ip][it, 2], phi_prev)
            phi_prev = phi
        if ntraj > 1:
            ntransits[ip] = np.round((phi - phi_init) / (2 * np.pi))
    return ntransits


def compute_poloidal_transits(res_tys, ma=None, flux=True):
    r"""
    Computes the number of poloidal transits of an orbit. For the case of
    particles traced in a :class:`MagneticField` (not a :class:`BoozerMagneticField`),
    the poloidal angle is computed using the arctangent angle in the poloidal plane with
    respect to the coordinate axis, ``ma``,

    .. math::
        \theta = \tan^{-1} \left( \frac{R(\phi)-R_{\mathrm{ma}}(\phi)}{Z(\phi)-Z_{\mathrm{ma}}(\phi)} \right),

    where :math:`(R,\phi,Z)` are the cylindrical coordinates of the trajectory
    and :math:`(R_{\mathrm{ma}}(\phi),Z_{\mathrm{ma}(\phi)})` is the position
    of the coordinate axis.

    Args:
        res_tys: trajectory solution computed from :func:`trace_particles` or
                :func:`trace_particles_boozer` with ``forget_exact_path=False``.
        ma: an instance of :class:`Curve` representing the coordinate axis with
                respect to which the poloidal angle is computed. If orbit is
                computed in Boozer coordinates, ``ma`` should be ``None``.
        flux: if ``True``, ``res_tys`` represents the position in flux coordinates
                (should be ``True`` if computed from :func:`trace_particles_boozer`).
                If ``True``, ``ma`` is not used.
    Returns:
        ntransits: array with length ``len(res_tys)``. Each element contains the
                number of poloidal transits of the orbit.
    """
    if not flux:
        assert ma is not None
    nparticles = len(res_tys)
    ntransits = np.zeros((nparticles,))
    gamma = np.zeros((1, 3))
    for ip in range(nparticles):
        ntraj = len(res_tys[ip][:, 0])
        if flux:
            theta_init = res_tys[ip][0, 2]
        else:
            R_init = np.sqrt(res_tys[ip][0, 1] ** 2 + res_tys[ip][0, 2] ** 2)
            Z_init = res_tys[ip][0, 3]
            phi_init = np.arctan2(res_tys[ip][0, 2], res_tys[ip][0, 1])
            ma.gamma_impl(gamma, phi_init / (2 * np.pi))
            R_ma = np.sqrt(gamma[0, 0] ** 2 + gamma[0, 1] ** 2)
            Z_ma = gamma[0, 2]
            theta_init = sopp.get_phi(R_init - R_ma, Z_init - Z_ma, np.pi)
        theta_prev = theta_init
        for it in range(1, ntraj):
            if flux:
                theta = res_tys[ip][it, 2]
            else:
                phi = np.arctan2(res_tys[ip][it, 2], res_tys[ip][it, 1])
                ma.gamma_impl(gamma, phi / (2 * np.pi))
                R_ma = np.sqrt(gamma[0, 0] ** 2 + gamma[0, 1] ** 2)
                Z_ma = gamma[0, 2]
                R = np.sqrt(res_tys[ip][it, 1] ** 2 + res_tys[ip][it, 2] ** 2)
                Z = res_tys[ip][it, 3]
                theta = sopp.get_phi(R - R_ma, Z - Z_ma, theta_prev)
            theta_prev = theta
        if ntraj > 1:
            ntransits[ip] = np.round((theta - theta_init) / (2 * np.pi))
    return ntransits


def compute_fieldlines(
    field, R0, Z0, tmax=200, tol=1e-7, phis=[], stopping_criteria=[], comm=None
):
    r"""
    Compute magnetic field lines by solving

    .. math::

        [\dot x, \dot y, \dot z] = B(x, y, z)

    Integration is initialized on the :math:`\phi = 0` plane.

    Args:
        field: the magnetic field :math:`B`
        R0: list of radial components of initial points
        Z0: list of vertical components of initial points
        tmax: for how long to trace. will do roughly ``|B|*tmax/(2*pi*r0)`` revolutions of the device
        tol: tolerance for the adaptive ode solver
        phis: list of angles in [0, 2pi] for which intersection with the plane
              corresponding to that phi should be computed
        stopping_criteria: list of stopping criteria, mostly used in
                           combination with the ``LevelsetStoppingCriterion``
                           accessed via :obj:`simsopt.field.tracing.SurfaceClassifier`.

    Returns: 2 element tuple containing
        - ``res_tys``:
            A list of numpy arrays (one for each particle) describing the
            solution over time. The numpy array is of shape (ntimesteps, 4).
            Each row contains the time and
            the position, i.e.`[t, x, y, z]`.
        - ``res_phi_hits``:
            A list of numpy arrays (one for each particle) containing
            information on each time the particle hits one of the phi planes or
            one of the stopping criteria. Each row of the array contains
            `[time, idx, x, y, z]`, where `idx` tells us which of the `phis`
            or `stopping_criteria` was hit.  If `idx>=0`, then `phis[int(idx)]`
            was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit.

    Backend routing
    ---------------
    When ``simsopt.backend.is_jax_backend()`` returns ``True`` the public
    wrapper routes JAX-native field wrappers to
    :func:`_compute_fieldlines_jax`, the in-repo JAX Dormand-Prince
    fieldline driver shipped under item 14
    (``src/simsopt/jax_core/tracing.py``). The JAX route requires the
    field object to expose ``jax_B_at(point)``; CPU ``MagneticField``
    instances are rejected instead of being bridged through host
    callbacks. MPI ``comm`` uses the same host-level split/gather as
    the CPU wrapper; each rank runs its assigned JAX traces locally. It
    supports ``phis`` and translated Python stopping-criterion wrappers
    through the fixed-shape event buffer; unsupported criterion types
    raise :class:`NotImplementedError`.

    No silent fallback to the C++ path is allowed: callers must either
    keep ``SIMSOPT_BACKEND`` on its default (CPU) value or stay within
    the JAX-supported scope.
    """
    assert len(R0) == len(Z0)
    if is_jax_backend():
        return _compute_fieldlines_jax(
            field,
            R0,
            Z0,
            tmax=tmax,
            tol=tol,
            phis=phis,
            stopping_criteria=stopping_criteria,
            comm=comm,
        )
    nlines = len(R0)
    xyz_inits = np.zeros((nlines, 3))
    xyz_inits[:, 0] = np.asarray(R0)
    xyz_inits[:, 2] = np.asarray(Z0)
    res_tys = []
    res_phi_hits = []
    first, last = parallel_loop_bounds(comm, nlines)
    for i in range(first, last):
        res_ty, res_phi_hit = sopp.fieldline_tracing(
            field,
            xyz_inits[i, :],
            tmax,
            tol,
            phis=phis,
            stopping_criteria=stopping_criteria,
        )
        res_tys.append(np.asarray(res_ty))
        res_phi_hits.append(np.asarray(res_phi_hit))
        dtavg = res_ty[-1][0] / len(res_ty)
        logger.debug(
            f"{i + 1:3d}/{nlines}, t_final={res_ty[-1][0]}, average timestep {dtavg:.10f}s"
        )
    if comm is not None:
        res_tys = [i for o in comm.allgather(res_tys) for i in o]
        res_phi_hits = [i for o in comm.allgather(res_phi_hits) for i in o]
    return res_tys, res_phi_hits


def _translate_stopping_criteria_to_jax(stopping_criteria: list) -> tuple:
    """Translate user-level CPU stopping criteria into JAX-side dataclasses.

    The public :mod:`simsopt.field.tracing` stopping-criterion classes
    are thin :mod:`simsoptpp` subclasses whose ``operator()`` lives in
    C++. The JAX driver in :mod:`simsopt.jax_core.tracing` cannot call
    those C++ predicates inside a ``jax.lax.while_loop``; instead we
    expose one frozen JAX dataclass per upstream class and route via
    isinstance dispatch. Levelset criteria are translated by rebuilding
    the signed-distance grid on the JAX side via
    :meth:`SurfaceClassifier.to_jax_classifier_fn`; the resulting closure
    captures concrete JAX arrays at host setup time and is queried inside
    ``jax.lax.while_loop`` through the
    :class:`~simsopt.jax_core.tracing.LevelsetStoppingCriterion` JAX
    dataclass. Unknown classes raise :class:`NotImplementedError` so the
    user always sees a precise failure mode rather than a silent fallback.

    The Python-level subclasses defined in this module mirror the
    constructor argument on the Python instance so the translator can
    read it without introspecting the bound C++ class. Raw ``sopp.``
    instances (no Python init) raise :class:`AttributeError` if their
    parameter cannot be recovered — the JAX path needs the user's
    threshold value to bake into its predicate.
    """

    from ..jax_core import tracing as jax_tracing

    def _attr(obj, name):
        if not hasattr(obj, name):
            raise NotImplementedError(
                f"JAX tracing path cannot read attribute '{name}' on "
                f"{type(obj).__name__}; use the simsopt.field.tracing "
                "Python wrapper class (it mirrors the constructor arg "
                "as a Python attribute) rather than the bound C++ "
                "``simsoptpp.*`` class directly."
            )
        return getattr(obj, name)

    translated = []
    for crit in stopping_criteria:
        if isinstance(crit, sopp.LevelsetStoppingCriterion):
            # Levelset criteria are wired through the JAX driver via
            # ``SurfaceClassifier.to_jax_classifier_fn()`` or the metadata
            # adapter registered for ``SurfaceClassifier.dist``.
            classifier_obj = getattr(crit, "_classifier", None)
            if classifier_obj is None:
                raise NotImplementedError(
                    "JAX tracing path cannot translate a raw "
                    "sopp.LevelsetStoppingCriterion: build the criterion "
                    "via simsopt.field.tracing.LevelsetStoppingCriterion("
                    "SurfaceClassifier(surface)) so the JAX-side interpolant "
                    "spec can be rebuilt from the classifier grid metadata."
                )
            if not hasattr(classifier_obj, "to_jax_classifier_fn"):
                raise NotImplementedError(
                    "JAX tracing path requires a SurfaceClassifier-derived "
                    "interpolant "
                    "as the LevelsetStoppingCriterion argument; received "
                    f"{type(classifier_obj).__name__}. The raw "
                    "sopp.RegularGridInterpolant3D path does not carry the "
                    "grid metadata needed to rebuild a JAX-side interpolant spec."
                )
            jax_classifier_fn = classifier_obj.to_jax_classifier_fn()
            translated.append(
                jax_tracing.LevelsetStoppingCriterion(classifier_fn=jax_classifier_fn)
            )
            continue
        if isinstance(crit, sopp.MinRStoppingCriterion):
            translated.append(
                jax_tracing.MinRStoppingCriterion(crit_r=float(_attr(crit, "crit_r")))
            )
            continue
        if isinstance(crit, sopp.MaxRStoppingCriterion):
            translated.append(
                jax_tracing.MaxRStoppingCriterion(crit_r=float(_attr(crit, "crit_r")))
            )
            continue
        if isinstance(crit, sopp.MinZStoppingCriterion):
            translated.append(
                jax_tracing.MinZStoppingCriterion(crit_z=float(_attr(crit, "crit_z")))
            )
            continue
        if isinstance(crit, sopp.MaxZStoppingCriterion):
            translated.append(
                jax_tracing.MaxZStoppingCriterion(crit_z=float(_attr(crit, "crit_z")))
            )
            continue
        if isinstance(crit, sopp.ToroidalTransitStoppingCriterion):
            translated.append(
                jax_tracing.ToroidalTransitStoppingCriterion(
                    max_transits=float(_attr(crit, "max_transits"))
                )
            )
            continue
        if isinstance(crit, sopp.IterationStoppingCriterion):
            translated.append(
                jax_tracing.IterStoppingCriterion(max_iter=int(_attr(crit, "max_iter")))
            )
            continue
        if isinstance(crit, sopp.MinToroidalFluxStoppingCriterion):
            translated.append(
                jax_tracing.MinToroidalFluxStoppingCriterion(
                    min_s=float(_attr(crit, "min_s"))
                )
            )
            continue
        if isinstance(crit, sopp.MaxToroidalFluxStoppingCriterion):
            translated.append(
                jax_tracing.MaxToroidalFluxStoppingCriterion(
                    max_s=float(_attr(crit, "max_s"))
                )
            )
            continue
        raise NotImplementedError(
            "JAX tracing path cannot translate stopping criterion of type "
            f"{type(crit).__name__}; supported classes are LevelsetStoppingCriterion, "
            "MinR/MaxR/MinZ/MaxZ/ToroidalTransit/Iteration/Min/MaxToroidalFlux."
        )
    return tuple(translated)


def _compute_fieldlines_jax(field, R0, Z0, tmax, tol, phis, stopping_criteria, comm):
    """JAX backend for :func:`compute_fieldlines`.

    Routes each initial point through
    :func:`simsopt.jax_core.tracing.trace_fieldline`. The driver is
    arc-length parametrised (the JAX RHS is ``B / |B|`` rather than the
    upstream ``B``); the wrapper reproduces the upstream return shape
    ``(res_tys, res_phi_hits)`` so call sites are agnostic to the
    backend choice.

    Phi-plane crossings (``phis`` argument) and translated stopping
    criteria are supported through the JAX driver's fixed-shape
    ``phi_hits`` buffer; the CPU stopping-criterion objects are
    translated to JAX dataclasses via isinstance dispatch in
    :func:`_translate_stopping_criteria_to_jax`. Unsupported criterion
    types raise :class:`NotImplementedError`.

    MPI ``comm`` uses the same host-level contiguous split/gather as
    the CPU wrapper. No compiled cross-rank collective is introduced.
    """
    import jax.numpy as jnp

    from ..jax_core.tracing import FieldlineTracingSpec, trace_fieldline

    field_fn = _require_jax_field_B(field)

    jax_stopping_criteria = _translate_stopping_criteria_to_jax(stopping_criteria)
    if len(phis) > 0:
        phis_arr = jnp.asarray(list(phis), dtype=jnp.float64)
    else:
        phis_arr = None

    # Static step budget for the JAX while-loop trajectory carry. We
    # size this with a generous margin over the controller's expected
    # step count (the lane gates this at ``step_count_max_ratio=1.25``
    # versus the CPU oracle); 4000 covers the contract fixtures and
    # leaves headroom for the controller to converge on stiff fields.
    max_steps = 4000
    # Phi-hits buffer cap. The C++ oracle records every crossing; this
    # cap covers a few hundred toroidal transits across all phi planes
    # plus the stopping-criterion epilogue. The JAX core counts every
    # detected crossing, and this wrapper rejects overflowing results.
    max_phi_hits = 4096
    nlines = len(R0)
    res_tys = []
    res_phi_hits = []
    R0_arr = np.asarray(R0, dtype=np.float64)
    Z0_arr = np.asarray(Z0, dtype=np.float64)
    first, last = parallel_loop_bounds(comm, nlines)
    for i in range(first, last):
        spec = FieldlineTracingSpec(
            tmax=float(tmax),
            rtol=float(tol),
            atol=float(tol),
            max_steps=max_steps,
            max_phi_hits=max_phi_hits,
        )
        y0 = jnp.asarray([float(R0_arr[i]), 0.0, float(Z0_arr[i])], dtype=jnp.float64)
        result = trace_fieldline(
            spec,
            y0,
            field_fn,
            phis=phis_arr,
            stopping_criteria=jax_stopping_criteria,
        )
        traj = np.asarray(result.trajectory, dtype=np.float64)
        mask = np.asarray(result.mask, dtype=bool)
        live = traj[mask]
        res_tys.append(live)
        res_phi_hits.append(
            _event_hits_prefix(
                result.phi_hits,
                result.phi_hits_count,
                context="JAX fieldline tracing",
            )
        )
        status = int(result.status)
        if status > 0:
            logger.debug(
                f"{i + 1:3d}/{nlines}, JAX fieldline status={status} "
                f"t_final={float(result.t_final)}, "
                f"steps_taken={int(result.steps_taken)}"
            )
        elif status < 0:
            logger.debug(
                f"{i + 1:3d}/{nlines}, JAX fieldline stopped by criterion "
                f"index {-1 - status}, t_final={float(result.t_final)}"
            )
        elif live.shape[0] > 0:
            dtavg = live[-1, 0] / max(live.shape[0], 1)
            logger.debug(
                f"{i + 1:3d}/{nlines}, t_final={live[-1, 0]}, "
                f"average timestep {dtavg:.10f}s (JAX backend)"
            )
    res_tys = _allgather_flat(comm, res_tys)
    res_phi_hits = _allgather_flat(comm, res_phi_hits)
    return res_tys, res_phi_hits


def particles_to_vtk(res_tys, filename):
    """
    Export particle tracing or field lines to a vtk file.
    Expects that the xyz positions can be obtained by ``xyz[:, 1:4]``.
    """
    from pyevtk.hl import polyLinesToVTK

    x = np.concatenate([xyz[:, 1] for xyz in res_tys])
    y = np.concatenate([xyz[:, 2] for xyz in res_tys])
    z = np.concatenate([xyz[:, 3] for xyz in res_tys])
    ppl = np.asarray([xyz.shape[0] for xyz in res_tys])
    data = np.concatenate(
        [i * np.ones((res_tys[i].shape[0],)) for i in range(len(res_tys))]
    )
    polyLinesToVTK(filename, x, y, z, pointsPerLine=ppl, pointData={"idx": data})


class LevelsetStoppingCriterion(sopp.LevelsetStoppingCriterion):
    r"""
    Based on a scalar function :math:`f:R^3\to R`, this criterion checks whether
    :math:`f(x, y, z) < 0` and stops the iteration once this is true.

    The idea is to use this for example with signed distance functions to a surface.
    """

    def __init__(self, classifier):
        assert isinstance(classifier, SurfaceClassifier) or isinstance(
            classifier, sopp.RegularGridInterpolant3D
        )
        # Retain the Python-side classifier so the JAX tracing translator
        # can request a JAX-traceable closure via
        # ``SurfaceClassifier.to_jax_classifier_fn()``. A raw
        # ``SurfaceClassifier.dist`` interpolant is mapped back to a
        # metadata adapter; unrelated raw interpolants remain CPU-only
        # because they do not carry enough grid metadata for JAX rebuild.
        if isinstance(classifier, sopp.RegularGridInterpolant3D):
            self._classifier = _surface_classifier_from_interpolant(classifier)
            if self._classifier is None:
                self._classifier = classifier
        else:
            self._classifier = classifier
        if isinstance(classifier, SurfaceClassifier):
            sopp.LevelsetStoppingCriterion.__init__(self, classifier.dist)
        else:
            sopp.LevelsetStoppingCriterion.__init__(self, classifier)


class MinToroidalFluxStoppingCriterion(sopp.MinToroidalFluxStoppingCriterion):
    """
    Stop the iteration once a particle falls below a critical value of
    ``s``, the normalized toroidal flux. This :class:`StoppingCriterion` is
    important to use when tracing particles in flux coordinates, as the poloidal
    angle becomes ill-defined at the magnetic axis. This should only be used
    when tracing trajectories in a flux coordinate system (i.e., :class:`trace_particles_boozer`).

    Usage:

    .. code-block::

        stopping_criteria=[MinToroidalFluxStopingCriterion(s)]

    where ``s`` is the value of the minimum normalized toroidal flux.
    """

    def __init__(self, min_s):
        sopp.MinToroidalFluxStoppingCriterion.__init__(self, min_s)
        # Python-side mirror of the C++ private member so the JAX
        # translator can read the value without re-introspecting the
        # bound C++ class.
        self.min_s = float(min_s)


class MaxToroidalFluxStoppingCriterion(sopp.MaxToroidalFluxStoppingCriterion):
    """
    Stop the iteration once a particle falls above a critical value of
    ``s``, the normalized toroidal flux. This should only be used when tracing
    trajectories in a flux coordinate system (i.e., :class:`trace_particles_boozer`).

    Usage:

    .. code-block::

        stopping_criteria=[MaxToroidalFluxStopingCriterion(s)]

    where ``s`` is the value of the maximum normalized toroidal flux.
    """

    def __init__(self, max_s):
        sopp.MaxToroidalFluxStoppingCriterion.__init__(self, max_s)
        self.max_s = float(max_s)


class ToroidalTransitStoppingCriterion(sopp.ToroidalTransitStoppingCriterion):
    """
    Stop the iteration once the maximum number of toroidal transits is reached.

    Usage:

    .. code-block::

        stopping_criteria=[ToroidalTransitStoppingCriterion(ntransits,flux)]

    where ``ntransits`` is the maximum number of toroidal transits and ``flux``
    is a boolean indicating whether tracing is being performed in a flux coordinate system.
    """

    def __init__(self, max_transits, flux):
        sopp.ToroidalTransitStoppingCriterion.__init__(self, max_transits, flux)
        self.max_transits = float(max_transits)
        self.flux = bool(flux)


class IterationStoppingCriterion(sopp.IterationStoppingCriterion):
    """
    Stop the iteration once the maximum number of iterations is reached.
    """

    def __init__(self, max_iter):
        sopp.IterationStoppingCriterion.__init__(self, max_iter)
        self.max_iter = int(max_iter)


class MinRStoppingCriterion(sopp.MinRStoppingCriterion):
    """
    Stop the iteration once a particle falls below a critical value of
    ``R``, the radial cylindrical coordinate.

    Usage:

    .. code-block::

        stopping_criteria=[MinRStopingCriterion(crit_r)]

    where ``crit_r`` is the value of the critical coordinate.
    """

    def __init__(self, crit_r):
        sopp.MinRStoppingCriterion.__init__(self, crit_r)
        self.crit_r = float(crit_r)


class MinZStoppingCriterion(sopp.MinZStoppingCriterion):
    """
    Stop the iteration once a particle falls below a critical value of
    ``Z``, the cylindrical vertical coordinate.

    Usage:

    .. code-block::

        stopping_criteria=[MinZStopingCriterion(crit_z)]

    where ``crit_z`` is the value of the critical coordinate.
    """

    def __init__(self, crit_z):
        sopp.MinZStoppingCriterion.__init__(self, crit_z)
        self.crit_z = float(crit_z)


class MaxRStoppingCriterion(sopp.MaxRStoppingCriterion):
    """
    Stop the iteration once a particle goes above a critical value of
    ``R``, the radial cylindrical coordinate.

    Usage:

    .. code-block::

        stopping_criteria=[MaxRStopingCriterion(crit_r)]

    where ``crit_r`` is the value of the critical coordinate.
    """

    def __init__(self, crit_r):
        sopp.MaxRStoppingCriterion.__init__(self, crit_r)
        self.crit_r = float(crit_r)


class MaxZStoppingCriterion(sopp.MaxZStoppingCriterion):
    """
    Stop the iteration once a particle gove above a critical value of
    ``Z``, the cylindrical vertical coordinate.

    Usage:

    .. code-block::

        stopping_criteria=[MaxZStopingCriterion(crit_z)]

    where ``crit_z`` is the value of the critical coordinate.
    """

    def __init__(self, crit_z):
        sopp.MaxZStoppingCriterion.__init__(self, crit_z)
        self.crit_z = float(crit_z)


def plot_poincare_data(
    fieldlines_phi_hits,
    phis,
    filename,
    mark_lost=False,
    aspect="equal",
    dpi=300,
    xlims=None,
    ylims=None,
    surf=None,
    s=2,
    marker="o",
):
    """
    Create a poincare plot. Usage:

    .. code-block::

        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')

    Requires matplotlib to be installed.

    """
    import matplotlib.pyplot as plt
    from math import ceil, sqrt

    nrowcol = ceil(sqrt(len(phis)))
    plt.figure()
    fig, axs = plt.subplots(nrowcol, nrowcol, figsize=(8, 5))
    for ax in axs.ravel():
        ax.set_aspect(aspect)
    color = None
    for i in range(len(phis)):
        row = i // nrowcol
        col = i % nrowcol
        if i != len(phis) - 1:
            axs[row, col].set_title(
                f"$\\phi = {phis[i] / np.pi:.2f}\\pi$ ", loc="left", y=0.0
            )
        else:
            axs[row, col].set_title(
                f"$\\phi = {phis[i] / np.pi:.2f}\\pi$ ", loc="right", y=0.0
            )
        if row == nrowcol - 1:
            axs[row, col].set_xlabel("$r$")
        if col == 0:
            axs[row, col].set_ylabel("$z$")
        if col == 1:
            axs[row, col].set_yticklabels([])
        if xlims is not None:
            axs[row, col].set_xlim(xlims)
        if ylims is not None:
            axs[row, col].set_ylim(ylims)
        for j in range(len(fieldlines_phi_hits)):
            lost = fieldlines_phi_hits[j][-1, 1] < 0
            if mark_lost:
                color = "r" if lost else "g"
            data_this_phi = fieldlines_phi_hits[j][
                np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :
            ]
            if data_this_phi.size == 0:
                continue
            r = np.sqrt(data_this_phi[:, 2] ** 2 + data_this_phi[:, 3] ** 2)
            axs[row, col].scatter(
                r, data_this_phi[:, 4], marker=marker, s=s, linewidths=0, c=color
            )

        plt.rc("axes", axisbelow=True)
        axs[row, col].grid(True, linewidth=0.5)

        # if passed a surface, plot the plasma surface outline
        if surf is not None:
            cross_section = surf.cross_section(phi=phis[i] / (2.0 * np.pi))
            r_interp = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
            z_interp = cross_section[:, 2]
            axs[row, col].plot(r_interp, z_interp, linewidth=1, c="k")

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.close()
