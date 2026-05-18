"""JAX-backed wrapper that freezes an upstream ``BoozerRadialInterpolant``.

``BoozerRadialInterpolantJAX`` mirrors the public ``BoozerMagneticField``
surface (``set_points``, ``modB``, ``K``, ``nu``, ``R``, ``Z``, ``iota``,
``G``, ``I``, ``psip`` and first-derivative bundles) while routing the
field-evaluation hot path through immutable JAX state captured at
construction time from an existing CPU ``BoozerRadialInterpolant``.

Architectural notes (item 33):

- This wrapper does **not** inherit from ``sopp.BoozerMagneticField``.
- It does **not** rewrite the upstream class. Construction reads the
  already-built ``InterpolatedUnivariateSpline`` objects and translates
  them into ``scipy.interpolate.PPoly`` coefficients so the JAX
  evaluator can compute spline values without leaving the compiled
  path.
- Frozen state semantics: mutating the wrapped CPU instance after
  construction does not propagate to the JAX wrapper. The wrapper
  exposes a fresh ``Optimizable`` node with no DOFs of its own.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .._core.optimizable import Optimizable
from .._core.json import GSONDecoder
from ..jax_core._math_utils import as_jax_float64 as _as_jax_float64
from ..jax_core.boozer_analytic import (
    BoozerAnalyticFrozenState,
    _eval_G as _eval_analytic_G,
    _eval_I as _eval_analytic_I,
    _eval_K as _eval_analytic_K,
    _eval_dGds as _eval_analytic_dGds,
    _eval_dIds as _eval_analytic_dIds,
    _eval_dKdtheta as _eval_analytic_dKdtheta,
    _eval_dKdzeta as _eval_analytic_dKdzeta,
    _eval_diotads as _eval_analytic_diotads,
    _eval_dmodBds as _eval_analytic_dmodBds,
    _eval_dmodBdtheta as _eval_analytic_dmodBdtheta,
    _eval_dmodBdzeta as _eval_analytic_dmodBdzeta,
    _eval_iota as _eval_analytic_iota,
    _eval_modB as _eval_analytic_modB,
    _eval_psip as _eval_analytic_psip,
    freeze_boozer_analytic_state,
)
from ..jax_core.boozer_radial_field import (
    BoozerRadialColumnBundle,
    BoozerRadialInterpolantFrozenState,
    _eval_G,
    _eval_I,
    _eval_K,
    _eval_K_from_columns,
    _eval_R,
    _eval_R_from_columns,
    _eval_Z,
    _eval_Z_from_columns,
    _eval_dGds,
    _eval_dIds,
    _eval_dKdtheta,
    _eval_dKdtheta_from_columns,
    _eval_dKdzeta,
    _eval_dKdzeta_from_columns,
    _eval_dRdtheta,
    _eval_dRdtheta_from_columns,
    _eval_dRds,
    _eval_dRds_from_columns,
    _eval_dRdzeta,
    _eval_dRdzeta_from_columns,
    _eval_dZdtheta,
    _eval_dZdtheta_from_columns,
    _eval_dZds,
    _eval_dZds_from_columns,
    _eval_dZdzeta,
    _eval_dZdzeta_from_columns,
    _eval_diotads,
    _eval_dmodBds,
    _eval_dmodBds_from_columns,
    _eval_dmodBdtheta,
    _eval_dmodBdtheta_from_columns,
    _eval_dmodBdzeta,
    _eval_dmodBdzeta_from_columns,
    _eval_dnuds,
    _eval_dnuds_from_columns,
    _eval_dnudtheta,
    _eval_dnudtheta_from_columns,
    _eval_dnudzeta,
    _eval_dnudzeta_from_columns,
    _eval_iota,
    _eval_modB,
    _eval_modB_from_columns,
    _eval_nu,
    _eval_nu_from_columns,
    _eval_psip,
    _eval_radial_columns,
    _frozen_state_from_host,
    _frozen_state_to_host,
    freeze_boozer_radial_state,
)
from ..jax_core.interpolated_boozer_field import (
    InterpolatedBoozerFieldFrozenState,
    _INTERP_EVALUATORS,
    build_spec_for_scalar as _interp_build_spec_for_scalar,
    freeze_interpolated_boozer_field_state,
)
from ..jax_core.regular_grid_interp import (
    RegularGridInterpolant3DSpec,
    UniformInterpolationRule as _jax_core_uniform_rule,
)

__all__ = [
    "BoozerAnalyticFrozenState",
    "BoozerAnalyticJAX",
    "BoozerRadialInterpolantFrozenState",
    "BoozerRadialInterpolantJAX",
    "InterpolatedBoozerFieldFrozenState",
    "InterpolatedBoozerFieldJAX",
    "freeze_boozer_analytic_state",
    "freeze_boozer_radial_state",
    "freeze_interpolated_boozer_field_state",
]


# ----------------------------------------------------------------------
# Public wrapper
# ----------------------------------------------------------------------


def _as_column(values: jax.Array) -> jax.Array:
    """Match the upstream ``_*_impl`` shape convention of ``(n, 1)``."""
    return values[:, None]


_RADIAL_COLUMN_EVALUATORS = {
    "modB": _eval_modB_from_columns,
    "dmodBds": _eval_dmodBds_from_columns,
    "dmodBdtheta": _eval_dmodBdtheta_from_columns,
    "dmodBdzeta": _eval_dmodBdzeta_from_columns,
    "K": _eval_K_from_columns,
    "dKdtheta": _eval_dKdtheta_from_columns,
    "dKdzeta": _eval_dKdzeta_from_columns,
    "nu": _eval_nu_from_columns,
    "dnuds": _eval_dnuds_from_columns,
    "dnudtheta": _eval_dnudtheta_from_columns,
    "dnudzeta": _eval_dnudzeta_from_columns,
    "R": _eval_R_from_columns,
    "dRds": _eval_dRds_from_columns,
    "dRdtheta": _eval_dRdtheta_from_columns,
    "dRdzeta": _eval_dRdzeta_from_columns,
    "Z": _eval_Z_from_columns,
    "dZds": _eval_dZds_from_columns,
    "dZdtheta": _eval_dZdtheta_from_columns,
    "dZdzeta": _eval_dZdzeta_from_columns,
    "psip": lambda _state, columns, _points: columns.psip,
    "G": lambda _state, columns, _points: columns.G,
    "I": lambda _state, columns, _points: columns.I,
    "iota": lambda _state, columns, _points: columns.iota,
    "dGds": lambda _state, columns, _points: columns.dGds,
    "dIds": lambda _state, columns, _points: columns.dIds,
    "diotads": lambda _state, columns, _points: columns.diotads,
}


class BoozerRadialInterpolantJAX(Optimizable):
    """JAX-backed wrapper that freezes an upstream ``BoozerRadialInterpolant``.

    Architectural note: this wrapper does **not** inherit from
    ``sopp.BoozerMagneticField``. It exposes the same public surface
    (``modB``, ``K``, ``nu``, ``R``, ``Z``, ``iota``, ``G``, ``I``,
    ``psip``, ``set_points``, ``get_points``, plus first-derivative
    bundles) while routing the field-evaluation hot path through
    immutable JAX state captured at construction time. State is frozen
    — modifying the wrapped CPU instance after construction does not
    propagate, including ``psi0`` changes that would require rebuilding
    upstream K splines. Construct a new wrapper after such CPU-side
    mutations.

    Args:
        upstream: an instance of
            :class:`simsopt.field.boozermagneticfield.BoozerRadialInterpolant`
            with splines already built (i.e. ``init_splines`` has run).
            ``compute_K`` must also have run unless ``upstream.no_K`` is
            True.
    """

    def __init__(self, upstream):
        Optimizable.__init__(self, x0=np.asarray([]))
        self._frozen_state = freeze_boozer_radial_state(upstream)
        self._psi0 = float(upstream.psi0)
        self._nfp = int(getattr(upstream.booz.bx, "nfp", 1))
        self._upstream = None
        self._points = jnp.zeros((0, 3), dtype=jnp.float64)
        self._cache: dict[str, jax.Array] = {}
        self._radial_columns_cache: BoozerRadialColumnBundle | None = None

    @classmethod
    def from_frozen_state(
        cls,
        frozen_state: BoozerRadialInterpolantFrozenState,
        *,
        psi0: float,
        nfp: int,
    ):
        wrapper = cls.__new__(cls)
        Optimizable.__init__(wrapper, x0=np.asarray([]))
        wrapper._frozen_state = frozen_state
        wrapper._psi0 = float(psi0)
        wrapper._nfp = int(nfp)
        wrapper._upstream = None
        wrapper._points = jnp.zeros((0, 3), dtype=jnp.float64)
        wrapper._cache = {}
        wrapper._radial_columns_cache = None
        return wrapper

    # ------------------------------------------------------------------
    # Points / cache management
    # ------------------------------------------------------------------

    @property
    def psi0(self) -> float:
        return self._psi0

    @property
    def stellsym(self) -> bool:
        return bool(self._frozen_state.stellsym)

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def no_K(self) -> bool:
        return bool(self._frozen_state.no_K)

    @property
    def frozen_state(self) -> BoozerRadialInterpolantFrozenState:
        return self._frozen_state

    def set_points(self, points):
        """Set the Boozer ``(s, theta, zeta)`` evaluation points.

        Returns ``self`` to match the upstream ``set_points`` signature.
        """
        arr = _as_jax_float64(np.asarray(points, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"points must have shape (n, 3); got shape={tuple(arr.shape)!r}"
            )
        self._points = arr
        self._cache.clear()
        self._radial_columns_cache = None
        return self

    def get_points(self) -> np.ndarray:
        return np.asarray(self._points)

    def get_points_ref(self) -> jax.Array:
        return self._points

    def clear_cached_properties(self):
        self._cache.clear()
        self._radial_columns_cache = None

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["frozen_state"] = _frozen_state_to_host(self._frozen_state)
        d["psi0"] = self._psi0
        d["nfp"] = self._nfp
        d["points"] = self.get_points()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        frozen_payload = decoder.process_decoded(
            d["frozen_state"], serial_objs_dict, recon_objs
        )
        points = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        wrapper = cls.from_frozen_state(
            _frozen_state_from_host(frozen_payload),
            psi0=float(d["psi0"]),
            nfp=int(d["nfp"]),
        )
        wrapper.set_points(points)
        return wrapper

    # ------------------------------------------------------------------
    # Field evaluators
    # ------------------------------------------------------------------

    def _radial_columns(self) -> BoozerRadialColumnBundle:
        cached = self._radial_columns_cache
        if cached is None:
            cached = _eval_radial_columns(self._frozen_state, self._points[:, 0])
            self._radial_columns_cache = cached
        return cached

    def _cached(self, name: str, fn) -> jax.Array:
        cached = self._cache.get(name)
        if cached is None:
            column_fn = _RADIAL_COLUMN_EVALUATORS.get(name)
            if column_fn is None:
                cached = fn(self._frozen_state, self._points)
            else:
                cached = column_fn(
                    self._frozen_state, self._radial_columns(), self._points
                )
            self._cache[name] = cached
        return cached

    def modB(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("modB", _eval_modB)))

    def dmodBdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBdtheta", _eval_dmodBdtheta)))

    def dmodBdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBdzeta", _eval_dmodBdzeta)))

    def dmodBds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBds", _eval_dmodBds)))

    def modB_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dmodBds", _eval_dmodBds))
        dtheta = np.asarray(self._cached("dmodBdtheta", _eval_dmodBdtheta))
        dzeta = np.asarray(self._cached("dmodBdzeta", _eval_dmodBdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def K(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("K", _eval_K)))

    def dKdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdtheta", _eval_dKdtheta)))

    def dKdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdzeta", _eval_dKdzeta)))

    def K_derivs(self) -> np.ndarray:
        dtheta = np.asarray(self._cached("dKdtheta", _eval_dKdtheta))
        dzeta = np.asarray(self._cached("dKdzeta", _eval_dKdzeta))
        return np.stack([dtheta, dzeta], axis=1)

    def nu(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("nu", _eval_nu)))

    def dnudtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dnudtheta", _eval_dnudtheta)))

    def dnudzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dnudzeta", _eval_dnudzeta)))

    def dnuds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dnuds", _eval_dnuds)))

    def nu_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dnuds", _eval_dnuds))
        dtheta = np.asarray(self._cached("dnudtheta", _eval_dnudtheta))
        dzeta = np.asarray(self._cached("dnudzeta", _eval_dnudzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def R(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("R", _eval_R)))

    def dRdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dRdtheta", _eval_dRdtheta)))

    def dRdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dRdzeta", _eval_dRdzeta)))

    def dRds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dRds", _eval_dRds)))

    def R_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dRds", _eval_dRds))
        dtheta = np.asarray(self._cached("dRdtheta", _eval_dRdtheta))
        dzeta = np.asarray(self._cached("dRdzeta", _eval_dRdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def Z(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("Z", _eval_Z)))

    def dZdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dZdtheta", _eval_dZdtheta)))

    def dZdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dZdzeta", _eval_dZdzeta)))

    def dZds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dZds", _eval_dZds)))

    def Z_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dZds", _eval_dZds))
        dtheta = np.asarray(self._cached("dZdtheta", _eval_dZdtheta))
        dzeta = np.asarray(self._cached("dZdzeta", _eval_dZdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def psip(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("psip", _eval_psip)))

    def G(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("G", _eval_G)))

    def I(self) -> np.ndarray:  # noqa: E743 — matches upstream API name
        return np.asarray(_as_column(self._cached("I", _eval_I)))

    def iota(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("iota", _eval_iota)))

    def dGds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dGds", _eval_dGds)))

    def dIds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dIds", _eval_dIds)))

    def diotads(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("diotads", _eval_diotads)))


# ----------------------------------------------------------------------
# BoozerAnalyticJAX — analytic Landreman-Sengupta field, pure JAX kernels
# ----------------------------------------------------------------------


class BoozerAnalyticJAX(Optimizable):
    """JAX-backed analytic Boozer field (Landreman & Sengupta, JPP 2018).

    Mirrors the public surface of
    :class:`simsopt.field.boozermagneticfield.BoozerAnalytic` (``set_points``,
    ``modB``, ``K``, ``G``, ``I``, ``iota``, ``psip``, derivative bundles)
    while routing the field-evaluation hot path through pure JAX kernels
    on an immutable frozen-state pytree.  This class does **not** inherit
    from ``sopp.BoozerMagneticField`` — it is a pure JAX-native sibling.

    Construction signature matches the CPU oracle exactly: ``(etabar, B0,
    N, G0, psi0, iota0, Bbar=1.0, I0=0.0, G1=0.0, I1=0.0, K1=0.0)``.

    Frozen-state semantics: the eleven scalar parameters are captured at
    construction time into an immutable ``BoozerAnalyticFrozenState``
    pytree.  Mutation requires constructing a new ``BoozerAnalyticJAX`` —
    there are no setters.
    """

    def __init__(
        self,
        etabar,
        B0,
        N,
        G0,
        psi0,
        iota0,
        Bbar=1.0,
        I0=0.0,
        G1=0.0,
        I1=0.0,
        K1=0.0,
    ):
        Optimizable.__init__(self, x0=np.asarray([]))
        self._frozen_state = freeze_boozer_analytic_state(
            etabar=etabar,
            B0=B0,
            N=N,
            G0=G0,
            psi0=psi0,
            iota0=iota0,
            Bbar=Bbar,
            I0=I0,
            G1=G1,
            I1=I1,
            K1=K1,
        )
        self._N_int = int(N)
        self._psi0_host = float(psi0)
        self._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        self._cache: dict[str, jax.Array] = {}

    @classmethod
    def from_frozen_state(
        cls,
        frozen_state: BoozerAnalyticFrozenState,
        *,
        N: int,
        psi0: float,
    ):
        """Build a wrapper directly from a pre-built frozen state.

        This bypasses scalar re-coercion and is useful for tests or
        downstream consumers that want to mutate one parameter without
        going through the full constructor.
        """
        wrapper = cls.__new__(cls)
        Optimizable.__init__(wrapper, x0=np.asarray([]))
        wrapper._frozen_state = frozen_state
        wrapper._N_int = int(N)
        wrapper._psi0_host = float(psi0)
        wrapper._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        wrapper._cache = {}
        return wrapper

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def psi0(self) -> float:
        return self._psi0_host

    @property
    def N(self) -> int:  # noqa: N802 — mirror CPU API
        return self._N_int

    @property
    def frozen_state(self) -> BoozerAnalyticFrozenState:
        return self._frozen_state

    # ------------------------------------------------------------------
    # Points / cache management
    # ------------------------------------------------------------------

    def set_points(self, points):
        """Set the Boozer ``(s, theta, zeta)`` evaluation points.

        Returns ``self`` to match the CPU ``set_points`` contract.
        """
        arr = _as_jax_float64(np.asarray(points, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"points must have shape (n, 3); got shape={tuple(arr.shape)!r}"
            )
        self._points = arr
        self._cache.clear()
        return self

    def get_points(self) -> np.ndarray:
        return np.asarray(self._points)

    def get_points_ref(self) -> jax.Array:
        return self._points

    def clear_cached_properties(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    # Field evaluators
    # ------------------------------------------------------------------

    def _cached(self, name: str, fn) -> jax.Array:
        cached = self._cache.get(name)
        if cached is None:
            cached = fn(self._frozen_state, self._points)
            self._cache[name] = cached
        return cached

    def modB(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("modB", _eval_analytic_modB)))

    def dmodBds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dmodBds", _eval_analytic_dmodBds)))

    def dmodBdtheta(self) -> np.ndarray:
        return np.asarray(
            _as_column(self._cached("dmodBdtheta", _eval_analytic_dmodBdtheta))
        )

    def dmodBdzeta(self) -> np.ndarray:
        return np.asarray(
            _as_column(self._cached("dmodBdzeta", _eval_analytic_dmodBdzeta))
        )

    def modB_derivs(self) -> np.ndarray:
        ds = np.asarray(self._cached("dmodBds", _eval_analytic_dmodBds))
        dtheta = np.asarray(self._cached("dmodBdtheta", _eval_analytic_dmodBdtheta))
        dzeta = np.asarray(self._cached("dmodBdzeta", _eval_analytic_dmodBdzeta))
        return np.stack([ds, dtheta, dzeta], axis=1)

    def K(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("K", _eval_analytic_K)))

    def dKdtheta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdtheta", _eval_analytic_dKdtheta)))

    def dKdzeta(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dKdzeta", _eval_analytic_dKdzeta)))

    def K_derivs(self) -> np.ndarray:
        dtheta = np.asarray(self._cached("dKdtheta", _eval_analytic_dKdtheta))
        dzeta = np.asarray(self._cached("dKdzeta", _eval_analytic_dKdzeta))
        return np.stack([dtheta, dzeta], axis=1)

    def G(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("G", _eval_analytic_G)))

    def dGds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dGds", _eval_analytic_dGds)))

    def I(self) -> np.ndarray:  # noqa: E743 — matches upstream API name
        return np.asarray(_as_column(self._cached("I", _eval_analytic_I)))

    def dIds(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("dIds", _eval_analytic_dIds)))

    def iota(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("iota", _eval_analytic_iota)))

    def diotads(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("diotads", _eval_analytic_diotads)))

    def psip(self) -> np.ndarray:
        return np.asarray(_as_column(self._cached("psip", _eval_analytic_psip)))


class InterpolatedBoozerFieldJAX(Optimizable):
    """JAX-native re-fit of a CPU :class:`BoozerMagneticField` on a regular grid.

    Mirrors the public surface of
    :class:`simsopt.field.boozermagneticfield.InterpolatedBoozerField`
    (``set_points``, ``modB``, ``K``, ``nu``, ``R``, ``Z``, ``G``, ``I``,
    ``iota``, ``psip``, and all first / second derivative bundles) while
    routing the field-evaluation hot path through pure JAX kernels on
    pre-fit :class:`InterpolatedBoozerFieldFrozenState` payloads.

    Construction signature exactly matches the CPU oracle:
    ``(field, degree, srange, thetarange, zetarange, extrapolate=True,
    nfp=1, stellsym=True)``.

    Architectural notes:

    - This wrapper does **not** inherit from ``sopp.BoozerMagneticField``
      or call into the C++ ``InterpolatedBoozerField`` class. It builds
      its own per-scalar interpolant set by sampling ``field``'s scalar
      getters on the regular grid.
    - Per-scalar interpolants are built **lazily** on the first call to
      each method, exactly mirroring the C++ template behaviour. The
      same base field may therefore be passed even if it only
      implements a subset of the 34 scalars: as long as the methods
      called on the wrapper map onto implemented getters on the base
      field, construction succeeds.
    - The wrapper exposes ``Optimizable`` with no DOFs of its own —
      mutating the wrapped CPU field after construction does NOT
      propagate to specs that have already been built. Newly-requested
      specs do sample the (possibly mutated) field state at the time of
      first request, just as the C++ template does.
    - The ``_simsopt_jax_native_field = True`` marker registers this
      class with the composition-strict-mode guard in
      :mod:`simsopt.field.magneticfield`.
    """

    _simsopt_jax_native_field = True

    def __init__(
        self,
        field,
        degree,
        srange,
        thetarange,
        zetarange,
        extrapolate: bool = True,
        nfp: int = 1,
        stellsym: bool = True,
        *,
        scalars: tuple[str, ...] | None = None,
    ):
        Optimizable.__init__(self, x0=np.asarray([]))
        self._field = field
        # Eagerly build the specified scalars at construction time.
        # ``scalars=None`` builds the full 34-scalar set (matches the
        # ``BoozerRadialInterpolant``-driven canonical use case). Pass a
        # tuple subset to match a base field that does not implement
        # every getter (e.g. ``BoozerAnalytic`` exposes only the 14
        # closed-form scalars).
        state = freeze_interpolated_boozer_field_state(
            field,
            degree=degree,
            srange=srange,
            thetarange=thetarange,
            zetarange=zetarange,
            extrapolate=extrapolate,
            nfp=nfp,
            stellsym=stellsym,
            scalars=scalars,
        )
        self._frozen_state = state
        self._lazy_specs: dict[str, RegularGridInterpolant3DSpec] = dict(state.specs)
        self._psi0_host = float(field.psi0)
        self._nfp = int(nfp)
        self._stellsym = bool(stellsym)
        self._extrapolate = bool(extrapolate)
        self._rule = _jax_core_uniform_rule(self._frozen_state.degree)
        self._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        self._cache: dict[str, jax.Array] = {}

    @classmethod
    def from_frozen_state(
        cls,
        frozen_state: InterpolatedBoozerFieldFrozenState,
        *,
        psi0: float,
        nfp: int | None = None,
    ):
        """Build a wrapper directly from a pre-built frozen state.

        ``nfp`` is read from ``frozen_state.nfp`` unless an explicit
        override is supplied. This keeps the metadata consistent with
        the underlying interpolant geometry. The resulting wrapper has
        no reference to a source ``field`` and therefore cannot build
        additional specs on demand — any scalar method absent from the
        frozen state will raise ``KeyError``.
        """
        wrapper = cls.__new__(cls)
        Optimizable.__init__(wrapper, x0=np.asarray([]))
        wrapper._field = None
        wrapper._frozen_state = frozen_state
        wrapper._lazy_specs = dict(frozen_state.specs)
        wrapper._psi0_host = float(psi0)
        wrapper._nfp = int(frozen_state.nfp if nfp is None else nfp)
        wrapper._stellsym = bool(frozen_state.stellsym)
        wrapper._extrapolate = bool(frozen_state.extrapolate)
        wrapper._rule = _jax_core_uniform_rule(frozen_state.degree)
        wrapper._points = _as_jax_float64(np.zeros((0, 3), dtype=np.float64))
        wrapper._cache = {}
        return wrapper

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def psi0(self) -> float:
        return self._psi0_host

    @property
    def nfp(self) -> int:
        return self._nfp

    @property
    def stellsym(self) -> bool:
        return self._stellsym

    @property
    def extrapolate(self) -> bool:
        return self._extrapolate

    @property
    def frozen_state(self) -> InterpolatedBoozerFieldFrozenState:
        return self._frozen_state

    # ------------------------------------------------------------------
    # Points / cache management
    # ------------------------------------------------------------------

    def set_points(self, points):
        """Set the Boozer ``(s, theta, zeta)`` evaluation points.

        Returns ``self`` to match the CPU ``set_points`` contract.
        """
        arr = _as_jax_float64(np.asarray(points, dtype=np.float64))
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"points must have shape (n, 3); got shape={tuple(arr.shape)!r}"
            )
        self._points = arr
        self._cache.clear()
        return self

    def get_points(self) -> np.ndarray:
        return np.asarray(self._points)

    def get_points_ref(self) -> jax.Array:
        return self._points

    def clear_cached_properties(self):
        self._cache.clear()

    # ------------------------------------------------------------------
    # Field evaluators
    # ------------------------------------------------------------------

    def _ensure_spec(self, name: str) -> None:
        """Lazy-build the per-scalar interpolant the first time it is read.

        Mirrors the C++ ``InterpolatedBoozerField`` lazy-build at header
        line 41-50 etc.: the interpolant is constructed and the base
        field is sampled only on the first call to the corresponding
        impl method. If the wrapper was built via
        :meth:`from_frozen_state` no base field reference is available,
        so unbuilt specs surface as ``KeyError``.
        """
        if name in self._lazy_specs:
            return
        if self._field is None:
            raise KeyError(
                f"spec for scalar {name!r} was not pre-fit and the wrapper "
                f"has no base field to lazy-fit against (likely built via "
                f"from_frozen_state). Available scalars: "
                f"{sorted(self._lazy_specs)}"
            )
        self._lazy_specs[name] = _interp_build_spec_for_scalar(
            self._field,
            scalar_name=name,
            rule=self._rule,
            s_range=self._frozen_state.s_range,
            theta_range=self._frozen_state.theta_range,
            zeta_range=self._frozen_state.zeta_range,
            extrapolate=self._frozen_state.extrapolate,
        )

    def _cached(self, name: str) -> jax.Array:
        cached = self._cache.get(name)
        if cached is None:
            self._ensure_spec(name)
            cached = _INTERP_EVALUATORS[name](
                self._frozen_state, self._lazy_specs, self._points
            )
            self._cache[name] = cached
        return cached

    # Flux-function scalars — all (N, 1) shape
    def psip(self) -> np.ndarray:
        return np.asarray(self._cached("psip"))

    def G(self) -> np.ndarray:
        return np.asarray(self._cached("G"))

    def I(self) -> np.ndarray:  # noqa: E743 — matches CPU API name
        return np.asarray(self._cached("I"))

    def iota(self) -> np.ndarray:
        return np.asarray(self._cached("iota"))

    def dGds(self) -> np.ndarray:
        return np.asarray(self._cached("dGds"))

    def dIds(self) -> np.ndarray:
        return np.asarray(self._cached("dIds"))

    def diotads(self) -> np.ndarray:
        return np.asarray(self._cached("diotads"))

    # modB family
    def modB(self) -> np.ndarray:
        return np.asarray(self._cached("modB"))

    def dmodBdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dmodBdtheta"))

    def dmodBdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dmodBdzeta"))

    def dmodBds(self) -> np.ndarray:
        return np.asarray(self._cached("dmodBds"))

    def modB_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("modB_derivs"))

    def d2modBdtheta2(self) -> np.ndarray:
        return np.asarray(self._cached("d2modBdtheta2"))

    def d2modBdzeta2(self) -> np.ndarray:
        return np.asarray(self._cached("d2modBdzeta2"))

    def d2modBdthetadzeta(self) -> np.ndarray:
        return np.asarray(self._cached("d2modBdthetadzeta"))

    # K family
    def K(self) -> np.ndarray:
        return np.asarray(self._cached("K"))

    def dKdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dKdtheta"))

    def dKdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dKdzeta"))

    def K_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("K_derivs"))

    # nu family
    def nu(self) -> np.ndarray:
        return np.asarray(self._cached("nu"))

    def dnudtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dnudtheta"))

    def dnudzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dnudzeta"))

    def dnuds(self) -> np.ndarray:
        return np.asarray(self._cached("dnuds"))

    def nu_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("nu_derivs"))

    # R family
    def R(self) -> np.ndarray:
        return np.asarray(self._cached("R"))

    def dRdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dRdtheta"))

    def dRdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dRdzeta"))

    def dRds(self) -> np.ndarray:
        return np.asarray(self._cached("dRds"))

    def R_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("R_derivs"))

    # Z family
    def Z(self) -> np.ndarray:
        return np.asarray(self._cached("Z"))

    def dZdtheta(self) -> np.ndarray:
        return np.asarray(self._cached("dZdtheta"))

    def dZdzeta(self) -> np.ndarray:
        return np.asarray(self._cached("dZdzeta"))

    def dZds(self) -> np.ndarray:
        return np.asarray(self._cached("dZds"))

    def Z_derivs(self) -> np.ndarray:
        return np.asarray(self._cached("Z_derivs"))
