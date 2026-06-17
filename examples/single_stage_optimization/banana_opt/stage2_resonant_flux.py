"""
Static resonant reweighting of the Stage-2 flux spectrum (2026-06-11
formulation audit 8, active half).

Motivation
----------
Islands at a rational surface ``iota = p/q`` are driven by the RESONANT
Fourier harmonics of the normal-field residual ``B.n`` in straight-field-line
angles: modes ``(m, n)`` with ``n/m = p/q``. The stock quadratic-flux
objective weights every harmonic equally, so the optimizer happily trades a
tiny broadband improvement for resonant content that seeds island chains at
the campaign's target iota. This module adds a STATIC (computed once at
setup, never updated in-loop -- no Boozer-trust risk) additive spectral
penalty

.. math::
    J_{res} = \\sum_{(a,b) \\in \\mathcal{R}} | \\hat f_{ab} |^2,
    \\qquad \\hat f = \\frac{1}{N_\\phi N_\\theta} \\, \\mathrm{FFT2}(B \\cdot \\hat n),

over the FFT bins :math:`\\mathcal{R}` marked resonant for the q <= 13
rationals within a window ``delta`` of the target iota. Combined as
``J_total = J_flux + w_res * J_res`` this multiplies the resonant L2 spectral
content by ``(1 + w_res / s)`` (``s`` the area normalisation ratio between
the two terms) while leaving every other mode untouched, so the limit
``w_res -> 0`` recovers the stock objective EXACTLY (the term is never even
constructed on the default path).

Normalisation: with the ``1/(N_phi N_theta)`` forward scaling a perturbation
``B.n = A cos(2 pi (m theta_hat - n phi_hat))`` contributes ``J_res = A^2/2``
(two Hermitian bins at ``A/2`` each), independent of grid resolution. Units
are Tesla^2 (``B`` dotted with the UNIT normal; no area weighting).

Angle-convention caveat (READ BEFORE TRUSTING THE MODE MAP)
-----------------------------------------------------------
The surface quadrature angles are the VMEC-convention angles of
``SurfaceRZFourier.from_wout(range="full torus")``: ``theta`` is the VMEC
poloidal angle of the Fourier representation, ``phi`` the geometric
(cylindrical) toroidal angle. These are NOT straight-field-line (Boozer)
angles. The resonance condition ``n/m = iota`` is exact only in
straight-field-line angles; in VMEC angles the resonant content at fixed
``n`` leaks into neighbouring ``m`` sidebands through the lambda
stream-function. This is acceptable for island-drive suppression near the
target resonance because (i) lambda is small for the low-shear vacuum
equilibria of this campaign, so the dominant projection of the
island-driving harmonic stays at ``(m, n)``, and (ii) the delta-window marks
the harmonics of ALL nearby rationals, partially covering the sideband
leakage. For strict island-width control a Boozer-angle mapping on the
quadrature surface would be required; the Stage-2 artifacts do not carry
one, so this module deliberately stops at the static VMEC-angle
approximation and documents it.

Sign convention: the mode basis is ``exp(i (m theta - n phi))`` with
resonance at ``n/m = +iota_target`` (signed), matching VMEC's
``cos(m theta - n phi)`` mode labelling and VMEC's signed iota. The
convention is pinned self-consistently by the helicity-injection unit tests;
it has NOT been verified against a real artifact's measured ``B.n`` spectrum
(flip the sign of ``iota_target`` if such a check ever shows the opposite
helicity carries the island drive).

Stellarator/NFP symmetry: a symmetry-expanded coil set on an NFP-symmetric
surface produces ``B.n`` content only at toroidal mode numbers ``n`` that
are multiples of NFP, so only those harmonics are marked (penalising the
identically-zero non-NFP bins would be a silent no-op). For NFP=5 and the
campaign's iota ~ 0.14 rationals this pushes the first resonant harmonic to
high m (e.g. 1/7 -> (m, n) = (35, 5)); the builder raises loudly when the
quadrature grid cannot represent any marked harmonic instead of silently
penalising nothing.

No-fallback contract: every misconfiguration (missing iota target, empty
rational window, q > 13 request, grid too coarse, non-uniform/partial-torus
quadrature grid) raises ``ValueError`` at setup. Nothing degrades silently.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

__all__ = [
    "MAX_RESONANT_DENOMINATOR",
    "ResonantFluxPenalty",
    "build_resonant_mode_mask",
    "build_stage2_resonant_flux_penalty",
    "enumerate_resonant_rationals",
    "resonant_mode_power",
    "resonant_mode_power_dbn",
]

# Campaign-band denominator cutoff. The original q<=8 cap was calibrated for
# the old iota~0.30 operating point; the current iota~0.08-0.13 band needs
# q<=13 to include the live 1/9..1/13 threats within delta=0.02.
MAX_RESONANT_DENOMINATOR = 13


def enumerate_resonant_rationals(iota_target, delta, q_max):
    """Enumerate the lowest-terms rationals ``p/q`` with ``q <= q_max`` and
    ``|p/q - iota_target| <= delta``, sorted ascending.

    Raises ``ValueError`` on a non-finite/missing target, non-positive or
    non-finite ``delta``, or ``q_max`` outside ``[1, MAX_RESONANT_DENOMINATOR]``.
    May legitimately return an empty tuple (no rational in the window); the
    weight-gated builder turns that into a loud error.
    """
    if iota_target is None:
        raise ValueError(
            "resonant reweighting requires an iota target; got None "
            "(pass --stage2-resonant-iota-target or --stage2-iota-target)."
        )
    iota = float(iota_target)
    if not math.isfinite(iota):
        raise ValueError(f"resonant iota target must be finite; got {iota!r}.")
    window = float(delta)
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError(
            f"resonant rational window delta must be finite and > 0; got {window!r}."
        )
    q_cap = int(q_max)
    if q_cap < 1:
        raise ValueError(f"resonant q_max must be >= 1; got {q_cap}.")
    if q_cap > MAX_RESONANT_DENOMINATOR:
        raise ValueError(
            f"resonant q_max must be <= {MAX_RESONANT_DENOMINATOR} "
            f"(campaign-band island relevance cutoff); got {q_cap}."
        )
    found = set()
    for q in range(1, q_cap + 1):
        p_lo = math.ceil((iota - window) * q)
        p_hi = math.floor((iota + window) * q)
        for p in range(p_lo, p_hi + 1):
            rational = Fraction(p, q)
            if rational.denominator == q:  # lowest terms only (dedupes 2/4 vs 1/2)
                found.add(rational)
    return tuple(sorted(found))


def build_resonant_mode_mask(nphi, ntheta, nfp, rationals):
    """Boolean FFT-bin mask of shape ``(nphi, ntheta)`` marking the resonant
    harmonics of ``rationals`` representable on the full-torus grid.

    For each rational ``p/q`` the resonant physical modes are
    ``(m, n) = s * (q, p)`` for integer ``s >= 1`` restricted to ``n`` a
    multiple of ``nfp`` (the only toroidal mode numbers an NFP-symmetric
    coil set populates) and to strictly sub-Nyquist ``m < ntheta/2``,
    ``|n| < nphi/2``. Each mode marks both Hermitian-partner bins of
    ``np.fft.fft2`` applied over the ``(phi, theta)`` axes:
    ``exp(i (m theta - n phi))`` lives at ``(a, b) = (-n mod nphi, m mod ntheta)``
    and its conjugate at ``(n mod nphi, -m mod ntheta)``.

    Raises ``ValueError`` when the rational set is empty or when no harmonic
    is representable on the grid (the penalty would be a silent no-op).
    """
    nphi_i = int(nphi)
    ntheta_i = int(ntheta)
    nfp_i = int(nfp)
    if nphi_i < 1 or ntheta_i < 1:
        raise ValueError(f"grid dims must be positive; got nphi={nphi_i}, ntheta={ntheta_i}.")
    if nfp_i < 1:
        raise ValueError(f"nfp must be >= 1; got {nfp_i}.")
    rationals = tuple(rationals)
    if not rationals:
        raise ValueError(
            "resonant mode mask requested with an empty rational set; "
            "widen --stage2-resonant-delta or fix the iota target."
        )
    m_nyq = (ntheta_i - 1) // 2
    n_nyq = (nphi_i - 1) // 2
    mask = np.zeros((nphi_i, ntheta_i), dtype=bool)
    smallest_required = None  # (m, |n|) of the lowest unrepresentable first harmonic
    for rational in rationals:
        p = rational.numerator
        q = rational.denominator
        stride = 1 if p == 0 else nfp_i // math.gcd(abs(p), nfp_i)
        first_m = q * stride
        first_n = abs(p) * stride
        if smallest_required is None or first_m < smallest_required[0]:
            smallest_required = (first_m, first_n)
        s = stride
        while q * s <= m_nyq and abs(p) * s <= n_nyq:
            m = q * s
            n = p * s
            mask[(-n) % nphi_i, m % ntheta_i] = True
            mask[n % nphi_i, (-m) % ntheta_i] = True
            s += stride
    if not mask.any():
        first_m, first_n = smallest_required
        raise ValueError(
            "no resonant harmonic of the selected rationals "
            f"{[str(r) for r in rationals]} is representable on the "
            f"nphi={nphi_i} x ntheta={ntheta_i} grid with nfp={nfp_i}: the "
            f"lowest harmonic needs (m, |n|) = ({first_m}, {first_n}), i.e. "
            f"ntheta >= {2 * first_m + 1} and nphi >= {2 * first_n + 1}. "
            "Raise the surface quadrature resolution; refusing to add a "
            "penalty that would be identically zero."
        )
    return mask


def resonant_mode_power(bn_grid, mode_mask):
    """Resonant spectral power ``sum_{mask} |FFT2(bn)/(Nphi*Ntheta)|^2`` of a
    real ``(nphi, ntheta)`` grid of ``B . n_hat`` values. Pure kernel."""
    bn = np.asarray(bn_grid, dtype=np.float64)
    mask = np.asarray(mode_mask)
    if bn.ndim != 2 or bn.shape != mask.shape:
        raise ValueError(
            f"bn grid shape {bn.shape} must match mode mask shape {mask.shape}."
        )
    coeffs = np.fft.fft2(bn) / bn.size
    return float(np.sum(np.abs(coeffs[mask]) ** 2))


def resonant_mode_power_dbn(bn_grid, mode_mask):
    """Exact gradient of :func:`resonant_mode_power` w.r.t. the grid values.

    Adjoint of the same FFT: ``dJ/dbn = (2/N) Re FFT2(mask * conj(FFT2(bn)/N))``.
    """
    bn = np.asarray(bn_grid, dtype=np.float64)
    mask = np.asarray(mode_mask)
    if bn.ndim != 2 or bn.shape != mask.shape:
        raise ValueError(
            f"bn grid shape {bn.shape} must match mode mask shape {mask.shape}."
        )
    scale = 1.0 / bn.size
    coeffs = np.fft.fft2(bn) * scale
    masked = np.where(mask, np.conj(coeffs), 0.0 + 0.0j)
    return 2.0 * scale * np.real(np.fft.fft2(masked))


class ResonantFluxPenalty(Optimizable):
    r"""
    Additive static spectral penalty on the resonant Fourier content of
    ``B \cdot \hat n`` over the surface quadrature grid (audit 8).

    Same dependency contract as :class:`simsopt.objectives.SquaredFlux`: the
    coil field is the optimizable parent, the (fixed in Stage-2) surface is a
    recompute dependency only, and ``dJ`` flows through ``field.B_vjp``.
    """

    def __init__(self, surface, field, mode_mask):
        mask = np.asarray(mode_mask)
        if mask.dtype != np.bool_:
            raise ValueError(f"mode mask must be boolean; got dtype {mask.dtype}.")
        grid_shape = surface.normal().shape[:2]
        if mask.shape != grid_shape:
            raise ValueError(
                f"mode mask shape {mask.shape} must match the surface "
                f"quadrature grid {grid_shape}."
            )
        if not mask.any():
            raise ValueError(
                "mode mask marks no bins; the resonant penalty would be "
                "identically zero. Refusing the silent no-op."
            )
        self.surface = surface
        self.field = field
        self.mode_mask = np.ascontiguousarray(mask)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])
        self.add_recompute_dependency(self.surface)
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        xyz = np.ascontiguousarray(self.surface.gamma().reshape((-1, 3)))
        self.field.set_points(xyz)

    def _bn_unitn(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1.0 / absn)[:, :, None]
        b = self.field.B().reshape(n.shape)
        return np.sum(b * unitn, axis=2), unitn

    def J(self):
        bn, _ = self._bn_unitn()
        return resonant_mode_power(bn, self.mode_mask)

    @derivative_dec
    def dJ(self):
        bn, unitn = self._bn_unitn()
        dbn = resonant_mode_power_dbn(bn, self.mode_mask)
        d_jd_b = (dbn[..., None] * unitn).reshape((-1, 3))
        return self.field.B_vjp(d_jd_b)

    return_fn_map = {'J': J, 'dJ': dJ}


def _validate_full_torus_uniform_grid(surface):
    """The FFT mode map is only valid on a uniform full-torus grid in both
    angles (``range="full torus"``); raise loudly otherwise."""
    qp_phi = np.asarray(surface.quadpoints_phi)
    qp_theta = np.asarray(surface.quadpoints_theta)
    expected_phi = np.arange(qp_phi.size) / qp_phi.size
    expected_theta = np.arange(qp_theta.size) / qp_theta.size
    if not (
        np.allclose(qp_phi, expected_phi, rtol=0.0, atol=1e-13)
        and np.allclose(qp_theta, expected_theta, rtol=0.0, atol=1e-13)
    ):
        raise ValueError(
            "resonant reweighting requires a uniform full-torus quadrature "
            "grid (surface range='full torus'); the surface quadrature points "
            "do not match arange(N)/N in both angles."
        )


def build_stage2_resonant_flux_penalty(surface, field, *, iota_target, delta, q_max):
    """Weight-gated audit-8 entry point: enumerate the rationals, build the
    static FFT-bin mask for ``surface`` and return
    ``(ResonantFluxPenalty, rationals)``.

    Raises ``ValueError`` on any misconfiguration (missing target, empty
    rational window, q > 8, partial-torus grid, grid too coarse).
    """
    rationals = enumerate_resonant_rationals(iota_target, delta, q_max)
    if not rationals:
        raise ValueError(
            f"no rationals p/q with q <= {int(q_max)} lie within "
            f"+/-{float(delta)} of iota target {float(iota_target)}; a nonzero "
            "resonant flux weight would be a silent no-op. Widen "
            "--stage2-resonant-delta or fix --stage2-resonant-iota-target."
        )
    _validate_full_torus_uniform_grid(surface)
    mask = build_resonant_mode_mask(
        len(surface.quadpoints_phi),
        len(surface.quadpoints_theta),
        surface.nfp,
        rationals,
    )
    return ResonantFluxPenalty(surface, field, mask), rationals
