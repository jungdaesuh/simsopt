"""Non-Boozer topology bridge metrics (Phase 3 of the Stage 2 / single-stage
Boozer handoff plan).

Two diagnostics, both pure-function and Boozer-independent:

- :func:`compute_helical_field_content_S_HEL`: spectral fraction of the
  on-surface ``|B|`` field that lives in helical (``n != 0``) modes. Scale-
  invariant; ``0`` for an axisymmetric field, ``1`` when all power is in
  helical modes. Used as a ranking signal and (in future, gated on gradient
  validation per plan line 308) as a ramped Stage 2 objective term.

- :func:`compute_fieldline_iota_proxy`: rotational-transform proxy derived
  from field-line tracing, with a convergence sub-study that fails closed
  when ``|iota_FL(2N) - iota_FL(N)| > 0.005`` (plan line 274). Designed for
  post-solve / on-checkpoint use only; plan line 270 ("evaluate post-solve
  or on checkpoint only, not every optimizer iteration") and line 283
  ("keep field-line iota as validation/truth signal, not the per-iteration
  objective") explicitly forbid use as a per-iteration objective.

Both metrics are persisted into the Stage 2 artifact via
:func:`boozer_topology_bridge_artifact_fields` so downstream reports can
rank hardware-clean donors without first running a Boozer solve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simsopt.field.tracing import (
    compute_fieldlines,
    compute_poloidal_transits,
    compute_toroidal_transits,
)
from topology_scorer import (
    build_stopping_criteria,
    midplane_seed_radii,
)


__all__ = [
    "FIELDLINE_IOTA_CONVERGENCE_TOL",
    "FieldlineIotaProxyResult",
    "HelicalFieldContentObjective",
    "boozer_topology_bridge_artifact_fields",
    "compute_fieldline_iota_proxy",
    "compute_helical_field_content_S_HEL",
    "compute_pre_boozer_topology_score",
    "compute_S_HEL_gradient_relative_error",
    "helical_field_content_from_modB_grid",
    "safe_compute_fieldline_iota_proxy",
    "safe_compute_helical_field_content_S_HEL",
]


# Plan line 274: ``|iota_FL(N=200) - iota_FL(N=100)| < 0.005``.
FIELDLINE_IOTA_CONVERGENCE_TOL = 0.005


@dataclass(frozen=True)
class FieldlineIotaProxyResult:
    """Diagnostic-only field-line iota proxy with convergence-gated trust.

    ``valid=True`` iff every surviving line's iota agreed within
    :data:`FIELDLINE_IOTA_CONVERGENCE_TOL` between trace-length levels.
    Callers must NOT use this as a per-iteration objective — plan lines
    270 and 283.
    """

    iota_proxy_mean: float | None
    iota_proxy_std: float | None
    iota_per_line: tuple[float, ...]
    n_lines_seeded: int
    n_lines_survived: int
    tmax: float
    valid: bool
    convergence_residual: float | None
    invalid_reason: str | None


def helical_field_content_from_modB_grid(modB: np.ndarray) -> float:
    """Compute ``S_HEL`` directly from a sampled ``|B|`` grid.

    ``modB`` has shape ``(nphi, ntheta)`` and must contain finite samples
    of the field magnitude on a uniform (phi, theta) grid covering one
    field period (or the full torus) at uniform spacing.

    Plan definition (Stage 2 → single-stage Boozer handoff plan, lines
    278-280)::

        S_HEL = sum(|B_hat[m,n]|^2 for n != 0) / sum(|B_hat[m,n]|^2)

    where ``np.fft.fft2`` lays out axis=0 = phi (toroidal, mode ``n``) and
    axis=1 = theta (poloidal, mode ``m``). The denominator is the full
    Parseval sum so the result is the fraction of total spectral power
    sitting in helical (``n != 0``) modes. Result is in ``[0, 1]``: ``0``
    for an axisymmetric field, ``1`` when all spectral power is helical.
    The metric is invariant to a uniform scaling of ``|B|`` because the
    numerator and denominator scale identically.

    Raises:
        ValueError: when ``modB`` is not 2D, contains non-finite samples,
            or has zero total spectral power (the metric is undefined).
    """
    modB = np.asarray(modB, dtype=float)
    if modB.ndim != 2:
        raise ValueError(
            f"|B| grid must be 2D (nphi, ntheta); got shape {modB.shape}"
        )
    if not np.all(np.isfinite(modB)):
        raise ValueError("|B| grid contains non-finite samples")
    spectrum = np.fft.fft2(modB)
    power = np.abs(spectrum) ** 2
    total_power = float(np.sum(power))
    if total_power == 0.0:
        raise ValueError("|B| grid has zero spectral power; S_HEL is undefined")
    helical_strip = power.copy()
    # Zero out the n=0 (axisymmetric) row from the numerator only. The
    # denominator keeps the full Parseval sum so the result reads as
    # "fraction of total power in helical modes".
    helical_strip[0, :] = 0.0
    return float(np.sum(helical_strip)) / total_power


def compute_helical_field_content_S_HEL(
    field, surface
) -> float:
    """Compute ``S_HEL`` for ``field`` sampled on ``surface``.

    Wraps :func:`helical_field_content_from_modB_grid` with the standard
    simsopt sampling protocol: set the field points to ``surface.gamma()``,
    read ``B``, take the magnitude on the (nphi, ntheta) grid. We
    transform ``|B|`` — NOT ``B`` itself, NOT ``B·n`` — so the metric
    captures field-magnitude modulation, which is what governs particle
    confinement on the surface.

    The caller is responsible for restoring ``field.set_points`` afterwards
    if it shares the field with the optimizer; this routine does not
    revert.
    """
    gamma = surface.gamma()
    if gamma.ndim != 3 or gamma.shape[-1] != 3:
        raise ValueError(
            f"surface.gamma() must be (nphi, ntheta, 3); got {gamma.shape}"
        )
    nphi, ntheta, _ = gamma.shape
    flat_points = gamma.reshape((-1, 3))
    field.set_points(flat_points)
    B = np.asarray(field.B(), dtype=float).reshape((nphi, ntheta, 3))
    modB = np.linalg.norm(B, axis=-1)
    return helical_field_content_from_modB_grid(modB)


def safe_compute_helical_field_content_S_HEL(
    field, surface
) -> float | None:
    """Best-effort wrapper that returns ``None`` on the documented failure modes.

    Per the Phase 3 plan ("Failed diagnostics fail closed as unavailable
    metrics, not as fake zero-quality success", line 305), the diagnostic
    path must not crash the parent solver when a degenerate ``|B|`` grid
    makes ``S_HEL`` mathematically undefined. The pure-function
    :func:`compute_helical_field_content_S_HEL` keeps strict semantics for
    unit tests; this wrapper is what the artifact pipeline calls.

    Returns ``None`` for the failure cases
    :func:`helical_field_content_from_modB_grid` raises ``ValueError`` for:
    a 2-D non-3-component gamma, non-finite ``|B|`` samples, or a grid with
    zero non-DC spectral power. All other exceptions propagate so they are
    not silently swallowed.
    """
    try:
        return compute_helical_field_content_S_HEL(field, surface)
    except ValueError:
        return None


def _trace_fieldlines_and_compute_iota(
    field,
    surface,
    *,
    n_lines: int,
    tmax: float,
    tol: float,
) -> tuple[list[float], int]:
    """Trace ``n_lines`` field lines and return per-line iota + survival."""

    radii = midplane_seed_radii(surface, int(n_lines))
    R0 = np.asarray(radii, dtype=float)
    Z0 = np.zeros_like(R0)
    stopping_criteria, _ = build_stopping_criteria(surface)
    histories, _ = compute_fieldlines(
        field,
        R0,
        Z0,
        tmax=float(tmax),
        tol=float(tol),
        phis=[0.0],
        stopping_criteria=stopping_criteria,
    )
    toroidal = compute_toroidal_transits(histories, flux=False)
    poloidal = compute_poloidal_transits(histories, flux=False)
    iota_per_line: list[float] = []
    survived = 0
    for ntor, npol in zip(toroidal, poloidal):
        if ntor <= 0:
            # Line escaped before completing a toroidal transit: no iota
            # can be defined.
            continue
        iota_per_line.append(float(npol) / float(ntor))
        survived += 1
    return iota_per_line, survived


def compute_fieldline_iota_proxy(
    field,
    surface,
    *,
    n_lines: int = 8,
    tmax: float = 200.0,
    tol: float = 1.0e-8,
    convergence_tol: float = FIELDLINE_IOTA_CONVERGENCE_TOL,
) -> FieldlineIotaProxyResult:
    """Compute the field-line iota proxy with a convergence sub-study.

    The proxy is the mean per-surviving-line ``poloidal_transits /
    toroidal_transits`` at trace length ``tmax``. The convergence
    sub-study traces the same seeds at ``2 * tmax`` and the proxy is
    only flagged ``valid=True`` when every surviving line agrees within
    ``convergence_tol`` between the two trace lengths (plan line 274).

    Failures (no surviving lines, divergent per-line iota, non-finite
    transit counts) yield ``valid=False`` and ``iota_proxy_mean=None``.
    """
    if n_lines <= 0:
        raise ValueError("n_lines must be positive")
    if tmax <= 0:
        raise ValueError("tmax must be positive")
    if tol <= 0:
        raise ValueError("tol must be positive")
    iota_short, survived_short = _trace_fieldlines_and_compute_iota(
        field,
        surface,
        n_lines=n_lines,
        tmax=tmax,
        tol=tol,
    )
    if survived_short == 0:
        return FieldlineIotaProxyResult(
            iota_proxy_mean=None,
            iota_proxy_std=None,
            iota_per_line=(),
            n_lines_seeded=int(n_lines),
            n_lines_survived=0,
            tmax=float(tmax),
            valid=False,
            convergence_residual=None,
            invalid_reason="no_lines_survived",
        )
    iota_long, _survived_long = _trace_fieldlines_and_compute_iota(
        field,
        surface,
        n_lines=n_lines,
        tmax=2.0 * tmax,
        tol=tol,
    )
    n_compare = min(len(iota_short), len(iota_long))
    if n_compare == 0:
        return FieldlineIotaProxyResult(
            iota_proxy_mean=None,
            iota_proxy_std=None,
            iota_per_line=tuple(iota_short),
            n_lines_seeded=int(n_lines),
            n_lines_survived=int(survived_short),
            tmax=float(tmax),
            valid=False,
            convergence_residual=None,
            invalid_reason="long_trace_had_no_survivors",
        )
    residuals = np.abs(
        np.asarray(iota_long[:n_compare]) - np.asarray(iota_short[:n_compare])
    )
    convergence_residual = float(np.max(residuals))
    valid = convergence_residual < float(convergence_tol)
    invalid_reason = None if valid else "convergence_tol_exceeded"
    iota_array = np.asarray(iota_short, dtype=float)
    return FieldlineIotaProxyResult(
        iota_proxy_mean=float(np.mean(iota_array)),
        iota_proxy_std=float(np.std(iota_array)),
        iota_per_line=tuple(iota_short),
        n_lines_seeded=int(n_lines),
        n_lines_survived=int(survived_short),
        tmax=float(tmax),
        valid=valid,
        convergence_residual=convergence_residual,
        invalid_reason=invalid_reason,
    )


def safe_compute_fieldline_iota_proxy(
    field,
    surface,
    *,
    n_lines: int = 4,
    tmax: float = 100.0,
    tol: float = 1.0e-8,
    convergence_tol: float = FIELDLINE_IOTA_CONVERGENCE_TOL,
) -> "FieldlineIotaProxyResult | None":
    """Best-effort wrapper that returns ``None`` on tracing failures.

    The artifact pipeline calls this post-solve to populate the Phase 3
    field-line proxy diagnostic without crashing the parent solver when
    field-line tracing fails (degenerate field, simsoptpp internal error).
    Per plan line 305 ("Failed diagnostics fail closed as unavailable
    metrics, not as fake zero-quality success"), a tracing failure must
    surface as ``None`` so downstream consumers can detect "diagnostic
    unavailable" rather than read a fabricated value.

    Only ``ValueError`` (the parameter-validation failure modes documented
    on :func:`compute_fieldline_iota_proxy`) and ``RuntimeError`` (raised by
    ``simsopt.field.tracing`` on numerical breakdown) are swallowed. Every
    other exception propagates so that genuine bugs are not masked.
    """
    try:
        return compute_fieldline_iota_proxy(
            field,
            surface,
            n_lines=n_lines,
            tmax=tmax,
            tol=tol,
            convergence_tol=convergence_tol,
        )
    except (ValueError, RuntimeError):
        return None


def compute_pre_boozer_topology_score(
    *,
    s_hel: float | None,
    fieldline_iota_proxy_mean: float | None,
    fieldline_iota_proxy_valid: bool,
    iota_target: float | None,
) -> float | None:
    """Composite ranking score combining helical content + iota proxy.

    Higher is better. ``None`` when neither diagnostic is available so
    downstream rankers can fail closed rather than reading a fabricated
    zero. The composite is intentionally simple — the goal is "good donor
    for Boozer activation", not a research-grade quality metric.

    Definition (one-sentence, recomputable from the artifact alone):

    .. code-block:: text

        score = S_HEL * (1 - |iota_proxy - iota_target|)   when both available
              = S_HEL                                       when only S_HEL
              = max(0, 1 - |iota_proxy - iota_target|)      when only iota_proxy
              = None                                        otherwise

    When the iota proxy is flagged invalid by the convergence gate, it is
    treated as unavailable.
    """
    proxy_available = (
        fieldline_iota_proxy_mean is not None
        and bool(fieldline_iota_proxy_valid)
    )
    if s_hel is None and not proxy_available:
        return None
    if s_hel is None:
        if iota_target is None:
            return None
        return max(
            0.0,
            1.0 - abs(float(fieldline_iota_proxy_mean) - float(iota_target)),
        )
    if not proxy_available or iota_target is None:
        return float(s_hel)
    proximity = 1.0 - abs(
        float(fieldline_iota_proxy_mean) - float(iota_target)
    )
    return float(s_hel) * max(0.0, proximity)


def boozer_topology_bridge_artifact_fields(
    *,
    s_hel: float | None,
    fieldline_result: FieldlineIotaProxyResult | None,
    s_hel_objective_weight: float | None,
    iota_target: float | None,
) -> dict[str, object]:
    """Build the artifact-key payload for the Phase 3 diagnostics.

    Always returns the five top-level diagnostic keys (with ``None``
    placeholders when a diagnostic was not computed) so the Stage 2
    artifact contract is identical across the off / soft / probe lanes
    and downstream consumers never see a missing key.
    """
    fieldline_iota_proxy = (
        None if fieldline_result is None else fieldline_result.iota_proxy_mean
    )
    fieldline_valid = (
        None if fieldline_result is None else bool(fieldline_result.valid)
    )
    composite = compute_pre_boozer_topology_score(
        s_hel=s_hel,
        fieldline_iota_proxy_mean=fieldline_iota_proxy,
        fieldline_iota_proxy_valid=bool(fieldline_valid),
        iota_target=iota_target,
    )
    return {
        "FIELDLINE_IOTA_PROXY": fieldline_iota_proxy,
        "FIELDLINE_IOTA_PROXY_VALID": fieldline_valid,
        "HELICAL_FIELD_CONTENT": (
            None if s_hel is None else float(s_hel)
        ),
        "S_HEL_OBJECTIVE_WEIGHT": (
            None
            if s_hel_objective_weight is None
            else float(s_hel_objective_weight)
        ),
        "PRE_BOOZER_TOPOLOGY_SCORE": (
            None if composite is None else float(composite)
        ),
    }


def _modB_and_B_unit_on_surface(
    field, surface
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``|B|`` and the unit ``B``-vector on the surface gamma grid.

    Returns:
        modB: shape ``(nphi, ntheta)``.
        B_unit: shape ``(nphi, ntheta, 3)``, the per-point unit vector
            ``B / |B|`` used by the chain rule.

    The flat ``(nphi * ntheta, 3)`` ordering matches what
    :func:`BiotSavart.B_vjp` expects as its co-tangent argument.
    """
    gamma = surface.gamma()
    if gamma.ndim != 3 or gamma.shape[-1] != 3:
        raise ValueError(
            f"surface.gamma() must be (nphi, ntheta, 3); got {gamma.shape}"
        )
    nphi, ntheta, _ = gamma.shape
    flat_points = gamma.reshape((-1, 3))
    field.set_points(flat_points)
    B = np.asarray(field.B(), dtype=float).reshape((nphi, ntheta, 3))
    modB = np.linalg.norm(B, axis=-1)
    B_unit = B / modB[..., None]
    return modB, B_unit


def _S_HEL_value_and_dS_HEL_by_dmodB(
    modB: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return ``S_HEL`` and ``dS_HEL/d(modB)`` from the |B| grid alone.

    Math (see plan derivation):

    - ``F = np.fft.fft2(modB)``.
    - Mask ``M`` zeroes the ``n = 0`` (axisymmetric) row of ``F``:
      ``M[0, :] = 0``, ``M[k, :] = 1`` otherwise. ``M`` is real-valued so
      ``M = M^2``.
    - Numerator ``N(modB) = || M ⊙ F ||^2``.
      ``dN/d(modB) = 2 * (nphi * ntheta) * Re( np.fft.ifft2(M^2 ⊙ F) )``
      (``np.fft.ifft2`` includes the ``1 / (nphi * ntheta)`` factor, so the
      explicit ``(nphi * ntheta)`` multiplier restores the Parseval-style
      adjoint).
    - Denominator ``D(modB) = || F ||^2 = (nphi * ntheta) * ||modB||^2``
      (Parseval).  ``dD/d(modB) = 2 * (nphi * ntheta) * modB``.
    - ``dS/d(modB) = (dN * D - N * dD) / D^2``.
    """
    nphi, ntheta = modB.shape
    F = np.fft.fft2(modB)
    power = np.abs(F) ** 2
    D = float(np.sum(power))
    if D == 0.0:
        raise ValueError("|B| grid has zero spectral power; S_HEL is undefined")
    masked_F = F.copy()
    masked_F[0, :] = 0.0
    N = float(np.sum(np.abs(masked_F) ** 2))
    S_HEL = N / D
    grid_size = float(nphi * ntheta)
    dN_dmodB = 2.0 * grid_size * np.real(np.fft.ifft2(masked_F))
    dD_dmodB = 2.0 * grid_size * modB
    dS_dmodB = (dN_dmodB * D - N * dD_dmodB) / (D * D)
    return S_HEL, dS_dmodB


class HelicalFieldContentObjective:
    """Stage 2 ramped helical-content objective with simsopt-style J / dJ API.

    Sampled on a fixed working surface (the optimizer's ``new_surf``); the
    field is a BiotSavart object whose coils are the optimization DOFs.
    The class caches the FFT and gradient and is the SSOT for ``S_HEL``
    contributions to ``make_stage2_fun``.
    """

    def __init__(self, field, surface):
        self.field = field
        self.surface = surface
        self._value: float | None = None
        self._grad: np.ndarray | None = None

    def recompute_bell(self) -> None:
        """Invalidate the J / dJ cache; the next call recomputes both."""
        self._value = None
        self._grad = None

    def _ensure_computed(self) -> None:
        if self._value is not None and self._grad is not None:
            return
        modB, B_unit = _modB_and_B_unit_on_surface(self.field, self.surface)
        S_HEL, dS_dmodB = _S_HEL_value_and_dS_HEL_by_dmodB(modB)
        # Chain rule from |B|_{ij} = ||B_{ij}||:
        #   d|B|/dB_k = B_k / |B|
        # so dS/dB_{ij,k} = dS/d|B|_{ij} * B_unit_{ij,k}.
        dS_dB = dS_dmodB[..., None] * B_unit
        nphi, ntheta, _ = dS_dB.shape
        flat_dS_dB = dS_dB.reshape((nphi * ntheta, 3))
        # BiotSavart.B_vjp returns a Derivative; calling it on ``field``
        # yields a flat numpy array of the field's free coil DOFs.
        derivative = self.field.B_vjp(flat_dS_dB)
        self._value = float(S_HEL)
        self._grad = np.asarray(derivative(self.field), dtype=float)

    def J(self) -> float:
        """Return scalar ``S_HEL`` in ``[0, 1]``."""
        self._ensure_computed()
        return self._value

    def dJ_by_dcoils(self) -> np.ndarray:
        """Return ``dS_HEL / d(coil DOFs)`` matching ``field.x`` ordering."""
        self._ensure_computed()
        return self._grad


def compute_S_HEL_gradient_relative_error(
    field,
    surface,
    *,
    perturbation: float = 1.0e-5,
    seed: int = 0,
) -> float:
    """Validate the analytic S_HEL gradient against centered finite differences.

    Samples up to four free coil DOFs (deterministic given ``seed``) of
    ``field``, perturbs each by ``± perturbation``, recomputes ``S_HEL``
    end-to-end at both points, and compares the centered-difference
    estimate to the analytic gradient component returned by
    :class:`HelicalFieldContentObjective`. Returns the maximum
    relative-error across the sampled components; a centered difference
    at ``eps = 1e-5`` carries an inherent ~1e-5 truncation floor so
    ``max_rel_error < 1e-3`` is the operational pass bar.
    """
    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive")
    objective = HelicalFieldContentObjective(field, surface)
    objective.recompute_bell()
    analytic_grad = objective.dJ_by_dcoils()
    n_dofs = analytic_grad.size
    if n_dofs == 0:
        raise ValueError("field has no free DOFs to validate against")
    rng = np.random.default_rng(int(seed))
    n_sample = int(min(4, n_dofs))
    indices = rng.choice(n_dofs, size=n_sample, replace=False)
    x0 = np.asarray(field.x, dtype=float).copy()
    max_rel_error = 0.0
    for idx in indices:
        x_plus = x0.copy()
        x_plus[idx] += perturbation
        field.x = x_plus
        plus = HelicalFieldContentObjective(field, surface)
        s_plus = plus.J()
        x_minus = x0.copy()
        x_minus[idx] -= perturbation
        field.x = x_minus
        minus = HelicalFieldContentObjective(field, surface)
        s_minus = minus.J()
        fd_grad = (s_plus - s_minus) / (2.0 * perturbation)
        analytic = float(analytic_grad[idx])
        denom = max(abs(fd_grad), abs(analytic), 1.0e-12)
        rel_error = abs(fd_grad - analytic) / denom
        if rel_error > max_rel_error:
            max_rel_error = rel_error
    field.x = x0
    return float(max_rel_error)
