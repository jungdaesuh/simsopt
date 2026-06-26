from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline, interp1d


EDGE_IOTA_STATUS_PASS = "pass"
EDGE_IOTA_STATUS_FAIL = "fail"
EDGE_IOTA_STATUS_INSUFFICIENT_SAMPLES = "insufficient_samples"
EDGE_IOTA_STATUS_MISSING_INPUTS = "missing_inputs"

EDGE_HELICITY_STATUS_CO = "co_helicity"
EDGE_HELICITY_STATUS_COUNTER = "counter_helicity"
EDGE_HELICITY_STATUS_MIXED = "mixed_helicity"
EDGE_HELICITY_STATUS_UNKNOWN = "unknown"

EDGE_IOTA_MODE_OFF = "off"
EDGE_IOTA_MODE_REPORT = "report"
EDGE_IOTA_MODE_SOFT = "soft"
EDGE_IOTA_ACTIVE_MODES = {EDGE_IOTA_MODE_REPORT, EDGE_IOTA_MODE_SOFT}

EDGE_IOTA_PROMOTION_BLOCK_EDGE_STATUS = "edge_iota_status"
EDGE_IOTA_PROMOTION_BLOCK_EDGE_P10 = "edge_delta_abs_iota_p10"
EDGE_IOTA_PROMOTION_BLOCK_EDGE_HELICITY = "edge_helicity_status"
EDGE_IOTA_PROMOTION_BLOCK_EDGE_SURVIVAL = "edge_surface_survival_fraction"
EDGE_IOTA_PROMOTION_BLOCK_EDGE_WIDTH = "edge_width_max"
EDGE_IOTA_PROMOTION_BLOCK_HARDWARE = "hardware_constraints"
EDGE_IOTA_PROMOTION_BLOCK_DIRECT_CAD = "direct_cad_contact_oracle"

DEFAULT_EDGE_BAND = (0.75, 1.0)
DEFAULT_EDGE_SAMPLE_COUNT = 6
DEFAULT_EDGE_TRACE_TURNS = 120
DEFAULT_EDGE_STEPS_PER_TURN = 240
DEFAULT_EDGE_Q_REL_TOL = 2.0e-3
DEFAULT_EDGE_DELTA_ABS_IOTA_TARGET_MIN = 0.10
DEFAULT_EDGE_SURVIVAL_FRACTION_MIN = 1.0

CylindricalField = Callable[[float, float, float], tuple[float, float, float]]


@dataclass(frozen=True, slots=True)
class EqdskEquilibrium:
    """Parsed G-EQDSK data plus interpolation helpers for psi, F(psi), and q(psi)."""

    path: str
    nw: int
    nh: int
    rdim: float
    zdim: float
    rcentr: float
    rleft: float
    zmid: float
    rmaxis: float
    zmaxis: float
    simag: float
    sibry: float
    bcentr: float
    current: float
    fpol: np.ndarray
    pres: np.ndarray
    ffprime: np.ndarray
    pprime: np.ndarray
    psirz: np.ndarray
    qpsi: np.ndarray
    rbbbs: np.ndarray
    zbbbs: np.ndarray
    rlim: np.ndarray
    zlim: np.ndarray
    r_grid: np.ndarray
    z_grid: np.ndarray
    psi_grid: np.ndarray

    def build_axisymmetric_field(self) -> "AxisymmetricTokamakField":
        return AxisymmetricTokamakField(self)

    def to_metadata(self) -> dict[str, object]:
        return {
            "eqdsk_path": self.path,
            "nw": int(self.nw),
            "nh": int(self.nh),
            "rmaxis": float(self.rmaxis),
            "zmaxis": float(self.zmaxis),
            "simag": float(self.simag),
            "sibry": float(self.sibry),
            "bcentr": float(self.bcentr),
            "current": float(self.current),
            "q_axis": float(self.qpsi[0]),
            "q_edge": float(self.qpsi[-1]),
        }


class AxisymmetricTokamakField:
    """Axisymmetric G-EQDSK field in cylindrical coordinates.

    The field uses the standard fixed-boundary convention
    ``B = F(psi) grad(phi) + grad(psi) x grad(phi)``:
    ``B_R = -(1/R) dpsi/dZ``, ``B_Z = (1/R) dpsi/dR``, and
    ``B_phi = F(psi)/R``. The caller must validate traced tokamak-only iota
    against the EQDSK q-profile before trusting hybrid profiles.
    """

    def __init__(self, equilibrium: EqdskEquilibrium):
        self.equilibrium = equilibrium
        self._psi = RectBivariateSpline(
            equilibrium.z_grid,
            equilibrium.r_grid,
            equilibrium.psirz,
            kx=3,
            ky=3,
        )
        self._fpol = interp1d(
            equilibrium.psi_grid,
            equilibrium.fpol,
            bounds_error=False,
            fill_value=(equilibrium.fpol[0], equilibrium.fpol[-1]),
        )
        self._qpsi = interp1d(
            equilibrium.psi_grid,
            equilibrium.qpsi,
            bounds_error=False,
            fill_value=(equilibrium.qpsi[0], equilibrium.qpsi[-1]),
        )

    def psi(self, R_m: float, Z_m: float) -> float:
        return float(self._psi.ev(float(Z_m), float(R_m)))

    def q_at(self, R_m: float, Z_m: float) -> float:
        return float(self._qpsi(self.psi(R_m, Z_m)))

    def __call__(self, R_m: float, phi_rad: float, Z_m: float) -> tuple[float, float, float]:
        del phi_rad
        R = float(R_m)
        Z = float(Z_m)
        dpsi_dz = float(self._psi.ev(Z, R, dx=1, dy=0))
        dpsi_dr = float(self._psi.ev(Z, R, dx=0, dy=1))
        psi = float(self._psi.ev(Z, R))
        fpol = float(self._fpol(psi))
        return (-dpsi_dz / R, fpol / R, dpsi_dr / R)


@dataclass(frozen=True, slots=True)
class EdgeIotaConfig:
    """Inputs and thresholds for the report-only edge-delivered-iota oracle."""

    eqdsk_path: str | None
    lcfs_path: str | None
    edge_band: tuple[float, float] = DEFAULT_EDGE_BAND
    sample_count: int = DEFAULT_EDGE_SAMPLE_COUNT
    helicity_sign: int = 1
    trace_turns: int = DEFAULT_EDGE_TRACE_TURNS
    steps_per_turn: int = DEFAULT_EDGE_STEPS_PER_TURN
    q_validation_rel_tol: float = DEFAULT_EDGE_Q_REL_TOL
    edge_delta_abs_iota_target_min: float = DEFAULT_EDGE_DELTA_ABS_IOTA_TARGET_MIN
    edge_survival_fraction_min: float = DEFAULT_EDGE_SURVIVAL_FRACTION_MIN
    edge_width_max: float | None = None
    coil_partition: Mapping[str, object] | None = None

    def validate(self, mode: str) -> None:
        validate_edge_iota_config(self, mode)

    def radial_labels(self) -> tuple[float, ...]:
        return edge_radial_labels(self.edge_band, self.sample_count)

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["edge_band"] = [float(self.edge_band[0]), float(self.edge_band[1])]
        payload["coil_partition"] = (
            None if self.coil_partition is None else dict(self.coil_partition)
        )
        return payload


@dataclass(frozen=True, slots=True)
class EdgeIotaSample:
    """One baseline/hybrid trace sample in the configured edge band."""

    radius_label: str
    r_over_a: float
    seed_r_m: float
    seed_z_m: float
    iota_tokamak: float | None
    iota_hybrid: float | None
    delta_iota_signed: float | None
    delta_abs_iota: float | None
    converged: bool
    survived: bool
    edge_width: float | None
    chaos_indicator: float | None
    reason: str = ""

    @classmethod
    def from_iotas(
        cls,
        *,
        radius_label: str,
        r_over_a: float,
        seed_point: tuple[float, float],
        iota_tokamak: float | None,
        iota_hybrid: float | None,
        converged: bool = True,
        survived: bool = True,
        edge_width: float | None = None,
        chaos_indicator: float | None = None,
        reason: str = "",
    ) -> "EdgeIotaSample":
        delta_signed = (
            None
            if iota_tokamak is None or iota_hybrid is None
            else float(iota_hybrid) - float(iota_tokamak)
        )
        delta_abs = (
            None
            if iota_tokamak is None or iota_hybrid is None
            else abs(float(iota_hybrid)) - abs(float(iota_tokamak))
        )
        return cls(
            radius_label=radius_label,
            r_over_a=float(r_over_a),
            seed_r_m=float(seed_point[0]),
            seed_z_m=float(seed_point[1]),
            iota_tokamak=None if iota_tokamak is None else float(iota_tokamak),
            iota_hybrid=None if iota_hybrid is None else float(iota_hybrid),
            delta_iota_signed=delta_signed,
            delta_abs_iota=delta_abs,
            converged=bool(converged),
            survived=bool(survived),
            edge_width=None if edge_width is None else float(edge_width),
            chaos_indicator=None if chaos_indicator is None else float(chaos_indicator),
            reason=reason,
        )

    def is_valid_for_summary(self) -> bool:
        return (
            self.converged
            and self.survived
            and _is_finite_optional(self.delta_iota_signed)
            and _is_finite_optional(self.delta_abs_iota)
        )

    def to_json_dict(self) -> dict[str, object]:
        return {
            "radius_label": self.radius_label,
            "r_over_a": float(self.r_over_a),
            "seed_point_m": [float(self.seed_r_m), float(self.seed_z_m)],
            "iota_tokamak": _json_float_or_none(self.iota_tokamak),
            "iota_hybrid": _json_float_or_none(self.iota_hybrid),
            "delta_iota_signed": _json_float_or_none(self.delta_iota_signed),
            "delta_abs_iota": _json_float_or_none(self.delta_abs_iota),
            "converged": bool(self.converged),
            "survived": bool(self.survived),
            "edge_width": _json_float_or_none(self.edge_width),
            "chaos_indicator": _json_float_or_none(self.chaos_indicator),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class EdgeIotaProfile:
    """Edge-band profile and scalar report fields for delivered external iota."""

    samples: tuple[EdgeIotaSample, ...]
    edge_delta_abs_iota_min: float | None
    edge_delta_abs_iota_p10: float | None
    edge_delta_abs_iota_mean: float | None
    edge_delta_signed_iota_min: float | None
    edge_delta_signed_iota_mean: float | None
    edge_surface_survival_fraction: float
    edge_width_max: float | None
    edge_iota_status: str
    edge_helicity_status: str

    def scalar_payload(self) -> dict[str, object]:
        return {
            "EDGE_IOTA_STATUS": self.edge_iota_status,
            "EDGE_DELTA_ABS_IOTA_MIN": _json_float_or_none(
                self.edge_delta_abs_iota_min
            ),
            "EDGE_DELTA_ABS_IOTA_P10": _json_float_or_none(
                self.edge_delta_abs_iota_p10
            ),
            "EDGE_DELTA_ABS_IOTA_MEAN": _json_float_or_none(
                self.edge_delta_abs_iota_mean
            ),
            "EDGE_DELTA_SIGNED_IOTA_MIN": _json_float_or_none(
                self.edge_delta_signed_iota_min
            ),
            "EDGE_DELTA_SIGNED_IOTA_MEAN": _json_float_or_none(
                self.edge_delta_signed_iota_mean
            ),
            "EDGE_SURFACE_SURVIVAL_FRACTION": float(
                self.edge_surface_survival_fraction
            ),
            "EDGE_WIDTH_MAX": _json_float_or_none(self.edge_width_max),
            "EDGE_HELICITY_STATUS": self.edge_helicity_status,
        }

    def to_json_dict(self) -> dict[str, object]:
        payload = self.scalar_payload()
        payload["samples"] = [sample.to_json_dict() for sample in self.samples]
        return payload


@dataclass(frozen=True, slots=True)
class EdgeIotaPromotionDecision:
    """Fail-closed Phase 6 promotion decision for edge-delivered iota."""

    promotion_ready: bool
    blocking_reasons: tuple[str, ...]
    edge_iota_status: str
    hardware_constraints_ok: bool
    direct_cad_contact_oracle_ok: bool
    sdf_proxy_clearance_ok: bool | None = None

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TraceIotaResult:
    iota: float
    hits_rz_m: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class TokamakIotaValidationSample:
    r_over_a: float
    seed_r_m: float
    seed_z_m: float
    traced_iota: float
    eqdsk_iota: float
    relative_error: float
    passed: bool

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokamakIotaValidation:
    samples: tuple[TokamakIotaValidationSample, ...]
    relative_tolerance: float
    passed: bool

    def max_relative_error(self) -> float | None:
        if len(self.samples) == 0:
            return None
        return max(float(sample.relative_error) for sample in self.samples)

    def to_json_dict(self) -> dict[str, object]:
        return {
            "relative_tolerance": float(self.relative_tolerance),
            "passed": bool(self.passed),
            "max_relative_error": _json_float_or_none(self.max_relative_error()),
            "samples": [sample.to_json_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class LcfsBoundary:
    r_m: tuple[float, ...]
    z_m: tuple[float, ...]

    def minor_radius_from_axis(self, axis_r_m: float) -> float:
        outward = max(self.r_m) - float(axis_r_m)
        inward = float(axis_r_m) - min(self.r_m)
        return float(min(outward, inward))

    def contains_point(
        self,
        r_m: float,
        z_m: float,
        *,
        tolerance_m: float = 1.0e-9,
    ) -> bool:
        """Return whether a point is inside or on the LCFS polygon."""
        r_value = float(r_m)
        z_value = float(z_m)
        vertices = tuple(zip(self.r_m, self.z_m))
        inside = False
        for index, (r0, z0) in enumerate(vertices):
            r1, z1 = vertices[(index + 1) % len(vertices)]
            if _point_on_segment(
                r_value,
                z_value,
                float(r0),
                float(z0),
                float(r1),
                float(z1),
                tolerance_m=tolerance_m,
            ):
                return True
            crosses_ray = (float(z0) > z_value) != (float(z1) > z_value)
            if crosses_ray:
                r_cross = float(r0) + (z_value - float(z0)) * (
                    float(r1) - float(r0)
                ) / (float(z1) - float(z0))
                if r_value < r_cross:
                    inside = not inside
        return inside


def validate_edge_iota_config(config: EdgeIotaConfig, mode: str) -> None:
    if mode not in {EDGE_IOTA_MODE_OFF, EDGE_IOTA_MODE_REPORT, EDGE_IOTA_MODE_SOFT}:
        raise ValueError(
            "--stage2-edge-iota-mode must be one of off, report, or soft."
        )
    if mode == EDGE_IOTA_MODE_OFF:
        return
    missing = []
    if config.eqdsk_path in {None, ""}:
        missing.append("--stage2-edge-iota-eqdsk")
    if config.lcfs_path in {None, ""}:
        missing.append("--stage2-edge-iota-lcfs")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{mode} edge-iota mode requires {joined}.")
    edge_radial_labels(config.edge_band, config.sample_count)
    if int(config.helicity_sign) not in {-1, 1}:
        raise ValueError("--stage2-edge-iota-helicity must resolve to +1 or -1.")
    if int(config.trace_turns) <= 0:
        raise ValueError("--stage2-edge-iota trace turns must be positive.")
    if int(config.steps_per_turn) <= 0:
        raise ValueError("--stage2-edge-iota steps per turn must be positive.")
    if float(config.q_validation_rel_tol) <= 0.0:
        raise ValueError("--stage2-edge-iota q validation tolerance must be positive.")
    if float(config.edge_delta_abs_iota_target_min) <= 0.0:
        raise ValueError("--stage2-edge-iota target minimum must be positive.")
    survival_floor = float(config.edge_survival_fraction_min)
    if survival_floor < 0.0 or survival_floor > 1.0:
        raise ValueError(
            "--stage2-edge-iota survival fraction minimum must be between 0 and 1."
        )
    if config.edge_width_max is not None and float(config.edge_width_max) <= 0.0:
        raise ValueError("--stage2-edge-iota width maximum must be positive.")


def edge_radial_labels(
    edge_band: tuple[float, float],
    sample_count: int,
) -> tuple[float, ...]:
    lower, upper = float(edge_band[0]), float(edge_band[1])
    count = int(sample_count)
    if count < 2:
        raise ValueError("edge-iota sample count must be at least two.")
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("edge-iota radial band must be finite.")
    if lower <= 0.0 or upper <= lower or upper > 1.0:
        raise ValueError("edge-iota radial band must satisfy 0 < lower < upper <= 1.")
    return tuple(float(value) for value in np.linspace(lower, upper, count))


def read_eqdsk(path: str | Path) -> EqdskEquilibrium:
    source = Path(path)
    lines = source.read_text(encoding="utf-8").splitlines()
    header_tokens = lines[0].split()
    nw = int(header_tokens[-2])
    nh = int(header_tokens[-1])

    scalar_values = _fixed_width_float_values(lines[1:5])
    rdim, zdim, rcentr, rleft, zmid = scalar_values[0:5]
    rmaxis, zmaxis, simag, sibry, bcentr = scalar_values[5:10]
    current = scalar_values[10]

    needed_profile_values = 4 * nw + nw * nh + nw
    profile_values: list[float] = []
    line_index = 5
    while len(profile_values) < needed_profile_values:
        profile_values.extend(_fixed_width_float_values([lines[line_index]]))
        line_index += 1
    if len(profile_values) != needed_profile_values:
        raise ValueError("G-EQDSK profile block has unexpected extra values.")

    offset = 0
    fpol = np.asarray(profile_values[offset : offset + nw], dtype=float)
    offset += nw
    pres = np.asarray(profile_values[offset : offset + nw], dtype=float)
    offset += nw
    ffprime = np.asarray(profile_values[offset : offset + nw], dtype=float)
    offset += nw
    pprime = np.asarray(profile_values[offset : offset + nw], dtype=float)
    offset += nw
    psirz = np.asarray(profile_values[offset : offset + nw * nh], dtype=float).reshape(
        nh,
        nw,
    )
    offset += nw * nh
    qpsi = np.asarray(profile_values[offset : offset + nw], dtype=float)

    counts = lines[line_index].split()
    nbbbs = int(counts[0])
    limitr = int(counts[1])
    boundary_values = _fixed_width_float_values(lines[line_index + 1 :])
    boundary = np.asarray(boundary_values[: 2 * nbbbs], dtype=float).reshape(nbbbs, 2)
    limiter_start = 2 * nbbbs
    limiter_stop = limiter_start + 2 * limitr
    limiter = np.asarray(boundary_values[limiter_start:limiter_stop], dtype=float)
    limiter = limiter.reshape(limitr, 2)

    r_grid = rleft + np.linspace(0.0, rdim, nw)
    z_grid = (zmid - zdim / 2.0) + np.linspace(0.0, zdim, nh)
    psi_grid = np.linspace(simag, sibry, nw)
    return EqdskEquilibrium(
        path=str(source),
        nw=nw,
        nh=nh,
        rdim=float(rdim),
        zdim=float(zdim),
        rcentr=float(rcentr),
        rleft=float(rleft),
        zmid=float(zmid),
        rmaxis=float(rmaxis),
        zmaxis=float(zmaxis),
        simag=float(simag),
        sibry=float(sibry),
        bcentr=float(bcentr),
        current=float(current),
        fpol=fpol,
        pres=pres,
        ffprime=ffprime,
        pprime=pprime,
        psirz=psirz,
        qpsi=qpsi,
        rbbbs=boundary[:, 0],
        zbbbs=boundary[:, 1],
        rlim=limiter[:, 0],
        zlim=limiter[:, 1],
        r_grid=r_grid,
        z_grid=z_grid,
        psi_grid=psi_grid,
    )


def load_lcfs_boundary(path: str | Path) -> LcfsBoundary:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "R_m" in payload and "Z_m" in payload:
        return _validated_lcfs_boundary(
            r_m=tuple(float(value) for value in payload["R_m"]),
            z_m=tuple(float(value) for value in payload["Z_m"]),
        )
    if "rbbbs" in payload and "zbbbs" in payload:
        return _validated_lcfs_boundary(
            r_m=tuple(float(value) for value in payload["rbbbs"]),
            z_m=tuple(float(value) for value in payload["zbbbs"]),
        )
    raise ValueError("LCFS JSON must contain R_m/Z_m or rbbbs/zbbbs arrays.")


def trace_iota(
    field: CylindricalField,
    *,
    seed_r_m: float,
    seed_z_m: float,
    axis_r_m: float,
    axis_z_m: float,
    turns: int,
    steps_per_turn: int,
) -> TraceIotaResult:
    turn_count = int(turns)
    step_count = int(steps_per_turn)
    if turn_count <= 0 or step_count <= 0:
        raise ValueError("trace_iota requires positive turns and steps_per_turn.")

    def rhs(phi: float, state: Sequence[float]) -> list[float]:
        R, Z = float(state[0]), float(state[1])
        b_r, b_phi, b_z = field(R, phi, Z)
        return [R * b_r / b_phi, R * b_z / b_phi]

    phi_end = 2.0 * math.pi * float(turn_count)
    phi_eval = np.linspace(0.0, phi_end, turn_count * step_count + 1)
    solution = solve_ivp(
        rhs,
        (0.0, phi_end),
        [float(seed_r_m), float(seed_z_m)],
        t_eval=phi_eval,
        rtol=1.0e-9,
        atol=1.0e-11,
        max_step=2.0 * math.pi / float(step_count),
        method="RK45",
        dense_output=False,
    )
    if not solution.success or solution.y.shape[1] < phi_eval.size:
        raise RuntimeError("field-line trace did not reach the requested turn count.")
    traced_r = solution.y[0]
    traced_z = solution.y[1]
    theta = np.unwrap(
        np.arctan2(traced_z - float(axis_z_m), traced_r - float(axis_r_m))
    )
    iota = float((theta[-1] - theta[0]) / (phi_eval[-1] - phi_eval[0]))
    hit_indices = np.arange(0, phi_eval.size, step_count)
    hits = tuple(
        (float(traced_r[index]), float(traced_z[index])) for index in hit_indices
    )
    return TraceIotaResult(iota=iota, hits_rz_m=hits)


def validate_tokamak_iota_against_q(
    tokamak_field: AxisymmetricTokamakField,
    *,
    edge_band: tuple[float, float],
    sample_count: int,
    minor_radius_m: float,
    relative_tolerance: float = DEFAULT_EDGE_Q_REL_TOL,
    turns: int = DEFAULT_EDGE_TRACE_TURNS,
    steps_per_turn: int = DEFAULT_EDGE_STEPS_PER_TURN,
) -> TokamakIotaValidation:
    equilibrium = tokamak_field.equilibrium
    labels = edge_radial_labels(edge_band, sample_count)
    samples: list[TokamakIotaValidationSample] = []
    for r_over_a in labels:
        seed_r = float(equilibrium.rmaxis) + float(r_over_a) * float(minor_radius_m)
        seed_z = float(equilibrium.zmaxis)
        traced = trace_iota(
            tokamak_field,
            seed_r_m=seed_r,
            seed_z_m=seed_z,
            axis_r_m=float(equilibrium.rmaxis),
            axis_z_m=float(equilibrium.zmaxis),
            turns=turns,
            steps_per_turn=steps_per_turn,
        )
        q_value = tokamak_field.q_at(seed_r, seed_z)
        eqdsk_iota = 1.0 / q_value
        relative_error = abs(abs(traced.iota) - abs(eqdsk_iota)) / abs(eqdsk_iota)
        samples.append(
            TokamakIotaValidationSample(
                r_over_a=float(r_over_a),
                seed_r_m=seed_r,
                seed_z_m=seed_z,
                traced_iota=float(traced.iota),
                eqdsk_iota=float(eqdsk_iota),
                relative_error=float(relative_error),
                passed=relative_error <= float(relative_tolerance),
            )
        )
    return TokamakIotaValidation(
        samples=tuple(samples),
        relative_tolerance=float(relative_tolerance),
        passed=all(sample.passed for sample in samples),
    )


def build_edge_iota_profile(
    samples: Sequence[EdgeIotaSample],
    *,
    helicity_sign: int,
    target_min: float = DEFAULT_EDGE_DELTA_ABS_IOTA_TARGET_MIN,
    survival_fraction_min: float = DEFAULT_EDGE_SURVIVAL_FRACTION_MIN,
    width_max: float | None = None,
) -> EdgeIotaProfile:
    ordered = tuple(sorted(samples, key=lambda sample: sample.r_over_a))
    valid = tuple(sample for sample in ordered if sample.is_valid_for_summary())
    total = len(ordered)
    survival_fraction = (
        0.0 if total == 0 else sum(1 for sample in ordered if sample.survived) / total
    )
    widths = [
        float(sample.edge_width)
        for sample in ordered
        if _is_finite_optional(sample.edge_width)
    ]
    edge_width_max = None if len(widths) == 0 else max(widths)
    abs_deltas = [float(sample.delta_abs_iota) for sample in valid]
    signed_deltas = [float(sample.delta_iota_signed) for sample in valid]
    helicity_status = classify_edge_helicity(valid, helicity_sign=helicity_sign)
    if len(valid) == 0:
        edge_status = EDGE_IOTA_STATUS_INSUFFICIENT_SAMPLES
    else:
        p10 = float(np.percentile(abs_deltas, 10))
        width_pass = width_max is None or (
            edge_width_max is not None and edge_width_max <= float(width_max)
        )
        edge_status = (
            EDGE_IOTA_STATUS_PASS
            if (
                p10 >= float(target_min)
                and survival_fraction >= float(survival_fraction_min)
                and width_pass
                and helicity_status == EDGE_HELICITY_STATUS_CO
            )
            else EDGE_IOTA_STATUS_FAIL
        )
    return EdgeIotaProfile(
        samples=ordered,
        edge_delta_abs_iota_min=None if len(abs_deltas) == 0 else min(abs_deltas),
        edge_delta_abs_iota_p10=(
            None if len(abs_deltas) == 0 else float(np.percentile(abs_deltas, 10))
        ),
        edge_delta_abs_iota_mean=(
            None if len(abs_deltas) == 0 else float(np.mean(abs_deltas))
        ),
        edge_delta_signed_iota_min=(
            None if len(signed_deltas) == 0 else min(signed_deltas)
        ),
        edge_delta_signed_iota_mean=(
            None if len(signed_deltas) == 0 else float(np.mean(signed_deltas))
        ),
        edge_surface_survival_fraction=float(survival_fraction),
        edge_width_max=edge_width_max,
        edge_iota_status=edge_status,
        edge_helicity_status=helicity_status,
    )


def evaluate_edge_iota_promotion_gate(
    profile: EdgeIotaProfile,
    *,
    hardware_constraints_ok: bool,
    direct_cad_contact_oracle_ok: bool,
    target_min: float = DEFAULT_EDGE_DELTA_ABS_IOTA_TARGET_MIN,
    survival_fraction_min: float = DEFAULT_EDGE_SURVIVAL_FRACTION_MIN,
    width_max: float | None = None,
    sdf_proxy_clearance_ok: bool | None = None,
) -> EdgeIotaPromotionDecision:
    """Combine edge-iota, hardware, and direct CAD/contact promotion gates.

    SDF/proxy clearance is accepted as recorded steering evidence only; it never
    substitutes for the direct CAD/contact oracle required for promotion.
    """

    blocking_reasons: list[str] = []
    if profile.edge_iota_status != EDGE_IOTA_STATUS_PASS:
        blocking_reasons.append(EDGE_IOTA_PROMOTION_BLOCK_EDGE_STATUS)
    if profile.edge_delta_abs_iota_p10 is None or (
        float(profile.edge_delta_abs_iota_p10) < float(target_min)
    ):
        blocking_reasons.append(EDGE_IOTA_PROMOTION_BLOCK_EDGE_P10)
    if profile.edge_helicity_status != EDGE_HELICITY_STATUS_CO:
        blocking_reasons.append(EDGE_IOTA_PROMOTION_BLOCK_EDGE_HELICITY)
    if float(profile.edge_surface_survival_fraction) < float(survival_fraction_min):
        blocking_reasons.append(EDGE_IOTA_PROMOTION_BLOCK_EDGE_SURVIVAL)
    if width_max is not None and (
        profile.edge_width_max is None or float(profile.edge_width_max) > float(width_max)
    ):
        blocking_reasons.append(EDGE_IOTA_PROMOTION_BLOCK_EDGE_WIDTH)
    if not bool(hardware_constraints_ok):
        blocking_reasons.append(EDGE_IOTA_PROMOTION_BLOCK_HARDWARE)
    if not bool(direct_cad_contact_oracle_ok):
        blocking_reasons.append(EDGE_IOTA_PROMOTION_BLOCK_DIRECT_CAD)
    return EdgeIotaPromotionDecision(
        promotion_ready=len(blocking_reasons) == 0,
        blocking_reasons=tuple(blocking_reasons),
        edge_iota_status=profile.edge_iota_status,
        hardware_constraints_ok=bool(hardware_constraints_ok),
        direct_cad_contact_oracle_ok=bool(direct_cad_contact_oracle_ok),
        sdf_proxy_clearance_ok=sdf_proxy_clearance_ok,
    )


def evaluate_edge_iota_profile(
    *,
    tokamak_field: CylindricalField,
    hybrid_field: CylindricalField,
    axis_r_m: float,
    axis_z_m: float,
    minor_radius_m: float,
    config: EdgeIotaConfig,
    lcfs_boundary: LcfsBoundary | None = None,
) -> EdgeIotaProfile:
    config.validate(EDGE_IOTA_MODE_REPORT)
    samples: list[EdgeIotaSample] = []
    for r_over_a in config.radial_labels():
        seed_r = float(axis_r_m) + float(r_over_a) * float(minor_radius_m)
        seed_z = float(axis_z_m)
        tokamak_trace = _trace_iota_for_profile(
            tokamak_field,
            seed_r_m=seed_r,
            seed_z_m=seed_z,
            axis_r_m=axis_r_m,
            axis_z_m=axis_z_m,
            turns=config.trace_turns,
            steps_per_turn=config.steps_per_turn,
        )
        if tokamak_trace is None:
            samples.append(
                EdgeIotaSample.from_iotas(
                    radius_label=f"r_over_a={r_over_a:.6f}",
                    r_over_a=r_over_a,
                    seed_point=(seed_r, seed_z),
                    iota_tokamak=None,
                    iota_hybrid=None,
                    converged=False,
                    survived=False,
                    reason="tokamak_trace_failed",
                )
            )
            continue
        hybrid_trace = _trace_iota_for_profile(
            hybrid_field,
            seed_r_m=seed_r,
            seed_z_m=seed_z,
            axis_r_m=axis_r_m,
            axis_z_m=axis_z_m,
            turns=config.trace_turns,
            steps_per_turn=config.steps_per_turn,
        )
        if hybrid_trace is None:
            samples.append(
                EdgeIotaSample.from_iotas(
                    radius_label=f"r_over_a={r_over_a:.6f}",
                    r_over_a=r_over_a,
                    seed_point=(seed_r, seed_z),
                    iota_tokamak=tokamak_trace.iota,
                    iota_hybrid=None,
                    converged=False,
                    survived=False,
                    reason="hybrid_trace_failed",
                )
            )
            continue
        samples.append(
            EdgeIotaSample.from_iotas(
                radius_label=f"r_over_a={r_over_a:.6f}",
                r_over_a=r_over_a,
                seed_point=(seed_r, seed_z),
                iota_tokamak=tokamak_trace.iota,
                iota_hybrid=hybrid_trace.iota,
                converged=True,
                survived=_trace_survived_edge_boundary(
                    hybrid_trace.hits_rz_m,
                    axis_r_m=axis_r_m,
                    axis_z_m=axis_z_m,
                    minor_radius_m=minor_radius_m,
                    lcfs_boundary=lcfs_boundary,
                ),
                edge_width=_trace_width(hybrid_trace.hits_rz_m),
                chaos_indicator=_trace_width(hybrid_trace.hits_rz_m),
            )
        )
    return build_edge_iota_profile(
        samples,
        helicity_sign=config.helicity_sign,
        target_min=config.edge_delta_abs_iota_target_min,
        survival_fraction_min=config.edge_survival_fraction_min,
        width_max=config.edge_width_max,
    )


def validate_trace_samples_against_q(
    tokamak_field: AxisymmetricTokamakField,
    trace_samples: Sequence[tuple[float, float, float, float]],
    *,
    relative_tolerance: float = DEFAULT_EDGE_Q_REL_TOL,
) -> TokamakIotaValidation:
    samples: list[TokamakIotaValidationSample] = []
    for r_over_a, seed_r, seed_z, traced_iota in trace_samples:
        q_value = tokamak_field.q_at(float(seed_r), float(seed_z))
        eqdsk_iota = 1.0 / q_value
        relative_error = abs(abs(float(traced_iota)) - abs(eqdsk_iota)) / abs(
            eqdsk_iota
        )
        samples.append(
            TokamakIotaValidationSample(
                r_over_a=float(r_over_a),
                seed_r_m=float(seed_r),
                seed_z_m=float(seed_z),
                traced_iota=float(traced_iota),
                eqdsk_iota=float(eqdsk_iota),
                relative_error=float(relative_error),
                passed=relative_error <= float(relative_tolerance),
            )
        )
    return TokamakIotaValidation(
        samples=tuple(samples),
        relative_tolerance=float(relative_tolerance),
        passed=all(sample.passed for sample in samples),
    )


def classify_edge_helicity(
    samples: Sequence[EdgeIotaSample],
    *,
    helicity_sign: int,
) -> str:
    sign = int(helicity_sign)
    if sign not in {-1, 1}:
        raise ValueError("edge helicity sign must be +1 or -1.")
    signed_deltas = [
        float(sample.delta_iota_signed)
        for sample in samples
        if _is_finite_optional(sample.delta_iota_signed)
    ]
    if len(signed_deltas) == 0:
        return EDGE_HELICITY_STATUS_UNKNOWN
    co_count = sum(1 for delta in signed_deltas if delta * float(sign) > 0.0)
    counter_count = sum(1 for delta in signed_deltas if delta * float(sign) < 0.0)
    if co_count > 0 and counter_count > 0:
        return EDGE_HELICITY_STATUS_MIXED
    if co_count > 0:
        return EDGE_HELICITY_STATUS_CO
    if counter_count > 0:
        return EDGE_HELICITY_STATUS_COUNTER
    return EDGE_HELICITY_STATUS_UNKNOWN


def profile_json_payload(
    profile: EdgeIotaProfile,
    *,
    config: EdgeIotaConfig | None = None,
    eqdsk: EqdskEquilibrium | None = None,
    validation: TokamakIotaValidation | None = None,
) -> dict[str, object]:
    payload = profile.to_json_dict()
    if config is not None:
        payload["config"] = config.to_json_dict()
    if eqdsk is not None:
        payload["eqdsk"] = eqdsk.to_metadata()
    if validation is not None:
        payload["tokamak_q_validation"] = validation.to_json_dict()
    return payload


def edge_iota_config_hash(config: EdgeIotaConfig) -> str:
    payload = json.dumps(config.to_json_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_profile_json(
    profile: EdgeIotaProfile,
    path: str | Path,
    *,
    config: EdgeIotaConfig | None = None,
    eqdsk: EqdskEquilibrium | None = None,
    validation: TokamakIotaValidation | None = None,
) -> None:
    payload = profile_json_payload(
        profile,
        config=config,
        eqdsk=eqdsk,
        validation=validation,
    )
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def edge_iota_missing_payload() -> dict[str, object]:
    return {
        "EDGE_IOTA_STATUS": EDGE_IOTA_STATUS_MISSING_INPUTS,
        "EDGE_IOTA_PROFILE_JSON": None,
        "EDGE_DELTA_ABS_IOTA_MIN": None,
        "EDGE_DELTA_ABS_IOTA_P10": None,
        "EDGE_DELTA_ABS_IOTA_MEAN": None,
        "EDGE_DELTA_SIGNED_IOTA_MIN": None,
        "EDGE_DELTA_SIGNED_IOTA_MEAN": None,
        "EDGE_SURFACE_SURVIVAL_FRACTION": None,
        "EDGE_WIDTH_MAX": None,
        "EDGE_HELICITY_STATUS": EDGE_HELICITY_STATUS_UNKNOWN,
    }


def edge_iota_report_payload(
    profile: EdgeIotaProfile,
    *,
    profile_json_path: str | Path | None,
) -> dict[str, object]:
    payload = profile.scalar_payload()
    payload["EDGE_IOTA_PROFILE_JSON"] = (
        None if profile_json_path is None else str(profile_json_path)
    )
    return payload


def _fixed_width_float_values(lines: Sequence[str]) -> list[float]:
    values: list[float] = []
    for line in lines:
        for start in range(0, len(line), 16):
            chunk = line[start : start + 16].strip()
            if chunk:
                values.append(float(chunk))
    return values


def _trace_width(hits_rz_m: Sequence[tuple[float, float]]) -> float:
    points = np.asarray(hits_rz_m, dtype=float)
    if points.size == 0:
        return 0.0
    span_r = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    span_z = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    return float(math.hypot(span_r, span_z))


def _trace_iota_for_profile(
    field: CylindricalField,
    *,
    seed_r_m: float,
    seed_z_m: float,
    axis_r_m: float,
    axis_z_m: float,
    turns: int,
    steps_per_turn: int,
) -> TraceIotaResult | None:
    try:
        return trace_iota(
            field,
            seed_r_m=seed_r_m,
            seed_z_m=seed_z_m,
            axis_r_m=axis_r_m,
            axis_z_m=axis_z_m,
            turns=turns,
            steps_per_turn=steps_per_turn,
        )
    except RuntimeError:
        return None


def _trace_survived_edge_boundary(
    hits_rz_m: Sequence[tuple[float, float]],
    *,
    axis_r_m: float,
    axis_z_m: float,
    minor_radius_m: float,
    lcfs_boundary: LcfsBoundary | None,
) -> bool:
    if len(hits_rz_m) == 0:
        return False
    if lcfs_boundary is not None:
        return all(lcfs_boundary.contains_point(r, z) for r, z in hits_rz_m)
    axis_r = float(axis_r_m)
    axis_z = float(axis_z_m)
    radius_limit = float(minor_radius_m)
    return all(
        math.hypot(float(r) - axis_r, float(z) - axis_z)
        <= radius_limit + 1.0e-9
        for r, z in hits_rz_m
    )


def _validated_lcfs_boundary(
    *,
    r_m: tuple[float, ...],
    z_m: tuple[float, ...],
) -> LcfsBoundary:
    if len(r_m) != len(z_m) or len(r_m) < 3:
        raise ValueError("LCFS boundary must contain at least three R/Z point pairs.")
    if not all(math.isfinite(float(value)) for value in (*r_m, *z_m)):
        raise ValueError("LCFS boundary coordinates must be finite.")
    return LcfsBoundary(r_m=r_m, z_m=z_m)


def _point_on_segment(
    r_m: float,
    z_m: float,
    r0_m: float,
    z0_m: float,
    r1_m: float,
    z1_m: float,
    *,
    tolerance_m: float,
) -> bool:
    segment_r = float(r1_m) - float(r0_m)
    segment_z = float(z1_m) - float(z0_m)
    point_r = float(r_m) - float(r0_m)
    point_z = float(z_m) - float(z0_m)
    cross = segment_r * point_z - segment_z * point_r
    if abs(cross) > float(tolerance_m) * max(
        math.hypot(segment_r, segment_z),
        1.0,
    ):
        return False
    dot = point_r * segment_r + point_z * segment_z
    if dot < -float(tolerance_m):
        return False
    segment_length_squared = segment_r * segment_r + segment_z * segment_z
    return dot <= segment_length_squared + float(tolerance_m)


def _is_finite_optional(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _json_float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)
