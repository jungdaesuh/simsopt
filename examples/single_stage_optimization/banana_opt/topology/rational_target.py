from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd, isfinite, pi


GREENE_IOTA_CONVENTION = "iota=p/q"
GREENE_FOURIER_CONVENTION = "m*iota-n=0"
GREENE_MAP_CONVENTION_FULL_TORUS = "full_torus_phi_return_map"
GREENE_BRANCH_O = "O"
GREENE_BRANCH_X = "X"
GREENE_BRANCHES = frozenset({GREENE_BRANCH_O, GREENE_BRANCH_X})


def _format_manifest_float(value: float | None) -> str:
    if value is None:
        return "none"
    return format(float(value), ".17g")


def _normalize_radial_window(
    radial_window: Sequence[float] | None,
) -> tuple[float, float] | None:
    if radial_window is None:
        return None
    if len(radial_window) != 2:
        raise ValueError("Greene rational target radial_window must have two entries")
    lower = float(radial_window[0])
    upper = float(radial_window[1])
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        raise ValueError(
            "Greene rational target radial_window must be finite and ordered"
        )
    return (lower, upper)


def _normalize_branches(branches: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(branch) for branch in branches)
    if len(normalized) == 0:
        raise ValueError("Greene rational target requires at least one branch")
    unknown = sorted(set(normalized) - GREENE_BRANCHES)
    if unknown:
        raise ValueError(f"Unknown Greene residue branch labels: {unknown}")
    return normalized


@dataclass(frozen=True, slots=True)
class RationalTarget:
    """Convention-locked p/q target for the full-torus Greene return map."""

    p: int
    q: int
    weight: float = 1.0
    radial_label: float | None = None
    radial_window: Sequence[float] | None = None
    branches: Sequence[str] = (GREENE_BRANCH_O, GREENE_BRANCH_X)
    phi0: float = 0.0
    nfp: int = 1
    convention: str = GREENE_IOTA_CONVENTION
    fourier_m: int | None = None
    fourier_n: int | None = None
    map_convention: str = GREENE_MAP_CONVENTION_FULL_TORUS

    def __post_init__(self) -> None:
        if self.convention != GREENE_IOTA_CONVENTION:
            raise ValueError(
                "Greene rational targets must declare convention='iota=p/q'"
            )
        if self.map_convention != GREENE_MAP_CONVENTION_FULL_TORUS:
            raise ValueError(
                "Greene rational targets currently require a full-torus return map"
            )
        p = int(self.p)
        q = int(self.q)
        if q <= 0:
            raise ValueError("Greene rational target q must be positive")
        if gcd(abs(p), q) != 1:
            raise ValueError("Greene rational target p/q must be reduced")
        weight = float(self.weight)
        phi0 = float(self.phi0)
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError(
                "Greene rational target weight must be finite and positive"
            )
        if not isfinite(phi0):
            raise ValueError("Greene rational target phi0 must be finite")
        radial_label = None if self.radial_label is None else float(self.radial_label)
        if radial_label is not None and not isfinite(radial_label):
            raise ValueError("Greene rational target radial_label must be finite")
        nfp = int(self.nfp)
        if nfp <= 0:
            raise ValueError("Greene rational target nfp must be positive")
        has_m = self.fourier_m is not None
        has_n = self.fourier_n is not None
        if has_m != has_n:
            raise ValueError("Greene Fourier metadata requires both m and n")
        fourier_m = None if self.fourier_m is None else int(self.fourier_m)
        fourier_n = None if self.fourier_n is None else int(self.fourier_n)
        if fourier_m is not None and fourier_n is not None:
            if fourier_m == 0:
                raise ValueError("Greene Fourier metadata requires nonzero m")
            if fourier_m * p != fourier_n * q:
                raise ValueError(
                    "Greene Fourier metadata must satisfy m * (p/q) - n = 0"
                )

        object.__setattr__(self, "p", p)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "radial_label", radial_label)
        object.__setattr__(
            self,
            "radial_window",
            _normalize_radial_window(self.radial_window),
        )
        object.__setattr__(self, "branches", _normalize_branches(self.branches))
        object.__setattr__(self, "phi0", phi0)
        object.__setattr__(self, "nfp", nfp)
        object.__setattr__(self, "fourier_m", fourier_m)
        object.__setattr__(self, "fourier_n", fourier_n)

    @property
    def iota(self) -> Fraction:
        return Fraction(self.p, self.q)

    @property
    def iota_float(self) -> float:
        return float(self.iota)

    @property
    def full_torus_span(self) -> tuple[float, float]:
        return (self.phi0, self.phi0 + 2.0 * pi)

    @property
    def periodic_span(self) -> tuple[float, float]:
        return (self.phi0, self.phi0 + 2.0 * pi * float(self.q))

    def expected_winding(self) -> int:
        return self.p

    def manifest_key(self) -> str:
        radial_window = self.radial_window
        window_payload = (
            "none"
            if radial_window is None
            else (
                f"{_format_manifest_float(radial_window[0])}:"
                f"{_format_manifest_float(radial_window[1])}"
            )
        )
        fourier_payload = (
            "none"
            if self.fourier_m is None or self.fourier_n is None
            else f"{self.fourier_m}:{self.fourier_n}:{GREENE_FOURIER_CONVENTION}"
        )
        return "|".join(
            (
                f"p={self.p}",
                f"q={self.q}",
                f"iota={self.p}/{self.q}",
                f"weight={_format_manifest_float(self.weight)}",
                f"radial_label={_format_manifest_float(self.radial_label)}",
                f"radial_window={window_payload}",
                f"branches={','.join(self.branches)}",
                f"phi0={_format_manifest_float(self.phi0)}",
                f"nfp={self.nfp}",
                f"convention={self.convention}",
                f"map={self.map_convention}",
                f"fourier={fourier_payload}",
            )
        )
