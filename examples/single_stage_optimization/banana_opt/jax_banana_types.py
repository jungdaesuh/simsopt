from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class ScalarObjective(Protocol):
    def J(self) -> float: ...

    def dJ(self) -> object: ...


@dataclass(frozen=True)
class BananaWindingSurface:
    major_radius: float = 0.903
    minor_radius: float = 0.142
    nfp: int = 5
    stellsym: bool = True


@dataclass(frozen=True)
class HBTTFCoils:
    major_radius: float = 0.976
    minor_radius: float = 0.400
    n_coils: int = 20
    nfp: int = 1
    order: int = 1
    stellsym: bool = False


@dataclass(frozen=True)
class HBTVFCoils:
    rs: tuple[float, ...] = (
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        0.260350,
        1.572133,
        1.572133,
        1.572133,
        1.572133,
        1.572133,
        1.572133,
        1.572133,
        1.572133,
    )
    zs: tuple[float, ...] = (
        0.3508375,
        0.3635375,
        0.3762375,
        0.3889375,
        0.4016375,
        0.4143375,
        -0.3508375,
        -0.3635375,
        -0.3762375,
        -0.3889375,
        -0.4016375,
        -0.4143375,
        0.6505194,
        0.6505194,
        0.6505194,
        0.6505194,
        -0.6505194,
        -0.6505194,
        -0.6505194,
        -0.6505194,
    )
    sign_curr: tuple[int, ...] = (
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
    )


@dataclass(frozen=True)
class HardwareLimits:
    tf_current_ka_limits: tuple[float, float] = (-80.0, 0.0)
    banana_current_ka_limits: tuple[float, float] = (0.0, 16.0)
    banana_curv_p: int = 4
    max_curvature: float = 100.0
    max_length: float = 1.900
    min_ccdist: float = 0.0462
    min_csdist: float = 0.010


@dataclass(frozen=True)
class Stage2Weights:
    sqflux: float = 1.0
    length: float = 5.0e-2
    ccdist: float = 1.0e6
    csdist: float = 1.0e2
    curvature: float = 1.0e-2
    poloidal: float = 1.0e2
    width: float = 1.0e1
    global_curvature_radius: float = 1.0e3
    currents: float = 1.0e6


@dataclass(frozen=True)
class SingleStageWeights:
    nonqs: float = 1.0
    bres: float = 1.0e3
    iota: float = 1.0e4
    length: float = 5.0e-2
    ccdist: float = 1.0e6
    csdist: float = 1.0e2
    curvature: float = 1.0e-2
    poloidal: float = 1.0e2
    width: float = 1.0e1
    global_curvature_radius: float = 1.0e3
    currents: float = 1.0e6


@dataclass(frozen=True)
class BananaTargets:
    poloidal_deg: float = 70.0
    width_max: float = 0.197
    width_min: float = 0.050
    global_curvature_radius_min: float = 0.010
    global_curvature_radius_exp_weight: float = 0.010


@dataclass(frozen=True)
class BananaDriverPaths:
    biotsavart: Path
    boozersurface: Path
    surface: Path | None
    results: Path
    log: Path


@dataclass(frozen=True)
class EvaluationTracker:
    iterations: int = 0
    evaluations: int = 0
    started_at: float = 0.0


@dataclass(frozen=True)
class BoozerSolveState:
    iota: float
    G: float


@dataclass(frozen=True)
class WeightedTerm:
    name: str
    weight: float
    objective: ScalarObjective


HBT_TF = HBTTFCoils()
HBT_BANANA_WS = BananaWindingSurface()
HBT_VF = HBTVFCoils()
HARDWARE_LIMITS = HardwareLimits()
BANANA_TARGETS = BananaTargets()

N_TF = HBT_TF.n_coils
N_BANANA = 2 * HBT_BANANA_WS.nfp
N_PROXY = 1
N_VF = len(HBT_VF.rs)
TF_IDX = 0
BANANA_IDX = N_TF
PROXY_IDX = N_TF + N_BANANA
VF_IDX = N_TF + N_BANANA + N_PROXY
N_COILS = VF_IDX + N_VF

DEFAULT_BANANA_ORDER = 4
DEFAULT_PROXY_RZ = (HBT_BANANA_WS.major_radius, 0.0)
DEFAULT_BANANA_DOFS = {
    "phic(0)": 0.0600,
    "phic(1)": 0.0300,
    "thetac(0)": 0.5000,
    "thetas(1)": 0.1000,
}
