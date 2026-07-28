"""Construction fingerprint coverage for stochastic Stage-II parity."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from examples.jax.parity.cases.native_stage_two_optimization_stochastic import (
    _effective_fingerprint,
)
from examples.jax.parity.input_bundle import InputBundle


@dataclass(frozen=True)
class _SurfaceConstruction:
    local_full_x: np.ndarray
    quadpoints_phi: np.ndarray
    quadpoints_theta: np.ndarray
    nfp: int

    def gamma(self) -> np.ndarray:
        raise AssertionError("derived geometry must not define construction identity")

    def normal(self) -> np.ndarray:
        raise AssertionError("derived geometry must not define construction identity")


@dataclass(frozen=True)
class _CurveConstruction:
    local_full_x: np.ndarray


def test_stochastic_fingerprint_uses_defining_coordinates() -> None:
    bundle = InputBundle(
        schema_version=2,
        case_id="native-stage-two-optimization-stochastic",
        scale="bounded",
        random_seed=0,
        configuration=MappingProxyType({"num_base_curves": 1}),
        configuration_fingerprint="configuration",
        arrays=MappingProxyType({}),
        input_fingerprint="input",
    )
    surface = _SurfaceConstruction(
        local_full_x=np.asarray((1.0, 2.0)),
        quadpoints_phi=np.asarray((0.0, 0.5)),
        quadpoints_theta=np.asarray((0.0, 0.25)),
        nfp=2,
    )
    curves = (_CurveConstruction(local_full_x=np.asarray((3.0, 4.0))),)

    fingerprint = _effective_fingerprint(
        bundle,
        {"initial_parameters": np.asarray((5.0, 6.0))},
        surface,
        curves,
    )

    assert len(fingerprint) == 64
