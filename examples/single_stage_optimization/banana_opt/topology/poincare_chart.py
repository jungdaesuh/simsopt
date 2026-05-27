from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import atan2, hypot, isfinite, pi

import numpy as np


@dataclass(frozen=True, slots=True)
class PoincareChart:
    """Maps section coordinates [R, Z] to radial labels and poloidal angles."""

    axis_r: float
    axis_z: float
    poloidal_orientation: int = 1
    radial_label_scale: float = 1.0

    def __post_init__(self) -> None:
        axis_r = float(self.axis_r)
        axis_z = float(self.axis_z)
        radial_label_scale = float(self.radial_label_scale)
        orientation = int(self.poloidal_orientation)
        if axis_r <= 0.0 or not isfinite(axis_r) or not isfinite(axis_z):
            raise ValueError("PoincareChart axis coordinates must be finite with R > 0")
        if orientation not in {-1, 1}:
            raise ValueError("PoincareChart poloidal_orientation must be +1 or -1")
        if radial_label_scale <= 0.0 or not isfinite(radial_label_scale):
            raise ValueError(
                "PoincareChart radial_label_scale must be finite and positive"
            )
        object.__setattr__(self, "axis_r", axis_r)
        object.__setattr__(self, "axis_z", axis_z)
        object.__setattr__(self, "poloidal_orientation", orientation)
        object.__setattr__(self, "radial_label_scale", radial_label_scale)

    def theta(self, state: Sequence[float]) -> float:
        if len(state) != 2:
            raise ValueError("PoincareChart state must be [R, Z]")
        return float(
            self.poloidal_orientation
            * atan2(float(state[1]) - self.axis_z, float(state[0]) - self.axis_r)
        )

    def radial_label(self, state: Sequence[float]) -> float:
        if len(state) != 2:
            raise ValueError("PoincareChart state must be [R, Z]")
        return float(
            hypot(float(state[0]) - self.axis_r, float(state[1]) - self.axis_z)
            / self.radial_label_scale
        )

    def unwrapped_thetas(self, states: Sequence[Sequence[float]]) -> np.ndarray:
        state_array = np.asarray(states, dtype=float)
        if state_array.ndim != 2 or state_array.shape[1] != 2:
            raise ValueError(
                f"PoincareChart states must have shape (n, 2), got {state_array.shape}"
            )
        angles = [
            self.poloidal_orientation
            * atan2(float(state[1]) - self.axis_z, float(state[0]) - self.axis_r)
            for state in state_array
        ]
        return np.unwrap(np.asarray(angles, dtype=float))

    def winding(self, states: Sequence[Sequence[float]]) -> float:
        angles = self.unwrapped_thetas(states)
        if angles.size < 2:
            raise ValueError("PoincareChart winding requires at least two states")
        return float((angles[-1] - angles[0]) / (2.0 * pi))
