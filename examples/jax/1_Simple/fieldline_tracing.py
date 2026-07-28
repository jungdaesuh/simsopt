"""Trace a field line and charged particle in a JAX Reiman field.

Host boundary: structured results are copied to the host only after immutable
traces finish; optional plotting belongs after this boundary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from simsopt_jax.examples import ExecutionScale, example_runtime_metadata
from simsopt_jax.core.tracing import (
    FieldlineTracingSpec,
    FullorbitTracingSpec,
    trace_fieldline,
    trace_fullorbit,
)
from simsopt_jax_adapters.field.reiman import ReimanJAX

EXAMPLE_ID = "fieldline-and-particle-tracing"
RADIUS = 1.4
FIELDLINE_TMAX = 0.7


@dataclass(frozen=True)
class ExampleResult:
    final_state: tuple[float, ...]
    analytic_final_state: tuple[float, ...]
    integrator_status: int
    event_count: int
    particle_status: int
    particle_energy_relative_error: float
    status: Literal["ok", "failed"]

    def json_object(self, scale: ExecutionScale) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            **example_runtime_metadata(scale),
            "status": self.status,
            "observables": {
                "final_state": self.final_state,
                "analytic_final_state": self.analytic_final_state,
                "integrator_status": self.integrator_status,
                "event_count": self.event_count,
                "particle_status": self.particle_status,
                "particle_energy_relative_error": self.particle_energy_relative_error,
            },
        }


def _solve(smoke: bool) -> ExampleResult:
    field = ReimanJAX(iota0=0.0, iota1=0.0, k=[2], epsilonk=[0.0], m0=1)
    fieldline_spec = FieldlineTracingSpec(
        tmax=FIELDLINE_TMAX,
        rtol=1.0e-11,
        atol=1.0e-12,
        max_steps=64 if smoke else 256,
        dtmax=0.05,
        max_phi_hits=8,
    )
    fieldline = trace_fieldline(
        fieldline_spec,
        jax.device_put(np.asarray((RADIUS, 0.0, 0.0), dtype=np.float64)),
        field.jax_B_at,
    )
    fieldline_trajectory = np.asarray(
        jax.device_get(fieldline.trajectory),
        dtype=np.float64,
    )
    final_state_array = fieldline_trajectory[
        int(jax.device_get(fieldline.steps_taken)), 1:4
    ]
    angle = -FIELDLINE_TMAX / RADIUS
    analytic = np.asarray(
        (RADIUS * np.cos(angle), RADIUS * np.sin(angle), 0.0), dtype=np.float64
    )

    particle_spec = FullorbitTracingSpec(
        tmax=0.02,
        rtol=1.0e-10,
        atol=1.0e-12,
        max_steps=96 if smoke else 384,
        dtmax=0.002,
        max_phi_hits=8,
    )
    particle = trace_fullorbit(
        particle_spec,
        jax.device_put(np.asarray((RADIUS, 0.0, 0.0, 0.1, 0.2, 0.3), dtype=np.float64)),
        field.jax_B_at,
        m=1.0,
        q=1.0,
    )
    particle_trajectory = np.asarray(
        jax.device_get(particle.trajectory),
        dtype=np.float64,
    )
    particle_final = particle_trajectory[int(jax.device_get(particle.steps_taken)), 1:7]
    initial_energy = 0.5 * float(np.dot((0.1, 0.2, 0.3), (0.1, 0.2, 0.3)))
    final_energy = 0.5 * float(np.dot(particle_final[3:], particle_final[3:]))
    energy_relative_error = abs(final_energy - initial_energy) / initial_energy
    fieldline_status = int(jax.device_get(fieldline.status))
    event_count = int(jax.device_get(fieldline.phi_hits_count))
    particle_status = int(jax.device_get(particle.status))
    is_correct = bool(
        fieldline_status == 0
        and event_count == 0
        and np.allclose(final_state_array, analytic, rtol=1.0e-9, atol=1.0e-10)
        and particle_status == 0
        and energy_relative_error <= 1.0e-9
    )
    return ExampleResult(
        final_state=tuple(float(value) for value in final_state_array),
        analytic_final_state=tuple(float(value) for value in analytic),
        integrator_status=fieldline_status,
        event_count=event_count,
        particle_status=particle_status,
        particle_energy_relative_error=energy_relative_error,
        status="ok" if is_correct else "failed",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = _solve(options.smoke)
    if options.json:
        scale: ExecutionScale = "bounded" if options.smoke else "native_default"
        print(json.dumps(result.json_object(scale), sort_keys=True))
    else:
        print(f"fieldline final state={result.final_state}")
        print(
            f"particle energy relative error={result.particle_energy_relative_error:.3e}"
        )
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
