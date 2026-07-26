"""Solve and publish bounded RCLS and GSCO wireframe states with JAX.

The host owns ``ToroidalWireframe`` geometry and accepted current assignment.
The constrained solve and fixed-shape multistep state transition stay in JAX.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax.core.wireframe_workflow import (
    WireframeGSCOLiveParams,
    wireframe_gsco_multistep_loop_jax,
)
from simsopt_jax_adapters.solve.wireframe import rcls_wireframe_jax

EXAMPLE_ID = "wireframe-optimization"


@dataclass(frozen=True)
class ExampleResult:
    solution_oracle_error: float
    constraint_residual_norm: float
    published_current_error: float
    objective: float
    gsco_nonfinal_steps: int
    gsco_enclosed_segments: int
    status: Literal["ok", "failed"]

    def json_object(self) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            "backend_mode": get_backend_mode(),
            "platform": jax.devices()[0].platform,
            "precision": get_resolved_precision(),
            "status": self.status,
            "observables": {
                "solution_oracle_error": self.solution_oracle_error,
                "constraint_residual_norm": self.constraint_residual_norm,
                "published_current_error": self.published_current_error,
                "objective": self.objective,
                "gsco_nonfinal_steps": self.gsco_nonfinal_steps,
                "gsco_enclosed_segments": self.gsco_enclosed_segments,
            },
        }


def _wireframe() -> ToroidalWireframe:
    surface = SurfaceRZFourier(nfp=2, mpol=1, ntor=0)
    surface.set_rc(0, 0, 2.0)
    surface.set_rc(1, 0, 0.7)
    surface.set_zs(1, 0, 0.7)
    return ToroidalWireframe(surface, 4, 6)


def _constrained_oracle(
    response: np.ndarray,
    target: np.ndarray,
    regularization: float,
    constraint_matrix: np.ndarray,
    constraint_target: np.ndarray,
) -> np.ndarray:
    hessian = response.T @ response + regularization**2 * np.eye(response.shape[1])
    kkt = np.block(
        [
            [hessian, constraint_matrix.T],
            [constraint_matrix, np.zeros((constraint_matrix.shape[0],) * 2)],
        ]
    )
    right_hand_side = np.vstack((response.T @ target, constraint_target))
    return np.linalg.solve(kkt, right_hand_side)[: response.shape[1]]


def _gsco_transition() -> tuple[int, int]:
    response = jnp.asarray(
        (
            (1.0, -0.3, 0.2, 0.0, 0.1, -0.2),
            (0.2, 0.8, -0.4, 0.3, 0.0, 0.1),
            (-0.1, 0.2, 0.7, -0.5, 0.3, 0.0),
        ),
        dtype=jnp.float64,
    )
    loops = jnp.asarray(((0, 1, 2, 3), (2, 3, 4, 5)), dtype=jnp.int32)
    segments = jnp.asarray(
        ((0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)),
        dtype=jnp.int32,
    )
    connections = jnp.asarray(
        ((0, 3, 4, 0), (0, 1, 5, 0), (1, 2, 4, 0), (2, 3, 5, 0)),
        dtype=jnp.int32,
    )
    params = WireframeGSCOLiveParams(
        A=response,
        loops=loops,
        free_loops=jnp.ones(2, dtype=jnp.int32),
        segments=segments,
        connections=connections,
        default_current=jnp.asarray(0.2, dtype=jnp.float64),
        max_current=jnp.asarray(1.0, dtype=jnp.float64),
        lambda_s=jnp.asarray(0.15, dtype=jnp.float64),
        tol=jnp.asarray(2.0e-4, dtype=jnp.float64),
        max_loop_count=1,
        no_crossing=False,
        no_new_coils=False,
        match_current=False,
    )
    multistep = wireframe_gsco_multistep_loop_jax(
        params,
        jnp.asarray((0.1, -0.2, 0.3), dtype=jnp.float64),
        jnp.zeros((6, 1), dtype=jnp.float64),
        jnp.asarray((1, 0), dtype=jnp.int32),
        loops,
        jnp.asarray(((1, 1, 1, 1), (0, 0, 0, 0)), dtype=jnp.int32),
        jnp.zeros(6, dtype=bool),
        max_iter_per_step=0,
        max_outer_steps=1,
        initial_current_fraction=0.2,
        current_scale=1.0,
        min_coil_size=1,
        final_max_current=0.22,
    )
    return (
        int(np.asarray(multistep.nonfinal_steps)),
        int(np.count_nonzero(np.asarray(multistep.enclosed_segment_mask))),
    )


def _solve() -> ExampleResult:
    wireframe = _wireframe()
    rng = np.random.default_rng(20260726)
    response = rng.standard_normal((wireframe.n_segments + 10, wireframe.n_segments))
    target = rng.standard_normal((response.shape[0], 1))
    regularization = 0.1
    constraint_matrix, constraint_target = wireframe.constraint_matrices(
        assume_no_crossings=False,
        remove_constrained_segments=True,
    )
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.intp)

    result = rcls_wireframe_jax(
        wireframe,
        response,
        target,
        regularization,
        assume_no_crossings=False,
    )
    solution = np.asarray(result.x, dtype=np.float64)
    oracle_free = _constrained_oracle(
        response[:, free_segments],
        target,
        regularization,
        np.asarray(constraint_matrix, dtype=np.float64),
        np.asarray(constraint_target, dtype=np.float64),
    )
    oracle = np.zeros_like(solution)
    oracle[free_segments] = oracle_free
    solution_oracle_error = float(np.linalg.norm(solution - oracle, ord=np.inf))
    constraint_residual_norm = float(
        np.linalg.norm(constraint_matrix @ solution[free_segments] - constraint_target)
    )

    wireframe.currents[:] = solution.ravel()
    published_current_error = float(
        np.linalg.norm(wireframe.currents - solution.ravel())
    )
    gsco_nonfinal_steps, gsco_enclosed_segments = _gsco_transition()
    is_correct = bool(
        solution_oracle_error <= 1.0e-10
        and constraint_residual_norm <= 1.0e-10
        and published_current_error == 0.0
        and gsco_nonfinal_steps == 1
        and gsco_enclosed_segments == 4
    )
    return ExampleResult(
        solution_oracle_error=solution_oracle_error,
        constraint_residual_norm=constraint_residual_norm,
        published_current_error=published_current_error,
        objective=float(np.asarray(result.f)),
        gsco_nonfinal_steps=gsco_nonfinal_steps,
        gsco_enclosed_segments=gsco_enclosed_segments,
        status="ok" if is_correct else "failed",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = _solve()
    if options.json:
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"RCLS objective={result.objective:.8e}")
        print(f"constraint residual={result.constraint_residual_norm:.3e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
