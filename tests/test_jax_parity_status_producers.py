"""Source ratchet: the optimizer-backed parity producers cannot relabel a budget stop.

Seven cases used ``"converged" if success else "failed"`` with ``success`` meaning
"finite and decreased", so an L-BFGS-B run that stopped on its iteration cap was
published as converged. They now take label and ``success`` from
``examples.jax.parity.terminal_status``. This pins that at the source level, where
it costs nothing to run; the lane-level behaviour is pinned by the integration
tests of each case.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CASES = Path(__file__).resolve().parents[1] / "examples" / "jax" / "parity" / "cases"
HELPERS = frozenset({"lane_terminal_status", "normalized_terminal_status"})

#: producer module -> optimizer-backed ``LaneObservation`` constructions in it
PRODUCERS = {
    "native_stage_two_optimization.py": 2,
    "native_stage_two_optimization_planar_coils.py": 2,
    "native_stage_two_optimization_stochastic.py": 2,
    "native_stage_two_optimization_finitebuild.py": 2,
    "native_coil_forces.py": 2,
    "native_strain_optimization.py": 2,
    "native_boozerqa.py": 1,
}


def _is_constant(node: ast.expr, value: str) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


@pytest.mark.parametrize(("module", "observations"), sorted(PRODUCERS.items()))
def test_producer_takes_its_label_from_the_shared_terminal_status(
    module: str,
    observations: int,
) -> None:
    tree = ast.parse((CASES / module).read_text(encoding="utf-8"))
    nodes = tuple(ast.walk(tree))

    predicate_ternaries = [
        node
        for node in nodes
        if isinstance(node, ast.IfExp)
        and _is_constant(node.body, "converged")
        and _is_constant(node.orelse, "failed")
    ]
    assert predicate_ternaries == []

    helper_calls = [
        node
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in HELPERS
    ]
    lane_observations = [
        node
        for node in nodes
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "LaneObservation"
    ]
    assert len(lane_observations) == observations
    assert len(helper_calls) == observations
