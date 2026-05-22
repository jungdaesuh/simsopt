"""Shared test fixtures for Boozer derivative parity artifact generation."""

from __future__ import annotations

import json
from pathlib import Path


def write_synthetic_outer_optimizer_progress(path: Path) -> Path:
    """Write the minimal progress event required by the repro CLI."""
    payload = {
        "events": [
            {
                "label": "objective_evaluation",
                "line_search_evaluation": 4,
                "accepted_iteration_target": 1,
                "optimizer_method": "synthetic-test",
                "backend": "cpu",
                "objective": {"value": 0.0},
                "candidate_optimizer_dofs": {
                    "values": [0.0],
                    "size": 1,
                    "inf_norm": 0.0,
                },
            }
        ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
