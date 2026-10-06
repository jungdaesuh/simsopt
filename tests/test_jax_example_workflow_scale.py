"""Workflow scale selection must be explicit and fail closed."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / ".github" / "workflows" / "jax_smoke.yml"
AUTHORITY = ROOT / ".github" / "workflows" / "jax_gpu_parity.yml"
_JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\n", re.MULTILINE)


def _jobs(path: Path) -> dict[str, str]:
    """Return each top-level job's text, keyed by job id."""
    jobs_section = path.read_text(encoding="utf-8").split("\njobs:\n", maxsplit=1)[1]
    headers = list(_JOB_HEADER.finditer(jobs_section))
    return {
        header.group(1): jobs_section[
            header.end() : (
                headers[index + 1].start()
                if index + 1 < len(headers)
                else len(jobs_section)
            )
        ]
        for index, header in enumerate(headers)
    }


def test_pr_example_commands_select_bounded_scale_explicitly() -> None:
    smoke_jobs = _jobs(SMOKE)
    smoke = "".join(smoke_jobs.values())
    # The strict GPU job runs from the dispatch/schedule workflow, so that no
    # pull_request-triggered job runs on the self-hosted runner.
    gpu_strict = _jobs(AUTHORITY)["jax-gpu-strict-purity"]

    assert "jax-gpu-strict-purity" not in smoke_jobs
    assert "run_examples.py --device cpu --scale bounded" in smoke
    assert "run_examples.py --device cpu --intent parity --scale bounded" in smoke
    assert "run_examples.py --device gpu --scale bounded" in gpu_strict
    assert "run_examples.py --device gpu --intent parity --scale bounded" in gpu_strict


def test_same_state_mirror_parity_runs_on_cpu_and_gpu() -> None:
    same_state = "tests/jax/test_mirror_same_state_parity.py"
    assert same_state in _jobs(SMOKE)["jax-public-integration"]
    gpu_parity = _jobs(AUTHORITY)["gpu-parity"]
    assert same_state in gpu_parity
    assert "steps.mirror_same_state.outcome" in gpu_parity
