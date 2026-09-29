"""Workflow scale selection must be explicit and fail closed."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / ".github" / "workflows" / "jax_smoke.yml"
AUTHORITY = ROOT / ".github" / "workflows" / "jax_gpu_parity.yml"
# Any invocation of the parity runner: a script path (relative, ``./`` or
# absolute, under any interpreter) or the ``-m`` module form, matched after
# ``_shell_normalised`` removes quotes and backslash-newline continuations.
# Text matching also counts a mention that is not executed (a comment, say);
# that can only add a job to the asserted set, so it fails closed.
_PARITY_INVOCATION = re.compile(
    r"(?:\S*/)?examples/jax/run_parity\.py\b|-m\s+examples\.jax\.run_parity\b"
)
_LINE_CONTINUATION = re.compile(r"\\\n\s*")
_SHELL_QUOTES = re.compile(r"[\"']")
_JOB_HEADER = re.compile(r"^  ([A-Za-z0-9_-]+):\n", re.MULTILINE)


def _jobs(path: Path) -> dict[str, str]:
    return _jobs_from_text(path.read_text(encoding="utf-8"))


def _jobs_from_text(workflow: str) -> dict[str, str]:
    """Return each top-level job's text, keyed by job id."""
    jobs_section = workflow.split("\njobs:\n", maxsplit=1)[1]
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


def _shell_normalised(text: str) -> str:
    """Join backslash-continued lines and drop shell quotes."""
    return _SHELL_QUOTES.sub("", _LINE_CONTINUATION.sub(" ", text))


def _parity_commands(job: str) -> list[str]:
    return _PARITY_INVOCATION.split(_shell_normalised(job))[1:]


def _parity_jobs(jobs: dict[str, str]) -> set[str]:
    return {job_id for job_id, job in jobs.items() if _parity_commands(job)}


def test_parity_invocations_are_recognised_in_every_form() -> None:
    forms = (
        "python examples/jax/run_parity.py --scale bounded",
        "python3 examples/jax/run_parity.py --scale bounded",
        "python ./examples/jax/run_parity.py --scale bounded",
        "python /work/simsopt/examples/jax/run_parity.py --scale bounded",
        "python -m examples.jax.run_parity --scale bounded",
        'python -m "examples.jax.run_parity" --scale bounded',
        "python -m 'examples.jax.run_parity' --scale bounded",
        "python -m \\\n              examples.jax.run_parity \\\n              --scale bounded",
        'python "examples/jax/run_parity.py" --scale bounded',
    )
    for form in forms:
        assert len(_parity_commands(form)) == 1, form
    assert _parity_commands("python examples/jax/run_examples.py") == []

    third_job_runs = {
        "python3 script": "      - run: python3 examples/jax/run_parity.py --scale bounded\n",
        "quoted module": (
            '      - run: python -m "examples.jax.run_parity" --scale bounded\n'
        ),
        "continued module": (
            "      - run: |\n"
            "          python -m \\\n"
            "            examples.jax.run_parity \\\n"
            "            --scale bounded\n"
        ),
    }
    for form, run in third_job_runs.items():
        extra_job = (
            f"  extra-parity:\n    runs-on: [self-hosted, gpu]\n    steps:\n{run}"
        )
        mutated = AUTHORITY.read_text(encoding="utf-8").rstrip("\n") + "\n" + extra_job
        assert _parity_jobs(_jobs_from_text(mutated)) == {
            "native-jax-example-parity",
            "jax-gpu-strict-purity",
            "extra-parity",
        }, form


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
    cpu_parity_commands = _parity_commands(smoke)
    assert len(cpu_parity_commands) == 1
    assert "--scale bounded" in cpu_parity_commands[0]


def test_scheduled_workflow_runs_bounded_parity_in_two_distinct_gpu_jobs() -> None:
    """The dispatch/schedule workflow runs the bounded native/JAX parity twice.

    ``native-jax-example-parity`` is the parity authority: bounded, plus the
    manual native-default run, with a long budget and 30-day receipts.
    ``jax-gpu-strict-purity`` runs it once more beside the example and strict
    transfer-guard smoke slices, under deterministic XLA GPU ops and a short
    budget, with its own receipts. Every other job runs no parity command.
    """
    jobs = _jobs(AUTHORITY)
    authority = jobs["native-jax-example-parity"]
    strict = jobs["jax-gpu-strict-purity"]

    assert _parity_jobs(jobs) == {"native-jax-example-parity", "jax-gpu-strict-purity"}
    authority_scales = [
        command.split("--scale ", maxsplit=1)[1].split(maxsplit=1)[0]
        for command in _parity_commands(authority)
    ]
    strict_scales = [
        command.split("--scale ", maxsplit=1)[1].split(maxsplit=1)[0]
        for command in _parity_commands(strict)
    ]
    assert authority_scales == ["bounded", "native_default"]
    assert strict_scales == ["bounded"]
    for job in (authority, strict):
        assert "runs-on: [self-hosted, gpu]" in job
        assert "--case all-applicable" in job
        assert "--lanes native-cpu,jax-cpu,jax-gpu" in job
        assert "SIMSOPT_JAX_TRANSFER_GUARD: disallow" in job
    assert "XLA_FLAGS: --xla_gpu_exclude_nondeterministic_ops=true" in strict
    assert "XLA_FLAGS" not in authority
    assert "timeout-minutes: 60" in strict
    assert "timeout-minutes: 720" in authority
    assert "name: jax-native-example-parity-gpu-strict" in strict
    assert "name: jax-native-example-parity-scheduled" in authority


def test_native_default_authority_is_manual_and_explicit() -> None:
    source = AUTHORITY.read_text(encoding="utf-8")

    assert "run_native_default:" in source
    assert (
        "if: github.event_name == 'workflow_dispatch' && inputs.run_native_default"
        in source
    )
    assert "--scale bounded" in source
    assert "--scale native_default" in source
