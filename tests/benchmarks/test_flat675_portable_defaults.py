"""Portable path configuration for the instrument-bound flat-675 CLIs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT_ENVIRONMENT = "SIMSOPT_GENUINE675_SOURCE_ROOT"


@pytest.fixture
def cli_environment(tmp_path: Path) -> dict[str, str]:
    """Supply only the import surface needed to exercise argument parsing."""
    package_root = tmp_path / "instrument"
    runtime_root = package_root / "simsopt_jax" / "runtime"
    runtime_root.mkdir(parents=True)
    (package_root / "simsopt_jax" / "__init__.py").write_text("")
    (runtime_root / "__init__.py").write_text("")
    (runtime_root / "genuine_675_dynamic.py").write_text(
        "class Genuine675LbfgsbPolicy:\n    pass\n"
    )
    (runtime_root / "single_stage_fullspace_675.py").write_text(
        "class _Formulation:\n"
        "    semantic_sha256 = '0' * 64\n"
        "GENUINE_FULLSPACE_675 = _Formulation()\n"
    )
    (runtime_root / "validation_ladder_common.py").write_text(
        "def repo_pythonpath_env(*args, **kwargs):\n    return {}\n"
    )
    environment = os.environ.copy()
    environment.pop(SOURCE_ROOT_ENVIRONMENT, None)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(package_root), str(REPOSITORY_ROOT))
    )
    return environment


@pytest.mark.parametrize(
    "script",
    (
        "benchmarks/flat675_fused_campaign.py",
        "benchmarks/flat675_promotion_robustness.py",
        "benchmarks/genuine_675_fair_bar.py",
    ),
)
def test_help_needs_no_external_input_configuration(
    script: str,
    cli_environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        (sys.executable, "-S", str(REPOSITORY_ROOT / script), "--help"),
        cwd=REPOSITORY_ROOT,
        env=cli_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_empty_source_root_environment_is_not_a_cwd_default(
    cli_environment: dict[str, str],
) -> None:
    environment = {**cli_environment, SOURCE_ROOT_ENVIRONMENT: ""}
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            str(REPOSITORY_ROOT / "benchmarks/genuine_675_fair_bar.py"),
            "phase1",
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert "--source-root is required" in completed.stderr


@pytest.mark.parametrize(
    ("script", "arguments"),
    (
        (
            "benchmarks/flat675_fused_campaign.py",
            ("pairs", "--output-root", "{output}"),
        ),
        (
            "benchmarks/flat675_promotion_robustness.py",
            ("run", "--run-root", "{output}"),
        ),
        (
            "benchmarks/genuine_675_fair_bar.py",
            ("--output-root", "{output}", "phase1"),
        ),
    ),
)
def test_missing_external_inputs_fail_before_creating_output(
    script: str,
    arguments: tuple[str, ...],
    cli_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "output"
    command_arguments = tuple(
        str(output_root) if value == "{output}" else value for value in arguments
    )

    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            str(REPOSITORY_ROOT / script),
            *command_arguments,
        ),
        cwd=REPOSITORY_ROOT,
        env=cli_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2, completed.stderr
    assert not output_root.exists()


def test_default_output_roots_are_repository_local(
    cli_environment: dict[str, str],
) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            (
                "from benchmarks import flat675_fused_campaign as fused\n"
                "from benchmarks import flat675_promotion_robustness as promotion\n"
                "from benchmarks import genuine_675_fair_bar as fair_bar\n"
                "print(fused.OUTPUT_ROOT)\n"
                "print(promotion.OUTPUT_ROOT)\n"
                "print(fair_bar.DEFAULT_OUTPUT_ROOT)\n"
            ),
        ),
        cwd=REPOSITORY_ROOT,
        env=cli_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(REPOSITORY_ROOT / ".artifacts" / "flat675_fused_campaign"),
        str(REPOSITORY_ROOT / ".artifacts" / "flat675_promotion"),
        str(REPOSITORY_ROOT / ".artifacts" / "genuine675_fair_bar"),
    ]


def test_default_output_roots_are_ignored_by_the_tracked_policy(
    tmp_path: Path,
) -> None:
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    (clone_root / ".gitignore").write_bytes(
        (REPOSITORY_ROOT / ".gitignore").read_bytes()
    )
    subprocess.run(("git", "init", "-q"), cwd=clone_root, check=True)
    subprocess.run(("git", "add", ".gitignore"), cwd=clone_root, check=True)

    output_files = (
        ".artifacts/flat675_fused_campaign/result.json",
        ".artifacts/flat675_promotion/result.json",
        ".artifacts/genuine675_fair_bar/result.json",
    )
    for relative_path in output_files:
        output_file = clone_root / relative_path
        output_file.parent.mkdir(parents=True)
        output_file.write_text("{}\n")

    ignored = subprocess.run(
        ("git", "check-ignore", "-v", *output_files),
        cwd=clone_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert ignored.returncode == 0, ignored.stderr
    assert len(ignored.stdout.splitlines()) == len(output_files)
    assert all(line.startswith(".gitignore:") for line in ignored.stdout.splitlines())

    unrelated_output = clone_root / ".artifacts" / "other" / "result.json"
    unrelated_output.parent.mkdir(parents=True)
    unrelated_output.write_text("{}\n")
    unrelated = subprocess.run(
        ("git", "check-ignore", str(unrelated_output.relative_to(clone_root))),
        cwd=clone_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert unrelated.returncode == 1


def test_source_root_environment_reaches_fused_parser(
    cli_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    environment_source_root = tmp_path / "environment-instrument"
    environment = {
        **cli_environment,
        SOURCE_ROOT_ENVIRONMENT: str(environment_source_root),
    }
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            (
                "from benchmarks import flat675_fused_campaign as fused\n"
                "arguments = fused._parser().parse_args([\n"
                "    'pairs', '--input-manifest', 'bundle.json', "
                "'--rung', 'b3'\n"
                "])\n"
                "print(arguments.source_root)\n"
            ),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [str(environment_source_root)]


def test_fair_bar_configure_runtime_paths_binds_both_roots(
    cli_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "instrument"
    output_root = tmp_path / "artifacts"
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            (
                "from pathlib import Path\n"
                "from benchmarks import genuine_675_fair_bar as fair_bar\n"
                "import sys\n"
                "fair_bar.configure_runtime_paths(\n"
                "    source_root=Path(sys.argv[1]), output_root=Path(sys.argv[2])\n"
                ")\n"
                "print(fair_bar.SOURCE_ROOT)\n"
                "print(fair_bar.OUTPUT_ROOT)\n"
            ),
            str(source_root),
            str(output_root),
        ),
        cwd=REPOSITORY_ROOT,
        env=cli_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(source_root.resolve()),
        str(output_root.resolve()),
    ]


def test_fair_bar_cli_source_root_overrides_environment_before_dispatch(
    cli_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    instrument_root = Path(cli_environment["PYTHONPATH"].split(os.pathsep)[0])
    environment = {
        **cli_environment,
        SOURCE_ROOT_ENVIRONMENT: str(tmp_path / "environment-instrument"),
    }
    output_root = tmp_path / "fair-output"
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            (
                "from benchmarks import genuine_675_fair_bar as fair_bar\n"
                "import sys\n"
                "def harmless_handler(arguments):\n"
                "    print(fair_bar.SOURCE_ROOT)\n"
                "    print(fair_bar.OUTPUT_ROOT)\n"
                "fair_bar.cmd_phase1 = harmless_handler\n"
                "sys.argv = [\n"
                "    'genuine_675_fair_bar.py',\n"
                "    '--source-root', sys.argv[1],\n"
                "    '--source-manifest', sys.argv[2],\n"
                "    '--output-root', sys.argv[3],\n"
                "    'phase1',\n"
                "]\n"
                "raise SystemExit(fair_bar.main())\n"
            ),
            str(instrument_root),
            str(tmp_path / "bundle" / "manifest.json"),
            str(output_root),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(instrument_root.resolve()),
        str(output_root.resolve()),
    ]
    assert output_root.is_dir()


def test_fused_main_binds_shared_roots_and_state_before_dispatch(
    cli_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cli-instrument"
    output_root = tmp_path / "fused-output"
    environment = {
        **cli_environment,
        SOURCE_ROOT_ENVIRONMENT: str(tmp_path / "environment-instrument"),
    }
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            (
                "from benchmarks import flat675_fused_campaign as fused\n"
                "from benchmarks import genuine_675_fair_bar as fair_bar\n"
                "import sys\n"
                "def harmless_handler(arguments):\n"
                "    print(fused.INSTRUMENT_ROOT)\n"
                "    print(fused.OUTPUT_ROOT)\n"
                "    print(fused.CAMPAIGN_STATE_PATH)\n"
                "    print(fair_bar.SOURCE_ROOT)\n"
                "    print(fair_bar.OUTPUT_ROOT)\n"
                "fused.cmd_pairs = harmless_handler\n"
                "raise SystemExit(fused.main([\n"
                "    'pairs',\n"
                "    '--input-manifest', 'bundle.json',\n"
                "    '--source-root', sys.argv[1],\n"
                "    '--output-root', sys.argv[2],\n"
                "    '--rung', 'b3',\n"
                "]))\n"
            ),
            str(source_root),
            str(output_root),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [
        str(source_root.resolve()),
        str(output_root.resolve()),
        str(output_root.resolve() / "campaign_state.json"),
        str(source_root.resolve()),
        str(output_root.resolve()),
    ]
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("source_root_case", "expected_returncode", "expected_stderr", "expected_stdout"),
    (
        ("missing", 2, "--source-root is required", ""),
        ("wrong", 1, "simsopt_jax resolved outside the instrument tree", ""),
        ("matching", 0, "", "validate-handler-reached\n"),
    ),
)
def test_fair_bar_validate_requires_and_verifies_the_source_root(
    source_root_case: str,
    expected_returncode: int,
    expected_stderr: str,
    expected_stdout: str,
    cli_environment: dict[str, str],
    tmp_path: Path,
) -> None:
    imported_root = Path(cli_environment["PYTHONPATH"].split(os.pathsep)[0])
    if source_root_case == "matching":
        source_root = imported_root
    elif source_root_case == "wrong":
        source_root = tmp_path / "wrong-instrument"
    else:
        source_root = None
    output_root = tmp_path / "validate-output"
    command = (
        "from benchmarks import genuine_675_fair_bar as fair_bar\n"
        "import sys\n"
        "def harmless_validate(arguments):\n"
        "    print('validate-handler-reached')\n"
        "fair_bar.cmd_validate = harmless_validate\n"
        "arguments = ['genuine_675_fair_bar.py', '--output-root', sys.argv[2]]\n"
        "if sys.argv[1] != 'NONE':\n"
        "    arguments.extend(('--source-root', sys.argv[1]))\n"
        "arguments.extend(('validate', 'run-directory'))\n"
        "sys.argv = arguments\n"
        "raise SystemExit(fair_bar.main())\n"
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            "-c",
            command,
            "NONE" if source_root is None else str(source_root),
            str(output_root),
        ),
        cwd=REPOSITORY_ROOT,
        env=cli_environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == expected_returncode, completed.stderr
    assert expected_stderr in completed.stderr
    assert completed.stdout == expected_stdout
    assert not output_root.exists()
