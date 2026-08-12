from pathlib import Path

from benchmarks.run_single_stage_fullspace_projected_hvp_canary import (
    SOURCE_PATHS,
    TRUST_RADIUS,
    _terminal_status,
    _tracked_paths,
)


def test_terminal_status_fails_closed() -> None:
    assert _terminal_status(usable=False, supported=True) == "CANARY_NOT_USABLE"
    assert (
        _terminal_status(usable=True, supported=False)
        == "NOT_SUPPORTED_BY_ONE_STEP_CANARY"
    )
    assert (
        _terminal_status(usable=True, supported=True) == "SUPPORTED_BY_ONE_STEP_CANARY"
    )


def test_frozen_radius_is_small_projected_canary_radius() -> None:
    assert TRUST_RADIUS == 2.0**-10


def test_source_manifest_includes_route_and_excludes_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = _tracked_paths(repo_root)

    assert set(SOURCE_PATHS).issubset(paths)
    assert all(
        path.name == ".env.example"
        or (path.name != ".env" and not path.name.startswith(".env."))
        for path in paths
    )
