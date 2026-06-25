"""Shared equilibrium-path policy for single-stage and Stage 2 entrypoints."""

from __future__ import annotations

from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_ROOT.parents[1]
REPO_EQUILIBRIA_DIR = EXAMPLE_ROOT / "equilibria"
WORKSPACE_EQUILIBRIA_DIR = REPO_ROOT.parent / "DATABASE" / "EQUILIBRIA"
DEFAULT_EQUILIBRIA_DIR = REPO_EQUILIBRIA_DIR


def equilibrium_candidate_paths(
    *,
    plasma_surf_filename: str,
    equilibria_dir: str | Path = DEFAULT_EQUILIBRIA_DIR,
    equilibrium_path: str | Path | None = None,
    fallback_dirs: tuple[str | Path, ...] = (WORKSPACE_EQUILIBRIA_DIR,),
) -> tuple[Path, ...]:
    """Return ordered candidate paths for the requested equilibrium file."""
    if equilibrium_path is not None:
        return (Path(equilibrium_path),)

    candidates: list[Path] = []
    seen: set[str] = set()
    for directory in (equilibria_dir, *fallback_dirs):
        candidate = Path(directory) / plasma_surf_filename
        candidate_key = str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        candidates.append(candidate)
    return tuple(candidates)


def resolve_equilibrium_path(
    *,
    plasma_surf_filename: str,
    equilibria_dir: str | Path = DEFAULT_EQUILIBRIA_DIR,
    equilibrium_path: str | Path | None = None,
    fallback_dirs: tuple[str | Path, ...] = (WORKSPACE_EQUILIBRIA_DIR,),
) -> Path:
    """Resolve the first existing equilibrium file from the ordered candidates."""
    path = maybe_resolve_equilibrium_path(
        plasma_surf_filename=plasma_surf_filename,
        equilibria_dir=equilibria_dir,
        equilibrium_path=equilibrium_path,
        fallback_dirs=fallback_dirs,
    )
    if path is not None:
        return path
    return equilibrium_candidate_paths(
        plasma_surf_filename=plasma_surf_filename,
        equilibria_dir=equilibria_dir,
        equilibrium_path=equilibrium_path,
        fallback_dirs=fallback_dirs,
    )[0]


def maybe_resolve_equilibrium_path(
    *,
    plasma_surf_filename: str,
    equilibria_dir: str | Path = DEFAULT_EQUILIBRIA_DIR,
    equilibrium_path: str | Path | None = None,
    fallback_dirs: tuple[str | Path, ...] = (WORKSPACE_EQUILIBRIA_DIR,),
) -> Path | None:
    """Resolve the first existing equilibrium file from the ordered candidates, or return None."""
    candidates = equilibrium_candidate_paths(
        plasma_surf_filename=plasma_surf_filename,
        equilibria_dir=equilibria_dir,
        equilibrium_path=equilibrium_path,
        fallback_dirs=fallback_dirs,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
