import os
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"


def _run_import_probe(source):
    env = os.environ.copy()
    local_pythonpath = os.pathsep.join((str(REPO_ROOT), str(SRC_ROOT)))
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        local_pythonpath
        if existing_pythonpath is None
        else os.pathsep.join((local_pythonpath, existing_pythonpath))
    )
    probe_source = (
        textwrap.dedent(
            """
            from pathlib import Path
            import sys

            sys.meta_path = [
                finder
                for finder in sys.meta_path
                if finder.__class__.__module__ != "_simsopt_editable"
            ]

            from repo_bootstrap import bootstrap_local_simsopt

            bootstrap_local_simsopt(Path.cwd() / "src")
            """
        )
        + "\n"
        + textwrap.dedent(source)
    )
    return subprocess.run(
        [sys.executable, "-c", probe_source],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


class _WeakKey:
    pass


def test_surface_classifier_import_does_not_enter_field_package():
    result = _run_import_probe(
        """
        import sys
        from simsopt.geo.surface import SurfaceClassifier

        assert SurfaceClassifier.__module__ == "simsopt.geo.surface"
        assert not any(name.startswith("simsopt.field") for name in sys.modules)
        """
    )

    assert result.returncode == 0, result.stderr


def test_tracing_metadata_is_core_owned():
    from simsopt._core import tracing_metadata as metadata

    assert metadata.__name__ == "simsopt._core.tracing_metadata"
    assert "simsopt_jax_adapters" not in metadata.__file__


def test_tracing_metadata_records_levelset_classifier():
    from simsopt._core import tracing_metadata as metadata

    criterion = _WeakKey()
    classifier = _WeakKey()

    metadata.register_levelset_classifier(criterion, classifier)

    assert metadata.levelset_classifier_for(criterion) is classifier
