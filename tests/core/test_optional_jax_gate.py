"""The optional-JAX gate skips only for an unsupported runtime, never for a broken package."""

import os
import subprocess
import sys
from pathlib import Path
from unittest import TestCase, skipIf

from unittest_jax_support import JAX_IMPORT_ERROR

_TESTS_ROOT = Path(__file__).resolve().parents[1]
_FRESH_PROCESS = Path(__file__).resolve().with_name("fresh_process")


def _run_child(script: str, *arguments: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(_TESTS_ROOT), environment.get("PYTHONPATH", ""))
    )
    return subprocess.run(
        (sys.executable, str(_FRESH_PROCESS / script), *arguments),
        cwd=_TESTS_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@skipIf(JAX_IMPORT_ERROR is not None, JAX_IMPORT_ERROR or "")
class TestOptionalJaxGate(TestCase):
    def test_broken_simsopt_jax_import_fails_instead_of_skipping(self):
        """An ImportError inside simsopt_jax on a supported runtime fails the run."""
        completed = _run_child(
            "unittest_with_broken_simsopt_jax.py", "core.test_state_tokens", "-v"
        )
        self.assertNotEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("import of simsopt_jax.core.curve_geometry halted", completed.stderr)
        self.assertNotIn("skipped", completed.stderr)

    def test_unsupported_jax_discovery_loads_every_test_module(self):
        """With an unsupported JAX every test module still imports, so discovery can skip."""
        completed = _run_child("discover_with_unsupported_jax.py")
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_version_gate_accepts_every_0_12_release_and_rejects_0_11(self):
        """The floor is the 0.12 series: four-part and local releases pass, 0.11 fails."""
        from simsopt_jax._requirements import _require_jax_version

        for version in ("0.12.0", "0.12.0.1", "0.12.1", "0.12.0.1+cuda12", "0.13.0", "1.0.0"):
            with self.subTest(version=version):
                _require_jax_version("jax", version)
        for version in ("0.11.2", "0.11.9.1", "0.10.0", "0.9.0"):
            with self.subTest(version=version), self.assertRaisesRegex(
                ImportError, "requires jax >= 0.12; found " + version.replace(".", r"\.")
            ):
                _require_jax_version("jax", version)
