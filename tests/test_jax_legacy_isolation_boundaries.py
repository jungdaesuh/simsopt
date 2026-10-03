"""The native ``simsopt`` packages stay independent of the JAX port.

Importing the native packages, and running native code through them, must
not load ``simsopt_jax`` or ``simsopt_jax_adapters``. Each check runs in a
fresh interpreter so modules imported by other tests cannot hide a leak.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
JAX_SOURCE_ROOTS = (SRC_ROOT / "simsopt_jax", SRC_ROOT / "simsopt_jax_adapters")

_ASSERT_NO_JAX_PORT_MODULES = """
loaded = sorted(
    name
    for name in sys.modules
    if name.split(".", maxsplit=1)[0] in ("simsopt_jax", "simsopt_jax_adapters")
)
if loaded:
    raise SystemExit(f"native simsopt loaded the JAX port: {loaded}")
"""


def _run_native_probe(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if existing_pythonpath is None
        else os.pathsep.join((str(SRC_ROOT), existing_pythonpath))
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(source) + _ASSERT_NO_JAX_PORT_MODULES,
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_native_packages_import_without_the_jax_port():
    result = _run_native_probe(
        """
        import sys

        import simsopt
        import simsopt._core
        import simsopt.configs
        import simsopt.field
        import simsopt.geo
        import simsopt.mhd
        import simsopt.objectives
        import simsopt.solve
        import simsopt.util
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_native_field_and_objective_evaluation_runs_without_the_jax_port():
    result = _run_native_probe(
        """
        import sys

        import numpy as np
        from simsopt.configs import get_data
        from simsopt.geo import CurveLength, SurfaceRZFourier
        from simsopt.objectives import SquaredFlux

        base_curves, _currents, _axis, nfp, field = get_data("ncsx")
        surface = SurfaceRZFourier.from_nphi_ntheta(
            nfp=nfp, mpol=1, ntor=0, nphi=8, ntheta=8, range="half period"
        )
        surface.set_rc(0, 0, 1.6)
        surface.set_rc(1, 0, 0.2)
        surface.set_zs(1, 0, 0.2)
        field.set_points(surface.gamma().reshape((-1, 3)))
        objective = SquaredFlux(surface, field)
        values = (objective.J(), CurveLength(base_curves[0]).J())
        gradient = objective.dJ()
        if not (np.all(np.isfinite(values)) and np.all(np.isfinite(gradient))):
            raise SystemExit(f"non-finite native evaluation: {values}")
        """
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_jax_source_packages_have_no_dynamic_imports():
    dynamic_imports = []
    for root in JAX_SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            dynamic_imports.extend(
                f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and (
                    isinstance(node.func, ast.Name)
                    and node.func.id == "__import__"
                    or isinstance(node.func, ast.Attribute)
                    and node.func.attr == "import_module"
                )
            )

    assert dynamic_imports == []
