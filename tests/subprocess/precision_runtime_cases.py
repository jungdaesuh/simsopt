"""Process-isolated typed precision-policy cases without JAX initialization."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal, cast

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SRC_DIR))

from repo_bootstrap import _strip_editable_finders

_strip_editable_finders()

import simsopt_jax.config as simsopt_config

PrecisionCase = Literal[
    "precision-env-inherited",
    "precision-explicit-mixed",
    "precision-explicit-mode-default",
    "precision-smoke-mode-default",
    "precision-invalid-environment",
    "precision-mixed-runtime-config",
    "precision-fp64-runtime-default",
]
_CASES: tuple[PrecisionCase, ...] = (
    "precision-env-inherited",
    "precision-explicit-mixed",
    "precision-explicit-mode-default",
    "precision-smoke-mode-default",
    "precision-invalid-environment",
    "precision-mixed-runtime-config",
    "precision-fp64-runtime-default",
)


def _run_case(case: PrecisionCase) -> None:
    os.environ.pop("SIMSOPT_MIXED_PRECISION", None)

    if case == "precision-env-inherited":
        os.environ["SIMSOPT_PRECISION"] = "fp64"
        config = simsopt_config.set_backend(
            "jax_cpu_parity",
            configure_runtime=False,
        )
        assert config.precision == "fp64"
        assert simsopt_config.get_resolved_precision() == "fp64"
    elif case == "precision-explicit-mixed":
        os.environ["SIMSOPT_PRECISION"] = "fp64"
        config = simsopt_config.set_backend(
            "jax_cpu_parity",
            precision="mixed",
            configure_runtime=False,
        )
        assert config.precision == "mixed"
        assert simsopt_config.get_resolved_precision() == "mixed"
    elif case == "precision-explicit-mode-default":
        os.environ["SIMSOPT_PRECISION"] = "mixed"
        config = simsopt_config.set_backend(
            "jax_cpu_parity",
            precision="mode_default",
            configure_runtime=False,
        )
        assert config.precision == "mode_default"
        assert simsopt_config.get_resolved_precision() == "fp64"
    elif case == "precision-smoke-mode-default":
        os.environ["SIMSOPT_PRECISION"] = "mode_default"
        config = simsopt_config.set_backend(
            "jax_cpu_float32_smoke",
            configure_runtime=False,
        )
        assert config.precision == "mode_default"
        assert simsopt_config.get_resolved_precision() == "fp32_smoke"
    elif case == "precision-invalid-environment":
        os.environ["SIMSOPT_PRECISION"] = "fp32"
        try:
            simsopt_config.set_backend(
                "jax_cpu_parity",
                configure_runtime=False,
            )
        except ValueError as exc:
            assert "SIMSOPT_PRECISION='fp32'" in str(exc)
            assert "jax" not in sys.modules
            return
        raise AssertionError("invalid precision environment was accepted")

    elif case == "precision-mixed-runtime-config":
        config = simsopt_config.set_backend(
            "jax_cpu_parity",
            precision="mixed",
        )
        import jax

        assert config.precision == "mixed"
        assert jax.config.jax_default_matmul_precision == "highest"
        return
    else:
        config = simsopt_config.set_backend(
            "jax_cpu_fast",
            precision="fp64",
        )
        import jax

        assert config.precision == "fp64"
        assert jax.config.jax_default_matmul_precision == "default"
        return

    assert os.environ["SIMSOPT_PRECISION"] == config.precision
    assert "jax" not in sys.modules


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=_CASES)
    args = parser.parse_args()
    _run_case(cast(PrecisionCase, args.case))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
