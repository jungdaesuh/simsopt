"""Native CPU NCSX Boozer-LS moving-coil reference for the JAX tutorial.

This reference uses the native SIMSOPT banana BFGS-then-Newton inner solver
and a host L-BFGS-B outer controller. Its nine-term native objective includes
SIMSOPT's ``ArclengthVariation``, which has a CPU JAX package dependency; it
is therefore kept with the JAX tutorial rather than the native-only examples.
``--smoke`` uses a two-mode 7x7 axis-tube seed and one outer iteration. The
default loads the six-mode NCSX seed at 48x48 with three outer iterations.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from simsopt._examples_runtime import ExampleResult, ExecutionScale, run_example
from simsopt_jax.backend.runtime import get_resolved_precision, get_runtime_jax_device
from simsopt_jax_adapters.geo.nested_ls_example import (
    nested_ls_example_result,
    run_native_nested_ls_example,
)

EXAMPLE_ID = "native-boozerqa-ls"


def _native_runtime_metadata(_scale: ExecutionScale) -> Mapping[str, str]:
    """Return the fixed runtime facts of the native CPU reference path."""

    device = get_runtime_jax_device()
    if device is None or device.platform != "cpu" or get_resolved_precision() != "fp64":
        raise RuntimeError(
            "native Boozer-LS reference requires the JAX CPU fp64 runtime profile."
        )
    return {
        "backend_mode": "native_cpu",
        "platform": "cpu",
        "precision": "fp64",
    }


def _solve(
    _output_directory: Path,
    max_steps: int,
    scale: ExecutionScale,
) -> ExampleResult:
    """Run one documented native nested-LS scale through the common contract."""

    run = run_native_nested_ls_example(
        smoke=scale == "bounded",
        max_steps=max_steps,
    )
    return nested_ls_example_result(run, example_id=EXAMPLE_ID)


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-native-boozerqa-ls-",
        bounded_steps=1,
        native_default_steps=3,
        solve=_solve,
        runtime_metadata=_native_runtime_metadata,
        result_filename="native-boozerQA_ls.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
