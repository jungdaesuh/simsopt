"""JAX NCSX moving-coil nested-LS optimization with a reduced-Schur inner.

This serial tutorial shares the NCSX input, nine-term outer objective, and
accepted-iterate transaction with the native CPU reference at
``examples/jax/native_reference/boozerQA_ls.py``. The native reference uses
banana BFGS/Newton; this one uses reduced-Schur Newton, so their timings are
not comparable. ``--smoke`` uses a low-order 7x7 axis-tube seed and one outer
iteration. The default loads the NCSX six-mode seed at 48x48 with three outer
iterations.

Host boundary: native NCSX input and surface construction plus the host
SciPy L-BFGS-B outer controller on the CPU; the reduced-Schur inner evaluation
is JAX-backed on the selected CPU or GPU.
"""

from __future__ import annotations

from pathlib import Path

from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax_adapters.geo.nested_ls_example import (
    nested_ls_example_result,
    run_jax_nested_ls_example,
)

EXAMPLE_ID = "native-boozerqa-ls"


def _solve(
    _output_directory: Path,
    max_steps: int,
    scale: ExecutionScale,
) -> ExampleResult:
    """Run one documented JAX nested-LS scale through the common contract."""

    run = run_jax_nested_ls_example(
        smoke=scale == "bounded",
        max_steps=max_steps,
    )
    result = nested_ls_example_result(run, example_id=EXAMPLE_ID)
    return ExampleResult(
        example_id=result.example_id,
        observables=result.observables,
        status=result.status,
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-boozerqa-ls-",
        bounded_steps=1,
        native_default_steps=3,
        solve=_solve,
        result_filename="boozerQA_ls.json",
    )


if __name__ == "__main__":
    raise SystemExit(main())
