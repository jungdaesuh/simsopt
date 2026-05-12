# Simsopt testing framework

## Overview

This directory contains integrated/regression tests. Source code for unit tests of each component is stored in the subdirectory for that component.

The layout of the subfolders within **tests** nearly mimics that of the simsopt code in **src/simsopt** folder. The test files (inputs, outputs or any other data files) are all collected into **tests/test_files**.

## Running tests

### With unittest 

To run the tests, you must first install simsopt with `pip install .` from the main simsopt directory.
Then, change to the `tests` directory, and run `python -m unittest` to run all tests.
The repository-level `./run_tests` and `./run_tests_mpi` scripts rebuild the
editable package before testing, so the native `simsoptpp` extension matches
the Python source in the checkout. Before direct `pytest` or `unittest` runs,
use:

```bash
python -m pip install --force-reinstall --no-deps -e .
```

Tests that instantiate `simsopt.mhd.Spec` require the full SPEC
runtime, not just `py_spec`. In those environments,
`import spec.spec_f90wrapped` must also succeed.

To run unittests only in one folder, for example, `geo`, run `python -m unittest -t . -s geo`.

To run unittests only in one file for example `geo/test_surface.py", run `python -m unittest geo.test_surface`.

To run only a single suite of unittests such as `QuadpointsTests` in `geo/test_surface.py` run `python -m unittest geo.test_surface.QuadpointsTests`

In this fashion, we can run only a single unit test: `python -m unittest geo.test_surface.QuadpointsTests.test_theta`.

See the [python unittest documentation](https://docs.python.org/3/library/unittest.html) for more options.


### Parallel testing with pytest

Requires installing pytest along with pytest-xdist or pytest-parallel.  Install either of them with pip. With pytest-xdist, run `pytest -n <N> geo` to run all tests in the `geo` folder in parallel. For pytest-parallel, use `pytest --workers <N> geo`. Here <N> is the number of cores you want to use. In place of a specific number for <N>, use `auto` to automatically use all the avaialable cores in the machine.

## Regression panel (`tests/regression/`)

`tests/regression/` contains a forward-pinned, snapshot-based panel that asserts the simsopt math layer (`src/simsopt`, `src/simsoptpp`) produces stable numerical outputs at fixed configurations supplied by collaborator artifacts. See `tests/regression/README.md` for the panel-specific docs and `docs/regression_panel_colleague_artifacts_2026-05-11.md` for the design rationale.

The panel is **platform-pinned**: snapshots are generated on macOS Silicon (Darwin/arm64) with Accelerate BLAS and `OMP_NUM_THREADS=1`. On any other platform the panel auto-skips via `tests/regression/conftest.py` to avoid spurious SHA mismatches.

To run locally:

```bash
OMP_NUM_THREADS=1 python -m pytest tests/regression/ -v
```

Forty panel tests + six negative-control resolution tests = 46 expected. The conftest sets `OMP_NUM_THREADS=1` if unset and warns if it has been set to anything else.
