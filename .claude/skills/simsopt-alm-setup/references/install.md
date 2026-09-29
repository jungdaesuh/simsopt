# Install routes

`check_env.py` picks the route; this page has the commands for each. In them,
`<python>` is the interpreter the optimization runs with, `<fork-url>` the
repository that publishes the `alm-library` branch
(`https://github.com/jungdaesuh/simsopt.git`, `check_env.py`'s default), and
`<clone>` the directory of an `alm-library` clone (`checkout.path` in the
report).

What gets installed: the pip package `simsopt-alm` (import name
`simsopt_alm`), which lives in `packages/simsopt-alm/` of the `alm-library`
branch. It is pure Python and needs Python >= 3.8, numpy and scipy. It
installs beside the simsopt the user already has (a fork, an editable
checkout or a PyPI release) and changes nothing in it. The solver
(`simsopt_alm` and its modules) imports only numpy, scipy and the standard
library; `simsopt_alm.signed_constraints` (the coil rows) also imports
simsopt's `Derivative`, so it needs a simsopt in the same interpreter. The
package directory also holds the four examples in `examples/`
(`stage_two_optimization_alm.py`, `boozerQA_alm.py`,
`alm_composition_example.py`, `alm_typed_evaluator_example.py`) and the tests
in `tests/`.

`ready` means `simsopt_alm` imports. The generic template needs only the
solver; the Stage-2 and Boozer templates also import the signed constraints,
so they also need simsopt. `templates` in the report says which templates can
run now. Without a simsopt, install the user's own first; for PyPI's, add
the `simsopt` extra: `"simsopt-alm[simsopt] @ git+<fork-url>@alm-library#subdirectory=packages/simsopt-alm"`.

In a uv environment without pip, use `uv pip install --python <python> ...`
instead of `<python> -m pip install ...` in the commands below.

## install

The default: install the package from the fork, with no local clone.

```sh
<python> -m pip install "git+<fork-url>@alm-library#subdirectory=packages/simsopt-alm"
```

This builds no C++ and takes seconds. To update to a newer `alm-library`
later, rerun it with `--force-reinstall --no-deps`.

## editable

`--checkout <clone>` was given: the user wants the solver's sources in a
clone they can read and change. `<clone>` is either a place to clone into
(a new or empty directory: `checkout.clone_target` in the report) or an
existing clone (`checkout.is_alm_clone`): the top level of a git repository,
on the `alm-library` branch, with `packages/simsopt-alm/pyproject.toml` and
`packages/simsopt-alm/src/simsopt_alm/__init__.py`. Any other directory is
a blocker (`checkout.problems` says why) and never this route: clone into a
new directory and rerun `check_env.py` with it. Skip the `git clone` line
when `<clone>` already is a clone.

```sh
git clone -b alm-library <fork-url> <clone>
<python> -m pip install -e <clone>/packages/simsopt-alm
```

Changes to the clone's Python files then take effect without reinstalling;
`git -C <clone> pull` updates it. When the report's notes say the interpreter
imports `simsopt_alm` from somewhere else, this install replaces that one.

## ready

Nothing to install. Run `smoke_toy.py`.

## After any route

Rerun `check_env.py` until `route` is `ready`, then run `smoke_toy.py` with
the same interpreter until it prints `"passed": true`.
