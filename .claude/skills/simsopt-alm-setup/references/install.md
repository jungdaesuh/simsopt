# Install routes

`check_env.py` picks the route; this page has the commands for each. In them,
`<checkout>` is the simsopt source checkout (`checkout.path` in the report),
`<python>` the interpreter the optimization runs with, `<fork-url>` the
repository that publishes the `alm-library` branch
(`https://github.com/<owner>/simsopt` until the owner is known) and
`<fork-remote>` its remote name in the checkout, and `<scratch>` an empty
directory outside the checkout for a temporary clone (the copy route; delete
it afterwards).

What gets installed: the self-contained package `src/simsopt/solve/alm/`
(numpy, scipy and the standard library only) and, for the coil rows, the
module `src/simsopt/geo/signed_constraints.py` (also needs simsopt's
`Derivative`). The branch also adds the four examples in
`examples/2_Intermediate/` (`stage_two_optimization_alm.py`,
`boozerQA_alm.py`, `alm_composition_example.py`,
`alm_typed_evaluator_example.py`), the tests in `tests/solve/test_alm*.py`,
and the docs page `docs/source/simsopt.solve.alm.rst`. The package needs
Python >= 3.8, the floor of upstream simsopt. The `alm-library` branch is
based on upstream commit `9e027eac3`.

## ready

Nothing to install. Run `smoke_toy.py`.

## reinstall

The checkout has the ALM sources but the interpreter does not import them:
either the interpreter's simsopt is another copy (the report's notes say
where it is imported from), or the install predates the sources.

```sh
<python> -m pip install -e <checkout>    # editable: new Python files are then picked up without reinstalling
<python> -m pip install <checkout>       # or non-editable: repeat after every source change
```

Both rebuild the C++ extension `simsoptpp` (minutes). In a uv environment
without pip use `uv pip install --python <python> ...` instead.

## upstream

A remote pointing at `github.com/hiddenSymmetries/simsopt` already has the
ALM package on its `master` (`upstream_remotes_with_alm` in the report).

```sh
git -C <checkout> merge --no-edit <upstream-remote>/master
```

Then, for a non-editable install, reinstall (see `reinstall`).

## merge-fork

A git checkout without the package. Needs a clean tree (commit or stash
first; `blockers` says so). When `contains_alm_upstream_base` is false the
merge also brings in the upstream commits up to `9e027eac3`; use `copy` if
that is unwanted.

```sh
git -C <checkout> remote add alm-fork <fork-url>    # skip when fork_remote is already set
git -C <checkout> fetch alm-fork alm-library
git -C <checkout> switch -c alm-setup                # optional: merge on a new branch
git -C <checkout> merge --no-edit alm-fork/alm-library
```

The branch adds files and touches three existing ones
(`src/simsopt/geo/__init__.py` exports the signed constraints;
`docs/source/simsopt.geo.rst` and `docs/source/simsopt.solve.rst` list the new
modules), so conflicts are rare and limited to those. Then, for a
non-editable install, reinstall.

## copy

No git checkout (a pip or conda install), or the user prefers not to merge.
Copy the package into the directory the interpreter imports simsopt from,
`<package>` below: the parent directory of
`report["simsopt"]["info"]["file"]` in the `CHECK_ENV` report.

```sh
git clone --depth 1 --branch alm-library <fork-url> <scratch>/simsopt-alm
cp -r <scratch>/simsopt-alm/src/simsopt/solve/alm <package>/solve/alm
cp <scratch>/simsopt-alm/src/simsopt/geo/signed_constraints.py <package>/geo/signed_constraints.py
```

The templates import `simsopt.geo.signed_constraints` by module path, so
`<package>/geo/__init__.py` needs no edit. Files copied into site-packages
are lost when simsopt is reinstalled or upgraded; copy them into a source
checkout instead when there is one.

## After any route

Rerun `check_env.py` until `route` is `ready`, then run `smoke_toy.py` with
the same interpreter until it prints `"passed": true`.
