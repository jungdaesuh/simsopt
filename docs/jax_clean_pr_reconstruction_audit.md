# JAX Clean PR Reconstruction Audit

This branch was reconstructed from `upstream_hss/master` with the mixed
`pr/jax-port-pure` branch used only as a committed donor. The PR diff is meant
to contain the isolated JAX packages, adapter package, JAX tests/docs/examples,
packaging support, and only legacy-path changes required for that port.

## Legacy Footprint

The retained `src/simsopt` diff is intentionally limited to the legacy files
required by the isolated JAX port:

- `src/simsopt/__init__.py`: keeps source-tree imports usable when the generated
  setuptools_scm `_version.py` file is absent. This is needed by import-smoke
  validation for an unbuilt local checkout.
- `src/simsopt/geo/__init__.py`: removes import-time `jax.config.update(...)`
  from the legacy package. This keeps legacy `simsopt.geo` from configuring or
  discovering JAX implicitly.
- `src/simsopt/_core/tracing_metadata.py`,
  `src/simsopt/field/tracing.py`, and `src/simsopt/geo/surface.py`: provide a
  neutral legacy-owned metadata registry for translating legacy tracing
  stopping criteria in the opt-in JAX adapter path. These files do not import
  `simsopt_jax` or `simsopt_jax_adapters`; the dependency direction remains
  adapter to legacy.

No `src/simsoptpp` files are changed on this branch.

## Displaced Work

The following classes of work are intentionally outside this clean JAX PR and
belong in separate branches/PRs if still needed:

- native C++/CPU correctness fixes under `src/simsoptpp` or legacy
  `src/simsopt`;
- native build-compatibility fixes not required by the isolated JAX packages;
- legacy plain-JAX public API shims or convenience facades under `src/simsopt`;
- maintenance-only lint tooling not required by the shipped JAX package split.

## Import-Style Audit

There are no dynamic imports in the reconstructed changed Python files as
detected by the plan's AST guard (`importlib.import_module` / `__import__`), and
the tracing-metadata cleanup adds no new function-local imports in changed
production files. The test-local metadata imports are part of import-boundary
probes that assert module ownership after the test subprocess/bootstrap setup.

Existing function-local imports elsewhere in the source tree are outside this
root fix. Notable retained cases include optional plotting/export dependencies
in legacy modules and the five `import jax` sites in
`src/simsopt_jax/backend/runtime.py`; the latter are load-bearing because CUDA
memory/runtime environment variables must be resolved before JAX is imported.
Moving those imports to module scope would make GPU runtime configuration fail
before `apply_jax_runtime_config()` can set the environment.
