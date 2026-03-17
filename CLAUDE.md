# CLAUDE.md — simsopt-jax

## Validation

After every code change, run lint, format, and typecheck:

```bash
ruff check <changed-files>
ruff format <changed-files>
mypy <changed-files> --ignore-missing-imports
```

Pre-existing mypy errors from upstream (pybind11 stubs, wildcard imports) are expected. Only zero-regression on files you touched.
