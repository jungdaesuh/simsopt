# Exact single-stage Boozer: full-default numerical review

This package preserves inspectable numerical results for run `20260917T035857Z-6a1f0ea9`, produced
from source commit `1311e9247ca882b327b046b0c5b2cb43b4404360` with native CPU, JAX CPU, and JAX GPU lanes.
The JAX lanes use the explicitly approved host SciPy BFGS outer optimizer over
JAX physics. GPU acceleration does not mean the entire optimizer executes on GPU.

## Result and limits

All three endpoint objectives satisfy the declared `1e-7` bound. **12 of 57
strict comparisons fail, and all three lanes exhaust 1000 iterations.** This is
endpoint-quality evidence, not strict parity, optimizer convergence, physical
acceptance, production certification, or an end-to-end speedup claim.

`review-summary.json` is a **derived numerical review package**. It contains all
57 comparison results, lane status/counters, input values, endpoint arrays,
original array hashes, source/build identifiers, and original receipt hashes.
It is not the original canonical publication bundle. Its embedded original
authority flags report the local audit; they do not independently establish
portable authority for this derivative.

Its bytes are bound by `derived_summary_sha256` in
`examples/jax/authority_evidence.json`, and
`python -m examples.jax.native_to_jax_index --check` fails when the file is
missing, unbound, or altered by one byte. Supplying it to
`--check --authority-summary` reports it as the derived summary of this run,
never as an authority summary.

`examples/jax/parity/derived_review_summary.py` generates this file from the
retained local run directory (which is not tracked in the repository):

```sh
python -m examples.jax.parity.derived_review_summary \
  --run PATH/TO/20260917T035857Z-6a1f0ea9 \
  --output examples/jax/parity/evidence/20260917T035857Z-6a1f0ea9/review-summary.json
```

The unchanged canonical bundle passed the repository's source/build and
publication audit in its clean recorded checkout. Repeating that audit requires
the original bundle, recorded checkout, native binary and adjacent local build
receipt. The receipt is unsigned and operator-writable; it is not a hermetic or
third-party build attestation. A new build need not reproduce the same binary hash.

## Reproduce

At `1311e9247ca882b327b046b0c5b2cb43b4404360`, install the repository and its JAX GPU dependencies, build the
native extension, and record its clean source/build identity using the existing
parity provenance tools. Run with float64 and one native BLAS/OpenMP thread:

```sh
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 XLA_FLAGS=--xla_gpu_autotune_level=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false MPI4PY_RC_INITIALIZE=false \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python examples/jax/run_parity.py \
  --case native-single-stage-boozer-vacuum-optimization \
  --lanes native-cpu,jax-cpu,jax-gpu --scale native_default \
  --artifact-root .artifacts/jax-example-parity
```

A fresh run receives its own identifiers and hashes; inspect failures and budget
status even when the endpoint-quality bound passes. Audit its retained bundle
from the same clean recorded checkout, substituting its new identifier for
`NEW_RUN_ID`:

```sh
python -m examples.jax.native_to_jax_index --print-authority-record \
  --authority-summary .artifacts/jax-example-parity/NEW_RUN_ID/summary.json
```
