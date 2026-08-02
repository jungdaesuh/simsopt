# Boozer initial-state parity

Date: 2026-08-02

Candidate: `bc97cb7540fc0232b2e088d1b08fd703f58e609a` in clean detached
worktree `/tmp/qn-broad-current`; FP64, 65 parameters, VMEC-free NCSX fixture.

The native and JAX outer objectives were evaluated at the identical initial
coil vector. This receipt covers initial observables only; it does not certify
outer BFGS convergence or performance.

| Lane | Native objective | JAX objective | Absolute difference | Native/JAX gradient infinity norm | Max gradient difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strict CPU | 0.0003902843220850033 | 0.0003902843220850035 | `2.168404344971009e-19` | 0.0038700625919864456 / 0.0038700625919863463 | `7.428953285870676e-16` |
| Strict RTX 5090 CUDA | 0.0003902843220850033 | 0.0003902843220850029 | `3.7947076036992655e-19` | 0.0038700625919864456 / 0.0038700625919860713 | `2.084703937255128e-15` |

Raw stdout is preserved under `raw/`:

| Lane | Exit | SHA-256 |
| --- | ---: | --- |
| CPU | 0 | `21c3c49c010a8b99ed86613d6e56f2eb81b12b578539b83ad7845e0167841da6` |
| GPU | 0 | `3a216d8e8b7f6d0808c0b6b787bf0c320e3746461f36903745cffb0ee057926f` |

Both lanes ran the source-owned `benchmarks.fixtures.custom_quasi_newton.fixture("boozer")`
builder with strict backend selection and `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
