# Broad Boozer compatibility partitions

Date: 2026-08-02

Candidate: `bc97cb7540fc0232b2e088d1b08fd703f58e609a` (clean detached
worktree `/tmp/qn-broad-current`)

Environment: Python 3.11, JAX/JAXLIB 0.10.0, SciPy 1.17.1, Optax 0.2.8,
FP64, `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`XLA_PYTHON_CLIENT_ALLOCATOR=platform`. Each selector ran in a fresh child
process with a 300-second CPU or 360-second GPU timeout.

The previous combined selector was sensitive to cumulative JAX compilation
cache pressure. The same collected nodes were therefore run as three explicit
partitions: private Boozer contracts, public Boozer contracts, and the two
legacy-shim contracts. This is the compatibility test evidence; it is not a
performance or A100 receipt. The rollback-base broad selector remains
incomplete; this receipt qualifies only the current candidate.

| Lane | Private | Public | Shim | Total |
| --- | --- | --- | --- | --- |
| Strict CPU | 64 passed, 1 skipped, 95 deselected | 45 passed, 2 skipped, 504 deselected | 2 passed, 29 deselected | **111 passed, 3 skipped** |
| Strict RTX 5090 CUDA | 65 passed, 95 deselected | 45 passed, 2 skipped, 504 deselected | 2 skipped, 29 deselected | **110 passed, 4 skipped** |

Commands used the documented `-k "bfgs_ondevice or lbfgs_ondevice or
limited_memory or traceable"` expression. No child timed out or produced a
failure. The CPU/GPU skip counts are the declared device-specific contracts.
The exact commands, exit codes, and raw stdout are preserved in `raw/` below.

| Lane/partition | Exit | Raw stdout SHA-256 |
| --- | ---: | --- |
| CPU/private | 0 | `8b0c62702d3652544e887ace4f959eaef776a04712f3f02f57f1f70cbf8c0455` |
| CPU/public | 0 | `7dba9cf2a18a917b658d8fff2a9721dcaf010b81df8c93d7f81ff42becac3335` |
| CPU/shim | 0 | `7f1a5b52daf0846210a70ac2c1401fd235b2ebf927c9d3ed3949027de8ead374` |
| GPU/private | 0 | `361f2ae933c7a3946ff6e77d230a4d2baa7ce74fb898f0236acf2c6903d01ad7` |
| GPU/public | 0 | `ad869557a2e9802e24d4c08f4c066d2a3a81123946a732f4af7a8aa817b9dfe9` |
| GPU/shim | 0 | `4cbb035afd81a3c1ee753f5f3226da12684e3e6b5ea682319fb95ddb178bc1c3` |

The common command prefix was:

```text
MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS={cpu|cuda} JAX_ENABLE_X64=true
SIMSOPT_BACKEND_MODE=jax_{cpu|gpu}_parity SIMSOPT_BACKEND_STRICT=1
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:.
```

Each partition appended `pytest -q`, its listed test path(s), and
`-k "bfgs_ondevice or lbfgs_ondevice or limited_memory or traceable"`.

Exact partition-to-path mapping:

| Partition | Test path argument(s) |
| --- | --- |
| CPU/private, GPU/private | `tests/geo/test_boozersurface_jax_private.py` |
| CPU/public, GPU/public | `tests/geo/test_boozersurface_jax.py` |
| CPU/shim, GPU/shim | `tests/jax/solve/test_driver_dispatch.py` `tests/jax/solve/test_compat_shim_translation.py` `tests/jax/examples/test_single_stage_vmec_hybrid_example.py` |

This supersedes the earlier rollback-base observation only as evidence for
current HEAD. That old receipt does not identify a failing node or per-test
RSS; the partitioned current-HEAD run closes the candidate compatibility
selector without changing solver code.
