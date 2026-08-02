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
performance or A100 receipt.

| Lane | Private | Public | Shim | Total |
| --- | --- | --- | --- | --- |
| Strict CPU | 64 passed, 1 skipped, 95 deselected | 45 passed, 2 skipped, 504 deselected | 2 passed, 29 deselected | **111 passed, 3 skipped** |
| Strict RTX 5090 CUDA | 65 passed, 95 deselected | 45 passed, 2 skipped, 504 deselected | 2 skipped, 29 deselected | **110 passed, 4 skipped** |

Commands used the documented `-k "bfgs_ondevice or lbfgs_ondevice or
limited_memory or traceable"` expression. No child timed out or produced a
failure. The CPU/GPU skip counts are the declared device-specific contracts.

This supersedes the earlier rollback-base observation of an approximately
8.8-GiB combined-process stop as evidence for current HEAD. That old receipt
does not identify a failing node or per-test RSS; the partitioned current-HEAD
run closes the broad compatibility selector without changing solver code.
