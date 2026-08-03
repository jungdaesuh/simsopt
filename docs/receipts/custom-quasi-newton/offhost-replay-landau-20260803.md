# Off-host archive replay — landau (2026-08-03)

## Second replay (same day, current corpus)

After the review iterations extended the corpus, the replay was
repeated: fresh archive tarball
(sha256 prefix `59fc61fe901bd1256a8c`, verified end to end), extracted
and re-write-protected (`chmod -R a-w`; the prior 47-receipt copy kept
at `qn-receipt-archives-47`), checkout fast-forwarded via delta bundle
and detached PRISTINE at `89a0a823c` — result:
`{"validated": 49, ...}` covering every current receipt, including the
republished performance receipts with recorded memory diagnostics, the
A100 quiet and contended receipts, and the rollback-rehearsal evidence
tree. The first replay below is retained as the historical record of
the 47-receipt corpus.

Different-host replay of the complete custom-quasi-newton receipt authority
bundle, per the closure plan's Phase-7 off-host gate.

## Result

`{"validated": 47, "root": "docs/receipts/custom-quasi-newton"}` — all 47
receipts green under the hardened validator (commit authentication via
`git cat-file -e <sha>^{commit}`, child `gpu_memory`/`trial_trace` SHA-256
recomputation, derivation and qualification recomputation from raw rows,
archive-root inventory cross-check), on a host that shares nothing with the
publishing machine.

Scope, precisely: `487d9ff89` below is the VALIDATOR checkout (the
then-HEAD whose object store authenticates every receipt-recorded
candidate commit such as `359fd41fc`). The replay re-verifies bytes,
hashes, derivations, and qualification arithmetic; it does not re-execute
solvers at the receipts' candidate trees.

## Replay identity

- Host: `landau` (Columbia AP fusion server, NVIDIA A100-PCIE-40GB,
  kernel 5.4.0-233-generic, NFS-backed home) — publishing host was
  `jungdaesuh-playstation` (RTX 5090).
- Checkout: fresh clone of a shallow single-branch replica
  (`--depth 200`, branch `pr/jax-port-squashed`), detached at
  `487d9ff89b6947e49c1e90fe4281a52399f66ee8`, `git status --porcelain`
  empty. Depth covers the oldest receipt-referenced commit
  (`9c64c2ef6`, 74 commits behind HEAD); presence proven with
  `git cat-file -e` before transfer.
- Interpreter: uv-managed CPython 3.11.15 with the environment-lock-pinned
  stack `jax==0.10.0 jaxlib==0.10.0 numpy==2.4.6 scipy==1.17.1`
  (`benchmarks/environments/custom_quasi_newton_cpu.lock.txt`).
- Archive copy: `~/qn-offhost-replay/qn-receipt-archives`, write-protected
  with `chmod -R a-w` after extraction (no-root equivalent of a read-only
  mount).
- Transfer integrity: both tarballs SHA-256-verified end to end —
  repo replica
  `6bf8833ea5e01c9bf87ec345c3b1966df82c99edc8866e649dabc38c555f53d4`,
  archive
  `2d24521363c7347e9e726671e4aa115f9ebfb5fd6532df2704514174bb268f31`
  (local manifest: session scratchpad `replay-transfer.sha256`).

## Exact replay command

```bash
cd ~/qn-offhost-replay/checkout
JAX_PLATFORMS=cpu PYTHONPATH=src:. ~/qn-offhost-replay/venv/bin/python \
  -m benchmarks.custom_quasi_newton_receipts validate-all \
  --root docs/receipts/custom-quasi-newton \
  --repo-root . \
  --archive-root ~/qn-offhost-replay/qn-receipt-archives
```
