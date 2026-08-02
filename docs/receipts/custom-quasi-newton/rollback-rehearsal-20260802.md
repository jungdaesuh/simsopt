# Custom quasi-Newton rollback rehearsal

Date: 2026-08-02

## Revisions

| Role | Revision | Git tree |
| --- | --- | --- |
| Rollback base | `9c64c2ef6cee45eb7eb1989bd5a41e2adf8bfc26` | `b74a857910ce4a69fbbd5f9994a27c1a3105835f` |
| Candidate | `3b2b9f40a` | `cb9218d93c49dde592e8f6a504bfd7777e484909` |
| Rehearsed index | revert `3b2b9f40a`, `41d95cf50`, `fd200f564` | `b74a857910ce4a69fbbd5f9994a27c1a3105835f` |

The revert order was newest to oldest. `git diff --cached --exit-code
9c64c2ef6 --` passed in the temporary rollback worktree, so the resulting
tracked tree is byte-identical to the rollback base.

## Frozen selectors

All runs used Python 3.11, FP64, strict CPU, and fresh worktrees.

| Selector | Base | Candidate |
| --- | --- | --- |
| Driver/legacy compatibility/result schema under `native_cpu` | 25 passed, 12.03 s | 26 passed, 11.08 s |
| Bounded Boozer eager on-device (`bfgs_ondevice or lbfgs_ondevice`) | 35 passed, 1 skipped, 61.33 s | 35 passed, 1 skipped, 59.95 s |

The broad traceable application selector was not promoted: the base run was
explicitly stopped at exit code `143` after reaching approximately `8.8 GiB`
RSS. It had already reported failures and reproduced the known unbounded
traceable compile/resource problem. The bounded traceable application gate
therefore remains open.

## Receipt verification

From the clean candidate worktree, a fresh process ran:

```text
validate-all --root docs/receipts/custom-quasi-newton \
  --repo-root /home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed
```

Result: `38` manifests validated against the external local archive. The
sorted tracked-manifest inventory has SHA-256
`d165979c72ec43c131d42fb646cb0b9c5cb5207a44445bb66f5cabbf39ca9539`.
The CPU and GPU environment-lock hashes are
`159e05a65796e76dfb502ea4f6a06b1f412af1c7bb147bb5ac5974b5888a6b35` and
`fc724b570ca23356b18df17da87a00217066fd42e5b02de5fe26b46cf20473f8`.

The archive remains in the ignored local `.artifacts/` tree; this receipt
proves the clean-checkout checksum path but is not durable external storage.
