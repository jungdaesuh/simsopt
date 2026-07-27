# JAX examples manifest-v2 migration dry-run report

Status: **approved and activated on 2026-07-27**.

The canonical [`examples/jax/manifest.json`](../examples/jax/manifest.json) is
the exact approved v2 candidate. Its SHA-256 is
`2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05`.

## Candidate identity and semantic result

- Candidate v2 SHA-256:
  `2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05`
- Canonical v1 input SHA-256:
  `e0b8cd713efd474ca2c6d525b8fca6ba231a0f2e862cddf393372e7a60fe13e2`
- Observed input schema: `1`
- Legacy adapter used: `true`
- Compatibility interval after activation: one release
- Rollback command: `git checkout -- examples/jax/manifest.json`

Normalized semantic diff:

```json
{"device_capabilities_equal":true,"example_ids_equal":true,"example_order_equal":true,"lineage_equal":true,"paths_equal":true,"readiness_equal":true,"semantic_equal":true,"source_catalog_equal":true}
```

The candidate contains `schema_version: 2` and per-example `devices`; it
contains no per-example `lanes` or `intents`. The v1 and v2 readers normalize
to identical typed records for source catalog, example IDs and order,
readiness, paths, lineage, and CPU/GPU capability.

## Read-only proof

Command:

```console
python examples/jax/migrate_manifest.py \
  --input examples/jax/manifest.json --dry-run
```

The canonical input hash was
`e0b8cd713efd474ca2c6d525b8fca6ba231a0f2e862cddf393372e7a60fe13e2`
both before and after the command. The complete working-diff hash was
`492249f14f897e4dcbad21763ab2f84c5d0bc4100f268cd5a2730982e8423c4f`
both before and after it. The command prints the exact candidate bytes after
the `candidate_v2:` line; it has no write mode.

## Compatibility and observability

- The absent `schema_version` is accepted only as legacy v1 and requires
  `lanes`.
- Explicit v2 requires `schema_version: 2` and `devices`.
- Explicit version 1, unknown versions, mixed `lanes`/`devices`, and
  per-example `intents` fail closed.
- Both the ordinary example runner and parity aggregate emit
  `manifest_schema_version` and `used_legacy_manifest_adapter`.
- Loading v1 emits an actionable migration warning.
- The read-only v1 adapter remains for one release after v2 activation.

The user approved option `1A` on 2026-07-27, covering the candidate digest,
semantic diff, one-release compatibility interval, observability fields, and
rollback command above. The retained v1 fixtures continue to exercise the
read-only compatibility adapter; the canonical v2 document does not use it.

## Activation RED -> GREEN

After the exact candidate bytes replaced canonical v1, the pre-activation
manifest tests failed in 10 places because their fixtures still assumed that
the canonical document was v1 (`10 failed, 17 passed`). The fixtures were then
made directionally correct: canonical v2 is the authority, and legacy v1 is
derived only for compatibility and migration tests. The focused suite passed
`28 passed`, including an exact digest assertion and byte-identical replay of
the approved v1-to-v2 conversion.
