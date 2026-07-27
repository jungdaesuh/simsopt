# JAX Manifest v3 Migration Candidate

Status: review-only, not activated.

The no-write generator read the frozen example-schema-v2, parity-schema-v1,
and 51-row inventory inputs. It did not modify either active manifest.

## Exact byte identities

| Contract | Active input SHA-256 | Review candidate SHA-256 |
|---|---|---|
| Example manifest | `2aeae6a63f631b205955c288e3308ad42c0191bbfcdef78b6cba7b2797db0b05` | `a2b60dfe4cd97b4f8092f315d26db952181ba5ee1121011edaec9bd30349ea0d` |
| Parity manifest | `060e55339194c203263da9d5690c2ff31bd6681f5713dc2ead0ce3313e313137` | `b51c69d7d4e08d2d08ba121f930133198f340684911565d1d3871fdaa15d78fa` |

The exact candidate bytes are checked in as
`docs/jax_examples_manifest_v3_candidate.json` and
`docs/jax_parity_manifest_v2_candidate.json`.

## Semantic diff

- All 51 native source rows remain present: 25 are `eligible`, one is
  `hybrid`, 23 are `blocked`, and two are `not_applicable`.
- The 11 existing JAX programs remain executable but become explicitly
  non-covering tutorials or compatibility lessons.
- Five single-source compatibility lessons bind their exact successor ID,
  runtime warning text, and one-interval removal contract. Combined tutorials
  remain non-covering without pretending to be aliases.
- Twenty-six planned one-to-one records are added. Every record uses the same
  tier and filename as its native source.
- The VMEC single-stage record is a hybrid. Its CPU scope is
  `host_and_jax_slice`; its GPU scope is `jax_slice_only`.
- The 28 old tutorial-lineage parity relationships are replaced by 26 exact
  native-source-to-mirror relationships.
- Every new relationship remains `unsupported` pending its own implemented
  mirror and replayable parity evidence. The migration promotes zero scientific
  parity claims.
- Example schema v2 plus parity schema v1 remains a read-only legacy pair for
  one documented deprecation interval. Mixed v2/v2 and v3/v1 pairs are
  rejected.

## Reproduction

```bash
python examples/jax/build_manifest_v3_candidate.py \
  --examples examples/jax/manifest.json \
  --parity examples/jax/parity_manifest.json \
  --inventory examples/jax/one_to_one_inventory.json \
  --dry-run
```

Activation is intentionally excluded from this candidate. The plan requires
explicit approval of both candidate SHA-256 values. After approval, activation
must use these byte-identical files and prove an atomic rollback of both
contracts, their activation readers and tests, artifact observability, and
compatibility behavior.
