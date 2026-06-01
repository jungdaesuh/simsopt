# Finite-Current Proxy Sign Conventions

This note is the operator-facing convention for materialized finite-current
seeds. The canonical software source is
`examples/single_stage_optimization/banana_opt/finite_current_profiles.py`.

## Mode contracts

| Mode | `--proxy-current-A` contract | Operator rule |
| --- | --- | --- |
| `wataru_proxy_field` | Nonnegative proxy/VF current magnitude. | Do not infer the sign from TF or banana current signs. Negative values are invalid. |
| `jhalpern30_proxy_field` | Signed upstream jhalpern30 proxy-loop scalar. | Do not infer the sign from TF or banana current signs. Do not flip the local proxy winding. |

The jhalpern30 proxy loop intentionally preserves the upstream
`banana_drivers` replay geometry: `xc(1)=R`, `ys(1)=R`. A local `ys(1)=-R`
change would break replay parity with the reference workflow even if it looked
like a convenient device-wide sign unification.

## Metadata

Materialized `results.json` files record the selected convention using:

- `PROXY_CURRENT_SIGN_CONVENTION`
- `PROXY_CURRENT_SIGN_FRAME`
- `PROXY_CURRENT_SIGNEDNESS`
- `PROXY_CURRENT_SIGN_REPLAY_REFERENCE`
- `PROXY_CURRENT_OPERATOR_WARNING`

The materializer CLI and current-sweep wrapper print the same descriptor in
their JSON summaries and help text.

## Future sign unification

A global rule like "negative current is always the operating/co direction" is a
separate upstream migration. It must update the upstream jhalpern/banana replay
contract and this materializer together, then rerun parity checks against the
reference fields. Do not implement it as a local winding change.

## Science gate

Materialized finite-current seeds are prescribed-field inputs, not certified
equilibria. Keep the no-collapse gate, iota stability check, and VMEC `curtor`
cross-check before trusting topology or promotion results.
