# GPU-Native SQP/Primal-Dual Single-Stage Results

**Status:** `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING`
**Route:** `CFS-SQP1`, SSOT Revision 2
**SSOT:** `docs/single_stage_jax_gpu_sqp_primal_dual_implementation_plan.md`
**Formal comparative verdict:** `NOT_PRODUCED`

## Current disposition

Revision 2 replaced the impossible standalone reciprocal-condition floor with
the computed KKT forward-error certificate
`zeta_2 / (rho_K - zeta_2) < 1e-7`, while retaining `rho_K > zeta_2`,
`zeta_2 <= 1e-10`, residual, Schur, rank, finiteness, and transfer gates.
The fresh RTX derivative gate and one-step canary passed. The ten-step canary
failed because the SQP trajectory did not maintain or reduce the required
endpoint measures. Per the SSOT chain, no cold endpoint or warm samples ran.

This is a bounded-negative optimization result, not evidence that the GPU
formulation or all SQP routes are impossible. The projected 100-iteration time
is diagnostic only and is not a speed verdict without endpoint certification.

## Revision 2 identity

| Field | Value |
| --- | --- |
| Live HEAD | `320e5cba814414a43e48cb5b6e53f4ad356a9925` |
| Plan SHA-256 | `3024b82b272dd72349c8c814b7b547dc6335357c1155b95cf60c1c5d252d0b78` |
| Budget SHA-256 | `d51c87c55793ebed63acf01e87ef3837f5abdccb95e8c61827758a8961482082` |
| Prior-campaign manifest SHA-256 | `6734e622f3f402875dffc3e381fd79318ddad611fb84accc829583bb41fea1e9` |
| Route-v2 bytes | 6891 bytes, `b3b36797924331721d36221c29f94a9d464c6f72812f47fad360085be0b37287` |
| Campaign-v2 contract SHA-256 | `b4824057c78e21014a6aad411c1ee3595f722816a02b0173713e7cd621979ce3` |
| Device | RTX 5090, `GPU-7951f78e-c05d-e01c-303f-d644f4341fe1` |
| Runtime | JAX `0.10.0`, driver `595.84`, FP64 |

## Gate ledger

| Gate | Status | Evidence |
| --- | --- | --- |
| Revision 2 SSOT/policy update | `PASS` | Direct forward-error bound; old receipt remains immutable |
| Prior artifact preservation | `PASS` | 18-entry manifest, SHA `6734e622...` |
| Generic SQP/core/route/receipt validation | `PASS` | 177 focused CPU tests, including null-edge and residual-tamper gates; Ruff and formatting pass |
| RTX derivative gate | `PASS` | `rho_K=6.397585277035713e-06`, `zeta_2=8.729023804486768e-16`, bound `1.3644247677339793e-10`; rank 255; zero hot H2D/D2H |
| RTX one-step canary | `PASS` | 2.487900536 s synchronized solve; zero hot transfers |
| RTX ten-step canary | `FAIL` | `FEASIBILITY_NOT_MAINTAINED_OR_DECREASED`, `RAW_KKT_NOT_DECREASED` |
| Complete cold endpoint | `NOT_RUN` | Prohibited after ten-step failure |
| Conditional warm samples | `NOT_RUN` | No certified cold endpoint |

## Revision 2 campaign evidence

Authoritative root:
`/home/jungdaesuh/campaigns/cfs-sqp1-r2-20260810T0100Z`

| Artifact | SHA-256 |
| --- | --- |
| `campaign.json` | `a3f40370adc2a0555fa3ce3cc863b8426579257e887cd584ca54324fe26089d4` |
| derivative gate receipt | `0c9920a3ed36b4d87aa9dbcd178a5ec12600169f2db84ee04ef6317237c884d1` |
| derivative raw result | `26c70f779d592763f16e1c3c03fc92087d11d65c99b7e7a40fc3c5f1f6809ed1` |
| one-step gate receipt | `407ba4805ab5b569bc0c01f21ae8ce5604450e1af5a1eca718f0ee8300e7076a` |
| one-step raw result | `bd21063c0bdd1e7ae37d6f6e6f4ea76c139d72ae56897b78585dbe11ef2af258` |
| ten-step gate receipt | `4242f35d97417f651ec06d01d01f0b82dd6ec4f1fc18aa0a1f57169557c9e186` |
| ten-step raw result | `f9c9e18bdde99ef4beef1ad52abb45b9bcd08a35aaf88c3667b6eae198ef0edf` |

The derivative gate synchronized for `25.966159451 s`, with whole-child peak
memory `25,906,118,656` bytes (`75.769%`). The one-step gate synchronized for
`2.487900536 s`, and the ten-step gate for `14.267211583 s` with projected
100-iteration time `142.67211583 s`. The ten-step endpoint changed from
`7.802097876149912e-16` to `1.1133440971064634e-08` scaled feasibility and
from `0.005108879270420846` to `0.006220502808346709` raw KKT stationarity.

All gate receipts and the campaign receipt independently validate. No claim is
made for a converged or faster endpoint.

## Immutable Revision 1 history

The earlier root
`/home/jungdaesuh/campaigns/cfs-sqp1-20260809T2000Z` remains byte-identical
Revision 1 evidence. It failed its static `rho_K > 0.0010000001` gate despite
the direct bound being `1.288868610089899e-10`; its campaign SHA is
`897df251810269b0cbf38881e57964c99faee08cdfb03178c5a65f6dd7420f5e`.
Revision 2 does not reinterpret or promote that receipt. The old contract is
still accepted only for historical validation, while all new evidence binds
the Revision 2 plan, budget, and contract digests above.

The formal comparative verdict remains `NOT_PRODUCED`: no matched native
baseline campaign was authorized or run.
