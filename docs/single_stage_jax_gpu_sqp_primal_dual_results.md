# GPU-Native SQP/Primal-Dual Single-Stage Results

**Status:** `CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING`
**Route:** `CFS-SQP1`, SSOT Revision 3 convergence closure
**SSOT:** `docs/single_stage_jax_gpu_sqp_primal_dual_implementation_plan_r3.md`
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

## Revision 3 convergence closure

Revision 3 added device-resident minimum-norm normal restoration to the
nonlinear line search and fixed-shape per-iteration telemetry. The focused
regression suites passed (`62` dense-SQP/certificate tests, `22` runner tests,
and `71` receipt/validator tests). One bootstrap ten-step RTX gate was
then run with no untimed warm solve, cold endpoint, A100 run, or speed campaign.

The revised trajectory maintained scaled feasibility below `1e-10` and reduced
the objective, but raw KKT stationarity increased from the bootstrap value
`0.005108879270420846` to `0.030689422261180984`. The ten-step gate therefore
fails `RAW_KKT_NOT_DECREASED`. This is the second convergence failure of this
SQP route. Per the Revision 3 SSOT, no further CFS-SQP1 tuning or timing is
authorized. The prior coupled AL routes are also closed, so continuation
requires a new SSOT for the filter/trust-region fullspace route.

| Iteration | Step | Merit | Feasibility inf | KKT residual | Multiplier update inf | BFGS reset | Restoration |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 8.3998071402e-05 | 1.2439553395e-11 | 1.4760362824e-17 | 5.7914759758e-04 | 0 | 1 |
| 2 | 3.90625e-03 | 8.3236389821e-05 | 5.6224383316e-11 | 3.2853795109e-18 | 2.4070268600e-05 | 0 | 1 |
| 3 | 1.25e-01 | 8.2684773507e-05 | 2.4994689297e-11 | 2.1869172662e-17 | 1.2505341561e-04 | 0 | 1 |
| 4 | 3.125e-02 | 8.1935079840e-05 | 7.4139323610e-11 | 5.1013207720e-18 | 7.1936487656e-05 | 0 | 1 |
| 5 | 1.5625e-02 | 8.1405028176e-05 | 2.8012875132e-11 | 3.4012399641e-18 | 3.1323627032e-05 | 0 | 1 |
| 6 | 1.5625e-02 | 8.0729722604e-05 | 6.1888915359e-11 | 3.1823499314e-18 | 2.7112259979e-05 | 0 | 1 |
| 7 | 7.8125e-03 | 8.0325980229e-05 | 1.3515016858e-11 | 1.9057808937e-18 | 1.1491570256e-05 | 0 | 1 |
| 8 | 7.8125e-03 | 7.9869401990e-05 | 2.0248004408e-11 | 2.1800693090e-18 | 9.6467310030e-06 | 0 | 1 |
| 9 | 7.8125e-03 | 7.9370882624e-05 | 2.6982998212e-11 | 1.6164650465e-18 | 7.9515823989e-06 | 0 | 1 |
| 10 | 7.8125e-03 | 7.8841807477e-05 | 3.3021264675e-11 | 1.4630216500e-18 | 6.4345818281e-06 | 0 | 1 |

The synchronized ten-step solve took `14.598303656 s`; this number is retained
as diagnostic evidence only and is not a GPU-speed claim.

### Revision 3 identity

| Field | Value |
| --- | --- |
| Plan SHA-256 | `e8ba9fe0513163038fd587427cc5199a00be954d9d9c3f9f51a79641136c9f4e` |
| Parent Revision 2 plan SHA-256 | `3024b82b272dd72349c8c814b7b547dc6335357c1155b95cf60c1c5d252d0b78` |
| Budget SHA-256 | `d51c87c55793ebed63acf01e87ef3837f5abdccb95e8c61827758a8961482082` |
| Source HEAD | `aeda1a02eb9706dd9aad5b9f97b7f3a72193c6ca` |
| Source manifest SHA-256 | `88ef17d13d14bdd48e403692b52fceccaaf1c37857b204f3ca15b2c87b4cc169` |
| Tracked diff SHA-256 | `997c766e43854b6aa5d2c600e95af2f9958d0827e90d1663d1c69d05a282f51c` |
| Untracked bytes manifest SHA-256 | `a90b88949a97c19448c7caa3cf7a7779cb402142c769f99e71d33c5ec6c5e3fc` |
| Device | RTX 5090, `GPU-7951f78e-c05d-e01c-303f-d644f4341fe1` |
| Gate disposition | `FAIL / RAW_KKT_NOT_DECREASED` |

Authoritative Revision 3 root:
`/home/jungdaesuh/campaigns/cfs-sqp1-r3-dirty-snapshot-20260809`

| Artifact | SHA-256 |
| --- | --- |
| gate receipt | `90d7e416ced18978b56061ee1f7e56cc1fb7a9bf4cdbe1e0ea620342fa49557d` |
| raw result | `e6cdd063044b6df6fbd393e1aa106721a41c0a0da5167d803f00e427413d476d` |
| GPU memory | `3ba0a3abfac19ab9d68f7fcbfac69e0c038b6c49d4d37cc3ed0d5289441ed20e` |
| runtime evidence | `0c062dc6f4ceeb2650fc7ac35d43e495292579f6708e76c962e0e2c31656ec20` |

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
| RTX Revision 3 ten-step canary | `FAIL` | `RAW_KKT_NOT_DECREASED`; feasibility maintained below `1e-10`; zero hot transfers |
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
