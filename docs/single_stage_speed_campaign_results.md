# Single-stage speed campaign results

**Campaign:** `single-stage-speed-20260804`
**Closed:** 2026-08-05
**Status:** `CLOSED_BOUNDED_NEGATIVE`
**Promotion status:** `NON_PROMOTING`
**Protocol verdict:** `NOT_PRODUCED`

## Bottom line

The native-default, nested single-stage Boozer formulation did not produce a
GPU speed win. The preserved RTX 5090 cold trajectories put the custom GPU
lane about 26.25 times behind the matched native CPU lane at the 1,000-iterate
cap. The Optax lane then timed out before producing an endpoint. A bounded A100
profile independently found a 9.096 s median custom accepted-iteration
increment and a 9.876 s median across three warm Optax value/gradient calls.
The bounded retry demonstrated no Optax-only improvement.

This is a bounded negative result, not a frozen-validator `LOSS`. The four-lane
r5 campaign never produced `campaign.json`; therefore no complete r5 receipt,
warm-median verdict, time-to-quality verdict, or promotion claim exists.

## Protocol binding

- Frozen protocol: `docs/single_stage_speed_campaign_protocol.md`
- Frozen-files tag: `campaign-20260804-frozen-r5`
- Protocol SHA-256:
  `32633251318fe9007740f379aef8c825b41b2a43276c7bfd04d99ef2c407bfc4`
- Frozen validator SHA-256:
  `d4fcb9cfe9809576ba5e402f436185516560430d174ff8e284ab9d1788f9a59d`
- The protocol remains byte-identical to r5 and retains its historical
  `Status: Active` text. This results record closes the attempted campaign
  without editing or retroactively reinterpreting that frozen file.

## Evidence inventory

### RTX 5090 partial campaign

Root:
`/home/jungdaesuh/simsopt-campaigns/.single-stage-speed-20260804.partial-20260805T052535Z-2add24ec`

The root is deliberately named `partial`. It contains two completed cold lane
observations and a censored Optax trajectory, but no aggregate `campaign.json`.
The native and custom receipts report repository commit
`22c6c8b049a6b3ce76abd38ca5e1a968a3001057` with a dirty source tree. Their
receipt-bound executed-source hashes, input identity, configuration identity,
construction identity, runtime policy, and device metadata are retained in the
lane observations; the commit field alone is not a sufficient source identity.

| Evidence | Rows/size | SHA-256 |
|---|---:|---|
| canonical `inputs/input_bundle.json` | 1,157 bytes | `9583586c7f2d3798b88eae1475a283b213d1579bd0378d47c89d73d99314b1b7` |
| native `lane_result.json` | 421,481 bytes | `8118529751f184f60f0c4d26f338cd1832aae579004d62866fb2a2f6617e9fe4` |
| native `trajectory.jsonl` | 1,000 rows | `fa81b533b7bd8127b021bc2aa206c01914f91a3ef2e34eee6e0636e2031fed8f` |
| custom GPU `lane_result.json` | 425,805 bytes | `ecb76d641d5fdac4cfcfd3c1d7c5722e48c94b3cd7c4f5845c454773c9e2b7b2` |
| custom GPU `trajectory.jsonl` | 1,000 rows | `e67baceda0e54c70f7317a38b055d7c9c577cff187a4eff2487bff55cb90b904` |
| Optax GPU `trajectory.jsonl` | 588 rows | `fb27c4647a0b7badbf671e0f60eb04512784481e71ef464e6cdaa6cbf79d52a5` |

The native and custom terminal receipts bind input fingerprint
`39f4edf6d1cad82cdcc567f3d7bd24c077af6ea0b314603b6622e25d7ce20f88`,
configuration fingerprint
`f8bc0b385f30fd34c8b5ab55dda4cf29dc0e0bcd6fe05d49b0fdbe54d9226667`
and construction fingerprint
`713c8530ec0569a659f902f832620846ed993744de99afe086d3cfe2a25c2b5b`.
The censored Optax directory is co-located under the same collector workspace,
and the monitor records that the collector launched it from that run, but no
Optax terminal receipt independently binds those identities.
The custom receipt identifies `NVIDIA GeForce RTX 5090`, FP64, strict
`jax_gpu_fast`, and direct exact-adjoint selection.

The controller sidecars establish the terminal cause and validator state:

| Evidence | Size | SHA-256 |
|---|---:|---|
| `single-stage-speed-20260804-controller-r1/run-status.txt` | 78 bytes | `a1a90fd93802168e18357183242dac912e04d94810bf8a8215be2352d0a4e327` |
| `single-stage-speed-20260804-controller-r1/MONITOR_STATE.md` | 5,700 bytes | `5109089783790dfa8b19cdefbfded2ea07da36a95297445638b37498958b9944` |

They record `campaign_exit=1`, `validator_exit=not_run`, and the monitored
SIGTERM at the 10,800-second child cap. During closeout, the exact campaign
affinity-guard unit was stopped through systemd and the stale receipt monitor
was terminated; both recorded PIDs are gone.

### One-time gradient agreement

Root:
`/home/jungdaesuh/simsopt-campaigns/single-stage-speed-20260804-gradient-agreement-r3`

The canonical `agreement.json` is 1,474 bytes with SHA-256
`3acc38093975bbd7f5ff4ab85a95bfe4aad47032c2cfa2d206009973002bb100`.
At the matched native-default initial point, the direct FP64 LU and parity-mode
exact gradients passed the frozen tolerance contract: maximum absolute
difference `9.865155567445605e-16`, maximum tolerance ratio
`0.0004844158171769726`, and 461 finite components. This establishes agreement
of those two adjoint routes at that point; it does not certify a full speed
campaign. The numerical comparison is independently reproducible from
`direct.json` and `parity.json`, but these artifacts do not bind a repository
commit, dirty diff, Python/runtime environment, or executed-source hashes.

### Landau A100 bounded T1 profile

Remote root: `/mnt/homes_global/jdsuh/simsopt-speed-r3-20260805`
Source: commit `092394b8b480286f92f8ce96720b544e66833451`, tag
`ship-20260805-landau`
Findings: `.Codex/t1-profile-findings.md`, 13,331 bytes, SHA-256
`bc2b16e38d049109356c3afee7cfa34bc9300d7f5978b53e01b41d5643b77d60`

The findings were independently re-hashed and reviewed. They explicitly mark
the run as bounded profiling rather than campaign evidence. Receipt-bound
provenance is incomplete: the GPU model, UUID, Python executable/version,
jaxlib, CUDA runtime/driver, launch command, and package inventory are
operator-reported rather than bound by those receipts.

## Bounded results

| Evidence | Native CPU | Custom GPU | Optax GPU |
|---|---:|---:|---:|
| RTX 5090-box cold accepted trajectory | 287.304 s / 1,000 | 7,541.455 s / 1,000 | 10,707.415 s / 588, then timeout |
| RTX 5090-box final objective | `4.4822247e-08` | `1.6135607e-07` | no endpoint |
| A100 accepted-iteration increment | not measured | 9.095761 s median, iterations 3-20 | not measured |
| A100 value/gradient call | not measured | not isolated | 9.875811 s median, last 3 warm calls |

The RTX figures are accepted-iteration trajectory timestamps, not promoted
campaign sample times. They show a diagnostic custom/native ratio of
`7541.455456477008 / 287.30421751597896 = 26.2490`. The missing Optax endpoint
prevents endpoint parity and aggregate receipt construction.

The A100 profile found that the first Optax value/gradient call took
154.297644 s and that its three warm calls remained near 9-10 s. Because the
RTX and A100 runs are not matched in source, runtime, hardware provenance, or
timing boundary, no cross-host performance ratio is claimed. The bounded A100
retry demonstrates no warm Optax value/gradient improvement.

## Profile interpretation

The A100 trace audit proves no causal eliminable wall-time fraction: the strict
proven share is 0%. A lossless cached same-point trace exposes only local upper
targets of 37.6203% for device-idle command-buffer preparation/dispatch and
41.1252% when allocator/free is included. Those percentages do not transfer to
the 9-10 s off-baseline evaluations. The broad optimizer trace is censored and
supports lower-bounded event counts and cumulative durations, not a
non-overlapping eliminable wall share.

Three warm off-baseline trials associated one additional observed Newton
iteration with 0.897671 s in a three-point linear model. That association does
not isolate Newton execution or prove an irreducible fraction. No defensible
residual "other" fraction was computed.

## Closure decision

1. Close this nested, host-driven single-stage GPU campaign as a
   profile-characterized bounded negative supported by preserved partial lane
   receipts and trajectories.
2. Do not run the full T2 Optax head-to-head. The 5090 timeout and the bounded
   A100 T1 retry show no demonstrated Optax-only speedup in the evidence that
   was collected. T2 was not run and would not repair the absent r5 campaign
   receipt.
3. Do not claim protocol `WIN`, `TIE`, or `LOSS`; `protocol_verdict` remains
   `NOT_PRODUCED`.
4. Do not infer that every GPU formulation is negative. Coupled,
   proximal-nested, batched, or otherwise divergent formulations are outside
   this result and require their own frozen contract and receipts.
5. Keep the direct FP64 LU route for the three matched custom measurement
   lanes. The operator-GMRES stagnation and exact-tail work remain separate,
   nonblocking root-cause tasks.

## Open follow-ups and risks

- Investigate native-default operator-GMRES stagnation without changing
  tolerances, damping, or fallback policy:
  `GPD/todos/pending/2026-08-05-investigate-gmres-stagnation-root-cause.md`.
- Implement exact, unpadded tail handling for
  `_point_direction_chunk_reduce`:
  `GPD/todos/pending/2026-08-05-implement-exact-tail-for-point-direction-chunk-reduction.md`.
- The partial 5090 run is not a promotion artifact and lacks warm samples,
  parity rows for all four endpoints, and an aggregate manifest.
- The bounded A100 profile has the provenance omissions listed above and must
  not be upgraded into campaign evidence by prose.
- Any future GPU execution-level timing check must synchronize the optimizer
  endpoint before closing the optimizer measurement window. CPU regressions
  cover the boundary, but no new GPU run was authorized during closeout.
