# Single-Stage JAX GPU Compute-Graph Optimization Results

**Status:** Faithful nested route stopped — engineering bounded-negative; formal shared closure incomplete
**Updated:** 2026-08-09

## Current outcome

The dense `jax.linearize` C1 route is a real same-process A100 changed-state
value-and-gradient improvement, but it does not pass the plan's formal isolated-
process promotion gate. With the same frozen candidate and FP64 objective, C1
width 8 reduced same-process warm median time from 6.7813 s to 4.1819 s, a
1.6216x speedup. In ten fresh processes per route after one excluded cache-prime
process per route, C1 reduced p50 from 29.8757 s to 26.4637 s, only 1.1289x.
The gate requires at least 1.25x.

Width 16 measured 4.1568 s, only 0.6% faster than width 8 in a two-sample
follow-up. That difference is not sufficient to select width 16, so width 8
remains the bounded candidate.

The C1 short Newton replay also passed the source-owned C0 trajectory contract.
Both routes made two accepted updates with matching acceptance, stopping,
persistence, and rollback decisions. Maximum terminal differences were
1.78e-15 for state, 2.13e-14 for residual, and 3.98e-13 for the 255-by-255
Jacobian, all within the declared C1 oracle tolerance.

The formal p95, process-tree RSS, GPU-memory, and numerical gates passed for
both dense routes. Only the p50 speed gate failed: C1 reached 1.1289x and C2
reached 1.0854x versus the required 1.25x. Both routes therefore remain
non-promoting, production routing remains unchanged, and no complete campaign
was launched.

## Measurements

| Route | Batch width | Cold (s) | Warm samples (s) | Warm median (s) | Speedup |
|---|---:|---:|---|---:|---:|
| C0 incremental GMRES | 8 | 112.918 | 6.933, 6.781, 6.778 | 6.781 | 1.000x |
| C1 dense LU/refinement | 8 | 105.843 | 4.313, 4.182, 4.175 | 4.182 | 1.622x |
| C1 dense LU/refinement | 16 | 107.083 | 4.208, 4.105 | 4.157 | 1.631x |

Formal isolated-process gate, ten measured processes per route:

| Metric | C0 | C1 width 8 | C2 width 8 | Gate |
|---|---:|---:|---:|---|
| p50 evaluation time | 29.8757 s | 26.4637 s | 27.5241 s | **FAIL** for both dense routes |
| p95 evaluation time | 30.1012 s | 26.8555 s | 27.8752 s | PASS |
| Peak process-tree RSS | 6,237,331,456 B | 5,055,459,328 B | 6,046,306,304 B | PASS |
| Peak GPU memory | 32,415,678,464 B | 32,409,387,008 B | 32,411,484,160 B | PASS |

The formal comparison had objective absolute difference 3.93e-19, gradient
maximum absolute difference 3.64e-15, and gradient L2 difference 1.81e-14. Each
route produced one stable gradient digest across all ten processes.

C2 first passed a bounded same-process canary: 4.1787 s warm median versus the
6.7813 s C0 reference. Its formal isolated-process comparison then had
objective absolute difference 3.39e-19, gradient maximum absolute difference
3.62e-15, and gradient L2 difference 1.86e-14. C2 also produced one stable
gradient digest across all ten processes.

Device: Landau NVIDIA A100-PCIE-40GB,
`GPU-250014ca-8cb3-bdcd-ad1d-2f6f64529b8d`. The current Landau qualification
receipt is `PASS` with SHA-256
`b7339c87be60458c981b35998641a09c754da6248b9ea3a6d3eb2c8c3fbbe4f7`.

Raw arrays, timings, trajectory documents, stderr files, hashes, and the
machine-readable non-promotion disposition are under
`.artifacts/compute-graph-phase0/landau-a100-lean-dense-v1/`.

The formal gate is
`formal-isolated-gate-v1/formal-gate.json`, SHA-256
`72cb2409f9a382f06bcc0f51fe4f6b9b2b5cd5fe99a8a29cf758e98bb4f715b8`.
All 22 child records were copied locally and their recorded byte hashes were
revalidated.

The C2 gate is `formal-isolated-c2-gate-v1/formal-c2-gate.json`, SHA-256
`b90941ed7bc1e2e4d434bab61f50dabab1114549256db7f563bcacec6213daf0`.
Its 11 child records include one excluded cache-prime process and ten measured
processes; the C0 baseline is byte-bound to the C1 artifact above.

The machine-readable dense-branch closure is `dense-branch-disposition.json`,
SHA-256
`bc454d3344be7587b9f7867b8ca95d564d4a8b2f14e1451f15534ac30afe4289`.
It records `ENGINEERING_BOUNDED_NEGATIVE`, leaves `campaign_verdict` null, and
keeps `C0_INCREMENTAL_GMRES` as the production default.

## Phase attribution and next bottleneck

The current RTX 5090 C0 profile attributes 98.35% of device-active time. Device
activity occupies only 44.73% of the 6.8603 s evaluation envelope, while
inter-launch gaps occupy 55.00%. Newton linear solve owns 84.53% of active
device time but only 37.81% of the full envelope. This explains why dense
linearization can win strongly inside one resident process yet miss the formal
fresh-process complete-boundary gate.

Command buffers are already active: graph-launched work covers 73.64% of the
classified device union. Capture tuning was therefore not selected as the next
primary lever. The plan's single-jitted accepted-incumbent candidate boundary
was measured next. It removed the host predicate between anchored forward,
gradient routing, and eligibility construction while retaining host L-BFGS-B
decisions.

That canary was numerically correct but non-promoting. On the immutable A100 v4
snapshot, default warm median was 6.8089 s and the fused boundary was 6.6596 s,
only 1.0224x. The phase gate requires 1.15x. Objective absolute difference was
6.51e-19, gradient maximum absolute difference was 6.40e-16, and gradient L2
difference was 3.72e-15. Both routes accepted the candidate and were internally
repeatable. The losing core route was removed; production routing did not
change.

The machine-readable disposition is `single-jit-v4/disposition.json`, SHA-256
`46f34aaa659198e8e7de69596bf689e59c96362a650b2b11fa3e8229bdb1eaf0`.
The immutable candidate manifest SHA-256 is
`420e2136f5442b62ff11369676a0684594a3d453bfc8cc42441884f2a039dbc8`.
This result shows that the large inter-launch share is mostly fragmentation
inside the nested device execution, rather than the one removable host
predicate at the complete candidate boundary.

The independent fused scalar-Lagrangian pullback canary also stopped at its
phase gate. On the same immutable A100 candidate, the existing split pullback
had a 0.031061 s warm median and the fused pullback had a 0.031920 s warm
median: 0.9731x, or 2.69% slower. Gradient maximum absolute difference was
2.35e-15 and L2 difference was 6.93e-15. The required phase speedup was 1.15x.
No production route changed, and no full campaign was launched.

The separate memory-only A100 rerun measured 32,415,678,464 B peak GPU memory
and 4,172,718,080 B peak self RSS for split, versus 32,424,067,072 B and
4,638,732,288 B for fused. GPU memory grew only 0.026%, but fused RSS grew
11.17%, exceeding the 10% gate. These samples do not replace the timing above.
The combined machine-readable disposition is
`scalar-pullback-v2/disposition.json`, SHA-256
`ceef25206fd90e39d91b01b4f915542390bfae49907c80f51c36eda711946437`.

The profile evidence SHA-256 is
`80509653a65d89c8b916b826dc72c1e082b3e50105a80e698b03ca6e2e0b1cd2`;
the promoting attribution-control evidence SHA-256 is
`f54ede760d6f462920f7acf7a19cd1d4c3b22ceca2ab2cb6b05edcf859f3325c`.
Matched current-source native and Optax complete-path evidence is still absent,
so these facts select an engineering canary but do not fire the plan's formal
faithful-route pivot rule.

## Immutable candidate

The measured candidate is the read-only Landau snapshot
`landau-a100-c1-width8-candidate-v2-immutable`. Its source-manifest SHA-256 is
`b56709728bd91ce8297eeb395d03503db8f787393f217af93e912c51e6d75a92` and
its import-attestation SHA-256 is
`18fbb7652b91f452fb48bdeb84c5e0e4d0a3c06cc7787cb75db25770611357ad`.
Copies of both documents are under `immutable-candidate-v2/` in the artifact
root.

## Portability correction

Landau reconstructed the fitted surface within 5.55e-17 maximum absolute error
but failed the former byte-exact cross-host comparison. The runtime now permits
only a two-epsilon scaled surface-fit reconstruction difference, then installs
the frozen FP64 surface DOFs as the authoritative state. Axis and coil DOFs
remain byte-exact. Material surface drift still fails closed.

Focused validation:

- Dense linearization: 23 passed.
- C1 one-step equations: 7 passed.
- Variant routing and strict-transfer contract: 14 passed.
- Retained-Jacobian transpose adjoint: 15 passed.
- Native-reference and cross-host reconstruction contract: 14 passed.
- Ruff, formatting, and targeted diff checks passed for the portability change.

## Current-source closure attempt and stop decision

The original RTX preflight blocker was resolved by gracefully closing only the
identified Loupe image-viewer process. The exact campaign preflight then passed
with background GPU-process memory fraction 0.0402980955, below the immutable
0.05 limit.

The first complete-path launch exposed a fail-closed manifest-role defect in
the benchmark harness before numerical work began. The isolated launcher
allowlisted `examples.jax.parity.child` but required every allowlisted module to
have manifest role `benchmark`; the child correctly has role
`execution_source`. The launcher now assigns that module its exact manifest
role, and its focused test file passes 19 tests. This is benchmark/provenance
infrastructure, not a numerical-core or optimizer change.

The corrected immutable RTX source snapshot is
`rtx5090-source-snapshot-v18`. Its source-manifest SHA-256 is
`7d6c2aa5370c1032f7d24eb00c57e3850921a8af94d8ec9a015c90120de84692`;
publication-file SHA-256 is
`838d6c60e071b341676e62cee48782c1167eb8c9c4e76c10e5b65c12e0a680f5`;
and import-attestation SHA-256 is
`db4a5da8e3907ec81032e7acea5704b8dda9427b7c6b29d5a13b6fbda8a9e965`.

The provenance-matched RTX v20 baseline passed its first-evaluation gate in
29.5663 s and completed ten isolated warm samples with p50 18.2794 s and p95
19.8821 s. Its source-state SHA-256 is
`2154b95571d5fe24744f70eae38f55e4892b70801e25c63f4d312dd8c8016c48`;
runtime-identity SHA-256 is
`2d335bfd7e604c7db83421a8afa41d2bd559ad4a196b03118c802272e7effd30`;
gate-checkpoint SHA-256 is
`0ca768b7b0ac56d7b356e37f99eb9270149b0dfa1be9beea51626f79fd579832`;
warm-checkpoint SHA-256 is
`4314b3a04e71d70f8b03f44d3dad4d29edc147260111424f4bcb575b89392a8f`.

The matched three-lane complete-path producer then started `native_cpu`. The
live orchestrator reported `native_cpu cold failed with wall_time_limit`; the
partial artifact itself contains no persisted monitored termination receipt.
The partial trajectory contains 383 records; its last record is iteration 383,
objective `1.5339465535214418e-07`, at 879.6247 s from start. Stderr is empty.
The trajectory byte SHA-256 is
`13c1138a1336163d98c55fc47b1d66d394416933fbfe9a6440abf9bd9bd806f0`.
Neither `C0` nor Optax started. Therefore no complete-path receipt, formal gap
budget, campaign `WIN`, or campaign `LOSS` was produced.

The earlier `plan-completion-audit-v1.json` remains a valid record of its own
preflight-time inputs and reports `INCOMPLETE_SHARED_EVIDENCE`, but its stated
desktop-memory blocker is superseded by the later v20 attempt. It must not be
used to claim current formal closure. The current missing evidence is a
complete matched native/C0/Optax path and the dependent gap budget/A100 Phase-0
receipt.

The faithful nested route is now stopped on its engineering gates. No further
native-CPU replay is selected for performance development: the existing
287–351 s native result is retained only as a historical engineering baseline,
not imported into a formal receipt. Production remains `C0` incremental GMRES,
and no measured candidate is promoted.

The next performance work is a separately contracted DESC-style
coupled/fullspace, device-resident formulation. It may reuse the verified
dense-linearization, retained-factor, attribution, parity, and provenance
primitives, but it must own new mathematical-equivalence, trajectory, endpoint,
memory, and timing gates. This result document does not claim that such a
formulation has already beaten native.
