# Wireframe GSCO siblings — native_default receipt (2026-08-16)

Scope: the two `examples/2_Intermediate` wireframe-GSCO mirrors and nothing
else —

- `native-wireframe-gsco-modular`
  (`examples/jax/2_Intermediate/wireframe_gsco_modular.py` vs
  `examples/2_Intermediate/wireframe_gsco_modular.py`)
- `native-wireframe-gsco-sector-saddle`
  (`examples/jax/2_Intermediate/wireframe_gsco_sector_saddle.py` vs
  `examples/2_Intermediate/wireframe_gsco_sector_saddle.py`)

This is the sibling campaign to
`docs/receipts/wireframe_gsco_multistep_native_default_receipt.md` and follows
that receipt's protocol: a native OpenMP sweep plus a shipped-default leg, JAX
GPU cold-plus-warm legs against a persistent compilation cache, and a
full-precision comparison of the final segment-currents vectors.

Grade: **DIAGNOSTIC — not certifying.** Every leg JSON in the campaign is
self-labelled `diagnostic-not-certifying`, and this document is untracked
until the orchestrator commits it. The physics result below is at the same
strength as the multistep receipt's; the speed result is a **bounded negative**
that places nothing on the GPU and so needs no promotion.

## Claim

At native_default scale (48x50 half-period wireframe, 4,800 segments, 1,024 x
4,800 area-weighted response matrix, 2,000 GSCO iterations, single stage):

1. **Physics: bitwise identity, both siblings.** The final 4,800-entry
   segment-currents vector is bit-for-bit identical (0 ULP, all entries)
   between the native C++/OpenMP example and the strict fp64 JAX lane on an
   RTX 5090. Verified across **88 native legs x 14-15 JAX-GPU legs per
   sibling** — every OpenMP thread count measured, both capture harnesses,
   cold and warm device legs, both batched and interleaved runs. Support,
   accepted-iteration count and the discrete current ladder agree exactly.
   This exceeds the governing `native_workflow` tolerance bucket (rtol 1e-6 /
   atol 1e-7, `src/simsopt_jax/parity_tolerances.py`) by the maximum possible
   margin, and extends the multistep receipt's fork-free greedy finding to the
   two single-stage siblings.

2. **Speed: the GPU lane LOSES to the best native configuration — bounded
   negative.** On the interleaved A/B (below), the warm device solve is
   **0.89x** (modular) and **0.79x** (sector-saddle) of native
   `OMP_NUM_THREADS=48` on the GSCO kernel, and **0.89x / 0.81x** across the
   whole numerical region. Against the campaign's fair-native reference
   (`OMP_NUM_THREADS=32`) the modular case is a **tie** — 1.11x on the kernel,
   1.01x on the region — while sector-saddle still loses at 0.86x / 0.85x. At
   no comparison level does the GPU beat best-configured native, and a cold
   JAX process is 2.75x slower than native on the solve alone, so both
   examples are placed on the CPU.

   The GPU is 7.0-10.3x faster on the kernel (5.1-7.1x on the region) than the
   *shipped* native default only because that default leaves
   `OMP_NUM_THREADS` unset and the 64-thread OpenMP lane collapses 8.9-11.6x
   on this box. **That comparison is not the headline** — it
   is the same trap as the MUSE mirror, and the fair comparison is against
   OMP_NUM_THREADS=32/48.

   The certified 3.5x multistep win therefore does **not** carry to the
   siblings. The discriminating quantity is per-step work volume, not the
   reduction dimension: both classes share the same 1,024-row reduction, but
   the siblings run one 2,000-iteration stage over 4,800 segments where the
   multistep runs seven 2,500-iteration stages over 19,200 segments.

## Primary evidence: interleaved A/B

Native and GPU batches measured minutes apart drifted by more than the margin
under test (native OMP=32 kernel medians moved 0.40-0.61 s between batches on
the modular case). The primary timing evidence is therefore an **interleaved
A/B** (`ab_interleave.py`): native OMP=32, native OMP=48 and the warm JAX GPU
lane round-robin *inside each of ten rounds*, each leg independently
quiet-gated, so every lane sees the same box conditions. Ratios are
`native / GPU`, so **> 1 means the GPU is faster**.

### `native-wireframe-gsco-modular` (10 rounds)

| Lane | GSCO kernel | numerical region |
| --- | --- | --- |
| native OMP=32 (fair) | 0.612 s (0.369-0.622) | 0.854 s |
| native OMP=48 (best) | **0.492 s** (0.477-0.543) | **0.754 s** |
| JAX GPU warm | 0.552 s (0.546-0.574) | 0.848 s (0.837-0.879) |
| **ratio vs OMP=32** | 1.108x | 1.007x |
| **ratio vs OMP=48** | **0.891x** | **0.889x** |

### `native-wireframe-gsco-sector-saddle` (10 rounds)

| Lane | GSCO kernel | numerical region |
| --- | --- | --- |
| native OMP=32 (fair) | 0.560 s (0.435-0.622) | 0.808 s |
| native OMP=48 (best) | **0.518 s** (0.503-0.526) | **0.775 s** |
| JAX GPU warm | 0.653 s (0.645-0.665) | 0.952 s (0.941-0.961) |
| **ratio vs OMP=32** | 0.858x | 0.849x |
| **ratio vs OMP=48** | **0.792x** | **0.814x** |

Paired per-round ratios agree with the ratio of medians to within 0.02 in
every cell (`receipt.json` → `ab_interleaved.ratios_native_over_gpu`), so the
comparison is not an artifact of pooling.

**Dispersion note.** The GPU lane is the *stable* one here: its ten warm solves
span 0.546-0.574 s (modular) and 0.645-0.665 s (sector-saddle), a 3-5 % spread.
Native OMP=32 is bimodal — modular sorted kernels are `0.369, 0.411, 0.464,
0.561, 0.610, 0.613, 0.616, 0.618, 0.618, 0.622` — a fast mode near 0.4 s and
a slow mode near 0.62 s, which is thread-placement luck rather than load
(every one of those legs passed the same quiet gate). OMP=48 is tight
(0.477-0.543) and is the configuration a native user should actually run. The
receipt quotes medians; a reader who prefers best-case native should note that
native's fast mode beats the GPU by up to 1.5x, never the other way round.

## Corroborating evidence: batched sweeps

Measured before the A/B, in separate batches, and consistent with it in
direction at every configuration.

- **Native example lane, unmodified scripts** (5 reps per configuration, the
  example's own `deltaT` timer around `optimize_wireframe`):

  | sibling | OMP=16 | OMP=32 | OMP=48 | shipped default (unset) |
  | --- | --- | --- | --- | --- |
  | modular | 0.893 s | 0.763 s | 0.711 s | **8.936 s** |
  | sector-saddle | 0.747 s | 0.714 s | 0.613 s | **6.448 s** |

  Whole-process wall for the same legs: modular 1.773 / 1.718 / 1.674 /
  9.903 s; sector-saddle 1.970 / 2.051 / 1.967 / 7.570 s.

- **Native stage split at the shipped default** (`OMP` unset, 5 reps): GSCO
  kernel 5.684 s (modular) and 4.580 s (sector-saddle), against warm GPU
  solves of 0.552 / 0.653 s — the 10.3x / 7.0x figure quoted above as *not*
  the headline. This configuration is also wildly unstable (kernel 1.36-6.57 s
  across the five modular reps, 3.57-4.71 s across the sector-saddle ones).

- **JAX GPU cold vs warm** (fresh compilation cache, then three warm legs):

  | sibling | cold solve | cold region | warm solve | warm region |
  | --- | --- | --- | --- | --- |
  | modular | 1.352 s | 1.816 s | 0.567 s | 1.005 s |
  | sector-saddle | 1.422 s | 1.874 s | 0.655 s | 1.056 s |

  A cold JAX process is 2.75x slower on the solve than native OMP=48 (both
  siblings), before its 0.72-0.96 s of module import. The cold leg writes 43 cache entries; every
  warm leg leaves the count at 43, which is the evidence that no warm leg
  recompiled. Four of the eight batched GPU legs (modular warm-r2 and all three
  sector-saddle warm legs) partially shared the device with the sibling LM_QR
  campaign — see the GPU-collision qualifier below. The interleaved A/B, which
  recorded a clean device, reproduces the batched warm figures (sector-saddle
  0.653 vs 0.655 s; modular 0.552 vs 0.567 s), so the table above stands as
  corroboration with that caveat.

### What the comparison levels mean

`optimize_wireframe` is two stages — the host Biot-Savart response build
(`bnorm_obj_matrices`) and the C++ GSCO solver (`gsco_wireframe`) — and the JAX
mirror splits the same two (`bnorm_obj_matrices_jax` then
`gsco_wireframe_jax`). The rows above compare matched stages:

- **GSCO kernel** — the ported computation itself. Both `gsco_wireframe` and
  `gsco_wireframe_jax` read the same four wireframe topology accessors on
  entry (`get_cell_key`, `get_free_cells('logical')`, `segments`,
  `connected_segments`), so those are inside the kernel figure on both lanes.
  Native figures come from `native_split_probe.py`, which reproduces each
  example's setup verbatim and times the stages separately; its currents vector
  is bitwise equal to the unmodified example's on every leg, so the split is a
  timing decomposition of the same computation, not a different one.
- **numerical region** — problem setup + response build + solve, on both
  lanes. The native response build is 0.198-0.242 s and is host C++ in both
  lanes, so it dilutes the ratio rather than creating it.
- **process wall** is recorded but deliberately **not** ratioed: the native
  example scripts also render a 2-D current plot and write VTK after the solve,
  which the JAX capture does not do. Those walls are not matched work.

The JAX lane additionally pays 0.72-0.96 s of interpreter/module import per
process, excluded from every ratio above. The native lane pays its own import,
which is why the process-wall column is reported but not ratioed.

**Topology-accessor asymmetry, resolved.** The sector-saddle example calls
`wf.make_plot_2d(quantity='constrained segments')` *before* its `t0`, which
warms those four accessors; with toroidal breaks they cost **0.126 s**, versus
0.000 s for the modular case which has none. That is why the sector-saddle
example's `deltaT` (0.714 s at OMP=32) sits below the split probe's
bnorm+kernel sum (0.809 s). For the modular case, where the accessor cost is
zero, `deltaT` and the sum agree to 0.4 % at OMP=48 (0.711 vs 0.7085 s —
different batches, but a configuration that is tight across batches); at
OMP=32 the split-probe figure is the bimodal one (0.400/0.573/0.611 s across
three batches) and the batch drift documented under Qualifiers (up to 53 %)
swamps the comparison. The
split probe times the accessors separately (`wall_topology_accessors_s`) and
counts them inside the kernel, because the JAX lane pays them inside
`solve_s`. Excluding them instead would move the sector-saddle kernel ratio
from 0.79x to 0.60x — further against the GPU, not for it.

## Parity

`compare_fullprec.py` (byte-identical to the committed multistep harness),
native OMP=32 vs JAX-GPU warm:

| Sibling | bitwise identical | differing entries | max abs diff | support | nonzero |
| --- | --- | --- | --- | --- | --- |
| modular | **true** | 0 / 4,800 | 0.0 A | identical | 596 |
| sector-saddle | **true** | 0 / 4,800 | 0.0 A | identical | 1,094 |

Extended to the whole leg set (`parity-fullprec/parity_all_legs.json`): for
each sibling, all **88** native captures (OMP 8/16/32/48/unset, example lane,
split probe and A/B, this session and the predecessor's) and all **15**
(modular) / **14** (sector-saddle) JAX-GPU captures produce one and the same
vector — **max 0 ULP**, zero non-identical legs on either lane. The native
reference is thread-count-robust for this greedy solver, exactly as the
multistep receipt found.

Discrete current ladder, identical on both lanes: modular
`+-208333.33333333334 A`; sector-saddle `{-416666.6666666667, -250000.0,
+250000.0} A`. Accepted iterations: modular 1,846, sector-saddle 1,698 (of
2,000 allowed) — identical on both lanes.

The full-precision final objective `f_B`, recomputed on the host from A, b and
x, agrees to **1.3e-15 relative / 6 ULP** (modular) and **1.9e-16 relative /
1 ULP** (sector-saddle). That residue is summation order in the diagnostic
recomputation, not a solution difference: the solution vectors are bitwise
equal, and the same `f_B` recomputation also drifts by 1-2 ULP *between native
OpenMP thread counts*.

## Method and environment

- Commit `829d92f2396b74a47ac72e90c05cb5ded6ee0658`, branch
  `pr/jax-port-squashed`; every leg JSON records its own `git_head` and
  `git_dirty_files`. The tree state varied across the campaign's phases: 43
  legs (the earliest sweeps) record a fully clean tree; 52 record exactly one
  untracked file (`docs/receipts/lm_qr_gpu_probe.md`, the sibling campaign's
  draft); the 110 interleaved-A/B and v2 split-probe legs record three
  campaign-doc changes (`docs/jax_example_device_assignment.md` modified, this
  receipt and the sibling receipt untracked). All dirt is under `docs/` and
  outside the GSCO import chain; no file under `src/` or `tests/` was touched
  by this campaign.
- Host: AMD Ryzen Threadripper 9970X (32 cores / 64 threads) + NVIDIA GeForce
  RTX 5090 (32 GB, driver 595.84), Linux 7.0.0-28-generic.
- Runtime: `.venv-qn-gpu` Python 3.11.15, jax/jaxlib 0.10.0 (CUDA), numpy
  2.4.6, `simsoptpp` extension sha256 `41b2ca791a720f32...`.
- Native legs: `MPI4PY_RC_INITIALIZE=false`, `JAX_PLATFORMS=cpu`,
  `CUDA_VISIBLE_DEVICES=` empty, all `SIMSOPT_*` variables unset, OpenMP thread
  count set per leg (or left unset for the shipped-default leg).
- JAX GPU legs: `SIMSOPT_BACKEND_MODE=jax_gpu_fast`,
  `SIMSOPT_BACKEND_STRICT=1`, `SIMSOPT_PRECISION=fp64`, `JAX_ENABLE_X64=1`,
  `JAX_PLATFORMS=cuda`, `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `OMP_NUM_THREADS=32`,
  `MPI4PY_RC_INITIALIZE=false`, and **`JAX_TRANSFER_GUARD=disallow` /
  `SIMSOPT_JAX_TRANSFER_GUARD=disallow`** — the strict setting. A preflight leg
  established that the capture harness runs clean under it, so no leg fell back
  to `log`.
- Two fixes were made to the predecessor session's inherited `run_jax.sh`: the
  determinism flag `XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops=true` was
  absent and is required by the handoff's strict example lane, and the transfer
  guard was hard-pinned to `log` instead of being parameterised. Nothing else
  in the inherited harnesses was changed; `compare_fullprec.py` is byte-for-byte
  the committed multistep version.
- Warm-cache protocol: `JAX_COMPILATION_CACHE_DIR` per sibling with
  `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0` and
  `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0`. The cold leg starts against a
  freshly emptied directory.
- Device occupancy during the GPU legs (`legs/*.nvsmi.csv`, 0.25 s sampling):
  169-185 W throughout, peak sampled utilization 88-96 % on the sector-saddle
  legs and 18-58 % on the modular legs. The modular figures are a **sampling
  artifact, not an idle device**: a 0.25 s sampler lands only 2-3 samples inside
  a ~0.55 s solve, so it frequently misses the burst. Treat the utilization
  column as corroboration that the device really runs the solve, not as a
  duty-cycle measurement.
- Quiet gate: every A/B leg, every native example-lane leg, every split-probe
  leg and every JAX leg waited for host CPU utilization (2-second `/proc/stat`
  delta) below 15 % before starting, and recorded box state on both sides.
  **All 158 gated legs passed as `idle`**; the worst value observed at any leg
  start was 14.7 %.

## Qualifiers and anomalies

- **Foreign load.** Long-running `pytest` processes from other sessions were
  resident on the box throughout — an orchestrator-session observation, **not**
  something the leg JSONs can show: the box-state schema records loadavg,
  aggregate `/proc/stat` busy % and GPU compute apps, not a host process list.
  The recorded gate evidence is the busy %: all 158 gated legs passed as
  `idle`, and the worst value at any leg start was 14.7 %. The 1-minute load
  average reached 70 at some native leg starts, but that is the decay tail of
  this campaign's own preceding 64-thread legs; the 2-second utilization gate
  is the meaningful signal.
- **GPU collision discipline — partial failure, disclosed.** The GPU legs
  waited on the sibling LM_QR campaign's `GPU_DONE.sentinel`, which appeared 7
  minutes into the wait (`gpu-gate/gate.log`, OPEN 08:35:17); the nvidia-smi
  fallback was never needed. The sentinel did not, however, mark the device
  free for good: the sibling re-took the GPU at 08:43-08:45:48 for an unplanned
  stability probe (its sentinel refresh, mtime 08:45:50, records this). Four of
  the eight batched JAX-GPU legs — modular warm-r2 and all three sector-saddle
  warm legs — overlapped that window, with sibling processes (pids
  1468461/1469670/1471061/1471727) holding 564-944 MiB on the device. The
  per-leg box-state JSONs classify those processes as `baseline: True` because
  `boxstate.py::_is_baseline` substring-matches `"code"` against the repo path
  — a classifier defect, disclosed here rather than fixed retroactively. The
  **primary evidence is unaffected**: every interleaved-A/B GPU leg
  (09:05-09:14) records a genuinely baseline-only device, and the A/B warm
  medians reproduce the batched ones (sector-saddle 0.653 vs 0.655 s; modular
  0.552 vs 0.567 s), so every ratio in this receipt rests on clean-device
  measurements and the batched table is corroboration under the caveat noted
  where it appears.
- **Batch drift.** The reason this receipt leads with an interleaved A/B is
  that it did not at first: separately-batched native runs of the same
  configuration disagreed by up to 53 % on the modular kernel (0.400 s vs
  0.573 s vs 0.611 s medians, all n=10 and all quiet-gated). Batched sweeps
  cannot resolve a 10 % margin on this box; the batched numbers are kept above
  as corroboration of direction only.
- **Stale artifact block.** `receipt.json`'s `siblings/modular/ratios` block
  predates the interleaved A/B and still records `best_native_omp = 32`; the
  authoritative ratios live in its `ab_interleaved` block (best native =
  OMP 48), which is what this document quotes throughout. The stale block is
  left unedited for provenance.
- **Shipped-default instability.** The `OMP_NUM_THREADS`-unset native leg is
  wildly unstable (kernel 1.4-6.6 s, example-lane solve 4.3-52.8 s). Its
  median is reported for completeness; it is a pathological configuration, not
  a baseline.
- **Direction vs magnitude.** The direction (best-configured native faster) is
  stable across both comparison levels and both siblings. The magnitude
  (0.79-0.89x) is a dated measurement on one host and would move with a
  different core count; on a host with fewer or slower cores the modular ratio
  in particular could cross 1. The `cpu` assignment rests on the direction
  against best native plus the cold-start cost, not on the magnitude.
- **Modular is close.** Against the OMP=32 fair-native reference the modular
  numerical region is 1.007x — a dead tie. It is placed `cpu` rather than
  `either` because (a) best-configured native (OMP=48) wins by 1.12x, (b) a
  cold JAX process loses 2.75x on the solve and the warm advantage requires a
  persistent compilation cache that nothing configures by default, and (c) the
  JAX lane pays 0.72-0.96 s of module import per process on top of its
  numerical region. A future host where
  those three stop holding would justify revisiting the row.
- This receipt makes no claim about any other mirror, and none about these two
  siblings at any scale other than `native_default`.

## Artifacts

Host-local campaign directory (**not** in this repository, not reviewable from
a clone): `~/simsopt-campaigns/gsco-siblings-20260816/`.

- `receipt.json` — the aggregated scoreboard this document quotes, with the
  `ab_interleaved` block as primary evidence.
- `native_capture.py`, `jax_capture.py`, `compare_fullprec.py` — capture and
  comparison harnesses, inherited from the predecessor session and adapted from
  the committed multistep set under `docs/receipts/wireframe_gsco_multistep/`;
  `compare_fullprec.py` is byte-identical to the committed one.
- `native_split_probe.py` — the native stage decomposition (setup, response
  build, topology accessors, GSCO core).
- `ab_interleave.py` — the interleaved native/GPU A/B driver.
- `run_native.sh`, `run_native_split.sh`, `run_jax.sh` — per-leg env drivers.
- `sweep_native_v2.py`, `sweep_jax.py`, `build_receipt.py`, `boxstate.py` —
  gated sweep drivers and aggregation.
- `legs/` — per-leg stdout logs and `nvidia-smi` samples.
- `boxstate/` — per-leg quiet-gate and box-state records (158 legs).
- `parity-fullprec/` — per-leg `.currents.npy` + `.meta.json`, the two
  `compare_<sibling>.json` verdicts, and `parity_all_legs.json`.
- `superseded/split-v1/` — the first split-probe batch, kept because the
  receipt's batch-drift qualifier cites it.
- `gpu-gate/` — the sentinel poller, its log and the `GATE_OPEN` marker.

## Consequences for the device-assignment record

`docs/jax_example_device_assignment.md` moves both rows off `unmeasured`:

| Example ID | before | after |
| --- | --- | --- |
| `native-wireframe-gsco-modular` | unmeasured / census-structural | **cpu / measured-diagnostic** |
| `native-wireframe-gsco-sector-saddle` | unmeasured / census-structural | **cpu / measured-diagnostic** |

Both rows open with the `sequential chain` mechanism family: GSCO's
2,000-iteration greedy chain is strictly loop-carried, and at 4,800 segments
the per-step work no longer covers the device's overheads — which is precisely
the quantity that separates these two from the certified multistep win. The
summary counts in that document move from 1 gpu / 23 cpu / 15 unmeasured to
**1 gpu / 25 cpu / 13 unmeasured**, and within the 27 `native-*` mirrors from
1/13/13 to **1/15/11**.
