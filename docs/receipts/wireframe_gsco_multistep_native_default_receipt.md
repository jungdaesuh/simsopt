# Wireframe GSCO multistep — native_default GPU receipt (2026-08-15)

First native_default-scale native-CPU vs JAX-GPU receipt in this repository
with full-precision physics agreement. Scope: the
`native-wireframe-gsco-multistep` mirror only
(`examples/jax/3_Advanced/wireframe_gsco_multistep.py` vs
`examples/3_Advanced/wireframe_gsco_multistep.py`). Nothing here generalizes
to other mirrors; the bounded-scale campaign verdicts stand unchanged.

Grade: CERTIFYING — the clean-tree confirmation run below reproduced the
bitwise identity at a committed SHA with `git_dirty_files: []`. The earlier
captures remain self-labelled `diagnostic-not-certifying` and are superseded
as evidence grade (not in content) by the confirmation run.

## Claim

At native_default scale (96x100 half-period wireframe, 19,200 segments, 9,600
cells, 2,500 iterations per step, seven multistep stages):

1. **Physics: bitwise identity.** The final 19,200-entry segment-currents
   vector is bit-for-bit identical (0 ULP, all entries) between the native
   C++/OpenMP example and the strict fp64 JAX lane on an RTX 5090. Support
   (1,750 nonzero segments), step count (7), and the dyadic current ladder
   (+-1e6 x 2^-k A) agree exactly. GSCO's greedy selection is exact
   arithmetic on both lanes and never forks. This exceeds the governing
   `native_workflow` tolerance bucket (rtol 1e-6 / atol 1e-7,
   `src/simsopt_jax/parity_tolerances.py`) by the maximum possible margin,
   and supersedes the campaign receipt's earlier physics verdict for this
   mirror ("CONSISTENT but NOT RESOLVED to the 1e-6 bucket -- print-limited"),
   which had only native stdout at 5 significant figures to compare against.
2. **Speed: ~3.5x on the warmed solve.** Warmed device solve 5.77-5.93 s vs
   20.49 s for the best measured native configuration (OMP_NUM_THREADS=32 on
   a quiet 32-core/64-thread TR 9970X; the shipped native example sets no
   thread count, and the pathological 64-thread default runs ~3.2x slower on
   the solve, 66.17 s vs 20.49 s, ~2.3x on leg wall). End-to-end wall ratio
   at matched workflow: ~4.4x.

## Evidence chain

Durable artifacts: `~/simsopt-campaigns/winnable-six-20260815/` (44-leg
`receipt.json`, per-leg JSON/logs, nvidia-smi samples) and
`~/simsopt-campaigns/winnable-six-20260815/parity-fullprec/` (this receipt's
full-precision captures and comparison output). The capture and comparison
harnesses are also committed in-tree under
`docs/receipts/wireframe_gsco_multistep/` so the numbers have a producing
script at this revision.

- Timing legs (2026-08-15, commit `2e6166505`): `gsco-jax-gpu-CLEAN-warm-r{1,2}`
  and `gsco-jax-gpu-split-CLEAN.split.json` (warm solve 5.765-5.919 s, GPU
  ~99% util / ~442 W); native sweep `gsco-native-CLEAN-omp{16,32,48,64}`
  (best total solve 20.49 s at OMP=32; OMP=48 measures 20.68 s). "CLEAN" is
  the campaign's
  quiet-box criterion (1-minute loadavg ~9-10 at leg start; the 15-minute
  average was still elevated from prior work, and the leg JSONs record
  `loadavg_start`). The leg JSONs do not record working-tree state; the
  parity captures below do. Native stage objectives are identical across
  OMP 4/16/32/48/64 at print precision -- the native reference is
  thread-count-robust for this greedy solver.
- Full-precision parity captures (2026-08-15, HEAD `2e6166505` plus in-flight
  edits recorded per capture): `parity-fullprec/native-omp32.currents.npy` vs
  `parity-fullprec/jax-gpu-head.currents.npy` -- `np.array_equal` True;
  comparison record `parity-fullprec/compare_fullprec_result.json`
  (support identical, max abs diff 0.0, 0 differing entries). The
  `jax-gpu-head` capture's `git_dirty_files` lists only files outside the
  GSCO import chain; the native capture records `git_head` only (its lane
  imports no JAX code). Full-precision final objective agrees to ~3e-15
  relative (summation-order-limited in the diagnostic recomputation; the
  solution vector itself is bitwise equal).
- Loop-columns hoist (landed in the integration commit this receipt
  accompanies): normalizing `WireframeGSCOLiveParams.loop_columns` once at
  the `wireframe_gsco_multistep_loop_jax` entry is bitwise-invariant
  (`jax-gpu-fixed-r{1,2}.currents.npy` == head == native, stage objectives
  bitwise equal; these two captures deliberately ran with the then-uncommitted
  `wireframe_workflow.py` edit dirty, that being the change under A/B) and
  does NOT change solve time (`solve_second_s` 5.876 s post-fix vs 5.765 s
  pre-fix, within box noise): XLA loop-invariant code motion was already
  hoisting the gather. The hypothesis that the per-step gather cost solve
  time is REFUTED at this scale; the hoist is retained as a structural
  guarantee rather than a measured win.

## Qualifiers

- The 3.5x compares warmed steady-state device solves against the best native
  configuration on a quiet box. Cold JAX (compile included) is slower than
  native for a single solve; the persistent compilation cache
  (`JAX_COMPILATION_CACHE_DIR`) amortizes this across processes.
- Contention behavior favors the GPU lane: under heavy host load the native
  OMP kernels degrade far more than the device solve (see campaign
  `receipt.json` contended legs).
- The campaign receipt's MUSE/GPMO narrative is partially superseded by the
  same integration commit: its "roughly half of the JAX iterations are
  fully-computed-then-discarded candidate scans" mechanism note describes the
  pre-change loop; the post-freeze `lax.cond` skip now elides that work. The
  recorded 4.05x MUSE ratio was measured PRE-skip and stands as a dated
  measurement; MUSE has not been re-measured post-skip (and still loses ~2x
  iteration-normalized, so the verdict is not expected to flip).
- This receipt certifies one mirror of the fork-free greedy class under the
  2026-08-15 native_default certification ruling
  (`docs/jax_native_example_end_to_end_parity_implementation_plan.md`,
  addendum). It is not evidence for continuous-optimizer mirrors.

## Clean-tree confirmation run (2026-08-16)

Executed at commit `b26eff5b6` with a clean tree, using the committed
harnesses in `docs/receipts/wireframe_gsco_multistep/`:

- Native lane (`OMP_NUM_THREADS=32`): `native-clean-b26eff5b6.{currents.npy,
  meta.json}` — `git_head b26eff5b6`, full-script wall 55.3 s.
- JAX GPU lane (strict fp64, warm cache): `jax-gpu-clean-b26eff5b6.{currents
  .npy,meta.json}` — `git_head b26eff5b6`, `git_dirty_files: []`,
  build+solve wall 7.4 s.
- Comparison (`compare_fullprec.py`, output
  `compare_clean_b26eff5b6.json`): `bitwise_identical: true`, 0 differing
  entries, support 1750/1750, dyadic ladder identical. The clean-tree JAX
  capture is also bitwise-equal to every earlier diagnostic capture
  (`jax-gpu-head`, `jax-gpu-fixed-r{1,2}`), so the diagnostic evidence chain
  and this certifying run describe one and the same solution vector.

Artifacts: `~/simsopt-campaigns/winnable-six-20260815/parity-fullprec/`.
