# Nested-LS eight-term outer charter (FD-0 → B3 → B37)

> **Pre-declared 2026-08-22 at `56b1dec05`, before any outer-loop lever
> exists in the tree.** This is the contract for the last open variant of
> the VMEC-free single-stage claim: a moving-coil outer optimization whose
> surface sits on the Boozer-LS manifold at every accepted iterate, timed
> end-to-end against an OMP-swept native nested bar. Successor to the
> Gate-6 inner-solve claims in
> `docs/receipts/nested_ls_reduced_track_20260820.md` (closed both hosts).
> Not F3 7.70× (that is the flat formulation; its endpoint is off-manifold
> per `docs/receipts/boozer_unnest_newton_reconstruct_diagnostic_20260820.md`).

## Problem

Outer variable: the **11 coil DOFs**. Vessel DOFs (3) and the 661 surface
Fourier DOFs are **not** outer variables — the surface is eliminated by the
reduced nested-LS inner solve `s*(c)` (255×64 grid, 661 DOFs,
`constraint_weight=1.0`, `weight_inv_modB=True`, free `G`, Volume target
0.1), with `y=(ι,G)` eliminated inside it by the 48960×2 QR. Outer
objective: the **eight-term flat-675 `J`** (GATE-1 lineage: `non_qs`,
`boozer_residual`, `iota_penalty`, `curve_length`, `curve_curve_distance`,
`curve_surface_distance`, `surface_vessel_distance`, `curve_curvature`,
frozen bundle weights) evaluated at `(c, v_frozen, s*(c))`. This matches
the native banana shape: coils-only outer, manifold-enforced surface.

## Gradient (the object under test)

`g(c) = ∇_c J_flat(c, v, s*) − λᵀ H_sc`, with `λ` from the stabilized
Schur dense-LU adjoint solve at cotangent `∇_s J_flat(c, v, s*)`. The flat
gradient differentiates **through the QR y-solve**, so the `y`-chain is
already inside both blocks; the flat gradient itself is certified
flat-vs-native by GATE 1. `H_sc` action and the Schur factor are the
validated pieces from the reduced track (Schur vs AD-through-QR
`rel_l2 ≤ 8.3e-16`; adjoint live-η machinery from the Volume canary).

## Gate FD-0 — multi-direction step-halved FD (blocks B3)

Frozen point: the dense-LU walk endpoint (same freeze as the Volume
canary). Fail-closed probe, one receipt:

- **All 11 coil-DOF unit directions.** Central differences at steps `ε`
  and `ε/2` (`ε` pre-chosen per direction from coil scale, recorded).
- Per direction: relative error of `g·d` vs FD at the better step
  **≤ 1e-5**, and halving must reduce the error (order sanity). The
  Volume canary's one-direction 4.32e-6 is the reference magnitude.
- Adjoint live-η ≤ the frozen `F3_B37_ADJOINT_LIVE_ETA_TOL` at the
  claim point.
- Every perturbed inner re-solve must succeed and pass the C++ rejudge
  no-op check (`iter=0`); coils delta exactly the FD step; any failure
  fails the gate closed. No timing content in FD-0.

## B3 → B37 contract (pre-declared claim boundary)

- Outer optimizer both lanes: **scipy L-BFGS-B over the 11 coil DOFs**,
  same `maxcor`, budgets `nit=3` (B3) then `nit=37` (B37). B37 runs only
  after B3 lands physics-green.
- **JAX lane**: reduced nested-LS inner (opt-in `dense_lu` /
  Shamanskii per the track's certified paths), persistent compile cache,
  rolling warm-start of `s*` across outer iterates, gradient per this
  charter. One lane per process.
- **Native lane**: the banana nested workflow — `run_code` inner
  (BFGS 1500 / Newton 50) under the same eight-term outer `J` and the
  same scipy L-BFGS-B budget. Building this native-twin driver is
  chartered implementation work, not a lever.
- **OMP law**: native swept over the host's frozen sweep set (5090:
  `{4,8,12,14,16,20,24,32}`; A100: contract set), pinned and observed,
  best-of-contract is the bar. Interleaved pairs, n=3 per rung.
- **Claim clock: full parent process wall on both sides.** No
  subtractions. Physics rejudge runs **outside** the timed window.
- **Aggregation: min**, medians published beside it, per-repeat rows in
  the artifact. Hosts: 5090 first, A100 rung second, each against its
  own bar.
- **Physics gate (per lane endpoint, untimed)**: C++ LS Newton
  reconstruct no-op (`iter=0`, coils frozen, `‖∇J_LS‖₂ ≤ 1e-13`) — the
  nested formulation must earn its defining property; plus endpoint
  eight-term `J` parity vs the native lane under the F3-style one-sided
  `1e-10` gate. Trajectory parity is NOT claimed.

## Amendment 1 (2026-08-22, pre-B3, no clock has run)

Two rulings from the implementation shakedown, frozen before any timed
rung:

- **Iota-branch guard, both lanes.** The B3 smoke measured L-BFGS-B's
  first unit-scale coil step throwing the nested inner solve onto a
  different Boozer branch (ι 0.1409 → −0.0024, J 0.0143 → 10.43) with
  every inner solve converging, so no rejection fired. The implicit
  surface `s*(c)` is only locally defined; an inner solve whose ι moves
  more than `NESTED_LS_OUTER_IOTA_BRANCH_GUARD` (contract module) from
  the last **accepted** anchor is a failed evaluation and takes the
  sealed rejection sentinel (anchor `J` + offset + scale·‖Δc‖, anchor
  gradient) in **both** lanes identically. FD-0 is unaffected (its
  steps move ι negligibly).
- **Inner-solve non-convergence sentinels in both lanes.** The sealed
  lane policy's rejection knobs exist for expected inner failures; the
  native child already sentinels them, so the JAX lane does the same
  (typed signal, receipt reason `inner_solve_failed`) rather than
  crashing — asymmetric brittleness would bias the comparison.
  Coil motion inside an inner solve stays fatal in both lanes (invariant
  violation, not a physics event); FD-0 legs never sentinel — a
  certification probe fails closed.
- **The endpoint eight-term J parity rtol is provisional.** Lane-vs-lane
  gradient agreement is ~1e-8 (native adjoint vs reduced Schur adjoint),
  so budget-truncated trajectories will fork; a hard 1e-10 endpoint gate
  is the known false-reject landmine class. B3 measures the achievable
  fork band; the B37 gate value is frozen from that measurement, in the
  receipt, before B37 runs. The reconstruct no-op gate is unchanged and
  not provisional.

## Amendment 2 (2026-08-23, pre-B3, no clock has run)

Mechanics pinned from external review before the first timed rung:

- **The B3 native OMP is artifact-bound, not CLI-trusted.** A native-only
  outer OMP sweep (frozen host set, n=2 interleaved, full parent wall,
  own evidence schema) precedes the B3 claim; the claim requires the
  sweep artifact, refuses an `--omp` that is not its best, and binds the
  artifact's sha. `omp_provenance="swept_artifact"` at B3;
  `"b3_receipt"` at B37, which additionally refuses a B3 whose own
  provenance is not the swept artifact.
- **J-parity is observational at B3 and frozen for B37.** B3 records the
  per-pair endpoint-J gap and publishes `measured_j_rel_gap_max`
  without gating on it (rejudge gates unchanged); B37 requires an
  explicit `--j-parity-rtol` that is ≥ the B3-measured value, gates
  one-sided at that frozen band, and records both numbers. Freezing the
  value is an adjudicated act recorded in the receipt.
- **Native-lane residual JAX overhead is disclosed-bounded, not zero.**
  The timed native child no longer runs any JAX solve or QR, but its
  loaders transitively import JAX and construct-and-discard the twin;
  the child publishes `module_import_seconds` and
  `problem_build_seconds` so every receipt can bound the padding
  (measured ~1.4 s + ~3.5 s against ~10³ s walls, and it biases
  against the JAX lane).

## Amendment 3 (2026-08-23, pre-FD-0-rerun, written while the red run executes)

The first FD-0 run (5090, all 11 directions, as-run receipt preserved)
failed every ε-floor-clamped direction and passed both derived-ε
directions, with **every failure improving under halving** — the
truncation signature, not the composition-bug plateau and not the noise
signature (worsening under halving). Root cause: the fixed two-rung ε
ladder with an absolute floor makes a fixed step a large *relative*
perturbation on small-|c| DOFs, outside the quadratic FD regime (the
same soft-mode nonlinearity the branch-guard smoke exposed at unit
scale). Third instance of the program's gate-parameter law: when a gate
fails, audit the gate's own parameter before touching the physics.

Replacement, fail-closed descent ladder per direction:

- Start from the existing per-direction ε rule; keep halving while the
  halved step improves the relative error and the error exceeds the
  1e-5 band, to a pre-declared maximum depth (contract constant).
- The noise floor is **measured, not guessed**: repeated inner
  re-solves at the base point publish the J scatter δJ, and the
  per-direction minimum step follows from δJ, the band, and |g·d|
  (contract-defined formula). Descending past it is forbidden.
- Distinct fail-closed reasons, all published with the full ladder
  (every rung's ε, FD values, error): `ladder_exhausted` (max depth),
  `halving_stopped_improving` (the composition-bug detector — a wrong
  gradient plateaus above the band), `noise_floor_reached_above_tol`
  (the valley never dips under the band — adjudicated in the open on
  the published scatter, never silently relaxed).
- The red first-run receipt is certification data, not an
  embarrassment: it lands as-run under a `.amendment2-red` suffix (its
  internal `execution_log` name predates the rename; disclosed here),
  and the Amendment-3 rerun writes the canonical receipt the
  skip-until test gates.

## Amendment 4 (2026-08-24, prose-only correction, no clock affected)

**Defect.** The B3 → B37 contract prose above (native lane bullet) reads
`(BFGS 1500 / Newton 50)`. The Newton budget was never 50. The frozen
constant is `NESTED_LS_BANANA_NEWTON_MAXITER = 40`
(`src/simsopt_jax_adapters/geo/nested_ls_contract.py:34`), consumed as
`newton_maxiter=` at `:175`, and it matches the upstream SIMSOPT default
of 40 (`src/simsopt/geo/boozersurface.py:103`, applied at `:134` and
`:141`). BFGS 1500 is correct and unchanged.

**Ruling.** 40 is, and has always been, the contract value. No executed
leg ever ran a Newton budget of 50, so no receipt, no gate, and no
published number is affected, and nothing is re-run on account of this.
The defect is descriptive prose only.

**Disposition.** Per the append-only discipline the original bullet is
left byte-identical rather than edited in place; this amendment carries
the correction, and a reader who reaches the prose before this section
should take 40. Recorded because the charter's value is that its text can
be checked against the code — a prose/constant mismatch found by a
reviewer costs more than the typo it describes.

## NO-GO (inherited and binding)

Mimicking native BFGS-shape gradients in JAX (measured ~273 s envelope);
claim-grade GMRES walk; dense `H_ss` materialization at 661 (~66 GB);
Optimistix/Lineax as a mid-campaign swap (they belong to a future charter
revision, adopted only between rungs); unpinned or unswept native
denominators; non-interleaved timing; editing sealed receipts (remint
only); claims from FD-0 (it is a physics gate, not a speed receipt).

## Evidence

`docs/receipts/evidence/nested_ls_outer_fd0_<date>.json` then
`nested_ls_outer_b3_<date>.{json,log}` / `nested_ls_outer_b37_<date>.{json,log}`
(+ `.a100` tags), skip-until tests in
`tests/geo/test_nested_ls_reduced_scale.py` per the established pattern,
clean implementation tree from the first byte of any claim run, producer
SHA recorded. Receipt narrative lands in the reduced-track doc as new
sections.

## Non-claims

Not F3 7.70× and not a supersession of it; not exact (`r=0`) Boozer; no
VMEC anywhere; no claim about formulations or hosts outside the two
rungs; a bounded negative at B37 is a legitimate closure of this charter.

## Amendment 4 (2026-08-24, pre-relaunch, after the pair-2 fault)

The first B37 run died at 9.6 h when an operator-launched GPU probe
starved its pair-2 JAX child of device memory (cuSolver init failure) —
an infrastructure fault, not a physics event; the incident, its
mechanism (default XLA preallocation on the probe), and the quiet-box
lesson are disclosed here and in the receipt narrative. Rulings for the
relaunch, frozen before its first byte:

- **The rerun's verdict is the physics gate, which is deterministic**:
  the JAX lane reproduced its trajectory bitwise across the prime and
  pair 0 (J identical to the last digit), and the endpoint-J parity
  outcome at the frozen 1e-9 band is therefore repeat-independent. The
  rerun runs **one interleaved pair** (`--pairs 1`); walls publish as
  informational, min==median==the single row, and no timing claim of
  any kind attaches to this receipt.
- **The untimed cache prime may be skipped** (`--skip-prime`): the
  persistent compile cache is demonstrably warm from the faulted run's
  own legs; the flag and reason land in the receipt.
- The first run's logged rows (prime, pair 0 both lanes, pair 1 both
  lanes) are preserved as evidence of the trending verdict and of
  trajectory determinism, but are not receipt rows; the receipt is the
  rerun's own.
