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
