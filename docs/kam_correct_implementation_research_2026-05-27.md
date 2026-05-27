# Correct Full Implementation of KAM for Banana Coil Optimization — GPD Research

Date: 2026-05-27
Status: Research deliverable (GPD research-phase artifact). Not a commitment; feeds planning.
Routed via GPD: `route_skill` → gpd-verify-work / gpd-literature-review. Verification framing from
the GPD `fluid_plasma` checklist. Literature was surfaced and cross-checked via the GPD arXiv tool
(IDs in §7); three papers were read in full (2403.19003, 2106.14930, 2102.04497), the remainder
confirmed at search-metadata level (title/author/abstract returned by the tool).
Builds on: the validation finding that `topology_scorer.py::kam_fraction` is a mislabeled
footprint-localization proxy (false-positives localized chaos, false-negatives intact outer tori),
and the existing `docs/greene_residue_topology_gradient_impl_plan_2026-05-11.md`.

> **Revision 2026-05-27 (post GPD cross-validation):** confirmed arXiv:2102.04497 (Geraldini–
> Landreman–Paul, adjoint island-width sensitivity) — earlier "verify" caveat removed; de-hardcoded
> the O/X residue labels (paper 2106.14930 and local `docs/source/example_islands.rst` give *opposite*
> θ→O/X maps — classify from residue sign, not the θ seed); tightened the "verified"
> wording into explicit [full]/[meta] levels; split the Track-B phase to separate the WBA classifier
> (~2 modules) from the multi-file schema rename (~7 production + ~4 test files;
> `single_stage_search_contracts.py` alone carries ≈56 `kam` references).

---

## 0. Executive summary

- **KAM is genuinely relevant.** Magnetic flux surfaces *are* the invariant tori (KAM tori) of the
  field-line flow. Their breakup into islands/chaos is exactly loss of confinement. A KAM/island
  criterion is a legitimate objective and certificate for stellarator coil optimization.
- **The central design error to avoid: conflating two different quantities.**
  - **(A) An island/chaos *optimization objective*** — smooth, differentiable, steers coils:
    the **Greene residue** of targeted rationals (→0 ⇔ island width →0). This is what the existing
    Greene-residue plan correctly targets.
  - **(B) A KAM-surface *existence certificate*** — per-field-line, replaces `kam_fraction`:
    the **weighted Birkhoff average (WBA)** convergence-rate classifier (regular/island/chaos),
    optionally backed by **converse KAM** (non-existence) and **QFM/ghost** almost-invariant surfaces.
  The broken `kam_fraction` was trying to be (B) and failed; the Greene plan implements (A). A
  *correct full implementation needs both*, and they are different algorithms.
- **Recommendation: two tracks.**
  - **Track B (do first, low risk, immediate):** replace `kam_fraction` with a WBA-based invariant-
    torus fraction + rotation-number diagnostic. Reuses existing `compute_fieldlines`; non-
    differentiable but a *correct* certificate. The classifier itself is ~2 new modules + a standard-map
    unit test; the production *schema rename* is a separate, larger step (~7 prod + ~4 test files — §6).
  - **Track A (the differentiable driver):** implement the direct-BiotSavart Greene residue per the
    existing plan, with the corrections/sharpenings in §3.2. This is the real KAM-aware optimizer term.
- **Banana caveat:** field-line KAM integrity is *necessary but not sufficient* for trapped/banana
  particle confinement. KAM ≠ particle confinement. Keep `Gamma_c`/`epsilon_eff` as the particle-side
  metric (currently stubbed "unavailable" in the scorer). Do not market a KAM fraction as confinement.

---

## 1. Physics framing

### 1.1 The field-line flow is Hamiltonian / volume-preserving

For a magnetic field **B**, field lines obey `dx/dτ = B(x)`. In a torus with toroidal angle φ as
"time", this is a 1½-degree-of-freedom Hamiltonian system; the Poincaré return map on a φ = const
plane is **area-preserving** (more generally the 3-D flow is volume-preserving). Its invariant circles
are the intersections of **invariant tori = magnetic flux surfaces**.

Key simplification for `banana_opt`: it is a **current- and pressure-free (vacuum) coil field**.
Then the field-line Hamiltonian *is the BiotSavart field directly* — no MHD equilibrium (VMEC/SPEC)
is required to define the tori. You can trace the BiotSavart field and analyze its return map directly.
(SPEC is only needed when a finite-β / finite-current equilibrium must be solved; `spec.py::Residue`
is for that case.)

### 1.2 What KAM actually says, and Greene's criterion

- **KAM theorem:** tori with *sufficiently irrational* (Diophantine) rotation number ι survive small
  perturbation; near-rational tori are destroyed first, breaking into island chains.
- **Chirikov:** when neighbouring island chains overlap, the region between them becomes chaotic.
- **Greene's residue criterion (Greene 1979, J. Math. Phys. 20, 1183):** the destruction of the KAM
  torus with a given (noble) irrational ι is signalled by the **residues of the periodic orbits whose
  rotation numbers are the continued-fraction convergents p_k/q_k → ι**. Residue
  `R = (2 − tr M)/4` where `M` is the tangent (monodromy) map of the periodic orbit:
  bounded residues (→ a fixed-point value, ≈0.25 at criticality for the golden mean) ⇒ torus exists;
  diverging residues ⇒ torus destroyed. The golden-mean circle is the *last* to die; for the standard
  map at `k_c ≈ 0.9716`.
- **Two uses of the residue:**
  - For a single rational p/q, `R → 0` ⇔ that island chain's width → 0 (**island healing**). This is
    the differentiable optimization target (Track A).
  - The *sequence* `{R(p_k/q_k)}` certifies existence of the irrational KAM torus (a Track-B certificate),
    but is more expensive than the WBA below.

### 1.3 Why the current metric is the wrong object (recap, grounds the redesign)

`kam_fraction` thresholds the (R,Z) bounding-box diagonal of pooled Poincaré hits vs
`0.25·cross_section_span`. That measures *footprint localization* (dominated by minor radius /
poloidal extent), not *torus integrity* (a 1-D invariant curve vs a 2-D area-filling chaotic orbit).
Proven: intact outer torus → 0.0 (false negative); localized chaos → 1.0 (false positive). Any correct
certificate must instead test **whether each orbit lies on a 1-D invariant circle**, which is precisely
what the WBA convergence rate decides.

---

## 2. Method taxonomy (the menu, with roles)

| Method | Measures | Differentiable? | Cost | In repo? | Role |
|---|---|---|---|---|---|
| **Greene residue, direct BiotSavart** | island stability of target p/q (R→0 ⇔ width→0) | **Yes** (tangent-map adjoint + `B_and_dB_vjp`) | moderate (Newton orbit + tangent map) | No (planned) | **(A) objective** |
| Greene residue via SPEC/pyoculus (`spec.py::Residue`) | same, on a SPEC equilibrium | yes (FD/adjoint via SPEC) | high (SPEC each eval) | **Yes** | (A) only if SPEC is in the loop (finite β/current) |
| Cary–Hanson island width (PFB 3, 1006, 1991); **adjoint: Geraldini–Landreman–Paul 2102.04497** | physical island width (Wb) | **yes — adjoint shape/coil gradient, verified on a vacuum NCSX coil config** | moderate | No | (A) most physical target; vacuum + coil shape gradient fits `banana_opt` directly |
| **Weighted Birkhoff average (WBA)** | per-orbit regular/island/chaos + rotation number | No (diagnostic) | **cheap** (reuse field-line trace) | No | **(B) certificate — replaces `kam_fraction`** |
| Converse KAM (MacKay 2018; Duignan–Meiss) | rigorous *non-existence* of a transport barrier | No | moderate | No | (B) strongest negative test |
| QFM / ghost surfaces (Hudson–Dewar; Dewar–Hudson–Gibson) | "almost-invariant" surface + residual flux | **Yes** (smooth flux functional) | moderate | **`simsopt.geo` ships `QfmSurface`** | (A) smooth surrogate **and** (B) quality measure |
| Survival / line-lifetime (existing) | field-line loss over τmax | No | cheap | Yes (correct) | keep as-is (confinement proxy, not KAM) |

Greene residue and WBA are the two pillars. QFM is an attractive third option *because simsopt already
has `QfmSurface`* — its residual quadratic flux is a smooth, optimizer-friendly "how non-invariant is
this surface" measure that could serve both as a Track-A surrogate and a Track-B quality number.

---

## 3. Recommended architecture

### 3.1 Track B — WBA KAM-existence certificate (do first; replaces the broken metric)

**Algorithm (per seed field line):**
1. Reuse the existing midplane radial seed sweep and `compute_fieldlines` (already the SSOT tracer).
   Record Poincaré returns on a fixed φ-plane; need enough returns N (WBA needs *far fewer* than a
   Lyapunov estimate — hundreds–few thousand).
2. Locate the magnetic axis (use `src/simsopt/field/magnetic_axis_helpers.py`) to define a poloidal
   angle θ_n for each return about the axis; track the unwrapped angle.
3. Form the **weighted Birkhoff average** of the angle increment Δθ:
   `WB_N = Σ_{n=0}^{N-1} ŵ_{n,N} Δθ_n`, with bump weight `w(t) = exp(−1/(t(1−t)))` on (0,1),
   `ŵ_{n,N} = w(n/N)/Σ_k w(k/N)`. `WB_N` estimates the **rotation number ι̂**.
4. **Classification by convergence rate** (the actual KAM test): compare the WBA over the first vs
   second half of the orbit (or `|WB_{2N} − WB_N|`). Number of matching digits D:
   - D large (e.g. ≥ ~8–10) ⇒ **invariant torus** (intact KAM surface);
   - D moderate with ι̂ ≈ rational p/q to working precision ⇒ **island chain** (period q);
   - D small ⇒ **chaotic** (no invariant circle through this seed).
5. **A meaningful KAM fraction** = (# seeds classified as invariant tori) / (# seeds *that did not exit*).
   This is finally a real KAM measure: near 1 for a clean machine, dropping as islands/chaos appear,
   and — unlike `kam_fraction` — *independent of whether the surface is near-axis or near-edge*.

**Optional upgrade (Ruth–Bindel Birkhoff RRE, arXiv:2403.19003):** a single linear least-squares solve
finds optimal weights; a subsequent eigenvalue problem returns the **number of islands** and ι̂ with
fewer map iterations, plus a Fourier parameterization of the invariant circle/island (i.e. it
*reconstructs the KAM surface*). Explicitly demonstrated on magnetic field-line dynamics and shown
robust to non-symplectic integrator noise — important because simsopt's adaptive RK tracer is *not*
symplectic.

**Wiring:**
- New module `examples/single_stage_optimization/banana_opt/topology/kam_birkhoff.py`.
- In `topology_scorer.score_topology`, replace the `kam_fraction(...)` call with the WBA classifier;
  keep `survival_fraction` and the `topology_broken` (iteration-limit) logic *unchanged* (they are
  correct). Emit `invariant_torus_fraction`, per-seed `{class, rotation_number, digits}`.
- **Rename** the result fields (`FRONTIER_KAM_FRACTION` → e.g. `INVARIANT_TORUS_FRACTION`) so the
  certificate name matches the physics. Recompute the certification floor in
  `frontier_kam_calibration.py` against this *true* metric (the old floor auto-collapsed to ~0 because
  the proxy scored good configs near 0).

**Validation (Track B):**
- Standard map ground truth: golden-mean circle exists for `k < k_c ≈ 0.9716`, gone above — classifier
  must flip there (GPD 5.16 benchmark reproduction).
- Analytic field with a known island (e.g. a tokamak field + single resonant perturbation, or a
  Dommaschk potential): WBA must label the island region as period-q and the core as tori.
- Direct-vs-proxy consistency (GPD 5.17): WBA fraction must track visible Poincaré islands and the SPEC
  residue on known `banana_opt` artifacts — the exact check the old metric failed.

### 3.2 Track A — direct-BiotSavart Greene residue objective (the differentiable driver)

The existing `greene_residue_topology_gradient_impl_plan_2026-05-11.md` is **mathematically correct and
should be the basis**: full-torus return map (avoids the nfp trap), `R_G=(2−tr M)/4`, tangent-map
integration with `det(M)≈1` gate, branch (O/X) tracking, frozen→FD→adjoint derivative ladder, Taylor
tests, iota-gaming guard, near-success freeze at `R_G→0`, kill criteria. Confirmations and sharpenings
from the verified literature:

- **API precedent (arXiv:2106.14930, Landreman–Medasani–Zhu; SIMSOPT `example_islands`):**
  seed *both* fixed points of the chain — `Residue(spec, p, q)` (θ=0 seed) and
  `Residue(spec, p, q, theta=π)` — and least-squares-target both to 0 with equal weight. They reduced
  residues 2e-3 → 2e-6 and eliminated the ι=2/5 island *without* moving the resonance out of the domain.
  Reproduce as a Track-A benchmark (GPD 5.16).
- **O/X labels are not convention-portable — classify, don't hardcode.** The sources openly
  contradict each other: Landreman 2106.14930's inline comment calls θ=0 the **X**-point and θ=π the
  **O**-point, whereas the local docs `docs/source/example_islands.rst:77-79` call θ=0 the **O**-point
  and θ=π the **X**-point. (The `simsopt.mhd.spec.Residue` docstring and
  `examples/2_Intermediate/eliminate_magnetic_islands.py` *code* are silent — θ is only the Newton
  seed, and which point it lands on depends on configuration geometry/symmetry.) The `example_islands`
  printed residues settle it empirically and show why to classify rather than label: **+0.0233 at θ=0,
  −0.0229 at θ=π** — the **sign** decides. So O/X identity is set by the tangent map:
  0 < R_G < 1 ⇒ elliptic/O, R_G < 0 or R_G > 1 ⇒ hyperbolic/X. Implementation must seed both branches
  and **classify O/X from the residue sign**, never from a hardcoded θ→label map.
- **Resonance-continuity trick (same paper):** run a preliminary optimization *without* residues first,
  because the ι profile shifts resonances in/out of the domain early and makes the residue objective
  discontinuous. This is exactly the plan's "freeze target rationals from the target ι profile."
- **Vacuum clarification (important):** for the current/pressure-free `banana_opt` coil field you do
  **not** need SPEC — trace the BiotSavart field directly (the plan's "direct BiotSavart residue").
  Use `spec.py::Residue` only if/when a finite-β/current SPEC equilibrium enters the loop.
- **Gradient path:** `M` depends on `∇B` along the orbit, so `dR_G/dcoil` needs variations of **both B
  and ∇B** ⇒ `BiotSavart.B_and_dB_vjp(v_B, v_gradB)` (repo exposes it; `d2B_by_dXdX` available).
  Differentiate through the periodic-orbit solve by the implicit function theorem
  (`F_z = M − I`). `B_vjp` alone is insufficient. Mandatory random-direction Taylor test
  (`J(c+εv) − J(c) − ε∇J·v = O(ε²)`, log-log slope ≈ 2) before any optimizer use.
- **Objective term (default weight 0, opt-in):**
  `J_total = J_NQS + RES_WEIGHT·J_BoozerResidual + IOTAS_WEIGHT·J_iota + engineering + RESIDUE_WEIGHT·Σ ρ(R_G/R_scale)`.

### 3.3 Optional Track C — QFM residual (cheap differentiable surrogate, already in simsopt)

Because `simsopt.geo.QfmSurface` exists, a **quadratic-flux-minimizing residual** on a target surface
is an inexpensive, smooth "non-invariance" measure (Hudson–Dewar 0909.2096; Dewar–Hudson–Gibson
1001.0483: QFM and ghost surfaces agree to O(ε²)). Worth a spike as either a Track-A surrogate or a
Track-B quality number, since it needs no new periodic-orbit machinery.

---

## 4. Verification plan (GPD `fluid_plasma` checklist mapping)

| GPD check | Applied to KAM implementation |
|---|---|
| 5.1 dimensional | residue & rotation number dimensionless; island width [L]; WBA digits dimensionless |
| 5.2 numerical spot-check | tangent-map `M v` vs finite-difference return-map perturbation; WBA on standard map vs known ι |
| 5.3 limiting cases | integrable field ⇒ WBA fraction → 1, all residues → 0; strongly perturbed ⇒ chaos detected, residues diverge |
| 5.4 conservation | `det(M) ≈ 1` (area preservation) under refinement; do **not** silently renormalize |
| 5.5 convergence | `z*`, `M`, `R_G`, WBA ι̂ and gradient stable under integrator/N refinement |
| 5.16 benchmark | standard map `k_c ≈ 0.9716`; reproduce 2106.14930 residue 2e-3→2e-6 island elimination |
| 5.17 direct-vs-proxy | WBA certificate must agree with SPEC residue and visible Poincaré islands (the check `kam_fraction` failed) |
| Taylor test | gradient of `J_residue` slope ≈ 2; branch id/winding unchanged across ε window |

---

## 5. Risks & pitfalls

- **Convention traps** (radian vs normalized-turn ι; full-torus vs field-period map; bare `(m,n)`/`(p,q)`).
  Lock conventions first (plan Phase 0); most residue bugs are convention bugs.
- **Singular branch at success:** near `R_G→0`, `M−I` is ill-conditioned (the isolated periodic point
  approaches a rational-surface family). Freeze the branch's gradient once `|R_G| < R_satisfied`; let
  BoozerResidual/Iotas/Poincaré hold quality.
- **Low |B_φ|** gating: do not regularize the `1/B_φ` denominator in v0; report and gate.
- **Non-symplectic tracer:** WBA is robust to this (Ruth–Bindel); the residue tangent map is not — use
  tight tolerance and the `det(M)≈1` gate.
- **Lost lines:** an exiting seed cannot be WBA-classified; count it via survival, not as "chaos."
  Keep survival and KAM as *separate* axes.
- **Field-line ≠ particle confinement:** intact KAM tori are necessary but not sufficient for banana-
  orbit confinement; pair with `Gamma_c`/`epsilon_eff` (guiding-center) for the particle claim.
- **Island-width adjoint is available (confirmed):** `arXiv:2102.04497` (Geraldini–Landreman–Paul,
  *An adjoint method for determining the sensitivity of island size to magnetic field variations*, JPP)
  gives an adjoint **island-width** gradient (Cary–Hanson 1991 width + residue gradient), verified on
  the Reiman–Greenside (1986) analytic field and applied to an **NCSX vacuum** config with a **coil
  shape gradient**. Near-direct fit for `banana_opt` (vacuum coil field) and arguably a better Track-A
  target than bare residue — adopt it as the island-width route in Track A.

---

## 6. Phasing / next steps

- **B0a (classifier, ~2 modules):** WBA classifier module + standard-map unit test; swap into
  `topology_scorer.score_topology` behind a flag, emitting the new `invariant_torus_fraction`
  *alongside* the legacy field for back-compat. Low risk, immediate value.
- **B0b (schema rename + recalibration, multi-file):** rename `FRONTIER_KAM_*`/`kam_fraction` across the
  ~7 production files that carry it (`topology_scorer`, `frontier_kam_calibration`,
  `frontier_pareto_trajectory`, `single_stage_search_contracts` [≈56 refs], `frontier_dominance`,
  `frontier_contracts`, `single_stage_banana_example` [≈37 refs]) plus ~4 test files, then recalibrate
  the certification floor against the true metric. Larger, schema-touching — budget separately from B0a.
- **B1:** optional Birkhoff RRE (island count + Fourier parameterization); converse-KAM negative test on
  edge regions flagged by strict Poincaré.
- **A0–A7:** follow the Greene-residue plan: value-only direct-BiotSavart residue → conventions/winding/
  tangent-map/`det(M)` tests → periodic-orbit solver + branch tracking → value-only diagnostics on
  existing artifacts → directional FD sensitivities → production `B_and_dB_vjp` adjoint → optional
  low-weight objective behind a flag. Decision gates per the plan's Final Decision Gate.
- **C (spike):** QFM residual via `simsopt.geo.QfmSurface` as a cheap differentiable surrogate.

---

## 7. Literature anchors (via GPD arXiv tool, 2026-05-27)

Verification levels: **[full]** = read in full this session (2403.19003, 2106.14930, 2102.04497).
All other entries are **[meta]** = title/author/abstract returned by the GPD arXiv `search_papers`
tool this session (existence + topic confirmed; not deep-read). 2005.07633, 1801.04317 and 2510.01957
were all surfaced by this session's GPD arXiv searches (so they are [meta], not unconfirmed).

**KAM-existence certificate (Track B):**
- 2001.00086 — Sander & Meiss, *Birkhoff Averages and Rotational Invariant Circles for Area-Preserving
  Maps* (WBA classifier; Greene noble/golden conjecture without symmetry).
- 2106.15024 — Meiss & Sander, *Birkhoff Averages and the Breakdown of Invariant Tori in
  Volume-Preserving Maps* (sharp regular/chaos split via convergence rate; critical-parameter detection).
- 2403.19003 — Ruth & Bindel, *Finding Birkhoff Averages via Adaptive Filtering* (Birkhoff RRE; island
  count + rotation number + Fourier parameterization; **magnetic field-line** demo; robust to
  non-symplectic noise).
- 2010.12116 — Duignan & Meiss, *Nonexistence of Invariant Tori Transverse to Foliations* (converse KAM
  for 3-D volume-preserving / magnetic flows).
- 2205.09496 — Tong & Li, *Exponential convergence of weighted Birkhoff average* (theory underpinning
  the convergence-rate classifier).
- Foundational (not in arXiv search): Das, Saiki, Sander & Yorke, *Quantitative quasiperiodicity*,
  Nonlinearity 30 (2017) 4111 — origin of the WBA chaos/regular test.

**Differentiable island/QS objective (Track A) & almost-invariant surfaces (Track C):**
- **2102.04497 [full]** — Geraldini, Landreman & Paul, *An adjoint method for determining the
  sensitivity of island size to magnetic field variations* (JPP). Adjoint **island-width** gradient
  (Cary–Hanson width + residue gradient), verified on Reiman–Greenside (1986), applied to an **NCSX
  vacuum coil** shape gradient. **Primary Track-A island-width route for `banana_opt`.**
- 2106.14930 [full] — Landreman, Medasani & Zhu, *Stellarator optimization for good magnetic surfaces
  at the same time as quasisymmetry* (VMEC+SPEC, **island-residue penalty**; SIMSOPT `Residue` API).
- 2108.11433 [meta] — Nies, Paul, Hudson & Bhattacharjee, *Adjoint methods for quasisymmetry of vacuum
  fields on a surface* (single-surface adjoint QS+ι; relevant to vacuum-field gradients).
- 0909.2096 [meta] — Hudson & Dewar, *Are ghost surfaces quadratic-flux-minimizing?*
- 1001.0483 [meta] — Dewar, Hudson & Gibson, *Unified Theory of Ghost and Quadratic-Flux-Minimizing Surfaces*.
- 2005.07633 [meta] (Paul thesis) and 1801.04317 [meta] (Paul, Landreman, Bader, Dorland) — adjoint
  stellarator shape/coil optimization foundations.
- 2510.01957 [meta] — Martinez-del-Rio & MacKay — volume between flux surfaces (flux-coordinate utility).

**Classic references (via 2106.14930 bibliography):**
- Greene 1979, J. Math. Phys. 20, 1183 (residue criterion).
- Hanson & Cary 1984, Phys. Fluids 27, 767; Cary & Hanson 1986, Phys. Fluids 29, 2464 (stochasticity
  reduction/elimination); Cary & Hanson 1991, Phys. Fluids B 3, 1006 (island width).
- Mather 1986, Publ. IHÉS 63, 153 (ΔW non-existence criterion).
- Meiss 1992, Rev. Mod. Phys. 64, 795 (symplectic maps, transport — review).
- MacKay 2018, Reg. Chaotic Dyn. 23, 797 (converse KAM via foliations).
- Boozer 1981, Phys. Fluids 24, 1999; SPEC: Hudson et al. 2012, Phys. Plasmas 19, 112502.
