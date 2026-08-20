# Flat single-stage rung-2 generality plan (F5)

Status: r2 reviewed PASS 2026-08-20 (r1 review: 6 findings F1–F6 + 1
recommendation, all addressed; r2 delta review confirmed all closed)
(engineering plan, reviewer-gated; no timing claim is made
or re-made; the sealed receipts and the certified 675 configuration are
invariants, not subjects).

This executes the follow-up F4 chartered: lift the rung-1 pins so the flat
single-stage constructor accepts **generic coil sets** and **arbitrary
surface resolutions**, while the certified 675 configuration remains
bit-identical everywhere it is consumed (`--bundle` example mode, campaign
child, tests, receipts).

## Requirements

Functional:
1. **Layout record.** The hardcoded 675/11/3/661 layout (module-level
   slice constants, `FLAT675_OUTER_DOF_COUNT`, per-validator pins) becomes
   a per-problem layout record (coil-block width, vessel-block width = 3,
   surface-block width) constructed once and threaded explicitly — no
   global state, no dual source. The certified 675 layout becomes THE
   distinguished instance of the record, exported under its existing
   names so every current consumer (example, campaign child, tests,
   `--bundle` mode) is untouched in behavior.
2. **Generic coil sets.** The constructor accepts any coil DOF extraction
   whose owner map covers `0..N_coil-1` contiguously — arbitrary owner
   count, arbitrary per-segment widths, any curve family the field
   machinery supports (including plain `CurveXYZFourier` + `Current`
   sets, the case rung 1 refused). The rung-1 refusal for non-certified
   coil layouts is lifted; its message is retired.
3. **Arbitrary surface resolutions.** The boundary fit accepts any
   `mpol >= 1, ntor >= 0` (surface-block width follows), `nfp` free,
   `stellsym=True`. **stellsym=False is decided by R1 recon** (below):
   included in G2 only if the recon shows the objective/Boozer math is
   genuinely parameterization-agnostic with no symmetry assumption;
   otherwise the asymmetric-boundary refusal stays and its message is
   re-pointed from "rung 2" to a named follow-up, in the same commit that
   would otherwise have lifted it.
4. **Gradient liveness for new blocks** (the F2 lesson: liveness gates
   need active anchors). For at least one generic coil configuration and
   one non-certified resolution: prove the new blocks carry gradient by
   evaluating at an anchor where the relevant terms are active, with
   exact-zero controls where a term is legitimately inactive.
5. **Small-layout end-to-end tests.** Generality makes cheap real solves
   possible (e.g. `mpol=ntor=4` → 121-DOF surface block; method
   cross-checked against `mpol=ntor=10` → 661): add artifact-
   free CPU tests that run a full fused solve on a small generic problem
   (finite endpoint, monotone improvement, endpoint gradient-inf strictly
   below start). These become the first fast whole-lane tests in CI.

Non-functional: production grade per repo law (SSOT/DRY/SOLID, typed, no
function-local imports in production code, no defensive guards); fp64
only; manifest cascade (digest refresh for edited members; `--admit` +
count twins only when a new module lands; expect-count currently 637) +
dual ruff (pinned 0.15.22 for any branch-wide check) + pyright; one test
file per pytest process.

## Scope decisions (settled — do not relitigate)

- **No package rename.** `flat675` stays — and the decisive reason is
  stronger than churn-avoidance: the sealed receipts, the
  `flat675-fused-lane.v1` schema, the run directories, and the scoreboard
  row all spell `flat675` and CANNOT be renamed, so a code rename would
  mint a permanent code-vs-evidence vocabulary mismatch. The layout
  record and any NEW public symbols use layout-neutral names; the package
  docstring AND the user-facing function docstrings
  (`build_flat675_problem`, `fit_flat675_boundary`,
  `assemble_flat675_problem`) each carry the one-line disclaimer that
  "675" names the certified configuration, not a constraint. Renaming would churn the manifest, tests, and
  every receipt citation for zero behavioral gain.
- **Sealed artifacts are invariants.** The F3 campaign machinery,
  `flat675-fused-lane.v1` schema, run directories, and receipts are not
  edited. `benchmarks/flat675_fused_lane_child.py` AND
  `benchmarks/flat675_promotion_robustness_child.py` keep consuming the
  certified instance (their sealed evidence is F3/C3 history); either may
  reference the distinguished layout record — the robustness child SHOULD,
  so future robustness runs can perturb non-certified layouts — but their
  behavior on the certified problem must be bit-identical.
- **The example does not change.** Rung-2 generality is API + test
  surface; the shipped example keeps the certified configuration and its
  scoreboard row keeps its scope. (A generic-layout demo lives in tests,
  not in a second example.)
- **No new timing claims.** Small-layout solves in tests and evidence are
  correctness runs; any seconds are incidental and labeled non-verdict.
- **Vessel block stays width-3** at rung 2 (the vessel template is a
  3-free-DOF record by construction; generalizing vessels is out of
  scope and not currently demanded by anything).

## Work packages (one writer; sequential scoped commits)

- **R1 — recon (report before code).** (a) Exact pin inventory: every
  reference to the 675/11/3/661 constants and slices across `src`,
  `benchmarks`, `examples`, AND `tests` (F4's ~30/5 count was partial;
  review measurement at charter time: ~158 refs / 13 files — src 69 incl.
  `flat675/__init__.py` 14, the re-export linchpin of requirement 1;
  benchmarks 20 incl. `flat675_promotion_robustness_child.py` 12, which
  uses the slices generically and is the natural first layout-record
  consumer; tests 67; example 2 — R1 enumerates authoritatively);
  (b) per-function polymorphism audit of the core (y-solve, objective
  terms, boozer material/system, candidate geometry, fused binding):
  which are already shape-polymorphic, which consume pinned constants;
  (c) the stellsym question: identify any symmetry assumption in the
  objective/Boozer math (residual assembly, QS terms, label solve) —
  answer INCLUDED/EXCLUDED for requirement 3 with evidence; (d) schema
  blast radius: which recorded strings/schemas mention 675 and which of
  those are sealed (untouchable) vs live. R1's findings amend this plan
  (dated, before G1 code).
- **G1 — layout record + core generalization.** Introduce the record;
  thread it through the core; certified instance exported under existing
  names. Gate: **bitwise self-consistency at the certified layout** — the
  full F4 fingerprint set: objective hex, gradient sha256, weighted-terms
  sha256, **surface-template dofs sha256** (the value most sensitive to a
  layout-record refactor), and the GATE-3 fused endpoint, bit-for-bit.
  The capture is taken AT THE PRE-F5 COMMIT (independent worktree or
  pre-change checkout, never a modified tree) and its five values are
  recorded in the commit message so a reviewer can re-derive them. Plus
  the 1.2e-15 archive control unchanged, plus one small-layout CPU solve
  test proving the generalized core actually runs off-675.
- **G2 — constructor generalization.** Generic coil owner maps; arbitrary
  resolutions; the stellsym outcome per R1; refusal messages updated
  (lifted ones retired; surviving ones re-pointed); gradient-liveness
  tests per requirement 4; F4-plan bookkeeping as a DATED, ADDITIVE edit —
  rung 2 is registered in the F4 plan's generality-ladder bullet
  ("reported not attempted"), NOT in its Chartered-follow-up section:
  append a dated "delivered by F5 (<commit>)" line beside that bullet (or
  a new dated Chartered-follow-up entry), never rewriting cleared text;
  the compile-cache lever entry stays open.
- **G3 — evidence + closure.** Small evidence note (tests are the record
  where possible; if any GPU witness runs are added — one generic-layout
  fused solve on the 5090 confirming the zero-transfer ledger off-675 —
  they follow the F4-C3 pattern: tracked summary, host-local raw,
  non-verdict seconds). Closing crucible pass over the delivery.

## Acceptance gates

1. Certified-layout bitwise gate at final HEAD (the five-value F4
   fingerprint set incl. surface-template dofs sha256, vs the recorded
   pre-F5-commit capture) + archive control.
2. A generic coil set (plain curves + currents) constructs, solves on a
   small layout, and passes finite/monotone/gradient-decrease gates,
   artifact-free in CI.
3. A non-certified resolution constructs and solves likewise.
4. Gradient liveness proven for new blocks at active anchors (with exact-
   zero inactive controls).
5. stellsym decision executed per R1 with evidence either way; no refusal
   message, inline error string, or module/function docstring left
   asserting a rung-1 restriction G2 lifted or pointing at a charter that
   no longer owns the change. Known pointer sites at charter time (R1
   re-enumerates): `_RUNG_TWO_SURFACE_MESSAGE` (construction.py:78),
   `_RUNG_TWO_COIL_MESSAGE` (:89), the inline rung-2 string inside
   `require_certified_surface_layout` (:268 — retired with its
   validator), and the module docstring's three-restriction sentence
   (:16-20).
6. Scoreboard/manifest/execution-source/ruff(0.15.22)/pyright green; the
   example and its scoreboard row byte-unchanged, full stop (the G2
   bookkeeping edit touches only the F4 plan document).
7. Closing crucible strict PASS.

## Non-goals

No package rename. No vessel generalization. No stellsym=False commitment
ahead of R1's evidence. No changes to sealed campaign machinery, run
directories, or receipts. No timing claims. No second example.
