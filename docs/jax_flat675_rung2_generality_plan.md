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

### R1 findings (2026-08-20)

Recon complete; no production code or tests were changed. This subsection
amends the R1 work package above. Where a finding contradicts an assumption
in the reviewer-cleared text, it says so rather than editing that text.

**(a) Pin inventory — the review's measurement is confirmed exactly, and is
a floor rather than the total.** The seven named layout symbols
(`FLAT675_{OUTER,COIL,VESSEL,SURFACE}_DOF_COUNT`,
`FLAT675_{COIL,VESSEL,SURFACE}_SLICE`) occur **158 times across 13 files**:
src 69 (`formulation.py` 25, `__init__.py` 14, `objective.py` 12,
`boozer_material.py` 11, `construction.py` 7), benchmarks 20
(`flat675_promotion_robustness_child.py` 12, `flat675_fused_lane_child.py`
8), tests 67 (`test_flat675_contracts.py` 30,
`test_flat675_construction.py` 19, `test_single_stage_flat675_example.py`
8, `test_flat675_objective.py` 7, `test_single_stage_flat675.py` 3),
example 2 — matching the charter's per-file figures term for term.

Four further classes of pin carry layout meaning and are **invisible to a
named-token grep**, so requirement 1 is not discharged by chasing the 158:

1. **The 661 is not defined by `FLAT675_SURFACE_DOF_COUNT`; it is produced
   by the resolution triple.** `CERTIFIED_MPOL/NTOR/STELLSYM`
   (`construction.py:62-64`) become a `SurfaceXYZTensorFourier` at
   `construction.py:225-228` and a spec at `:251-253`; the DOF count
   constant only *cross-checks* the result downstream
   (`boozer_material.py:116-119`). The six `CERTIFIED_*` constants have 14
   production use sites and 11 test use sites. A layout record that
   generalizes the widths but leaves this triple pinned generalizes
   nothing.
2. **Bare numeric layout literals in executable test code — 16 sites, none
   of which the 158 counts.** `test_flat675_contracts.py`: `:922`
   `dof_count=660` and `:1314` `jnp.zeros(660)` (surface width − 1
   negative cases), `:1101` `_spec_array(..., [661])` (wire payload),
   `:770/:772/:973/:1123/:1127` current segments `((9,10,0,1),)` /
   `((10,11,0,1),)`, `:827/:828/:880/:884` owner maps and
   `frozenset(range(11))`, `:1280` `.at[10]`, `:1307` `jnp.zeros(10)`;
   `test_flat675_objective.py:274` `np.zeros(3)`;
   `test_flat675_construction.py:717` the regex
   `"does not carry owner DOFs 1-10 contiguously"`. The width − 1 cases
   and the `1-10` regex do not fail loudly on a width change — they become
   false passes or spurious failures.
3. **One string-encoded pin.** The archived certificate's gradient key is
   literally `"full_675"` (`test_flat675_objective.py:109`,
   `benchmarks/genuine_675_fair_bar.py:791`).
4. **The vessel's width-3 is also unnamed at its source.**
   `synthesize_flat675_vessel` builds `np.zeros((2, 1))` rc/zs blocks
   (`construction.py:366-372`); the 3 is a consequence of that shape, and
   `FLAT675_VESSEL_DOF_COUNT` is never referenced there. Harmless while the
   vessel stays width-3 (a settled scope decision), recorded because the
   constructor could otherwise emit a vessel its own material rejects.

**(b) Polymorphism audit — the core math is already generic; every pin is a
validator.**

| Unit | Verdict | Evidence |
| --- | --- | --- |
| `solve_flat675_y_qr` | **Polymorphic** in row count | `y_solve.py:40-45` constrains only the 2 columns (the inner state, out of scope); rows free |
| `build_flat675_boozer_system_arrays` | **Polymorphic** | `_boozer_arrays.py:34-41`: pointwise columns, `.reshape(-1)`, normalized by `magnetic_field.size` |
| `build_flat675_boozer_system` | **Polymorphic**, zero pins | `boozer_material.py:190-207` uses `.reshape((-1, 3))` and `.shape` only |
| `flat675_candidate_geometry` | Polymorphic body, **2 validator pins** | pins `boozer_material.py:163,167`; body `:176-186` is width-free |
| `Flat675BoozerMaterial.__post_init__` | **4 validator pins** | `boozer_material.py:51,116,143` |
| `flat675_weighted_terms` | **1 validator pin + 3 slices** | `objective.py:88` shape gate; `:93-95` block slices; everything after is generic |
| `Flat675Candidate.__post_init__` | **3 validator pins**, table-driven | `formulation.py:89-100` — already a per-block width table; the natural layout-record seat |
| `bind_flat675_programs` | **No pins** | `objective.py:187-214` closes over material only |
| `require_certified_coil_layout` | **Pinned by design** (rung-1 gate) | `construction.py:299,320,323` |
| `assemble_flat675_problem` | **1 validator pin** | `construction.py:432` |

Downstream of the package the shared single-stage math is already width-free:
`_split_x_inner_runtime` derives the surface width as
`x_inner.shape[0] - 2` (`surface_objectives.py:1051-1058`), so the
`661 + 2` concatenation at `objective.py:113-114` needs no change.

**(c) stellsym — verdict: INCLUDED for requirement 3, but the charter's
framing of the question is wrong and the amendment states so.**

The charter asks whether "the objective/Boozer math is genuinely
parameterization-agnostic with no symmetry assumption". It is — but that is
not what decides the question, because there *is* a real symmetry
assumption and it is not in the math.

*Evidence that the math is agnostic* (positive, not argument-from-absence —
these three kernels do not accept a `stellsym` or `nfp` argument at all):
- `boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB)` —
  `simsopt_jax/geo/boozer_residual.py:263-306`; pointwise residual
  normalized by `3 * nphi * ntheta` (`:295`).
- `surface_volume(gamma, normal)`, `surface_area`,
  `surface_mean_cross_sectional_area`, `surface_major_radius` —
  `simsopt_jax/core/surface_integrals.py:19,26,34,51`; each is a *mean over
  the supplied grid*, dividing by the actual `nphi*ntheta`.
- `non_quasi_symmetric_residual_primitives(xphi, xtheta, B, axis)` —
  `simsopt_jax/core/quasisymmetry.py:14-34`; area-weighted variance of
  `|B|` along one axis.

*Evidence that the `stellsym=False` parameterization is implemented, not
stubbed:* `stellsym` reaches the flat path only as a descriptor of which
Fourier coefficients exist, always paired with `scatter_indices`
(`objective.py:122,130` → `surface_objectives.py:1417,1447,1466` →
`boozer_residual.py:736-750`). At `boozer_residual.py:744-748` the
non-symmetric case substitutes an empty scatter array, and
`surface_fourier_kernels.py:1231-1244` takes the complementary branch
`_split_flat_to_xyzc` (`:1115-1132`), which unpacks
`3 x (2*mpol+1) x (2*ntor+1)` coefficients — 1323 at `mpol=ntor=10`,
matching the figure the F4 plan already quotes. The flat path never touches
the stellsym-masked BoozerExact machinery
(`surface_objectives.py:2061-2108`), which belongs to the nested
formulation.

*Where the symmetry assumption actually lives:* precisely because those
integrals are means over the supplied grid, a half-field-period grid stands
for the whole torus **only by symmetry**. That assumption enters at exactly
one line — `range=CERTIFIED_SURFACE_RANGE` (`construction.py:220`,
`CERTIFIED_SURFACE_RANGE = "half period"` at `:70`), a constructor input
with one production use site. **G2 can therefore include stellsym=False by
selecting the quadrature range from the boundary's symmetry instead of
hardcoding "half period"; no kernel changes.** The refusal at
`construction.py:271-275` is retired with the rest of
`require_certified_surface_layout`, and the F4-era refusal in
`fit_flat675_boundary` (`_RUNG_TWO_SURFACE_MESSAGE`) is retired with it.
Recommended G2 gate: a stellsym=False problem constructs and solves, and
its volume label is compared against the same boundary evaluated on a
full-period grid, so the range selection is proven rather than assumed.

**(d) Schema blast radius — 12 recorded strings mention 675; all 12 are
sealed, and none blocks the work.**
- **Sealed by file mode (0440/0550), and therefore pinned *in live code*:**
  the frozen bundle's six schema strings —
  `simsopt.single_stage.fixed_state_genuine_675.{frozen_input_manifest.v1,
  objective_policy.v2, material_identity.v1, source_identity.v1}`,
  `simsopt.single_stage.experimental_fullspace_675.{boozer_system_policy.v1,
  candidate.v1}`. The loaders that must keep accepting them byte-for-byte
  are `flat675/manifest.py:42,44,77,249,311` and
  `flat675/runtime_spec_loader.py:32,160`. These are the only 675-bearing
  strings inside `src/`, and they are data contracts, not layout pins.
- **Sealed by tracked evidence:** `genuine-675-fair-bar-manifest.v1`
  (8 tracked receipt files), `flat675-fused-campaign-manifest.v1` (7),
  `flat675-promotion-robustness.v1` (2).
- **Sealed by host-local campaign evidence:** `flat675-fused-lane.v1`
  (named an invariant by this charter), `flat675-promotion-robustness-child.v1`,
  `genuine-675-fair-bar-{input,row,loader-selftest}.v1`,
  `genuine-675-fair-bar-oracle.v1`.
- **Live and safe to extend:** none of the above needs a version bump for
  rung 2, because no schema encodes a width — they carry named coordinate
  blocks whose lengths are implicit in the payload.

**(e) Rung-2 pointer sites — the charter's four are accurate but the
enumeration is incomplete; the sweep finds 7 in `construction.py` and 14
more outside it.** The four cited line numbers verify exactly (`:78`,
`:89`, `:268`, `:16-20`). Additional sites:
- `construction.py:61` — inline comment "the pair is forced at rung 1".
- `construction.py:266` — `"rung 1 forces mpol=ntor="` (the rung-1 half of
  the message whose rung-2 half the charter cites at `:268`).
- `construction.py:272-274` — `"rung 1 forces stellsym=True: ... carries
  1323 DOFs"`, a message separate from `:266-268` inside the same validator.
- `construction.py:60`, `:192-206` (`fit_flat675_boundary`'s two-loss-mode
  docstring), `:279` (section comment), `:478` (`build_flat675_problem`'s
  pointer to those loss modes) — restriction prose that survives the
  validators.
- `tests/jax/objectives/test_flat675_construction.py:14, 611, 619, 623,
  634, 636, 650` — seven sites that *assert* the rung-1 message text.
- `docs/jax_flat675_promotion_plan.md:21, 27, 29, 89, 92, 94, 95` — the F4
  plan's own rung-1/rung-2 statements.

**Charter contradictions and gaps found (flagged, not fixed):**

1. **Acceptance gates 5 and 6 collide on the example.** Gate 6 freezes
   `examples/jax/3_Advanced/single_stage_flat675.py` byte-for-byte "full
   stop"; gate 5 forbids leaving any "module/function docstring ...
   asserting a rung-1 restriction G2 lifted". That example's docstring
   (`:29-38`) is headed "THE LAYOUT IS FIXED AT 11 + 3 + 661" and states
   that `build_flat675_problem` "fits any compatible simsopt boundary onto
   that layout" — a claim about the *constructor* that G2 falsifies, even
   though it stays true of this example's own configuration. The two gates
   cannot both be satisfied as written. **Needs adjudication before G2**;
   R1 recommends the narrow reading (gate 6 wins, the sentence is about the
   example's configuration) with the ambiguity resolved by a one-word
   change under an explicit documented exception if the reviewer disagrees.
2. **Gate 5 does not cover the tests that assert the retired messages.**
   Seven assertions in `test_flat675_construction.py` pin rung-1 message
   text; they must change in the same commit that retires the messages or
   CI fails. Gate 5's vocabulary ("refusal message, inline error string,
   module/function docstring") should be read to include them.
3. **Requirement 3's decision procedure is mis-framed** (see (c)): it
   conditions stellsym=False on the math being symmetry-agnostic, which is
   true but not sufficient. The operative condition is the quadrature-range
   selection at `construction.py:220`. The verdict is INCLUDED on that
   basis, not on the basis the charter names.
4. **R1 process note.** The first bare-literal sweep run for (a) reported a
   false clean: the filter excluded any line whose text matched `flat675`,
   and `grep -n` prefixes each line with a path that itself contains
   `flat675`, so every hit inside the flat675-named files was discarded.
   The 16 literal sites in (a)(2) were recovered by a second, independent
   sweep and each was then read at its line. Recorded because the same
   filter shape would silently under-report any future inventory.

### Adjudication of R1's flags (2026-08-20, orchestrator; delta-reviewed)

1. **Gates 5/6 collision — resolved with NO example edit, secured by a G2
   API constraint.** In the generalized constructor, surface-layout
   selection (`mpol`, `ntor`, `stellsym`) is EXPLICIT with certified
   defaults; the constructor never derives the target layout from the
   boundary it is given. Under that API every sentence of the example
   docstring stays true after G2: the example's own layout IS fixed at
   11 + 3 + 661, and at the certified defaults the constructor still fits
   any compatible boundary onto that layout and still REFUSES a boundary
   that layout cannot represent — an asymmetric boundary targeted at a
   stellsym=True layout is a projection onto a proper closed subspace and
   stays refused per the F4 coercion line. What G2 changes is the
   refusal's message (from "rung 2 will lift this" to "pass
   stellsym=False"), not the refusal behavior at the certified target.
   This supersedes one clause of finding (c): `_RUNG_TWO_SURFACE_MESSAGE`'s
   refusal in `fit_flat675_boundary` is RE-POINTED, not retired — only the
   `require_certified_surface_layout` validator (the certified-triple pin)
   is retired outright. Gate 6 therefore holds as written (example
   byte-unchanged, full stop) and gate 5 holds because the docstring
   asserts no lifted restriction under the explicit-layout API. G2 adds
   the pinning test: asymmetric boundary + stellsym=True target → typed
   refusal naming the stellsym=False alternative.
2. **Gate 5 vocabulary — clarified to include test assertions.** The seven
   message-text assertions in `test_flat675_construction.py` (:14, :611,
   :619, :623, :634, :636, :650) are in gate 5's scope: any commit that
   retires or re-points a message updates the tests asserting its text in
   the same commit.
3. **Requirement 3 — stellsym=False INCLUDED, on R1's quadrature-range
   basis.** The charter's literal condition ("no symmetry assumption") was
   a proxy for "no kernel surgery needed"; R1 shows the kernels are
   symmetry-free and the single assumption is the constructor's
   quadrature-range input (`construction.py:220`). G2 includes
   stellsym=False by selecting the range from the REQUESTED layout's
   symmetry (stellsym=True → "half period", stellsym=False → full torus),
   and adopts R1's recommended gate as a requirement-3 acceptance
   addition: a stellsym=False problem constructs and solves, and its
   volume label matches the same boundary evaluated on a full-period
   grid — the range selection is proven, not assumed. (For an
   `nfp`-periodic asymmetric boundary, "field period" tiles the torus at
   1/nfp the phi cost and is equally correct; G2 may select either — the
   volume gate proves the choice.)
4. **Process note accepted**; the grep-filter false-clean trap is recorded
   here and in the session learnings.
5. **R1 finding (a)(1) — where the resolution triple lives** (added on
   delta review, which correctly flagged this as unresolved). The layout
   record is constructed FROM `(mpol, ntor, stellsym)` and CARRIES the
   triple as members alongside the surface-block width it produces; the
   width is derived inside the record's construction and is never
   supplied independently, so triple and width are one source and
   requirement 1's "no dual source" clause is satisfied explicitly, not
   by inference. The constructor's explicit layout parameters
   (adjudication 1) are the record-construction inputs, not a parallel
   channel — everything downstream of construction reads the record. The
   certified instance pins the triple `(10, 10, True)` and therefore the
   661.

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

## G3 closure (2026-08-20)

Appended at close, after G1/G2 landed and were reviewed. Nothing above is
rewritten; the plan's own text and the R1 amendment stand as written.

### Delivery

| Commit | What it did |
| --- | --- |
| `c5123bd09` | G1. Introduced `geo/flat675/layout.py` and threaded the record through the core (candidate, Boozer material, objective). The `FLAT675_*` constants became the certified instance read out under their existing names, so every consumer was untouched. |
| `1755e2265` | G1 review fix. Pinned the surface template to the resolution its layout declares — width alone does not name a resolution. |
| `8fe5d70ea` | G2. Generic coil sets, explicit `mpol`/`ntor`/`stellsym` with certified defaults, `stellsym=False` included with the quadrature range selected from the requested symmetry, the coil refusal retired and the surface refusal re-pointed. |
| `21930ce22` | G2 bookkeeping. Dated additive line beside the F4 plan's generality-ladder bullet recording rung 2 as delivered. |
| `6ad7bb349` | G2 review fix. Derived the certified triple from the record instead of re-declaring it as literals. |
| this commit | G3. This closure section. Docs-only; no code, no manifest cascade. |

### Acceptance gates

1. **Certified-layout bitwise gate — PASS.** The five-value fingerprint,
   recomputed on each work package's final tree against a baseline captured
   at `2661ab1df` in a detached worktree (never a modified tree; that tree's
   code is byte-identical to F4-final, `git diff 5ef18d852..2661ab1df` over
   `src/ benchmarks/ examples/ tests/` being empty):
   objective hex `0x1.5223a865c5038p+4`; gradient sha256
   `29f834ca6f794481135a04a24e33b9723770b4f2addc0d0236d6187ee05b977a`;
   weighted-terms sha256
   `ca76c5e8430039aa2511966c483ce618a3962d87ae20c38181161c4972b88534`;
   surface-template dofs sha256
   `74b00942edd6e7678002725bafc375c8c93cdfe5f9d04b0ef371ba45e446343b`;
   GATE-3 fused endpoint sha256
   `91ca47a99fa0e4f6d30aabe76aa9804bdb7464019893ec27a0ade7d0dfe4996d`.
   Bit-identical every time, with the full record — counters and transfer
   ledger included — identical too. Baseline and every post-change capture
   came from one unmodified script whose mtime predates all of them, so the
   construction is identical by construction rather than by assertion.
   Archive control unchanged throughout at `1.176746e-15` objective and
   `2.081422e-15` gradient.
2. **Generic coil set constructs and solves — PASS.**
   `tests/jax/objectives/test_flat675_generality.py::test_generic_coil_set_constructs_and_solves`:
   `create_equally_spaced_curves` plus a `Current` each — the configuration
   rung 1 refused — on a small layout, finite endpoint, objective decreased,
   endpoint gradient below start, zero `advance`/`callback` in the ledger.
3. **Non-certified resolution constructs and solves — PASS.** Same file,
   `test_non_certified_resolution_constructs_and_solves` at `(4, 4)` and
   `(5, 2)`, plus `test_asymmetric_problem_constructs_and_solves` for the
   `stellsym=False` layout. Requirement 3's acceptance addition — that the
   selected quadrature range is proven rather than assumed — is
   `test_asymmetric_range_selection_is_proven_by_the_volume_label`: the same
   asymmetric boundary fitted on the selected `field period` range and on an
   independently built full-torus grid gives volume labels agreeing to
   **5.3e-16** (the certified symmetric convention sits at 1.2e-15 by the
   same measurement, so the asymmetric selection is in the same class rather
   than a new assumption).
4. **Gradient liveness at active anchors — PASS.**
   `test_new_blocks_carry_gradient` across three generalized configurations
   (generic coils, non-certified resolution, asymmetric layout) asserts every
   coil coordinate nonzero and the surface block live, read at the
   constructor's own start where the residual, iota and non-QS terms are all
   active. The F2 lesson is applied by pairing: the vessel block is exactly
   zero when its hinge is inactive
   (`test_vessel_block_is_exactly_zero_when_its_term_is_inactive`) and fully
   nonzero once the threshold is raised on the same problem and the same
   start (`test_vessel_block_is_live_once_its_term_switches_on`), so a zero
   means an inactive term rather than a disconnected block.
5. **stellsym decision executed, no stale pointer left — PASS.**
   `stellsym=False` is INCLUDED on R1's quadrature-range basis.
   `_RUNG_TWO_COIL_MESSAGE` retired with its validator;
   `require_certified_surface_layout` retired outright;
   `_RUNG_TWO_SURFACE_MESSAGE` re-pointed, not retired — the refusal survives
   because an asymmetric boundary aimed at a symmetric target is still a
   projection onto a proper subspace — and renamed to
   `_ASYMMETRIC_BOUNDARY_REFUSAL`, since a name reading `RUNG_TWO` would
   itself point at a charter that no longer owns the change. **Twelve**
   message-text assertions named retired or re-pointed text, and all twelve
   moved in `8fe5d70ea`, the same commit as the messages: three on the
   re-pointed surface refusal (its two in-test assertions plus the
   front-door `match=`), two on the retired `require_certified_surface_layout`
   (`rung 1 forces mpol=ntor`, `rung 1 forces stellsym=True`), and seven on
   the retired coil validator and its message (`Rung 1 accepts only the
   certified coil owner layout`, its charter-path assertion, `claims owner
   DOFs`, `has curve type OrientedCurveXYZFourierSpec`, `declares no free
   winding-surface coil`, `does not carry owner DOFs 1-10 contiguously`,
   `does not share the single free current`). One now asserts the surface
   message contains no "rung" at all. No message-text assertion moved in any
   other commit of this delivery, which is the property gate 5 actually
   needs. *(The closing review counted seven; that undercounted the retired
   coil validator's group, where two of the seven were tallied. Recorded
   because the count is the evidence, and R1's finding (e) in this same file
   enumerates the pointer sites it derives from.)*
   Content sweep at close: zero "rung" references remain anywhere in
   `src/simsopt_jax_adapters/geo/flat675/`, and no stale restriction prose.
   **Toolchain, named as run rather than as pinned.** Two ruff binaries
   were used and they are different versions. Iterative `check`/`format`
   during development ran the interpreter-local
   `.venv-qn-gpu/bin/python -m ruff`, which is **ruff 0.16.1**. Every
   branch-wide check quoted in this delivery ran the charter's pin explicitly
   as `uvx ruff@0.15.22 check` / `uvx ruff@0.15.22 format --check`, verified
   to report **ruff 0.15.22**. Both were clean on every commit; the record
   names both because the installed toolchain is not the pinned one and a
   reader reproducing this should know which produced which result. pyright
   ran at the repository pin, 0 errors 0 warnings throughout.
6. **Example and scoreboard byte-unchanged — PASS, by diff rather than by
   assertion.** `git diff 2661ab1df..HEAD` over
   `examples/`, `docs/jax_example_device_assignment.md`,
   `benchmarks/flat675_fused_lane_child.py` and
   `benchmarks/flat675_promotion_robustness_child.py` is empty. The
   execution-source manifest ends at 638 — one module landed (`layout.py`, at
   G1) and nothing since; G2 and G3 were digest refreshes and a docs edit.
7. **Closing crucible pass (2026-08-20) — PASS on the work, with two
   closure-record corrections.** The work was verified across the full
   `2661ab1df..HEAD` range: the five-value gate independently re-derived a
   third time at `3b422579f`, bit-identical, certifying the `6ad7bb349` code
   state; requirement 2 confirmed BY EXECUTION rather than by reading the
   tests — a `create_equally_spaced_curves` + `Current` set building a
   48-coil-DOF / 121-surface / 172-outer problem at triple `(4, 4, True)`;
   whole-range gate-6 empty diff; both charters append-only against their
   reviewed predecessors; no assertion loosened anywhere across the arc.
   The verdict was FAIL_ITERATE on two clauses of this closure record — the
   gate-5 assertion count and the GPU-witness "unmodified" qualifier — and
   both are corrected above in the commit that carries this line, under the
   reviewer's pre-authorization. Correcting the count raised it further than
   the review asked, from three to twelve. **F5 CLOSED.**

### Requirement 5: the "fast whole-lane tests" expectation is measured-false

Requirement 5 anticipated that small layouts would make cheap real solves
possible and that these would become "the first fast whole-lane tests in CI".
Half of that holds and half does not, and the half that does not is recorded
here rather than quietly accepted.

The tests are real: `test_flat675_generality.py` runs full fused solves on
layouts that are not 675, and it passes. It is not fast. It costs **624 s**,
and the cost is compile, not arithmetic. Five distinct problem shapes each
pay their own XLA compile, and each of those five maps one-to-one onto a
distinct proof obligation — generic coils, two non-certified resolutions, the
asymmetric layout, and the liveness anchors — so the cost is intrinsic to the
coverage rather than an artifact of how the file is written. This matches
what F4 measured on the shipped example (~150 s, unmoved by a 4x problem
shrink and only 12% by the frozen L-BFGS history): compile dominance in this
formulation is a property of the objective graph, which the 661-DOF surface
block fixes, and it does not fall away when the layout gets smaller.

No trimming was done and none is chartered here. Reducing the wall time means
reducing the number of distinct shapes, which means reducing what the suite
proves, and that trade is not one this work package should make on its own.
The mechanism that would make these tests fast without giving up coverage is
the persistent XLA compile cache already chartered as follow-up in
`docs/jax_flat675_promotion_plan.md`; **this suite is now that lever's second
named consumer**, after the shipped example.

### GPU witness (optional, run)

The charter makes this optional and forbids minting a harness for it, so it
was done with the existing test file and nothing else: the same
`tests/jax/objectives/test_flat675_generality.py`, run once under
`JAX_PLATFORMS=cuda` on the RTX 5090 after a bounded sustained-quiet check on
the device. The file was unmodified **for** the GPU run — nothing was changed
to make it run there — but it is not byte-identical to its G2 form: its only
change since `8fe5d70ea` is the semantically inert import hoist in
`6ad7bb349`, so the witness ran against the `6ad7bb349` tree. **11/11
passed.** The off-675 zero-transfer ledger is therefore
witnessed on the device and not only on CPU: the same assertions that hold
`advance == 0` and `callback == 0` for a generic coil set, a non-certified
resolution and an asymmetric layout passed with the solves executing on GPU.

Seconds are incidental and non-verdict, recorded only so the run is
identifiable: 735.65 s wall, against 624 s for the same file on CPU. Nothing
is claimed from that comparison — the two ran under different contention on a
box whose desktop compositor shares the device, and F5 makes no timing claims
at all.

### The dual-source class, caught three times

Requirement 1's "no dual source" clause turned out to be the load-bearing one,
and the same failure shape recurred at every layer. Recorded because a future
editor adding a fourth path should expect it rather than rediscover it:

- **Record versus template.** The material validated the surface block's
  WIDTH against the layout but never tied the layout's `(mpol, ntor,
  stellsym)` to the template's own. Width does not name a resolution — 202 of
  244 widths under mpol 1..15 have several producers, and the certified 661
  has three of its own at 1..25 — so a template could match the declared size
  while parameterizing a different surface. Fixed in `1755e2265`.
- **Owner width as a default parameter.** `_map_owner_indices` kept the
  certified coil width as a DEFAULT, which would have refused any wider coil
  set at G2 through an upper bound no caller passed, with an error message
  naming the wrong reason. Fixed in `c5123bd09` (found by re-reading the diff,
  not by a failing test).
- **Constructor defaults as literals.** `CERTIFIED_MPOL`/`NTOR`/`STELLSYM`
  were re-declared as literals rather than read from the record, so drifting
  one would have handed the DEFAULT path a non-certified problem while the
  record and every `FLAT675_*` constant still described 661 and 675 — and the
  first fix above would not have caught it, because under drift the layout and
  the template derive from the same wrong parameters and agree. Fixed in
  `6ad7bb349`.

The durable lesson is that an equality between two stored values is not a
single source, however carefully it is checked: each fix replaced a pair with
a derivation, which is why the surface width is a property rather than a
field and the certified triple is a read rather than a literal.
