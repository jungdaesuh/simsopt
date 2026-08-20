# Flat-675 promotion plan (F4) — from certified instrument to official example

Status: r3 — reviewer-cleared for execution (r1: 7 findings; r2 delta: 1 stale
cross-reference; all closed). Engineering plan, reviewer-gated; not a measurement
charter — no timing claim is made or re-made here. The speed and physics
claims live in the sealed receipts and are cited, never re-derived).

Operator decision (2026-08-19): the flat coupled formulation is blessed as
the official production VMEC-free single-stage problem statement. This plan
converts the certified campaign instrument into a user-facing workflow.

## Requirements

Functional:
1. **Constructor (same-input capability).** A public API that builds a
   flat-675 problem from simsopt objects, honoring the certified layout's
   real contracts:
   - **Boundary:** any `SurfaceRZFourier`/compatible boundary, fitted to
     the certified surface layout via simsopt's own `least_squares_fit`
     onto `SurfaceXYZTensorFourier(mpol=10, ntor=10, stellsym=True)` —
     exactly 661 DOFs. Rung-1 constraint stated to users: `stellsym=True`
     at `mpol=ntor=10` is forced (stellsym=False yields 1323 DOFs and the
     validators refuse it); `nfp` is free.
   - **Coils:** the certified owner layout — one free current (owner DOF
     0) plus a curve-on-winding-surface family (`CurveCWSFourierRZ`)
     carrying owner DOFs 1-10, with fixed `CurveXYZFourier` TF coils
     (empty owner maps). Rung 1 does NOT accept generic coil sets
     (`create_equally_spaced_curves` + `Current` cannot satisfy the
     11-owner validator); generalizing the coil validator is a rung-2
     chartered change, reported not attempted.
   - **Vessel:** an optional vessel surface; when omitted, the
     constructor synthesizes a default 3-free-DOF vessel template placed
     far enough that the hinge penalty (and its gradient) is exactly
     inactive at the start — the 11+3+661 layout is always satisfied, and
     "no vessel" means "inactive vessel term", never a 672-DOF vector.
   - **Policy:** documented defaults = the campaign's frozen weights and
     L-BFGS-B policy (`maxcor=300, maxls=8`).
   The frozen-bundle loader becomes one caller of this general path (SSOT:
   the bundle path must not keep a private construction route).
2. **Example script.** `examples/jax/` gains a VMEC-free single-stage
   flat example that is a thin driver over the production module,
   self-contained from a clone: it builds its problem from repo test-file
   geometry (the existing `tests/test_files` surfaces/coils the other
   examples already use), NOT from the host-local frozen bundle.
3. **Registration.** `examples/jax/manifest.json` entry (all 13 required
   keys; **`host_boundaries` must be `[]`** — the fused lane's GATE-3
   transfer discipline is the checkable substance of that claim);
   `docs/jax_example_device_assignment.md` assignment row appended per
   that record's amendment procedure (dated changelog row + table row,
   same commit); drift-gate suite green. **Row/receipt scope law:** the
   example ships a selectable `--bundle` mode that runs the certified
   frozen-bundle configuration when the host-local bundle is present
   (repo geometry is the clone-runnable default), and the row's evidence
   text scopes the citation explicitly: the F3 receipt certifies the
   frozen-bundle configuration of this formulation at one archived start;
   the shipped default runs the same production lane on repo geometry
   with no timing claim of its own. The row never extends the sealed
   receipt past its own scope sentence.
4. **Multi-start robustness evidence.** The fused lane dropped the host
   rejection protocol; all sealed evidence is one start candidate. Produce
   a small evidence record: solves from (a) the archived start (control),
   (b) seeded perturbations of it at 1e-3/1e-2/1e-1 relative amplitude
   (surface-block, coil-block, and full-vector variants), (c) at least one
   constructor-built generic start (repo-geometry example problem). Gates
   per run: finite endpoint, objective decreased, native-oracle
   cross-evaluation where the bundle problem is used (oracle is
   bundle-scoped; constructor-built problems gate on finiteness +
   monotone improvement + endpoint gradient-inf strictly below the start
   gradient-inf, stated as such). **The licensed claim is pre-committed
   and perturbation-scoped:** "robust to perturbations of the certified
   start at relative amplitudes <= 1e-1 (surface-block, coil-block,
   full-vector) and to one constructor-built start" — never "robust from
   arbitrary starts", regardless of outcomes.
   **No device-side rejection guard is added preemptively** (repo law: no
   defensive code); a guard is designed only if a failure appears, as its
   own reviewed change.

Non-functional: production grade (SSOT/DRY/SOLID, typed, no function-local
imports in production code, no defensive fallbacks); fp64 only; artifact-
free tests for everything constructor-path (CI must exercise the new
surface without the bundle); manifest cascade + count twins + dual ruff +
pyright per repo law; one test file per pytest process.

## Scope decisions (settled — do not relitigate)

- **Generality fallback ladder covers BOTH surfaces and coils.** The
  shape-polymorphic core
  (y-solve, objective math) generalizes freely; validators pinned to
  675/11/3/661 may stay pinned in step one. Rung 1 (required): constructor
  accepts user geometry at the campaign resolution (resample/expand user
  surfaces to the 661-DOF layout where simsopt's own APIs make that
  natural). Rung 2 (only if it does not explode scope — implementer
  reports first): arbitrary resolutions AND generic coil sets (the
  11-owner coil validator relaxation). Shipping rung 1 alone is an
  acceptable F4 outcome if rung 2 is reported as chartered follow-up.
- **The native side stays an oracle, not a shipped twin.** The C++
  evaluation of this formulation lives in the pinned instrument worktree;
  F4 does not port it. The example is registered as a JAX-native workflow
  whose certification oracle is external; the scoreboard row's receipt
  citation carries the native comparison. (A native twin example is out of
  scope — recorded as a possible follow-up, not attempted.)
- **No new timing claims.** The example's scoreboard row cites the sealed
  F3 receipt. The robustness runs are convergence/quality evidence, never
  timed claims; if any timing appears in their records it is labeled
  incidental and non-verdict.
- Cold-start disclosure language from the receipt carries into the example
  docstring (first solve in a process pays XLA compile; repeated/warm is
  the win regime).

## Work packages (one writer; sequential scoped commits)

- **C1 — constructor + tests.** Public build API; bundle loader refactored
  onto it; weight/policy mapping documented; artifact-free tests covering
  the construction path, the vessel-optional branch, the layout validators
  against constructor output, and SSOT (bundle path == general path on the
  bundle's own inputs, bitwise objective at the archived candidate).
- **C2 — example + registration + docs.** Example script (clone-runnable,
  CPU-defaulting with GPU flag per the examples convention, plus the
  `--bundle` certified-configuration mode), manifest entry
  (`host_boundaries: []`), device-assignment row + changelog, module
  docstrings updated to the promoted status; execution-source manifest
  cascade (`--admit` + count twins for every new `src/` file, digest
  refresh for edited members, SAME commit — this cascade bit the F3
  instrument work twice); drift gates green.
- **C3 — robustness evidence.** The runs in requirement 4, recorded as a
  compact tracked evidence note under `docs/receipts/` (or an addendum
  section format the reviewer prefers) with per-run gates listed;
  reviewer-checked before commit.

## Acceptance gates (all must hold before close)

1. SSOT gate, two separate standards: (a) **bitwise self-consistency** —
   the refactored bundle path reproduces its own pre-refactor objective,
   gradient, and GATE-3 fused endpoint **bit-for-bit** (an SSOT refactor
   of the same code path admits zero drift; 1e-15 is exactly the scale a
   construction-order regression hides at); (b) the cross-implementation
   archive comparison (1.2e-15 objective vs the native certificate)
   re-run unchanged as a control, not re-derived.
2. Example gate: the example script runs from a clean clone (CPU) in CI
   scale; its GPU scope declared per manifest convention.
3. Robustness gate: every chartered run meets its stated per-run gates, or
   failures are reported with a designed (not yet implemented) guard
   proposal — failures do NOT block C1/C2 landing; they block only the
   pre-committed perturbation-scoped claim of requirement 4, which is then
   narrowed to the evidence actually obtained.
4. Scoreboard drift suite green; execution-source manifest green; dual
   ruff; pyright 0.
5. Crucible strict PASS on the full delivery (requirements-e2e-review-loop
   closure).

## Non-goals

No mixed precision. No native twin example. No arbitrary-resolution
guarantee beyond the reported rung. No re-timing. No changes to the sealed
receipts or campaign evidence. No P-as-matvec (track C stays parked).

## Chartered follow-up

- **Persistent XLA compile cache for the examples lanes.** This document is
  the SSOT for the decision; the example docstring discloses the symptom and
  does not restate the lever. C2 measured the shipped example's CPU smoke at
  ~150 s and established that the cost is compile, not arithmetic and not the
  frozen L-BFGS history: a 4x problem shrink moved it 0.4% (136.97 s ->
  136.43 s) and `maxcor` 300 -> 10 moved it 12% (142.07 s -> 125.05 s), which
  leaves ~125 s in the objective and gradient program whose size the
  formulation fixes at 661 surface DOFs. There is therefore no lever inside
  the example, and shrinking it further would only shrink the lesson. The
  accepted answer is a persistent compile cache configured by the lane
  environment, which is also the mechanism the finite-build row already
  depends on for its warm claim. Not attempted here: it changes shared
  examples infrastructure, so it is a scoped change of its own rather than a
  rider on this promotion.
