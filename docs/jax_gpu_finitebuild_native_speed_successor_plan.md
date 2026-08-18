# Finite-build Stage-II native-vs-GPU speed: successor campaign

**Preregistered:** 2026-08-17, before any successor evidence exists.
**Predecessor:** `docs/jax_gpu_finitebuild_native_speed_implementation_plan.md`,
closed `CLOSED_BOUNDED_NEGATIVE` at commit `9d7f8d974`
(`docs/receipts/stage_two_finitebuild_native_gpu.md`). This is the successor
charter that receipt names: a **new preregistration with its own gate
derivation**, never an amendment to the closed campaign.

## Why a successor exists

The predecessor closed on two structural defects of its own protocol, both
measured and archived in its receipt:

1. **Endpoint-protocol asymmetry.** The native lane published its *first
   rung-crossing accepted iterate* (stopping callback); the fused GPU lane
   published only its *budget-terminal iterate*. The GPU crossed the
   objective rung near iteration 545 but that iterate was never captured.
2. **A landing-condition gradient clause.** The `1.05×`-anchor gradient
   infinity-norm cap was anchored to one lucky iterate: the reference run's
   own converged endpoint failed it (ratio 2.08), the callback-stopped
   shipped default failed it (1.638), and only 6 of 24 callback-stopped
   native legs landed under it (measured landings spanned 0.88–2.41).

The successor removes both while keeping every part of the predecessor's
machinery that worked: fp64 environment pins with child-observed
conformance, gate-source conformance (physics pins fail-closed,
harness/plan drift disclosed), oracle mediation of every GPU endpoint,
box-idle gates, attested environments, callback-stop native protocol,
fail-closed recompute-from-rows validation, and the receipt/evidence-bundle
publication standard set by commits `6bce010d0`–`71a5d1cb7`.

## The symmetric endpoint protocol (the one substantive change)

**Both lanes publish their first objective-crossing accepted iterate.**

- **Native** (unchanged): the stopping callback raises `StopIteration` at
  the first accepted iterate whose scaled objective clears the frozen
  target; cap `NATIVE_STOP_MAX_STEPS = 800`.
- **GPU** (new instrument, no solver changes): the fused loop is bitwise
  deterministic and the L-BFGS line search enforces sufficient decrease,
  so the terminal iterate of a fixed-budget-`b` run **is** the `b`-th
  accepted iterate of the same trajectory (verified empirically on the
  predecessor's archived legs: separately-run, separately-compiled
  fixed-budget legs at b560/b800 produce bitwise-identical solution
  vectors and identical `nit`). The sweep finds the crossing iteration
  `k*` — the smallest `b ≤ 800` whose terminal objective clears the
  target — by **bisection over plain fixed-budget runs**: virtual lower
  bound `b=0` (the initial objective, no leg), upper seed `b=800`,
  standard bisection to bracket width 1, ≤ 10 probe legs per history — a
  nominal bound (`[0, 800]` to width 1 is exactly 10 halvings; a lost or
  retried probe leg replaces its slot rather than adding one) — each a
  fresh untimed process. Two objectives play two preregistered
  roles: the **crossing decision** for every probe is the probe endpoint's
  **native oracle** objective (`≤ target`; gate mediation stays native,
  and the `1.001×` margin sits ~13 orders above the oracle's measured
  1-ULP evaluation noise), while the **monotonicity guard** reads the
  probes' **GPU self-reported** objectives — the deterministic quantity
  the line-search argument covers — and fails closed (`NOT_PRODUCED`)
  only on an increase exceeding `rtol 1e-12` (the predecessor's archived
  oracle re-evaluations of bitwise-identical endpoints differ by 1 ULP,
  which a strict guard would false-reject). The full quality gate is then
  evaluated at `k*` through the native oracle. There is **no forward scan
  past `k*`**: if the gate fails at the crossing iterate, that history is
  ineligible, exactly as a native repetition that fails the gate at its
  callback stop is disqualified. A non-minimal (high) `k*` from a
  bisection defect can only slow the GPU lane, never flatter it.
  Bisection probes are **untimed and unranked**; after `k*` is found, one
  **timed** leg per crossing history runs at exactly `k*` under the
  box-idle gates (one discard + `SELECTION_REPETITIONS` warm solves) and
  supplies the selection statistic. Deliberately, the timed leg runs for
  every history that crosses and the **full gate is evaluated on the timed
  leg's own native re-evaluation** — the gated endpoint is the endpoint
  whose time is published, never a sibling's. The determinism the bitwise
  clauses assert is evidenced on the exact protocol axes they compare
  (2026-08-17, h10/b100, three fresh processes): a cold-cache first
  solve (genuine 42.7 s compile), a primed-cache first solve (9.0 s
  load), and a primed-cache fourth solve produce bitwise-identical
  solution vectors — cache state, autotune, and warm repetition count do
  not perturb the trajectory. Probe rows are tracked at
  `docs/receipts/evidence/stage_two_finitebuild_native_gpu_successor/protocol-determinism/`
  (`probe-a.json` sha256 `7de8e1b9920c6588…`, `probe-b.json`
  `bba4c7dc922ccf24…`, `probe-c.json` `aae18076f7631cbd…`).
- **Final pairs**: the GPU lane runs plain fixed-budget `k*`; the native
  lane runs its callback-stop protocol. Both lanes therefore time "solve
  until the first accepted iterate at native-equivalent quality", with
  one disclosed asymmetry: the GPU's stopping point is frozen once from
  the sweep, while the native lane's is redetermined every repetition
  under its measured ~1% OpenMP trajectory fork — priced against the
  native lane already by the 3-of-3 qualification rule. **Every** pair's
  GPU endpoint solution is verified bitwise against the sweep's stored
  crossing solution; any mismatch is `NOT_PRODUCED` (broken determinism
  is broken evidence, not a slow lane).

## Frozen quality contract (schema `stage-two-finitebuild-quality-contract-v4-successor`)

Derived fresh by the successor's own gate phase (fp64 environment, OMP=8
reference, 400-step forced-tolerance reference run, truncated
same-trajectory anchor at the first qualifying iterate — all as the
predecessor's amended derivation):

- target objective = `1.001×` the converged reference endpoint;
- one-sided caps: endpoint objective ≤ target; squared flux, length
  penalty, distance penalty each ≤ `atol 1e-9` + `1.05×` anchor;
- two-sided geometry bands: minimum clearance and each coil length within
  `rtol 5e-2 / atol 1e-9` of the anchor; positive clearance;
- **gradient infinity norm ≤ `2.3×` the reference gradient scale**, where
  the reference scale is the **median `|g|∞` over the reference
  trajectory's `min(21, available)` accepted iterates up to and including
  the anchor** (window values and count published in the contract, floor
  `1e-12`). Mechanism, named here because the contract's own multiplicand
  depends on it: the gate reference leg already retains every accepted
  iterate for anchor derivation, so the window norms are computed **after
  the solve** by native evaluation of the last 21 stored iterates (~21
  untimed evaluations; the solve, its `nfev`, and its stopping callback
  are untouched). The window ends *at* the anchor deliberately: the
  pre-crossing descent regime is where a crossing lane actually lands,
  and its `|g|∞` runs systematically higher than at convergence. The
  window median — not the single anchor draw — is the multiplicand
  because the predecessor measured `|g|∞` swinging
  `1.349e-06 → 2.808e-06` within two iterations of one trajectory: a
  single crossing-iterate anchor is a lottery draw, which was the
  predecessor's failure class. **Basis conversion, disclosed:** the
  archived calibration ratios are quoted against the predecessor's anchor
  draw, and that draw was a local *minimum* — across the 24 archived
  callback-stopped native legs, first-crossing `|g|∞` sits at median
  `1.31×` (mean `1.43×`) that anchor, so a tail-window median is expected
  *above* the old anchor, not below it. Re-expressed on the
  window-median basis, the worst measured honest landing (`2.41` in
  old-anchor units) is `≈ 1.84`, and the multiplier `2.3` reproduces the
  intended `≈ 1.25×` headroom over it — carrying `3.0` onto the new
  basis would have silently loosened the clause to `≈ 3.9×` the old
  anchor, a pro-GPU drift this charter rejects. The stalls the clause
  exists to reject sit at `≈ 18–60` on the new basis. The gate phase
  publishes the window median, the anchor draw, and their ratio so the
  effective cap in archived-ratio units is auditable at freeze time — and
  the audit has a named response, not just a disclosure: if the measured
  ratio implies an admission threshold below the worst archived honest
  landing (`2.3 × median < 2.41 × anchor`), the gate phase **halts
  fail-closed before any lane runs** and the multiplier is re-derived
  under a dated amendment, which the amendment discipline permits because
  the gate audit precedes all lane evidence. The
  GPU lane's own archived value (`1.98` old-anchor ≈ `1.51` new-basis)
  sits below the native maximum and does not bind the cap — the clause is
  calibrated by native behavior alone.
- **The v4 contract is frozen at gate-freeze and is never amendable
  thereafter, on any lane's evidence.** The predecessor's most contested
  moment was whether its gradient clause could be amended after becoming
  the sole binding clause; the successor forecloses the question.

**Disclosed denominator effect (archived-data estimate, not a
prediction):** re-scoring the predecessor's archived native matrix under a
`3.0×`-class cap admits ~7 configurations and moves the fastest qualifying
native lane from `81.657 s` (omp2-h10) to `≈ 44.8 s` (omp2-h400) — a
`≈ 1.8×` **harder** bar for the GPU. The clause's measurable effect on the
denominator is anti-GPU; the successor's own fresh gate and matrix decide
the real number.

Oracle mediation, gate-sha binding, source fingerprints, fp64 and
gate-source conformance clauses: identical to the predecessor's hardened
validator (commit `71a5d1cb7`). The successor extends the source
fingerprints with this charter's own hash as a disclosed pin (absent at
pre-successor commits, recorded as such — archived runs keep validating).

**Archived-verdict recomputability:** the successor modifies the shared
harness in place (`benchmarks/stage_two_finitebuild_native_gpu.py`), and
every archived verdict must keep recomputing from its own rows. This is
enforcement, not convention: every row is cryptographically bound to its
contract (`gate_sha256` per row — an archived run cannot be scored
against any contract but its own), quality clauses read tolerances from
each run's **stored** contract, and the sweep reducer discriminates by
the rows' own role field — the predecessor's ladder rows keep their
ladder reduction, successor bisection rows get the bisection reduction.
One carve-out: the lane cross-check (`_lane_cross_check`) reads its
`rtol`/`atol` from module constants, not stored tolerances — it is an
evidence-integrity check rather than a quality clause, those constants
have never changed, and changing them falls under the fork rule below. A
change that would break recomputation of any archived terminal verdict
requires a forked harness instead.

## Selection and final measurement

1. **Gate → baseline → kernel canary** exactly as the predecessor
   (canary kill: warm value/grad GPU ≥ `1.10×` best native across OMP
   2–48, else close).
2. **Native matrix**: callback-stop time-to-quality over OMP
   `{2,4,8,16,32,48}` × history `{10,20,40,400}`, 3 repetitions, cap 800,
   under the v4 gate. Fastest fully-qualifying configuration is the
   denominator. Shipped-default disclosure lane unchanged.
3. **GPU sweep**: per history `{10,20,40}`, bisection for `k*` as
   specified above (virtual `b=0` lower bound, `b=800` upper seed,
   bracket width 1, oracle-decided crossings, ≤ 10 untimed probe legs,
   every probe a full fresh-process solve with row publication), full
   gate at `k*` via oracle, then one timed leg at `k*` per qualifying
   history for the selection statistic. Smallest-median-warm-time
   qualifying history is selected; its `k*` and its crossing solution
   vector (with sha256) are frozen into the selection.
4. **Five final pairs** (alternating order, pinned affinity, serialized
   GPU, identity-checked): warm legs (1 discard + 3 timed, median) and
   wall legs (single solve process wall), primed persistent cache; the
   fresh-empty-cache compile/process time reported separately, no cold
   claim unless it independently passes the same rule.
   **WIN** requires median paired `native_seconds / gpu_seconds ≥ 1.10`
   for both `warm_solve_seconds` and warm persistent-cache
   `process_wall_seconds`, with **every** pair `> 1.00`. Anything else is
   `CLOSED_BOUNDED_NEGATIVE`. `NOT_PRODUCED` stays what it always was:
   broken evidence, never a verdict.

## Kill criteria (all fail-closed, none amendable post-evidence)

- Kernel canary `< 1.10×` → close.
- No GPU history qualifies at its crossing iterate within `b ≤ 800` →
  close (budget parity with the native cap; final).
- Bisection monotonicity violation (GPU self-reported objectives, beyond
  `rtol 1e-12`) → `NOT_PRODUCED` (instrument defect, investigate before
  any rerun).
- Any final-pair speed gate failure → close.
- Any final-pair GPU endpoint deviating bitwise from the frozen crossing
  solution → `NOT_PRODUCED`.
- Lane cross-check (`rtol 5e-2`) or any conformance clause →
  `NOT_PRODUCED`.

## Production deliverable (win path)

`src/simsopt_jax/examples/stage_two_finitebuild.py`: the internal workflow
module of the predecessor plan's Step 4, baking the frozen selection
(history, `k*`) with the cache-marked `problem._solver_value_and_grad_fn`
routed through `dispatch.minimize`; both JAX callers rerouted; behavioral
tests replace source-shape tests (predecessor plan Step 5, verbatim);
strict-transfer integration test under `jax.transfer_guard("disallow")`.
Execution-source manifest grows 615 → 616 (the module), count twins
bumped, regenerated in the same commit. On the close path the module is
not created and the manifest stays 615.

## Publication

Terminal receipt at
`docs/receipts/stage_two_finitebuild_native_gpu_successor.md` with full
hashes and a tracked evidence bundle under
`docs/receipts/evidence/stage_two_finitebuild_native_gpu_successor/`;
device-assignment row + dated amendment-log entry in the same
measured-verdict commit (a win moves the row to `gpu` with the receipt
cited; a close leaves it `unmeasured` unless the completed evidence
actually establishes the native lane faster). Both timers' medians, every
pair ratio, and the fresh-cache numbers publish regardless of verdict.

## Amendment discipline

Dated amendments are permitted only before the evidence they govern
exists, and never to the kill criteria above after any sweep or pair has
run. Every amendment cites its empirical basis by artifact path and hash.
