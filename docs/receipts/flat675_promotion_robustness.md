# Flat-675 promotion robustness (F4 / C3) — evidence note

**Verdict: all 11 runs passed their chartered gates** (2026-08-19, RTX 5090,
`cuda:0`, fp64, fused on-device L-BFGS-B at the frozen campaign policy
`maxcor=300, maxls=8`, budget 37 for every run).

**Licensed claim, pre-committed before any run and reproduced verbatim from
the harness constant** (`LICENSED_CLAIM` in
`benchmarks/flat675_promotion_robustness.py`, and `licensed_claim` in the
tracked summary):

> robust to perturbations of the certified start at relative amplitudes
> <= 1e-1 (surface-block, coil-block, full-vector) and to one
> constructor-built start

Nothing stronger is claimed, and the outcomes do not widen it. In
particular this note does **not** claim robustness at amplitudes above 1e-1,
robustness of any other constructor-built start, or convergence — every run
spent its full 37-step budget and none of them is a converged solution.

**No timing claim is made.** The run records carry
`incidental_run_seconds` and `incidental_process_seconds`; both are labelled
non-verdict in the schema and exist only so a reader can tell a finished run
from a hung one.

Charter: `docs/jax_flat675_promotion_plan.md`, requirement 4 / work package
C3. All sealed flat-675 evidence before this note stood on one start
candidate; this is the evidence that bounds how far that generalizes.

## Runs

Amplitudes are relative in the block-2-norm sense: a perturbation is a
Gaussian direction, normalized, scaled so that
`||delta||_2 / ||block||_2` is the nominal amplitude by construction. The
achieved value is recomputed from the perturbed vector and recorded beside
the nominal one in every row, so the two cannot silently differ. They agree
to floating-point rounding rather than bitwise: across the nine draws the
largest relative departure is **8.46e-15** (`coil-0.001`, the block with the
smallest norm and therefore the least headroom), and four of the nine differ
by one ULP or less. Draws are reproducible from the recorded seed alone: each
uses `numpy.random.default_rng([20260819, amplitude_index, block_index])`, so
any single row can be reproduced without reproducing the grid.

| Run | Config | Block | Rel. amp | Start objective | Endpoint objective | grad-inf start | grad-inf end | Oracle objective | Oracle gap | Gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `control` | bundle | — | 0 | 21.1337055 | 0.013957202 | 14856.6 | 4.03708 | 0.013957202 | 4.85e-15 | PASS |
| `surface-0.001` | bundle | surface | 1e-3 | 21.300112 | 0.0146528198 | 14855.9 | 17.9736 | 0.0146528198 | 4.26e-15 | PASS |
| `coil-0.001` | bundle | coil | 1e-3 | 22.0484467 | 0.0170398028 | 16620.4 | 3.47595 | 0.0170398028 | 2.48e-14 | PASS |
| `full-0.001` | bundle | full | 1e-3 | 21.6366593 | 0.00916726047 | 14315.6 | 3.72991 | 0.00916726047 | 1.70e-15 | PASS |
| `surface-0.01` | bundle | surface | 1e-2 | 37.6952844 | 0.0116523461 | 14847.7 | 6.66299 | 0.0116523461 | 5.21e-15 | PASS |
| `coil-0.01` | bundle | coil | 1e-2 | 56.9651263 | 0.105313959 | 40979.4 | 11.5074 | 0.105313959 | 1.82e-14 | PASS |
| `full-0.01` | bundle | full | 1e-2 | 318.639904 | 0.0396655607 | 551624 | 10.3993 | 0.0396655607 | 4.90e-15 | PASS |
| `surface-0.1` | bundle | surface | 1e-1 | 1693.28297 | 0.28267653 | 16015.3 | 21.9671 | 0.28267653 | 5.30e-15 | PASS |
| `coil-0.1` | bundle | coil | 1e-1 | 125.082392 | 1.45913689 | 8984.41 | 110.716 | 1.45913689 | 4.57e-16 | PASS |
| `full-0.1` | bundle | full | 1e-1 | 9805.99262 | 2.78465165 | 235391 | 62.2066 | 2.78465165 | 4.78e-16 | PASS |
| `constructor` | repo geometry | — | 0 | 615.005562 | 11.7929372 | 290371 | 11.3581 | n/a | n/a | PASS |

Every run spent `nit = 37`; `nfev` ranged 40–62. Every run's host-transfer
ledger was `{initialization: 0, final_result: 15}` — zero `advance`, zero
`callback`, zero `unclassified` — so the fused lane's GATE-3 discipline held
on the GPU across all eleven solves, not only in the CPU example smoke.

## Gates, and what each one is worth

**Bundle-problem runs (the control and the nine perturbations).** Finite
endpoint; objective strictly decreased; and a native-oracle
cross-evaluation. The oracle is the fair-bar's own
(`benchmarks/genuine_675_fair_bar_oracle.py`) run against the pinned
instrument tree at `1c23f6c5f896`, re-evaluating each endpoint through the
archived native C++ evaluator with that endpoint's own `(iota, G)` as
anchor. The gate is the fair bar's own `ENDPOINT_OBJECTIVE_RTOL = 1e-10`,
imported rather than restated. Observed agreement spanned **4.6e-16 to
2.5e-14**, four to six orders inside the tolerance: the perturbed endpoints
are the same physics under an independent implementation, not merely
self-consistent JAX.

**The constructor-built run.** Finite; objective strictly decreased; and
endpoint gradient infinity norm strictly below the start's (290371 →
11.3581). **The oracle does not apply to this run and its absence is a scope
statement, not a missing gate**: the fair-bar oracle is wired to the archived
bundle's native material and can only evaluate candidates of that problem.
Nothing in this note cross-validates the repository-geometry configuration
against a native implementation, and the claim sentence does not say
otherwise.

One reading worth stating plainly: `surface-0.001` ends with a *larger*
gradient norm (17.97) than the unperturbed control (4.04) while reaching a
comparable objective, and `coil-0.1` ends at 110.7. Neither is a gate
failure — the chartered gate on bundle runs is objective decrease, not
gradient decrease — but they are the honest signal that a perturbed start
lands in a different place on the same surface at a fixed 37-step budget,
which is what a robustness note should show rather than smooth over.

## How it ran, and what that costs the reader

One process solved all ten bundle-problem starts, reusing one compiled
program: the perturbed starts are shape-identical to the archived one, and
**no timing claim depends on process isolation**, so per-solve processes
would have bought nothing but compile time. The constructor problem has
different shapes and compiled once more in its own process. Both children
ran under a persistent JAX compilation cache in the run root.

The layering is the F3 split and is forced rather than chosen: the native
oracle resolves its imports from the pinned instrument tree while the lane
under test must resolve only from the production tree, so the solves live in
a child process with a production-only `PYTHONPATH`. The child fails closed
if any `simsopt_jax` module resolves outside the production root.

Harness failure policy, exercised by design rather than by luck: a native
oracle *refusal* is recorded as the gate outcome `oracle_refused_endpoint`
with the native error text, and the summary is written before the harness
raises. No run hit that path here, but the campaign could not have lost a
failure to a crash. No guard was added anywhere: the charter reserves guard
design for review, and a measurement is not allowed to repair what it
measures.

## Evidence

- Tracked, compact:
  `docs/receipts/evidence/flat675_promotion_robustness/c3-20260820T023140Z/summary.json`
  (schema `flat675-promotion-robustness.v1`; every row above, plus per-run
  seeds, achieved amplitudes, transfer ledgers, inner states and gate
  verdicts).
- Host-local, disclosed not tracked (19 MB, dominated by the compilation
  cache):
  `~/simsopt_mixed_artifacts/flat675_promotion/c3-20260820T023140Z/` — the
  two child payloads (`bundle_child.json`, `constructor_child.json`, which
  additionally carry each endpoint's full 675 coordinates), the eleven
  per-run oracle directories (`oracle-<run>/request.json`,
  `oracle-<run>/oracle.json`), the provenance shim and the JAX cache.
- Producers: `benchmarks/flat675_promotion_robustness.py` (harness, oracle,
  gates, record) and `benchmarks/flat675_promotion_robustness_child.py`
  (solves only). Invocation contract is documented in the harness module
  docstring.
