# Custom JAX BFGS/L-BFGS Step Runtime Implementation Plan

**Status:** In progress — core runtime green; physics and promotion gates open

**Last updated:** 2026-08-02

**Change tier:** Tier 3 — public solver behavior and traced call paths

## Current status (2026-08-02)

**Verdict:** the fixed-step runtime and focused CPU/GPU contracts are green,
but the change is not promotion-ready.

| Area | Current evidence | Open gate |
| --- | --- | --- |
| Runtime | Eager BFGS uses the typed host driver; eager L-BFGS uses the specialized design-B facade. Traced whole-solve routes remain available. The broad Boozer compatibility set is green when partitioned into fresh processes. | Complete application-scale endpoint closure. |
| Numerical parity | Rosenbrock accepted states are byte-identical to the pre-refactor solver. Matched `coil47` native CPU/custom CPU/custom GPU/Optax endpoints converge at the recorded FP64 objective tolerance. | Close a matched converged Boozer endpoint and the full accepted-state matrix. |
| GPU | The 41 non-slow runtime contracts and the broader Boozer/traceable compatibility selector pass on strict RTX 5090 CUDA. Custom `coil47` converges on GPU; the Boozer outer BFGS now has a current-HEAD 20-iteration CPU/GPU diagnostic pair. | Qualify a converged Boozer endpoint and run the declared A100 lane. |
| Performance and memory | Bounded diagnostics favor specialized design B over generic design A. Custom now exposes a production `prepare_lbfgs_private`/`PreparedLBFGS` boundary; Optax uses the same prepare/run split. Both reuse fixed-shape programs for warm runs; a clean strict-RTX-5090 `coil47` five-sample receipt now includes process-attributed VRAM and phase-scoped RSS. | Complete the StableHLO/compile promotion comparison and the A100 gate. |
| Evidence | Local candidate receipts have artifact checksum tests. A clean candidate worktree validates all `39` manifests against the external bundle `/home/jungdaesuh/simsopt-jax-quasi-newton-evidence/20260802`; bundle inventory SHA-256 is `43ea7e7dd07c7914d16054c1dae206c3a6cdbc2293b493c255dfb87b43659ea9`. CPU lock SHA-256 is `159e05a65796e76dfb502ea4f6a06b1f412af1c7bb147bb5ac5974b5888a6b35` and GPU lock SHA-256 is `fc724b570ca23356b18df17da87a00217066fd42e5b02de5fe26b46cf20473f8`. | Replicate the bundle off-host and re-run the same validator there. |
| Quality | Focused tests and compile/diff checks are green. The current large Boozer source/test files still report existing Ruff findings. | Close scoped Pyright, project-wide Ruff, application-scale endpoint, and clean-checkout gates. |
| TDD | New defects have preserved RED -> GREEN evidence where recorded. | The already-implemented core has post-hoc tests only; historical RED revisions cannot be recreated and are not a completion claim. |

<details>
<summary>Detailed diagnostic ledger</summary>

The entries below preserve raw measurements and failed probes. They are not a
substitute for the open promotion gates above.

Measurement caveat: legacy fields labeled `solver_boundary_rss_peak_kib` or
`solver RSS delta` use process-lifetime `ru_maxrss` after the solver window.
They can include earlier fixture, compilation, or instrumentation peaks and are
diagnostic only until Phase 5 phase-scoped measurements replace those legacy
fields in promotion receipts. The Optax warm
path now reuses its prepared state and compiled programs, but it remains an
explicit comparator rather than a custom-solver parity oracle.

- Green: typed eager runtime, BFGS eager steps, L-BFGS dynamic-budget facade,
  and SciPy Rosenbrock accepted-step parity. The private wrapper cache now has
  capacity-8 LRU/single-flight admission with tests for owner lifecycle,
  eviction of an in-use wrapper, key identity, failed compilation, cardinality,
  and concurrent first use. Tree-definition key separation is covered for
  structured adapters; full adapter execution collision coverage and XLA
  executable reclamation are not yet qualified.
- Green current qualification rerun: the documented strict-CPU selector
  completed with `131 passed, 4 deselected` in `385.00 s`; the matching strict
  RTX-5090 selector completed with `48 passed, 2 deselected` in `157.25 s`.
  Both lanes used FP64, strict backend selection, and the current worktree.
  These are contract-suite results, not application-scale Boozer or promotion
  receipts.
- Green historical pre-wiring selector: the combined solver/trajectory/runner
  tests completed with `82 passed, 4 deselected` on strict CPU and again on
  strict RTX-5090 CUDA (`153.82 s`). The later runner integration check found
  that its prepared custom object was not passed into the timed solve; those
  timing claims are superseded by the production-boundary rerun below.
- Green current production prepared-boundary rerun: custom and Optax now use
  the production custom `PreparedLBFGS` boundary and the existing Optax
  prepared step. After vectorizing the history updates, strict CPU `coil47`
  (`maxiter=20`) native/custom/Optax converged in `14/13/15` iterations;
  final objectives were `0.13786263284430203/0.137862632844302/
  0.137862632844302`, with cold/warm times `0.5794/0.8750 s`,
  `5.1284/0.01533 s`, and `1.4539/0.0090 s`. Custom and Optax solver RSS
  deltas were `316044/155700 KiB`; native was `0 KiB`. On strict RTX-5090
  CUDA (`maxiter=20`), custom and Optax converged in `13/15` iterations to
  `0.13786263284430203/0.13786263284430206`; cold/warm times were
  `14.077/0.04490 s` and `3.921/0.02169 s`, with RSS deltas
  `385420/193984 KiB`. Receipts:
  `docs/receipts/custom-quasi-newton/coil47-native-custom-optax-cpu-maxiter20-prepared-vectorized-20260802/`
  and
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-maxiter20-prepared-vectorized-20260802/`.
- RED -> GREEN compatibility closure: Boozer reference tests now select
  `native_cpu` explicitly, and traceable tests patch the adapter's directly
  imported private-solver symbols. A stale functional-result fixture now uses
  the production Newton reporting-field helper. The full selected compatibility
  gate passed `111 passed, 3 skipped, 628 deselected` in `159.87 s` under strict
  CPU FP64; the three skips are the declared GPU-only cases. The prior failures
  were test-contract defects, not solver failures.
- RED -> GREEN GPU compatibility closure: the same bounded selector completed
  `110 passed, 4 skipped, 628 deselected` in `362.21 s` under strict RTX 5090
  FP64. Two GPU-only stale test contracts were corrected: traceable exact Newton
  is operator-only and therefore returns no dense Jacobian, and the mocked Newton
  callback accepts the production `materialize_hessian` and
  `max_dense_hessian_bytes` keywords. No solver implementation change was needed.
- RED -> GREEN surface-metadata compatibility: exact KKT-mask construction now
  belongs only to the optimized surface, so an independently quadratured label
  surface cannot poison an LS setup. If an LS object later enters the legacy
  exact path, the mask is built lazily from the cached optimized-grid metadata.
  The long strict-CPU selector reached `711 passed, 30 skipped` before exposing
  this compatibility case; the repair then passed the exact selector (`97 passed,
  4 skipped`) and the upstream factory matrix (`90 passed`).
- Green: CPU runner records fixture-build elapsed time and the RSS observed at
  the build boundary separately from cold/warm solver time, provider-child
  RSS, status, counters, full initial/final parameter vectors, and explicit
  Optax comparison on deterministic contract fixtures. Each measurement now
  also records generator/source hashes, full fixture-contract metadata,
  expected initial observables, solver options, predeclared tolerances, and
  final certificate fields. The measurement payload is now schema version 6;
  version 3 adds the phase-separated warm transfer audit, version 4 adds
  legacy solver-window RSS start, process-lifetime peak, and derived delta
  fields, version 5 records the JAX/XLA/SIMSOPT runtime environment, and
  version 6 records named phase-scoped RSS windows.
  Commit/dirty-state and dependency versions remain recorded; the supported
  Rosenbrock command was re-run from the documented environment.
- Green diagnostic: the corrected runner now samples named fixture-build,
  preparation, cold-solver, and warm-solver RSS windows in each provider child.
  Fresh strict-CPU `coil47` custom/Optax preparation peaks were
  `979264/828940 KiB` and solver-window peaks were `979828/836496 KiB`; warm
  times were `0.01582/0.009710 s`. Strict RTX-5090 preparation peaks were
  `1856496/1676456 KiB` and solver-window peaks were `1861528/1681804 KiB`;
  warm times were `0.03790/0.01963 s`.
  These are host-RSS diagnostics, not device-memory measurements. Receipts:
  The current phase receipts use measurement schema 6; earlier schema-5
  receipts remain historical diagnostics.
  `docs/receipts/custom-quasi-newton/coil47-native-custom-optax-cpu-phase-rss-schema6-20260802/`
  and
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-phase-rss-schema6-20260802/`.
- Green diagnostic: six fresh strict-RTX-5090 `coil47` provider runs were
  collected; the first was discarded and samples 1--5 were retained. Custom
  warm time had median/range `0.044907/0.041051--0.055912 s`; Optax had
  `0.027435/0.026636--0.029175 s`. The corresponding maximum warm-window
  RSS was `1817316/1632504 KiB` (custom/Optax), with no monotonic increase
  across the retained samples. All endpoints succeeded and matched the
  objective within `3e-17`; custom took 12--13 iterations versus Optax's 15.
  Raw outputs are archived under
  `.artifacts/custom-quasi-newton-archive/gpu-five-sample-coil47-20260802/`.
  No device-memory telemetry or clean-checkout provenance was captured, so
  this remains diagnostic rather than promotion evidence.
- Green clean-candidate GPU receipt: one discarded warm-up plus five retained
  custom and Optax samples ran from clean commit `3b2b9f40a` with
  process-attributed `nvidia-smi` VRAM. Custom/Optax medians were
  `14.43294/4.07767 s` cold and `0.044042/0.027639 s` warm; maximum solver RSS
  was `1786248/1605436 KiB`, and maximum process VRAM was `1514/2602 MiB`.
  Objectives differed by at most `5.55e-17`. Receipt and raw evidence:
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-five-sample-vram-20260802/`.
  This qualifies only the RTX-5090 lane; A100 and StableHLO promotion remain
  open.
- Green clean strict-CUDA contract selector: `48 passed, 2 deselected` in
  `81.23 s` from the candidate worktree, with FP64 and no CPU fallback.
- Blocked external lane: `ssh landau` returned `No route to host` on
  2026-08-02; no A100 qualification claim is made.
- Green clean-candidate replay: detached checkout `41d95cf502240a686cc968e690f62d4a85a2d1a3`
  was clean; the receipt and runtime subset passed `51` tests with `2`
  deselected, and a fresh-process `validate-all` verified all `38` tracked
  manifests against the local archive. The later external-bundle replay is
  the portable form of this check.
- Green external archive replay: the same `38`-manifest validator passed from
  clean candidate `3b2b9f40a` with `--repo-root` bound to the external bundle;
  its inventory hash and path are recorded in the rollback receipt.
- Green final receipt replay: clean candidate `8f29ca7a6` validates `39`
  manifests, including the five-sample GPU/VRAM receipt, against the same
  external bundle. The fresh receipt tests pass `7/7`.
- Green bounded rollback rehearsal: the candidate was reverted in reverse
  order to the declared base and the resulting index tree was byte-identical.
  The native-CPU compatibility and bounded Boozer eager selectors passed on
  both base and candidate. The broad traceable selector reproduced the known
  resource blow-up and was stopped; the receipt is
  `docs/receipts/custom-quasi-newton/rollback-rehearsal-20260802.md`.
- TDD limitation: the phase-RSS and dense-analysis-phase regressions were added
  after the implementation was present. They are current post-hoc GREEN tests;
  no historical RED revision is being claimed.
- Green: schema-4 matched CPU `coil47` receipts now expose solver-boundary
  memory. Native/custom/Optax solver RSS deltas were `0 / 355824 / 1699508`
  KiB; final objectives remained `0.13786263284430203 / 0.137862632844302 /
  0.137862632844302`. Receipts are under
  `.artifacts/custom-quasi-newton/20260801T-coil47-{native,custom,optax}-cpu-schema4-fixed/`.
  This is CPU evidence only; strict GPU and A100 lanes remain open.
- Green: provider children now have direct-PID 120-second and 8-GiB RSS
  fail-closed watchdogs with TERM/KILL grace; both limit paths have bounded
  contract tests without allocating an 8-GiB fixture.
- Green diagnostic: a fresh strict-GPU Optax `coil47` two-step run sampled
  process VRAM and completed with a checksummed receipt. The provider peaked
  at `834 MiB` of `32607 MiB`; all 14 current receipt manifests validate their
  raw-artifact hashes only. Environment-lock validation remains open, including
  the known CPU-receipt/GPU-lock mismatch. This remains fixed-budget evidence
  on a dirty checkout.
- Green: after the typed BFGS transition and observation annotations, the
  focused BFGS contracts passed `27/27` on strict CPU and `27/27` on strict
  CUDA. The shared optimizer-input normalizer now has an explicit typed
  callable contract, and BFGS scalar/value-and-gradient branches are narrowed
  explicitly. The combined runtime/trajectory selector passed `45/45` on
  strict CPU and `45/45` on strict CUDA. The current scoped Pyright profile
  reports `797` diagnostics; remaining findings are dependency-stub and
  benchmark/test call-site typing work, not runtime failures. The runner
  contract selector is also green at `38/38` (two slow probes deselected); it now fails closed when the
  requested device/intent pair does not select the canonical
  `SIMSOPT_BACKEND_MODE`. The prepared custom-program boundary also fails
  closed on mismatched reuse. Compileall, touched-file Ruff, formatting, and
  `git diff --check` pass.
- RED -> GREEN: the Optax benchmark comparator now prepares its immutable
  solver transformation, fixed-shape step, and endpoint value/gradient
  executable once per measurement. Each cold and warm solve initializes fresh
  state from the same parameters, while the warm path reuses the compiled step
  and accepted line-search state. Its
  stop check now reuses the accepted value/gradient stored by Optax's
  line-search state after each update, so one-step quadratic convergence does
  not incur a second zero-step iteration or an extra objective evaluation.
  Initial convergence, nonfinite endpoints, and line-search failure are
  labeled directly. Prepared programs are bound to the objective, initial
  vector's exact FP64 bit pattern, shape, and history size; mismatched reuse
  fails closed, including signed-zero and NaN inputs. The focused runner
  selector is `38 passed` (two slow probes deselected). The custom provider now exposes the production
  `PreparedLBFGS` boundary over the same fixed-shape private transitions;
  phase-scoped host-RSS sampling is now green; clean promotion receipts remain
  open.
- RED -> GREEN runner wiring: the first prepared-boundary integration check
  exposed that custom preparation was not passed into the timed solve and that
  native received an invalid prepared argument. The runner now routes native
  without preparation and custom with its prepared program; regression tests
  cover both branches.
- Green: `benchmarks/custom_quasi_newton_receipts.py` now publishes runner
  output atomically with lock/artifact hashes and validates both tracked and
  archive bytes. Its tamper, missing-lock, dirty-receipt, and fresh-process
  tests pass; the existing 18 custom quasi-Newton receipts revalidate from the
  current checkout. The publisher now rejects an archive URI that aliases the
  tracked destination before creating either side.
- Green diagnostic: the CPU `coil47` native/custom/Optax `maxiter=20` bundle
  was rerun against `benchmarks/environments/custom_quasi_newton_cpu.lock.txt`
  and published as
  `docs/receipts/custom-quasi-newton/coil47-native-custom-optax-cpu-maxiter20-lockbound-current/`.
  The tracked receipt and distinct `.artifacts/custom-quasi-newton-archive/`
  copy validate from a fresh process. This pre-JIT comparator bundle is
  retained as historical diagnostic evidence.
- Green diagnostic: after the Optax comparator was changed to one JIT-compiled
  fixed-shape step, a fresh lock-bound CPU bundle was published at
  `docs/receipts/custom-quasi-newton/coil47-native-custom-optax-cpu-maxiter20-jitted-current/`.
  Native/custom/Optax took `12/13/16` iterations, with warm times
  `0.9686/0.0581/2.4474 s`; all three converged. The dirty checkout keeps it
  diagnostic-only.
- Green diagnostic (Optax-only preparation boundary, superseded): a fresh
  strict-CPU `coil47` custom/Optax `maxiter=20` run reuses the prepared Optax
  step and endpoint program for the warm measurement. Both endpoints
  converged to `0.137862632844302`; custom/Optax took `13/15` iterations,
  with cold/warm times `5.3460/0.05532 s` and `1.6090/0.006305 s`, respectively.
  Solver RSS deltas were `346824/150240 KiB`. The checksummed receipt is
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-cpu-maxiter20-prepared-20260802/`;
  it remains diagnostic and predates the custom prepared-provider boundary.
- Green diagnostic (both prepared): the current strict-CPU `coil47`
  custom/Optax `maxiter=20` run compiles both providers before the timed
  solves. Both endpoints again converged to `0.137862632844302` in `13/15`
  iterations. Cold/warm times were `10.0737/0.05186 s` for custom and
  `1.4427/0.008220 s` for Optax; solver RSS deltas were `515428/151404 KiB`.
  The receipt is
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-cpu-maxiter20-prepared-both-20260802/`;
  it remains diagnostic and is superseded by the fixed production-boundary
  receipt because the runner wiring was corrected afterward.
- Green diagnostic: the matching strict-RTX-5090 custom/Optax bundle is
  published at
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-maxiter20-jitted-current/`.
  Both providers converged (`12/16` iterations) with final-objective
  difference `2.78e-17`; warm times were `0.0783/6.1386 s`. Custom peak
  solver RSS was `532472 KiB` versus Optax `243112 KiB`; this exceeds the
  predeclared `1.5x` diagnostic ratio and remains non-promotion evidence until
  phase-scoped measurements replace the legacy process-lifetime field.
- Green diagnostic (both prepared, strict CUDA): the current `coil47`
  `maxiter=2` run compiled and reused both providers' fixed-shape programs.
  Custom/Optax warm times were `0.04779/0.006254 s`; final objectives were
  `0.13786469682070215/0.13786469652455854` and both endpoints were correctly
  labeled `iteration-limit`. The checksummed receipt is
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-maxiter2-prepared-both-20260802/`.
  This is fixed-budget GPU evidence, not a converged parity or A100 result;
  it is superseded by the fixed production-boundary receipt.
- Green: the full non-slow CPU qualification selector passed `127/127` with
  four intentionally deselected slow probes in `533.75 s`; no failures or
  skips were hidden. The long tail is compilation-heavy, not a hung solver.
- Green: a strict-CUDA public `Driver.SIMSOPT_LBFGSB` smoke on the pinned GPU
  environment converged in two iterations and returned the typed
  `LbfgsInvHessProduct` (`shape=(2, 2)`) without densifying it.
- Green: eager BFGS now preserves nonfinite/failed initial-stop semantics and
  maps callback `StopIteration` to unsuccessful status 99 through the typed
  result boundary.
- Green: traced whole-solve BFGS is reachable again; tracer closure constants
  now remain under the enclosing JIT's placement instead of probing tracer
  shardings.
- Green: traceable Newton refinement now normalizes optional linear-solve status
  fields before `lax.cond`; the previously failing six-case traceable private
  compatibility slice passes.
- Green: traceable Boozer penalty objective and residual closures are now
  cached per instance by the existing structural key. Repeated LM/Boozer
  calls reuse the same callables; target-label and option changes still build
  fresh closures. The RED regression was caused by rebuilding the objective
  on every call and is now covered by stable/rebuild tests.
- Green: eager BFGS and stepwise L-BFGS observer paths pack accepted-step
  payloads at their explicit host boundary; monolithic compatibility callbacks
  still use their legacy boundary and remain separately qualified.
- Green: dense-BFGS receipts now include the logical no-donation contract and
  an isolated XLA buffer analysis for the update itself. On the 47-variable
  FP64 case, the logical upper bound is `108,560 bytes`; the compiled update
  peak is `89,424 bytes` on CPU and `33,645,600 bytes` on the RTX 5090 GPU.
  The GPU number is compiler temporary allocation, not a claim that the
  Hessian itself is 33 MB; it records the backend amplification explicitly.
- Diagnostic: the bounded CPU Boozer BFGS receipt (65 variables, two-step cap)
  measured a custom solver-boundary RSS delta of `1,521,644 KiB` versus native
  `0 KiB`; the custom logical upper bound is only `204,880` bytes. This is
  process-level compile/allocator evidence, not proof of dense-update device
  bytes, and both endpoints are capped (`status=1`, `success=false`).
- Green: L-BFGS eager budget scalars now use the device-placement owner rather
  than raw host `jnp.asarray`; the explicit value/gradient transfer-guard test
  and the fast public BFGS/L-BFGS CPU subset pass after the fix.
- Decision: use physical design B. An exploratory monolithic L-BFGS probe
  exceeded 120 seconds and 5 GiB RSS; an exploratory specialized-kernel facade
  completed the same Rosenbrock probe in about 3.9 seconds cold and 0.6 GiB
  RSS. These figures are not promotion receipts.
- Green high-history diagnostic: after replacing Python-unrolled history
  algebra with JAX loop/vectorized updates, fresh `PreparedLBFGS` Rosenbrock
  runs with `maxcor=300` completed on CPU (`2.21 s` cold, `0.43 GiB` RSS for
  two steps) and strict RTX-5090 (`6.62 s` cold, `0.21 GiB` RSS for two
  steps). The `maxiter=20` CPU run stayed within the bound and reported its
  iteration cap (`status=1`); it is a stability result, not convergence proof.
  Receipts:
  `docs/receipts/custom-quasi-newton/rosenbrock-custom-maxcor300-cpu-20260802/`
  and
  `docs/receipts/custom-quasi-newton/rosenbrock-custom-maxcor300-gpu-20260802/`.
- Green diagnostic A/B comparison: the same 47-variable FP64 quadratic,
  `maxcor=10`, and two-step budget used the old generic transition and the
  specialized transition. Design A timed out at 120.48 s and 1,676,884 KiB
  RSS before lowering completed; design B lowered in 2.37 s, completed in
  10.35 s, and used 420,228 KiB RSS. The local candidate comparison is under
  `docs/receipts/custom-quasi-newton/compile-design-ab/`; the cheap quadratic
  isolates optimizer graph cost, so no objective-specific refactor was opened.
- Green diagnostic correction: the compile-shape probe now passes
  unconstrained_fast_path=True to the specialized no-bounds transitions, so
  it measures the production route rather than the full bounded branch graph.
  The corrected quadratic maxcor=10 lowering shrank from about 24.9 MB to
  0.48 MB StableHLO and from about 329k to 9.6k JAXPR lines; lowering completed
  in 0.797 s on the recorded CPU probe.
- Diagnostic decision: warm cells skip duplicate StableHLO text generation and
  isolate executable/solver timing; compile-only cells provide the StableHLO
  and lowering measurements. All cells still record objective cold/warm calls,
  RSS samples, compile events, executable counts, and accepted-step progress
  when a solver run is requested.
- Green matrix receipt:
  .artifacts/lbfgs-ondevice/root-cause-matrix-fastpath-full.json.
  Four of five declared cells completed under the 120-second/8-GiB direct-PID
  policy: quadratic/maxcor=10 warm completed in 26.73 s with peak 661,672 KiB
  RSS and two iterations; coil47/maxcor=10 compile-only completed in 27.32 s
  with 1,504,656 StableHLO bytes and 844,004 KiB RSS; coil47/maxcor=10 warm
  completed in 57.59 s with 11 accepted iterations, converged status 0, and
  1,022,488 KiB RSS; and coil47/maxcor=300 compile-only completed in 25.41 s
  with 1,505,243 StableHLO bytes and 837,428 KiB RSS. Warm maxcor=10 cells
  compiled five stepwise executables and did not recompile across three calls.
- Bounded high-history finding: quadratic/maxcor=300 warm reached its first
  accepted-step progress event, then timed out at 120.60 s with 1,762,368 KiB
  RSS. This is a runtime/history-size finding, not a compile failure; the
  receipt is incomplete by contract and is not promoted.
- Green clean design-B matrix: the four completed cheap/coil cells were
  rerun from clean commit `975f6b722`. Coil47 lowering was
  `1,571,730/1,572,317` StableHLO bytes for `maxcor=10/300`; the warm coil
  cell used `1,024,812 KiB` RSS, reached 11 iterations, and did not recompile
  across repeated calls. Raw evidence is
  `docs/receipts/custom-quasi-newton/compile-matrix-clean-20260802/`.
  The legacy design-A timeout remains the reason the full A/B promotion gate
  is open.
- Historical pre-fix probes remain diagnostic only: the unbounded combined
  attempt reached 6.1 GiB RSS, and the old general-branch compile reached
  9,420,664 KiB RSS before interruption. Legacy kernels remain explicit opt-in.
- Safety finding: an exploratory unbounded attempt to combine the four root-
  cause cells in one process reached 6.1 GiB RSS while compiling the
  `coil47/maxcor=300` cell and was manually terminated without a receipt. The
  matrix must use the existing direct-PID watchdog before it can produce
  authority evidence; no result from that attempt is promoted.
- Green scaffolding: `benchmarks/lbfgs_compile_root_cause_matrix.py` now runs
  the five declared CPU cells in independent direct-PID children, samples RSS,
  and records completed, timeout, RSS-limit, or failed outcomes. Its contract
  tests cover the cheap/coil, `maxcor=10/300`, short/long, and compile-only/warm
  axes. The child now writes an atomic `--progress-json` sidecar after each
  construction, lowering, compile, and solver-run checkpoint, and the parent
  preserves that sidecar when a watchdog kills the child. A pre-fix bounded
  quadratic rerun reached `new_x_state_start` before timing out at 20 seconds
  and 461,316 KiB RSS. The diagnostic now uses a shape-compatible initial
  state for re-entry, so that extra line-search execution is not counted in
  compile-shape measurements.
- Historical probe: the earlier source-owned Boozer custom route used a
  different inner root (`iota=-0.05134074584230428`) and is not parity
  evidence. Its receipt remains archived under
  `.artifacts/custom-quasi-newton/20260801T0720Z-cpu-boozer-custom/`.
- RED -> GREEN: the Boozer fixture had passed
  `constraint_weight=11.1232` to the JAX exact inner solve, unlike the native
  reference. The regression failed on the old fixture, then passed after the
  option was removed and a SIMSOPT-native `BoozerSurface` objective callback
  was added. The corrected initial inner state is
  `iota=-0.1924157185150927`, `G=14.035365807510038`.
- Corrected Boozer CPU callback probe (`maxiter=2`) records native/JAX initial
  objective `3.902843220850033e-4` versus
  `3.902843220850035e-4` (absolute difference
  `2.168404344971009e-19`) and initial gradient-infinity norms
  `3.8700625919864456e-3` versus `3.8700625919863463e-3` (difference
  `9.93129189996722e-17`). Native/custom cold/warm times were
  `2.257643948076293/2.2763357399962842 s` and
  `69.86358919506893/0.038925772067159414 s`; child RSS was
  `970348/3268868 KiB`. The two BFGS implementations took different capped
  endpoints (`2.707807798339409e-4` versus `3.279124574199254e-4`), so this
  receipt certifies initial objective/gradient alignment only, not endpoint or
  convergence parity.
- Probe: matched native SIMSOPT/simsoptpp `BiotSavart` + `SquaredFlux` versus
  custom JAX on source-owned coil47 with `maxiter=20`. The latest schema-5
  run reached native/custom `12/19` and `13/23` iterations/evaluations.
  Initial and final objective differences were `2.7755575615628914e-17` and
  `2.7755575615628914e-17`; maximum final-parameter difference was
  `4.149377871680293e-09`; final gradient-infinity-norm difference was
  `2.889977063078508e-09`. Native/custom cold and warm times were
  `1.0344/1.0469 s` and `6.3525/0.06247 s`; solver RSS deltas were
  `0/345948 KiB`. Receipt:
  `.artifacts/custom-quasi-newton/20260801T-coil47-native-custom-cpu-parity-maxiter20-schema5/`.
  This is matched native/JAX CPU objective and endpoint evidence, not GPU or
  performance-promotion evidence.
- Green: an isolated `.venv-qn-gpu` with JAX 0.10.0 CUDA 12 support sees
  `cuda:0`; the recorded GPU run passed the optimizer trajectory/step tests
  and source-owned fixture checks under FP64 `jax_gpu_parity`. Existing schema-5
  receipts record the
  JAX/XLA/SIMSOPT environment; the CPU environment remains separate.
- Green: source-owned coil47 custom L-BFGS on strict GPU, with
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, converged in 12 iterations / 15
  evaluations. Final objective was `0.13786263284430206`; versus native CPU,
  the absolute objective difference was `2.7755575615628914e-17`, maximum
  parameter difference `4.143977844033031e-09`, and gradient-infinity-norm
  difference `2.8879265835549204e-09`. Cold/warm time was `22.1284/0.08087 s`
  and solver RSS delta `536252 KiB`. The GPU fast-intent run also converged to
  the same objective (`22.2542/0.08540 s`, `535620 KiB` delta). Receipts are
  under `.artifacts/custom-quasi-newton/20260801T-coil47-custom-gpu-*`.
- Historical diagnostic: the pre-JIT Optax `coil47` strict-GPU 20-step run
  timed out at 120 seconds, and default preallocation reached CUDA allocation
  failures up to `23.52 GiB`. The current comparator uses an explicit JIT
  step and `XLA_PYTHON_CLIENT_PREALLOCATE=false`/`platform`; its bounded
  converged result is recorded above. Allocator settings remain part of the
  GPU receipt contract, not hidden setup.
- Open transfer-guard issue: a global `JAX_TRANSFER_GUARD=disallow` rerun
  failed 32/46 optimizer tests during fixture setup or scalar assertions
  (`jnp.asarray` host literals and implicit boolean conversion), before solver
  execution. The normal strict-GPU lane passes; the dedicated boundary lane
  needs explicit device-placed test inputs and host-result assertions.
- Open: matched converged Boozer endpoint, compile/device-memory and StableHLO
  accounting, and tracked promotion receipts. Provider-child RSS isolation is
  green for the synthetic runner. Four source-owned fixture tests marked
  `slow` pass in the required isolated CPU environment in 49.78 s combined.
  The current-HEAD direct Boozer outer-BFGS diagnostic now completes on CPU and
  strict RTX 5090 GPU at the two-step cap: initial objective difference is
  `2.168404344971009e-19`; custom CPU/GPU final-objective difference is
  `3.008661028647275e-17` and maximum parameter difference is
  `1.1934897514720433e-15`. Native and custom endpoints differ by
  `5.713167758598453e-05`, and all three are explicitly
  `status=1`/`iteration-limit`, not convergence. GPU cold/warm time was
  `144.7162/0.5821 s` with `1,630,208 KiB` solver RSS delta. Receipts:
  `docs/receipts/custom-quasi-newton/boozer-outer-bfgs-native-custom-cpu-maxiter2-current/`
  and `docs/receipts/custom-quasi-newton/boozer-outer-bfgs-gpu-maxiter2-current/`.
  An earlier unisolated selector exceeded 180 seconds and about 17 GiB RSS; it
  is invalid diagnostic evidence, not a current fixture result. Runner
  receipts now time fixture construction separately from the solver, and child
  stdout is discarded so large JSON output cannot deadlock the watchdog pipe.
- Diagnostic follow-up: the same current-HEAD Boozer fixture was run for 20 BFGS
  iterations on native CPU, custom CPU, and strict RTX 5090 GPU. All three hit the
  iteration limit (`status=1`, `success=false`). Native CPU ended at objective
  `3.2370187845523733e-05`; custom CPU/GPU ended at
  `2.83571085184401e-05`/`2.8357108518436739e-05`, with a custom CPU/GPU
  objective difference of `3.3610267347050637e-18` and maximum parameter
  difference `5.9396931817445875e-14`. Native versus custom objective differed
  by `4.0130793270869935e-06` (relative `12.3974546%`) at this capped endpoint.
  Cold/warm times were `5.6538/6.9635 s` native CPU,
  `69.9314/0.5852 s` custom CPU, and `146.4187/1.1891 s` custom GPU; custom
  solver RSS deltas were `0`, `2,048,680`, and `1,630,472 KiB`, respectively.
  Receipts: `docs/receipts/custom-quasi-newton/boozer-outer-bfgs-native-custom-cpu-maxiter20-current/`
  and `docs/receipts/custom-quasi-newton/boozer-outer-bfgs-gpu-maxiter20-current/`.
- Diagnostic follow-up at `maxiter=50` confirms this is not only a short cap:
  native stopped at the first outer step (`status=2`, `nfev=62`, objective
  `1000.0`), while custom reached the 50-step limit (`status=1`, objective
  `1.6400701660477423e-05`). Initial objective values remained matched to
  `2.17e-19`. The native inner-solve failure and JAX rejected-objective policy
  remain the unresolved parity boundary; no solver tolerance was weakened.
  Receipt:
  `docs/receipts/custom-quasi-newton/boozer-outer-bfgs-native-custom-cpu-maxiter50-20260802/`.
- RED -> GREEN diagnostic: the exact traceable Newton residual now crosses a
  nested JIT boundary and rematerializes intermediates with the FP64
  `nothing_saveable` policy. The focused exact-Newton selector is `8 passed,
  1 skipped`; the source-owned strict-GPU Boozer fixture passes in `126.55 s`
  with `2,631,308 KiB` peak RSS under a direct 180-second bound. This is
  fixture-construction evidence only; the capped outer BFGS pair is recorded
  separately above. StableHLO and device-memory receipts remain open. Receipt:
  `docs/receipts/custom-quasi-newton/boozer-fixture-gpu-current/`.
- Recorded checks (before the latest environment-schema test): 55 fast
  focused tests (48 core optimizer, trajectory, and result-schema tests plus
  seven runner/watchdog/measurement
  tests), plus four slow source-owned fixture tests
  passed. Direct BFGS edge
  and typed-callback checks, source-owned coil47/Boozer runner probes plus
  synthetic CPU BFGS/L-BFGS/Optax probes, the runner's Ruff check, formatting,
  compileall, and `git diff --check` pass. The targeted runtime/runner Ruff and
  formatting checks pass. The standalone typed step-runtime module passes
  Pyright with zero errors; a historical strict scoped configuration reported
  1,828 errors before the current annotation pass. The historical profile
  reported `793`; the current profile reports `797` because private optimizer
  dependencies and benchmark/test call sites are not yet typed. The project-wide
  Ruff command remains open under
  Ruff 0.16.1 (856 findings across
  the current source/test/benchmark tree); the touched Boozer source/test
  selector reports 84 findings. The isolated `.venv-qn-gpu` uses CUDA 12 on
  the local RTX 5090; the configured Landau A100 host remains unreachable
  (`No route to host`). The GPU lock is recorded in
  `benchmarks/environments/custom_quasi_newton_gpu.txt`. The GPU reruns add
  optimizer and source-owned fixture checks; their receipts are
  local working evidence only.
- Open diagnostic: a fresh-process `bfgs_quadratic` 47-variable pair produced
  no receipt; a native child remained around 300 MiB RSS for more than a minute
  before the probe was stopped. The same 47-variable custom call completed in
  about 0.56 s in an already-running process. Do not use the failed fresh
  process as a solver timing or parity result.
- Selected optimizer result/dispatch compatibility checks: 25 passed.
- Post-annotation reruns are green: the focused runtime/runner/trajectory
  selector passed 19 tests in 203.0 s; the four-test trajectory selector passed
  in 65.84 s; five Boozer private routing/callback/cache tests passed in 51.17 s.
  Touched-file Ruff and format checks are clean. A pre-annotation strict
  Pyright run reported 1,031 diagnostics; that historical profile reported
  793, while the current profile reports 798. These remain an open typing gate,
  not solver failures.
- Green: the full custom step-runtime contract file now passes 43/43 strict-CPU
  tests in 200.89 s, including the no-observer host-NumPy guard and the updated
  macro-step entry-point assertion.
- Green: the local candidate trajectory receipt checksum/byte-identity contract passes
  as an additional benchmark test; its manifest, metrics, script, and raw JSON
  are self-consistent.
- Green diagnostic: the pre-refactor BFGS implementation at `9ba1ad057` and
  the candidate produced byte-identical three-step FP64 Rosenbrock accepted
  states, counters, objective, gradient, parameters, and status. Receipt:
  `docs/receipts/custom-quasi-newton/bfgs-pre-refactor-trajectory/`; the
  candidate tree is dirty, so promotion closure remains open.
- Diagnostic: a fresh matched two-step CPU `coil47` run gave native/custom
  final objectives `0.13786469682070213/0.13786469682070215` (absolute
  difference `2.8e-17`), both status 1; custom warm time was `0.0794 s` versus
  native `0.5230 s`. Optax at the same two-step budget reached
  `0.13786469652455854`, warm time `7.9936 s`, and solver RSS delta `410400`
  KiB. The Optax 20-step child hit the 120-second watchdog; these are fixed-
  budget diagnostics, not convergence or promotion evidence.
- Green receipt packaging: the matched two-step CPU `coil47` native/custom/
  Optax measurements are packaged locally with historical schema-5 raw payloads, a hashed
  manifest, metrics, and a concise summary under
  `docs/receipts/custom-quasi-newton/coil47-fixed-budget-cpu/`; the receipt
  checksum check passes. It remains diagnostic because the candidate tree is
  dirty and the budget is capped before convergence.
- Historical runner diagnostic: a one-step strict-GPU Boozer child hit the
  120-second watchdog during fixture construction/traceable Boozer setup before
  writing a solver measurement. GPU allocation was only about 546 MiB; this is
  a fixture compile/build bottleneck, not evidence of dense-BFGS iteration
  memory. The direct slow-fixture probe now passes under the updated exact
  residual boundary; the runner endpoint remains open.
- Diagnostic-pass: a fresh current-head strict-GPU `coil47` two-step custom
  run matched native CPU to `2.2e-16` maximum parameter difference and
  `2.2e-17` final-gradient difference. Cold/warm times were `81.30/0.162 s`
  and solver RSS delta was `546292 KiB`; the local candidate receipt is
  `docs/receipts/custom-quasi-newton/coil47-fixed-budget-gpu-current/`.
  Cold time includes fixture setup and XLA compilation, so this is not a
  performance-promotion result; matched GPU Optax and A100 lanes remain open.
- Diagnostic-pass: a clean pre-refactor worktree at `c0dc94580` and the
  candidate implementation produced byte-identical three-step FP64 Rosenbrock
  JSON (`max_abs_difference=0`) for accepted states, counters, and status.
  The local candidate receipt is
  `docs/receipts/custom-quasi-newton/rosenbrock-pre-refactor-trajectory/`.
  The candidate checkout was dirty, so this is not promotion evidence.
- Green routing inventory: the public `Driver`/method mapping, target and
  reference entry points, `lbfgs_run_mode`, solver limits, callbacks, seeded
  gradients, result fields, and production callers are recorded in
  `docs/receipts/custom-quasi-newton/public-routing-inventory.md`. The strict
  CPU driver and compatibility-shim selectors passed 21/21 in 49.84 s; full
  application-scale endpoint parity remains separate.
- Green caller smoke: the VMEC-hybrid single-stage, Stage-II objective, and
  dynamic-surface caller selectors passed 22/22 in 32.68 s under strict CPU
  FP64. This confirms the current public construction and derivative contracts;
  full eager/traceable Boozer endpoint coverage remains open.
- Green trajectory rerun: the accepted-step SciPy/custom L-BFGS parity selector
  passed 4/4 in 74.91 s on strict CPU, including observer-equivalence and
  frozen accepted-state checks.
- Green observer contract: the no-observer, requested-trace, callback-stop,
  and cached-wrapper selectors passed 7/7 in 50.02 s. Together with the
  trajectory selector, this verifies that observational hooks do not alter the
  accepted path and that `StopIteration` freezes the expected prefix; the
  unbounded monolithic Boozer application path remains separate.
- Green bounded-state contract: the `maxcor`-shaped history, no-trace default,
  explicit bounded trace, and logical memory-accounting tests are green. The
  normal eager result retains no accepted-iterate trajectory; trace capture is
  opt-in and rejects an over-budget allocation.
- Diagnostic GPU split: the 41 non-slow runtime contracts pass in fresh
  processes as groups `18/18`, `17/17`, and `6/6` (58.04 s, 160.58 s, and
  72.80 s). A single-process run reached 35 completed tests and timed out as
  cumulative compilation grew to about 1.9 GiB RSS; this process-lifetime
  cache/compile issue is recorded separately from solver correctness. The two
  slow physics fixtures and full GPU application qualification remain open.
- Green/diagnostic slow fixtures: the strict-GPU `coil47` source fixture passed
  1/1 in 18.26 s. The strict-GPU traceable Boozer fixture hit the 120-second
  watchdog during construction/lowering with about 2.76 GiB child RSS and no
  result payload; this is the same Boozer compile bottleneck seen in the runner,
  so Boozer GPU qualification remains open.
- The focused public compatibility selector (with `MPI4PY_RC_INITIALIZE=0`)
  passed 27 tests in 22.36 seconds; 133 unrelated tests were deselected.
- Public private-optimizer CPU checks: the Boozer private on-device selector
  passed 35 tests with one GPU-only closure test skipped; the broader public
  on-device selector passed 53 tests with seven GPU-only skips. Seven focused
  traceable exact/LM callable-cache tests passed after the cache fix. The full
  project compatibility suite remains open: the unbounded `traceable` selector
  was stopped at about 17.7 GiB RSS before completion.
- Green: all 13 legacy compatibility-shim tests pass when host-reference
  methods explicitly enter the `native_cpu` lane; on-device and traceable shim
  methods remain on the JAX lane. The prior RED was the validation process
  forcing `jax_cpu_parity` onto intentionally host-side legacy methods.
- Green: direct `target_minimize` coverage passed 12 tests; stage-two objective
  and dynamic-surface coverage passed 13 tests; and the planar, VMEC-free
  Boozer, and VMEC-hybrid example contract tests passed 16 tests on strict CPU.
  These are caller/contract checks, not matched native endpoint evidence.
- Green: the dedicated `_lbfgsb_scipy` transition/kernel compatibility suite
  passed 61/61 strict-CPU tests in 4m38s, including reverse-communication
  status ordering and result-schema coverage. End-to-end application parity is
  still a separate open gate.
- Green: the post-rematerialization private compatibility selector passed
  30/30 strict-CPU tests, and the combined eager-runtime plus trajectory
  selector passed 47/47 in 244.91 s. These cover direct target routing,
  callback-stop/status mapping, inverse-Hessian results, no-SciPy entry, and
  accepted-step trajectory checks after the exact-Newton compile change.
- Green: the current strict-CPU application selector passed 32/32 across
  Stage-II objectives, dynamic surfaces, finite-build/planar/stochastic
  examples, and both Boozer vacuum and VMEC-hybrid single-stage contracts.
  This extends caller construction coverage only; matched converged Boozer
  endpoint, strict-GPU, and performance receipts remain open.
- Green: the current strict-GPU non-slow runtime selector passed 41/41 in
  257.21 s with FP64, CUDA-only execution, and the platform allocator. The
  slow source-owned Boozer fixture remains separately bounded because its
  exact-Newton compile dominates the process; no GPU fallback was enabled.
- Green: the direct BoozerSurface target/run-code selector passed 21/21 on
  strict CPU, covering public target routing, on-device LM/BFGS paths,
  backend rejection, and run-code construction. This does not close the
  matched Boozer endpoint or strict-GPU outer-BFGS gate.
- Diagnostic boundary: the current strict-GPU Boozer outer-BFGS runner was
  attempted with a one-step cap and still hit the declared 120-second child
  watchdog before a solver measurement was emitted. This confirms that the
  remaining Boozer gate is whole outer-objective compilation, not a missing
  iteration budget; the failed child is not promotion evidence.
- Green: strict-CPU end-to-end application parity passed for stage-two minimal
  (1/1), stage-two standard (1/1), and VMEC-free Boozer single-stage (2/2,
  including the executable parity-contract check). These tests compare native
  and JAX CPU inputs, construction fingerprints, initial observables, and
  bounded final outcomes; they are not GPU or performance receipts.
- Green: the remaining Stage-II parity cases passed on strict CPU: finite-build
  (2/2), planar coils (2/2), and stochastic (1/1). Across these mirrors,
  native/JAX CPU construction and bounded outcome contracts are green; strict
  GPU, wall-time, RSS, and device-memory qualification remain open.
- Probe: fixed-budget CPU `coil47` comparison (`maxiter=20`) reached the same
  final objective `0.137862632844302` for native, custom, and Optax. Native /
  custom / Optax iterations were `12 / 13 / 16`; cold seconds were
  `0.970634 / 5.692719 / 24.275724`; warm seconds were
  `0.766777 / 0.050979 / 23.552837`; child RSS was
  `460844 / 1010952 / 2375072 KiB`. Maximum final-parameter differences from
  native were `3.6456e-9` (custom) and `1.3207e-4` (Optax). This is dirty-tree
  exploratory CPU evidence, not the A100 promotion receipt.
- Green diagnostic receipt: a fresh current-head native/custom `coil47` CPU
  run with a 20-step cap converged on both providers (`12/13` iterations,
  `15/44` evaluations, status 0). Final objective difference was zero at the
  recorded precision; maximum final-parameter difference was `8.10e-15`.
  Raw JSON, metrics, summary, and checksummed manifest are present locally under
  `docs/receipts/custom-quasi-newton/coil47-native-custom-cpu-maxiter20-current/`.
  The dirty checkout keeps it diagnostic, and strict-GPU/A100 promotion lanes
  remain open.
- Green diagnostic receipt: the matching strict-GPU custom `coil47` run
  converged in `12/15` iterations/evaluations with status 0. Relative to the
  native CPU reference, final objective, gradient-infinity-norm, and parameter
  differences were `5.55e-17`, `4.61e-18`, and `2.94e-15`. Cold/warm solver
  time was `75.169/0.354 s`; solver RSS delta was `509964 KiB`. The local candidate
  manifest is under
  `docs/receipts/custom-quasi-newton/coil47-custom-gpu-maxiter20-current/`;
  the dirty checkout and missing Optax/A100 lanes keep it diagnostic.
- Green diagnostic receipt: a matched strict-GPU two-step `coil47` comparison
  now includes custom JAX and Optax. Warm time was `0.162 s` custom versus
  `21.338 s` Optax; final objective difference was `2.96e-10` absolute. Both
  endpoints were capped and non-converged, so the checksummed receipt under
  `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-fixed-budget-current/`
  is fixed-budget evidence only.
- Green diagnostic receipt: the strict-GPU `coil47` step-from-start kernel
  lowered in `2.453 s` with `1,504,879` StableHLO bytes and `1,645,584` JAXPR
  bytes. The local candidate lowering receipt is under
  `docs/receipts/custom-quasi-newton/coil47-lbfgs-gpu-compile-shape-current/`;
  runtime executable count and device memory are still unmeasured.
- Probe: the same CPU `coil47` custom/Optax run in explicit `fast` intent reached
  the same final objective; custom / Optax cold seconds were `5.451687 /
  21.986368`, warm seconds `0.066188 / 21.502429`, and child RSS
  `989340 / 2171908 KiB`. This confirms the fast lane is executable, but is
  still dirty-tree CPU evidence without native or GPU comparison.
- Green: schema-4 transfer-audit rerun of the matched CPU `coil47` case kept
  the native/custom/Optax final objective at `0.137862632844302` (custom
  max-parameter difference from native `1.0232e-9`). The custom warm audit
  recorded `0` initialization, `26` advance, and `15` final-result transfer
  calls (`156` and `8,700` bytes for the latter two); native and Optax rows
  intentionally report empty custom-ledger fields. This remains CPU evidence
  on a dirty tree, not a GPU promotion receipt.
- Diagnostic: a fresh matched CPU Boozer run at `maxiter=100` did not close an
  endpoint parity gate. Native SciPy BFGS stopped after one failed inner-solve
  evaluation with `status=2` and objective `1000.0`; custom BFGS consumed all
  100 iterations with `status=1` and objective `8.469027533302147e-6`. Initial
  objective values still differed by only `2.168404344971009e-19`. The
  native failure penalty and the JAX rejected-objective policy are therefore a
  confirmed trajectory-semantics mismatch, not convergence evidence.
- Controlled follow-up: a fresh native-only Boozer run with the same fixture
  reached 50 iterations and 54 evaluations without an inner-solve failure;
  its endpoint was `1.6644268825804663e-5` with gradient infinity norm
  `1.40605699279515e-4`. This shows the failure is trajectory- and
  horizon-dependent rather than an immediate fixture defect. The earlier
  `maxiter=100` failure remains a separate long-horizon diagnostic; it does
  not justify changing the native oracle or declaring endpoint parity.

Review gate: not promotion-ready. The runner still needs matched converged
Boozer endpoint evidence, strict-GPU results, compile/device-memory and StableHLO
receipts, and a clean tracked manifest. The default eager L-BFGS path is the
existing three-kernel
design-B facade; the
generic accepted-step helper is not yet the production route. These are open
qualification items, not parity evidence. Historical RED revisions were not
preserved; current test files are post-hoc green evidence.

</details>

## Purpose

Refactor SIMSOPT's custom JAX BFGS and L-BFGS around the useful part of the
Optax structure: immutable state, fixed-shape compiled transitions, and a small
host driver for ordinary eager solves. SIMSOPT continues to own the algorithms,
SciPy-compatible behavior, and public results. Optax remains an explicit,
optional comparator; the custom solvers must not call Optax internals.

Optax's [official interface](https://github.com/google-deepmind/optax/blob/main/docs/getting_started.ipynb)
is a composable `GradientTransformation` with `init`/`update`; its
[`scale_by_lbfgs`](https://github.com/google-deepmind/optax/blob/main/examples/lbfgs.ipynb)
is a gradient transform, not a SciPy-compatible L-BFGS-B
result/callback/status implementation. It therefore remains an explicit
comparator in this plan.

This plan supersedes the earlier proposal to remove custom BFGS/L-BFGS after
Optax qualification. The architecture/routing delivery must keep
`docs/jax_solver_algorithm_matrix.md` and
`docs/jax_solver_provider_coexistence_implementation_plan.md` aligned. No
custom-provider removal or default-provider change is allowed until this plan's
compatibility, science, and performance gates pass.

## Goals

- [x] Preserve `bfgs-ondevice` and `lbfgs-ondevice` method names, options,
      callbacks, statuses, counters, result fields, and SciPy parity behavior.
- [x] Use a fixed-shape accepted-step interface for normal eager solves, with
      total budgets outside step compilation.
- [x] Retain a whole-solve JAX route for callers that execute the optimizer
      under `jax.jit`; a Python host loop cannot consume traced optimizer state.
- [x] Keep line search on device with bounded JAX control flow.
- [x] Allocate only current solver state and bounded history during normal
      execution; retain no full trajectory unless requested.
- [x] Keep JAX fast intent as the default after a JAX device is selected;
      parity remains explicit.
- [ ] For every remaining change and newly found defect, preserve authentic
      RED -> GREEN -> REFACTOR evidence and durable numerical, compile, timing,
      RSS, and device-memory receipts. Existing post-hoc tests remain labeled
      post-hoc; this plan does not manufacture historical RED revisions.

## Non-goals

- [ ] Do not replace the custom solvers with `optax.lbfgs`.
- [ ] Do not require Optax for custom BFGS/L-BFGS.
- [ ] Do not promise identical Optax, custom JAX, and SciPy trajectories.
- [ ] Do not add new constrained optimization behavior.
- [ ] Do not remove or deprecate the traced whole-solve route in this change.
- [ ] Do not hide objective compile or memory costs inside optimizer claims.
- [ ] Do not create a large benchmark matrix or fabricate historical RED
      revisions.
- [ ] Do not add an automatic dense-BFGS routing threshold or silently switch
      algorithms.

## Current facts and evidence limits

- `_bfgs.py` uses a fixed-shape eager step plus host driver for concrete inputs;
  the traced route still compiles the full solve with `lax.while_loop`, keeps
  staged `maxiter` in its cache identity, and uses the existing callback path.
- `_lbfgs.py` defaults to a host-observed driver over `start`, `search`, and
  `new_x_reentry` kernels. The eager dynamic-limit variants exclude
  `maxiter`/`maxfun` from their cache identity; the traced/static variants keep
  staged limits in their compilation contract.
- `BoozerSurfaceJAX.run_code_traceable()` and the traceable single-stage
  objective invoke whole-solve BFGS/L-BFGS inside a larger JIT. L-BFGS
  `monolithic_debug` is therefore a live compatibility route, not dead debug
  code.
- Before this refactor, L-BFGS history had shape `min(maxcor, maxiter)`. The
  eager path now allocates exactly `maxcor` slots; the old/new bytes and
  traced/static compatibility still require qualification below.
- BFGS inverse-Hessian state is `O(n^2)`. L-BFGS history is
  `O(n * maxcor)`, while the SciPy-compatible workspace also contains an
  `O(maxcor^2)` term (`2mn + 5n + 11m^2 + 8m` floating slots). Logical state,
  compiler/allocator amplification, and peak live bytes must be reported
  separately.
- `src/simsopt_jax/geo/optimizers/optimizer.py` owns legacy
  `target_minimize` callback/result conversion;
  `src/simsopt_jax/solve/dispatch.py` owns the typed solve API's callback,
  timing, and result normalization. `src/simsopt_jax/runtime/host_boundary.py`
  owns explicit device-to-host materialization. A private step runtime must not
  duplicate these owners.

The prior A100 observations—about 36 seconds for the custom 10-step probe,
about 12 seconds for the matched Optax probe, and roughly 34--36 GiB host RSS
during a stopped custom long run—were not archived with a durable raw result.
They are exploratory session notes, not a baseline or promotion evidence. Phase
0 must reproduce the failure from a clean revision before it influences design
or thresholds.

## Design

### Two execution paths, one algorithm

Each algorithm retains one mathematical transition implementation and exposes
two execution paths:

1. **Eager host-stepped path:** fixed-shape compiled transition plus a Python
   driver. This is the normal `bfgs-ondevice`/`lbfgs-ondevice` path when state is
   not traced.
2. **Traceable whole-solve path:** JAX control flow around the same transition
   primitives. This remains supported for `run_code_traceable()` and callers
   under `jax.jit`.

The route is selected from tracing context and the existing explicit
`lbfgs_run_mode` contract. No new public mode or compatibility flag is added.
Removal or deprecation of the whole-solve route requires a separate Tier-3
proposal with caller migration, warnings, a release timeline, and rollback.

### Ownership

Add `src/simsopt_jax/geo/optimizers/private/_step_runtime.py` with only:

```text
StepOps[
    StateT, TransitionT, ObservationT, HostObservationT, PayloadT, InitialT
] = immutable typed callable bundle
TransitionSink[HostObservationT, PayloadT] -> ContinueDecision
run_eager(ops, x0, limits, sink: TransitionSink[...]) -> StateT
BFGSStepOps: StepOps[
    _BFGSResults, _BFGSTransition, _BFGSObservation,
    _BFGSHostObservation, _BFGSResults, _BFGSResults
]
LBFGSBStepOps: typed facade over the specialized entry kernels
```

- `StepOps` has typed `initialize`, `advance`, `observe`,
  `host_observation`, and `payload` callables. The current BFGS eager path uses
  this generic driver. With physical design B, the current L-BFGS-B eager path
  remains a typed facade over the three specialized entry kernels rather than
  a merged branch graph; it is not yet routed through `run_eager`. The generic
  driver receives its bundle explicitly and performs no algorithm-tag dispatch.
- `TransitionSink` is supplied by the existing public owner. It receives the
  packed observation and, when requested, the typed payload, then returns the
  typed decision `CONTINUE` or `STOP`. A null sink requests no payload. The
  private driver never stores a callback in `StepOps` or in a compiled cache.
  `optimizer.py`/`dispatch.py` remain responsible for adapting callbacks and
  `StopIteration` into the sink decision.
- The BFGS state (`_BFGSResults`) and L-BFGS-B state
  (`_lbfgsb_scipy.LbfgsbState`) remain separate immutable pytrees.
- Algorithms own budget-transition timing, line search, status, and counters.
  The shared driver must not impose a generic zero-budget policy.
- Target eager contract: the no-observer path packs all stop/status/counter
  scalars into one fixed observation pytree and performs exactly one explicit
  `device_get` per eager `advance`, including terminal or nonaccepted
  transitions. Initialization and final packaging are counted separately. The
  observer path may add one typed callback/trace payload transfer containing
  `x`, `f`, gradient, counters, and status; its transfer count and bytes are
  recorded separately. Current BFGS and stepwise L-BFGS observer paths satisfy
  the packed-observation and phase-ledger contract; monolithic compatibility
  callbacks remain separately qualified.
- `geo/optimizers/optimizer.py` remains the compatibility owner for direct
  `target_minimize`; `solve/dispatch.py` remains the typed API owner. They keep
  callback conversion, `StopIteration`, timing, and public result
  normalization out of the private runtime.
- `runtime/host_boundary.py` remains the sole device-to-host boundary owner.
- State is per-run and immutable. Each objective keeps its own lock-protected
  LRU with capacity 8. The current implementation uses immutable tuple keys;
  the following is the target key contract and remains subject to the open
  collision tests:
  - common fields: algorithm, objective mode, dtype, flat parameter shape,
    structured cache token, pytree-adapter cache identity, closure-constant
    signature, and callback/trace compile policy;
  - BFGS fields: `value_and_grad`, norm, line-search bound, `gtol`, and `xrtol`;
  - L-BFGS-B fields: `maxcor`, `ftol`, `gtol`, `maxls`, bounds signature, and
    seeded-value/gradient compile policy.
  `maxiter` and `maxfun` are excluded only from the eager key because they are
  dynamic there. The current L-BFGS key does not yet include every listed
  semantic field; same-owner closure mutation and callback/trace-policy
  collision tests remain open. No global strong reference retains the
  objective or its closures. A per-key pending
  entry makes first compilation single-flight; failure removes the pending
  entry, and eviction never invalidates an executable already held by a solve.

### Physical compile boundary: decide by measurement

The public design requires one logical accepted-step operation. Two physical
implementations are allowed:

- **A:** one JIT containing reverse-communication branches; or
- **B:** a typed facade over the existing specialized entry kernels.

Phase 0 compares StableHLO size, compile time, peak RSS, and warm step time on a
cheap fixture and the coil objective. Choose the smaller implementation and
record the decision. Do not merge kernels merely for structural symmetry: a
unified branch graph can compile more code and consume more memory.

For the eager path, `maxiter` and `maxfun` are dynamic host/runtime limits.
The traceable whole-solve path may still compile for its staged loop bound.
`maxcor`, dtype, parameter shape, bounds shape, and objective closure structure
remain valid compilation identities because they change state shape or
generated code.

### Observable behavior and compatibility matrix

This Tier-3 change is complete only when each observable delta below has a
direct test or receipt. Unlisted public behavior must remain unchanged.

| Surface | Before | Intended result | Required proof |
| --- | --- | --- | --- |
| Eager BFGS execution | Whole-solve compiled loop | Fixed-shape compiled step with a host driver | Frozen accepted states, status/counters, callback order, and compile-count tests |
| Eager L-BFGS execution | Static-budget compiled wrappers | Specialized fixed-shape design-B kernels with dynamic total budgets | SciPy trajectory parity, budget-change compile count, and inverse-Hessian tests |
| Traced BFGS/L-BFGS | Whole-solve JAX control flow | Supported whole-solve route using the same algorithm-owned transitions | Direct traced and Boozer `run_code_traceable()` tests |
| Observation | Legacy callback/trace boundaries | One packed host observation per eager transition; payload only when requested | Transfer-ledger and observer-equivalence tests |
| Wrapper cache | Objective-attached unbounded mapping | Per-objective capacity-8 LRU with single-flight construction | Identity, collision, eviction, failure, lifetime, and concurrency tests |
| L-BFGS history | `min(maxcor, maxiter)` slots | Exactly `maxcor` slots | State-shape and logical/peak-memory receipts |
| Public API | Existing method names, options, routing, results, and statuses | No public migration or deprecation | Routing inventory plus direct public compatibility tests |

Rollback must remove every implementation commit that changes active public
solver behavior unless an old/new implementation gate is added and proven at
the public seam. Frozen tests and receipts remain. No persisted-state migration
is required.

### Runtime modes and providers

These are orthogonal axes:

| Axis | Values | Meaning |
|---|---|---|
| Device | CPU, GPU | Selected JAX device |
| Execution intent | fast, parity | Existing runtime policy; fast is the JAX default |
| Provider | custom SIMSOPT, Optax, SciPy | Algorithm implementation; Optax is explicit, SciPy is the CPU oracle |

Optax is never a third execution intent and is never silently selected for a
custom method name.

## Implementation

### Phase 0 — provenance, root cause, and RED

- [x] Work from an isolated worktree at a recorded commit. Record source
      status, Python/JAX/JAXLIB/SciPy/Optax versions, device, FP64 state,
      options, commands, exit codes, and fixture hashes. The clean candidate
      receipts bind these fields to commit `3b2b9f40a` and the CPU/GPU locks;
      current-head `validate-all` revalidates the 39 tracked manifests and
      archive checksums.
- [x] Add the reviewed build-requirement input and relocatable hash-locked CPU/
      GPU dependency files described under Supported environments. The CPU
      and GPU locks resolve with `uv pip sync --dry-run --require-hashes`; the
      editable checkout path is excluded. Full clean-environment replay remains
      a separate promotion gate.
- [x] Add the versioned runner
      `benchmarks/custom_quasi_newton_runtime.py` for deterministic quadratic,
      Rosenbrock, coil47, and Boozer fixtures, with synchronization,
      child-process measurements, and a JSON schema. The physics cases remain
      endpoint/promotion evidence only after matched native/C++ endpoint
      contracts close; coil47 and Boozer now expose matched SIMSOPT-native
      objective callbacks for initial-state checks.
- [x] Add `benchmarks/fixtures/custom_quasi_newton.py` and
      `benchmarks/fixtures/custom_quasi_newton_cases.json` as the current
      runtime-contract fixture SSOT. The source-owned physics builders are
      present; full endpoint and device qualification remain open:
  - [x] `coil47` is a deterministic, VMEC-free coil preoptimization slice
        derived from the curve/current construction in
        `examples/jax/3_Advanced/single_stage_optimization.py`; its frozen
        analytic surface keeps the fixture independent of VMEC and mutable
        files. The generator asserts 47 free variables and records geometry,
        quadrature, current-coordinate, objective, FP64, and seed metadata.
        Its certificate includes matched native/JAX objective parity on CPU;
        the custom GPU endpoint is now recorded; Optax and full performance
        qualification remain open.
  - [x] `boozer` is the deterministic vacuum case derived from
        `examples/jax/3_Advanced/single_stage_boozer_vacuum_optimization.py`;
        it records surface/field construction, exercises the compiled
        `run_code_traceable()` route, and exposes a matched native
        `BoozerSurface` objective callback. The fixture selects
        `optimizer_backend="ondevice"` explicitly. The host-controlled eager
        `run_code()` route is intentionally not invoked by this strict fixture:
        strict JAX rejects that fallback and its compatibility tests remain a
        separate gate.
  - [x] The runner JSON records generator/source hashes, full initial vectors,
        solver option vectors, expected initial observables, final certificate
        fields, and predeclared tolerances. Neither fixture may read VMEC, a
        network path, or mutable user data.
  - [x] Add the focused runtime and SciPy trajectory tests:
  - [x] `tests/jax/solve/test_custom_quasi_newton_step_runtime.py` (43 tests;
        41 fast contracts and two `slow` source-owned physics probes);
  - [x] `tests/jax/solve/test_lbfgsb_trajectory_parity.py` (accepted-step,
        frozen FP64 fields, deferred-`maxfun`, and observer-equivalence tests).
        These are current-worktree evidence until committed;
        they are not tracked promotion receipts.
  - [x] `tests/benchmarks/test_custom_quasi_newton_runtime.py` (40 collected
        cases: 38 non-slow provider-child, provenance, watchdog, receipt,
        memory, intent, prepared-program, and phase-RSS contracts; two slow
        fixture cases).
  - [x] focused additions/updates to
        `tests/geo/test_boozersurface_jax_private.py` and
        `tests/geo/test_boozersurface_jax.py`, including the traceable closure
        cache and signature-contract regressions.
- [x] Freeze accepted-step behavior for FP64 quadratic and Rosenbrock cases:
      `x`, objective, gradient, step length, line-search task/status,
      iterations, `nfev`, and `njev` are pinned in the focused BFGS/L-BFGS
      tests. These are tolerance-based frozen contracts, not GPU bitwise claims.
- [x] Add compile-count regression tests: after one warm solve, changing only
      `maxiter` or `maxfun` creates no new eager step executable. The tests
      observe compilation behavior without pinning private cache-key spelling.
- [x] Pin separate zero-budget semantics: BFGS `maxiter=0` takes no step;
      L-BFGS-B preserves SciPy's deferred stop check and accepts one `NEW_X`
      before reporting the limit. `test_zero_budget_preserves_bfgs_and_lbfgs_limit_timing`
      covers both contracts.
- [x] Run the five-cell root-cause matrix: cheap versus coil objective,
      `maxcor=10` versus `300`, short versus long budget, and compile-only
      versus warm execution. Record iteration progress, compile events,
      per-step timing, objective-only timing, an RSS time series, StableHLO
      size, and executable count. Do not infer a compile defect merely from a
      long elapsed time.
      Use `benchmarks/lbfgs_ondevice_compile_shape.py`; its default diagnostic
      excludes the legacy generic/monolithic kernels. Add
      `--include-legacy-kernels` only under an externally watched process.
- [x] Preserve partial matrix evidence with atomic progress sidecars. A killed
      child must leave its last completed phase and the parent RSS time series;
      a missing final payload is recorded as incomplete, not as a solver result.
- [x] Record the bounded diagnostic that provisionally selects physical design
      B: design A exceeded the watchdog while design B completed on the matched
      quadratic.
- [ ] Complete the promotion-grade A/B comparison for StableHLO size, compile
      time, peak RSS, and warm step time on both the cheap and coil fixtures.
      If the objective graph—not optimizer control/state—is dominant, stop this
      refactor claim and open an objective-specific plan.
- [x] Bound each diagnostic in an exact child process. The runner sends TERM,
      then KILL after a grace period, at 120 seconds or 8 GiB RSS, and tracks
      the child PID directly; no broad process-name matching is used. The
      timeout and RSS branches have bounded contract tests.
- [x] Add one deterministic receipt publisher and aggregate validator. It builds
      `manifest.json`, `metrics.json`, and `summary.md` from runner output,
      binds environment locks and every artifact by SHA-256, copies raw evidence
      to the declared archive, and fails closed when any tracked or archived byte
      is absent or different. Tamper and fresh-process replay tests pass.

  A direct bounded design probe is also recorded under
  `.artifacts/lbfgs-ondevice/compile-design-ab/`: the specialized transition
  lowered in 2.37 s with 1,180,046 bytes of StableHLO, while the old generic
  transition stopped during lowering without a final payload. This is useful
  root-cause evidence and supports the provisional choice of B, but it is not
  a completed promotion-grade A/B receipt.

### Phase 1 — shared eager driver

- [x] Implement the typed private protocol without `Any`, dynamic imports,
      mutable dictionaries, runtime algorithm tags, or public-result logic.
- [x] Keep runtime limits immutable host data. Where L-BFGS-B needs limits in a
      compiled transition, pass total limits dynamically and preserve SciPy's
      post-`NEW_X` checks. Never truncate an in-progress line search to prevent
      `maxfun` overshoot.
- [x] Implement separate scalar-only and callback/trace observation paths.
- [x] Complete the transfer audit: assert one packed scalar transfer per eager
      `advance`, including terminal and nonaccepted transitions, and report
      separate initialization, callback/trace, and final-result transfers.
      The context-local host-boundary ledger is emitted in runner schema 5;
      tests cover full eager L-BFGS transitions with and without callbacks,
      terminal/nonaccepted packets, and BFGS observer-phase attribution.
- [x] Cover initial convergence, zero budgets, evaluation exhaustion,
      nonfinite state, callback `StopIteration`, status mapping, and concurrent
      independent solves with no shared mutable state or crossed callbacks.
      Initial-convergence, zero-budget, and L-BFGS-B evaluation-exhaustion
      behavior are covered for both public and private paths; two-thread BFGS
      and L-BFGS isolation tests cover independent callback ownership.
- [x] Prove the null sink and every non-stopping observer produce the same
      trajectory. For a stopping observer, prove the accepted prefix, callback
      order, status, counters, and stop point match the frozen contract.
      Current tests cover non-stopping BFGS/L-BFGS-B and multi-step callback
      order, accepted-prefix stopping, and concurrent callback ownership.

### Phase 2 — BFGS

- [x] Extract immutable initialization and one direction/line-search/curvature
      transition from `_minimize_bfgs_private`.
- [x] Preserve strong-Wolfe constants, lower-precision decreasing fallback,
      curvature validation, `gtol`, `xrtol`, line-search status, counters, and
      final re-evaluation; the focused private compatibility selectors and
      frozen contracts cover these fields.
- [x] Route ordinary eager calls through the host driver; retain the traceable
      whole-solve route using the same transition primitives.
- [x] Report dense inverse-Hessian logical bytes and a conservative derived
      peak-live upper bound during the update, including simultaneous old/new
      Hessians and intermediates. Record the no-donation policy. The runner
      emits this accounting without adding a routing threshold.
- [x] Measure compiled dense-BFGS update peak live bytes on supported CPU/GPU
      devices and compare them with the derived logical upper bound. Keep the
      backend temporary allocation separate from the Hessian-state accounting;
      this does not replace a future allocator timeline if one is needed.

### Phase 3 — L-BFGS-B

- [x] Preserve `_lbfgsb_scipy.py` transition equations, reverse-communication
      task/status ordering, callback order, counters, `ftol`, `gtol`, `maxls`,
      seeded value/gradient, and inverse-Hessian extraction.
- [x] Implement the logical accepted-step operation using the Phase-0-selected
      physical compile design.
- [x] Remove `maxiter`/`maxfun` from eager init/step/result static closures and
      cache identities without changing algorithm-owned stop timing.
- [x] Allocate exactly `maxcor` history slots, independent of `maxiter`; update
      the pinned state-shape test and report the old/new byte difference for
      `maxiter < maxcor`.
- [x] Replace Python-unrolled history-update and two-loop algebra with
      JAX loop/vectorized updates. Fresh `maxcor=300` CPU/GPU probes now stay
      within the bounded compile/RSS envelope; normal accepted-state tests
      remain green.
- [x] Retain no accepted iterates normally. Keep explicit trace capture within
      its existing byte cap; the no-trace and oversized-trace tests cover both
      paths.
- [x] Preserve `monolithic_debug` and traced `run_code_traceable()` behavior.
- [x] Replace the objective-attached unbounded compiled-wrapper dictionary with
      the per-objective capacity-8 LRU and single-flight admission specified
      above.
- [x] Complete cache qualification: test key identity, owner garbage
      collection, eviction while another solve holds an entry, failed-
      compilation cleanup, cardinality under shape/policy churn, concurrent
      single-flight first use, every initial-state semantic field, and
      same-owner structured-cache-token mutation. Structured adapters are
      exercised through distinct tuple/list kernels and their outputs are
      checked, so tree-definition collisions are not only compared as keys.

### Phase 4 — public compatibility and application tests

- [x] Keep public method names and provider routing unchanged.
- [x] Inventory all callers of `lbfgs_run_mode`, `maxcor`, `maxfun`, callbacks,
      seeded gradients, `hess_inv`, status, and counters.
- [x] Preserve the direct custom L-BFGS inverse-Hessian operator through the
      typed `Driver.SIMSOPT_LBFGSB` result. `OptimizerResult.hess_inv` now
      carries the SciPy-compatible operator without densifying it; the public
      dispatch regression verifies identity and application semantics.
- [ ] Complete direct `target_minimize`, BoozerSurface eager and traceable
      paths, stage-two, and single-stage caller coverage. Contract smoke
      selectors are green; full eager/traceable and application-scale closure
      remains open.
- [x] Legacy `jax_minimize`/`jax_least_squares` shim routes are covered for
      host-reference, on-device, trace, and Optax/Optimistix methods; host
      reference tests select `native_cpu` explicitly.
- [x] Direct target, stage-two objective, and single-stage example contract
      selectors pass in the isolated strict CPU lane; full application-scale
      endpoint parity remains open.
- [ ] Compare every accepted state against the pre-refactor custom solver.
      The current synthetic three-step BFGS and L-BFGS receipts are explicitly
      diagnostic and do not cover the full application matrix.
- [x] Compare the pinned SciPy L-BFGS-B accepted-step trajectory with matched
      options. `tests/jax/solve/test_lbfgsb_trajectory_parity.py` covers the
      accepted states, counters, deferred `maxfun`, and observer-equivalence
      contracts; bitwise, tolerance, and equivalent-endpoint verdicts remain
      separate where applicable.
- [x] Verify non-stopping callback/no-callback and trace/no-trace configurations
      do not change numerical results. Verify `StopIteration` produces the
      intended frozen trajectory prefix and terminal result.
- [ ] Verify eager transition/math kernels and no-observer paths import or
      execute no SciPy, Optax, Optimistix, NumPy numerical work, or host
      callback. For supported callbacks inside the traced whole-solve route,
      retain `jax.debug.callback` only at the accepted-step observation
      boundary and freeze its current ordering flag, payload, callback order,
      status, and counter behavior.
  - [x] The focused no-SciPy/stepwise/unconstrained selectors passed on strict
        CPU (`5/5`) and strict CUDA (`4/4`); the CUDA lane also passed the
        explicit closure-constant placement and transfer-guard checks. This
        closes the selected optimizer boundary paths; the full application
        matrix remains a separate gate.

### Phase 5 — lean physics and performance qualification

- [ ] Qualify L-BFGS on the runner's 47-parameter coil case and BFGS on its
      representative Boozer case using native CPU, JAX CPU, and strict JAX GPU.
- [x] Strict-CPU application parity smoke is green for stage-two minimal,
      stage-two standard, and VMEC-free Boozer single-stage; the custom GPU
      runner lane for the 47-parameter coil is also green, while the combined
      Boozer/performance qualification remains open.
- [x] The clean-candidate VMEC-free Boozer fixture now has matched initial
      native/JAX objective and gradient evidence on strict CPU and RTX 5090
      CUDA. Objective absolute differences are `2.1684e-19` and `3.7947e-19`,
      and maximum gradient differences are `7.429e-16` and `2.085e-15`;
      receipt: `docs/receipts/custom-quasi-newton/boozer-initial-parity-20260802.md`.
      This does not close the outer endpoint/convergence gate.
- [x] Strict-CPU application parity is also green for finite-build, planar,
      and stochastic Stage-II mirrors; these are outcome-contract evidence,
      not performance receipts.
- [ ] Compare initial and final objective components, parameters, invariant
      geometry observables, gradient infinity norm, constraints, iterations,
      evaluations, raw status, and stopping reason.
- [x] Label capped, converged, failed, and callback-stopped states directly.
      A finite decrease or lower objective alone is not convergence; the
      runner's `stopping_reason` contract covers these terminal classes.
- [x] The runner now emits an explicit `stopping_reason` beside status and
      success, with RED/GREEN coverage for convergence, iteration limits,
      line-search failure, nonfinite termination, callback stop, and Optax
      failure. Existing receipts remain diagnostic until regenerated on a
      clean checkout.
- [ ] Measure cold compile, warm optimizer, total wall time, peak RSS, peak
      device memory, StableHLO size, and executable count. Synchronize timed
      boundaries and exclude export/plotting.
- [x] Repair the measurement boundary before using any performance result for
      promotion. The clean strict-RTX-5090 five-sample receipt now exercises
      the repaired boundary; A100 and promotion-threshold qualification remain
      separate gates:
  - [x] expose a prepared-provider interface for the flat runner fixtures:
        custom uses production `prepare_lbfgs_private`/
        `PreparedLBFGS`, while Optax constructs its equivalent prepared step
        once outside warm timing. Both providers reuse fixed-shape programs for
        timed runs, with exact input-binding tests. General structured-adapter
        qualification remains outside this bounded comparator.
  - [x] run dense-BFGS buffer analysis in a named
        `algorithm_memory_analysis` phase so compilation instrumentation is
        absent from cold and warm solver timing; the phase contract is covered
        on strict CPU and CUDA;
  - [x] measure current RSS through the fixture, preparation, cold-solver, and
        warm-solver windows with 10-ms `/proc/self/status` polling in provider
        children; retain process-lifetime `ru_maxrss` only as a separately
        labeled diagnostic; and
  - [x] invalidate the old custom-versus-Optax warm-time and solver-RSS
        promotion ratios. They compared unmatched provider preparation and a
        process-lifetime high-water mark; the replacement phase-scoped clean
        receipt is
        `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-five-sample-vram-20260802/`.
- [x] The strict-RTX-5090 `coil47` L-BFGS step-from-start runtime-compile
      diagnostic measured `20.303 s` to one executable, peak host RSS
      `1,816,276,992` bytes (`358,379,520`-byte delta), and recorded the
      receipt at
      `docs/receipts/custom-quasi-newton/coil47-lbfgs-gpu-runtime-compile-current/`.
      Device VRAM telemetry was not enabled, so the full measurement gate
      remains open.
- [x] A targeted strict-RTX-5090 custom `coil47` maxiter-20 run sampled
      `nvidia-smi` every 0.2 seconds. Peak runner-process GPU memory was
      `1076 MiB` of `32607 MiB`; cold/warm time was `78.599/0.253 s`, and
      solver RSS delta was `524468 KiB`. Receipt:
      `docs/receipts/custom-quasi-newton/coil47-custom-gpu-vram-current/`.
      This closes only the custom coil measurement; Optax, Boozer, and A100
      memory qualification remain open.
- [x] Compare Optax and custom overhead at a fixed accepted-step budget, then
      compare time to the same scientific certificate. The runner now
      precompiles the production custom fixed-shape transitions and the Optax step;
      CPU and strict-GPU `coil47` bundles both reach the same final objective
      when allowed to converge. Optax line-search
      evaluations remain unavailable and are not inferred from outer
      iterations. Phase-scoped memory and clean-checkout promotion remain
      open.
- [x] A fixed-budget CPU `coil47` Optax/custom/native comparison is recorded
      above, and the strict-GPU custom/Optax repetition is recorded above;
      A100 repetition and certificate-time promotion remain open.
- [x] A strict-GPU fixed-budget `coil47` comparison is also archived at
      `docs/receipts/custom-quasi-newton/coil47-custom-optax-gpu-fixed-budget-current/`.
      At two steps, custom/Optax warm time was `0.162/21.338 s`, RSS delta
      `546292/444792 KiB`, and final-objective difference was `2.96e-10`
      absolute. Both endpoints were capped; this is diagnostic evidence only.
- [x] The corresponding Optax VRAM sample is archived at
      `docs/receipts/custom-quasi-newton/coil47-optax-gpu-vram-current/`.
      The provider reached `834 MiB` of `32607 MiB` (`2.56%`), with
      `22.147 s` warm time and `444348 KiB` solver RSS delta. The two-step
      cap and dirty checkout keep this diagnostic-only.
- [x] The bounded strict-GPU `coil47` Optax `maxiter=20` attempt is recorded as
      incomplete: its provider child exceeded the declared 120-second watchdog
      before producing a solver payload. Receipt:
      `docs/receipts/custom-quasi-newton/coil47-optax-gpu-maxiter20-current/`.
- [x] A fresh strict-CPU `coil47` native/custom/Optax `maxiter=20` comparison
      reached successful endpoints for all three providers. The final
      objective was identical at recorded precision; custom versus native had
      `1.03e-9` maximum parameter difference, while Optax had `1.32e-4`.
      Warm time / solver RSS delta were `1.910 s / 0 KiB` native,
      `0.171 s / 354368 KiB` custom, and `49.584 s / 1548672 KiB` Optax.
      The original bundle remains preserved as historical diagnostic evidence.
      A lock-bound rerun is recorded at
      `docs/receipts/custom-quasi-newton/coil47-native-custom-optax-cpu-maxiter20-lockbound-current/`
      with the CPU lock hash. It remains diagnostic because the checkout is
      dirty and the budget is a bounded comparison.
- [ ] For the 47-parameter A100 case, require:
  - [ ] no 120-second or 8-GiB guard trip;
  - [ ] warm custom time no more than `2x` matched Optax warm time;
  - [ ] custom peak RSS no more than `1.5x` matched Optax peak RSS;
  - [ ] zero new eager step compilations when only budgets change; and
  - [ ] no monotonic RSS/device-memory or executable-count growth across five
        warm repeats.
- [ ] Collect one discarded warm-up followed by five synchronized samples per
      provider in fresh, otherwise idle children. Apply timing thresholds to
      the median, memory thresholds to the maximum, report every raw sample and
      range, and rerun when background GPU activity or thermal/power state
      changes during the pair.
- [ ] These `2x` and `1.5x` thresholds are predeclared initial promotion gates.
      Change them only by reviewing this plan before GREEN data are collected.
      The work need not prove every small example is faster than native CPU.

### Phase 6 — refactor, rollout, and rollback

The authoritative rollback base is
`9c64c2ef6cee45eb7eb1989bd5a41e2adf8bfc26`, the clean committed revision
before the current custom-runtime implementation slice. The older
`9ba1ad057...` and `c0dc94580...` revisions above are trajectory-comparison
fixtures only; neither is the rollback base.

The behavior-changing implementation slice is committed in `fd200f564`:

- `src/simsopt_jax/geo/optimizers/_shared.py`;
- `src/simsopt_jax/geo/optimizers/optimizer.py`;
- `src/simsopt_jax/geo/optimizers/private/{_step_runtime,_bfgs,_common,_lbfgs,_lbfgsb_scipy,_line_search,_result_converters,_types}.py`;
- `src/simsopt_jax/runtime/{host_boundary,jaxpr_closure}.py`;
- `src/simsopt_jax/solve/{__init__,contracts,dispatch}.py`;
- `src/simsopt_jax/objectives/stage_two.py`;
- `src/simsopt_jax_adapters/geo/boozer_surface.py`;
- `examples/jax/2_Intermediate/stage_two_optimization_planar_coils.py`;
- `examples/jax/3_Advanced/single_stage_optimization.py`; and
- `examples/jax/parity/cases/native_stage_two_optimization_planar_coils.py`.

The lock-only follow-up is `41d95cf502240a686cc968e690f62d4a85a2d1a3`.
No unrelated commit is part of the implementation slice. The rollback
rehearsal and bounded frozen-selector replay are recorded in
`docs/receipts/custom-quasi-newton/rollback-rehearsal-20260802.md`; the
rollback-base broad traceable application selector remains incomplete because
its run hit the declared RSS guard. Current-candidate compatibility closure is
recorded separately in
`docs/receipts/custom-quasi-newton/boozer-compatibility-partitions-20260802.md`.

- [ ] Remove duplicated eager host-loop mechanics only after BFGS and L-BFGS
      pass independently. Keep all mathematics algorithm-owned.
- [ ] Keep runtime, algorithm transitions, public routing, and tests in
      reviewable commits. Record exactly which commits alter active public
      behavior. The current implementation SHA is recorded, but the large
      implementation commit still needs this separation review.
- [x] Expand `[tool.pyright].include` and add
      `pyright.custom-quasi-newton.json`, scoped to the changed private
      optimizer, fixture, runner, and test paths with
      `typeCheckingMode: "strict"`. Do not add blanket ignores or permit
      unknown/`Any` types in the new protocol.
- [x] Update the solver matrix, provider plan, public docs, and examples to use
      the device/intent/provider taxonomy above. The custom provider remains the
      production BFGS/L-BFGS lane; Optax is explicit and comparative.
- [x] Record the exact pre-implementation rollback base above. Solver state is
      process-local, so no persisted-state migration is required.
- [x] After the implementation commits exist, record their SHAs oldest to
      newest. Rehearse `git revert --no-commit` for those SHAs in reverse order
      in a clean candidate worktree; the resulting source tree for every path
      above must be byte-identical to the rollback base. If a tested public
      old/new gate is added instead, prove that it selects untouched baseline
      primitives.
- [x] Run the frozen optimizer and bounded eager Boozer compatibility selectors
      on the rehearsed rollback tree, and archive the base, candidate, ordered
      SHA list, commands, exit codes, tree hashes, and receipt hashes. Return
      to the candidate commit and rerun the same bounded selectors.
- [x] Complete the current-candidate broad traceable Boozer compatibility
      selector in fresh private/public/shim processes on strict CPU and RTX
      5090 CUDA. The partitioned receipt records `111 passed, 3 skipped` on
      CPU and `110 passed, 4 skipped` on GPU. The rollback-base broad selector
      remains an incomplete rollback-side comparison.
- [x] Do not remove the traceable whole-solve implementation in this plan.

## Receipt contract

The runner writes local working data to
`.artifacts/custom-quasi-newton/<run-id>/`. Promotion additionally requires:

- [x] a tracked manifest at
      `docs/receipts/custom-quasi-newton/<run-id>/manifest.json` with schema
      version, commit, clean status, environment lock hashes, device, commands,
      exit codes, artifact checksums, verdicts, and archive URI;
- [x] tracked compact `metrics.json` and `summary.md` beside the manifest; and
- [x] raw logs/JSON copied to the archive URI and verified from a fresh process.

Current-head validation: `validate-all` rechecked `39` manifests with exit
code 0. The rollback receipt records the clean base/candidate tree hashes and
the bounded selector results; the external archive is local and not yet
replicated off-host.

The ignored local `.artifacts/` copy alone is never authority. A result is
incomplete if the archive is absent, a checksum differs, the environment is an
unsupported overlay, or a child is timed out/killed.

Phase 0 adds `benchmarks/custom_quasi_newton_receipts.py` with `publish` and
`validate-all` subcommands. `publish` accepts an explicit set of case-qualified
runner directories, an environment lock, a tracked destination, and an archive
URI; `validate-all` rehashes every
tracked and archived byte from a fresh process. Both commands must be covered
by missing-file, wrong-lock, artifact-tamper, and archive-tamper tests.

## Validation

### Supported environments

Phase 0 creates reviewed, relocatable, hash-locked dependency files. The lock
input includes the exact build-system requirements but omits the local `simsopt`
package itself; checkout identity is bound separately by Git commit. Generate
the locks once, review them, and commit them before collecting evidence:

```bash
uv pip compile pyproject.toml \
  benchmarks/environments/custom_quasi_newton_build.in \
  --python 3.11 --extra JAX --extra dev --extra ALGS \
  --no-emit-package simsopt --generate-hashes \
  -c benchmarks/environments/custom_quasi_newton_constraints.txt \
  -o benchmarks/environments/custom_quasi_newton_cpu.lock.txt

uv pip compile pyproject.toml \
  benchmarks/environments/custom_quasi_newton_build.in \
  --python 3.11 --extra JAX_GPU --extra dev --extra ALGS \
  --no-emit-package simsopt --generate-hashes \
  -c benchmarks/environments/custom_quasi_newton_constraints.txt \
  -o benchmarks/environments/custom_quasi_newton_gpu.lock.txt
```

Replay clean Python 3.11 environments from those committed locks, then install
only the candidate checkout without resolving dependencies again:

```bash
uv venv --python 3.11 .venv-qn-cpu
uv pip sync --python .venv-qn-cpu/bin/python --require-hashes \
  benchmarks/environments/custom_quasi_newton_cpu.lock.txt
uv pip install --python .venv-qn-cpu/bin/python --no-deps \
  --no-build-isolation -e .

uv venv --python 3.11 .venv-qn-gpu
uv pip sync --python .venv-qn-gpu/bin/python --require-hashes \
  benchmarks/environments/custom_quasi_newton_gpu.lock.txt
uv pip install --python .venv-qn-gpu/bin/python --no-deps \
  --no-build-isolation -e .
```

The constraints file pins SciPy and Optax versions compatible with the
project-pinned JAX/JAXLIB `0.10.0`; the initial Optax comparator pin is `0.2.8`.
The manifest binds the appropriate lock hash, Python ABI, platform tag, and
candidate commit. A `pip freeze` may be retained as a diagnostic, but it is not
the replay authority because editable paths are machine-specific. Do not use
the system Python or an overlay from another checkout.

For non-MPI optimizer tests, set `MPI4PY_RC_INITIALIZE=0` in the child
environment so importing `simsopt.field` does not auto-initialize MPI or open a
host X11 session. Do not apply that setting to a VMEC/MPI workflow that
explicitly owns MPI initialization.

### CPU

```bash
MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/jax/solve/test_lbfgsb_trajectory_parity.py \
  tests/jax/solve/test_optimizer_result_schema.py \
  tests/geo/test_lbfgsb_scipy_jax_kernels.py \
  tests/benchmarks/test_custom_quasi_newton_runtime.py \
  -m "not slow"

# Source-owned fixture construction is intentionally a separate slow lane.
MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/benchmarks/test_custom_quasi_newton_runtime.py -m slow

MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python -m pytest -q \
  tests/geo/test_boozersurface_jax_private.py \
  tests/geo/test_boozersurface_jax.py \
  tests/jax/solve/test_driver_dispatch.py \
  tests/jax/solve/test_compat_shim_translation.py \
  tests/jax/examples/test_single_stage_vmec_hybrid_example.py \
  -k "bfgs_ondevice or lbfgs_ondevice or limited_memory or traceable"
```

Each CPU process must assert `jax.default_backend() == "cpu"` and record
`jax.devices()` before tests or probes.

Run the matched native/custom CPU cases through the versioned runner. The
source-owned physics fixture builders are separate slow probes; the runner's
solver timings begin after fixture construction.

The current executable smoke cases are `--cases rosenbrock`, `--cases coil47`,
and `--cases boozer` (or `bfgs_quadratic` with `--providers native,custom` for
dense BFGS). Coil and Boozer use different solver methods, so they run as
separate commands. Coil47 and Boozer now have matched native/JAX CPU
initial-objective paths; endpoint and GPU qualification remain separate gates.

```bash
RUN_ROOT=".artifacts/custom-quasi-newton/$(date -u +%Y%m%dT%H%M%SZ)-cpu-smoke"
ROSENBROCK_RUN_DIR="$RUN_ROOT/rosenbrock"
COIL_RUN_DIR="$RUN_ROOT/coil47"
BOOZER_RUN_DIR="$RUN_ROOT/boozer"
mkdir -p "$ROSENBROCK_RUN_DIR" "$COIL_RUN_DIR" "$BOOZER_RUN_DIR"

MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device cpu --intent parity --providers native,custom \
  --cases rosenbrock --output "$ROSENBROCK_RUN_DIR"

# Source-owned coil physics smoke; native/JAX CPU objective parity is covered.
MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device cpu --intent parity --providers native,custom \
  --cases coil47 --output "$COIL_RUN_DIR"

# Source-owned Boozer physics smoke; native and custom providers share the
# initial objective/gradient contract. Capped endpoints are diagnostic only.
MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_cpu_parity \
SIMSOPT_BACKEND_STRICT=1 SIMSOPT_PRECISION=fp64 PYTHONPATH=src:. \
.venv-qn-cpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device cpu --intent parity --providers native,custom \
  --cases boozer --output "$BOOZER_RUN_DIR"
```

### Strict GPU

Run the fast focused tests and the physics cases in fresh CUDA-only processes.
The `slow` fixture-construction tests are a separate qualification step; do
not include them in the fast contract command:

```bash
RUN_ROOT=".artifacts/custom-quasi-newton/$(date -u +%Y%m%dT%H%M%SZ)-gpu-parity"
CUSTOM_COIL_RUN_DIR="$RUN_ROOT/custom/coil47"
CUSTOM_BOOZER_RUN_DIR="$RUN_ROOT/custom/boozer"
OPTAX_COIL_RUN_DIR="$RUN_ROOT/optax/coil47"
mkdir -p "$CUSTOM_COIL_RUN_DIR" "$CUSTOM_BOOZER_RUN_DIR" \
  "$OPTAX_COIL_RUN_DIR"

MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/jax/solve/test_lbfgsb_trajectory_parity.py \
  -m "not slow"

# Broader Boozer compatibility selectors are a separate bounded qualification
# run; do not fold their fixture construction into the fast contract lane.
MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python -m pytest -q \
  tests/geo/test_boozersurface_jax_private.py \
  -k "bfgs_ondevice or lbfgs_ondevice or limited_memory or traceable"

# Slow source-owned fixture construction is run separately when GPU resources
# are available; it is not part of the fast contract result.
MPI4PY_RC_INITIALIZE=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python -m pytest -q \
  tests/jax/solve/test_custom_quasi_newton_step_runtime.py \
  tests/benchmarks/test_custom_quasi_newton_runtime.py -m slow

MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers custom \
  --cases coil47 --output "$CUSTOM_COIL_RUN_DIR"

MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers custom \
  --cases boozer --output "$CUSTOM_BOOZER_RUN_DIR"

MPI4PY_RC_INITIALIZE=0 MPLBACKEND=Agg JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
SIMSOPT_BACKEND_MODE=jax_gpu_parity SIMSOPT_BACKEND_STRICT=1 \
SIMSOPT_PRECISION=fp64 XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=src:. \
.venv-qn-gpu/bin/python benchmarks/custom_quasi_newton_runtime.py \
  --device gpu --intent parity --providers optax \
  --cases coil47 --output "$OPTAX_COIL_RUN_DIR"
```

The runner must fail before execution unless the requested CPU lane reports
`jax.default_backend() == "cpu"`, or the requested GPU lane reports a CUDA/
ROCm/GPU backend and every visible JAX device is a GPU. Every leaf of inputs,
closure constants, initialized solver state, step output, and pre-host final
result is placed on a GPU. Use
`XLA_PYTHON_CLIENT_PREALLOCATE=false` with
`XLA_PYTHON_CLIENT_ALLOCATOR=platform` for matched GPU comparisons and record
both settings in the schema-6 receipt. Transfer-guard checks are a separate
boundary lane: do not enable a global disallow guard for this fixture suite,
whose setup intentionally constructs device inputs from host literals.

### Quality and closure

```bash
# After the Phase-0 receipt publisher lands:
PYTHONPATH=src:. .venv-qn-cpu/bin/python \
  benchmarks/custom_quasi_newton_receipts.py validate-all \
  --root docs/receipts/custom-quasi-newton

.venv-qn-cpu/bin/pyright --project pyright.custom-quasi-newton.json --warnings
.venv-qn-cpu/bin/ruff check src/simsopt_jax tests/jax tests/geo benchmarks
.venv-qn-cpu/bin/ruff format --check src/simsopt_jax tests/jax tests/geo benchmarks
.venv-qn-cpu/bin/python -m compileall -q src/simsopt_jax tests/jax benchmarks
```

Run the full project suite after focused validation. Final closure must be
replayed from a clean detached worktree at the candidate commit; run
`git diff --check` there, then verify the tracked manifest against the external
archive. This prevents unrelated dirty-tree files from entering the verdict.

## Completion criteria

- [x] Eager custom BFGS/L-BFGS use the fixed-shape step runtime; traced callers
      retain a supported whole-solve JAX path.
- [x] Changing only `maxiter`/`maxfun` causes no new eager init/step executable,
      and the `maxcor` allocation change is measured and documented.
- [x] Public options, callbacks, statuses, counters, result fields, zero-budget
      semantics, and L-BFGS inverse-Hessian behavior pass frozen tests.
- [x] Native SciPy/SIMSOPT remains the parity oracle; Optax remains explicit.
- [x] CPU and strict-GPU tests pass in supported isolated environments without
      weakened tolerances or CPU fallback.
- [ ] Physics cases meet the predeclared gates with durable raw evidence.
- [x] Normal execution retains no trajectory and uses only audited host
      boundaries.
- [ ] Every remaining behavior change and newly found defect has preserved
      RED -> GREEN -> REFACTOR evidence. The already-implemented core remains
      explicitly post-hoc and is not granted synthetic historical RED credit.
- [ ] Focused and broad tests, Pyright, Ruff, formatting, compileall, and clean
      `git diff --check` pass.
- [ ] Solver architecture docs agree with this plan, and rollback is proven.

## Resolved decisions

- Keep the whole-solve route for traced production callers.
- Use one packed scalar host observation per eager transition; defer fixed-size
  multi-step chunks until this path is correct and measured.
- Report dense-BFGS memory; do not add automatic routing.
- Compare Optax in the supported project JAX/JAXLIB environment and record the
  exact resolved Optax version.
- Use the `2x` warm-time and `1.5x` RSS gates above unless this plan is revised
  before implementation.
