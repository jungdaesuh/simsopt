# Priority 7 — Permanent-magnet solver kernel parity audit (MwPGP & GPMO)

**Audit timestamp**: 2026-05-16
**Auditor**: Claude (parity-audit fresh agent)

## Files audited

| Role | Path | Lines |
|------|------|-------|
| JAX kernel | `src/simsopt/jax_core/pm_optimization.py` | 2485 |
| C++ reference | `src/simsoptpp/permanent_magnet_optimization.cpp` | 1332 |
| C++ header | `src/simsoptpp/permanent_magnet_optimization.h` | 39 |
| Python orchestrator | `src/simsopt/solve/permanent_magnet_optimization.py` | 480 |
| JAX kernel parity test | `tests/jax_core/test_pm_optimization_jax_item25.py` | 1495 |
| JAX orchestrator parity test | `tests/solve/test_permanent_magnet_optimization_jax_item28.py` | 878 |
| CPU orchestrator test (oracle ref) | `tests/solve/test_pm_optimization.py` | 197 |
| Parity ladder | `benchmarks/validation_ladder_contract.py:210-222` (`pm_mwpgp_fixed_step`) | — |

## Executive summary — top 3 findings

1. **HIGH — `projection_l2_balls` produces NaNs whenever any `m_maxima_i == 0`** (`src/simsopt/jax_core/pm_optimization.py:2122-2125`). The JAX implementation computes `unit = m_maxima / m_maxima` as its "one" tensor instead of `jnp.ones_like(m_maxima)`. With a zero entry this evaluates to `0/0 = NaN`, then `max(NaN, anything) = NaN`, then `m / NaN = NaN`. The C++ reference uses the literal `1.0` (line 14: `std::max(1.0, sqrt(...) / m_maxima)`) so a zero `m_maxima` simply blows up `denom` to `+inf` and the projected vector becomes zero — finite but degenerate. The CPU orchestrator (`solve/permanent_magnet_optimization.py:79-84`) likewise uses `np.maximum(np.ones(...), ...)`. There is **no test** with `m_maxima_i == 0`; every fixture uses `0.3 + rng.random(...)` or `np.full(N, ...)` so this latent bug is not exercised. Trivial fix: replace line 2123 with `unit = jnp.ones_like(m_maxima)`.
2. **MEDIUM (intentional, but reviewer-visible) — JAX `mwpgp_solve` deliberately drops the C++ early-stop and history-snapshot machinery.** The C++ kernel runs up to `max_iter` but bails out on `x_sum = sum|x_k - x_k_prev| < epsilon` (line 319) or `R2_history(print_iter) < min_fb` (line 307); it also records iterates every `max_iter/5` into `m_history`. The JAX `mwpgp_solve` is documented as fixed-step (file docstring lines 24-45) and exposes neither convergence early-exit nor the 21-slot history. State-trace parity tests therefore (correctly) force the C++ side to use `epsilon=0, min_fb=0` (test lines 1244-1266) — but a caller substituting `mwpgp_solve` for `MwPGP_algorithm` in the orchestrator without remembering this will quietly burn extra iterations after convergence and not produce the `objective_history`/`m_history`/`R2_history` arrays that downstream code reads.
3. **INFO — coverage gaps**: (a) no L0/L1 regularizer parity at the kernel level (the C++ kernel only diagnostically reports L0/L1, so the omission is correct per the convex-only contract, but no test asserts this); (b) no `single_direction` coverage for `GPMO_backtracking` JAX (the C++ supports `single_direction` and the JAX `GPMOBacktrackingSpec` carries it as a meta field but tests only exercise `single_direction=-1`); (c) `find_max_alphaf` is exercised only at a single value of `(a, b, c)` and never with `a ≤ tol` returning the sentinel under `_step_body` (sentinel asserted only in the standalone helper test at lines 241-246); (d) no test confirms the JAX `mwpgp_step` "expand" branch is hit (`alpha_cg >= alpha_f`) — `count_jaxpr_primitives(..., "cond") == 2` (test line 1495) only proves both `cond`s compile, not both branches execute on the trace.

## Function-by-function parity matrix

| C++ function | JAX function | Math | Physics | Algorithm | Computation | Verdict |
|--------------|--------------|------|---------|-----------|-------------|---------|
| `projection_L2_balls` (cpp:13) | `projection_l2_balls` (py:2108) | OK except `unit = m_maxima/m_maxima` instead of 1.0 — NaN-prone for `m_maxima=0` | OK | OK | NaN risk for zero radius | **HIGH** (Finding 1) |
| `phi_MwPGP` (cpp:19) | `phi_mwpgp` (py:2142) | OK | OK | OK | Vectorized; deterministic | PASS |
| `beta_tilde` (cpp:35) | `_beta_tilde` (py:2167) | OK; uses `safe_norm` guard when `||m||=0` (justified by `on_ball` mask) | OK | OK | OK | PASS |
| `g_reduced_gradient` (cpp:61) | `g_reduced_gradient` (py:2153) | OK | OK | OK | OK | PASS |
| `g_reduced_projected_gradient` (cpp:69) | `g_reduced_projected_gradient` (py:2195) | OK (φ + β̃ sum) | OK | OK | OK | PASS |
| `find_max_alphaf` (cpp:79) | `find_max_alphaf` (py:2207) | OK; extra `sqrt(max(disc, 0))` guard (documented py:2234-2239) | OK | OK | Single-row case test only | PASS (with minor test gap) |
| `MwPGP_algorithm` main loop (cpp:153-325) | `_step_body` + `mwpgp_solve` (py:2278-2485) | OK (cost J = ½‖Am−b‖² + reg_l2 ‖m‖² + ½ν⁻¹ ‖m−m_proxy‖²); Hessian action `H v = AᵀAv + 2(reg_l2 + ½ν⁻¹)v` matches cpp:188 / cpp:217 | L0/L1 dropped (correct: cpp reports only) | Fixed-step scan; **no convergence early-exit, no history snapshot** | Three-branch `lax.cond` (CG / expand / projected); residual proxy omits constant `‖b‖²` (documented py:2444-2448) | **MEDIUM** (Finding 2) |
| `GPMO_baseline` (cpp:1238-1332) | `gpmo_baseline_solve` (py:669) | OK (cost = ‖res ± Aⱼ‖² + reg_l2·m_maxima_{j//3}²) | OK | `argmin` order: plus-then-minus matches `std::min_element` tie order | Sentinel `1e50` used for unavailable; matches cpp `R2s` initial | PASS |
| `GPMO_multi` (cpp:584-713) | `gpmo_multi_solve` (py:1678) | OK including the C++ index-quirk (penalty uses `mmax_ptr[cj]` indexing 3N-long vector by dipole id — JAX mirrors at py:1591-1592) | OK | Nearest-`Nadjacent`-available selection via cumsum-rank (py:1571-1573) reproduces C++ `while not Gamma_ptr[...]` traversal because both pick the first `Nadjacent` available entries in `connectivity[seed]` | OK | PASS |
| `GPMO_backtracking` (cpp:381-580) | `gpmo_backtracking_solve` (py:1997) | OK | OK | "Dewyrming" condition `(k >= backtracking) and (k % backtracking == 0)` matches cpp:487 exactly (py:1931-1933); cascading removals propagate within a pass via `lax.scan` state carry | Carry-forward `done` mask after stop predicate; no host break | PASS |
| `GPMO_ArbVec` (cpp:1124-1233) | `gpmo_arbvec_solve` (py:841) | OK; penalty uses first-N entries of component-mmax (mirrors cpp `mmax_ptr[j]` index quirk — documented py:770-771) | OK; arbitrary polarization vectors | Candidate order is dipole-major / pol-vector-minor with plus-then-minus | OK | PASS |
| `GPMO_ArbVec_backtracking` (cpp:730-988) | `gpmo_arbvec_backtracking_solve` (py:1363) | OK | OK; thresh_angle in radians, cos check `min_cos_angle <= cos(thresh_angle)` matches cpp:899 | Dewyrming gate `(k % backtracking == 0)` triggers on `k=0` (matches cpp:862 — explicitly distinguished from baseline `GPMO_backtracking` in py:1287-1291) | Sentinel `cos_angle = 2.0` for not-placed neighbors | PASS |
| `initialize_GPMO_ArbVec` (cpp:995-1118) | `initialize_gpmo_arbvec` (py:906) | OK; chooses argmin over (pos, neg, null) candidates per dipole; mirrors cpp tie order plus-then-minus-then-null (py:982-1004) | OK | OK | Single `argmin` per row replaces C++ stateful loop; verified by direct C++ oracle parity (test lines 813-874) | PASS |
| `connectivity_matrix` (cpp:355-377) | `gpmo_connectivity_matrix` (py:729) | OK | OK | Stable argsort vs incremental `std::min_element` per slot; tie order may differ for exactly-equal distances | OK | INFO (see §c) |
| `print_MwPGP` (cpp:98-145) | (not ported) | — | — | C++ diagnostic only | — | INFO (no port required) |
| `print_GPMO` (cpp:329-352) | (not ported) | — | — | C++ diagnostic only | — | INFO (no port required) |

## (a) MwPGP parity

**Cost function**. The C++ objective evaluated in `print_MwPGP` (cpp:132-141) is `J = R2 + N2 + L2 = ½‖Am−b‖² + (1/(2ν))‖m−m_proxy‖² + reg_l2 ‖m‖²`. L0 and L1 terms are computed and printed but **explicitly excluded from `cost`** (cpp:139-141). The JAX test mirrors this exactly in `_cost` (test:124-137). The Hessian-times-vector action used in `_step_body` (py:2256-2272) is `H v = AᵀA v + 2(reg_l2 + 1/(2ν)) v`, matching cpp:188 and cpp:217 (`+ 2 eigen_v * (reg_l2 + 1.0/(2.0*nu))`).

**Gradient (= residual)**. Both implementations carry `g = H m − ATb_rs` where `ATb_rs = ATb + m_proxy/ν`. JAX initializes via `_initial_state` (py:2371-2385) and incrementally updates `g` either via `g - α_cg ATAp` (CG branch, py:2311 ≡ cpp:242) or by full recomputation `g = H x_proj - ATb_rs` (expand and projected-gradient branches, py:2320, 2327 ≡ cpp:272-280, cpp:292-300).

**Decision tree**. Three branches: `inner_branch` when `‖g_α_p‖² ≤ ‖φ‖²` and `α_cg < α_f` (CG); `expand_branch` when `‖g_α_p‖² ≤ ‖φ‖²` and `α_cg ≥ α_f`; `projected_gradient_branch` otherwise. Exactly mirrors cpp:234-302. The JAX expand-branch update (py:2318) `(x − α_f p) − α (g − α_f ATAp)` is bitwise the same construction as cpp:268; both then project onto the L2 balls and recompute `g` from scratch.

**Step-size choice**. The user passes a fixed `alpha`. Both kernels use a single scalar `α_f = min_i find_max_alphaf_i` (cpp:230-231 via `std::min_element`; JAX py:2307 via `jnp.min(find_max_alphaf(...))`). The CG step `α_cg = gp / pATAp` matches cpp:232 (py:2306). No Barzilai-Borwein or backtracking line search exists in either side — the contract is pre-tuned `α < 2/λ_max(H)` set by the orchestrator at `solve/permanent_magnet_optimization.py:197-198` (`alpha_max = 2.0 / pm_opt.ATA_scale * (1 - 1e-5)`).

**Convergence criteria — DIVERGENCE (intentional)**. C++ terminates when `x_sum = Σ |x_k − x_k_prev| < epsilon` (cpp:319) or `R2_history < min_fb` (cpp:307). JAX has neither — `mwpgp_solve` runs exactly `n_steps` iterations and never short-circuits (py:32-34, 2431-2432). Tests work around this by forcing `epsilon=0, min_fb=0` on the C++ side (test:1259-1264). **Caveat**: setting `epsilon=0` does not actually disable the C++ early-exit on the *first* iteration where `x_k1 == x_k_prev` (it would if `epsilon > 0`); the test docstring (lines 1244-1250) notes that strict-less-than `x_sum < epsilon` with `x_sum = 0` and `epsilon = 0` is false, so the loop continues — verified.

**Verified parity** (`test_cpp_oracle_parity_state_trace`, `test_cpp_oracle_parity_with_l2_regularization`, `test_cpp_oracle_parity_with_relax_and_split`, lines 1221-1383): JAX `mwpgp_solve` matches `simsoptpp.MwPGP_algorithm` final iterate to `rtol=1e-9, atol=1e-11` (the `pm_mwpgp_fixed_step.state_trace_*` ladder lane). Also: monotone-decreasing cost test (1116-1172), unconstrained-interior optimality (1174-1219), and `lax.cond` static count (1479-1495).

## (b) GPMO parity

GPMO is a **greedy** algorithm: at each step it scans all available `(dipole, [component|pol_vector], sign)` candidates and picks the one whose placement minimizes the next residual `‖A m − b‖² + penalty`. **Ordering is load-bearing**: ties must be broken identically for state-trace parity.

**Tie-breaking convention**. C++ `std::min_element` returns the **first** minimum in scan order. The scan order for `GPMO_baseline` is `R2s` of length `6N`: `[plus over j=0..3N-1; minus over j=0..3N-1]`. JAX `jnp.argmin` on a freshly built `concatenate([plus, minus])` of length `6N` matches the C++ ordering. JAX `gpmo_baseline_step` (py:625-630) decodes `is_minus`, `component_index`, then `dipole = component_index // 3`, `component = component_index % 3`, identical to cpp:456-468.

**Sentinel for unavailable slots**. C++ initializes `R2s = 1e50` for unavailable entries; JAX uses `1.0e50` via `jnp.where(allowed, ..., sentinel)` (py:609-611, 776-778, 1062-1065, 1597-1600). A real candidate cost above `1e50` would corrupt the ordering — for `b` and `A` drawn from N(0,1) at the scale used in tests this never occurs, but in pathological inputs it could; **no test asserts that real costs never exceed the sentinel**. This is a shared design hazard with the C++ reference (not a divergence).

**Penalty quirk faithfully mirrored**. The C++ `GPMO_multi` (cpp:662) and `GPMO_ArbVec*` (cpp:822, 1193) accumulate `mmax_partial_sum += mmax_ptr[cj] * mmax_ptr[cj]` where `mmax_ptr` points at the component-expanded `np.sqrt(reg_l2) * np.repeat(m_maxima, 3)` vector but is indexed by **dipole id** `cj`, not component id `3*cj + l`. This is an index-bug-by-convention in the C++ reference: it picks the first-of-three components for each dipole. The JAX deliberately mirrors this by `_component_mmax(m_maxima)[: m_maxima.shape[0]]` (py:771) and `[ordered_dipoles]` (py:1591-1592). For typical `reg_l2` and uniform `m_maxima` the user-observable penalty is `reg_l2 * m_maxima[dipole]²` per placement, which is what the user likely expects.

**Connectivity matrix**. C++ `connectivity_matrix` (cpp:355-377) does `Nadjacent = 2000` slots (hard-coded width regardless of the user's `Nadjacent` arg) using repeated `std::min_element` with the picked slot blanked to `1e10`. JAX `gpmo_connectivity_matrix` (py:729-735) uses `argsort(stable=True)`. For non-degenerate distances both produce the same row. For exactly-tied distances, C++ keeps insertion order (first encountered wins each `std::min_element`); JAX `argsort(stable=True)` keeps input-array order. Both pick the same elements; tie order may rarely differ — uncovered by tests but would only affect candidate scoring in the case of geometric coincidences.

**Backtracking gate divergence (intentional)**. `GPMO_backtracking` (`baseline-with-cleanup`) only dewyrms when `k >= backtracking and k % backtracking == 0` (cpp:487 ≡ py:1931-1933). `GPMO_ArbVec_backtracking` dewyrms whenever `k % backtracking == 0`, including `k=0` (cpp:862 ≡ py:1292). The JAX file comments the discrepancy and is correct.

**Dewyrming pair selection**. The two backtracking variants disagree on what "wyrm" means:
- `GPMO_backtracking`: prior-pair removal when **adjacent placed dipoles have the same Cartesian component selected with opposite signs** (cpp:498: `sk_sign_fac[jk] == -sk_sign_fac[cj] && skjj_ind[jk] == skjj_ind[cj]`). JAX `_gpmo_backtracking_remove_pairs` (py:1820-1827) mirrors exactly.
- `GPMO_ArbVec_backtracking`: prior-pair removal when **the most-anti-aligned adjacent placed neighbor has `cos_angle <= cos(thresh_angle)`** (cpp:899). JAX `_gpmo_arbvec_remove_pairs` (py:1156-1179) mirrors exactly.

**Cascading removals**. C++ outer loops over `j = 0..N-1` for the ArbVec variant (cpp:867) and over the placement history `j = 0..k-1` for the baseline backtracking variant (cpp:490). After a pair is removed, both the seed and the neighbor become available, which can affect subsequent seeds in the same pass. JAX correctly propagates the updated `(x, residual, available, signs, components)` state through `jax.lax.scan` so cascading removals match (py:1210-1221 for ArbVec; py:1878-1882 for baseline).

**Verified parity** (tests lines 331-998): C++ oracle parity confirmed for all five GPMO variants (`baseline`, `multi`, `backtracking`, `ArbVec`, `ArbVec_backtracking`) at `pm_mwpgp_fixed_step` ladder (`rtol=1e-9, atol=1e-11`), including `reg_l2 > 0`, all `single_direction` modes for baseline/multi, `max_nMagnets` stop condition, nonzero `x_init` seeding, and at-least-one dewyrming pass for both backtracking variants.

## (c) Regularizer parity

| Term | Convex sub-problem | Outer relax-and-split | Used in MwPGP cost? | Used in GPMO score? | JAX coverage |
|------|---------------------|------------------------|---------------------|---------------------|--------------|
| L2 (`reg_l2`) | Yes — adds `2 reg_l2 v` to Hessian action; adds `reg_l2 ‖m‖²` to cost | Yes (always) | YES (cpp:188, 217 ≡ py:2271) | YES (penalty `reg_l2 m_maxima_j²` per candidate) | PASS — kernel parity + ladder |
| L0 (`reg_l0`) | NO — cost reports only (cpp:117, 137) | YES — `prox_l0` (`solve:13-39`) in outer loop | NO | NO | INFO — only `prox_l0_jax` parity tested at item 28 (orchestrator), no kernel test asserts JAX `mwpgp_solve` ignores `reg_l0` |
| L1 (`reg_l1`) | NO — cost reports only (cpp:116, 136) | YES — `prox_l1` (`solve:42-64`) in outer loop | NO | NO | INFO — only `prox_l1_jax` parity tested at item 28, no kernel test asserts JAX ignores `reg_l1` |
| Relax-and-split (1/ν, m_proxy) | YES — adds `1/ν` to Hessian and `(1/ν) m_proxy` to gradient source; cost has `½‖m−m_proxy‖²/ν` | YES (couples convex and prox steps) | YES (cpp:179, 188, 217 ≡ py:2407, 2271, 2464) | NO (GPMO score has no `m_proxy` term) | PASS — `test_cpp_oracle_parity_with_relax_and_split` line 1336 |

**Notes**:
- The orchestrator (`solve/permanent_magnet_optimization.py:118-275`) forwards `reg_l0`/`reg_l1` only to the **outer** prox step, never to `MwPGP_algorithm`. So a JAX kernel that silently dropped `reg_l0`/`reg_l1` from its signature would still produce correct outer-loop behavior. The JAX `PMOptimizationSpec` only has `m_maxima, m_proxy, nu, reg_l2, alpha` — consistent.
- `prox_l0` and `prox_l1` (CPU) and `prox_l0_jax`/`prox_l1_jax` (item 28) are exercised in `tests/solve/test_permanent_magnet_optimization_jax_item28.py:184-197` with element-wise NumPy parity. **There is no end-to-end relax-and-split parity test that pins JAX against CPU after multiple outer iterations.**
- `relax_and_split` (CPU orchestrator) at `solve/permanent_magnet_optimization.py:118-275` raises if both `reg_l0` and `reg_l1` are nonzero (line 210-211). The `relax_and_split_jax` orchestrator covered at item 28 mirrors this restriction (not re-checked here, but recorded as INFO so reviewers verify).

## (d) Convergence criteria parity

| Aspect | C++ `MwPGP_algorithm` | JAX `mwpgp_solve` | Verdict |
|--------|------------------------|---------------------|---------|
| Termination on iterate diff (`x_sum < epsilon`) | YES (cpp:319-322) | NO | INTENTIONAL DIVERGENCE — fixed-step contract documented py:24-34 |
| Termination on residual (`R2 < min_fb`) | YES (cpp:307) | NO | INTENTIONAL DIVERGENCE — same |
| Max-iteration cap (`max_iter`) | YES | YES (`n_steps`, static) | OK — JAX requires static int (py:2450-2451) |
| History snapshots (21-slot `m_history`, `objective_history`, `R2_history`) | YES (cpp:174-176, snapped at every `max_iter/5`) | NO (returns only final `m` and a per-step residual-proxy scan) | INTENTIONAL — caller substituting JAX must not depend on `(objective_history, R2_history, m_history)` shape from C++ |
| Per-step residual proxy correctness | `R2 = ½ Σ (Am − b)²` recomputed at snapshots | `Σ (Am)² − 2 m·ATb` (omits constant `‖b‖²`, documented py:2444-2448) | OK for monotonicity, not bit-equal to C++ `R2` series |

| Aspect | C++ GPMO variants | JAX GPMO variants | Verdict |
|--------|---------------------|---------------------|---------|
| Termination on `num_nonzero >= N` | YES (cpp:559, 965) | YES (py:1327, 1967 — sets `done` mask, carry-forward state) | OK |
| Termination on `num_nonzero >= max_nMagnets` | YES (cpp:559, 965) | YES (same lines) | OK |
| Termination on stalled nonzero count (3 consecutive snapshot rounds with no change, cpp:547-554, 953-961) | YES | NO — JAX has no host-side print history to interrogate | INTENTIONAL DIVERGENCE (only triggers under verbose printing in C++; not asserted as user-visible) |
| Hard cap (`K`) | YES (outer `for` bound) | YES (`scan` length) | OK |

## Test coverage gaps

| Gap | Severity | Affected JAX function | Suggested test |
|-----|----------|-----------------------|----------------|
| `projection_l2_balls` with `m_maxima_i == 0` | HIGH | py:2108 (Finding 1) | Assert `out` is finite when one entry is zero; expect zero radius behavior matching CPU `solve.projection_L2_balls` |
| `find_max_alphaf` with `a ≤ tol` returning sentinel inside `_step_body` | LOW | py:2207 + py:2278 | Construct `p = 0` row and verify the `expand` branch never selects a NaN step |
| `mwpgp_step` expand-branch (taken when `alpha_cg ≥ alpha_f`) | INFO | py:2317-2321 | Synthesize a row near the L2 boundary with `p` pointing outward and check the `lax.cond` traces the expand body |
| `GPMO_backtracking` with `single_direction ∈ {0,1,2}` | INFO | py:1886 (uses baseline candidate costs which honor `single_direction`) | Mirror `test_solver_matches_cpp_baseline_for_all_single_direction_modes` for backtracking |
| `relax_and_split` end-to-end JAX vs CPU parity over ≥2 outer iterations with reg_l0 > 0 | INFO | `solve.permanent_magnet_optimization_jax.relax_and_split_jax` | Existing tests in item 28 exercise individual prox kernels but not the outer loop's iterate trajectory |
| Penalty index quirk in `GPMO_multi`/`GPMO_ArbVec` with non-uniform `m_maxima` | INFO | py:771, 1591 | Add a test with `m_maxima` chosen so the per-component vs per-dipole indexing would disagree; assert JAX matches C++ (it does, but the regression flag would catch a "fix" that breaks parity) |
| Behavior when real GPMO candidate cost exceeds the `1e50` sentinel | INFO | py:609, 776, 1062, 1597 | Scale `A`/`b` such that the smallest plus cost is `> 1e50` and verify the unavailable-vs-real distinction holds (both kernels share the hazard) |
| L0/L1 silently ignored in `mwpgp_solve` | INFO | py:2411 | Assert `mwpgp_solve(spec, A, ATb, m0, n_steps=K)` is invariant under setting `reg_l0`/`reg_l1` on a future spec extension (defensive only — neither field exists today) |

## Recommended actions ordered by severity

1. **HIGH — Fix `projection_l2_balls` (py:2123)**: change `unit = m_maxima / m_maxima` to `unit = jnp.ones_like(m_maxima)`. Add a regression test with `m_maxima = [1.0, 0.0, 1.0]` confirming finiteness and matching the CPU `projection_L2_balls` zero-collapse behavior.
2. **MEDIUM — Document the early-stop omission in `mwpgp_solve`'s public docstring with a "CPU semantics" callout**: explicitly state that callers translating `MwPGP_algorithm(..., epsilon=1e-4, min_fb=1e-20)` should pre-compute `n_steps` from a representative run, or accept the (often-cheap) extra iterations. This is the single most common porting trap for downstream users.
3. **MEDIUM — Add a structural test that the JAX kernel's three `_step_body` branches are all reachable**: the existing test counts two `cond` primitives in the jaxpr but does not prove the expand branch ever fires. A small synthetic problem near the L2 boundary would cover it.
4. **LOW — Reinforce the multi/ArbVec penalty-index quirk with a non-uniform `m_maxima` fixture** so accidental "fixes" of the dipole-id-vs-component-id indexing get caught.
5. **LOW — Add a `single_direction` parity sweep for `GPMO_backtracking`** mirroring the existing baseline/multi sweeps.
6. **INFO — Add an end-to-end `relax_and_split_jax` vs `relax_and_split` (CPU) parity test** that runs ≥2 outer iterations with `reg_l0 > 0` (and again with `reg_l1 > 0`) on a small synthetic problem; this guards the orchestrator-level wiring, which is currently only spot-checked at the prox primitives.
7. **INFO — Smoke-check the sentinel-vs-real-cost assumption** by scaling `b` to be enormous; both kernels share the `1e50` hazard but the JAX path could be made more robust by switching to `jnp.inf` (and casting back to `float64` for `jnp.argmin`) without changing C++ parity for normal inputs.
