# simsopt Performance & Safety Audit

Audit target: `simsopt` at HEAD `1b0cc3a9` (master).
Scope: `src/simsoptpp/` (C++ extension) and `src/simsopt/` (Python).
Status: static review validated against the tree. **Speedup magnitudes are unvalidated** until the benchmark harness in §6 is in place.

---

## 1. Executive summary

simsopt's compute path is a C++ extension (`simsoptpp`) exposing Biot–Savart, surface Fourier, dipole / permanent-magnet, and regular-grid interpolation kernels, plus an ODE particle tracer. XSIMD + OpenMP are already used inside kernels.

Three classes of issues were identified:

- **Memory / thread-safety bugs** (A-series): unambiguous bugs, small diffs.
- **Structural performance issues** (B-series): real, directional wins, mechanical to implement.
- **Design hypotheses** (C-series): observations with plausible payoff that need microbenchmarks before the gain can be claimed.

A prior "3×/5×/20×" speedup table was withdrawn. It was derived from inspection, not measurement. The measurement-first plan in §6 produces defensible numbers as each fix lands.

---

## 2. Ledger

16 confirmed, 4 partial, 2 withdrawn. Per-item file:line evidence.

| # | Status | Item | Evidence |
|---|--------|------|----------|
| A1 | Confirmed | Heap leak in surface Fourier VJPs — **3 active sites per build**, not 6 | `surfacerzfourier.cpp:472, 553, 680, 764, 1004, 1088` (SIMD/non-SIMD pairs; only one active) |
| A2 | Confirmed | Missing braces in `ANGLE_RECOMPUTE` — performance bug, not correctness | `surfacerzfourier.cpp:581, 794, 1118` |
| A3 | Confirmed | `RegularGridInterpolant3D` scratch is class-member, thread-unsafe | `regular_grid_interpolant_3d.h:115`; `regular_grid_interpolant_3d_impl.h:114` |
| B1 | Confirmed | Biot–Savart parallelizes over coils, not points | `magneticfield_biotsavart.cpp:44`, `biot_savart_py.cpp:27`, `biot_savart_vjp_py.cpp:34` |
| B2 | Confirmed | VJP kernel: scalar `hadd` / scatter on inner quadrature loop | `biot_savart_vjp_impl.h:74` |
| B3 | Confirmed | Particle tracing: Python-serial + no GIL release | `src/simsopt/field/tracing.py:179, 288`; `python_tracing.cpp:37` |
| B4 | Confirmed | Serial per-coil reduction tail for B / dB / ddB | `magneticfield_biotsavart.cpp:69` (and `compute_A()`) |
| B5 | Confirmed | `std::map<string,…>` cache + `fmt::format` in hot loop | `cache.h:18`; `magneticfield_biotsavart.cpp:36` |
| B6 | Confirmed | Top-level Biot–Savart wrappers repack points per call | `biot_savart_py.cpp:5`, `biot_savart_vjp_py.cpp:5` vs. `magneticfield_biotsavart.h:29` |
| B7 | Confirmed | Tracing output vectors grow with no `reserve()` | `tracing.cpp:371` |
| B8 | Confirmed | `omp simd` hint commented out on non-XSIMD Biot–Savart path | `biot_savart_impl.h:214` |
| B9 | Confirmed | `RegularGridInterpolant3D::evaluate_batch` is serial (gated on A3) | `regular_grid_interpolant_3d_impl.h:59` |
| B10 | Confirmed | Python VJP waste: list comps, `np.sum(v*arr)` temporaries, `any([...])` | `biotsavart.py:34, 44, 54, 73, 83, 85, 111, 118, 125, 135, 145, 164, 174, 176, 202, 209` |
| B11 | Confirmed | `RegularizedCoil.force` rebuilds BiotSavart per call — **4× worse via `coils_to_vtk`** | `coil.py:158, 583` |
| B12 | Confirmed | Redundant `current->get_value()` rereads | `magneticfield_biotsavart.cpp:71, 77, 84, 154, 160, 167` |
| B13 | Confirmed | Small-array stack copies in `solve()` / `join<...>` | `tracing.cpp:353, 389–443` |
| C1 | Partial | CI builds `-march=westmere`; no AVX/FMA in wheels | `CMakeLists.txt:43`; `.github/workflows/wheel.yml:15` — magnitude unbenchmarked |
| C2 | Partial | `std::map<string,…>` cache design vs. dense-indexed vector | `cache.h:18` — gain unbenchmarked |
| C3 | Partial | Dipole kernel untiled point×dipole loops | `dipole_field.cpp:36, 150, 218` — tiling gain speculative |
| D1 | Withdrawn | "Map insert inside `#pragma omp parallel`" was overstated | Keys pre-created serially at `magneticfield_biotsavart.cpp:33–41`; downgraded to B5 |
| D2 | Withdrawn | "Redundant `set_array_to_zero(ddBi)`" was wrong | `Cache::get_or_create` only zeros on first alloc/resize (`cache.h:30-42`) |

---

## 3. Confirmed bugs (A-series)

### A1 — Heap leak in surface Fourier VJPs

Six literal `new double[num_dofs()]` sites; of these, one of each SIMD/non-SIMD pair compiles, so **3 are active per build**. Each sits inside `#pragma omp parallel` and is never freed.

```cpp
// surfacerzfourier.cpp:472 (pattern repeats at 553, 680, 764, 1004, 1088)
#pragma omp parallel
{
    double* resptr_private = new double[num_dofs()];  // never deleted
    ...
    #pragma omp critical
    { for (int i = 0; i < num_dofs(); ++i) resptr[i] += resptr_private[i]; }
}
```

**Fix**: `std::vector<double> resptr_private(num_dofs(), 0.0);` — RAII handles cleanup; the `critical` block then reads `resptr_private.data()`.

**Validation**: AddressSanitizer build + a pytest run that exercises each VJP. Expect "definitely lost" bytes → 0.

### A2 — Missing braces break angle recurrence

```cpp
// surfacerzfourier.cpp:581 (also 794, 1118)
if(i % ANGLE_RECOMPUTE == 0)
    sinterm = sin(m*theta-n*nfp*phi);
    costerm = cos(m*theta-n*nfp*phi);  // runs every iteration
```

`costerm` is unguarded, so the recurrence-based update a few lines below is immediately overwritten. Numerically still correct — just slower on the non-XSIMD build.

**Fix**: add `{ }` around both assignments. One-line diff × 3.

**Validation**: an instrumented counter (compile-time flag) that increments on `cos()` call; ratio of `cos` calls to inner-iterations should be ~`1/ANGLE_RECOMPUTE` after, ~1 before.

### A3 — Thread-unsafe scratch in `RegularGridInterpolant3D`

```cpp
// regular_grid_interpolant_3d.h:115
Vec pkxs, pkys, pkzs;   // class members

// regular_grid_interpolant_3d_impl.h:114
void RegularGridInterpolant3D<Array>::evaluate_local(...) {
    ...
    pkxs[k] = temp[0];  // writes shared state
    pkys[k] = temp[1];
    pkzs[k] = temp[2];
    ...
}
```

Blocks any parallel evaluation of a shared interpolant — notably, multicore particle tracing (B3).

**Fix**: move `pkxs/pkys/pkzs` to function-local arrays (degree is small, so stack allocation is fine), or mark `thread_local`. Prefer function-local.

**Validation**: concurrent `evaluate_batch` on 32 threads against a ground-truth serial run; max |Δ| must be 0.

---

## 4. Confirmed performance issues (B-series)

Summary — see §2 for the ledger and the prior detailed audit for rationale:

- **B1, B2**: change Biot–Savart parallelism axis from coils → points (or `collapse(2)`); restructure VJP inner reduction to stripe over j.
- **B3**: lift tracing loop into C++ with `#pragma omp parallel for` + `py::gil_scoped_release`. Gated on A3.
- **B4**: fold the B/dB/ddB serial tail into the parallel region with a reduction.
- **B5**: replace `std::map<string, CachedArray>` with `std::vector<CachedArray>` indexed by coil id; eliminates `fmt::format` allocation in the hot path.
- **B6**: reuse `pointsx/y/z` buffers across wrapper calls (mirror what `magneticfield_biotsavart.h:29` already does).
- **B7**: `res.reserve(tmax / dtmax_estimate)` in `solve()`.
- **B8**: uncomment `#pragma omp simd` with `aligned(...)` clause on non-XSIMD Biot–Savart.
- **B9**: parallelize `evaluate_batch` (after A3).
- **B10**: cache pre-allocated per-coil buffers in `BiotSavart`; use `np.einsum` / `np.dot` instead of `np.sum(v*arr)`; generator expressions instead of `any([...])`.
- **B11**: cache one `BiotSavart` in `coils_to_vtk`; compute B-fields once before the force/torque loop.
- **B12**: drop redundant `current->get_value()` re-reads.
- **B13**: minor — view step/state instead of copying.

---

## 5. Design hypotheses (C-series) — gated on benchmarks

| Id | Hypothesis | What to measure before claiming gain |
|----|-----------|--------------------------------------|
| C1 | `-march=westmere` CI flag costs wheel users 2–4× on Biot–Savart | Build wheels with `x86-64-v3` baseline; compare `BiotSavart.B` throughput on AVX2 host |
| C2 | `std::map<string>` cache has measurable lookup cost vs. `std::vector` | Per-call profile of `get_or_create` under a realistic optimizer step |
| C3 | Dipole kernel benefits from point×dipole tiling | Roofline + tiled variant benchmark at `n_dipoles ~ 10⁵` |

Until measured: these are **plausible**, not quantified.

---

## 6. Measurement-first upgrade plan

### Phase 0 — Benchmark harness (blocks all perf work)

Goal: every subsequent PR can say *"this fix moves benchmark X by Δ% on reference hardware"* with a reproducible command.

**Tooling choices**

- **asv (airspeed velocity)** for perf tracking over time. Matches numpy/scipy conventions, gives git-bisection and HTML diff reports. Lives in `benchmarks/`.
- **pytest + AddressSanitizer** build for memory-safety gates (A1, A3). `CC=clang CXX=clang++ CFLAGS="-fsanitize=address,undefined"` build, run the existing suite, fail on any leak/UB.
- **ThreadSanitizer** build for A3 / B3 concurrent-access tests.
- **py-spy / perf / Instruments** for on-demand profiling. Not in CI — ad hoc for each PR.
- **`/usr/bin/time -v`** for peak RSS regressions across a reference optimizer run.

**Reference hardware & config**

- Linux x86_64, AVX2-capable (document exact model in `benchmarks/README.md`).
- Pinned thread count (`OMP_NUM_THREADS=16`) for reproducibility of perf runs.
- `taskset` to a fixed cpu set; disable turbo in CI reference runs.
- A second profile: single-thread (`OMP_NUM_THREADS=1`) to isolate single-core kernel gains from parallel speedup.

**Benchmarks to implement (Phase 0 deliverable)**

Representative of real workloads:

1. `bench_biotsavart_forward` — `ncoils=16`, `n_quad=200`, `npoints=10⁴`, 0/1/2 derivs.
2. `bench_biotsavart_vjp` — same shape, `B_vjp` + `B_and_dB_vjp`.
3. `bench_coil_optimizer_step` — one forward + one gradient at realistic stage-II dofs (~200 dofs).
4. `bench_trace_particles` — `nparticles=1000`, synthetic `InterpolatedField`, `tmax=1e-5`.
5. `bench_surface_vjp` — `SurfaceRZFourier`, `mpol=ntor=8`, `numquadpoints=64×64`, all four `_by_dcoeff_vjp` variants.
6. `bench_dipole_field` — `n_dipoles=10⁵`, `npoints=10³`, B + dB.
7. `bench_rss_long_run` — loop the optimizer step 500× under `/usr/bin/time -v`; asserts peak RSS bound (A1 regression guard).

Each benchmark returns a single scalar so asv can track trends.

**Baseline capture**

```bash
asv run HEAD^!                 # baseline the current master
asv publish && asv preview     # HTML dashboard
```

Baseline numbers go into `benchmarks/BASELINE.md` with hardware fingerprint.

### Phase 1 — Bug fixes (A1, A2, A3)

One PR per bug, small and reviewable.

| PR | Target | Merge gate |
|----|--------|-----------|
| P1.1 | A1 leaks | ASan clean on full pytest suite; `bench_rss_long_run` peak RSS bounded (no growth across 500 iters) |
| P1.2 | A2 braces | Dedicated test counting `cos` calls via instrumentation build; `bench_surface_vjp` non-regression |
| P1.3 | A3 interpolant scratch | TSan-clean concurrent `evaluate_batch` test on 32 threads; numerical parity vs. serial |

A1–A3 are independent — can proceed in parallel after Phase 0.

### Phase 2 — Structural perf (B1, B2, B3)

One PR per item, each gated on benchmark numbers in the PR description.

| PR | Target | Expected dominant benchmark | Gate |
|----|--------|----------------------------|------|
| P2.1 | B1 (parallelism axis) | `bench_biotsavart_forward`, `bench_coil_optimizer_step` | ≥20 % wall-time reduction at `OMP_NUM_THREADS=16`; no single-thread regression |
| P2.2 | B2 (VJP stripe reduction) | `bench_biotsavart_vjp` | ≥15 % wall-time reduction; numerical parity to 1e-12 |
| P2.3 | B3 (C++ tracing + GIL release) | `bench_trace_particles` | Linear-ish scaling to `OMP_NUM_THREADS`; parity of particle endpoints |

Prerequisites: P1.3 before P2.3 (interpolant must be thread-safe).

### Phase 3 — Mechanical cleanups (B4–B13)

Group thematically to reduce review load:

- **P3.1** — B4 (reduction tail) + B5 (cache redesign) + B12 (current re-reads): one PR on `magneticfield_biotsavart.cpp`.
- **P3.2** — B6 (point-buffer reuse) + B7 (reserve) + B8 (`omp simd` uncomment): small C++ cleanup PR.
- **P3.3** — B9 (`evaluate_batch` parallel): depends on P1.3.
- **P3.4** — B10 (biotsavart.py) + B11 (coils_to_vtk): Python-side PR.
- **P3.5** — B13: trivial.

Gate: each PR attaches benchmark delta for whichever benchmarks move. If no benchmark moves materially, the PR is justified on correctness/readability alone and labeled accordingly.

### Phase 4 — Hypothesis-driven (C1, C2, C3)

Only merge after measurement shows the expected win on reference hardware.

- **C1** — build a wheel with `-march=x86-64-v3` (or a two-tier wheel strategy), run `bench_biotsavart_forward` and `bench_surface_vjp` on an AVX2 host. If Δ > 25 %, land the CMake flag change + document the baseline bump.
- **C2** — first microbenchmark `Cache::get_or_create` in isolation at 16 keys × 10⁴ lookups. Only pursue the vector-indexed rewrite if the hot-path cost is > 2 % of a full `BiotSavart.compute`.
- **C3** — prototype a tiled dipole kernel on a branch; if `bench_dipole_field` improves by > 20 % at large n_dipoles and no regression at small n_dipoles, land.

### Cross-cutting hygiene

- Every PR runs the full pytest suite (correctness) + the relevant asv benchmark (perf).
- Regression guard: asv comparison vs. master — PR fails CI if any benchmark degrades > 5 % unintentionally.
- Numerical-parity tests in the PR for every kernel whose inner loop was touched. Tolerance ≤ 1e-12 relative (the operation is the same, just re-ordered).
- ASan + TSan builds run on a nightly schedule, not per-PR (too slow), but must be clean before cutting a release.

---

## 7. Rollout order (TL;DR)

1. **Phase 0** — benchmark harness + baseline. Nothing merges until this is in.
2. **A1, A2, A3** in parallel (pure bugs).
3. **B1, B2, B3** in parallel after A3 (structural perf, biggest wins).
4. **B4–B13** in grouped PRs (mechanical).
5. **C1, C2, C3** only after microbenchmarks justify them.

The final `BASELINE.md` delta table becomes the validated replacement for the withdrawn speedup table — one row per fix, measured numbers, reference hardware, reproducible command.

---

## 8. Appendix — tooling commands

```bash
# Phase 0: asv setup (one-time)
pip install asv
asv machine --yes
asv run HEAD^! --show-stderr

# ASan build for A1 / A3 gating
CC=clang CXX=clang++ \
  CFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" \
  CXXFLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" \
  pip install -e . --no-build-isolation
ASAN_OPTIONS=detect_leaks=1 pytest tests/

# TSan build for A3 / B3 gating
CC=clang CXX=clang++ \
  CFLAGS="-fsanitize=thread -g -O1" \
  CXXFLAGS="-fsanitize=thread -g -O1" \
  pip install -e . --no-build-isolation
OMP_NUM_THREADS=16 pytest tests/test_concurrent_trace.py

# Per-PR benchmark delta
asv continuous master HEAD

# RSS regression
/usr/bin/time -v python benchmarks/long_optimizer_run.py
```
