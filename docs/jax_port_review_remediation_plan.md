# JAX Port — Review Remediation Plan

## Purpose

Track and execute remediation of every issue surfaced by the 2026-05-29
three-dimension review of the C++→JAX port (architecture/structure matching ·
JAX best-practices & modularity · `SOFTWARE_DESIGN.md` conformance). The review
ran 7 read-only Opus subagents (lens × file-region decomposition) over ~69K LOC
JAX source + ~90K LOC tests across 428 commits since 2026-03-01.

This file is the single backlog for the findings. Each item carries its review
ID, `file:line` evidence, severity, and confidence so it can be executed without
re-deriving the analysis.

## Headline

**No Critical findings and no confirmed correctness bugs at any layer.** The
`jax_core` purity boundary holds with zero import-level violations; the
implicit-diff failed-adjoint→non-finite-gradient integrity contract is
implemented as documented; the previously-flagged test tautologies are fixed.
**Every item below is maintainability (SSOT/DRY), module-responsibility,
perf/idiom, or doc-accuracy — not correctness.** Severity is therefore relative
to *complexity/perf leverage*, not bug risk.

Legend — Sev: `Med` (change-amplification / measurable perf / SSOT) · `Low`
(localized smell / nit). Conf: `H/M/L`. `✓verified` = independently confirmed
against the live tree during the review.

## Goals

- Eliminate the confirmed SSOT/change-amplification hotspots (2π device scalar,
  `__all__`/`_EXPORT_MODULES`, `host_cache_array` duplication).
- Fix the one clear JAX-idiom inconsistency (surface form-Jacobian `jacrev`→`jacfwd`).
- Reduce the three solver/objective giants' design debt (untyped result dict,
  resolver pass-through matrix, multi-responsibility module) without changing
  observable behavior.
- Remove measurable perf waste in the tracing hot path and the curve DOF mapping.
- Bring `docs/` and `CLAUDE.md` back into line with the live tree.
- Resolve the test-quality nits and close the named review coverage gaps.

## Non-Goals

- No change to numerical contracts, parity-ladder tolerances, or public
  Optimizable APIs (Tier-3 surface) unless a dedicated API-evolution gate is run.
- No CPU/C++ reference-oracle changes.
- Not a re-derivation of the review; not a correctness re-audit (the review
  found none — Phase 7 only *closes coverage gaps*, it does not presume bugs).
- No crusade refactors beyond the file/function scope named in each task.

## Current Context

- Branch: `gpu-purity-stage2-20260405`. Remediation implementation and status
  packaging had reached `1162ae851` (`docs(jax): record current CUDA signoff
  gate`) before the current-head CUDA signoff-packet update. The broad
  remediation implementation is in `f287bde96` (`refactor(jax): close port
  remediation review`) and tracing-caveat closure is in `8fa1b9a41`
  (`test(jax): close tracing comm replay caveat`). The remaining untracked
  worktree entries are local artifacts (`.antigravitycli/`, `.conda/`,
  `analysis/`, `runs/`); preserve them and scope any future staging operation
  to the intended remediation files only.
- Layering convention (verified working): `jax_core/*.py` = pure compute SSOT
  (simsoptpp-free, no `Optimizable`); `{field,geo,objectives,solve,mhd}/*_jax.py`
  = Optimizable adapters/shims; `*_cpu_ordered.py` = parity twins; `backend/` =
  mode-selection SSOT.
- Canonical device-scalar helper already exists: `_device_scalars.two_pi`
  (`src/simsopt/jax_core/_device_scalars.py:13`, device-pure via `arccos(-1)`).
- Canonical host-boundary helper already exists:
  `field/_jax_common.host_cache_array` / `_core/jax_host_boundary.host_array`.
- Resolver SSOT table already exists: `_BOOZER_INNER_DRIVER_OPTIONS`
  (`geo/optimizer_jax.py:601-642`).

## Rationale

The port's structure is sound, so remediation is sequenced by **leverage ×
risk**, not by subsystem. Phase 1 batches the Low-risk, High-confidence SSOT and
idiom wins (disjoint files, no contract change → one clean Tier-1b/2 PR). The
Medium-risk module restructures (Phases 3) are split into individually-reviewed
PRs because each touches a 5–7K-line giant and must prove behavior preservation.
Faithful-port AD-safety items (Phase 4) are conditional — they are only defects
*if those kernels are ever differentiated near their physical singularities*, so
they are gated on confirming an AD consumer rather than fixed speculatively.

## Assumptions

- `surface_rzfourier.py`'s `jacfwd` choice is the correct precedent for the
  XYZ/Tensor form-Jacobians (same tall DOF→grid shape). [High confidence —
  ✓verified the twin asymmetry.]
- `boozersurface_jax.res` dict shape is constrained by CPU `BoozerSurface.res`
  dict-compatibility; a typed-record migration must keep dict-style access or
  provide a compat view. [Confirmed 2026-05-29 — item 12 keeps a `dict`
  subclass result record for stateful public solver paths.]
- The `*_fast` parity-lane semantics and `transfer_guard("disallow")` policy are
  unchanged by 2π consolidation as long as the canonical helper is device-pure.
  [High confidence, but Phase 1 validation must include a transfer-guard smoke.]

## Implementation Plan

### Phase 1 — Bounded SSOT + idiom cleanup (Low-risk, High-confidence; single PR)

1. **Surface form-Jacobian `jacrev`→`jacfwd`** [B-1 / J1 · Med · H · ✓verified]
   - [x] In `jax_core/surface_fourier.py`, switch the 6 `jax.jacobian(...)` form/curvature
     Jacobian sites to `jax.jacfwd`: lines `446`, `455`, `464`, `874`, `883`, `892`
     (`dfirst_fund_form`, `dsecond_fund_form`, `dsurface_curvatures` for both the
     `_from_dofs` and paired variants). Match `surface_rzfourier.py:566`
     (`_evaluate_jacobian_from_dofs` already uses `jacfwd`).
   - [x] Replace any remaining implicit `jax.jacobian` derivative sites with an
     explicit `jacfwd`/`jacrev` so the forward/reverse choice is visible [J6].

2. **Consolidate 2π onto the device-pure canonical helper** [C-1 / C1-F1 · Med · H · ✓verified]
   - [x] Replace host-literal 2π constructions *inside `jax_core`* with
     `_device_scalars.two_pi(reference)`:
     `jax_core/surface_henneberg.py:46-48` (`_two_pi_like` → host literal),
     `jax_core/tracing.py` (`_device_array(2.0*np.pi, dtype)` at `:609,645,695,1145,1831,3007`),
     `jax_core/interpolated_boozer_field.py:280` (`_device_float64(2.0*np.pi)`).
   - [x] Decide the policy for the duplicate `_two_pi_like` defs in `geo/curve.py:104`
     (+ `_TWO_PI = 2.0*np.pi` at `:95`) and `geo/curve_rz_fourier.py` /
     `geo/surface_henneberg.py`: either route through `_device_scalars` or add a
     one-line comment justifying a host literal in a CPU/host-only path. Do NOT
     touch pure-host CPU modules (`geo/surfacerzfourier.py`, `geo/curvecwsfourier.py`)
     where a host literal is correct.
   - [x] Confirm no behavioral delta under `transfer_guard("disallow")` (the whole
     point of the device-pure form).

3. **Single-source the `jax_core` export lists** [C-5 / C1-F2 · Med · H · ✓verified]
   - [x] In `jax_core/__init__.py`, derive `__all__` from `_EXPORT_MODULES`
     (`:18` and `:335`, both currently 314 hand-synced entries) so one list is the
     SSOT; keep the lazy `__getattr__` (`:670`) mechanism. Verify `_EXPORT_MODULE_OBJECTS`
     (`:652`) stays consistent or is likewise derived.

4. **Remove `host_cache_array` duplication** [C-6 / C1-F3 · Med · H]
   - [x] In `geo/finitebuild.py:47`, import and use
     `field/_jax_common.host_cache_array` (or the underlying
     `_core/jax_host_boundary.host_array`) instead of the hand-rolled
     `_host_cache_array`.
   - [x] Decide `host_cache_array`'s own status [C1-F4 · Low]: it is a thin
     pass-through (`_jax_common.py:28-30`). Either keep it as the documented
     field-wrapper boundary alias (pairs with `points_device`) or collapse callers
     onto `host_array`. Document the decision in a one-line comment.

5. **Doc-accuracy refresh** (bundled into Phase 1 since it's zero-risk)
   - [x] `docs/cpp_to_jax_port_file_map.md`: add the ~7 omitted shipped wrappers
     (`objectives/stage2_target_objective_jax`, `field/sampling_jax`,
     `solve/serial_jax`, `solve/mpi_jax`, `mhd/vmec_diagnostics_jax`,
     `mhd/bootstrap_jax`, `mhd/profiles_jax`) under the Python-only / solve-mhd
     section [A-1 · Med].
   - [x] `CLAUDE.md`: list `field/magneticfieldclasses_jax.py` as the 4th
     re-export shim in the M6/shim inventory [A-2 · Low].
   - [x] `CLAUDE.md`: refresh stale line anchors — LS-PLU callbacks
     `3514-3540`→`_build_runtime_linear_solve_callbacks` at
     `boozersurface_jax.py:3938`; `_traceable_solve_plu_linearization`
     `3167-3220`→`surfaceobjectives_traceable_jax.py:419`
     [C-8 / C2-F7 · Low · ✓verified].

### Phase 2 — Localized perf / idiom (Low-risk; can ride Phase 1 or a follow-up)

6. **Curve DOF mapping: dense matmul → slice/scatter** [B-3 / J2 · Med · M]
   - [x] Replace the concrete-evaluation `O(N²)` one-hot selector matmuls in
     `jax_core/curve_geometry.py:91-106` (`_slice_1d_static` /
     `_update_1d_static`) with static `lax.slice_in_dim` segment assembly.
     Traced/autodiff calls keep an explicit device-placed selector fallback
     because JAX's reverse-mode transpose for `lax.slice` currently introduces
     an implicit scalar zero under strict transfer guard. The `curve_kernels.py`
     reference is stale for this helper; remaining selector matrices are
     Fourier-mode selectors and stay benchmark-gated.

7. **`boozer_fixed_state` static-condition dead branch** [B-5 / J4 · Low · H]
   - [x] In `jax_core/boozer_fixed_state.py:246-256`, replace
     `jnp.where(spec.no_K, ...)` (static `meta_field`) with a Python `if spec.no_K:`
     so the dead K/dKdθ/dKdζ Fourier sum is not computed (mirror
     `boozer_radial_field.py:461,479`).

8. **CPU-ordered Biot-Savart scalar stacking** [J3 · Low · H]
   - [x] In `jax_core/biotsavart_cpu_ordered.py` (`:68,95,116-118,121-123`), replace
     `jnp.array([cx,cy,cz])` of traced scalars with `jnp.stack((cx,cy,cz))`.
     (Parity-only twin; bounded impact.)

9. **Single-point interpolated-field eval** [J5 · Low · M]
   - [x] Confirm the caller of `interpolated_field_state_B[_GradAbsB]`
     (`jax_core/interpolated_field.py:600-612,637-657`) is the traced `lax.scan`
     body (then it's a non-issue and gets a clarifying comment). If it is called
     in a hot Python step-loop, add a non-`(1,3)`-wrapping scalar path.

10. **PM scale via top-singular-value, not full SVD** [F9 · Low · H]
    - [x] Deferred: in `geo/permanent_magnet_grid_jax.py:151`, replacing
      `jnp.linalg.svd(A, compute_uv=False)` (used for one number) with a
      power-iteration / `eigh(AᵀA)` top-value estimate remains measurement-gated;
      no current benchmark proves the swap is worth the behavior risk.

11. **`mhd_bootstrap` knot-matrix vectorization** [F10 · Low · H]
    - [x] In `jax_core/mhd_bootstrap.py:33-57` (`_not_a_knot_coefficients`), replace
      the per-row Python-loop `.at[].set()` with a single batched
      `.at[rows,cols].set(vals)`.

### Phase 3 — Medium-risk module restructures (each its own reviewed PR; behavior-preserving)

12. **`boozersurface_jax.res`: untyped dict → typed records** [C-3 / C2-F1 · Med · H · done]
    - [x] Replace the 10-mode-shape dict + parallel `_BoozerResultSchema`
      validator (`geo/boozersurface_jax.py:195-456`; schema instances `lbfgs`,
      `ls_manual`, `ls_lm`, `newton`, `exact`, `exact_constraints`, `traceable`,
      `traceable_exact`, `traceable_ls`) with typed result records (NamedTuple/
      frozen dataclass per mode, or a sealed union), keeping a dict-compat view if
      CPU `BoozerSurface.res` access requires it (see Open Questions).
    - [x] Once typed, the `_BoozerResultSchema` validation layer becomes redundant —
      remove or downgrade it.
    - [x] 2026-05-29 implementation: `geo/boozersurface_jax.py` now uses
      `_BoozerResultRecordType` plus mutable `_BoozerResultRecord` dict-subclass
      wrappers for stateful public `BoozerSurfaceJAX.res` results. The actual
      solver paths store records through `_store_boozer_result(...)`, preserving
      `isinstance(res, dict)`, `res["..."]`, `.get()`, `.keys()`, `dict(res)`,
      `.copy()`, and targeted callback mutation used by existing SIMSOPT-style
      consumers. The result record is registered as a JAX pytree with the same
      sorted-key leaf behavior as a plain dict. Direct incomplete test/compat
      assignments remain plain dicts. The traceable pure-array path remains a
      plain JAX pytree and is validated against the typed traceable record
      contracts without wrapping the traced return.
    - [x] Validation: py-compile for `boozersurface_jax.py` and the touched
      Boozer tests; 9 focused public/exact/traceable result-record tests; 8
      adjoint-runtime-state mapping tests; 9 integration wrapper crash-guard
      tests that mutate `booz_jax.res`; and a broader 29-test Boozer selector
      covering result-record, exact-constraints, LS, exact, traceable, and
      mixed-quadrature result paths all passed locally on the CPU JAX backend.
      CUDA validation is not claimed here because local `nvidia-smi` is absent
      and JAX reports only CPU devices in this environment.
    - [x] Official-doc cross-check: SIMSOPT `Optimizable`/objectives docs keep
      the stateful optimization-object compatibility frame; JAX transfer-guard
      and benchmarking docs keep host/device and timing claims scoped; JAX GPU
      memory and NVIDIA CUDA best-practices docs inform CUDA caveats; SciPy
      `minimize`/`least_squares` docs confirm the reference optimizer result
      adapters remain external-library boundary objects rather than JAX pytrees.
      A read-only subagent review found two item-12 regressions in the initial
      implementation (`isinstance(res, dict)` consumers and stale open-status
      text); both were fixed before final validation by making the record a
      `dict` subclass with JAX pytree registration and by updating this status
      section/Open Questions.

13. **`optimizer_jax` resolver matrix collapse** [C-4 / C2-F2 · Med · H · done]
    - [x] Collapse the ~18 `resolve_{reference,target}{,_least_squares,_outer_loop}_optimizer_{driver,method,contract}`
      functions (`geo/optimizer_jax.py:1124-1451`; pass-throughs via
      `_jax_driver.py:80-121`) so `*_method` variants stop re-deriving backend/lane
      logic; route all resolution through the existing typed
      `_BOOZER_INNER_DRIVER_OPTIONS` table (`:601-642`).
    - [x] Implemented with a table-derived inverse
      `_BOOZER_INNER_DRIVER_BY_OPTIONS` and shared method/contract helpers
      (`geo/optimizer_jax.py:643-735`). Public backend, reference, target,
      residual least-squares, SciPy-control, and public-LBFGS lanes now reuse the
      same typed driver/contract route instead of rebuilding lane-specific
      method strings.

14. **`surfaceobjectives_jax` responsibility split** [C-2 / C2-F3 · Med · M · done]
    - [x] Extract the `_traceable_*` runtime-bundle / custom-VJP / cache subsystem
      (~3000–6284, ~60 `_traceable_*` helpers) from
      `geo/surfaceobjectives_jax.py` into a sibling module (e.g.
      `surfaceobjectives_traceable_jax.py`), leaving the IFT wrapper classes
      (`_BoozerObjectiveBase`, `BoozerResidualJAX`, `IotasJAX`,
      `NonQuasiSymmetricRatioJAX`, `MajorRadiusJAX`) and surface-metric kernels.
      Preserve `__all__` (`:138-181`) re-exports for import-path stability.
    - [x] 2026-05-29 implementation: `geo/surfaceobjectives_traceable_jax.py`
      now owns the traceable runtime/cache/custom-VJP block from
      `_traceable_iota_from_x_inner` through
      `make_traceable_objective_profile_suite`. `surfaceobjectives_jax.py`
      keeps the IFT wrappers/surface kernels and explicit compatibility
      re-exports for legacy public builders and direct private-helper imports.
      Traceable-internal tests patch the new implementation module, while a
      compatibility test asserts old-path public/private helper identity.
    - [x] 2026-05-30 follow-up: the traceable least-squares method gate now
      reuses `boozersurface_jax._ONDEVICE_OPTIMIZER_METHODS` instead of a stale
      literal `{bfgs, lbfgs, lm}-ondevice` allow-list, so supported
      `lm-minpack-ondevice` and `optimistix-lm-ondevice` single-stage target
      lanes are not rejected before runtime-state construction.

15. **`_normalize_solver_options` → declarative incompat table** [C2-F4 · Low-Med · M · done]
    - [x] Refactor the ~120-line procedural cross-knob compatibility matrix
      (`geo/boozersurface_jax.py:3381-3499` + `_apply_inner_driver_option:3342-3378`)
      into a declarative incompatibility table. Behavior-preserving; validation
      stays loud.

16. **Remove `_BoozerSolverOptions(dict)` reactive mutation** [C-7 / C2-F5 · Low · M · done]
    - [x] In `geo/boozersurface_jax.py:3302-3335`, stop overriding
      `__setitem__`/`update` to silently recompute the dense-linearization default
      on a *different* key; compute that default explicitly at construction using
      the existing explicit-flag plumbing.

### Phase 4 — Faithful-port AD-safety audit (conditional; investigate-then-fix)

> These divide/sqrt sites are faithful ports; they are defects only if the kernel
> is differentiated near its physical singularity. Gate each on confirming an AD
> consumer before adding a `jnp.where` guard.

17. **Audit and guard differentiated diagnostics** [B-6 / F4–F7 · Low · M]
    - [x] `jax_core/vmec_geometry.py`:
      current AD consumers exist away from the named poles; source and public
      wrapper docs now record the inherited `s=0` drift-normalization and
      zero-`grad_B` diagnostic-domain limits. Add numerical guards only with a
      model-specific limiting treatment and a failing singular AD regression.
    - [x] `jax_core/qfm_solver.py:215,224`: division by `norm_normal` and
      `∫|B|²dA` is differentiated via `value_and_grad`; implemented
      `_row_norm_without_zero_sqrt_gradient`-style safe operands and regression
      coverage.
    - [x] `jax_core/redl_current.py:153,234` (`sqrt(Zeff-1)`, `1/(iota-helicity_N)`):
      current AD consumers stay away from the poles; source and caller docs now
      document faithful-port singularities. Guard only if a supported objective
      differentiates at a pole.
    - [x] `jax_core/mhd_bootstrap.py` and
      `jax_core/magnetic_axis_helpers.py`: pre-mask operands before
      `where` / `sqrt` so `jacfwd` does not pick up dead-branch NaNs. Regression
      coverage now includes flat `|B|` trapped-fraction JVP and zero-discriminant
      magnetic-axis JVP finiteness.
    - [x] `jax_core/tracing.py` (`dv_par` banana-tip `v_par=0`):
      source docs now mark this as an inherited physical singularity; no fix
      unless AD through tip-crossing is required.

18. **MwPGP `lax.cond` under future vmap** [F3 · Low · H — preventive]
    - [x] If `pm_optimization.py` solvers are ever `vmap`'d across geometries,
      switch `lax.cond` at `:3202,3209` to `lax.select` (both arms already execute
      under vmap). Currently compile-per-geometry, so no action needed now — leave
      a comment.

### Phase 5 — Architecture convention nits (Low; opportunistic)

19. **`objectives/__init__.py` guard idiom** [A-3 · Low]
    - [x] Optionally normalize `objectives/__init__.py:27-34` (`try/except`) to the
      `_has_jax = importlib.util.find_spec("jax") is not None` flag pattern used by
      `field/geo/solve/mhd/__init__.py`. Consistency-only.

20. **`stage2_target_objective_jax.py` privacy signal** [A-4 · Low]
    - [x] Either prefix `_stage2_...` (matches `_distance_jax.py`) or add a header
      comment + `CLAUDE.md` note documenting it as a private optimizer-lane bundle
      (currently unexported, discoverable by path).

21. **XYZ vs XYZTensor cross-reference** [A-5 · Low]
    - [x] Add a one-line header cross-ref in `jax_core/surface_fourier.py` (XYZ)
      and `jax_core/surface_fourier_kernels.py` (XYZTensor) so the two are not
      conflated.

22. **Infra-helper nits** [C1-F5/F6/F7 · Low]
    - [x] `jax_core/_finite_difference.py:54-61`: reuse one representation (or
      rename) the abs/rel step values that are built twice (NumPy then device).
    - [x] Consider `ArrayLike` instead of `: object` on `jax_core/field.py` public
      eval entries where `points` flows into `.shape` (e.g. `:286`); leave the
      `specs.make_*` `: object` coercion-boundary params as-is. Decision: leave
      `: object` on trace/coercion boundaries to avoid overpromising accepted
      array protocols.
    - [x] Trim the `field/_jax_common.py:16-25` `points_device` interface comment to
      ≤5 lines (noun + contract), letting the linked helper own the mechanism.

### Phase 6 — Test-quality nits (Low)

23. **Remove export-trivia / circular-verdict tests** [D-F2, D-F3 · Low · H]
    - [x] Delete or fold the re-export `is`-identity tests at
      `tests/solve/test_serial_jax.py:316-318` (the audit already deleted two
      sibling files; this one slipped through).
    - [x] In `tests/test_jax_import_smoke.py:1094`, assert component fields instead
      of the driver's self-reported `payload["passed"] is True` (mirror the CUDA
      probe pattern #22).

24. **Label borderline self-consistency oracles** [D-F1, D-F5 · Low · H/M]
    - [x] Add a one-line docstring to
      `tests/geo/test_boozer_residual_jax.py:489,586,616` labeling it "Tier-4
      reduction-order self-consistency, not C++ parity" (independent C++ parity is
      `test_scalar_matches_cpp_oracle:555`).
    - [x] Confirm `test_coil_cotangent_projection_matches_explicit_sum`
      (`tests/integration/test_single_stage_jax_cpu_reference.py:2719`) is
      adequately backed by the BiotSavart C++ parity suite; no change if so
      (D-F4 streamed-VJP smoke is already self-flagged — no action).

### Phase 7 — Close review coverage gaps (investigate-only; no presumed bugs)

25. **`tracing.py` design-lens review + driver dedup** [B1b pattern · Med · H]
    - [x] Independently design-review `jax_core/tracing.py` (only idiom-reviewed so
      far). Factor the shared ~120-line body duplicated across the 4 trajectory
      drivers (fieldline / GC-Cartesian / GC-Boozer / fullorbit) so a fix like the
      event-localizer recompute (item 26) is applied once, not 3–4×.

26. **Event-localizer DOPRI5 recompute** [B-2 / F1 · Med · H]
    - [x] Deferred: in `jax_core/tracing.py:1410` (fieldline), `:2070` (Cartesian GC),
      `:3241` (Boozer zeta), and `:3968` (full orbit), avoid re-running a full
      7-eval DOPRI5 step per Illinois iteration × event plane: cache the
      dense-output / 5th-order interpolant once per accepted step, or reduce
      `max_root_iters`. Item 25 centralizes the angle-plane helper; a dense-output
      patch remains profiler-gated to preserve event accuracy.

27. **`per-dipole/segment lax.scan` GPU serialization** [B-4 / F2 · Med · M]
    - [x] Evaluate a chunked/`vmap`-batched variant of the contributor loops in
      `jax_core/dipole_field.py:163,199,253,325` and `jax_core/wireframe.py:354,375,423,442`
      for the case where `npoints×M` fits memory (current `scan` trades parallelism
      for memory deliberately — measurement gate keeps current streaming structure).

28. **`_from_spec` / `_from_dofs` twin-API DRY question** [C1 gap · Low · M]
    - [x] Trace whether the ~120 `surface_*_..._from_spec` / `_from_dofs` twin
      symbols (`jax_core/__init__.py` exports) are genuine shared-knowledge
      duplication or an intentional two-entry contract; deduplicate only if they
      change together.

29. **Exact-path batched adjoint VJP-pullback reuse** [B-7 / J1(B2) · Low · H]
    - [x] In `geo/optimizer_jax.py:4168-4220`, build the VJP pullback once per
      adjoint-state and reuse `matvec`/`transpose_matvec` across RHS columns rather
      than rebuilding `_jacobian_linear_operator` per RHS
      (`surfaceobjectives_jax.py:1955-1957`). Bounded (few RHS); low priority.

30. **Latent / cosmetic micro-items**
    - [x] `geo/surfaceobjectives_traceable_jax.py`: use the shared runtime int32
      scalar helper for traceable `nit` placeholders instead of
      `jnp.asarray(..., int64)` [B-8 / J2 · Low].
    - [x] `geo/boozersurface_jax.py`: collapse the Newton-polish PLU finalizer's
      eager dual-branch + `jnp.where` into a single `lax.cond` on `finite`
      [J3(B2) · Low].
    - [x] `jax_core/specs.py:26-134`: add `eq=False`/`unsafe_hash` discipline or a
      comment so a future use of a `frozen=True` array-bearing spec as a dict key
      cannot hit the auto-`__hash__`-over-arrays foot-gun [B-9 / J4(B2) · Low,
      latent — no current trigger].
    - [x] Note `C2-F6` (dozens–hundreds of underscore-private helpers per giant) is
      a *symptom* of items 12/13/14, not an independent task — re-check after those
      land.

## Validation Plan

The checklist below is the original release-gate plan. Checked boxes now mean
the current worktree has focused evidence recorded in the execution log; broad
hardware or full-suite gates stay unchecked unless they actually completed.

Per `CLAUDE.md`, with `PYTHONNOUSERSITE=1 PYTHONPATH=src JAX_ENABLE_X64=True JAX_PLATFORM_NAME=cpu`:

- [x] `ruff check` + `ruff format` on every changed file (`.conda/jax/bin/python -m ruff ...`).
- [x] **Phase 1 item 1** (surface jacfwd): `tests/geo/test_surface_fourier_jax.py`,
      `tests/geo/test_surface_rzfourier_jax.py`, `tests/geo/test_boozer_derivatives_jax.py`.
- [x] **Phase 1 item 2** (2π): import smoke `tests/test_jax_import_smoke.py` +
      a `transfer_guard("disallow")` smoke (the device-purity invariant) +
      `tests/jax_core/test_tracing_jax_*` (tracing uses 2π).
- [x] **Phase 1 item 3** (exports): `python -c "import simsopt.jax_core"` and a
      `len(__all__) == len(_EXPORT_MODULES)` assertion; import smoke suite.
- [x] **Phase 1 item 4** (host_cache_array): `tests/...finitebuild...` +
      `tests/objectives/test_fluxobjective_jax_parity.py`.
- [x] **Phase 2** (curve/biotsavart/interp perf): `tests/field/test_biotsavart_jax.py`,
      curve geometry tests, `tests/jax_core/test_tracing_jax_*`.
- [x] **Phase 3** (giants): full Boozer + single-stage lanes —
      `tests/geo/test_boozersurface_jax.py`, `tests/geo/test_boozersurface_jax_private.py`
      (`-m private_optimizer_runtime`), `tests/integration/test_single_stage_jax*.py`;
      diff `self.res` keys before/after to prove no observable result-dict change.
- [x] **Phase 4** (AD-safety): add an FD/`jacfwd`-finiteness regression for each
      guarded kernel that proves the old behavior produced NaN-grad at the
      singularity (per SOFTWARE_DESIGN "prove-it-fails"); record source-level
      domain documentation for faithful physical singularities that are not
      numerically regularized.
- [x] **Phase 6** (tests): run the edited test files; confirm net assertion count
      did not silently drop coverage.
- [x] Full integration sweep before declaring any Phase 3 PR done:
      `.conda/jax/bin/python -m pytest tests/integration/ -v` equivalent
      completed on CPU; counts recorded below.
- [x] No mypy/ruff regression on touched files (pre-existing upstream errors OK).

## Risks and Mitigations

- **Risk:** 2π consolidation changes a parity-lane byte result if a touched site
  was relying on the host-literal path.
  **Mitigation:** the canonical helper is device-pure; run the `transfer_guard`
  smoke + tracing tests; keep host literals in pure-host CPU modules.
- **Risk:** Typing `boozersurface_jax.res` breaks a CPU-compat consumer that does
  `res["key"]`.
  **Mitigation:** item 12 keeps stateful public solver results as a `dict`
  subclass, preserves mapping mutation/copy/key APIs, registers the record as a
  JAX pytree, and diffs result-record keys in focused tests.
- **Risk:** `surfaceobjectives_jax` module split breaks an import path used by
  research scripts.
  **Mitigation:** keep `__all__` re-exports from the original module; grep the
  workspace for `from simsopt.geo.surfaceobjectives_jax import`.
- **Risk:** `jacrev`→`jacfwd` changes a derivative value beyond float noise.
  **Mitigation:** the RZ twin already uses `jacfwd` at the same shape; assert
  parity to the existing FD/C++ derivative oracles at the `derivative-heavy` lane
  tolerance, not byte-identity.
- **Risk:** Phase 7 perf changes (driver dedup, scan→vmap) regress memory on GPU.
  **Mitigation:** measure before/after per SOFTWARE_DESIGN Performance rule; roll
  back any change that doesn't measurably help and didn't simplify.

## Completion Criteria

Current status for this local remediation execution:

- [x] Phase 1 (items 1–5) implemented in the worktree; SSOT duplications gone;
      surface Jacobian uses `jacfwd`; docs/CLAUDE.md anchors refreshed.
- [x] Phases 2 perf/idiom items either done or explicitly deferred with a
      reason.
- [x] Each Phase 3 giant-restructure has focused behavior-preservation evidence,
      result-dict/import-path parity checks where applicable, and subagent review
      evidence recorded below.
- [x] Phase 4 items are either guarded with regression tests or marked as
      faithful physical singularities with no supported AD consumer at that pole.
- [x] Phase 6 test nits resolved; no intentional coverage drop.
- [x] Phase 7 coverage-gap investigations produced written verdicts (fix,
      defer, or confirmed-non-issue) for items 25–29.
- [x] Touched files pass ruff/format/compile and focused lane tests recorded
      below; unrelated dirty/untracked files are preserved.
- [x] Full unfiltered `tests/integration/` pytest pass counts recorded:
      `486 passed, 9 skipped, 8 warnings in 3769.76s (1:02:49)`. The shell
      wrapper exited nonzero after pytest because it assigned zsh's reserved
      `status` parameter; no pytest failure remained.
- [ ] CUDA/GPU signoff recorded on a CUDA-capable host, or explicitly waived by
      the release owner.
- [x] Local commit packaging completed for the implementation commits:
      broad remediation in `f287bde96`, tracing-caveat closure in `8fa1b9a41`,
      and pre-packet CUDA signoff-gate status in `1162ae851`. The current-head
      CUDA packet below intentionally derives the expected SHA from
      `git rev-parse HEAD` at run time so the packet remains valid after doc-only
      packaging commits. PR/merge publication remains external if this plan is
      used as a release-merge gate.

## Execution Status — 2026-05-29 Live Tree

This status was written during the `$requirements-e2e-review-loop` implementation
pass. It is intentionally conservative: anything not proven by current-tree code
or a focused command remains open.

### Official documentation cross-check

- **JAX:** the official transfer-guard docs distinguish explicit
  `jax.device_put*()` / `jax.device_get()` transfers from implicit transfers,
  and `disallow` blocks implicit transfers while allowing explicit transfers
  (<https://docs.jax.dev/en/latest/transfer_guard.html>). This supports the
  current transfer-guard framing and keeps host/device crossings explicit,
  including the item 14 host-wrapper-only `transfer_guard_host_to_device("allow")`
  setup boundary.
- **JAX autodiff and timing:** the official autodiff cookbook recommends
  `jacfwd` for tall Jacobians and `jacrev` for wide Jacobians
  (<https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html>), matching
  the Phase 1 surface Fourier `jacfwd` change. The official benchmarking guide
  requires `.block_until_ready()` for real JAX timings and recommends
  pre-placing benchmark inputs with `jax.device_put`
  (<https://docs.jax.dev/en/latest/benchmarking.html>). The GPU memory guide
  documents JAX's default 75% GPU preallocation and the fragmentation tradeoff
  when `XLA_PYTHON_CLIENT_PREALLOCATE=false`
  (<https://docs.jax.dev/en/latest/gpu_memory_allocation.html>).
  The official JAX FAQ documents NaN gradients from undefined `jnp.where`
  branches and recommends protecting the operand fed to the undefined operation,
  which is the Phase 4 safe-operand pattern for `sqrt` / division sites
  (<https://docs.jax.dev/en/latest/faq.html#gradients-contain-nan-where-using-where>).
  The official `lax.scan` and control-flow docs keep the Phase 7 item 25
  tracing refactor bounded: loop-carried values must keep fixed shape/dtype,
  and static loop bounds / compile-time control values should stay explicit
  (<https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>,
  <https://docs.jax.dev/en/latest/control-flow.html>).
- **NVIDIA CUDA:** the CUDA C++ Best Practices Guide says to minimize
  host/device transfers, create and operate on intermediates in device memory,
  measure effective bandwidth, and use synchronized CUDA-event timing for GPU
  work (<https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>).
  It also marks coalesced global-memory access as high priority. NVIDIA's
  Nsight Systems docs cover CUDA API/workload timeline tracing, including
  host-device copies and kernel executions
  (<https://docs.nvidia.com/nsight-systems/UserGuide/index.html>), and Nsight
  Compute is the kernel-profiler path for per-kernel metrics
  (<https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>). This
  supports deferring Phase 2 item 10 and Phase 7 items 26/27 until a synchronized
  GPU benchmark or profiler trace proves the replacement is beneficial. Phase 7
  item 29 is a direct reuse of an already-built exact-Jacobian operator rather
  than a new numerical method.
- **SIMSOPT:** the official Optimizable docs define the DAG/dof ownership model,
  `x` / `full_x` access contract, and cache invalidation on dof changes
  (<https://simsopt.readthedocs.io/latest/optimizable.html>). The SIMSOPT API
  docs expose `SquaredFlux(surface, field, target=None, definition=...)` and
  Boozer residual semantics
  (<https://simsopt.readthedocs.io/v1.10.0/simsopt.objectives.html>,
  <https://simsopt.readthedocs.io/stable/simsopt.geo.html>). SIMSOPT's MHD docs
  expose VMEC fieldline diagnostics and Redl bootstrap-current objectives as
  public physics APIs
  (<https://simsopt.readthedocs.io/stable/simsopt.mhd.html>). The upstream
  README also documents modular optional physics modules
  (<https://github.com/hiddenSymmetries/simsopt>). These sources support the
  Phase 3 requirement to preserve CPU public API / result-dict compatibility and
  the Phase 4 requirement to distinguish faithful physical singularities from
  JAX-only dead-branch AD hazards, plus the Phase 5 optional-JAX import gate
  instead of broad import suppression.
- **Optimizer APIs:** SciPy's official `minimize` docs expose `BFGS` and
  `L-BFGS-B` as method strings and document the `fun`/`jac` contract
  (<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>).
  SciPy's `least_squares` docs keep `method='lm'` as the MINPACK
  Levenberg-Marquardt lane with residual-vector/Jacobian semantics
  (<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>).
  Optax documents its JAX L-BFGS optimizer and value/gradient state flow
  (<https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html>),
  and Optimistix documents solver function-information variants including
  `EvalGrad`, `Residual`, and `ResidualJac`
  (<https://docs.kidger.site/optimistix/api/searches/function_info/>). These
  support preserving the existing scalar-method versus residual-least-squares
  lane split in Phase 3 item 13.

### Implemented in the current worktree

- **Phase 1 items 1–5:** implemented in the current diff. `surface_fourier.py`
  uses explicit `jacfwd`; device-2π helpers are consolidated in touched
  `jax_core` paths; `jax_core.__all__` is derived from `_EXPORT_MODULES`;
  `finitebuild.py` uses the shared host-boundary helper; `CLAUDE.md` and
  `docs/cpp_to_jax_port_file_map.md` are refreshed.
- **Phase 2 items 7, 8, 11:** implemented. `boozer_fixed_state.py` now branches
  on static `spec.no_K`; `biotsavart_cpu_ordered.py` uses `jnp.stack` for traced
  scalar vector assembly; `mhd_bootstrap.py` builds the knot matrix with one
  batched indexed update.
- **Phase 2 item 9:** confirmed non-issue for the current traced path. The
  state-form interpolated-field evaluators are used by the tracing drivers inside
  vmapped scan bodies; comments now document the fixed one-row batch shape.
- **Phase 4 item 17, guarded subitems:** `qfm_solver.py`,
  `mhd_bootstrap.py`, and `magnetic_axis_helpers.py` have real AD consumers or
  forward-AD contracts and now mask dead-branch zero divisions / square roots
  before JVP. Regression tests cover flat `|B|` trapped-fraction JVP, degenerate
  QFM normal JVP finiteness, and zero-discriminant magnetic-axis JVP finiteness.
- **Phase 4 item 17, documented faithful singularities:** `vmec_geometry.py` is
  differentiated by frozen-coefficient / fieldline VMEC tests, and
  `redl_current.py` is differentiated through the public bootstrap wrapper; the
  current AD consumers stay away from the named axis / zero-`L_grad_B` /
  `Zeff=1` / `iota=helicity_N` poles. Source docs now record those
  faithful-port domain limits. The Boozer `mu / v_par` tracing sites now carry
  source docs marking the inherited banana-tip singularity; no guard is planned
  unless AD through tip-crossing becomes a supported objective path.
- **Phase 5 items 19–22 and Phase 7 item 30/specs:**
  `objectives/__init__.py` now uses the same optional-JAX `find_spec` gate as
  sibling packages; Stage-2 target objective privacy is documented in the
  module header and `CLAUDE.md`; XYZ/XYZTensor cross-refs were added;
  `_finite_difference.py` separates host/device step names;
  `_jax_common.points_device` was trimmed in the current diff; `specs.py` now
  documents array-bearing specs as tracing payloads, not dict keys.
- **Phase 7 item 25:** implemented as a bounded tracing driver-body dedup.
  `tracing.py` now shares the DOPRI5 trial / PI-controller accepted-state update
  in `_dopri5_adaptive_step`, shares fixed-shape angle-plane event scanning in
  `_scan_angle_plane_events`, and shares stopping-criterion event row / status
  handling in `_apply_stopping_criteria_events`. The public fieldline,
  Cartesian guiding-center, Boozer guiding-center, and full-orbit drivers keep
  separate top-level functions because their state dimensions, axis handling,
  result row widths, and public SIMSOPT wrapper contracts differ. A read-only
  subagent independently confirmed the duplicated regions and recommended
  keeping the first production slice bounded; the implemented helper set keeps
  the JAX `lax.scan` shape contract and SIMSOPT `res_tys` / `res_phi_hits` row
  layouts unchanged while removing the duplicated event/stopping bodies that
  blocked one-place item 26 work.
- **Phase 3 item 13:** implemented as a behavior-preserving resolver refactor.
  `optimizer_jax.py` now derives concrete Boozer inner drivers from the typed
  `_BOOZER_INNER_DRIVER_OPTIONS` table, shares target objective-route construction
  through one contract helper, and keeps reference residual least-squares
  coalesced onto the single host `lm` lane.
- **Phase 3 item 14:** implemented as a behavior-preserving module split.
  `surfaceobjectives_traceable_jax.py` owns the traceable runtime/cache/custom-VJP
  builders; `surfaceobjectives_jax.py` retains the IFT wrapper classes and
  surface kernels plus compatibility re-exports for existing public and private
  imports. Host-wrapper baseline-gradient setup uses a narrow
  `transfer_guard_host_to_device("allow")` block, matching the official JAX
  transfer-guard split between explicit host-boundary staging and implicit
  transfers while leaving the pure runtime entrypoints strict.
  Follow-up closure also moved the single-stage traceable on-device method gate
  onto `_ONDEVICE_OPTIMIZER_METHODS`, so `lm-minpack-ondevice` and
  `optimistix-lm-ondevice` share the same SSOT allow-set as Boozer target LS
  dispatch.
- **Phase 3 item 15:** implemented as a behavior-preserving local refactor.
  `_normalize_solver_options` now applies LS cross-option incompatibilities
  through a declarative rule table, and `inner_driver` conflict detection uses
  the same table-driven shape before normalizing legacy options.
- **Phase 3 item 16:** implemented as a bounded local change. `BoozerSurfaceJAX`
  now stores normalized solver options in a plain dict after construction-time
  default resolution; backend mutation no longer silently rewrites the separate
  `materialize_dense_linearization` option.
- **Phase 7 item 26:** investigated and deferred. The DOPRI5 recompute pattern is
  still live through each driver's `state_at_fraction` hook, but the angle-plane
  scan / hit-row contract is now centralized by item 25. A production
  dense-output/interpolant patch should still wait for a profiler trace proving
  event localization dominates runtime; lowering `max_root_iters` would change
  the event-accuracy contract.
- **Phase 7 item 27:** investigated and deferred. Dipole kernels intentionally
  scan over dipoles, wireframe totals intentionally scan over segments, and
  current tests assert that streaming structure. A chunked/`vmap` production
  switch remains measurement-gated because it trades GPU parallelism against
  staging `npoints * contributors` intermediates.
- **Phase 7 item 28:** investigated. The `_from_spec` / `_from_dofs` pairs are
  an intentional two-entry contract: spec entry points are the immutable
  JAX-core kernels; DOF entry points materialize a derived spec for Optimizable
  adapters and derivative wrappers. Keep the public twin API and deduplicate
  only local helper bodies when a concrete change-amplification site appears.
- **Phase 7 item 29:** implemented. Exact-Jacobian adjoint runtime callbacks now
  reuse the linear operator built once by `BoozerSurfaceJAX`, and
  `_solve_boozer_adjoint_batch` submits RHS rows as one column-batched private
  solve. Focused tests assert single operator-build reuse and one batched RHS
  solve rather than one solve per row.

### Deferred or still open

- **Phase 2 item 6:** the `curve_geometry.py` DOF-map slice/update helper now
  uses static `lax.slice_in_dim` segment assembly for concrete evaluation and an
  explicit device-placed selector fallback for traced/autodiff evaluation. The
  fallback is a strict-transfer compromise: JAX's `lax.slice` transpose currently
  lowers an implicit scalar zero under `transfer_guard("disallow")`, while the
  selector fallback passes both VJP and JVP strict-transfer checks. The
  `curve_kernels.py` reference in the plan is stale for `_update_1d_static`; its
  remaining selector matrices are Fourier-mode selectors and should be handled
  only with a targeted benchmark if still considered hot.
- **Phase 2 item 10:** deferred. Replacing the one-time permanent-magnet SVD
  scale with `eigh(A.T @ A)` or power iteration needs numerical/perf evidence;
  no current benchmark proves the swap is worth the behavior risk.
- **Phase 3 item 12:** implemented in this pass. Residual risk is limited to
  broader downstream script coverage beyond the focused result-shape and wrapper
  selectors captured below.
- **Phase 4 remaining work:** if future tests drive VMEC/Redl/tracing through
  the documented physical singularities, add a model-specific limiting treatment
  with a failing `jacfwd`/`jax.grad` regression first.
- **Phase 7 item 25 residual:** no further generic-driver collapse is planned
  without a concrete change-amplification site. The remaining repeated pieces
  are driver-specific setup/final assembly for distinct public result layouts.
  Item 26 should be revisited with profiler evidence or a dense-output design
  that plugs into the shared angle-plane helper without changing event accuracy.

### Focused validation captured in this pass

- `ruff check` on the changed Python slice: passed.
- `ruff format --check` on the changed Python slice: passed.
- `git diff --check`: passed.
- `simsopt.jax_core` export smoke: `len(__all__) == len(_EXPORT_MODULES) == 314`.
- Bootstrap / Boozer fixed-state / CPU-ordered Biot-Savart focused tests:
  `10 passed, 52 deselected`.
- QFM AD-safety focused tests: `1 passed`.
- Curve geometry focused tests: `5 passed, 239 deselected`.
- Fluxobjective current-diff focused tests: `6 passed, 6 skipped`.
- Tracing helper focused tests: `3 passed`.
- Objectives package-entry import smoke for `SquaredFluxJAX`: `1 passed`.
- Lightweight JAX surface/Boozer-derivative/fluxobjective/import packet:
  `tests/geo/test_surface_fourier_jax.py`,
  `tests/geo/test_surface_rzfourier_jax.py`,
  `tests/geo/test_boozer_derivatives_jax.py`,
  `tests/objectives/test_fluxobjective_jax_parity.py`, and
  `tests/test_jax_import_smoke.py`
  → `383 passed, 106 skipped in 1018.86s`.
- Phase 4 closeout selectors:
  `tests/mhd/test_bootstrap_jax.py -k 'trapped_fraction_jax_modb_jvp_matches_centered_fd or compute_trapped_fraction_jax_flat_modb_jvp_is_finite or redl_bootstrap_jax_profile_dof_jacobians_match_finite_difference or j_dot_B_Redl_jax_grad_matches_finite_difference_profile_coefficients'`
  → `4 passed, 14 deselected`;
  `tests/mhd/test_vmec_compute_geometry_jax.py -k 'frozen_coeff_gradient_matches_finite_difference'`
  → `1 passed, 8 deselected`;
  `tests/geo/test_surface_objectives_jax.py -k 'qfm_surface_norm_zero_jvp_is_finite'`
  → `1 passed, 328 deselected`;
  `tests/geo/test_qfmsurface_jax.py -k 'qfm_augmented_lagrangian_info_reports_qfm_gradient'`
  → `1 passed, 22 deselected`;
  `tests/field/test_magnetic_axis_helpers_jax_item21.py -k 'first_eigenvalue_angle_2x2'`
  → `5 passed, 15 deselected`.
- Phase 4 closeout `ruff check`, `ruff format --check`, and diff whitespace
  checks passed on the touched Python/docs slice.
- Exact-Jacobian adjoint reuse focused tests:
  `tests/geo/test_surface_objectives_jax.py -k 'exact_batched_adjoint or traceable_solve_exact_linearization or traceable_predict_warmstart'`
  → `2 passed, 326 deselected`;
  `tests/geo/test_boozersurface_jax.py -k 'exact_jacobian_uses_host_tolerance_boundary or exact_adjoint_dense_metadata_does_not_change_operator_runtime or exposes_runtime_callbacks_and_stream'`
  → `3 passed, 474 deselected`;
  `tests/geo/test_boozersurface_jax_private.py -k 'matrix_rhs_linear_operators_apply_columns'`
  → `1 passed, 103 deselected`.
- Broader exact-path selector:
  `tests/geo/test_boozersurface_jax.py tests/geo/test_surface_objectives_jax.py -k 'exact_jacobian or exact_adjoint or exact_batched_adjoint or runtime_callbacks_and_stream'`
  → `7 passed, 798 deselected`.
- Phase 7 item 26 current-contract checks:
  `tests/jax_core/test_tracing_jax_item14.py -k 'bracket_root_finds_zero_crossing or trace_fieldline_jaxpr_uses_scan'`
  → `2 passed, 41 deselected`.
- Phase 7 item 25 tracing driver-body dedup checks:
  `tests/jax_core/test_tracing_jax_item14.py -k 'trace_fieldline_jaxpr_uses_scan_and_supports_reverse_mode_ad or trace_guiding_center_and_fullorbit_jaxprs_use_scan or boozer_axis_status_ignores_rejected_trial_steps or bracket_root_finds_zero_crossing'`
  → `4 passed, 39 deselected`;
  `tests/jax_core/test_tracing_jax_phi_events.py tests/jax_core/test_tracing_jax_fullorbit_events.py tests/jax_core/test_tracing_jax_levelset_events.py`
  → `16 passed`;
  `tests/jax_core/test_tracing_jax_guiding_center.py tests/jax_core/test_tracing_jax_fullorbit.py`
  → `18 passed`;
  `tests/jax_core/test_tracing_jax_boozer_zeta_events.py -k 'zeta_plane or trace_particles_boozer_jax_records_zeta_hits or flux_coordinate_stopping'`
  → `2 passed, 1 deselected`;
  `tests/jax_core/test_tracing_jax_gc_boozer.py -k 'axis_violation or rhs_recovers or endpoint_matches_cpp_oracle or routes_when_field_is_jax_wrapper'`
  → `6 passed, 2 deselected`;
  `tests/jax_core/test_tracing_jax_conservation.py`
  → `3 passed`.
- Phase 7 item 27 current-contract checks:
  `tests/jax_core/test_dipole_field_item24.py -k 'total_field_kernels_stream_over_dipoles or dipole_field_Bn_symmetry_axis_is_vectorized'`
  → `2 passed, 21 deselected`;
  `tests/jax_core/test_wireframe_jax_item29.py -k 'total_field_kernels_stream_over_segments or contribution_kernels_scan_over_half_periods'`
  → `2 passed, 12 deselected`.
- Phase 3 item 16 options normalization checks:
  `tests/geo/test_boozersurface_jax.py -k 'materialize_dense_linearization_from_backend or backend_mutation_does_not_rewrite_dense_linearization_default or backend_mutation_preserves_explicit_dense_linearization_request or resolve_ls_optimizer_method_contract or normalize_solver_options'`
  → `16 passed, 461 deselected`.
- Phase 3 item 15 option-incompatibility checks:
  `tests/geo/test_boozersurface_jax.py -k 'inner_driver or materialize_dense_linearization_from_backend or backend_mutation_does_not_rewrite_dense_linearization_default or backend_mutation_preserves_explicit_dense_linearization_request or optimizer_backend_from_runtime_contract or private_ls_option_validation or optimistix_lm_rejects or unknown_option_rejected or parity_mode_rejects_damped_boozer_linearization or private_options_rejected_with_scipy_backend or scipy_limited_memory_options_are_accepted or removed_hybrid_backend_is_rejected or ls_constructor_rejects_outer_only_optimizer_backend'`
  → `32 passed, 445 deselected`.
- Phase 3 item 15 adjacent contract checks:
  `tests/geo/test_optimizer_jax_item19.py::test_item19_boozer_inner_driver_contract_stays_typed`,
  `tests/geo/test_lm_optimistix_contract.py::{test_target_optimistix_lm_lane_rejects_callbacks,test_target_optimistix_lm_lane_rejects_nondefault_lm_tuning}`,
  and
  `tests/geo/test_boozersurface_jax_private.py::TestBoozerSurfaceJAXClassPrivate::test_run_code_ondevice_force_limited_memory_routes_to_lbfgs`
  → `14 passed`;
  `tests/test_runtime_dtype_policy.py -k 'boozer_optimizer_backend_auto_uses_policy_default or boozer_ls_mps_smoke_default_avoids_target_x64_gate or boozer_ls_mps_smoke_default_reaches_reference_method'`
  → `3 passed, 35 deselected`;
  `tests/geo/test_single_stage_example.py -k 'initialize_boozer_surface_limited_memory_disables_dense_linearization or resolve_single_stage_boozer_inner_driver_from_compat_options'`
  → `2 passed, 425 deselected`.
- Full BoozerSurfaceJAX regression file after the item 15 / item 16 / item 29 updates:
  `tests/geo/test_boozersurface_jax.py`
  → `473 passed, 4 skipped`.
- Phase 3 item 13 resolver matrix checks:
  `tests/geo/test_boozersurface_jax.py -k 'resolve_ls_optimizer_method_contract or resolve_ls_optimizer_method_rejects_invalid_backend or resolve_least_squares_optimizer_method_contract or resolve_least_squares_optimizer_method_rejects_invalid_backend or resolve_least_squares_optimizer_method_rejects_limited_memory_lm or resolve_least_squares_optimizer_method_rejects_scipy_control_lm or resolve_optimizer_method_rejects_non_ondevice_ls_lane_in_target_jax_backend_modes'`
  → `43 passed, 434 deselected`;
  `tests/geo/test_optimizer_jax_item19.py` → `12 passed`;
  `tests/geo/test_single_stage_example.py -k 'resolve_single_stage_boozer_inner_driver_from_compat_options or run_single_stage_optimizer_ondevice_does_not_enter_scipy_minimize or run_single_stage_optimizer_rejects_unknown_outer_lane or resolve_single_stage_outer_optimizer_method_rejects_unknown_backend or resolve_single_stage_outer_optimizer_method_rejects_cpu_ondevice'`
  → `5 passed, 370 deselected`;
  `tests/solve/jax/test_compat_shim_translation.py tests/solve/jax/test_driver_dispatch.py`
  → `19 passed`.
- Phase 3 item 14 traceable split checks:
  direct fresh import of `simsopt.geo.surfaceobjectives_traceable_jax` succeeded;
  legacy `surfaceobjectives_jax` public/private traceable helper imports were
  identity-equal to the new module; focused runtime/cache/custom-VJP selector
  across `tests/jax_core/test_tree_signature.py` and
  `tests/geo/test_surface_objectives_jax.py` → `12 passed`;
  traceable CPU-reference runtime-bundle selector excluding the long fused
  value/grad comparison → `9 passed`.
- Phase 3 item 14 single-stage gate follow-up:
  `tests/geo/test_surface_objectives_jax.py -k 'traceable_cache_state_accepts_ondevice_least_squares_methods or traceable_cache_state_rejects_non_ondevice_methods'`
  → `4 passed`; resolver contract selectors → `13 passed`; traceable
  `lm-minpack` integration selector → `1 passed`; static `ruff`, `py_compile`,
  and `git diff --check` clean.
- Item 30 tail checks:
  `tests/geo/test_surface_objectives_jax.py -k 'traceable_exact_warmstart_prediction_uses_operator_solve or traceable_exact_warmstart_success_matches_reference_operator_linearization or traceable_exact_warmstart_failure_keeps_failed_operator_step or traceable_exact_warmstart_failure_surfaces_unsuccessful_forward_result or traceable_seeded_initial_value_surfaces_failed_solve_gradient'`
  → `5 passed, 324 deselected`;
  `tests/geo/test_surface_objectives_jax.py -k 'traceable_solve_exact_linearization_uses_operator_with_factors_present or traceable_exact_operator_and_dense_reference_share_residual_contract or traceable_inner_stationarity_coil_jvp_matches_full_stationarity_jvp or traceable_objective_gradient_parts_use_strict_vjp_helpers or traceable_term_adjoint_solve_report_serializes_unknown_iterations_as_null'`
  → `5 passed, 324 deselected`;
  `tests/geo/test_boozersurface_jax.py -k 'run_code_traceable_ls_skips_lu_for_nonfinite_newton_result or run_code_traceable_ls_reuses_newton_fun_and_grad or run_code_traceable_ls_skip_policy_does_not_call_newton'`
  → `3 passed, 474 deselected`.
- Tail completion audit after a broad integration attempt exposed six failures:
  `tests/integration/test_single_stage_jax.py::test_traceable_iota_target_penalty_uses_runtime_scalar_constants`,
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestNonQSRatioValue::test_dj_allows_strict_transfer_guard`,
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestCompositeObjective::test_public_wrapper_dj_boundaries_allow_strict_transfer_guard_real_fixture`,
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestExactSolveCPUJAXParity::test_operator_adjoint_signoff_gate_on_exact_state`,
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestTraceableObjective::test_target_lane_accepted_step_sync_matches_legacy_mutable_surface_lane`,
  and
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestTraceableObjective::test_traceable_runtime_bundle_matches_sharded_field_contract`
  → `6 passed in 57.76s` after the strict-transfer, module-split, exact-status,
  nested-metric, and sharding-test fixes.
- Slow private-LBFGS traceable selectors were validated individually on CPU:
  `test_traceable_scalar_routes_through_private_lbfgs_path`
  → `1 passed, 1 warning in 357.32s`;
  `test_traceable_value_and_grad_routes_through_ondevice_private_path`
  → `1 passed, 1 warning in 311.68s`;
  `test_traceable_solver_path_localizes_delta_to_optimizer_driver`
  → `1 passed, 1 warning in 377.33s`; and
  `test_traceable_matches_fused_value_and_grad_path`
  → `1 passed, 2 warnings in 846.08s`.
- The subsequent full CPU integration sweep exposed a post-59% artifact-contract
  failure in
  `tests/integration/test_single_stage_physics_parity.py::TestSingleStagePhysicsSmokeParity::test_outer_loop_physics_quantity_single_step_budget_smoke_parity`:
  fixed-iteration one-step runs can legitimately write `REJECTED.json` with a
  diagnostic payload when SciPy/JAX L-BFGS-B stop at the iteration limit. The
  shared single-stage final-payload loader now accepts either `results.json` or
  the diagnostic payload from `REJECTED.json`; the init-parity ladder and
  physics-parity test use the same helper. The same test also had a stale
  cross-lane equality assertion for one-step CPU SciPy L-BFGS-B versus private
  on-device L-BFGS-B. Per SciPy's official L-BFGS-B docs, `maxiter` and
  `maxfun` are iteration/evaluation termination limits and status 1 means the
  run stopped at those limits, not convergence; the smoke now requires
  lane-local finite objective descent, weighted-component recomposition, and
  shared physics/hardware ceilings instead of identical first-step endpoints.
  Validation:
  `tests/integration/test_single_stage_physics_parity.py::TestSingleStagePhysicsSmokeParity::test_outer_loop_physics_quantity_single_step_budget_smoke_parity`
  → `1 passed in 660.91s`;
  `test_init_state_sensitivity_smoke_parity_under_small_initial_coil_perturbation`
  plus the CUDA outer-loop probe selector
  → `1 passed, 1 skipped in 33.26s`;
  `test_single_stage_subprocess_env_preserves_existing_xla_flags` plus
  `tests/test_benchmark_helpers.py::test_single_stage_init_loads_rejected_diagnostic_payload`
  → `2 passed in 1.72s`.
- The single-stage physics parity JAX subprocess cache now uses the stable
  source-hashed `JAX_COMPILATION_CACHE_DIR` directly instead of creating a fresh
  `run-*` subdirectory for every invocation; this matches the existing source
  hash invalidation comment and avoids forcing repeated cold CPU XLA compiles.
- The public `--backend jax` optimizer default is now regular `scipy-jax` for
  both Stage 2 and single-stage CLIs. Explicit `ondevice` and
  `scipy-jax-fullgraph` remain available, but the implicit JAX lane no longer
  switches by local JAX platform. Current public docs were refreshed in
  `docs/using_jax_backend.md`, `docs/source/jax_migration.rst`, and
  `examples/single_stage_optimization/BETA_QUICKSTART.md`. Focused validation:
  `tests/test_cli_defaults.py`,
  `tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::test_parse_args_defaults_jax_backend_to_scipy_jax_optimizer_lane`,
  `tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::test_parse_args_accepts_disabling_accepted_step_callback`,
  `tests/integration/test_stage2_jax.py::TestStage2OptimizerContract::test_parse_args_accepts_least_squares_algorithm_override`,
  and
  `tests/geo/test_single_stage_example.py::SingleStageExampleTests::test_parse_args_defaults_jax_backend_to_scipy_jax_optimizer_lane`
  plus the single-stage artifact/cache helper selectors
  `tests/integration/test_single_stage_physics_parity.py::test_single_stage_subprocess_env_preserves_existing_xla_flags`
  and
  `tests/test_benchmark_helpers.py::test_single_stage_init_loads_rejected_diagnostic_payload`
  → `14 passed in 4.31s`. Static checks on the touched files passed:
  `ruff check`, `ruff format --check`, `py_compile`, `git diff --check`, and a
  stale-string scan for the removed platform-aware default language.
- The benchmark validation harness defaults were also re-based onto the regular
  `scipy-jax` outer lane without adding `scipy-jax` to the native/on-device
  branch group. `benchmarks/single_stage_smoke_fixture.py`,
  `benchmarks/single_stage_init_parity.py`,
  `benchmarks/single_stage_outer_loop_probe.py`,
  `benchmarks/stage2_e2e_comparison.py`, and
  `benchmarks/validation_ladder_contract.py` now keep plain JAX benchmark
  invocations on host-SciPy control while preserving explicit `ondevice` stress
  lanes. Focused validation:
  `tests/test_benchmark_helpers.py -k 'single_stage_init_defaults_to_reduced_grid_smoke_fixture or single_stage_fixture_optimizer_backend_defaults_by_backend or stage2_benchmark_scripts_default_to_repo_fixture_equilibria_dir or single_stage_outer_loop_contract_matches_probe_defaults or single_stage_outer_loop_probe_accepts_finite_target_lane_result or single_stage_outer_loop_probe_resolves_boozer_backend or single_stage_init_parity_accepts_fullgraph_target_optimizer_method or single_stage_init_parity_requires_accepted_step_on_outer_loop_probe or single_stage_init_parity_reports_real_gate_failures'`
  → `9 passed, 314 deselected`; and
  `tests/geo/test_single_stage_example.py -k 'defaults_jax_backend_to_scipy_jax_optimizer_lane or defaults_boozer_algorithm_from_explicit_inner_backend or explicit_target_lane_outer_maxls_to_tighter_budget or explicit_target_lane_benchmark_mode_preserves_boozer_precision'`
  → `4 passed, 379 deselected`.
- A broad CPU integration sweep with `tests/integration/ -q -k 'not lbfgs'`
  progressed to 59% before a failure in
  `test_traceable_solver_path_localizes_delta_to_optimizer_driver`. Root cause:
  two earlier traceable-runtime tests changed `booz_jax.options` on the
  module-scoped fixture, but the function-scoped restore did not restore solver
  options or invalidate the option-keyed traceable runtime cache. The fixture
  restore now snapshots/restores solver options and drops the cached traceable
  runtime entry when options changed. Targeted order repro:
  `test_runtime_bundle_rebuilds_after_solver_option_change_post_compile`,
  `test_traceable_objective_accepts_lm_ondevice_inner_solve`, and
  `test_traceable_solver_path_localizes_delta_to_optimizer_driver`
  → `3 passed, 1 warning in 405.13s`. The broad sweep was not rerun to
  completion after the targeted fix because the private-LBFGS selectors take
  5-14 minutes each on this CPU-only host.
- Item 14 traceable least-squares allow-list follow-up: the gate now imports
  `_ONDEVICE_OPTIMIZER_METHODS` from `boozersurface_jax.py`, whose SSOT is
  `{"bfgs-ondevice", "lbfgs-ondevice"} | optimizer_jax._TARGET_LEAST_SQUARES_METHODS`.
  Focused validation:
  `tests/geo/test_surface_objectives_jax.py::test_traceable_cache_state_accepts_ondevice_least_squares_methods`
  plus
  `tests/geo/test_surface_objectives_jax.py::test_traceable_cache_state_rejects_non_ondevice_methods`
  → `4 passed in 4.19s`;
  `tests/geo/test_boozersurface_jax.py::TestBoozerSurfaceJAXClass::test_resolve_least_squares_optimizer_method_contract`
  → `13 passed in 4.72s`; and
  `tests/integration/test_single_stage_jax_cpu_reference.py::TestTraceableObjective::test_traceable_objective_accepts_lm_ondevice_inner_solve`
  → `1 passed in 80.94s`. Static checks on
  `src/simsopt/geo/surfaceobjectives_traceable_jax.py` and
  `tests/geo/test_surface_objectives_jax.py` passed: `ruff check`,
  `ruff format --check`, `py_compile`, `git diff --check`, and a stale literal
  scan for the old error text / 3-method allow-list.
- Final static checks passed for the current tail: `ruff check`, `ruff format --check`,
  `py_compile`, and `git diff --check` on the 55 changed Python files
  (including `surfaceobjectives_traceable_jax.py` and the benchmark
  default-rebase files) plus this plan emitted no findings.

### Validation caveats

- Local CUDA validation was not run in this pass. The local runtime probe found
  `jax 0.10.0`, backend `cpu`, devices `['cpu']`, and no `nvidia-smi` binary
  on `PATH`; CUDA conclusions above are documentation-backed design gates, not
  a GPU signoff.
- Current-head CUDA signoff remains external as of 2026-05-30. A local re-probe
  before the signoff-packet update at `1162ae851` again found local JAX backend
  `cpu`, devices `[('cpu', 'cpu:0')]`, and no local `nvidia-smi` binary. The
  prior Runpod A100 SSH endpoint recorded in `HANDOFF.md`
  (`154.54.102.24:16628`) refused connection during a non-invasive status
  probe, and `runpodctl pod list -o json` returned `[]`, so the previous venue
  could not provide current-head GPU evidence. The release gate is therefore
  still either a fresh CUDA-host run or an explicit release owner waiver.
- Exact current-head CUDA signoff packet for a CUDA-capable host is now the
  executable SSOT at `scripts/current_head_cuda_signoff.sh`. Run it from the
  current checkout:
  ```bash
  PYTHON_BIN="${PYTHON_BIN:-$PWD/.conda/jax/bin/python}" \
  RESULTS_DIR="${RESULTS_DIR:-$PWD/.artifacts/current_head_cuda_signoff}" \
  bash scripts/current_head_cuda_signoff.sh
  ```
  The script fails closed unless the tracked checkout is clean, no non-artifact
  untracked path or inherited `PYTHONPATH` can influence the run, `nvidia-smi`
  exists, `PYTHON_BIN` is executable, JAX reports a CUDA/GPU backend, a live
  JAX `transfer_guard` disallow probe blocks an implicit transfer, both
  JSON-producing single-stage runs pass, both JSON payloads record the active
  checkout's `git rev-parse HEAD`, and both payloads record
  `transfer_guard=disallow`.
  Successful execution is the required artifact evidence for checking the
  CUDA/GPU signoff box above; a skipped pytest selector is not sufficient
  evidence for this release gate because the signoff must fail closed when CUDA
  is unavailable.
- The full unfiltered `tests/integration/` sweep now has current-tree pytest
  pass counts: `486 passed, 9 skipped, 8 warnings in 3769.76s (1:02:49)`.
  The wrapping zsh command exited `1` after pytest completed because it tried
  to assign the reserved parameter name `status`; this is recorded as a harness
  typo, not a product or pytest failure. Earlier partial-run caveats remain
  historical only: the stale option-leak failure and the explicit on-device
  parser/default blockers are now covered by focused passing tests and the full
  integration pass above.
- A broad selector over `tests/mhd/test_bootstrap_jax.py` also pulled in
  `test_mhd_import_propagates_broken_installed_jax_importerror`, which failed
  because the package treated a broken installed JAX import as absent in the
  subprocess. The same focused test was reproduced as failing on a clean
  detached `HEAD` worktree at `2497f0281`, so it is a pre-existing MHD import
  contract issue outside the touched bootstrap kernel.
- The previous tracing comm-replay caveat is closed on the current tree:
  `tests/jax_core/test_tracing_jax_gc_boozer.py::test_trace_particles_boozer_jax_rejects_unsupported_shapes_and_replays_comm`
  now passes (`1 passed in 5.78s`). Root cause was the test harness comparing
  rank-local one-particle replay payloads against slices from a two-particle
  batched no-comm trace; the helper now accepts rank-local expected payloads
  while preserving the existing global no-comm comparison for other callers.
- Bounded read-only reviewer subagents checked Phase 3 item 13, Phase 3 item 14,
  the Phase 3 item 14 gate follow-up, Phase 7 item 29, and the Phase 3 item 16 /
  item 26–27 documentation updates. The item 13/item 14 reviews returned PASS
  with no actionable findings; the only actionable later finding was the stale
  full-orbit item 26 anchor, corrected above to `tracing.py:3968`.

## Open Questions

- How tightly is `boozersurface_jax.res` bound to CPU `BoozerSurface.res`
  dict-compat? Resolved for item 12: the stateful JAX result record remains a
  `dict` subclass so existing CPU-style `isinstance(res, dict)`, key, copy, and
  mutation consumers keep working.
- Phase 4 AD-consumer audit resolved: VMEC, QFM, Redl, MHD bootstrap, and
  magnetic-axis kernels all have current AD consumers or forward-AD contracts.
  QFM/MHD/magnetic-axis dead-branch AD hazards are guarded with focused
  regressions; VMEC/Redl/Boozer-tip poles remain documented faithful physical
  singularities until a supported objective differentiates at those poles.
- Phase 1 / Phase 2 bundling resolved for the current diff: the zero-risk
  Phase 2 items 7, 8, and 11 are included with focused validation evidence
  captured above.
- `field/sampling_jax.py` confirmed as a public wrapper that re-exports the
  `jax_core/sampling.py` weighted curve/surface sampling kernels; the port map
  "misc" row is current.
