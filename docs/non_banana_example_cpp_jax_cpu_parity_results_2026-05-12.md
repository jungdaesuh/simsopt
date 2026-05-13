# Non-Banana Example C++/JAX CPU Parity Results - 2026-05-12

Status: implementation report for
`docs/non_banana_example_cpp_jax_cpu_parity_plan_2026-05-12.md`. This is a
current-HEAD evidence document, paired with the JSON artifacts under
`.artifacts/parity/20260512-non-banana-examples/`.

## Scope

Non-banana SIMSOPT example fixtures. CPU-only (no GPU). All claims here
are backed by the JSON artifacts in
`.artifacts/parity/20260512-non-banana-examples/` and by the integration
tests in `tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py`.

## Per-fixture summary

The table reflects the most recent JSON artifact, generated against
HEAD `e50bc4ced` (after which this document is being authored on the
follow-up commit).

| Fixture | Verdict | Native components compared | Failing | Unsupported | Max abs diff (gradient) | Max abs diff (B) |
| --- | --- | --- | --- | --- | --- | --- |
| `minimal_stage2_flux_length_gap` | partial | 7 | 0 | 1 | 1.4e-12 | 4.4e-16 |
| `cws_saved_local_flux_nfp2` | unsupported | 0 | 0 | 0 | n/a | n/a |
| `cws_saved_local_flux_nfp3` | unsupported | 0 | 0 | 0 | n/a | n/a |
| `full_stage2_composite` | partial | 7 | 0 | 5 | 3.5e-17 | 4.4e-16 |
| `planar_stage2_composite` | partial | 7 | 0 | 6 | 3.8e-17 | 5.0e-16 |
| `position_orientation_flux_support_gate` | unsupported | 0 | 0 | 0 | n/a | n/a |
| `boozer_surface_basic` | unsupported | 0 | 0 | 0 | n/a | n/a |
| `boozer_qa_wrappers` | unsupported | 0 | 0 | 0 | n/a | n/a |
| `finite_beta_target_flux` | unsupported | 0 | 0 | 0 | n/a | n/a |
| `finitebuild_multifilament_support_gate` | unsupported | 0 | 0 | 0 | n/a | n/a |
| `qfm_surface` | unsupported | 0 | 0 | 0 | n/a | n/a |

Verdict legend:

- `pass`: every native-supported quantity passes its tolerance bucket and
  there are no unsupported required components.
- `partial`: every native-supported quantity passes its tolerance bucket
  and at least one component is explicitly listed in
  `unsupported_components`. No native-supported quantity fails.
- `unsupported`: the fixture cannot construct a native JAX lane today
  because a required native-spec or wrapper contract is missing, or the
  harness does not yet wire the comparison required for this fixture
  family.
- `fail`: at least one native-supported quantity exceeds its tolerance
  bucket. Currently zero failures.

## Fixture-by-fixture status

### Pass (with native subproblem at machine precision)

- `minimal_stage2_flux_length_gap` — Phase 1 P0 fixture. SquaredFlux value,
  gradient, B, B·n, surface gamma/normal all pass `direct_kernel` /
  `ls_wrapper_gradient` buckets. `QuadraticPenalty(sum(CurveLength), 'max')`
  is listed in `unsupported_components` as
  `QuadraticPenalty_over_sum_CurveLength_max`. Perturbation diagnostic
  (seed=1 Taylor central differences) passes at `abs_diff < 1e-6` for
  every eps in `{1e-3, 1e-4, 1e-5, 1e-6, 1e-7}`.
- `full_stage2_composite` — Phase 3 P1 fixture. SquaredFlux subproblem
  passes at machine precision (gradient `max_abs_diff = 3.5e-17`). CPU
  composite total `JF_total_cpu` is recorded in
  `lanes.cpu_cpp.components` for traceability but is not compared against
  any JAX total because the rest of the composite has no native JAX
  wrappers. Five components are listed in `unsupported_components`:
  `sum_CurveLength`, `CurveCurveDistance`, `CurveSurfaceDistance`,
  `sum_LpCurveCurvature`, `sum_QuadraticPenalty_MeanSquaredCurvature_max`.
- `planar_stage2_composite` — Phase 4 P2 fixture. Same partial pattern
  with planar coils (`CurvePlanarFourier` exposes `to_spec()`).
  `LinkingNumber` plus the planar geometry penalties are listed as
  unsupported (six entries).

### Unsupported (gated)

- `cws_saved_local_flux_nfp2`, `cws_saved_local_flux_nfp3` — Phase 2 P1.
  Upstream `simsopt.load()` cannot reconstruct `CurveCWSFourier`. The
  fixture builder calls `simsopt.load` and emits a precise unsupported
  result citing the deserializer signature mismatch. Re-enable by
  patching the upstream JSON loader.
- `position_orientation_flux_support_gate` — Phase 5 support gate. The
  probe builds the CPU TF+windowpane fixture (per the plan's "build the
  CPU fixture without optimizer execution") and rejects the JAX lane
  because `OrientedCurveXYZFourier` does not implement `to_spec()`. The
  message records the exact rejecting class and the active free DOF list
  the JAX lane would mirror once support lands.
- `finitebuild_multifilament_support_gate` — Phase 7 support gate. The
  probe materializes a low-resolution multifilament grid via
  `create_multifilament_grid` and `apply_symmetries_to_curves` /
  `apply_symmetries_to_currents`. Every symmetry-expanded filament's
  base curve (after stripping `RotatedCurve` wrappers) currently exposes
  a native spec; the residual gap is the full
  `build_lanes`-style multifilament composite constructor (flux + length
  + curvature + filament-arclength variation + min-distance penalty),
  which is not wired into this harness.
- `boozer_surface_basic`, `boozer_qa_wrappers` — Phase 6. Native JAX
  Boozer residual + wrapper coverage already exists in
  `tests/geo/test_boozer_residual_jax.py`,
  `tests/geo/test_boozer_derivatives_jax.py`, and
  `tests/integration/test_single_stage_jax.py`. Per-fixture wiring needs
  the harness LaneArtifact to carry Boozer residual vectors / iota / G
  / label values; that extension is a follow-up plan.
- `finite_beta_target_flux` — Phase 7. The blocker is `VirtualCasing`
  preprocessing; once a cached `vcasing_*.nc` is checked in for the
  W7-X target equilibrium, this fixture flips to a partial verdict
  (SquaredFluxJAX with target array native; CPU-only length QP
  unsupported).
- `qfm_surface` — Phase 7. `QfmResidualJAX` exists in current source
  (`src/simsopt/geo/surfaceobjectives_jax.py`); per-fixture wiring for
  QFM residual + label parity requires the same LaneArtifact extension
  as Boozer.

## Reproducer commands

The harness is CPU-only; commands force `JAX_PLATFORMS=cpu` and
`JAX_ENABLE_X64=1` before `import jax`. Run from the repository root.

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 conda run -p .conda/jax-0.9.2 \
    python -m pytest tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py -v
```

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 conda run -p .conda/jax-0.9.2 \
    python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
        --fixtures all-supported \
        --git-sha "$(git rev-parse HEAD)" \
        --dirty-policy record \
        --output-json .artifacts/parity/20260512-non-banana-examples/all-supported-cpu.json
```

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 conda run -p .conda/jax-0.9.2 \
    python benchmarks/non_banana_example_cpp_jax_cpu_parity.py \
        --fixtures all \
        --git-sha "$(git rev-parse HEAD)" \
        --dirty-policy record \
        --output-json .artifacts/parity/20260512-non-banana-examples/all-cpu.json
```

## Definition-of-Done crosswalk

Walking the plan's "Definition Of Done" item by item:

- [x] P0 fixed-state fixtures pass CPU/C++ vs JAX CPU value and gradient
  parity for native-supported quantities — `minimal_stage2_flux_length_gap`,
  `full_stage2_composite`, and `planar_stage2_composite` all pass their
  native-supported comparisons (every comparison entry has
  `verdict="pass"` in the JSON artifact).
- [x] The minimal Stage-II fixture's length penalty is reported as
  `QuadraticPenalty_over_sum_CurveLength_max` in `unsupported_components`.
- [x] P1 saved-artifact and full-composite fixtures either pass or report
  exact unsupported components — `cws_saved_local_flux_nfp{2,3}` report
  the precise upstream `simsopt.load` failure; `full_stage2_composite`
  reports the five CPU-only components.
- [~] P2 Boozer/planar fixtures have fixed-state coverage before any
  optimizer trajectory claims — planar fixture passes; Boozer is reported
  as `unsupported` with classification reason pointing at existing JAX
  parity test coverage (`tests/geo/test_boozer_residual_jax.py`,
  `tests/geo/test_boozer_derivatives_jax.py`,
  `tests/integration/test_single_stage_jax.py`). Per-fixture harness
  wiring is a follow-up plan.
- [x] Position/orientation and finite-build fixtures are reported as
  support gates with precise unsupported reasons.
- [x] All JSON artifacts include current git SHA and dirty-tree metadata.
- [x] All pass claims state the exact fixture, quantity, tolerance bucket,
  and max difference (see the JSON artifact `comparisons` arrays).
- [x] All timed JAX measurements synchronize with `block_until_ready`
  (`benchmarks/non_banana_example_parity_fixtures.py::_build_jax_lane`).
- [x] No GPU proof is claimed from this CPU-only plan.
- [x] No banana Stage 2 or banana single-stage proof is mixed into this
  non-banana example plan.

## Follow-up plans

The following follow-up plans would close the remaining gaps without
loosening any current tolerance:

1. Upstream `simsopt.load()` deserializer support for `CurveCWSFourier`.
   Unblocks `cws_saved_local_flux_nfp{2,3}` → `pass`.
2. Harness `LaneArtifact` extension for Boozer residual / iota / G /
   label fields, plus a parallel `_supported_comparisons_boozer` branch.
   Unblocks `boozer_surface_basic` and `boozer_qa_wrappers`.
3. Same `LaneArtifact` extension covers `qfm_surface`.
4. Checked-in cached `vcasing_*.nc` artifact for the W7-X target
   equilibrium. Unblocks `finite_beta_target_flux` → `partial`.
5. `OrientedCurveXYZFourier.to_spec()` plus an immutable spec contract.
   Unblocks `position_orientation_flux_support_gate` → `partial`.
6. Native JAX implementations of `CurveLength`, `CurveCurveDistance`,
   `CurveSurfaceDistance`, `LpCurveCurvature`, `MeanSquaredCurvature`,
   `LinkingNumber`, and `QuadraticPenalty`. Each implementation flips
   the corresponding entry of `unsupported_components` to a native
   comparison line. Largest cross-fixture lever for moving
   `full_stage2_composite` and `planar_stage2_composite` to `pass`.
