# Closeout Coverage Template

This template is the SSOT for parity-row evidence in JAX port closeout
artifacts (`.artifacts/jax_port_goal/plans/*.md`, `REPORT.md`, etc.).
It was introduced after the 2026-05-13 test-quality audit
(`.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md`, AI-2)
revealed that several closeout citations rested on tautological tests
(JAX-vs-JAX comparisons with no independent oracle).

## Requirement

Every parity-row claim in a closeout artifact MUST cite:

1. **Test file**: full path, ideally with line-number anchor (e.g.,
   `tests/geo/test_framedcurve_jax_item18.py:327`). The cited test must
   exist in the current tree.
2. **Oracle**: the independent source of truth the test compares against.
   One of:
   - **C++ reference symbol** — cite the simsoptpp class/function (e.g.,
     `sopp.biot_savart_B`, `BoozerSurface._call_boozer_residual_ds`,
     `surfacerzfourier.cpp::dgamma_by_dcoeff_vjp`).
   - **Closed-form analytic expression** — write the expression out so
     the reader can audit it without re-running the test (e.g., planar
     circle `γ(t) = (R cos 2πt, R sin 2πt, 0)`, analytic torus area
     `4π² R r`).
   - **External dataset or pinned baseline** — cite the source (VMEC
     wout file, FOCUS input, pinned bundle path).
   - **Finite-difference vs JAX** — acceptable as a gradient oracle, but
     the value path itself still needs one of the above.
3. **Parity-ladder tolerance lane**: which lane in
   `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`
   the test uses. Cite both `rtol` and `atol` (or the relevant tier of
   the lane dict).

**Reject claims like "JAX matches host"** when the "host" wrapper
invokes the same JAX kernel under the hood. That is not an oracle; it
is a function-vs-itself tautology. The 2026-05-13 audit findings #1–#4
catalogue concrete examples.

## Example (good)

| Coverage row | Test | Oracle | Lane |
| --- | --- | --- | --- |
| FrameRotation analytic | `tests/geo/test_framedcurve_jax_item18.py::test_rotated_centroid_frame_matches_planar_circle_analytic` | Closed-form planar-circle centroid frame: `T=(-sin 2πt, cos 2πt, 0)`, `N₀=(cos 2πt, sin 2πt, 0)`, `B₀=(0,0,-1)`; with rotation α, `N = cos(α) N₀ − sin(α) B₀` | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| Build filaments analytic | `tests/geo/test_finitebuild_jax_ssot_item20.py::test_build_filament_gammas_matches_planar_circle_closed_form` | Closed-form filament positions on planar circle with hand-computed `(dn, db)` offsets | `direct_kernel`, `rtol=1e-10`, `atol=1e-12` |
| CurveFilament VJP FD | `tests/geo/test_finitebuild_jax_item20.py::test_curvefilament_jax_gamma_vjp_matches_central_fd` | Central finite-difference of JAX gamma forward pass (independent gradient oracle) | `fd_gradient`, `directional_fd_rtol=1e-5`, `directional_fd_atol=1e-7` |

## Example (bad — REJECT)

| Coverage row | Test | Oracle | Lane |
| --- | --- | --- | --- |
| FrameRotation parity | `tests/geo/test_framedcurve_jax_wrappers_item18.py::test_frame_rotation_jax_matches_host` | Host wrapper output | — |

The "host" `FramedCurveFrenet`/`FramedCurveCentroid` re-export the same
JAX kernel the JAX wrapper uses, so this asserts `f(x) == f(x)`. It is
not parity coverage.

## How to use this template

1. Before marking an item closeout `complete`, walk the coverage matrix
   row by row. For each row:
   - Open the cited test and read the actual assertion.
   - Identify the oracle. If the oracle is another JAX function in the
     same call graph, the row is tautological — fix the test, then
     update the citation.
   - Write the oracle into the citation (one short phrase is fine if
     the test docstring spells out the formula).
   - Tag the parity-ladder lane (use `parity_ladder_tolerances(...)`
     names; do not invent new lane names in closeouts).
2. When deleting or renaming a tautological test, propagate the change
   to every closeout citation (`.artifacts/jax_port_goal/REPORT.md`,
   `plans/*-coverage.md`, `restart/*.md`, `bench/*.json`, `state.json`).
3. New closeout artifacts must include an "Oracle" column or annotation
   on every `current` row.
