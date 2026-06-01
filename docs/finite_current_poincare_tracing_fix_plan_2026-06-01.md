# Finite-Current Poincare Tracing Fix Plan

## Purpose

This plan defines the fix for the finite-current materialized-field Poincare
crash observed after validation tracing completed for
`fc_p400_cocur/poincare_default`. The goal is to make validation, diagnostic,
and default Poincare metrics measurable for finite-current fields without
conflating their different seed and stopping contracts.

## Goals

- Make `examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py`
  produce validation, diagnostic, default, and aggregate metric sidecars for
  finite-current materialized fields without a `simsoptpp` segmentation fault.
- Make the metric contract explicit:
  validation uses inset midplane seeds plus Boozer-surface exit stopping;
  diagnostic uses inset midplane seeds plus the same Boozer-surface exit guard;
  default uses extended-surface seeds plus the same Boozer-surface exit guard.
  Box guardrails remain present in all modes, but they are no longer the first
  line of defense against runaway finite-current traces.
- Keep topology field preparation and stopping criteria centralized in
  `examples/single_stage_optimization/topology_scorer.py`.
- Record the actual field model, interpolation bounds, extrapolation policy,
  and stopping-domain metadata in emitted Poincare artifacts.
- Validate the fix on a fresh copy of the `fc_p400_cocur` artifact and rerun
  vacuum controls under the same metric definitions.

## Non-Goals

- Do not change finite-current materializer physics, proxy/VF coil geometry, or
  Boozer-current conventions.
- Do not treat validation survival as a substitute for diagnostic or default
  survival.
- Do not trust materialized-field iota or promote confinement claims without the
  existing finite-current stability and VMEC cross-check gates.
- Do not patch SIMSOPT C++ tracer internals in this task.
- Do not rewrite historical run directories or stale `results.json` notes in
  place.

## Current Context

- The failing artifact is:
  `/Users/suhjungdae/code/columbia/autoresearch/runs/box_lever_matrix_2026-05-28/fc_p400_cocur/poincare_default`.
- That directory contains `PoincareMetrics_opt_validation.json` but does not
  contain `PoincareMetrics_opt_diagnostic.json`,
  `PoincareMetrics_opt_default.json`, or aggregate
  `PoincareMetrics_opt.json`.
- Its `poincare.log` shows validation completed first:
  `survival=50/50`, `status=validated`, and maximum interpolation error
  `2.2635132625661874e-4`.
- Immediately after validation, the process crashed with
  `Signal: Segmentation fault: 11`, `Signal code: Invalid permissions (2)`,
  and a stack inside `simsoptpp.cpython-311-darwin.so`.
- The local vacuum comparison artifacts have all three modes:
  `r4_rim_diag` has validation `29/50`, diagnostic `45/50`, default `28/50`;
  `control` has validation `39/50`, diagnostic `49/50`, default `28/50`.
- `poincare_surfaces.py` traces validation first, then diagnostic, then default
  in one process. The crash therefore happened while starting the old box-only
  diagnostic mode, before default mode was measured.
- `topology_scorer.prepare_topology_field()` builds an `InterpolatedField` from
  surface-sized cylindrical bounds. It currently records grid resolution and
  interpolation error, but not the actual `rrange`, `phirange`, `zrange`, or
  extrapolation flag.
- `topology_scorer.build_stopping_criteria()` is the shared stopping-criteria
  factory. It builds `MaxZStoppingCriterion`, `MinZStoppingCriterion`,
  `MinRStoppingCriterion`, `MaxRStoppingCriterion`, and optionally
  `LevelsetStoppingCriterion`.
- The finite-current field includes real proxy/VF current sources. The
  `fc_p400_cocur` coil artifact has 51 coils, with VF rings at approximately
  `R = 1.572133 m`, `Z = +/-0.6505194 m`, and current
  `-61.53846153846154 A`; the optimized surface spans roughly
  `R = 0.82156..1.01929 m` and `|Z| <= 0.07576 m`.
- For this artifact, the current interpolation domain from `padded_bounds()` is
  approximately `R = 0.80156..1.03929 m`, `|Z| <= 0.08575 m`. The current
  box-stopping contract is approximately `R = 0.78048..1.07026 m`,
  `|Z| <= 0.07954 m`. The R stopping box therefore extends outside the
  interpolation domain before any finite-current-specific policy is applied.
- The VF coil locations are not themselves required to be inside the
  interpolation grid; field-line evaluation points are what matter. They are
  still relevant because they change the external field and can drive extended
  default/diagnostic traces toward larger excursions than the vacuum cases.
- Official SIMSOPT documentation and local docstrings define
  `InterpolatedField(field, degree, rrange, phirange, zrange, extrapolate=True,
  nfp=..., stellsym=...)` as cylindrical-grid interpolation. With
  `extrapolate=True`, the interpolant can be evaluated outside its configured
  domain; with `extrapolate=False`, the local wrapper throws a Python exception
  for out-of-domain index probes.
- Official SIMSOPT examples use `InterpolatedField(..., extrapolate=True, ...)`
  together with `SurfaceClassifier` and `LevelsetStoppingCriterion` to stop
  field lines when they leave the target surface. Official stopping criteria
  also include `MaxRStoppingCriterion`, `MinRStoppingCriterion`,
  `MaxZStoppingCriterion`, and `MinZStoppingCriterion`.
- Local validation showed that simply switching to `extrapolate=False` is not a
  viable long-trace fix: the adaptive RK solver probes trial stages outside
  even a guard-expanded interpolation grid before an accepted-step stopping
  criterion can fire. The clean failure is `RuntimeError: zidxs=... not within
  [...]`, rather than a sidecar-producing trace.
- Local validation also isolated a Python/C++ lifetime hazard: constructing the
  `InterpolatedField` inside `prepare_topology_field()` and returning only the
  field could leave the constructor range tuples dead before long tracing.
  Keeping those range objects alive on the returned field removes the
  helper-only segfault/non-finite divergence observed against the manual smoke.
- The supported diagnosis is therefore: finite-current diagnostic/default
  traces are entering a numerically unsafe region for the old box-only stopping
  policy. The documented SIMSOPT pattern is to use a surface levelset stop to
  bound escape before box-only extrapolated tracing can run away.

## Rationale

The fix should be a trace-policy and observability fix, not a physics-model
change. The materialized finite-current field is doing the intended thing by
adding real proxy/VF coils; the tracer must be made robust enough to measure
the three existing Poincare modes honestly.

The shared topology helpers are the right single source of truth. Field
preparation already lives in `prepare_topology_field()`, and stopping criteria
already live in `build_stopping_criteria()`. Adding ad hoc finite-current
logic inside the plotting script would make future topology scoring and
promotion artifacts diverge.

Validation mode should remain surface-exit stopped. Diagnostic and default
should use the same surface-exit guard when run through the interpolated
Poincare script, while retaining their distinct seed populations. This changes
the old box-only diagnostic/default contract, so old vacuum numbers are not
comparable until rerun under the same guarded contract.

For interpolated tracing, the emitted artifact must record the interpolation
domain and whether it covers the seed and nominal stopping domain. Local
testing showed that forcing the interpolation grid to cover the entire nominal
domain can destabilize this finite-current trace; the stable documented pattern
is a surface-sized interpolant with SIMSOPT's `extrapolate=True` behavior plus
surface levelset stopping.

Because the failure is a native-code crash, Python exception handling is not a
complete recovery mechanism. The implementation must avoid the unsafe state
upfront through documented stopping domains and field-policy choices, then
prove the path by producing all expected sidecars in a fresh run directory.

## Assumptions

- The current sidecar schema is the compatibility target for downstream
  promotion and comparison scripts.
- Validation, diagnostic, and default mode names are part of the science
  contract and should not silently change semantics.
- `auto` field policy should be changed from the current tmax-only interpolation
  threshold into a documented guarded policy: record the interpolation ranges
  and nominal-domain coverage, use `extrapolate=True` as in SIMSOPT's official
  tracing example, and rely on the surface levelset stop to prevent runaway
  finite-current traces.
- Exact-field tracing is acceptable as an opt-in or automatic fallback if
  interpolation remains unstable, but it may be slower and should be reported.
- Existing materialized finite-current artifacts remain valid inputs; the fix
  should not require regenerating proxy/VF coil fields.

## Implementation Plan

1. Make trace-domain metadata explicit in `topology_scorer.py`.
   - [ ] Extend `prepare_topology_field()` field-model metadata to include
     `rrange`, `phirange`, `zrange`, `extrapolate`, `surface_rmin`,
     `surface_rmax`, `surface_zmax`, and padding values when interpolation is
     selected.
   - [ ] Keep the existing `grid` keys for backward compatibility.
   - [ ] Add a small typed helper, for example `TopologyTraceDomain`, that
     carries surface bounds, seed bounds, historical box-stopping bounds, and
     resolved interpolation bounds.
   - [ ] Keep the interpolation range objects alive for returned
     `InterpolatedField` instances so native tracing never observes dead
     constructor state after the helper returns.
   - [ ] Assert in metadata and tests whether interpolated mode covers the
     selected seed and nominal stopping bounds across `R`, `phi`, and `Z`. If
     an explicit caller-supplied interpolation range does not cover them,
     `always` must fail loudly and `auto` must resolve to native field instead
     of interpolated field.
   - [ ] Do not infer safety from coil locations alone; record field-line
     evaluation bounds and stopping bounds instead.

2. Make the stopping semantics docs-aligned and explicit.
   - [ ] Keep the historical box-stopping formula in the guardrail list:
     `rmin * (1 - box_padding)`, `rmax * (1 + box_padding)`, and
     `zmax * (1 + box_padding)`.
   - [ ] Feed those resolved stopping bounds into `prepare_topology_field()` so
     artifacts can report whether the surface-sized interpolant covers the
     nominal guardrails, without shrinking or redefining the guards to fit the
     interpolation domain.
   - [ ] Use `include_surface_exit=True` for validation, diagnostic, and
     default Poincare render modes in `poincare_surfaces.py`.
   - [ ] Record the concrete stopping bounds and stop-label order in every
     mode sidecar.
   - [ ] Add tests that diagnostic/default sidecars include `surface_exit` in
     their stop-label contract, so the metric shift is visible and cannot be
     mistaken for the old box-only contract.

3. Add a documented field-policy control to `poincare_surfaces.py`.
   - [ ] Replace the hard-coded `interpolate = True` switch with a validated
     environment variable, `POINCARE_FIELD_POLICY=auto|always|never`. Use an
     environment variable for this first fix because the script already uses
     `POINCARE_OUT_DIR` and has no `argparse` boundary today.
   - [ ] Default the plotting script to `auto`, with `auto` using
     `extrapolate=True`, surface-sized interpolation ranges, and surface-exit
     stopping for all long render modes.
   - [ ] Change `auto` so it records whether the resolved interpolation domain
     covers the requested nominal render-mode domains; explicit caller-supplied
     uncovered domains still fail/fallback according to `always`/`auto`.
   - [ ] Make `always` fail loudly when a caller supplies an explicit
     interpolation range that does not cover the requested `R`/`phi`/`Z` metric
     domain. For the default surface-sized interpolant, record the coverage flag
     and use SIMSOPT's documented `extrapolate=True` behavior plus surface-exit
     stopping.
   - [ ] Allow `never` to run exact-field tracing for crash triage and final
     confirmation when interpolation is unsafe.
   - [ ] Write the selected policy and resolved field mode into validation,
     diagnostic, default, and aggregate artifacts.
   - [ ] Fail loudly on unsupported policy values.

4. Refactor the plotting script into testable mode configuration.
   - [ ] Extract a pure helper that returns the three render-mode contracts:
     seed radii, seed contract, stopping criteria, stop labels, metric suffix,
     display label, and concrete trace-domain bounds.
   - [ ] Keep the execution order validation -> diagnostic -> default.
   - [ ] Ensure each mode writes its sidecar immediately after tracing, as it
     does today, so a later failure never erases completed evidence.
   - [ ] Avoid broad `try/except` around native tracer calls; rely on safer
     policies and explicit validation instead.

5. Add tests for the crash contract.
   - [ ] Test that `prepare_topology_field()` records interpolation ranges and
     extrapolation status when interpolation is selected.
   - [ ] Test that `prepare_topology_field(..., field_policy="never")` returns
     the native field and records `selected_mode = "native"`.
   - [ ] Test that `field_policy="auto"` records when the nominal stopping
     bounds exceed the surface-sized interpolation domain.
   - [ ] Test that `field_policy="always"` fails before tracing if an explicit
     caller-supplied interpolation range cannot cover the requested
     `R`/`phi`/`Z` domain.
   - [ ] Test that the Poincare render-mode helper preserves validation,
     diagnostic, and default seed semantics and uses the surface-exit guard for
     all three render modes.
   - [ ] Test that unsupported field-policy values fail before any field-line
     tracing starts.
   - [ ] Add a regression fixture or lightweight synthetic case with external
     coils that exercises the finite-current field-policy path without running
     long `tmax=7000` traces in unit tests.

6. Run fresh acceptance traces.
   - [ ] Copy the `fc_p400_cocur` `biot_savart_opt.json` and `surf_opt.json`
     into a new scratch output directory; do not mutate the original run.
   - [ ] Run `poincare_surfaces.py` with the default `auto` policy and confirm it
     chooses an extrapolating interpolation domain with `surface_exit` in the
     stop-label contract for every render mode.
   - [ ] Run a shorter or explicitly time-bounded `POINCARE_FIELD_POLICY=never`
     smoke only to prove the native path is still available; full `tmax=7000`
     native tracing may be too slow for the normal acceptance lane.
   - [ ] If explicit `auto` selects interpolation, verify the emitted `rrange`,
     `zrange`, and coverage flag match the emitted diagnostic/default nominal
     stopping bounds before accepting the result.
   - [ ] Rerun `r4_rim_diag` and `control` through the same fixed script and
     policy so finite-current and vacuum comparisons are metric-matched.
   - [ ] Report validation, diagnostic, and default survival separately.

7. Clean up stale promotion interpretation.
   - [ ] Do not edit the historical `fc_p400_cocur/results.json` in place.
   - [ ] In any new report or generated summary, mark the old promotion note as
     stale unless diagnostic/default sidecars exist in the same run directory.
   - [ ] Ensure future promotion checks require the sidecar files they cite,
     rather than trusting a free-form `_NOTE`.

## Validation Plan

- [ ] Static sanity:
  `cd /Users/suhjungdae/code/columbia/simsopt-surrogate && git diff --check`.
- [ ] Syntax:
  `PYTHONNOUSERSITE=1 PYTHONPATH=examples/single_stage_optimization .conda-env/bin/python3.11 -m compileall examples/single_stage_optimization/topology_scorer.py examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py`.
- [ ] Focused topology tests:
  `PYTHONNOUSERSITE=1 PYTHONPATH=examples/single_stage_optimization .conda-env/bin/python3.11 -m pytest -q tests/geo/test_topology_scorer_wba_contract.py tests/geo/test_topology_bridge.py`.
- [ ] Focused field/preparation tests:
  `PYTHONNOUSERSITE=1 PYTHONPATH=examples/single_stage_optimization .conda-env/bin/python3.11 -m pytest -q tests/geo/test_single_stage_example.py -k "prepare_topology_field or stopping_criteria or poincare"`.
- [ ] Fresh finite-current acceptance trace:

  ```bash
  cd /Users/suhjungdae/code/columbia/simsopt-surrogate
  set -o pipefail
  rm -rf /tmp/fc_p400_cocur_poincare_retry
  mkdir -p /tmp/fc_p400_cocur_poincare_retry
  cp /Users/suhjungdae/code/columbia/autoresearch/runs/box_lever_matrix_2026-05-28/fc_p400_cocur/poincare_default/biot_savart_opt.json /tmp/fc_p400_cocur_poincare_retry/
  cp /Users/suhjungdae/code/columbia/autoresearch/runs/box_lever_matrix_2026-05-28/fc_p400_cocur/poincare_default/surf_opt.json /tmp/fc_p400_cocur_poincare_retry/
  POINCARE_OUT_DIR=/tmp/fc_p400_cocur_poincare_retry \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH=examples/single_stage_optimization \
    .conda-env/bin/python3.11 \
    examples/single_stage_optimization/POINCARE_PLOTTING/poincare_surfaces.py \
    2>&1 | tee /tmp/fc_p400_cocur_poincare_retry/poincare.log
  ```

- [ ] Acceptance artifact check:
  confirm the scratch directory contains
  `PoincareMetrics_opt_validation.json`,
  `PoincareMetrics_opt_diagnostic.json`,
  `PoincareMetrics_opt_default.json`, and `PoincareMetrics_opt.json`, and that
  its log contains no `Segmentation fault`.
- [ ] Native-path smoke:
  rerun a time-bounded native smoke with `POINCARE_FIELD_POLICY=never` only to
  verify the option is wired; do not require full `tmax=7000` native tracing for
  acceptance unless runtime budget permits.
- [ ] Metric-matched comparison:
  rerun the same script/policy on fresh scratch copies of `r4_rim_diag` and
  `control`, then compare validation, diagnostic, and default survival in
  separate rows.

## Risks And Mitigations

- Risk: Exact-field tracing is too slow at `tmax=7000`.
  Mitigation: Keep interpolation as the default when stable; use exact field as
  a named policy or acceptance fallback, and record the selected field mode.

- Risk: Adding surface-exit stopping to diagnostic/default changes the old
  box-only metric.
  Mitigation: Record `surface_exit` in every sidecar's `stop_labels`, rerun
  vacuum controls under the same guarded contract, and never compare guarded
  finite-current metrics to stale box-only vacuum numbers.

- Risk: A native segfault cannot be caught by Python cleanup code.
  Mitigation: Avoid the unsafe state with stronger stopping domains and
  field-policy controls; run acceptance in scratch directories so partial
  evidence cannot corrupt historical artifacts.

- Risk: Future reports may cite stale `_NOTE` fields instead of actual sidecars.
  Mitigation: Promotion/reporting code must require the cited sidecar files and
  metric keys before declaring diagnostic/default success.

## Completion Criteria

- [ ] The implementation changes only topology tracing/plotting helpers and
  focused tests, not finite-current materializer physics.
- [ ] `poincare_surfaces.py` can run the `fc_p400_cocur` scratch artifact without
  segfaulting.
- [ ] The scratch finite-current run writes validation, diagnostic, default, and
  aggregate metric JSON files.
- [ ] The field-model metadata records actual interpolation ranges,
  extrapolation policy, selected field mode, stopping bounds, and whether the
  interpolation domain coverage flag for the traced metric domain.
- [ ] Vacuum controls are rerun under the same guarded metric definitions before
  any finite-current improvement claim is made.
- [ ] Any report separates validation, diagnostic, and default survival instead
  of replacing missing default metrics with validation metrics.

## Open Questions

- If a future box-only diagnostic is required, should it run exact-field only or
  in a subprocess-isolated tracer rather than through the interpolated-field
  plotting path?
- Should render modes eventually run in subprocesses so a future native-code
  crash in one mode cannot prevent failure reporting for the others? This is a
  hardening follow-up, not the first root-cause fix.
