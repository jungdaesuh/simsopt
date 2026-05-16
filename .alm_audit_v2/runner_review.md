# ALM Runner Audit v2 — 2026-05-08

## Summary

Audited the user-facing ALM runner scripts (`run_single_stage_thresholded_physics_alm.py`, `run_stage2_alm.py`, `run_stage2_to_single_stage.py`, `run_stage2_iota_decision_gate.py`, `run_alm_normalization_fixture_benchmark.py`, `collect_alm_autoresearch_baseline.py`) and their shared helpers (`workflow_runner_common.py`, `workflow_helpers.py`, `alm_utils.py`, `banana_opt/alm_benchmarking.py`, `banana_opt/alm_fixture_benchmarking.py`).

Headline findings:

- **CRITICAL F1**: Runner default `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0` violates the `require_positive_alm_threshold` contract added in commit `bf936a0a4`/H1. The runner cannot be invoked with default arguments — every default invocation fails inside `validate_single_stage_alm_formulation_args` with `ValueError`. This same value is forwarded into `run_stage2_to_single_stage.py` recovery commands, breaking that handoff path under the same conditions.
- **HIGH F2**: Operator-facing `--alm-iota-penalty-threshold` help text is mathematically incorrect. The doc claims `Jiota_penalty = (iota - iota_target)^2 * iotas_weight`, but `QuadraticPenalty.J()` returns `0.5 * (iota - iota_target)^2`, so the worked example (`iotas_weight=1.0`, `deviation 0.01 -> threshold 1e-4`) is off by 2× from the actual code path. Operators following the help text under-budget every iota constraint by 2×.
- **HIGH F3**: `run_single_stage_thresholded_physics_alm.py` strips help text from the four physics threshold flags it owns. The same flags carry detailed unit explanations on the inner `single_stage_banana_example.py`, but the runner the operator actually invokes shows no unit guidance — an explicit regression vs the "operator-facing units" goal of commit `2e9acced2`.
- **HIGH F4**: `run_stage2_alm.py` does not expose `--basin-seed`/`--basin-hops` as CLI flags. `_normalize_basin_seed` silently calls `os.urandom(4)` when a JSON spec sets `basin_hops > 0` with `basin_seed=None`. The seed is then encoded into the artifact path via `format_stage2_basin_suffix`, so each run materialises a NEW path, defeating artifact reuse and reproducibility.
- **MEDIUM F5**: Runner-emitted summary JSONs use `json.dump(summary, f, indent=2)` without `allow_nan=False`. `alm_final_max_feasibility_violation`, `alm_final_stationarity_norm`, etc. can be NaN/Inf on degenerate runs — these emit non-portable `NaN`/`Infinity` tokens that downstream consumers (autoresearch baseline collector) silently mis-parse. `workflow_runner_common.write_json` already enforces `allow_nan=False`; the runners bypass it.
- **MEDIUM F6**: `run_stage2_alm.py` never calls `validate_alm_cli_args`. ALM tuning fields supplied via `--stage2-spec-json` are validated only inside the subprocess (`banana_coil_solver.py:1411`). A bad value triggers a stack trace from the subprocess instead of an immediate parser-level rejection.
- **LOW F7**: ALM CLI fields `--alm-qs-threshold` / `--alm-boozer-threshold` / `--alm-iota-penalty-threshold` carry no `help=` text in `run_single_stage_thresholded_physics_alm.py:190-205`. Compare `run_single_stage_goal_mode_comparison.py:252-263`, which uses the correct `default=None + append_optional_flag` pattern and lets the inner script own units.
- **LOW F8**: `alm_fixture_benchmarking.run_fixture_benchmark` accepts a `seed` kwarg, records it under `settings.seed`, but never uses it to seed any RNG. Fixtures are deterministic regardless. Operators may believe `--seed N` changes behavior; it does not.

No remaining `ACCEPT_OFFSPEC*` references in the audited runners. Constraint registration order is deterministic (set-typed `available_names` is filtered through the canonical `HARDWARE_CONSTRAINT_SCHEMA` order in `hardware_constraint_specs`). `validate_initial_multipliers` is wired into `_normalize_alm_run_inputs` and is not directly exposed at any runner CLI.

## Methodology

1. Walked the runner CLI list and collected all `--alm-*` flags + defaults.
2. For each threshold flag, traced the CLI value through to the receiver (`SINGLE_STAGE/single_stage_banana_example.py` or `STAGE_2/banana_coil_solver.py`).
3. Confirmed validators (`validate_alm_cli_args`, `require_positive_alm_threshold`) are reached on the production path.
4. Cross-referenced the recently committed unit-clarification work (commits `2e9acced2`, `bf936a0a4`) against the runner-side help text and defaults.
5. Diffed `d61648f50` (off-spec removal) against the audited files to confirm no surviving `ACCEPT_OFFSPEC*` paths.
6. Read the prior audit (`.alm_audit/FIX_PLAN.md`, `.alm_audit/algorithm_review.md`) for known runner items, especially S1 / H1 / M10.
7. Verified atomic-write / determinism claims by reading the JSON write paths and the basin-seed flow.

## Findings

### F1: Default `DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0` is rejected by the post-`bf936a0a4` validator [CRITICAL]

- File: `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py:43`
- Code:
  ```python
  DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD = 0.0
  ```
- Receiver: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:2994-2995`
  ```python
  for flag_name, value in required_thresholds.items():
      require_positive_alm_threshold(flag_name, value)
  ```
- Helper: `examples/single_stage_optimization/alm_utils.py:381-385`
  ```python
  value_f = float(value)
  if not np.isfinite(value_f) or value_f <= 0.0:
      raise ValueError(
          f"ALM threshold {name!r} must be a finite positive value; got {value!r}"
      )
  ```
- Bug: The runner unconditionally injects `--alm-length-penalty-threshold 0.0` into the subprocess command (`run_single_stage_thresholded_physics_alm.py:313-314` always emits `str(args.alm_length_penalty_threshold)`). The single-stage entrypoint hardcodes `--alm-formulation thresholded_physics` (line 277-278), which forces `validate_single_stage_alm_formulation_args` to call `require_positive_alm_threshold` for every threshold flag including `length_penalty`. With the default 0.0, `value_f <= 0.0` ⇒ `ValueError` — every default invocation crashes.
- Why: `.alm_audit/FIX_PLAN.md` H1 (now landed via `bf936a0a4`) intentionally tightened `< 0.0` to `<= 0.0` to plug the "10¹² scale-floor blow-up" path. The runner-side default was never updated. The fix plan even called this out (line 173: "Saved configs with literal `0.0` thresholds become load-error after fix"), but missed this hardcoded constant.
- Impact: `python run_single_stage_thresholded_physics_alm.py --plasma-surf-filename X --stage2-bs-path Y` fails with no operator-actionable message about which knob to set; the only working invocation requires the operator to know `--alm-length-penalty-threshold` exists and pick a finite positive value. The same constant is referenced from `run_stage2_to_single_stage.py:538-542`, propagating the same failure into the recovery handoff path.
- Suggested fix: Either (a) pick a defensible non-zero default consistent with the in-tree fix-plan recipe (e.g. `length_target * length_target * COIL_LENGTH_TARGET_M^2 * weight`) and document the unit, OR (b) follow the `run_single_stage_goal_mode_comparison.py:252-263` pattern: `default=None`, then forward via `append_optional_flag` and let the inner parser raise when None is incompatible with the formulation. Option (b) keeps the strict failure mode but gives the operator a clear "missing required flag" error.

### F2: `--alm-iota-penalty-threshold` help text is off by 2× from the actual code path [HIGH]

- File: `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:1397-1416`
- Code (help text):
  ```python
  "Units: squared-penalty units, NOT iota deviation. The constraint is "
  "Jiota_penalty = (iota - iota_target)**2 * iotas_weight <= threshold, "
  "so this threshold scales with iotas_weight. To target a desired iota "
  "deviation d, set --alm-iota-penalty-threshold = (d**2) * iotas_weight "
  "(e.g. iotas_weight=1.0 with target deviation 0.01 -> threshold 1e-4). "
  ```
- Code (actual implementation, `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:3495-3496`):
  ```python
  if goal_mode == "target":
      return QuadraticPenalty(surface_iota_term, iota_target)
  ```
- Code (`src/simsopt/objectives/utilities.py:185-194`):
  ```python
  def J(self):
      val = self.obj.J()
      diff = float(val - self.cons)
      ...
      elif self.f == 'identity':
          return 0.5*diff**2
  ```
- Bug: The QuadraticPenalty objective returns `J = 0.5 * (iota - iota_target)^2`, NOT `(iota - iota_target)^2`. The help text is missing the `0.5` prefactor and the worked example is therefore off by 2×. With `iotas_weight = 1.0` and target deviation `d = 0.01`, the actual `Jiota_penalty = 0.5 * 1e-4 = 5e-5`, not `1e-4`.
- Why: Commit `2e9acced2` ("docs(alm-runner): clarify iota threshold operator-facing units") clarified the `--stage2-iota-tolerance` semantic but did not re-derive the `--alm-iota-penalty-threshold` help text. The objective wraps a `QuadraticPenalty` whose `J()` formula has had the `0.5` factor since pre-port simsopt, so the help text was never internally consistent.
- Impact: Operators following the help text recipe under-budget every iota inequality by 2×. Some downstream operators may have hand-tuned thresholds to compensate; flipping to the correct formula will move the optimization landscape silently.
- Suggested fix: Either (a) fix the help text to read `Jiota_penalty = 0.5 * (iota - iota_target)**2 * iotas_weight` and update the example (`(d**2) * 0.5 * iotas_weight`, so `1.0 * (0.01)^2 * 0.5 = 5e-5`), OR (b) wrap the objective in a `2 *` so the help-text formula is the actual one and the Stage 2 deviation converts more cleanly (a one-line change in `build_single_stage_iota_objective`).

### F3: Runner strips help text from the threshold flags it owns [HIGH]

- File: `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py:190-205`
- Code:
  ```python
  parser.add_argument("--alm-qs-threshold", type=float, default=DEFAULT_ALM_QS_THRESHOLD)
  parser.add_argument(
      "--alm-boozer-threshold",
      type=float,
      default=DEFAULT_ALM_BOOZER_THRESHOLD,
  )
  parser.add_argument(
      "--alm-iota-penalty-threshold",
      type=float,
      default=DEFAULT_ALM_IOTA_PENALTY_THRESHOLD,
  )
  parser.add_argument(
      "--alm-length-penalty-threshold",
      type=float,
      default=DEFAULT_ALM_LENGTH_PENALTY_THRESHOLD,
  )
  ```
- Bug: None of these four flags carries a `help=` argument. The inner `single_stage_banana_example.py` parser has detailed help text (especially for `--alm-iota-penalty-threshold`, including unit semantics and a worked example), but `argparse` help text does not propagate from the subprocess to the wrapper. A user running `python run_single_stage_thresholded_physics_alm.py --help` sees no guidance on units.
- Why: When `run_single_stage_thresholded_physics_alm.py` was created, the `--alm-iota-penalty-threshold` flag had no operator-facing unit doc anywhere. Commit `2e9acced2` added unit doc to `run_stage2_alm.py:308-313` and to `single_stage_banana_example.py:1407-1415`, but skipped this runner. Since this runner is the canonical entrypoint for thresholded_physics single-stage, this is the shipped operator UX.
- Impact: Operators without insider knowledge cannot tell `--alm-iota-penalty-threshold` is in squared-penalty units (vs deviation units used by Stage 2's `--stage2-iota-tolerance`). The cross-reference docstring in `run_stage2_alm.py` points at the README, but the operator sees nothing here.
- Suggested fix: Mirror the help text from `single_stage_banana_example.py:1397-1416` onto each of the four threshold flags. After F2, this can be a copy with the 0.5 fix applied.

### F4: Non-deterministic `basin_seed` from `os.urandom(4)` breaks artifact reuse [HIGH]

- File: `examples/single_stage_optimization/run_stage2_alm.py:367-372`
- Code:
  ```python
  def _normalize_basin_seed(*, basin_hops: int, basin_seed: int | None) -> int | None:
      if basin_hops <= 0:
          return None
      if basin_seed is not None and int(basin_seed) >= 0:
          return int(basin_seed)
      return int.from_bytes(os.urandom(4), "big")
  ```
- File: `examples/single_stage_optimization/workflow_helpers.py:436-448`
- Code:
  ```python
  def format_stage2_basin_suffix(
      basin_hops: int,
      basin_stepsize: float,
      basin_temperature: float = 1.0,
      basin_niter_success: int = 0,
      basin_seed: int | None = None,
  ) -> str:
      ...
      seed_value = "none" if basin_seed is None else str(basin_seed)
      suffix = (
          f"-BH={basin_hops}"
          f"-BS={format_compact_float(basin_stepsize)}"
          f"-BSeed={seed_value}"
      )
  ```
- Bug:
  1. `--basin-seed` and `--basin-hops` are NOT exposed as CLI flags on `run_stage2_alm.py` (`grep -n "add_argument.*basin"` returns 0 hits in `run_stage2_alm.py`); they are only configurable via `--stage2-spec-json`.
  2. When the spec JSON omits or sets `basin_seed=None` and sets `basin_hops > 0`, `_normalize_basin_seed` invents a fresh random seed via `os.urandom(4)`.
  3. The seed is then injected into the Stage 2 artifact path via `format_stage2_basin_suffix` (line 442, `BSeed=<n>`).
  4. Effect: every run materialises a NEW directory; `ensure_stage2_artifact_result.artifact_reused` will be False; identical configs run different solves.
- Why: Stage 2 wrappers grew incrementally; the spec JSON contract added basin tuning later, and the runner-side seed stitching was implemented for symmetry with `banana_coil_solver.py:1709` (which has the same `os.urandom` fallback). The runner's path-encoding step was not updated to make the absent-seed case deterministic.
- Impact: Researchers who configure basin-hopping cannot get reproducible artifact paths or reuse prior solves. The summary JSON does record the resolved seed (`resolved_stage2_config`), so the run is *retroactively* reconstructable, but only by reading the summary first.
- Suggested fix: Either (a) require `basin_seed` to be set explicitly when `basin_hops > 0` (raise instead of urandom in `_normalize_basin_seed`), or (b) make the seed deterministic from the surrounding spec — e.g. SHA256 of the canonical spec JSON, truncated to 4 bytes. Option (a) is the strict-CLI-contract path consistent with the rest of the audit history.

### F5: Summary JSON writes accept NaN/Inf, producing non-portable JSON [MEDIUM]

- File: `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py:435-437`
- Code:
  ```python
  with summary_path.open("w", encoding="utf-8") as outfile:
      json.dump(summary, outfile, indent=2)
  print(json.dumps(summary, indent=2))
  ```
- File: `examples/single_stage_optimization/run_stage2_alm.py:877-879`
- Code:
  ```python
  with summary_path.open("w", encoding="utf-8") as outfile:
      json.dump(summary, outfile, indent=2)
  print(json.dumps(summary, indent=2))
  ```
- Compare `examples/single_stage_optimization/workflow_runner_common.py:1148-1152`:
  ```python
  def write_json(path: str | Path, payload: object) -> None:
      target_path = Path(path)
      target_path.parent.mkdir(parents=True, exist_ok=True)
      with target_path.open("w", encoding="utf-8") as outfile:
          json.dump(_json_portable_value(payload), outfile, indent=2, allow_nan=False)
  ```
- Bug: The runners bypass `write_json` and call `json.dump` directly with default `allow_nan=True`. Fields like `alm_final_max_feasibility_violation`, `alm_final_stationarity_norm`, `boozer_residual` are populated from solver output and may be `NaN` / `Inf`. Python's default JSON encoder writes those as bare `NaN` / `Infinity` tokens, which violate RFC 8259 and break strict downstream parsers.
- Why: The two runner main() bodies were written before `write_json`/`_json_portable_value` was added to `workflow_runner_common.py`. Switch was never made.
- Impact: `collect_alm_autoresearch_baseline.py` (a downstream consumer) uses Python's stdlib json.loads for ledger rows, which silently accepts these tokens via `parse_constant`. Other tooling (e.g. `jq`, JS readers, the autoresearch CSV pipeline) does not. So the runner output is well-formed-ish from Python but corrupt for cross-language consumers, and the schema invariant in the rest of the codebase is violated. It's also non-atomic — if `json.dump` raises mid-write (e.g. unserializable type), the file is left half-written.
- Suggested fix: Replace both `with summary_path.open("w") ...; json.dump(summary, outfile, indent=2)` blocks with `write_json(summary_path, summary)`. Same for `print(json.dumps(...))` — use `json.dumps(_json_portable_value(summary), indent=2, allow_nan=False)`.

### F6: `run_stage2_alm.py` does not call `validate_alm_cli_args` at the wrapper boundary [MEDIUM]

- File: `examples/single_stage_optimization/run_stage2_alm.py:798-812`
- Code:
  ```python
  def main(argv: list[str] | None = None) -> int:
      args = parse_args(argv)
      validate_stage2_iota_args(
          stage2_iota_mode=args.stage2_iota_mode,
          ...
      )
  ```
- Compare `examples/single_stage_optimization/STAGE_2/banana_coil_solver.py:1410-1411`:
  ```python
  args = parse_args() if parsed_args is None else parsed_args
  validate_alm_cli_args(args)
  ```
- Compare `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py:8109`:
  ```python
  validate_alm_cli_args(args)
  ```
- Bug: ALM tuning fields supplied through `--stage2-spec-json` (e.g. `alm_max_outer_iters`, `alm_penalty_init`, `alm_penalty_scale`, `alm_feas_tol`, etc.) are forwarded into `Stage2ArtifactConfig` and then into the subprocess `--alm-*` flags. The values are validated only inside the subprocess, AFTER `subprocess.run` boots a Python interpreter, imports the heavy stack, and eventually reaches `validate_alm_cli_args`. The wrapper boundary itself accepts e.g. `alm_penalty_max < alm_penalty_init` and produces an artifact path that the subprocess will reject minutes later.
- Why: `run_stage2_alm.py` was designed under the assumption that the spec JSON is the contract, and the contract is enforced at the place that consumes the values (`banana_coil_solver.py`). For wrapper UX, the validation should also run early.
- Impact: Bad spec JSON wastes time materialising directories, locks, and subprocess startup before the failure is reported.
- Suggested fix: Add a `validate_alm_cli_args(SimpleNamespace(**resolved_spec))` after `resolve_stage2_spec_payload` returns. The `Stage2AlmControls` dataclass can also gain a `__post_init__` that mirrors the same checks, providing SSOT (the project memory says this is the project convention; cf. `ALMSettings.__post_init__` at `alm_utils.py:37-77`).

### F7: Threshold flags lack `help=` in `run_single_stage_thresholded_physics_alm.py` [LOW]

- File: `examples/single_stage_optimization/run_single_stage_thresholded_physics_alm.py:190-205`
- Code: see F3.
- Compare `examples/single_stage_optimization/run_single_stage_goal_mode_comparison.py:252-263`:
  ```python
  parser.add_argument("--alm-qs-threshold", type=float, default=float(os.environ["ALM_QS_THRESHOLD"]) if "ALM_QS_THRESHOLD" in os.environ else None)
  parser.add_argument("--alm-boozer-threshold", type=float, default=float(os.environ["ALM_BOOZER_THRESHOLD"]) if "ALM_BOOZER_THRESHOLD" in os.environ else None)
  ```
- Bug: The goal_mode_comparison runner uses the proper `default=None` + `append_optional_flag` pattern (line 679-682), letting the inner parser own units and defaults. The thresholded_physics runner instead hardcodes module-level constants (`DEFAULT_ALM_QS_THRESHOLD = 3.0e-3`, etc.), which (a) duplicates SSOT, (b) silently changes behavior if the inner default ever changes, and (c) couples F1's failure to all four thresholds rather than letting `None` propagate cleanly.
- Why: When the runner was authored, the operator UX goal was "single canonical entrypoint with stable defaults". Since then, the multiple-runner pattern has stabilised on `append_optional_flag`. This runner was not converted.
- Impact: Operator sees no help; SSOT drift is possible if the inner defaults change.
- Suggested fix: Adopt the goal_mode_comparison pattern. After F1, this is essentially the same change — drop the module-level constants, set `default=None`, and use `append_optional_flag`.

### F8: `alm_fixture_benchmarking.run_fixture_benchmark` records `seed` but never uses it [LOW]

- File: `examples/single_stage_optimization/banana_opt/alm_fixture_benchmarking.py:296-361`
- Code:
  ```python
  def run_fixture_benchmark(
      *,
      ...
      seed: int = DEFAULT_FIXTURE_SEED,
  ) -> dict[str, object]:
      active_fixtures = tuple(default_fixtures() if fixtures is None else fixtures)
      ...
      return {
          ...
          "settings": {
              "seed": int(seed),
              ...
          },
          ...
      }
  ```
- Bug: `seed` is passed in, written into the output `settings` block, but no RNG is ever seeded by it. The fixtures are deterministic functions of `x0`, `target`, `upper_bounds`, `objective_weights`, `constraint_scales` (all literal tuples in `default_fixtures()`), and the ALM run is itself deterministic for a given start point. Changing `--seed` cannot affect the run.
- Why: Provenance scaffolding pre-dates the actual RNG usage; the fixtures were intended to grow stochastic noise (e.g. randomized x0 perturbations) but never did.
- Impact: Operators who try to ablate the benchmark with `--seed` see identical results and may believe the benchmark machinery is broken. The recorded "seed" in the output is misleading provenance.
- Suggested fix: Either (a) actually use the seed (`np.random.default_rng(seed)` to perturb x0 by a small amount per fixture), or (b) drop the parameter and the recorded field. Option (a) is more in spirit with what the benchmark seems to want to measure; option (b) is the strictly correct cleanup.

## CLI Argument Inventory

### `run_single_stage_thresholded_physics_alm.py`

| Flag | Type | Default | Validator | Unit / Notes |
|---|---|---|---|---|
| `--python-executable` | str | `sys.executable` | - | path |
| `--dry-run` | bool flag | False | - | - |
| `--plasma-surf-filename` | str (required) | - | `.name` extraction only | filename, not path |
| `--stage2-bs-path` | str (required) | - | `resolved_path` | path |
| `--allow-init-only-stage2-seed` | bool | False | enforced in `validate_stage2_seed_not_init_only` | - |
| `--equilibria-dir` | str | None | - | path |
| `--equilibrium-path` | str | None | - | path |
| `--seed-order-upgrade` | int | env `SEED_ORDER_UPGRADE` | - | Fourier order |
| `--stage2-seed-surf-path` | str | env | - | path |
| `--warm-start-surface-stem` | str | None | - | path |
| `--output-root` | str | repo subdir | - | path |
| `--summary-json` | str | None (auto) | - | path |
| `--single-stage-timeout-seconds` | float | 0.0 (=> None) | `timeout_or_none` | seconds |
| `--nphi` | int | 91 | - | quadrature |
| `--ntheta` | int | 32 | - | quadrature |
| `--mpol` | int | 8 | - | Fourier |
| `--ntor` | int | 6 | - | Fourier |
| `--maxiter` | int | 300 | - | L-BFGS-B |
| `--iota-target` | float | 0.20 | - | dimensionless |
| `--vol-target` | float | 0.10 | - | normalized |
| `--constraint-weight` | float | env `CONSTRAINT_WEIGHT` else 1.0 | - | Boozer mode select; <0 = exact |
| `--boozer-I` | float | env or None | - | internal |
| `--plasma-current-A` | float | env or None | - | SI A |
| `--single-stage-banana-current-mode` | choice | env or `shared` | - | enum |
| `--num-tf-coils` | int | env or 20 | - | count |
| `--banana-surf-radius` | float | env or None | - | meters |
| `--stage2-seed-tf-current-A` | float | env or None | - | SI A (legacy backfill) |
| `--cc-dist` | float | 0.05 | - | meters |
| `--cs-dist` | float | 0.015 | - | meters |
| `--curvature-threshold` | float | 100.0 | `validate_constraint_cli_overrides` | 1/m |
| `--hardware-search-mode` | choice | `warn` (only) | - | enum |
| `--alm-max-outer-iters` | int | 20 | inner `validate_alm_cli_args` | count |
| `--alm-max-subproblem-continuations` | int | 4 | inner | count |
| `--alm-penalty-init` | float | 1.0 | inner | dimensionless |
| `--alm-penalty-scale` | float | 10.0 | inner (>1) | dimensionless |
| `--alm-penalty-max` | float | 1e8 | inner (>= init) | dimensionless |
| `--alm-feas-tol` | float | 1e-4 | inner (>0) | normalized violation |
| `--alm-stationarity-tol` | float | 1e-4 | inner (>0) | augmented gradient norm |
| `--alm-trust-radius-init` | float | 0.05 | inner (>=0) | relative |
| `--alm-trust-radius-min` | float | 1e-4 | inner (>0) | relative |
| `--alm-trust-radius-shrink` | float | 0.5 | inner (0,1) | factor |
| `--alm-trust-radius-grow` | float | 1.5 | inner (>1) | factor |
| `--alm-max-inner-attempts` | int | 4 | inner (>0) | count |
| `--alm-distance-smoothing` | float | 0.005 | inner (>0) | meters |
| `--alm-curvature-smoothing` | float | 0.05 | inner (>0) | meters/curvature |
| `--alm-qs-threshold` | float | 3.0e-3 | inner `require_positive_alm_threshold` | dimensionless QS |
| `--alm-boozer-threshold` | float | 1.0e-2 | inner | residual unit |
| `--alm-iota-penalty-threshold` | float | 1.0e-4 | inner | **squared-penalty (see F2)** |
| `--alm-length-penalty-threshold` | float | **0.0 (F1: rejected!)** | inner | squared-penalty meters^2 |

### `run_stage2_alm.py`

| Flag | Type | Default | Validator | Unit / Notes |
|---|---|---|---|---|
| `--python-executable` | str | sys | - | path |
| `--dry-run` | bool | False | - | - |
| `--plasma-surf-filename` | str (required) | - | `.name` | filename |
| `--profile` / `--stage2-spec-json` | mutex required | - | profile choices / spec JSON validator | spec source |
| `--equilibria-dir` | str | None | - | path |
| `--output-root` | str | repo subdir | - | path |
| `--summary-json` | str | None (auto) | - | path |
| `--stage2-timeout-seconds` | float | 0.0 | `timeout_or_none` | seconds |
| `--cc-threshold` | float | None | downstream contract | meters |
| `--curvature-threshold` | float | None | downstream contract | 1/m |
| `--order` | int | None | downstream | Fourier |
| `--tf-current-A` | float | None | downstream contract | SI A |
| `--banana-current-max-A` | float | None | downstream contract | SI A |
| `--length-target` | float | None | downstream | meters |
| `--target-lcfs-max-major-radius-m` | float | None | downstream | meters |
| `--target-lcfs-max-minor-radius-m` | float | None | downstream | meters |
| `--banana-surf-radius` | float | None | downstream | meters |
| `--toroidal-flux` | float | None | downstream | normalized [0,1] |
| `--stage2-iota-mode` | choice | `off` | `validate_stage2_iota_args` | off/report/alm |
| `--stage2-iota-target` | float | None | required if mode != off | dimensionless iota |
| `--stage2-iota-tolerance` | float | 5e-3 | `validate_stage2_iota_args` (>0) | **deviation: \|iota - target\|** |
| `--stage2-iota-vol-target` | float | 0.10 | (>0) | normalized |
| `--stage2-iota-constraint-weight` | float | 1.0 | - | Boozer-mode select |
| `--stage2-iota-num-tf-coils` | int | 20 | (>0) | count |
| `--stage2-iota-nphi` | int | 91 | (>0) | quadrature |
| `--stage2-iota-ntheta` | int | 32 | (>0) | quadrature |
| `--stage2-iota-mpol` | int | 8 | (>0) | Fourier |
| `--stage2-iota-ntor` | int | 6 | (>0) | Fourier |
| _ALM tuning_ | _via JSON spec only_ | _BASE profile_ | _**F6: not validated at wrapper boundary**_ | _various_ |
| _basin tuning_ | _via JSON spec only_ | basin_hops=0, basin_seed=None | `_normalize_basin_seed` (**F4**) | counts/floats |

### `run_stage2_iota_decision_gate.py`

Wraps `run_stage2_alm`. Adds `--benchmark-modes`, `--baseline-mode`, `--minimum-iota-error-improvement`, `--max-acceptable-runtime-multiplier`, `--donor-repair-summary`. All validated in `validate_args`.

### `run_alm_normalization_fixture_benchmark.py`

Thin wrapper over `banana_opt.alm_fixture_benchmarking.main`. Inherits CLI:
- `--autoresearch-root` (Path | None)
- `--output` (Path)
- `--seed` (int, default 0) — **F8: recorded only, no RNG seeding**
- `--stamp` (default = utc_stamp())

### `collect_alm_autoresearch_baseline.py`

Thin wrapper over `banana_opt.alm_benchmarking.main`:
- `--autoresearch-root`
- `--output-dir`
- `--stamp`
- `--baseline-jsonl` (optional comparison input)

## Verdict

The runners are operationally close to correct but have two shipping defects:

1. **F1 is critical and immediate**: the canonical `run_single_stage_thresholded_physics_alm.py` runner does not start with default arguments. Every CI / manual / autoresearch invocation that does not explicitly pass `--alm-length-penalty-threshold <positive>` aborts inside the subprocess. Combined with F3 (no help text), an operator who hits this has no in-tool guidance to recover. **Fix this before any further ALM hardening lands.**

2. **F2 is a documentation defect with operational impact**: the `--alm-iota-penalty-threshold` help text is mathematically wrong by 2×. Users following the recipe systematically under-budget iota. The fix is one line in either the help text or the objective wrapper.

The remaining HIGHs (F3, F4) and MEDIUMs (F5, F6) are quality-of-life and reproducibility items that align with the project's SSOT/IMMUTABLE/REPRODUCIBILITY principles in `CLAUDE.md`. F4 specifically is a path-encoding determinism bug that masquerades as benign because it always "works" on first invocation.

LOWs (F7, F8) are cosmetic / provenance hygiene — clean up while in the area.

No `ACCEPT_OFFSPEC*` survivals were found in the audited runner scope (commit `d61648f50` cleanup is complete in this surface).
