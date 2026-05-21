# Backend Mode Rename Proposal — 2026-05-20

Status: draft, awaiting target-vocabulary approval. No code or doc consumer
changes have been made yet.

## Motivation

The current `SIMSOPT_BACKEND_MODE` vocabulary (declared as
`VALID_BACKEND_MODES` in `simsopt.backend.runtime`) reads like CI/QA lane
labels rather than user-facing product modes:

```
native_cpu
jax_cpu_fast
jax_cpu_parity
jax_cpu_float32_smoke
jax_gpu_fast
jax_gpu_parity
jax_mps_smoke
```

Concrete complaints:

1. `_smoke` is dev jargon. Users see "smoke" and reasonably interpret it as
   "CI smoke test only, not real" — which is what we mean for
   `jax_cpu_float32_smoke` and `jax_mps_smoke`, but the suffix doesn't say
   *why* a real user might still want to opt in (e.g. for Apple Silicon
   acceleration via `jax-mps`).
2. `_fast`/`_parity` collapse two orthogonal axes (device, validation tier)
   into a single underscore-joined string. Adding a third tier (e.g. quantized,
   mixed-precision, sharded) would require yet another combined name.
3. `jax_` prefix is redundant once `native_cpu` is the only non-JAX lane.
4. `jax_cpu_float32_smoke` encodes four axes in one identifier (backend,
   device, precision, tier).
5. `using_jax_backend.md` already hides the dev-flavored names: it advertises
   only `native_cpu`, `jax_cpu_parity`, `jax_gpu_parity`, `jax_gpu_fast`. The
   public surface and the SSOT disagree.

## Semantic axes that matter

Inferred from `_MODE_POLICY_DEFAULTS` in `simsopt.backend.runtime`:

| Axis              | Values today                          | Surfaces as |
|-------------------|---------------------------------------|-------------|
| backend kernel    | `cpu` (C++), `jax`                    | first segment |
| device platform   | `cpu`, `cuda`, `mps`                  | second segment |
| runtime dtype     | `float64`, `float32`                  | `_float32_` infix (smoke only) |
| validation tier   | parity-strict, perf, experimental     | `_parity` / `_fast` / `_smoke` suffix |
| parity_mode flag  | `True` only for `*_parity`            | derivable from suffix |

`tolerance_tier` (`cpu_reference`, `parity`, `fast`, `float32_smoke`) is the
*policy-side* expression of the same validation axis.

## Naming options

All three keep the `<device>_<tier>` shape, use `_` separators (consistent
with current style), and add deprecation aliases for the seven existing names.

### Option A — minimal product polish (small delta)

| Old                       | New                  | Why |
|---------------------------|----------------------|-----|
| `native_cpu`              | `native_cpu`         | unchanged; already neutral |
| `jax_cpu_parity`          | `jax_cpu_strict`     | "strict" is the contract noun, used elsewhere |
| `jax_cpu_fast`            | `jax_cpu_perf`       | "perf" reads as the speed-tier counterpart |
| `jax_gpu_parity`          | `jax_gpu_strict`     | mirror cpu |
| `jax_gpu_fast`            | `jax_gpu_perf`       | mirror cpu |
| `jax_cpu_float32_smoke`   | `jax_cpu_f32_preview`| `_preview` flags experimental + dtype shorthand |
| `jax_mps_smoke`           | `jax_mps_preview`    | `_preview` instead of `_smoke` |

**Strengths:** Smallest churn. Keeps the `jax_` prefix and underscore shape.
Replaces only the load-bearing tier words. `_strict` matches the
`SIMSOPT_BACKEND_STRICT` env var already in the codebase, so the vocabulary is
internally consistent.

**Weaknesses:** Still mixes backend + device + tier in one string.

### Option B — drop redundant `jax_` prefix

| Old                       | New                |
|---------------------------|--------------------|
| `native_cpu`              | `cpu_native`       |
| `jax_cpu_parity`          | `cpu_strict`       |
| `jax_cpu_fast`            | `cpu_perf`         |
| `jax_gpu_parity`          | `gpu_strict`       |
| `jax_gpu_fast`            | `gpu_perf`         |
| `jax_cpu_float32_smoke`   | `cpu_f32_preview`  |
| `jax_mps_smoke`           | `mps_preview`      |

**Strengths:** Shorter, device-led, easier to scan in CI matrix YAML. The
backend (C++ vs JAX) is an internal kernel detail; users care about *device*
and *validation contract*.

**Weaknesses:** Slightly bigger churn. Users skimming logs might miss the
backend identity (we'd compensate via `provenance_label` in the policy).

### Option C — explicit two-segment grammar

Use `device.tier` (or `device:tier`) instead of a single string, and split
parsing in `_validate_mode`:

```
SIMSOPT_BACKEND_MODE=gpu.strict
SIMSOPT_BACKEND_MODE=cpu.perf
SIMSOPT_BACKEND_MODE=mps.preview
SIMSOPT_BACKEND_MODE=cpu.native        # C++
SIMSOPT_BACKEND_MODE=cpu.f32.preview
```

**Strengths:** Most extensible. Future tiers (`gpu.quantized`,
`gpu.sharded.strict`) compose without inventing new monoliths.

**Weaknesses:** Largest churn — every consumer (tests, YAML, docs, env files,
shell snippets) changes shape, not just spelling. Period or colon in env var
values is shell-safe but historically unusual for this codebase.

## Recommendation

**Option A**, with one tweak: also rename `_smoke` → `_preview` in the
`tolerance_tier` policy field so policy and mode use the same word. The
churn-to-clarity ratio is best:

- 7 mode strings change, 5 of them in tier-word only.
- The validation contract (`*_strict` is the parity gate, `*_perf` is the
  no-parity lane) stays a one-word suffix matching the existing
  `SIMSOPT_BACKEND_STRICT` vocabulary.
- `_preview` is a real product word ("preview release"). It honestly conveys
  what `jax_mps_smoke` and `jax_cpu_float32_smoke` are: experimental lanes
  intended for users opting in, not just internal smoke tests.
- The `jax_` prefix stays, which keeps grep-ability and survives the case
  where someone adds a future `c_gpu_*` or `pure_cpu_*` kernel without
  breaking the existing names again.

If we instead prefer the device-led Option B style, the same `_strict` /
`_perf` / `_preview` tier suffixes apply, just with the prefix dropped.

## Migration plan (applies to whichever option lands)

The existing deprecation pathway (`_validate_mode` / `_raise_deprecated_selector`
in `simsopt.backend.runtime`) *raises* on a legacy value. That model worked for
`jax_metal_smoke` because the underlying plugin was being removed. For a
wholesale rename across 45 consumer files (including
`.github/workflows/jax_*.yml`, `docs/source/*.rst`, `tests/test_backend.py`
with ~50 literal `set_backend(...)` sites, and the `CLAUDE.md` quick-reference),
a hard cutover would be reckless.

Proposed staged approach:

**Phase 1 — Add new names, soft-deprecate old ones (one PR)**

- Extend `VALID_BACKEND_MODES` to include both the new and old names.
- Add a `_LEGACY_MODE_ALIASES = {old: new}` map. `_validate_mode` resolves
  old → new transparently and emits a `DeprecationWarning` (not
  `ValueError`).
- All `_MODE_POLICY_DEFAULTS`, `_MODE_TO_RUNTIME`, `_FIELD_KERNEL_DEFAULTS`,
  `_MODE_SHARDING_DEFAULTS`, `_TRANSFER_GUARD_DEFAULTS`,
  `_DEFAULT_RESIDENCY_BY_MODE`, `provenance_label`, etc. key on the new
  names. The legacy aliases route through the new keys.
- `get_backend_mode()` returns the new name even if the user set the old one
  (with a one-time warning per process).
- Update `using_jax_backend.md`, `CLAUDE.md` env-snippets, and
  `docs/source/jax_gpu_setup.rst` to advertise the new names.

**Phase 2 — Migrate first-party consumers (separate PR per file family)**

Tests (~45 files), benchmarks (~6 files), CI workflows (3 yml files),
internal docs.

Order suggested:

1. `tests/test_backend.py` (highest concentration of literal mode strings).
2. `.github/workflows/jax_*.yml` and `scripts/jax_ci_contract.py`.
3. Other `tests/**` and `benchmarks/**`.
4. `docs/**` and `examples/**` shell snippets.

After each PR, run the relevant test slice with the deprecation warning
promoted to error (`-W error::DeprecationWarning`) to detect leftover usages.

**Phase 3 — Hard cutover (one PR, ≥1 release after Phase 1)**

- Move legacy names from soft aliases into `_DEPRECATED_MODES` (existing
  hard-fail mechanism). Error message points at the new name.
- Drop the legacy entries from `VALID_BACKEND_MODES`.
- Update `tolerance_tier="float32_smoke"` → `"float32_preview"` in
  `_MODE_POLICY_DEFAULTS` in `simsopt.backend.runtime`, and any callers
  (`tests/**`, `examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py`,
  search for `"float32_smoke"`).

## Consumer impact summary (Option A)

The mode literal appears in 45 files outside the runtime SSOT. Spot-checked
counts:

- `tests/test_backend.py`: ~50 literal `set_backend(...)` calls.
- `tests/integration/**`: 10 files, mostly one or two literals each.
- `.github/workflows/jax_smoke.yml`, `jax_gpu_parity.yml`,
  `jax_benchmark_reporting.yml`: env-var literals embedded in step shells.
- `docs/source/jax_gpu_setup.rst`, `docs/using_jax_backend.md`,
  `CLAUDE.md`: documentation prose.
- `docs/parity_dual_mode_contract_2026-05-08.md` and several other
  `docs/*_plan_*.md` reference mode names by string — but planning docs may
  stay frozen (historical record); only living docs need rewriting.

No public Python API name changes are required: `set_backend(mode)` continues
to accept either spelling during Phase 1.

## Open questions for the reviewer

1. **Option A, B, or C?** (Recommended: A, with `_preview`.)
2. **Hard cutover or soft alias?** (Recommended: soft alias for ≥1 release,
   then hard cutover.)
3. **Should `tolerance_tier` strings rename in lockstep** (`float32_smoke` →
   `float32_preview`, `parity` → `strict`, `fast` → `perf`), or stay separate
   from mode-name rename to keep the diff small?
4. **Should planning docs under `docs/*_plan_*.md` be rewritten,** or left as
   historical artifacts with a one-line note that mode names changed?
5. **Bonus: rename `SIMSOPT_BACKEND_STRICT`?** Currently this env var
   toggles strict mode independently of the validation tier. If we adopt
   `_strict` as the tier suffix, the env-var name becomes ambiguous; renaming
   to `SIMSOPT_STRICT_REJECT` or similar would help.

## Files that change in Phase 1 only

If we land the proposal:

```
src/simsopt/backend/runtime.py                     # SSOT + aliases + warning
docs/using_jax_backend.md                          # advertise new names
docs/source/jax_gpu_setup.rst                      # GPU runbook
CLAUDE.md                                          # quick-reference env snippet
tests/test_backend.py                              # add cover for alias warning
```

Everything else is Phase 2/3 churn that can be sequenced safely behind the
soft alias.
