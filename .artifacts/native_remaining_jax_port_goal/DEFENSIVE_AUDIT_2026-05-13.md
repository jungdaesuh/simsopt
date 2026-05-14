# Defensive-Code Adversarial Audit — N02/N04b/N03 §D Closure
Date: 2026-05-13
Branch: gpu-purity-stage2-20260405
Auditor: adversarial pass over the production code (no test-side coverage)

## Verdict
REQUIRES-FIX with 8 findings (1 MAJOR-MAJOR, 2 MAJOR, 3 MINOR, 2 LOW).

The Stage-2 closure is broadly clean — zero `try`/`except` blocks, no
dynamic imports (`importlib`/`__import__`), no `# TODO`/`# HACK`/`# XXX`
markers, no bare `pass`, no `typing.Any` annotations. Validation at
constructor boundaries is contract-correct.

However: the two lazy-import claims are **false** (no real cycle), the
N02 lazy-build mutation contract conflicts with the dataclass `frozen`
posture, the `_apply_symmetry` helper has a silent-fallback branch that
is dead under the current rule inventory but would mask a future bug if
a `value_size=2` rule with `apply_odd=True` is ever added, and a few
type erasures (`object`, bare `tuple`/`dict`) leak through.

## Findings by severity

### CRITICAL
None.

### MAJOR

**M1: False import-cycle claim in `_boozer_field_evaluators`** —
`src/simsopt/jax_core/tracing.py:1909-1953`. The docstring lines
1927-1934 state:

> The imports inside this function are deliberately lazy: top-level
> importing `simsopt.field.boozermagneticfield_jax` from
> `simsopt.jax_core.tracing` would pull in `simsopt._core.optimizable`
> and the rest of the field surface, which depend on this very module
> via `simsopt.field.tracing` in some import orderings.

I verified empirically that moving the imports at lines 1939-1953 and
1954-1968 to the top of `tracing.py` does **not** create a circular
import. Test method: patched a copy of `tracing.py` to add those
imports at module top, then `import simsopt.jax_core.tracing` — no
`ImportError`, no cycle. `simsopt.field.tracing` imports
`simsopt.jax_core.tracing` lazily inside functions (lines 395, 750),
so it never participates in the module-load graph for
`simsopt.jax_core.tracing` initialization. The cycle justification is
**fabricated**. Recommend either: (a) hoist the imports to module top
and delete the docstring paragraph, or (b) if the lazy posture is
desired for cold-start reasons, rewrite the docstring honestly (e.g.
"deferred for JAX-init cost amortization") rather than claiming a
non-existent cycle. As written, this is exactly the kind of
"defensive optimization disguised as necessity" that the CLAUDE.md
guardrails reject.

**M2: Undocumented lazy import in `SurfaceHenneberg.to_spec`** —
`src/simsopt/geo/surfacehenneberg.py:319-341`. The function lazily
imports `make_surface_henneberg_spec` from `..jax_core.specs` (line
328) with **no comment** explaining why. Empirically verified there is
**no real cycle**: hoisting the import to module top alongside
`import simsoptpp as sopp` does not break import resolution. The
audit prompt expected the claimed `simsoptpp ↔ jax_core` cycle to be
documented; it is not even claimed. The pattern is codebase-wide
across `surfacexyzfourier.py`, `curvexyzfourier.py`, etc., suggesting
an unstated optimization (avoid JAX import cost in CPU-only code
paths), but the new `SurfaceHenneberg.to_spec` ships without a
comment and without parity with the other lazy `to_spec`s'
docstrings. **Fix**: add a one-line comment matching the project's
existing convention, or hoist the import.

### MINOR

**N1: `_apply_symmetry` silent fall-through under an
inventory-impossible rule** —
`src/simsopt/jax_core/interpolated_boozer_field.py:580-607`. The
exhaustive branch tree is correct under the **current**
`SYMMETRY_EXPLOIT_SCALARS` inventory: every `apply_odd=True` entry has
`value_size=1`, the `apply_odd_vector_first_only=True` and
`apply_even=True` entries all have `value_size=3`, and `K_derivs`
(value_size=2) has all-False flags and hits the early `return raw` on
line 585. But the inner control flow is structured so that if a future
contributor introduces a rule with `apply_odd=True, value_size=2`
(or any value_size ∉ {1, 3}), execution falls through line 597 (which
only handles `value_size==3`), skips lines 598 and 602 (both False),
and silently `return raw` on line 607 — masking a real sign-flip bug.
Demonstrated this empirically by constructing a synthetic
`_SymmetryRule(value_size=2, apply_odd=True, ..., False)` and calling
`_apply_symmetry`: returns `raw` unchanged where the contract demands
column-0 negation. **Fix**: either replace the trailing `return raw`
with `raise AssertionError(f"unhandled rule: {rule}")`, or assert at
table-construction time that every `apply_odd=True` rule has
`value_size in (1, 3)`.

**N2: Mutable `specs: dict` inside a `@dataclass(frozen=True)`
container** —
`src/simsopt/jax_core/interpolated_boozer_field.py:164-219` plus
`src/simsopt/field/boozermagneticfield_jax.py:1431-1462`. The
dataclass is decorated `frozen=True`, the docstring says "Immutable
container", and line 197 marks the `specs` field with the comment
"immutable in spirit; not registered as a pytree". The wrapper
class then mutates the dict in `_ensure_spec` on line 1462
(`self._frozen_state.specs[name] = spec`). This is technically valid
Python — `frozen=True` only locks attribute *rebinding*, not the
mutability of nested objects — but it conflicts with the docstring
contract and with the CLAUDE.md "IMMUTABLE" guardrail. The C++
lazy-build motivation is legitimate, but the implementation pattern
should be one of: (a) drop `frozen=True` and document the lazy-build
mutation explicitly on `specs`, (b) use `types.MappingProxyType`
exposed publicly with the live mutable dict held privately, or (c)
return a new `InterpolatedBoozerFieldFrozenState` from
`_ensure_spec`. The current state lies to readers (docstring says
"immutable" while code mutates).

**N3: Type erasure `dict[str, object]` on `_INTERP_EVALUATORS`** —
`src/simsopt/field/boozermagneticfield_jax.py:1262`. The real type is
`dict[str, Callable[[InterpolatedBoozerFieldFrozenState, jax.Array],
jax.Array]]`. Annotating as `object` hides the contract and exactly
matches the CLAUDE.md guardrail against "`object` annotations that
hide the real type". **Fix**: add a `Callable` typedef or annotate
inline.

### LOW

**L1: Bare `tuple` / `dict` annotations on
`InterpolatedBoozerFieldFrozenState`** —
`src/simsopt/jax_core/interpolated_boozer_field.py:197, 202-204`.
`specs: dict`, `s_range: tuple`, `theta_range: tuple`,
`zeta_range: tuple` lack parameters. Real shapes are
`dict[str, RegularGridInterpolant3DSpec]` and
`tuple[float, float, int]`. Type-checker can't catch shape mismatches.

**L2: `dict.get(...)` + `is None` check on line 213-214 of
`interpolated_boozer_field.py`**. The `get` call returns None for
missing keys and conflates "absent" with "explicitly stored None".
Today the dict never stores None as a value, but the wrapper would
mask a future regression where None gets stored. Using
`if scalar_name not in self.specs: raise KeyError(...)` followed by
`return self.specs[scalar_name]` would be a less defensive contract.
This is borderline `LOW` — pedantic.

## Pattern-by-pattern audit

### try/except blocks
Zero hits across all six audit-target files
(`interpolated_boozer_field.py`, `boozermagneticfield_jax.py` lines
1240-1578, `surface_henneberg.py`, `specs.py` lines 154-205 and
973-1050, `surfacehenneberg.py:319-341`, `tracing.py:1835-2310`).
PASS.

### Silent fallbacks
1. `_apply_symmetry` line 607: terminal `return raw` — see **N1**
   (silent fall-through for `apply_odd=True, value_size ∉ {1, 3}`).
2. No silent `or default` fallbacks found.
3. `_resolve_boozer_field_state` lines 1863-1871: `getattr(boozer_field,
   "frozen_state", None)` + `getattr(boozer_field, "psi0", None)` +
   `is None` check that raises `TypeError`. The pattern is
   defensive-but-explicit: failure raises a typed error with the
   actual `type(...).__name__` in the message. **VERDICT: not a
   silent fallback** — it is a typed boundary check at a public
   factory entry point. ACCEPT.

### Defensive None guards
1. `interpolated_boozer_field.py:214` `if spec is None:` after
   `dict.get` — see **L2**.
2. `interpolated_boozer_field.py:514` `selected = tuple(ALL_SCALARS
   if scalars is None else scalars)` — this is the user-facing
   keyword-argument default sentinel; legitimate optional API. ACCEPT.
3. `boozermagneticfield_jax.py:1368` `wrapper._nfp = int(frozen_state
   .nfp if nfp is None else nfp)` — optional override of the meta
   field; legitimate API. ACCEPT.
4. `boozermagneticfield_jax.py:1443` `if self._field is None:
   raise KeyError(...)` — documented at the `from_frozen_state`
   contract (line 1357-1362). ACCEPT.
5. `boozermagneticfield_jax.py:1466` `if cached is None:` cache miss —
   standard memoize idiom. ACCEPT.

### Dynamic imports
Zero `importlib.import_module` or `__import__` hits. The four lazy
`from ...` imports inside functions (`_boozer_field_evaluators`,
`SurfaceHenneberg.to_spec`) are LAZY MODULE IMPORTS, not dynamic
imports — see **M1** and **M2** for cycle-claim verification (both
are phantom cycles).

### Hacky markers (TODO/HACK/XXX/FIXME)
Zero hits. PASS.

### Mutable-where-immutable
1. `InterpolatedBoozerFieldFrozenState.specs: dict` mutated by
   `InterpolatedBoozerFieldJAX._ensure_spec` — see **N2**.

### Type-erasing annotations
1. `_INTERP_EVALUATORS: dict[str, object]` — see **N3**.
2. Bare `dict` / `tuple` annotations on frozen-state fields — see
   **L1**.

### `_BOOZER_RHS_EVAL_KEYS` SSOT enforcement
Defined at `tracing.py:1893` with a 30-line docstring claiming it
holds "the SSOT for what each frozen-state branch must provide". In
production code it is **never indexed against** — the three RHS
factories (`guiding_center_vacuum_boozer_rhs`,
`guiding_center_no_k_boozer_rhs`, `guiding_center_boozer_rhs`)
duplicate the key lookups by string literal (`evals["modB"]`,
`evals["dGds"]`, etc.). The SSOT claim is enforced only by the test
`test_trace_boozer_analytic_jax.py:136, 206, 254`. This is a
**marginal** SSOT — the constant exists for tests, not for
production. **LOW** — fix by either consuming the tuple in
`_boozer_field_evaluators` to drive the registered key set, or
deleting the docstring claim and renaming to
`_TEST_EXPECTED_RHS_KEYS`.

### `_linear_state_at` deletion safety
Verified no remaining callers of `_linear_state_at` in
`src/`, `tests/`, or `benchmarks/`. ACCEPT.

## Confirmed safe

- N02 `freeze_interpolated_boozer_field_state` validation
  (`_validate_range`, `degree`/`nfp` bounds, `unknown` scalar list
  check at lines 503-520): contract-correct user-input boundary
  validation. The redundancy with `OneofIntegers`-style descriptors is
  intentional — these are constructor arguments to a *function*, not
  a class with a descriptor.
- N02 `_make_callback_for_scalar` (line 322) uses `getattr(field,
  scalar_name)` with **no default**, so missing getters surface as
  `AttributeError`. The shape check on line 332-336 raises
  `ValueError` with the actual shape. PASS.
- N04b `make_surface_henneberg_spec` `alpha_fac` validation matches
  CPU `OneofIntegers(-1, 0, 1)` descriptor. Shape validation against
  `(nmax+1,)` and `(mmax+1, 2*nmax+1)` is contract-strict (no
  permissive "if shape close enough" fallback). PASS.
- N04b `surface_henneberg.py` math: alpha = `0.5·nfp·alpha_fac` matches
  the CPU oracle at `surfacehenneberg.py:594-595`. The `_z0_n_mask`
  zeroes index 0 of `Z0nH` matching the CPU loop start at n=1
  (line 605 of the host file). The `_mn_indices_2d` valid mask zeroes
  `(m=0, n<=0)` per host convention (line 614). PASS.
- N03 §D `_BOOZER_RHS_EVAL_KEYS` consistency: each branch of
  `_boozer_field_evaluators` returns exactly those 12 keys, matching
  the analytic and radial evaluator inventories. PASS (though
  enforcement is test-side, see SSOT note).
- N03 §D `_resolve_boozer_field_state` tuple/instance dispatch: the
  contract documents both `(frozen_state, psi0)` tuple AND the
  wrapper instance, so the polymorphism is part of the documented
  API, not a hidden compatibility layer. PASS.
- The frozen state `get` raises `KeyError` instead of returning a
  sentinel — failure surfaces at the boundary as the docstring
  promises. PASS.

## Pattern-summary

| Pattern | Hits | Verdict |
|---|---|---|
| `try`/`except` | 0 | PASS |
| `# TODO`/`# HACK`/`# XXX`/`# FIXME` | 0 | PASS |
| `importlib.import_module` / `__import__` | 0 | PASS |
| `typing.Any` annotations | 0 | PASS |
| `object` annotations on internal helpers | 1 | **N3** |
| Bare `tuple` / `dict` annotations | 4 | **L1** |
| Lazy imports inside functions | 2 | **M1** (false cycle), **M2** (undocumented) |
| Mutable state in `frozen=True` dataclass | 1 | **N2** |
| Silent `return raw` fall-through | 1 | **N1** |
| `getattr(..., default)` | 2 | one in `_resolve_boozer_field_state` (typed boundary), accept |
| `if x is None` guards | 5 | all contract-correct except L2 |
| Magic numbers | 0 critical (only `0.5` in `α` formula, mathematically justified) | PASS |

## Recommendation
The N02/N04b/N03 §D work is high quality on the easy axes (no
try/except, no `typing.Any`, no `# TODO`, no real dynamic imports,
no hacky shortcuts), but **the two lazy-import docstring claims
should be fixed before this lands**: M1 contains a false cycle
justification that misleads future maintainers, and M2 ships without
any justification at all. The `_apply_symmetry` fall-through (N1)
and the `frozen=True` + mutable dict (N2) should be fixed in the
same pass to match the CLAUDE.md IMMUTABLE / SSOT guardrails.

---

# Iteration 2 — Re-audit Verdict
Date: 2026-05-13 (same session, post-fixes)

## Verdict
**REQUIRES-FIX** (2 small residual findings — N4 contradiction, L3 stale ref). The five original findings are all closed; the residuals are cleanup of the N2 docstring edit, not new defensiveness.

## Findings status

- **M1 (false cycle)**: CLOSED. `tracing.py:79-108` now imports the 13 radial + 13 analytic symbols at module top. `_boozer_field_evaluators` body contains only the two `isinstance` branches plus the `TypeError`. Empirically `import simsopt.jax_core.tracing` + `import simsopt.field.boozermagneticfield_jax` + `import simsopt` all succeed. Docstring cycle-break paragraph removed.
- **M2 (undocumented to_spec lazy import)**: CLOSED. `surfacehenneberg.py:328-335` now has an 8-line comment naming three sibling `to_spec` precedents and stating "empirically cycle-free; here purely to keep JAX optional." Honest.
- **N1 (silent `_apply_symmetry` fall-through)**: CLOSED. `interpolated_boozer_field.py:606-651` now raises `ValueError` with named rule fields on every unreachable branch (`apply_odd` with `value_size ∉ {1, 3}`, `apply_odd_vector_first_only` with `value_size != 3`, `apply_even` with `value_size != 3`, terminal multi-flag combination). Verified empirically by constructing all three hostile `value_size=2` rules — each raises.
- **N2 (frozen/mutable mismatch)**: PARTIAL. Docstring at lines 178-188 now honestly documents the mutability contract. CLOSED for the docstring. **See N4.**
- **N3 (object→Callable)**: CLOSED. `boozermagneticfield_jax.py:25` imports `Callable`; line 1263-1265 types `_INTERP_EVALUATORS: dict[str, Callable[[InterpolatedBoozerFieldFrozenState, jax.Array], jax.Array]]`.

## New findings introduced by iteration-2 fixes

- **N4 (MINOR)** — Stale inline comment at `interpolated_boozer_field.py:210`: `specs: dict  # immutable in spirit; not registered as a pytree`. The new docstring (lines 178-188) explicitly documents in-place mutation; the inline comment still claims "immutable in spirit". The two statements contradict. Fix: replace the inline comment with `# mutated in place by lazy-build; not registered as a pytree`.
- **L3 (LOW)** — Docstring at line 183 references `InterpolatedBoozerFieldJAX._ensure_spec_built`; the actual method (line 1434 of `boozermagneticfield_jax.py`) is named `_ensure_spec`. Dead cross-reference.

No new try/except, dynamic imports, silent fallbacks, `Any`/`object` annotations, or hacky markers were introduced by the fixes.

---

# Iteration 3 — Final Verdict
Date: 2026-05-13 (same session)

## Verdict
**PASS**.

- **N4 closed**: `interpolated_boozer_field.py:210-213` now carries a 3-line comment ("Append-only dict mutated in place... see 'Mutability contract'... not registered as a JAX pytree...") that is consistent with the class docstring at 178-188. No contradiction.
- **L3 closed**: docstring line 183 now reads `:meth:\`InterpolatedBoozerFieldJAX._ensure_spec\`` matching the actual method at `boozermagneticfield_jax.py:1434`.

No new defensive patterns, dynamic imports, silent fallbacks, type erasures, or stale comments introduced. Audit complete.
