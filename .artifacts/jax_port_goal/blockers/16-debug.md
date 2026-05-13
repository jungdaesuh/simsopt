# Item 16 Blocker — `field/tracing.py` JAX-native wrapper port

Status: **BLOCKED** with category `missing_dependency`.

Closure level: `blocked_dependency`.

## Item 16 scope

`src/simsopt/field/tracing.py` (936 LOC) is the Python public wrapper around
the C++ tracing kernels (`particle_guiding_center_tracing`,
`particle_guiding_center_boozer_tracing`, `particle_fullorbit_tracing`,
`fieldline_tracing`) plus the 9 `StoppingCriterion` subclasses re-exported
from `simsoptpp`. Per the goal prompt's manifest, item 16 depends on
item 14.

## Why this is BLOCKED

Item 14 is BLOCKED with `missing_dependency` (see
`.artifacts/jax_port_goal/blockers/14-debug.md`). Until item 14's JAX
kernel exists with parity coverage, item 16 cannot wire a JAX backend
selection path or replace the `sopp.*tracing` calls in
`src/simsopt/field/tracing.py:186, 295, 300, 707`. Touching this file
without a JAX kernel would either:

- Add `try/except ImportError` around a non-existent JAX path (violates
  the no-broad-except invariant), or
- Add a feature-flag scaffold for a non-existent backend (violates the
  no-shims invariant), or
- Silently fall back to the CPU C++ path (violates the no-silent-fallback
  invariant).

None of these is acceptable per the goal prompt's section-2 architecture
invariants and section-6 anti-patterns.

## Diagnostic budget

Not consumed. `missing_dependency` does not require the two-timebox
budget per goal prompt section 5.

## Proposed user decision

Same as item 14. If the human expands `active_scope` to permit the item
14 MVP carve-out, item 16 can immediately follow once item 14 is closed.
Otherwise, item 16 remains BLOCKED.

## State.json entry

```json
{
  "id": "16",
  "tier": "P2",
  "title": "field tracing wrappers",
  "status": "blocked",
  "closure_level": "blocked_dependency",
  "blocker": {
    "category": "missing_dependency",
    "detail": "Item 16 depends on item 14 (tracing RK path), which is BLOCKED. No JAX backend for trace_particles/fieldline_tracing exists yet, so the wrapper cannot route to a JAX path without violating anti-pattern invariants.",
    "debug_artifact": ".artifacts/jax_port_goal/blockers/16-debug.md",
    "needs_user": true
  },
  "evidence": {
    "source_audit": "src/simsopt/field/tracing.py (936 LOC, 8 sopp.* call sites + 9 StoppingCriterion subclasses)",
    "upstream_oracle": "upstream byte-identical at SHA 1b0cc3a96063197cdbdd01559e04c25456fbe6ff",
    "upstream_audit_sha": "1b0cc3a96063197cdbdd01559e04c25456fbe6ff",
    "depends_on": ["14"],
    "cuda_smoke": "not_claimed"
  }
}
```
