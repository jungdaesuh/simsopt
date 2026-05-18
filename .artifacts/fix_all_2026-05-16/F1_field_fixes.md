# F1 — Field-adapter surgical fixes (2026-05-16)

Four targeted fixes to the simsopt-jax field-adapter layer.

## Fix 1 — H-9: SpecBackedBiotSavartJAX `_set_coil_dofs` shape assertion

**Issue.** `x.setter` writes the input vector into `self._dofs.full_x`
without verifying its shape matches the DOF count, allowing latent
shape mismatches when any DOF is fixed (`free_x != full_x`).

**Before** (`src/simsopt/field/biotsavart_jax_backend.py:496-499`,
pre-fix):

```python
def _set_coil_dofs(self, coil_dofs: object) -> None:
    self._x = _as_jax_float64(coil_dofs)
    self._coil_dofs_generation += 1
    self._coil_dof_state_token = _new_coil_dof_state_token()
```

**After** (`src/simsopt/field/biotsavart_jax_backend.py:496-503`):

```python
def _set_coil_dofs(self, coil_dofs: object) -> None:
    self._x = _as_jax_float64(coil_dofs)
    assert self._x.shape[0] == self.dof_size, (
        f"BiotSavartSpec DOF count {self._x.shape[0]} != "
        f"Optimizable.dof_size {self.dof_size}"
    )
    self._coil_dofs_generation += 1
    self._coil_dof_state_token = _new_coil_dof_state_token()
```

**Verification.**
- `ruff check`: All checks passed.
- `ruff format`: 1 file left unchanged.
- Import smoke: `SpecBackedBiotSavartJAX` imports correctly.
- `tests/field/test_biotsavart_jax.py`: 40 passed.

## Fix 2 — H-11: `BoozerRadialInterpolantJAX.as_dict` delegate to super

**Issue.** The method manually rebuilt the
`@module`/`@class`/`@name`/`@version` metadata block. Other JAX
wrappers (poloidal/scalar_potential/interpolated/toroidal/circular/
mirror/reiman/dipole/dommaschk) delegate to `super().as_dict()`.

**Before**
(`src/simsopt/field/boozermagneticfield_jax.py:904-915`, pre-fix):

```python
def as_dict(self, serial_objs_dict) -> dict:
    d = {
        "@module": self.__class__.__module__,
        "@class": self.__class__.__name__,
        "@name": getattr(self, "name", str(id(self))),
        "@version": None,
    }
    d["frozen_state"] = _frozen_state_to_host(self._frozen_state)
    d["psi0"] = self._psi0
    d["nfp"] = self._nfp
    d["points"] = self.get_points()
    return d
```

**After** (`src/simsopt/field/boozermagneticfield_jax.py:905-911`):

```python
def as_dict(self, serial_objs_dict) -> dict:
    d = super().as_dict(serial_objs_dict=serial_objs_dict)
    d["frozen_state"] = _frozen_state_to_host(self._frozen_state)
    d["psi0"] = self._psi0
    d["nfp"] = self._nfp
    d["points"] = self.get_points()
    return d
```

`super().as_dict()` chains into `Optimizable.as_dict ->
GSONable.as_dict`, which iterates `__init__` args and reads
`getattr(self, name)` / `getattr(self, "_" + name)`. The wrapper's
init takes `upstream` but historically did not retain it; a
companion edit at
`src/simsopt/field/boozermagneticfield_jax.py:836` (in `__init__`)
and `:853` (in `from_frozen_state`) sets
`self._upstream = None` so the iteration finds it and emits
`upstream: None` (downstream-safe; `from_dict` reconstructs via
`from_frozen_state` and ignores the key). The merged dict carries
the same `frozen_state` / `psi0` / `nfp` / `points` payload plus the
proper `@version` from the import-time module attribute.

**Verification.**
- `ruff check`: All checks passed.
- `ruff format`: 1 file left unchanged.
- Import smoke: `BoozerRadialInterpolantJAX` imports correctly.
- Manual `as_dict({})` invocation on a hand-rolled wrapper instance
  emits the keys `@module`, `@class`, `@name`, `@version`,
  `upstream` (None), `frozen_state`, `psi0`, `nfp`, `points`.

## Fix 3 — H-12: `InterpolatedFieldJAX.dB_by_dX` explicit Python error

**Issue.** `dB_by_dX()` was intentionally not implemented; the C++
trampoline forwarded the call to a binding that raised a confusing
low-level error. The class docstring buried the intention in a
multi-paragraph aside.

**Before** (`src/simsopt/field/interpolated_field_jax.py:21-27`,
class docstring + no method):

```rst
The wrapper deliberately does NOT implement ``_dB_by_dX_impl``. The CPU
:class:`InterpolatedField` does not implement it either (it raises a
runtime error inside the C++ binding); the upstream class exposes
``_GradAbsB_impl`` instead, computed from a separately-interpolated
``\\nabla |B|`` table. The JAX wrapper preserves this semantic: a
caller that needs Cartesian Jacobians of ``B`` should evaluate the
underlying source field directly.
```

**After** (`src/simsopt/field/interpolated_field_jax.py:21-23`,
class docstring; lines 229-234 add the explicit method):

```rst
The wrapper does NOT expose Cartesian Jacobians of ``B`` — see
:meth:`InterpolatedFieldJAX.dB_by_dX` for the explicit Python error
and the supported alternatives.
```

```python
def dB_by_dX(self):
    raise RuntimeError(
        "InterpolatedFieldJAX does not expose dB_by_dX in Cartesian "
        "coordinates. Use the source field directly, or call "
        "GradAbsB() for the physical gradient table."
    )
```

The override sits adjacent to `GradAbsB()` (the supported sibling
method) and short-circuits the C++ trampoline before it can emit
its low-level message. No fallback.

**Verification.**
- `ruff check`: All checks passed.
- `ruff format`: 1 file left unchanged.
- Manual call: `InterpolatedFieldJAX.__new__(...).dB_by_dX()` raises
  `RuntimeError` with the documented message.
- `tests/field/test_interpolated_field_jax_item15.py`: 20 passed.

## Fix 4 — H-13: `InterpolatedBoozerFieldFrozenState` frozen+mutable dict

**Issue.** The dataclass declared `frozen=True` while
`InterpolatedBoozerFieldJAX._ensure_spec` mutated the contained
`specs: dict` in place — a contradiction between the freeze
contract and the lazy-build semantic.

**Files.**
- `src/simsopt/jax_core/interpolated_boozer_field.py`
- `src/simsopt/field/boozermagneticfield_jax.py`
- `src/simsopt/jax_core/tracing.py` (downstream evaluator dispatch)

**Resolution (option B).** Specs ownership moves off the frozen
dataclass and onto the wrapper.

1. `InterpolatedBoozerFieldFrozenState.specs` is retyped as
   `Mapping[str, RegularGridInterpolant3DSpec]` and populated by
   `freeze_interpolated_boozer_field_state` via
   `MappingProxyType(dict(specs))` — an immutable read-only view of
   the initial frozen set
   (`src/simsopt/jax_core/interpolated_boozer_field.py:188`,
   `:530`).
2. `evaluate_scalar` now takes `specs` as an explicit `Mapping`
   parameter rather than reading `state.get(scalar_name)` against a
   mutable field
   (`src/simsopt/jax_core/interpolated_boozer_field.py:633-667`).
3. `InterpolatedBoozerFieldJAX` owns
   `self._lazy_specs: dict[str, RegularGridInterpolant3DSpec]`,
   initialized as `dict(state.specs)` at construction time and
   appended by `_ensure_spec`
   (`src/simsopt/field/boozermagneticfield_jax.py:1351`, `:1381`,
   `:1465`).
4. `_INTERP_EVALUATORS` value signature is now `(state, specs,
   points)`; `_eval_scalar_factory` threads `specs` to the
   underlying `evaluate_scalar`
   (`src/simsopt/field/boozermagneticfield_jax.py:1244-1277`).
5. `simsopt.jax_core.tracing._resolve_boozer_field_state` returns
   the triple `(state, specs, psi0)`; the
   `InterpolatedBoozerFieldJAX` branch extracts `_lazy_specs`, the
   tuple branches accept either `(state, psi0)` (specs pulled from
   the immutable `state.specs`) or `(state, specs, psi0)`. The
   `_boozer_field_evaluators` dispatcher binds the spec dict into
   the per-scalar callables so the inner-loop call signature stays
   `(state, point)` (`src/simsopt/jax_core/tracing.py:1883-1937`,
   `:1974-2058`).

**Before** (`src/simsopt/field/boozermagneticfield_jax.py:1473-1476`,
pre-fix lazy-build):

```python
# The frozen state is a frozen dataclass; ``specs`` is a regular
# dict held inside it, and we mutate that dict in place so the
# add is observable through the public ``specs`` attribute.
self._frozen_state.specs[name] = spec
```

**After** (`src/simsopt/field/boozermagneticfield_jax.py:1465-1473`):

```python
self._lazy_specs[name] = _interp_build_spec_for_scalar(
    self._field,
    scalar_name=name,
    rule=self._rule,
    s_range=self._frozen_state.s_range,
    theta_range=self._frozen_state.theta_range,
    zeta_range=self._frozen_state.zeta_range,
    extrapolate=self._frozen_state.extrapolate,
)
```

The frozen dataclass remains genuinely frozen — its
`MappingProxyType` cannot be mutated. The lazy-build cache lives
on the wrapper, where mutation is consistent with the wrapper's
non-frozen Optimizable contract. `state.has` and `state.get`
remain on the dataclass and now exclusively reflect the
construction-time frozen set; the wrapper's `_lazy_specs`
superset is the operational mapping fed to evaluators.

**Verification.**
- `ruff check`: All checks passed (3 files).
- `ruff format`: 3 files left unchanged.
- Import smoke: `InterpolatedBoozerFieldJAX`,
  `InterpolatedBoozerFieldFrozenState` import correctly.
- `tests/field/test_interpolated_boozer_field_jax.py`: 39 passed
  (covers `state.has`, lazy-build, from_frozen_state, evaluator
  parity vs CPU).
- `tests/field/test_trace_boozer_analytic_jax.py`: 25 passed
  (evaluator dispatch + RHS factory acceptance).
- `tests/field/test_boozer_analytic_jax.py`: 23 passed.
- `tests/jax_core/test_boozer_fixed_state_jax_item33.py`: 6 passed.
