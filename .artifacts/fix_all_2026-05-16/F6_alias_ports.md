# F6 — Resolve alias-only JAX ports in `simsopt/field/force.py`

Audit reference: `00_SYNTHESIS.md` line 249 (M-24) — "Public-API alias-only ports:
`B2EnergyJAX = B2Energy`, `LpCurveForceJAX = LpCurveForce` in `field/force.py:1320, 2284`.
Deceptive `JAX` suffix."

## §1 Investigation findings

### Parent class implementations are already pure JAX

`B2Energy` (`src/simsopt/field/force.py:1231-1317`) and `LpCurveForce`
(`src/simsopt/field/force.py:2064-2281`) both:

- Subclass `Optimizable`.
- Bind `self.J_jax` / `self.dJ_jax` to module-level `jit(...)` and
  `jit(grad(...))` closures over pure JAX kernels.

Source evidence (`force.py`):

```
951  _b2energy_eval = _b2energy_eval  # pure JAX
955  _B2ENERGY_JAX  = jit(_b2energy_eval, static_argnums=(3,))
956  _B2ENERGY_GRAD = jit(grad(_b2energy_eval, argnums=(0, 1, 2)), static_argnums=(3,))
964  _LP_FORCE_JAX  = jit(_lp_force_eval, static_argnums=(14,))
965  _LP_FORCE_GRAD = jit(grad(...), static_argnums=(14,))
...
1267-1268  self.J_jax  = _B2ENERGY_JAX
           self.dJ_jax = _B2ENERGY_GRAD
2171-2172  self.J_jax  = _LP_FORCE_JAX
           self.dJ_jax = _LP_FORCE_GRAD
```

`grep -n "simsoptpp\|sopp" src/simsopt/field/force.py` returns zero hits. There is
no C++ codepath: `J()` and `dJ()` route exclusively through JAX-jitted kernels.
The `*JAX` suffix on the deleted aliases conveyed nothing the canonical names
did not already carry.

### Cross-codebase references to the aliases (pre-fix)

```
$ grep -rn "B2EnergyJAX\|LpCurveForceJAX" src/simsopt/ tests/ benchmarks/ examples/ --include='*.py'
src/simsopt/field/force.py:49:    "B2EnergyJAX",
src/simsopt/field/force.py:52:    "LpCurveForceJAX",
src/simsopt/field/force.py:1320: B2EnergyJAX = B2Energy
src/simsopt/field/force.py:2284: LpCurveForceJAX = LpCurveForce
tests/field/test_force_item09_closeout.py:21:    B2EnergyJAX,
tests/field/test_force_item09_closeout.py:24:    LpCurveForceJAX,
tests/field/test_force_item09_closeout.py:102: assert field_mod.B2EnergyJAX is B2Energy
tests/field/test_force_item09_closeout.py:103: assert field_mod.LpCurveForceJAX is LpCurveForce
tests/field/test_force_item09_closeout.py:104: assert B2EnergyJAX is B2Energy
tests/field/test_force_item09_closeout.py:105: assert LpCurveForceJAX is LpCurveForce
tests/field/test_force_item09_closeout.py:111: LpCurveForceJAX,
tests/field/test_force_item09_closeout.py:112: B2EnergyJAX,
benchmarks/non_banana_example_cpp_jax_cpu_parity.py:778: jax_value=jax_lane.components["LpCurveForceJAX"]
benchmarks/non_banana_example_cpp_jax_cpu_parity.py:784: jax_value=jax_lane.components["LpCurveForceJAX"]
benchmarks/non_banana_example_cpp_jax_cpu_parity.py:790: jax_value=jax_lane.components["B2EnergyJAX"]
benchmarks/non_banana_example_cpp_jax_cpu_parity.py:796: jax_value=jax_lane.components["B2EnergyJAX"]
benchmarks/non_banana_example_parity_fixtures.py:6063: B2EnergyJAX = field_mod.B2EnergyJAX
benchmarks/non_banana_example_parity_fixtures.py:6065: LpCurveForceJAX = field_mod.LpCurveForceJAX
benchmarks/non_banana_example_parity_fixtures.py:6173: LpCurveForceJAX,
benchmarks/non_banana_example_parity_fixtures.py:6174: B2EnergyJAX,
benchmarks/non_banana_example_parity_fixtures.py:6219: "LpCurveForceJAX": jax_eval["force_value"],
benchmarks/non_banana_example_parity_fixtures.py:6220: "B2EnergyJAX": jax_eval["energy_value"],
benchmarks/non_banana_example_parity_fixtures.py:7397: "...LpCurveForceJAX and B2EnergyJAX..."
benchmarks/non_banana_example_parity_fixtures.py:7410: "...LpCurveForceJAX and B2EnergyJAX..."
```

The consumer tests/benchmarks explicitly documented that
`B2EnergyJAX is B2Energy` and `LpCurveForceJAX is LpCurveForce` (see
`test_force_item09_closeout.py:102-105` and the fixture spec text). The
"public lazy export" claim was the only thing the aliases provided.

### `Optimizable` contract

The parent contract (`src/simsopt/_core/optimizable.py`) requires `J()` /
`dJ()`. Both canonical classes already satisfy it via JAX. The aliases
inherit transparently; no additional contract is satisfied by the
identity assignment.

## §2 Decision

**Option A — delete the aliases and migrate consumers to the canonical
names.**

Rationale:

1. The parent classes are already JAX kernel-backed (`jit + grad` on pure
   JAX functions; no `simsoptpp` call sites in the methods). The aliases
   add no behaviour and only mislead readers into thinking there are two
   implementations.
2. Audit M-24 prescribes "delete or rename". Delete is the cleanest path:
   it eliminates the deceptive surface and forces consumers onto the
   honest canonical name.
3. All consumer sites (1 test, 2 benchmark modules) can use `B2Energy` /
   `LpCurveForce` directly without behavioural change because they were
   already running the same object via the identity alias.

The Option-A precondition in the prompt ("not referenced anywhere
outside `force.py`") strictly required zero external references, but the
task allowed Option B only when "the parents call into `simsoptpp`"; the
parents do not call `simsoptpp` here, so Option B is contraindicated.
The external references are migrated as part of the deletion.

## §3 Edits applied

### `src/simsopt/field/force.py`

`__all__` (lines 43-55, pre-fix had `"B2EnergyJAX"` at L49 and
`"LpCurveForceJAX"` at L52). Removed both entries.

```
Before:
__all__ = [
    "_coil_coil_inductances_pure",
    "_coil_coil_inductances_inv_pure",
    "_induced_currents_pure",
    "NetFluxes",
    "B2Energy",
    "B2EnergyJAX",
    "SquaredMeanForce",
    "LpCurveForce",
    "LpCurveForceJAX",
    "SquaredMeanTorque",
    "LpCurveTorque",
]

After:
__all__ = [
    "_coil_coil_inductances_pure",
    "_coil_coil_inductances_inv_pure",
    "_induced_currents_pure",
    "NetFluxes",
    "B2Energy",
    "SquaredMeanForce",
    "LpCurveForce",
    "SquaredMeanTorque",
    "LpCurveTorque",
]
```

`force.py:1320` (B2Energy alias) — removed:

```
Before:
    return_fn_map = {"J": J, "dJ": dJ}


B2EnergyJAX = B2Energy


def _net_fluxes_pure(

After:
    return_fn_map = {"J": J, "dJ": dJ}


def _net_fluxes_pure(
```

`force.py:2284` (LpCurveForce alias) — removed:

```
Before:
    return_fn_map = {"J": J, "dJ": dJ}


LpCurveForceJAX = LpCurveForce


def lp_torque_pure(

After:
    return_fn_map = {"J": J, "dJ": dJ}


def lp_torque_pure(
```

### `tests/field/test_force_item09_closeout.py`

Imports L19-26 — dropped `B2EnergyJAX` / `LpCurveForceJAX`. Renamed
`test_force_energy_jax_wrappers_are_public_lazy_exports` →
`test_force_energy_wrappers_are_public_lazy_exports`; the assertions
now check the canonical names resolve through the lazy export map.
`test_reduced_force_energy_wrappers_match_independent_cpu_lane` builds
both lanes via `(LpCurveForce, B2Energy)` (the JAX-vs-CPU separation was
illusory under identity aliasing; the test still independently
constructs two object graphs and gates on independent CPU oracles).

### `benchmarks/non_banana_example_parity_fixtures.py`

- L6063, L6065: dropped the alias unpackings.
- L6173-6174: pass canonical classes to `_build_terms`.
- L6219-6220: renamed lane component keys
  `"LpCurveForceJAX"`/`"B2EnergyJAX"` → `"LpCurveForce"`/`"B2Energy"`
  to match the CPU lane's keys.
- L7395-7410 (FixtureSpec text): rewrote the classification reason
  and acceptance criteria to reference the canonical class names and
  the honest "JAX-kernel-backed wrappers (jit + grad on pure JAX
  kernels with no simsoptpp call in J()/dJ())" framing.

### `benchmarks/non_banana_example_cpp_jax_cpu_parity.py`

L778, L784, L790, L796: switched the JAX lane's
`components["…JAX"]` lookups to `components["LpCurveForce"]` /
`components["B2Energy"]` (matches the renamed keys in
`non_banana_example_parity_fixtures.py`).

## §4 Verification

### ruff

```
$ ruff check src/simsopt/field/force.py tests/field/test_force_item09_closeout.py \
              benchmarks/non_banana_example_parity_fixtures.py \
              benchmarks/non_banana_example_cpp_jax_cpu_parity.py
All checks passed!

$ ruff format src/simsopt/field/force.py tests/field/test_force_item09_closeout.py \
              benchmarks/non_banana_example_parity_fixtures.py \
              benchmarks/non_banana_example_cpp_jax_cpu_parity.py
1 file reformatted, 3 files left unchanged
```

(Reformatted file: `benchmarks/non_banana_example_parity_fixtures.py` —
ruff trivially adjusted long-line splits around the rewritten text.)

### Import smoke

```
$ .conda/jax/bin/python -c "from simsopt.field.force import B2Energy, LpCurveForce; ..."
canonical imports ok
B2Energy simsopt.field.force
LpCurveForce simsopt.field.force

$ .conda/jax/bin/python -c "from simsopt.field.force import B2EnergyJAX"
ImportError: cannot import name 'B2EnergyJAX' from 'simsopt.field.force'

$ .conda/jax/bin/python -c "from simsopt.field.force import LpCurveForceJAX"
ImportError: cannot import name 'LpCurveForceJAX' from 'simsopt.field.force'

$ .conda/jax/bin/python  # lazy-export check
import simsopt.field as fm
assert 'B2EnergyJAX' not in fm.__all__
assert 'LpCurveForceJAX' not in fm.__all__
assert 'B2Energy' in fm.__all__
assert 'LpCurveForce' in fm.__all__
field __all__ clean
```

### Targeted test rerun

```
$ .conda/jax/bin/python -m pytest tests/field/test_force_item09_closeout.py -v
collected 3 items

tests/field/test_force_item09_closeout.py::test_force_energy_wrappers_are_public_lazy_exports PASSED
tests/field/test_force_item09_closeout.py::test_reduced_force_energy_wrappers_match_independent_cpu_lane PASSED
tests/field/test_force_item09_closeout.py::test_lp_curve_force_production_scale_taylor_parity_under_strict_transfer_guard PASSED

3 passed in 3.87s
```

### Zero residual references

```
$ grep -rn "B2EnergyJAX\|LpCurveForceJAX" src/simsopt/ tests/ benchmarks/ examples/ --include='*.py'
(no output)
```

Two parity-baseline JSON snapshots under
`.artifacts/parity/20260514-*/` still contain the historical
`"LpCurveForceJAX"` / `"B2EnergyJAX"` component keys. Those are frozen
audit artifacts (immutable evidence files), not source code. They are
not loaded by any consumer that survives this change and are left
untouched on purpose.
