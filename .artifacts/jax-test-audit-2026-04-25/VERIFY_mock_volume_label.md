# VERIFY — `_MockVolumeLabel.J()` audit finding

Branch: `gpu-purity-stage2-20260405`
Date: 2026-04-25
Subject: rebuttal to `bucket2_m3_m4_boozer.md` Top-issue #1 + #40 + P0 #2 (the
claim that `_MockVolumeLabel.J() -> 0.0` "silently zeroes the constraint
contribution to the penalty in ~70% of M4 tests").

## 1. Code call sites

### 1a. `src/simsopt/geo/boozersurface_jax.py`

A whole-file `grep` for `\.J(` returns exactly **one** hit, and it is a
docstring sentence — there is no executable `label.J()` anywhere in the
JIT-traced forward pass, exact-residual builder, adjoint, or gradient
plumbing:

```
1955:            for downstream consumers that call ``boozer_surface.label.J()``.
```

The constructor stores the label object only to derive a *string token*
(`label_type`):

```python
# boozersurface_jax.py L1990-2013
self.label = label
self.targetlabel = float(targetlabel)
...
label_cls = type(label).__name__
if "Volume" in label_cls:
    self.label_type = "volume"
elif "Area" in label_cls:
    self.label_type = "area"
elif "ToroidalFlux" in label_cls:
    self.label_type = "toroidal_flux"
else:
    raise ValueError(...)
```

The penalty value used in the JIT objective comes from
`_label_from_geometry_and_field_terms` (L743-758), which dispatches purely
on the string and recomputes from geometry/field arrays:

```python
def _label_from_geometry_and_field_terms(geometry, field_terms, params):
    normal = _cross_product(geometry.xphi, geometry.xtheta)
    if params.label_type == "volume":
        return volume_jax(geometry.gamma, normal)
    if params.label_type == "area":
        return area_jax(normal)
    ntheta = geometry.gamma.shape[1]
    return toroidal_flux_jax(
        _select_axis0(field_terms.A, params.phi_idx),
        _select_axis0(geometry.xtheta, params.phi_idx),
        ntheta,
    )
```

The penalty at L774-778 is then:

```python
label_val = _label_from_geometry_and_field_terms(geometry, field_terms, params)
gamma_axis_z = _surface_sample_z(geometry.gamma)
half = _as_jax_float64(0.5)
label_delta = label_val - params.targetlabel
J_label = half * params.constraint_weight * label_delta * label_delta
```

Note `label_val - params.targetlabel`, NOT `label.J() - params.targetlabel`.

The exact-constraints residual builder uses the same recomputed value
through `_compute_label` / `_compute_label_and_axis_z` (L783-837),
which again calls `volume_jax`/`area_jax`/`toroidal_flux_jax` — never
`self.label.J()`. Call sites: L971-982 (residual vector), L1102-1117
(exact residual), L2792-2803 (`_exact_constraints_vector`),
L3307-3318 (LS residual). Every one of them performs:

```python
label_value, gamma_axis_z = _compute_label_and_axis_z(
    gamma=..., xphi=..., xtheta=..., points=...,
    label_type=self.label_type, ..., coil_set_spec=...,
)
... = label_value - _as_jax_float64(self.targetlabel)
```

`self.label_type` is the inferred string token (`"volume"`/`"area"`/
`"toroidal_flux"`); `self.label` itself is never accessed inside any
JIT/autodiff path.

### 1b. `src/simsopt/geo/label_constraints_jax.py`

Pure functions of geometry / field; no external label object:

- `volume_jax = surface_volume(gamma, normal)` — re-exported from
  `surface_fourier_jax.py`; closed-form Stokes integral on `gamma`,
  `normal`.
- `area_jax = surface_area(normal)` — closed-form area integral on
  `normal`.
- `toroidal_flux_jax(A, gammadash2_at_phi, ntheta)` — Stokes line
  integral; no external state.
- `compute_G_from_currents(currents)` — `mu0 * sum(|I_k|)`.

None of these read any object that has a `.J()` method.

## 2. Test usage analysis

`_MockVolumeLabel` is used by `_make_mock_boozer_surface(_exact)`, by
`_make_mock_boozer_surface_with_free_currents`, by
`_make_basic_mock_surface_and_label`, and by ~10 ad-hoc construction
tests in `test_boozersurface_jax.py`. A repo-wide grep for `\.J\(\)`
across `tests/geo/test_boozersurface_jax.py`,
`tests/geo/test_boozersurface_jax_private.py`, and
`tests/geo/boozersurface_jax_test_helpers.py` returns **zero** hits.
No test code ever invokes `_MockVolumeLabel.J()`.

Categorising every test that constructs `_MockVolumeLabel` (directly or
via a `_make_mock_boozer_surface*` helper):

- **Pure construction / validation tests** (e.g. L2255-2414, L5342-5396,
  L6648-6715): assert that the constructor accepts/rejects option dicts
  and that `boozer_type`, `label_type`, `boozer_type == "ls"` etc. are
  set. The mock's only required behaviour is "type name contains the
  word `Volume`" — assertion is independent of `J()`.
- **Routing / monkeypatch tests** (e.g. `test_run_code_routes_backend_*`,
  `test_public_exact_constraints_newton_*`, almost every entry in
  `test_boozersurface_jax_private.py:1690-1899`): override
  `target_minimize`, `_run_newton_polish_for_method`, or
  `_make_exact_constraints_residual_with` so the actual penalty/residual
  callable is replaced before the solver runs. The result dict is
  fabricated by the fake. `J()` cannot influence anything.
- **Real run_code / penalty solves on the mock torus** (`test_run_code_ls_converges`
  L2594, `test_pack_unpack_roundtrip` L2586, the entire mixed-quad and
  composed-penalty cluster): assert `success=True`, the presence of
  `residual`/`jacobian`/`hessian`/`PLU`/`vjp` keys, and finiteness. The
  penalty value is computed via `volume_jax(gamma, normal)` against the
  closed-form torus target `2π² R r²` set by `_make_mock_boozer_surface`.
  Re-running the same kernel via `label.J()` would produce *literally the
  same number*; the mock returning `0.0` does not change anything because
  the kernel never reads it.

If `_MockVolumeLabel.J()` returned the analytic torus volume instead of
`0.0`, **no assertion in any of these tests would change**, because no
assertion's RHS is derived from the mock's `J()` value — every
penalty-relevant value is recomputed from `gamma`/`normal`/`A` inside
the JIT path.

## 3. Verdict

**Verdict: (a)** The user's claim is correct.

`BoozerSurfaceJAX` recomputes labels from geometry via
`volume_jax`/`area_jax`/`toroidal_flux_jax` (a.k.a. `_compute_label` /
`_label_from_geometry_and_field_terms`); it never reads `label.J()` in
the JIT forward path, exact residual, LS residual, adjoint, or any
public API. `self.label` is stored as an Optimizable dependency for
downstream CPU consumers and as a class-name token to set
`self.label_type`. The `_MockVolumeLabel.J() -> 0.0` is plumbing only
and was a P0 false alarm in the audit.

Citations: `boozersurface_jax.py` L100-106 imports `volume_jax`/`area_jax`
/`toroidal_flux_jax`/`compute_G_from_currents`; L743-758 dispatches the
penalty label by string; L774-778 forms `label_delta = label_val -
params.targetlabel`; L1990-2013 stores `self.label` only to extract
`self.label_type`; the only `.J(` token in the file is the docstring at
L1955.

## 4. Real residual finding (if any)

The audit's mechanical observation that `_MockVolumeLabel.J()` returns
`0.0` is *true but inert*. Two minor follow-ups remain useful:

1. **Naming clarity.** The mock is fine as plumbing, but the `J()`
   method is dead code. Either delete it (the constructor only inspects
   `type(label).__name__`), or rename the class to
   `_MockVolumeLabelMarker` / `_VolumeLabelStub` to make the
   "marker-only" intent obvious. Future audits will keep flagging the
   `0.0` return value otherwise.
2. **Audit hygiene.** Bucket2 P0 #2 (and the headline "70% of M4 tests
   silently null-test the constraint contribution") should be retracted
   in `bucket2_m3_m4_boozer.md` and `SYNTHESIS.md`. The constraint
   contribution is exercised by the recomputed `volume_jax(gamma, normal)
   - targetlabel` term, with `targetlabel = 2π² R r²` set by
   `_make_mock_boozer_surface`. The volume penalty has nonzero gradient
   w.r.t. surface DOFs in those tests.

What the audit *did* correctly identify, separately, is that
`_MockBiotSavart` is field-state-shallow (it only carries a coil-spec
and never refreshes field state), and that some run_code tests assert
only on routing/key-presence rather than on physical value. Those are
real (small) coverage holes, but they are independent of the
`_MockVolumeLabel.J()` claim and should not be conflated.
