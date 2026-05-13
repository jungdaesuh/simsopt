# Item 17 Coverage Matrix

Status: complete for CPU/JAX oracle closure. CUDA proof is not claimed.

Upstream audit SHA:
`1b0cc3a96063197cdbdd01559e04c25456fbe6ff`.

Repo HEAD audited: `a9da18facfda9a51f14534ea9d06e76803213db0`.

Greps used to populate the matrix:

```
git grep -nE "NormalField|CoilNormalField|normal_field" tests/ src/
git -C /Users/suhjungdae/code/opensource/simsopt grep -nE \
    "NormalField|CoilNormalField|normal_field" tests/
```

The repo grep returns 147 lines across 4 files; the upstream grep returns
146 lines across 2 files. Each row below merges multi-line grep hits into
a single test node when they exercise the same kernel or class.

| Source | Path / Node | Intent | Classification | Evidence |
| --- | --- | --- | --- | --- |
| current | `src/simsopt/field/normal_field.py:20` | `NormalField(Optimizable)` class definition | `covered_by_unit_parity` | `tests/field/test_normal_field.py::NormalFieldTests` plus `tests/field/test_normal_field_item17_closeout.py::test_normal_field_real_space_round_trip_bit_tight` |
| current | `src/simsopt/field/normal_field.py:50-83` | `NormalField.__init__` with DOF packing via `get_dofs` | `covered_by_unit_parity` | `test_normal_field.py::NormalFieldTests::test_get_index`, `test_dofs`, `test_getter_setter` |
| current | `src/simsopt/field/normal_field.py:100-141` | `NormalField.from_spec` and `from_spec_object` SPEC-input loaders | `wrapper_only` | `test_normal_field.py::test_initialize_normal_field_from_spec`, `test_dofs`, `test_make_names`, `test_change_resolution`, `test_get_set_vns_vnc_asarray`, `test_get_real_space_field` — all skipped without `py_spec`; out of `direct_kernel` scope because SPEC I/O is not portable |
| current | `src/simsopt/field/normal_field.py:167-260` | `get_dofs`, `get_index_in_array`, `get_index_in_dofs` index plumbing | `covered_by_unit_parity` | `test_normal_field.py::NormalFieldTests::test_get_index`, `test_getter_setter`, `test_wrong_index`, `test_check_mn`, `test_asarray_getter_setter_raises` |
| current | `src/simsopt/field/normal_field.py:300-336` | `_make_names` / `_make_names_helper` DOF naming | `covered_by_unit_parity` | `test_normal_field.py::test_make_names` |
| current | `src/simsopt/field/normal_field.py:338-399` | `change_resolution` and `fixed_range` | `covered_by_unit_parity` | `test_normal_field.py::test_change_resolution`, `test_fixed_range` |
| current | `src/simsopt/field/normal_field.py:401-491` | array-wise getters/setters `get_vns_asarray`, `set_vns_asarray`, etc. | `covered_by_unit_parity` | `test_normal_field.py::test_get_set_vns_vnc_asarray`, `test_asarray_getter_setter_raises` |
| current | `src/simsopt/field/normal_field.py:510-519` | `NormalField.get_real_space_field` via `surface.inverse_fourier_transform_scalar` and `surface.normal()` | `covered_by_unit_parity` | `test_normal_field.py::test_get_real_space_field` (SPEC-gated, atol-free, shape only) plus `tests/field/test_normal_field_item17_closeout.py::test_normal_field_real_space_round_trip_bit_tight` (production-scale, direct_kernel) |
| current | `src/simsopt/field/normal_field.py:522-555` | `CoilNormalField(NormalField)` constructor and CoilSet binding | `covered_by_unit_parity` | `test_normal_field.py::CoilNormalFieldTests::test_empty_init`, `test_inherited_methods_handled_correctly`, `test_vns_vns_setter_raises` |
| current | `src/simsopt/field/normal_field.py:573-597` | `CoilNormalField.vns` / `.vnc` property cache — `np.sum(coilset.bs.B().reshape((nphi, ntheta, 3)) * surface.normal() * -1, axis=2)` → `surface.fourier_transform_scalar(..., normalization=(2*pi)**2)` | `covered_by_unit_parity` | `test_normal_field.py::CoilNormalFieldTests::test_spec_coil_correspondence_on_converged_output` (SPEC-gated, places=6) plus `tests/field/test_normal_field_item17_closeout.py::test_coil_normal_field_vns_vnc_match_direct_cpu_oracle` (production-scale, direct_kernel, both symmetries) |
| current | `src/simsopt/field/normal_field.py:599-637` | `coilset` setter and `reduce_coilset` | `wrapper_only` | `test_normal_field.py::CoilNormalFieldTests::test_reduce_coilset`, `test_nonstellsym_reduce`, `test_double_reduction` — SPEC-runtime gated, exercise the Optimizable graph rather than the JAX hot path |
| current | `src/simsopt/field/normal_field.py:639-641` | `recompute_bell` — clears `_vns` / `_vnc` on parent DOF change | `covered_by_unit_parity` | `tests/field/test_normal_field_item17_closeout.py::test_coil_normal_field_recompute_bell_invalidates_cache` (production-scale, direct_kernel, both symmetries) |
| current | `src/simsopt/field/normal_field.py:643-675` | `CoilNormalField.get_vns/get_vnc/get_*_asarray` and the read-only setters that raise | `covered_by_unit_parity` | `test_normal_field.py::CoilNormalFieldTests::test_vns_vnc_asarray`, `test_vns_vns_setter_raises`, `test_wrong_index` |
| current | `src/simsopt/field/normal_field.py:677-713` | `CoilNormalField.change_resolution / fixed_range / get_dofs / get_index_in_dofs` — all raise | `covered_by_unit_parity` | `test_normal_field.py::CoilNormalFieldTests::test_inherited_methods_handled_correctly` |
| current | `src/simsopt/field/normal_field.py:715-751` | `CoilNormalField.optimize_coils` SciPy-driven flux-penalty optimizer | `wrapper_only` | `test_normal_field.py::CoilNormalFieldTests::test_optimize_coils` — exercises the SciPy / `coilset.flux_penalty` path, owned by item 03 / 04 not item 17 |
| current | `src/simsopt/geo/surfacerzfourier.py:2169` | `SurfaceRZFourier.fourier_transform_scalar` (NumPy) | `covered_by_unit_parity` | `tests/field/test_normal_field_item17_closeout.py::test_fourier_pair_identity_at_production_scale`, `::test_normal_field_real_space_round_trip_bit_tight` |
| current | `src/simsopt/geo/surfacerzfourier.py:2269` | `SurfaceRZFourier.inverse_fourier_transform_scalar` (NumPy) | `covered_by_unit_parity` | `tests/field/test_normal_field_item17_closeout.py::test_fourier_pair_identity_at_production_scale`, `::test_normal_field_real_space_round_trip_bit_tight` |
| upstream | `upstream_hss/master:src/simsopt/field/normal_field.py` | upstream class definitions, Fourier-pair convention, CoilSet binding | `oracle_only` | byte-identical to the fork at audit SHA; fork preserves upstream public surface and numerical contract |
| upstream | `upstream_hss/master:tests/field/test_normal_field.py` | upstream unit-test coverage (146 grep lines, 28 tests) | `oracle_only` | fork test file matches upstream modulo the three documented cosmetic edits in `17.md` (`requires_*` decorator hoist, `spec_wrapper` import, PEP-8 line-wrapping) |
| upstream | `upstream_hss/master:src/simsopt/geo/surfacerzfourier.py:2249-2253` | comment block documenting the "inverse Fourier transform(Fourier transform) = identity" band-limited contract | `oracle_only` | cited as the hand-derived reference for `test_fourier_pair_identity_at_production_scale` |
| current | `tests/mhd/test_spec.py:28,94-97,99,111-115,130-131` | SPEC-runtime wrapper test exercising `NormalField.get_vns` / `set_vns` | `wrapper_only` | SPEC F90 runtime gated; orchestrates the SPEC executable, not portable to JAX |
| current | `src/simsopt/mhd/spec.py:46,240,245,402-403` | SPEC Optimizable wrapper that consumes `NormalField` instances | `wrapper_only` | SPEC F90 runtime gated downstream consumer |
| current | `src/simsopt/field/coilset.py:35` | CoilSet docstring that references `NormalField` as a downstream user | `wrapper_only` | doc-only reference; CoilSet construction owned by item 04 |
| current | `tests/field/test_normal_field_item17_closeout.py` | New parity test added by item 17 | `covered_by_unit_parity` | new closeout test using `parity_ladder_tolerances("direct_kernel")`; five parameterized cases |

No matrix row is `unclassified`. Item 17 has no `blocked` row, no
`not_applicable` row, and no row that depends on a missing upstream
parity oracle.

## Empty-oracle note

The `direct_kernel` lane requires a `direct_cpp_oracle`. The
upstream-Python NumPy implementation of `fourier_transform_scalar` /
`inverse_fourier_transform_scalar` is the documented and audited oracle
for the SPEC-convention Fourier pair (no C++ counterpart exists in
`src/simsoptpp/`). For the `CoilNormalField` reduction-vs-direct parity,
the C++ binding `sopp.SurfaceRZFourier.normal()` and the C++
`BiotSavart.B()` parity oracle both anchor the reduction inputs at C++
oracle fidelity, and the NumPy `np.sum(...) * -1` reduction itself is
the hand-derived reference (matches the upstream `CoilNormalField.vns`
property definition byte-for-byte).

Empty-oracle items add at least one new parity test built against a
hand-derived / closed-form / NumPy oracle, with the oracle source cited
in the test docstring (section 4a). Item 17 is NOT an empty-oracle item:
upstream `tests/field/test_normal_field.py` provides 28 existing tests
that exercise every public method. The new closeout test in item 17
targets the section-4c "partial-coverage closeout" carve-out (NEW
invariants: production-scale floor + dual-symmetry coverage + strict
transfer-guard discipline + bit-tight `direct_kernel` lane on the
Fourier pair identity and the `CoilNormalField` reduction).
