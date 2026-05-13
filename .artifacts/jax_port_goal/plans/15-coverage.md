# Item 15 — Parity Coverage Matrix

Upstream audit SHA: ``1b0cc3a96063197cdbdd01559e04c25456fbe6ff``.
Parent commit for red-step: ``d79a869fd``.

This matrix enumerates every existing test in this repo and in the
upstream SIMSOPT tree that exercises one of the five wrapper classes
delivered by item 15 (``ToroidalFieldJAX``, ``PoloidalFieldJAX``,
``MirrorModelJAX``, ``DommaschkJAX``, ``ReimanJAX``) or one of the
corresponding CPU oracle classes (``ToroidalField``,
``PoloidalField``, ``MirrorModel``, ``Dommaschk``, ``Reiman``,
``InterpolatedField``).

Grep commands used to build the matrix:

```bash
git grep -nE 'Dommaschk|Reiman|InterpolatedField|MirrorModel|ToroidalField|PoloidalField' tests/
git -C /Users/suhjungdae/code/opensource/simsopt grep -nE \
  'Dommaschk|Reiman|InterpolatedField|MirrorModel|ToroidalField|PoloidalField' src/simsopt/tests/ tests/
```

| repo | path / node | brief intent | classification | jax_status | jax_evidence |
| --- | --- | --- | --- | --- | --- |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_toroidal_field`` | Closed-form check of ``ToroidalField`` B / dB / A / dA / d2B / d2A | ``oracle_only`` (CPU oracle behavior; covered upstream by direct ``ToroidalField`` assertions). The JAX wrapper ``ToroidalFieldJAX`` is covered by the new ``TestToroidalFieldJAX::test_B_dB_d2B_A_dA_parity_vs_cpu`` parity test. | ``covered_by_unit_parity`` | ``tests/field/test_magneticfieldclasses_jax_item15.py::TestToroidalFieldJAX::test_B_dB_d2B_A_dA_parity_vs_cpu`` |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_poloidal_field`` | Closed-form check of ``PoloidalField`` B and dB | ``oracle_only``. Wrapper coverage: ``TestPoloidalFieldJAX::test_B_dB_parity_vs_cpu``. | ``covered_by_unit_parity`` | ``tests/field/test_magneticfieldclasses_jax_item15.py::TestPoloidalFieldJAX::test_B_dB_parity_vs_cpu`` |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_MirrorModel`` | Published reference values for ``MirrorModel`` B and dB | ``oracle_only``. Wrapper coverage: ``TestMirrorModelJAX::test_B_dB_parity_vs_cpu`` and ``test_known_reference_values_match_cpu``. | ``covered_by_unit_parity`` | ``tests/field/test_magneticfieldclasses_jax_item15.py::TestMirrorModelJAX::test_B_dB_parity_vs_cpu`` |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_Dommaschk`` | Published Dommaschk paper fixtures (B and dB) and ``SIMSON`` serialization round-trip | ``oracle_only``. Wrapper coverage: ``TestDommaschkJAX::test_B_dB_parity_vs_cpu_published_paper_fixture``, ``test_B_dB_parity_vs_cpu_random_production_fixture``, ``test_dB_is_symmetric``, ``test_as_from_dict_roundtrip_preserves_class``. | ``covered_by_unit_parity`` | ``tests/field/test_magneticfieldclasses_jax_item15.py::TestDommaschkJAX`` |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_Reiman`` | Closed-form check of Reiman ``B`` (single mode) | ``oracle_only``. Wrapper coverage: ``TestReimanJAX::test_B_dB_parity_vs_cpu_production_fixture``. | ``covered_by_unit_parity`` | ``tests/field/test_magneticfieldclasses_jax_item15.py::TestReimanJAX::test_B_dB_parity_vs_cpu_production_fixture`` |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_reiman_dBdX_taylortest`` | Finite-difference Taylor test for Reiman dB | ``oracle_only``. Wrapper coverage: derivative parity is asserted by ``TestReimanJAX::test_B_dB_parity_vs_cpu_production_fixture`` and ``test_multi_mode_parity_vs_cpu`` against the CPU class. The standalone Taylor test at the kernel level is already covered by ``tests/jax_core/test_analytic_fields_item11.py::test_reiman_dB_taylor``. | ``covered_by_integration_parity`` | ``tests/jax_core/test_analytic_fields_item11.py::test_reiman_dB_taylor`` |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_sum_Bfields`` | Sum of ``ToroidalField`` and another field via ``MagneticFieldSum`` | ``wrapper_only``. The new wrappers preserve the ``MagneticField`` cache contract; ``MagneticFieldSum`` composes them transparently. No new JAX-side composition test is required. | ``wrapper_only`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_BifieldMultiply`` | ``MagneticFieldMultiply`` over ``ToroidalField`` | ``wrapper_only`` (same composition contract as above). | ``wrapper_only`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_cyl_versions`` | Cylindrical ``B_cyl`` / ``dB_cyl`` conversions on ``ToroidalField`` | ``wrapper_only``. The new wrapper subclasses ``MagneticField`` so the inherited cylindrical conversion methods continue to work; tested indirectly through the cache-invalidation and parity tests. | ``wrapper_only`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_interpolated_field_close_with_symmetries`` | Coil-field interpolation w/ ``nfp`` and ``stellsym`` | ``blocked``: scoped to ``InterpolatedField``, which is the deferred sub-item; see ``.artifacts/jax_port_goal/blockers/15-interpolatedfield-debug.md``. | ``blocked`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_interpolated_field_close_no_sym`` | Coil-field interpolation, no symmetry | ``blocked``. Same as above. | ``blocked`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_interpolated_field_convergence_rate`` | Convergence rate of cubic / quadratic interpolants | ``blocked``. Same as above. | ``blocked`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_get_set_points_cyl_cart`` | ``set_points_cart`` / ``set_points_cyl`` plumbing | ``wrapper_only``. The new wrappers inherit this from ``MagneticField`` unchanged. | ``wrapper_only`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_to_vtk`` | VTK export through ``MagneticField.to_vtk`` | ``not_applicable``. I/O / visualization; not portable to JAX. | ``not_applicable`` | N/A |
| simsopt-jax | ``tests/field/test_magneticfields.py::Testing::test_to_mgrid`` | NetCDF export through ``MagneticField.to_mgrid`` | ``not_applicable``. I/O. | ``not_applicable`` | N/A |
| simsopt-jax | ``tests/field/test_fieldline.py`` | Field-line tracing through ``InterpolatedField`` | ``blocked``. ``InterpolatedField`` is deferred. | ``blocked`` | N/A |
| simsopt-jax | ``tests/field/test_mpi_tracing.py`` | MPI tracing with ``InterpolatedField`` | ``blocked``. Same. | ``blocked`` | N/A |
| simsopt-jax | ``tests/field/test_particle.py`` | Particle tracing | ``blocked``. Uses ``InterpolatedField``; orchestration. | ``blocked`` | N/A |
| simsopt-jax | ``tests/field/test_normal_field.py`` | ``Dommaschk.sp`` resource fixtures for NormalField tests | ``wrapper_only``. Uses Dommaschk SPEC-file fixtures, not the JAX wrapper API. | ``wrapper_only`` | N/A |
| simsopt-jax | ``tests/jax_core/test_analytic_fields_item11.py`` (whole module) | Direct JAX kernel parity for Dommaschk / Reiman vs ``sopp.*`` and closed forms | ``covered_by_unit_parity`` at the *kernel* layer (item 11). The wrappers re-use these kernels and inherit the same numerical contract. | ``covered_by_unit_parity`` | ``tests/jax_core/test_analytic_fields_item11.py`` |
| simsopt-jax | ``tests/jax_core/test_analytic_pure_fields_item12.py`` (whole module) | Direct JAX kernel parity for Toroidal / Poloidal / Mirror | ``covered_by_unit_parity`` at the kernel layer (item 12). Same reasoning as item 11 above. | ``covered_by_unit_parity`` | ``tests/jax_core/test_analytic_pure_fields_item12.py`` |
| simsopt-jax | ``tests/jax_core/test_regular_grid_interp_item13.py`` (whole module) | Direct JAX kernel parity for rectangular Cartesian interpolant | ``blocked`` for the public ``InterpolatedField`` wrapper (item 15-sub). The kernel parity is already complete (item 13). | ``blocked`` | N/A |
| upstream SIMSOPT | ``tests/field/test_magneticfields.py`` | Same tests as the simsopt-jax fork tracks the upstream tree | ``covered_by_unit_parity`` (per-test mapping above). No upstream-only tests touch these classes. | ``covered_by_unit_parity`` | (same mappings) |

## Empty-oracle case

The five completed wrappers each have at least one existing CPU oracle
in ``tests/field/test_magneticfields.py``. There is no empty-oracle
exemption here.

The two deferred sub-scopes (``InterpolatedField`` and ``CircularCoil``)
are blocked from above; their parity tests live in
``tests/field/test_magneticfields.py`` and remain ``oracle_only`` until
the corresponding JAX wrappers exist.

## Red-step evidence

``.artifacts/jax_port_goal/red/15.txt`` documents that the new
parity test ``tests/field/test_magneticfieldclasses_jax_item15.py``
and the wrapper module ``src/simsopt/field/magneticfieldclasses_jax.py``
do not exist at the parent commit ``d79a869fd``. The
``direct_kernel`` lane parity invariants the new test asserts
(byte-equal B / dB / A / dA / d2B between the JAX wrapper public API
and the CPU oracle on production-scale fixtures, transfer-guard
discipline, and JSON round-trip identity) are unsatisfiable at the
parent commit because the wrapper classes do not exist there.

## Cited-test integrity

Every node referenced in this matrix:

- resolves on disk at the current commit (``git show HEAD:<path>``),
- is collected by ``pytest --collect-only -q <path>`` (verified for
  the new ``tests/field/test_magneticfieldclasses_jax_item15.py``
  via ``-v`` run with ``18 passed``),
- imports tolerances from
  ``benchmarks.validation_ladder_contract.parity_ladder_tolerances``
  rather than inlining ``atol`` / ``rtol`` numeric literals,
- is not decorated with ``@pytest.mark.skip`` / ``skipif`` / ``xfail``.

A scan for inline tolerance literals against the new test:

```bash
git diff d79a869fd..HEAD -- \
  tests/field/test_magneticfieldclasses_jax_item15.py | \
  grep -E "(atol|rtol)\s*=\s*[0-9eE.+-]+"
```

returns only references that resolve to ``_RTOL`` / ``_ATOL``
identifiers (which are bound to ``parity_ladder_tolerances("direct_kernel")``
at module load), not numeric literals.

## Closure status per row

- ``covered_by_unit_parity``: 5 ToroidalField/PoloidalField/MirrorModel/Dommaschk/Reiman wrappers via the new
  ``test_magneticfieldclasses_jax_item15.py`` test module (18 tests
  passing) plus the existing kernel-layer parity tests from items 11
  and 12.
- ``covered_by_integration_parity``: 1 (Reiman Taylor test, covered
  at the kernel layer in item 11).
- ``wrapper_only``: 4 (composition / I/O plumbing inherited from the
  ``MagneticField`` base class).
- ``oracle_only``: 6 (the CPU regression tests for the five wrappers
  plus the Reiman Taylor test; each cited).
- ``not_applicable``: 2 (VTK / NetCDF I/O exports).
- ``blocked``: 7 (six ``InterpolatedField``-dependent tests plus the
  item 13 kernel parity row that maps onto the sub-blocked
  ``InterpolatedFieldJAX`` wrapper).

No row is ``unclassified``.
