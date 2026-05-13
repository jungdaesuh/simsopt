# Item 12 — Coverage Matrix

`upstream_audit_sha: 1b0cc3a96063197cdbdd01559e04c25456fbe6ff`

Sources scanned:

```bash
git grep -nE "ToroidalField|PoloidalField|MirrorModel|CircularCoil" tests/
git -C /Users/suhjungdae/code/opensource/simsopt grep -nE \
  "ToroidalField|PoloidalField|MirrorModel|CircularCoil" src/simsopt/tests/
```

## ToroidalField

| Repo / upstream | Test path                                                              | Node ID                                  | Intent                                                                                                     | Classification             |
| --------------- | ---------------------------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------- |
| repo            | `tests/field/test_magneticfields.py:37`                                | `Testing::test_toroidal_field`           | CPU oracle: B, dB, d2B, A, dA, d2A on 20-point random fixture; checks divergence and Hessian symmetry.     | `oracle_only` (CPU oracle) |
| repo            | `tests/field/test_magneticfields.py:86`                                | `Testing::test_sum_Bfields`              | CPU sum-of-fields integration (helical + toroidal); item-10 / item-15 territory.                           | `wrapper_only`             |
| repo            | `tests/jax_core/test_analytic_pure_fields_item12.py` (new)             | `test_toroidal_field_jax_vs_cpu`         | JAX parity vs CPU class at `direct_kernel` lane: B, dB, d2B, A, dA at 50 random points.                    | `covered_by_unit_parity`   |
| repo            | `tests/jax_core/test_analytic_pure_fields_item12.py` (new)             | `test_jax_paths_under_strict_transfer_guard` | Strict `transfer_guard("disallow")` discipline for ToroidalField kernels (B, dB, d2B, A, dA).         | `covered_by_unit_parity`   |
| upstream        | `tests/field/test_magneticfields.py::Testing::test_toroidal_field`     | upstream node                            | Same CPU oracle (this repo's test is the migrated copy of upstream).                                       | `oracle_only`              |

## PoloidalField

| Repo / upstream | Test path                                                              | Node ID                                | Intent                                                                          | Classification             |
| --------------- | ---------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------- | -------------------------- |
| repo            | `tests/field/test_magneticfields.py:1261`                              | `Testing::test_poloidal_field`         | CPU oracle: B and dB at two reference points against hand-derived analytical.   | `oracle_only`              |
| repo            | `tests/jax_core/test_analytic_pure_fields_item12.py` (new)             | `test_poloidal_field_jax_vs_cpu`       | JAX parity vs CPU class at `direct_kernel` lane: B, dB at ≥ 50 filtered points. | `covered_by_unit_parity`   |
| repo            | `tests/jax_core/test_analytic_pure_fields_item12.py` (new)             | `test_jax_paths_under_strict_transfer_guard` | Strict transfer-guard discipline for PoloidalField kernels.               | `covered_by_unit_parity`   |
| upstream        | `tests/field/test_magneticfields.py::Testing::test_poloidal_field`     | upstream node                          | Mirror of repo CPU oracle.                                                      | `oracle_only`              |

## MirrorModel

| Repo / upstream | Test path                                                              | Node ID                              | Intent                                                                              | Classification             |
| --------------- | ---------------------------------------------------------------------- | ------------------------------------ | ----------------------------------------------------------------------------------- | -------------------------- |
| repo            | `tests/field/test_magneticfields.py:708`                               | `Testing::test_MirrorModel`          | CPU oracle: B and dB at one Mathematica-reference point (Rogerio Jorge notebook).   | `oracle_only`              |
| repo            | `tests/jax_core/test_analytic_pure_fields_item12.py` (new)             | `test_mirror_field_jax_vs_cpu`       | JAX parity vs CPU class at `direct_kernel` lane: B, dB at ≥ 50 filtered points.     | `covered_by_unit_parity`   |
| repo            | `tests/jax_core/test_analytic_pure_fields_item12.py` (new)             | `test_jax_paths_under_strict_transfer_guard` | Strict transfer-guard discipline for MirrorModel kernels.                     | `covered_by_unit_parity`   |
| upstream        | `tests/field/test_magneticfields.py::Testing::test_MirrorModel`        | upstream node                        | Mirror of repo CPU oracle.                                                          | `oracle_only`              |

## CircularCoil (deferred sub-item)

| Repo / upstream | Test path                                                              | Node ID                                          | Intent                                                                            | Classification            |
| --------------- | ---------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------- | ------------------------- |
| repo            | `tests/field/test_magneticfields.py` (multiple subtests)               | `Testing::test_CircularCoil`, related            | CPU oracle for off-axis circular coil via elliptic integrals.                     | `blocked` (see blocker)   |
| upstream        | `tests/field/test_magneticfields.py::Testing::test_CircularCoil`       | upstream node                                    | Mirror of repo CPU oracle.                                                        | `oracle_only`             |

The `blocked` row references
`.artifacts/jax_port_goal/blockers/12-circularcoil-debug.md` and points at
the missing `jax.scipy.special.ellipk` / `ellipe` symbols. No JAX parity
test is added for CircularCoil in this run.

## Auxiliary shape / dtype / validation

| Test path                                                              | Node ID                                       | Classification             |
| ---------------------------------------------------------------------- | --------------------------------------------- | -------------------------- |
| `tests/jax_core/test_analytic_pure_fields_item12.py`                   | `test_kernel_output_shapes_and_dtypes`        | `covered_by_unit_parity`   |
| `tests/jax_core/test_analytic_pure_fields_item12.py`                   | `test_rejects_malformed_points`               | `covered_by_unit_parity`   |

Coverage gate verdict: all rows classified; the only `blocked` row links
the CircularCoil debug artifact. The remaining three analytic fields are
covered by JAX unit parity tests at the `direct_kernel` lane tolerance
under both default and strict transfer-guard modes.
