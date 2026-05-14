# Reviewer Oracle Lint

This document defines the **oracle lint** rule for new tests in this repository, especially `test_*_jax_*.py` files. Reviewers run this lint before approving a PR that adds or modifies tests.

## The Question

> For every assertion of equality or near-equality, name the independent oracle.

If the oracle is a re-implementation of the system under test (or another code path that calls the same kernel), the test is **tautological** — flag and reject.

## Acceptable Oracles

A valid oracle is one of:

1. **C++ reference symbol**: e.g., `simsoptpp::biot_savart_B`, `simsoptpp::surface_gamma`. Cite the C++ class/function and parameter set.
2. **Closed-form analytic expression**: e.g., planar circle centroid frame, helix with known constant torsion, `np.exp(0.1)`. Write the formula out.
3. **Pinned external dataset**: e.g., VMEC equilibrium reference, recorded baseline from an independent run. Cite the dataset version/path.
4. **Finite-difference gradient** of an analytic objective: valid for gradient tests, with tolerance derived from FD step size and condition number.

## Unacceptable Oracles (tautologies)

Reject these patterns:

- `assert wrapper.J() == kernel.J()` when `wrapper.J()` delegates to `kernel.J()`.
- `assert jax_path(x) == host_path(x)` when `host_path` invokes the same JAX kernel as `jax_path`.
- `assert module.foo is other_module.foo` (re-export identity).
- `assert numpy_reproduction(x) == jax_kernel(x)` when `numpy_reproduction` literally reimplements the same formula as `jax_kernel`.
- `assert payload["passed"] is True` when the same driver writes `payload["passed"]` and the other payload fields.
- `assert apply_transpose(solve_transpose(b)) ≈ b` as the only check — self-consistency of an operator and its inverse is trivially true.

## Required Annotations

Every parity-row claim in a closeout artifact under `.artifacts/jax_port_goal/` must cite:
1. **Test file**: full path with line anchor.
2. **Oracle**: which acceptable type (1-4 above) + the specific reference (C++ symbol / formula / dataset).
3. **Parity-ladder tolerance lane**: which lane from `benchmarks/validation_ladder_contract.py::PARITY_LADDER_TOLERANCES`. Cite both `rtol` and `atol`.

## Reviewer Checklist

When reviewing a PR that adds `test_*_jax_*.py`:

- [ ] Every `assert_allclose`/`np.allclose`/`assertEqual` has a named oracle (one of types 1-4).
- [ ] No `function is other_function` identity check on re-exports.
- [ ] No `jax_path(x) == host_path(x)` where `host_path` routes through JAX.
- [ ] No NumPy reproduction of the JAX formula being tested.
- [ ] No `payload["passed"]`/`payload["failures"] == []` as the headline assertion (verdict-circularity).
- [ ] Tolerance is derived from theory (FP analysis, truncation order, FD step), not chosen empirically to make the test pass.
- [ ] Test docstring names the oracle and the gate-tier (parity / smoke / routing / Tier-4 self-consistency).

## Related Audits

- `.artifacts/jax-test-audit-2026-05-13/TEST_QUALITY_TODOS.md` (Tier 1 tautologies #1-#6, with examples of each anti-pattern).

## Examples

### Good

```python
def test_rotated_centroid_frame_matches_planar_circle_analytic():
    """Anchor against closed-form planar-circle centroid frame.

    Oracle: closed-form analytic expression (type 2).
    Lane: direct-kernel, rtol=1e-12, atol=1e-12.
    """
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    gamma = np.column_stack([np.cos(2*np.pi*quadpoints), np.sin(2*np.pi*quadpoints), np.zeros(64)])
    tangent = np.column_stack([-np.sin(2*np.pi*quadpoints), np.cos(2*np.pi*quadpoints), np.zeros(64)])
    N0_analytic = np.column_stack([np.cos(2*np.pi*quadpoints), np.sin(2*np.pi*quadpoints), np.zeros(64)])
    B0_analytic = np.cross(tangent, N0_analytic)
    alpha = 0.0
    N_expected = np.cos(alpha) * N0_analytic - np.sin(alpha) * B0_analytic
    N_jax = rotated_centroid_frame(gamma, tangent, alpha=alpha, ...)["N"]
    np.testing.assert_allclose(N_jax, N_expected, rtol=1e-12, atol=1e-12)
```

### Bad — tautology

```python
def test_rotated_centroid_frame_matches_upstream():
    # rotated_centroid_frame and upstream_rotated_centroid_frame are the SAME
    # Python function object via re-export. This test is a tautology.
    from simsopt.geo.framedcurve import rotated_centroid_frame
    from simsopt.jax_core.framedcurve import rotated_centroid_frame as upstream_rotated_centroid_frame
    assert rotated_centroid_frame(...) == upstream_rotated_centroid_frame(...)
```
