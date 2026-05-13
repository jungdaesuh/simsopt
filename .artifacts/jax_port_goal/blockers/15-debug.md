# Item 15 Blocker -- public magneticfieldclasses JAX wrappers

Status: `blocked`.

Closure level: `blocked_dependency`.

## Item 15 Scope

Prompt item 15 owns the public wrappers in
`src/simsopt/field/magneticfieldclasses.py` that depend on Tier P1
ports:

- `Dommaschk` and `Reiman` wrappers over item 11
  `src/simsopt/jax_core/analytic_fields.py`.
- Public wrappers for item 12 analytic fields:
  `ToroidalField`, `PoloidalField`, `CircularCoil`, and `MirrorModel`.
- `InterpolatedField` over item 13
  `src/simsopt/jax_core/regular_grid_interp.py`.

## Blocking Dependency

The current worktree contains a partial implementation:

- `src/simsopt/field/magneticfieldclasses_jax.py`
- `tests/field/test_magneticfieldclasses_jax_item15.py`

That partial implementation covers `ToroidalFieldJAX`,
`PoloidalFieldJAX`, `MirrorModelJAX`, `DommaschkJAX`, and `ReimanJAX`.
It does not complete the prompt item because two scoped public surfaces
remain missing.

First, item 12 is intentionally partial. `ToroidalField`,
`PoloidalField`, and `MirrorModel` have JAX-native specs and
direct-kernel tests, but `CircularCoil` is deferred as
`12-circularcoil` because its analytic field requires complete elliptic
integrals. The current repo-local runtime is JAX/JAXLIB 0.10.0, and
`jax.scipy.special` does not expose `ellipk` / `ellipe`.

Second, `InterpolatedField` remains an architecture blocker. Item 13
ported the rectangular Cartesian `RegularGridInterpolant3D` kernel, but
the public Python `InterpolatedField` wrapper has cylindrical
`(r, phi, z)` semantics, `nfp` period folding, optional stellarator
symmetry folding, skip masks, and source-field sampling through the
public `MagneticField` API. See
`.artifacts/jax_port_goal/blockers/15-interpolatedfield-debug.md`.

The prompt forbids adding a new runtime dependency. Completing the item
15 public wrapper surface without `CircularCoil` and `InterpolatedField`
would report an incomplete public API as complete.

## Required Unblock

Resolve `12-circularcoil` first by adding an in-repo JAX-native complete
elliptic-integral implementation, for example a Carlson `R_F` / `R_D`
or Bulirsch `cel` kernel, with:

- direct CPU oracle parity against `scipy.special.ellipk` /
  `scipy.special.ellipe` in the regimes used by `CircularCoil`,
- value and `dB_by_dX` parity against the existing CPU `CircularCoil`
  wrapper,
- strict transfer-guard coverage,
- no new dependency.

After those dependencies land, item 15 can be promoted from blocked to
complete.

## Partial Validation

```bash
JAX_ENABLE_X64=True JAX_PLATFORMS=cpu \
  .conda/jax-0.9.2/bin/python -m pytest \
  tests/field/test_magneticfieldclasses_jax_item15.py -q
# 18 passed

SIMSOPT_JAX_TRANSFER_GUARD=disallow JAX_ENABLE_X64=True JAX_PLATFORMS=cpu \
  .conda/jax-0.9.2/bin/python -m pytest \
  tests/field/test_magneticfieldclasses_jax_item15.py -q
# 18 passed
```

## CUDA Status

CUDA proof is `not_claimed`. The user requested CPU JAX only.
