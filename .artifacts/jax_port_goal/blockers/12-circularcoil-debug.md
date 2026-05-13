# Item 12-sub — CircularCoil JAX port blocker

`category: missing_dependency`
`closure_level: blocked_dependency`
`needs_user: false` (resolution path is implementable without a new
runtime dependency; sub-item is parked until a future run promotes it.)

## Summary

The JAX port of `simsopt.field.magneticfieldclasses.CircularCoil` is
deferred as a sub-item of P1 item 12. The CPU class evaluates the off-axis
circular-coil B-field via complete elliptic integrals
`K(k^2)`, `E(k^2)`. Neither symbol is exposed in `jax.scipy.special` at
the runtime version pinned for this repo (`jax==0.10.0`,
`jaxlib==0.10.0`).

## Specific missing API

Import probe on the repo-local interpreter
`/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python`:

```bash
$ JAX_ENABLE_X64=True JAX_PLATFORMS=cpu \
    /Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python \
    -c "from jax.scipy.special import ellipk, ellipe; print('ok')"
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ImportError: cannot import name 'ellipk' from 'jax.scipy.special'
  (/Users/suhjungdae/.local/lib/python3.11/site-packages/jax/scipy/special.py)
```

`scipy.special.ellipk` and `scipy.special.ellipe` are available in
NumPy/SciPy as host functions, but cannot be traced inside a `jit` /
`vmap` compiled JAX kernel without exiting the trace.

## Why CircularCoil specifically depends on these

The CPU `_B_impl` and `_dB_by_dX_impl` in
`src/simsopt/field/magneticfieldclasses.py` (lines 281-499) use the
classical Jackson Eq. 5.37 expression for the off-axis B-field of a
circular current loop. Both `K(k^2)` and `E(k^2)` appear in the
analytic expression, where `k^2 = 4 R r / ((R + r)^2 + Z^2)` with `R`
the field-point cylindrical radius, `r` the coil radius, and `Z` the
axial offset. The first derivatives of these integrals enter the
`dB_by_dX` expression as well. There is no rational simplification that
removes the elliptic-integral dependency on general 3-D field points.

## Why a runtime dependency is not the right answer

A drop-in `mpmath` / `scipy.special` host-callback would force every
compiled JAX path that consumes `CircularCoil` to leave the device, run
on the host, then re-enter the device for the rest of the trace. That
violates the `jax.transfer_guard("disallow")` contract and the
"no silent host fallback" architecture invariant in section 2 of the
goal prompt.

## Proposed resolution path

Implement a JAX-native helper for the complete elliptic integrals
inside `src/simsopt/jax_core/`:

- **Option A (recommended)**: Carlson symmetric-form `R_F`, `R_D`, with
  Carlson's iterative reduction (Numerical Recipes 6.11). Both `K(k^2)`
  and `E(k^2)` reduce to short combinations of `R_F` / `R_D`. The
  iteration is a fixed-iteration `jax.lax.scan` and is fully traceable.
- **Option B**: Bulirsch `cel` (`Numerische Mathematik 7 (1965) 78`).
  Same shape; slightly fewer ops per iteration but trickier convergence.

Either helper would live as a private module (e.g.
`src/simsopt/jax_core/_elliptic.py`) with its own parity test against
`scipy.special.ellipk` / `ellipe` over a dense `k^2 ∈ (0, 1 - eps)`
grid at the `direct_kernel` lane tolerance. After the helper exists,
`CircularCoil` becomes a medium-effort JAX port: one B kernel, one dB
kernel (autodiff via `jacfwd` of the B kernel, or closed-form),
parity tests against the upstream CPU class.

## Sub-item recording

In `state.json` for item 12, record the sub-item under the
`evidence.partial_completion` key:

```json
"partial_completion": {
  "deferred_sub_item": "CircularCoil",
  "blocker": ".artifacts/jax_port_goal/blockers/12-circularcoil-debug.md",
  "category": "missing_dependency",
  "missing_api": [
    "jax.scipy.special.ellipk",
    "jax.scipy.special.ellipe"
  ],
  "proposed_resolution": "Implement Carlson R_F/R_D in src/simsopt/jax_core/_elliptic.py with scipy.special parity, then port CircularCoil on top."
}
```

Item 12 itself remains `status=complete` and
`closure_level=cpu_oracle_complete` for `ToroidalField`, `PoloidalField`,
and `MirrorModel` — three of the four analytic fields originally in the
prompt scope.
