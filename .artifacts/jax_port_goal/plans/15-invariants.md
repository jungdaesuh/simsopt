# Item 15 — Math And Physics Invariants

## Units and scales

| Field | Units | Scale convention |
| --- | --- | --- |
| ``ToroidalFieldJAX`` | Tesla | ``B = B0 * R0 / R`` along ``e_phi``; ``R = sqrt(x^2 + y^2)`` |
| ``PoloidalFieldJAX`` | Tesla | ``B = (B0 / (R0 * q)) * r`` along ``e_theta``; ``r = sqrt((R - R0)^2 + z^2)`` |
| ``MirrorModelJAX`` | Tesla | ``B_R`` and ``B_Z`` derived from the WHAM double-Lorentzian flux function ``psi(R, Z)``; SI parameters (B0 in Tesla, gamma and Z_m in metres) |
| ``DommaschkJAX`` | Tesla | Cartesian field per mode + ``ToroidalField(R0=1, B0=1)`` baseline (matches upstream CPU ``Dommaschk``) |
| ``ReimanJAX`` | Dimensionless model | ``B_R``, ``B_Z`` derived from the island-model series; ``B_phi = -1`` is constant by construction |

Points ``x = (x_p_0, x_p_1, x_p_2)`` are Cartesian metres in all cases.

## Sign and orientation conventions

- All wrappers preserve the upstream CPU class's sign and orientation.
  The CPU class is the parity oracle. Any deviation would be caught
  by the ``direct_kernel`` lane parity test on the production
  fixture.
- ``ToroidalField``: ``e_phi`` is the right-handed cylindrical
  azimuthal unit vector; ``B0`` and ``R0`` are positive by convention.
- ``PoloidalField``: ``e_theta`` follows the
  ``(theta, phi)`` poloidal-toroidal convention with ``theta``
  measured from the magnetic axis ``R = R0``.
- ``Dommaschk``: ``Phi`` is the scalar potential and ``B = grad Phi``;
  the wrapper folds in the ``ToroidalField(R0=1, B0=1)`` baseline so
  the published-paper reference values match.

## Stellarator symmetry coverage

None of the five analytic-field wrappers participate in the
``stellsym=True`` / ``stellsym=False`` switch directly — they expose a
single closed-form field over Cartesian ``(x, y, z)`` and do not own
a discrete-symmetry contract.

``InterpolatedField`` (the deferred sub-item) IS where
``stellsym=True`` / ``stellsym=False`` matter: the public wrapper
folds ``z`` through the stellarator-symmetric map when constructing
the cylindrical interpolant table. That coverage is part of the
unblocked work tracked in
``.artifacts/jax_port_goal/blockers/15-interpolatedfield-debug.md``.

## Derivative shape conventions

| Method | Shape | Index convention |
| --- | --- | --- |
| ``B()`` | ``(N, 3)`` | ``B[p, l] = B_l(x_p)`` |
| ``dB_by_dX()`` | ``(N, 3, 3)`` | ``dB[p, j, l] = d B_l(x_p) / d x_j`` (matches upstream simsopt ``fields.rst`` convention; ``j`` is the derivative direction, ``l`` is the field component) |
| ``A()`` | ``(N, 3)`` | ``A[p, l] = A_l(x_p)`` |
| ``dA_by_dX()`` | ``(N, 3, 3)`` | ``dA[p, j, l] = d A_l(x_p) / d x_j`` |
| ``d2B_by_dXdX()`` | ``(N, 3, 3, 3)`` | ``ddB[p, j, k, l]`` matches the literal CPU storage in upstream ``_d2B_by_dXdX_impl`` (see the item 12 invariants note on the upstream typo in ``ToroidalField._d2B_by_dXdX_impl``) |

## Excluded / singular regimes

- ``PoloidalField`` is singular at ``R = R0`` (magnetic axis). The
  CPU class returns ``NaN`` on the axis ring; the JAX kernel matches.
  The new parity test filters points away from ``|R - R0| > 0.2`` m
  before asserting parity (``test_B_dB_parity_vs_cpu``).
- ``MirrorModel`` is singular at ``R = sqrt(x^2 + y^2) = 0`` (axis).
  Same filtering convention; the parity test keeps points with
  ``R > 0.2`` m.
- ``Reiman`` is singular at the axis ring ``R = R_axis = 1`` (the
  Cartesian ``r_min`` argument is zero). The parity test filters
  points away from ``|R - 1| > 0.1``.
- ``Dommaschk`` requires ``R > 0`` (otherwise ``log(R)`` and
  ``B_phi / R`` diverge); the production fixture samples Cartesian
  ``(x, y)`` from ``[0.4, 1.8]`` so ``R`` is bounded well above zero.
- The wrapper does NOT add defensive guards beyond what the upstream
  CPU class provides; behaviour on the singular ring is consistent
  between the CPU oracle and the JAX wrapper.

## Oracle contract per wrapper

All five wrappers are tested under the **fixed-state scalar / fixed
gradient** oracle contract (``oracle_contract = fixed_state_scalar``
+ ``fixed_gradient_vjp`` per the goal-prompt taxonomy). The new
parity tests assert byte-equality (up to ``direct_kernel`` lane
tolerance) of the wrapper outputs against the CPU oracle at a
fixed set of evaluation points; no optimization-trajectory or
optimizer-envelope contract is in scope.

## Tolerance lane

``direct_kernel`` (``rtol = 1e-10``, ``atol = 1e-12``,
``requires_same_state = True``, ``requires_direct_cpp_oracle = True``,
``vector_parity_required = True``). All wrappers are exercised at
production-scale fixtures (60-point Cartesian point clouds, or
120-point clouds filtered down to ``>= 50`` points away from the
singular regimes documented above).

## Negative-control / red-step evidence

``.artifacts/jax_port_goal/red/15.txt`` records that the new
parity tests cannot be collected at the parent commit ``d79a869fd``
(the wrapper module and tests do not exist there). After
implementation, the same parity assertions pass at the production
scale, demonstrating that the new test does not ride a previously
green lane: it asserts an invariant (byte-equal B/dB/A/dA/d2B
between the new JAX wrapper public API and the CPU oracle) that did
not hold before because the wrapper API did not exist.

The transfer-guard discipline test, in particular, asserts an
invariant that none of the pre-existing tests required: that the
new wrappers' specs and ``_points_device`` staging are clean under
:func:`jax.transfer_guard("disallow")`. The kernel-level item 11
test did not exercise this (it had no transfer-guard test), so the
new test closes a specific edge.

## Serialization / restart contract

All five wrappers implement ``as_dict`` and ``from_dict`` mirroring
the upstream CPU classes' SIMSON serializer. The new parity test
``test_as_from_dict_roundtrip_preserves_class`` (one per wrapper)
asserts:

1. ``json.dumps(SIMSON(field), cls=GSONEncoder)`` succeeds.
2. ``json.loads(payload, cls=GSONDecoder)`` returns an instance of
   the same JAX wrapper class.
3. The re-loaded instance evaluates ``B()`` to the same value as the
   original at the ``direct_kernel`` tolerance.

Restart compatibility evidence is therefore covered inline by the
parity test module. The dedicated restart artifact is
``.artifacts/jax_port_goal/restart/15.md``.
