# Item 11 Math And Physics Invariants

## Units And Frames

- Both kernels return Cartesian-frame ``B(x)`` in Tesla and the
  Cartesian gradient ``dB/dx`` in T/m using the SIMSOPT axis
  convention ``dB[..., i, j] = d B_j / d x_i``. This matches the
  ``sopp.DommaschkdB`` and ``sopp.ReimandB`` index layout: the
  third axis is the derivative direction and the fourth axis is
  the magnetic-field component (cf. ``dommaschk.cpp`` lines
  519-528 and ``reiman.cpp`` lines 96-104).
- Cylindrical inputs to the C++ scalar-potential helpers are
  ``(R, phi, Z)`` with ``R = sqrt(x^2 + y^2)``, ``phi = atan2(y, x)``,
  ``Z = z``. The JAX kernels reproduce the same conversion in
  pure JAX.

## Domain Constraints

- ``R > 0`` is required at every evaluation point. ``Bphi``
  divides by ``R`` (Dommaschk line 311; Reiman ``BR``/``BZ``
  divides by ``RR`` at ``reiman.cpp`` lines 33-34), and Dommaschk
  uses ``log(R)`` inside the polynomial-term log branches. The
  module docstring documents this invariant.
- For Reiman, additionally ``rmin = sqrt((R - R_axis)^2 + Z^2) >
  0`` whenever ``min(k_theta) <= 2`` (so that ``rmin^(k_theta -
  2)`` does not require a negative power). The upstream
  ``test_Reiman`` fixture uses ``k = [6]`` so the canonical
  reference path is power-positive.
- Mode indices ``m, n, k_theta`` must be Python integers. The
  Dommaschk ``alpha / beta / gamma1`` helpers branch to ``0`` for
  ``l < 0`` (and ``l >= m`` for ``beta``); this avoids feeding
  ``math.gamma`` with non-positive integer arguments at trace
  time.

## Series Convergence

- The Dommaschk sums over ``k`` (``0 .. n // 2``) and ``j``
  (``0 .. k``) are finite for finite ``n`` -- no convergence
  concern; the kernel materializes every term explicitly.
- The Reiman sum over ``k_theta`` indices is finite (length
  ``M = len(k_theta)``).

## Symmetric Gradient (Vacuum Field)

The Dommaschk field is derived from a scalar potential, so the
Cartesian gradient ``dB[..., i, j]`` is symmetric in ``(i, j)``
at every evaluation point and for every mode contribution. This
is verified by ``test_dommaschk_grad_symmetric`` at
``direct_kernel`` tolerance.

## Reiman Constant ``Bphi``

``Bphi(x) = -1`` for all points, and consequently
``dRBphi = dZBphi = dphiBphi = 0``. The Cartesian gradient
contribution from ``Bphi`` therefore vanishes in the
``cylindrical -> Cartesian`` assembly except through the
``cos(phi)``, ``sin(phi)`` rotation factors. The JAX kernel
reproduces this exactly.

## Linearity In Coefficients

- Dommaschk ``BR`` / ``BZ`` / ``Bphi`` are linear in each
  ``(coeff1, coeff2)`` pair per mode (lines 264-313 of the C++
  source). The per-mode JAX kernel preserves this linearity.
- The Dommaschk gradient is similarly linear in
  ``(coeff1, coeff2)`` per mode.

## ``stellsym`` And ``nfp``

Item 11 ports the raw analytic-field kernels. Neither
``stellsym`` nor ``nfp`` are direct inputs -- they enter only at
the public ``Optimizable`` wrapper level (item 15) where
quadrature grids and field-periodicity unfolding are managed.

## Boundary With Wrapper

The Dommaschk public wrapper
``simsopt.field.magneticfieldclasses.Dommaschk`` adds a
``ToroidalField(R0=1, B0=1)`` baseline to the raw kernel sum
(see ``magneticfieldclasses.py`` lines 770-781). Item 11 keeps
the JAX kernel **raw** and lets item 15 own the baseline
addition; ``test_dommaschk_paper_fixtures`` re-adds the
baseline at the test boundary to match the printed wrapper
references.

The Reiman public wrapper has no baseline -- ``sopp.ReimanB``
output is the wrapper output (see ``magneticfieldclasses.py``
lines 825-831).
