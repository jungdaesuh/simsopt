Verification-only Aristotle request.

Use the existing Lean project in this directory.  Do not create
`SimsoptSurrogate.lean` or any other new Lean file.  The only Lean source file
to inspect or repair is:

```text
signed_g_convention.lean
```

Run the explicit Lake target:

```text
lake build signed_g_convention
```

The required theorem boundary is arithmetic-only:

1. `signedG (mu0 signedCurrentSum : ℝ) : ℝ := mu0 * signedCurrentSum`
2. `magnitudeG (mu0 positiveMagnitude : ℝ) : ℝ := mu0 * positiveMagnitude`
3. `absG (mu0 signedCurrentSum : ℝ) : ℝ := mu0 * |signedCurrentSum|`
4. If `mu0 > 0`, `signedCurrentSum < 0`, and `positiveMagnitude > 0`, prove
   `signedG mu0 signedCurrentSum ≠ magnitudeG mu0 positiveMagnitude`.
5. If `mu0 > 0` and `signedCurrentSum < 0`, prove
   `signedG mu0 signedCurrentSum ≠ absG mu0 signedCurrentSum`.
6. If `signedG < 0` and `0 < absG`, prove `signedG ≠ absG`.

Verify the file contains no `sorry`, no `admit`, and no custom axioms.  It is
acceptable for Lean/mathlib to report the standard foundational axioms used by
real-number arithmetic, such as `propext`, `Classical.choice`, or `Quot.sound`.

Do not prove or claim that every nonlinear Boozer solve is impossible.  That
would require formalizing the Boozer residual, surface geometry, and numerical
solver.  This request is only about the signed-current arithmetic contract.
