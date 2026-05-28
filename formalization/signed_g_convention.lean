import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Signed Boozer G convention

This file formalizes the arithmetic contract behind the signed Boozer `G` seed.
It does not formalize the Boozer residual, Biot-Savart geometry, coil
partitioning, or a Newton/least-squares solver.

If `mu0 > 0`, a negative signed TF current sum gives a negative signed
`G = mu0 * signedCurrentSum`.  Any absolute-value or sign-blind magnitude
`mu0 * positiveMagnitude` is positive, so the two values cannot be equal.
-/

namespace SimsoptSurrogate

/-- Signed TF-only Boozer `G` seed, abstracting `mu0 * sum_signed(I_TF)`. -/
def signedG (μ signedCurrentSum : ℝ) : ℝ :=
  μ * signedCurrentSum

/-- Sign-blind positive-magnitude Boozer `G` seed. -/
def magnitudeG (μ positiveMagnitude : ℝ) : ℝ :=
  μ * positiveMagnitude

/-- Sign-blind Boozer `G` seed, abstracting `mu0 * abs(sum_signed(I_TF))`. -/
def absG (μ signedCurrentSum : ℝ) : ℝ :=
  μ * |signedCurrentSum|

/-- If `μ > 0` and the signed TF-current sum is negative, `signedG` is negative. -/
theorem signedG_neg_of_pos_of_neg {μ signedCurrentSum : ℝ}
    (hμ : 0 < μ) (hsigned : signedCurrentSum < 0) :
    signedG μ signedCurrentSum < 0 := by
  unfold signedG
  exact mul_neg_of_pos_of_neg hμ hsigned

/-- If `μ > 0` and the magnitude is positive, `magnitudeG` is positive. -/
theorem magnitudeG_pos_of_pos_of_pos {μ positiveMagnitude : ℝ}
    (hμ : 0 < μ) (hmagnitude : 0 < positiveMagnitude) :
    0 < magnitudeG μ positiveMagnitude := by
  unfold magnitudeG
  exact mul_pos hμ hmagnitude

/-- Direct sign contradiction: a negative value cannot equal a positive value. -/
theorem negative_ne_positive {negativeG positiveG : ℝ}
    (hnegative : negativeG < 0) (hpositive : 0 < positiveG) :
    negativeG ≠ positiveG := by
  intro h
  linarith

/-- General contract theorem for any positive sign-blind current magnitude. -/
theorem signedG_ne_positive_magnitudeG
    {μ signedCurrentSum positiveMagnitude : ℝ}
    (hμ : 0 < μ)
    (hsigned : signedCurrentSum < 0)
    (hmagnitude : 0 < positiveMagnitude) :
    signedG μ signedCurrentSum ≠
      magnitudeG μ positiveMagnitude := by
  exact negative_ne_positive
    (signedG_neg_of_pos_of_neg hμ hsigned)
    (magnitudeG_pos_of_pos_of_pos hμ hmagnitude)

/-- If `μ > 0` and the signed TF-current sum is negative, `absG` is positive. -/
theorem absG_pos_of_pos_of_neg {μ signedCurrentSum : ℝ}
    (hμ : 0 < μ) (hsigned : signedCurrentSum < 0) :
    0 < absG μ signedCurrentSum := by
  unfold absG
  exact mul_pos hμ (abs_pos.mpr (ne_of_lt hsigned))

theorem signedG_ne_absG_of_negative_current_sum
    {μ signedCurrentSum : ℝ}
    (hμ : 0 < μ)
    (hsigned : signedCurrentSum < 0) :
    signedG μ signedCurrentSum ≠
      absG μ signedCurrentSum := by
  exact negative_ne_positive
    (signedG_neg_of_pos_of_neg hμ hsigned)
    (absG_pos_of_pos_of_neg hμ hsigned)

theorem opposite_sign_G_values_are_not_equal
    {signedG absG : ℝ}
    (hsignedG : signedG < 0)
    (habsG : 0 < absG) :
    signedG ≠ absG := by
  exact negative_ne_positive hsignedG habsG

end SimsoptSurrogate
