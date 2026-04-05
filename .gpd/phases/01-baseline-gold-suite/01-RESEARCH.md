# Research: Phase 1 — Baseline & Gold Suite

## Mathematical Framework — Incumbents

### 1. DESC `lsq-auglag`
- **Formalism:** Gauss-Newton trust-region Augmented Lagrangian.
- **Objective:** $\min_x \frac{1}{2} \|f(x)\|_2^2$ subject to $c(x) = 0$ and $h(x) \geq 0$.
- **Augmented Lagrangian:** $L(x, \lambda, \mu) = \frac{1}{2}\|f(x)\|_2^2 + \lambda^T c(x) + \frac{\mu}{2}\|c(x)\|_2^2$.
- **Derivatives:** Exact Jacobians via JAX Automatic Differentiation.
- **Scaling:** Excellent for high-dimensional boundary modes.

### 2. SIMSOPT 2025 ALM (ReBCO Strain)
- **Formalism:** Single-stage ALM for engineering-constrained coil optimization.
- **Constraints:**
  - Binormal curvature: $\kappa_b = \frac{|\mathbf{r}' \times \mathbf{r}'' \cdot \mathbf{r}'''|}{|\mathbf{r}' \times \mathbf{r}''|^2}$ (minimized to stay below critical strain $\sim 0.4\%$).
  - Torsion: $\tau = \frac{(\mathbf{r}' \times \mathbf{r}'') \cdot \mathbf{r}'''}{|\mathbf{r}' \times \mathbf{r}''|^2}$.
  - Coil-to-coil distance.
- **Optimization:** Simultaneously optimizes plasma boundary and coil geometry.

## Gold Suite Problem Definition

The Gold Suite must span three dimensions of stellarator optimization:

| Dimension | Problem Case | Metric | Budget Metric |
| --- | --- | --- | --- |
| **Boundary** | Fixed-boundary QS optimization | Quasisymmetry error ($\chi^2_{QS}$) | JAX op count / time |
| **Equilibrium** | Free-boundary equilibrium solve | Force balance residual ($|f_\rho|$) | JAX op count / time |
| **Coil** | Single-stage ReBCO coil design | Pareto front (QS vs Strain) | JAX op count / time |

### Matched-Budget Baseline Contract
- **Budget Allocation:** Total wall-clock time on standardized hardware (e.g., A100 GPU).
- **Secondary Budget:** Total number of objective and gradient evaluations.
- **Parity Guard:** Discovered optimizers must achieve better final objective *within* the same budget as incumbents.

## Literature & Anchors
- [1] "DESC: A JAX-based stellarator equilibrium and optimization suite," Princeton.
- [2] "Strain optimisation for ReBCO high-temperature superconducting stellarator coils in SIMSOPT," JPP 2025.
- [3] "Single-Stage Stellarator Optimization with Automatic Differentiation," SIMSOPT-JAX.

## Known Challenges
- **Scaling:** Ill-conditioning of the Hessian in high-dimensional spaces.
- **Anisotropy:** Sensitivity of the objective to different Fourier modes varies by orders of magnitude.
- **Constraints:** Precise satisfaction of force balance is computationally expensive.
