# Gold Suite Specification

## Boundary Optimization
- **Case:** Ginsburg-15 fixed-boundary Quasisymmetry (QA) optimization.
- **Metric:** Quasisymmetry error $\chi^2_{QS}$.
- **Resolution:** `nphi=255`, `ntheta=64`, `mpol=8`, `ntor=6`.

## Equilibrium Solve
- **Case:** Free-boundary equilibrium solution for the Ginsburg-15 coil set.
- **Metric:** Force balance residual $|f_\rho|$.
- **Target Volume:** `0.10`.
- **Target Iota:** `0.15`.

## Coil Optimization
- **Case:** Single-stage ReBCO coil optimization.
- **Constraints:**
  - Binormal curvature: $\kappa_b < 0.04 \text{ cm}^{-1}$
  - Torsion: $\tau$ limit
  - Coil-to-coil distance: $d > 0.05 \text{ m}$
- **Metric:** Pareto front distance comparing QA vs Maximum Strain.

## Evaluation Interface
```python
def evaluate_optimizer(optimizer_cls, budget_constraint):
    """
    Evaluates an optimizer on the Gold Suite under the given budget constraint.
    """
    pass
```