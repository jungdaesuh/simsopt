# Physics, Math, and Computational Validation Report

This report validates the proposed improvements for the single-stage stellarator optimization routine, focusing on physics consistency, mathematical correctness, and computational feasibility within the `simsopt` framework.

## 1. Current-Only Polish Block
**Verdict: Highly Valid and Physically Sound**

*   **Physics & Math:** The Biot-Savart law dictates that the magnetic field $\mathbf{B}$ is strictly linear with respect to the coil currents $I_j$:
    $$ \mathbf{B}(\mathbf{x}; \mathbf{g}, \mathbf{I}) = \sum_j I_j \mathbf{B}_j(\mathbf{x}; \mathbf{g}_j) $$
    where $\mathbf{B}_j$ is the field produced by the $j$-th coil with unit current. Because the dominant objective (e.g., quadratic flux $\int (\mathbf{B}\cdot\mathbf{n})^2 dA$) is quadratic in $\mathbf{B}$, it is strictly quadratic in $\mathbf{I}$ when the geometry $\mathbf{g}$ is frozen.
*   **Computation:** By freezing $\mathbf{g}$, the highly non-linear, non-convex optimization of coil shapes is temporarily bypassed. Solving for $\mathbf{I}$ becomes a convex Quadratic Programming (QP) problem (or linear least-squares if unconstrained). This guarantees a global minimum for the given geometry.
*   **Validation:** This approach perfectly mirrors established codes like `REGCOIL`, which solve a linear system for surface currents. Applying it to discrete coils will absolutely eliminate "false rejections" where a good geometry is discarded simply because the coupled optimizer took a poor step in current space.

## 2. Optimize Current Modes (Fourier Basis)
**Verdict: Valid, but Requires Careful Symmetry Handling**

*   **Physics & Math:** Decomposing $I_j$ into a Fourier basis along the toroidal angle is a powerful regularization technique.
    $$ I_j = I_0 + \sum_m \left[ a_m \cos\left(\frac{2\pi m j}{N}\right) + b_m \sin\left(\frac{2\pi m j}{N}\right) \right] $$
    Low $m$ modes naturally control long-wavelength features (rotational transform $\iota$, gross magnetic well, global error fields), while high $m$ modes generate localized modular ripple.
*   **Computation:** This acts as a physics-informed preconditioner. Optimizing $(a_m, b_m)$ instead of raw $I_j$ structurally prevents the optimizer from rapidly oscillating adjacent coil currents in unphysical ways (which often happens in gradient descent on raw currents).
*   **Validation:** You must ensure this basis respects stellarator symmetry and the number of field periods ($N_{FP}$). Typically, coils are grouped into symmetry classes (e.g., 5 independent coils per half-period). The Fourier modes should be defined over these symmetry classes, not the raw $1..N$ index of all coils in the machine, to preserve the inherent modular periodicity.

## 3. Current-Only Gauss-Newton / QP Predictor
**Verdict: Mathematically Optimal for Currents**

*   **Physics & Math:** As established in Point 1, the objective is approximately (or exactly) quadratic in $\delta \mathbf{I}$. A Gauss-Newton step:
    $$ \delta \mathbf{I} = -(J_I^T J_I + \lambda D^T D)^{-1} J_I^T \mathbf{r} $$
    (where $J_I$ is the Jacobian with respect to currents and $\mathbf{r}$ is the residual vector) is the mathematically exact minimizer for the unconstrained quadratic flux.
*   **Computation:** `simsopt` already computes the Jacobian of the objective with respect to all DOFs. Partitioning this Jacobian to extract $J_I$ is computationally cheap. Solving the dense $10 \times 10$ (or similar small dimension) linear system takes negligible time compared to a single Biot-Savart evaluation. Adding bound constraints ($\leq 16$ kA) simply turns this into a very small, ultra-fast QP solve.
*   **Validation:** This is heavily supported by `QUADCOIL` and `RCLS` methodologies. It provides a "perfect" current update in a single step, rather than waiting for L-BFGS-B to build up an approximate Hessian over many iterations.

## 4. Split Line Search
**Verdict: Computationally Critical for Coupled Problems**

*   **Physics & Math:** Geometry updates $\delta \mathbf{g}$ and current updates $\delta \mathbf{I}$ have drastically different physical effects and characteristic scales. Geometry changes alter the Biot-Savart kernel non-linearly and are heavily restricted by physical collisions (coil-coil distance) and curvature penalties. Current changes are linear and only restricted by engineering bounds (e.g., max current density).
*   **Computation:** Standard L-BFGS-B uses a single step length $\alpha$ for the concatenated vector $[\mathbf{g}, \mathbf{I}]$. If a geometry step violates a collision penalty, the line search cuts $\alpha$ for *both*. Splitting the line search:
    1. Try full step $(\mathbf{g} + \alpha \delta \mathbf{g}, \mathbf{I} + \alpha \delta \mathbf{I})$.
    2. If rejected due to geometry constraints, decouple and try $(\mathbf{g} + \beta \delta \mathbf{g}, \mathbf{I} + \alpha \delta \mathbf{I})$ with $\beta < \alpha$.
*   **Validation:** This directly addresses the pathology where the optimizer stagnates because beneficial current updates are repeatedly vetoed by aggressive geometry steps. It allows the currents to continuously relax to their optimal state even when geometry is tight against constraint boundaries.

## Conclusion and Recommendations
The proposed roadmap is highly robust and grounded in established stellarator optimization theory.

**Recommended Implementation Order:**
1. **Current-Only Polish:** Easiest to implement in `simsopt`. Simply freeze geometry DOFs (`coil.fix_all()`) and run a fast bounded optimization on currents (`coil.unfix_current()`).
2. **Gauss-Newton/QP Predictor:** Can be integrated into the polish step. Instead of SciPy's L-BFGS-B for the polish, use a direct least-squares/QP solver (like `scipy.optimize.lsq_linear` or `cvxpy`) using the existing `simsopt` Jacobian.
3. **Split Line Search:** Harder to implement as it requires writing a custom optimizer loop or heavily wrapping the objective function to intercept the line search logic of standard SciPy optimizers.
4. **Fourier Basis:** Requires mapping the raw current DOFs to a Fourier matrix. Very effective, but the mapping must rigorously respect the $N_{FP}$ and stellarator symmetries of the specific device.
