"""
JAX optimizer adapter for the Boozer inner solve.

Per M0 contract §4:
  - BFGS is the default (stable, matches current SciPy inner solve).
  - L-BFGS-B available via ``method="lbfgs"``.
  - The adapter preserves SIMSOPT's flat-DOF contract.

Device residency note (plan §3):
  The BFGS optimizer loop runs via SciPy on the host, with JAX-compiled
  objective/gradient evaluation on-device.  ``jax.scipy.optimize.minimize``
  was evaluated as a fully on-device alternative but its simplified line
  search (backtracking Armijo) fails to converge on the non-convex Boozer
  penalty landscape from cold starts.  SciPy's Moré-Thuente line search
  is required for reliable convergence.  This is accepted as a v1 fallback
  per plan §3 ("unless explicitly accepted as a v1 fallback").
  Newton polish and Newton exact use JAX arrays for all linear algebra;
  only scalar convergence checks read a single float from device.
"""

import numpy as np

import jax
import jax.numpy as jnp
from scipy.optimize import minimize as scipy_minimize

__all__ = ["jax_minimize", "newton_polish", "newton_exact"]


def jax_minimize(fun, x0, *, method="bfgs", tol=1e-10, maxiter=1500, options=None):
    """Optimizer adapter: SciPy BFGS with JAX value_and_grad.

    The objective ``fun(x)`` is JIT-compiled and differentiated via
    ``jax.value_and_grad``.  The optimizer loop itself runs via SciPy,
    which provides a robust Moré-Thuente line search needed for the
    non-convex Boozer penalty landscape.

    Args:
        fun: Callable ``(x) -> scalar``.  Must be JAX-traceable.
        x0: (n,) ``jax.Array`` of initial DOFs.
        method: ``"bfgs"`` (default) or ``"lbfgs"``.
        tol: Gradient tolerance for convergence.
        maxiter: Maximum number of iterations.
        options: Dict of method-specific options passed to SciPy.

    Returns:
        ``scipy.optimize.OptimizeResult`` with ``.x`` (as jax.Array),
        ``.fun``, ``.jac``, ``.nit``, ``.success``.
    """
    if options is None:
        options = {}

    val_and_grad_fn = jax.jit(jax.value_and_grad(fun))

    def scipy_fun(x_np):
        x_jax = jnp.asarray(x_np)
        val, grad = val_and_grad_fn(x_jax)
        return float(val), np.asarray(grad)

    if method == "bfgs":
        scipy_method = "BFGS"
        scipy_opts = {"maxiter": maxiter, "gtol": tol, **options}
    elif method == "lbfgs":
        scipy_method = "L-BFGS-B"
        scipy_opts = {"maxiter": maxiter, "gtol": tol, "maxcor": 200, **options}
    else:
        raise ValueError(f"Unknown method {method!r}. Supported: 'bfgs', 'lbfgs'.")

    result = scipy_minimize(
        scipy_fun,
        np.asarray(x0),
        jac=True,
        method=scipy_method,
        options=scipy_opts,
    )

    # Convert result arrays back to JAX
    result.x = jnp.asarray(result.x)
    result.jac = jnp.asarray(result.jac)
    result.nit = int(result.nit)
    return result


def newton_polish(
    objective_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-11,
    stab=0.0,
):
    """Newton polish using JAX Hessian + ``jnp.linalg.solve``.

    Runs a Python-level Newton loop with JIT-compiled steps.
    Each iteration computes value, gradient, and Hessian of the objective
    and solves the Newton system ``H dx = g``.

    All linear algebra stays on JAX arrays; only the scalar convergence
    check ``float(norm)`` touches the host.

    Args:
        objective_fn: Callable ``(x) -> scalar``, JAX-traceable.
        x0: (n,) initial DOFs (from prior BFGS).
        maxiter: Maximum Newton iterations.
        tol: Gradient norm convergence threshold.
        stab: Tikhonov stabilization added to diagonal of Hessian.

    Returns:
        dict with keys: ``x``, ``fun``, ``grad``, ``hessian``, ``nit``,
        ``success``.
    """
    val_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))
    hessian_fn = jax.jit(jax.hessian(objective_fn))

    x = x0
    val, grad = val_and_grad_fn(x)
    norm = jnp.linalg.norm(grad)
    H = hessian_fn(x)

    nit = 0
    while nit < maxiter and float(norm) > tol:
        if stab > 0:
            H = H + stab * jnp.eye(H.shape[0])
        dx = jnp.linalg.solve(H, grad)
        # Iterative refinement (Wilkinson) when close to solution
        if float(norm) < 1e-9:
            dx = dx + jnp.linalg.solve(H, grad - H @ dx)
        x = x - dx
        val, grad = val_and_grad_fn(x)
        norm = jnp.linalg.norm(grad)
        H = hessian_fn(x)
        nit += 1

    return {
        "x": x,
        "fun": val,
        "grad": grad,
        "hessian": H,
        "nit": nit,
        "success": bool(float(norm) <= tol),
    }


def newton_exact(
    residual_fn,
    x0,
    *,
    maxiter=40,
    tol=1e-13,
):
    """Newton solver for the exact Boozer residual system r(x) = 0.

    Computes Jacobian via ``jax.jacfwd`` and solves ``J dx = r``
    at each step with iterative refinement.

    All linear algebra stays on JAX arrays; only the scalar convergence
    check ``float(norm)`` touches the host.

    Args:
        residual_fn: Callable ``(x) -> (n_eq,)`` residual vector.
        x0: (n,) initial guess.
        maxiter: Maximum Newton iterations.
        tol: Residual norm convergence threshold.

    Returns:
        dict with keys: ``x``, ``residual``, ``jacobian``, ``nit``,
        ``success``.
    """
    jac_fn = jax.jit(jax.jacfwd(residual_fn))
    res_fn = jax.jit(residual_fn)

    x = x0
    r = res_fn(x)
    J = jac_fn(x)
    norm = jnp.linalg.norm(r)

    nit = 0
    while nit < maxiter and float(norm) > tol:
        dx = jnp.linalg.solve(J, r)
        # Unconditional iterative refinement: exact systems are more
        # sensitive to conditioning than the LS penalty Hessian, so
        # refinement is always applied (unlike newton_polish which
        # guards on norm < 1e-9).
        dx = dx + jnp.linalg.solve(J, r - J @ dx)
        x = x - dx
        r = res_fn(x)
        J = jac_fn(x)
        norm = jnp.linalg.norm(r)
        nit += 1

    return {
        "x": x,
        "residual": r,
        "jacobian": J,
        "nit": nit,
        "success": bool(float(norm) <= tol),
    }
