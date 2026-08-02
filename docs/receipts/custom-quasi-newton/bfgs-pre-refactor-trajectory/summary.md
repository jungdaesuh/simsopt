# BFGS accepted-state comparison

The pre-refactor solver at `9ba1ad057` and the candidate worktree produced
byte-identical JSON for three FP64 BFGS accepted steps. Final status,
iterations, function/gradient evaluations, objective, gradient, and
parameters all matched.

This is diagnostic pre-refactor evidence only: the candidate worktree was
dirty, and the case is a synthetic CPU Rosenbrock fixture.
