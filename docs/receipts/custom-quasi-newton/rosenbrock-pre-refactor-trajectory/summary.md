# Rosenbrock accepted-state comparison

The pre-refactor solver at `c0dc94580` and the candidate worktree produced
byte-identical JSON for three FP64 L-BFGS accepted steps. Final status,
iterations, evaluations, objective, gradient, and parameters all matched.

This closes the diagnostic pre-refactor trajectory check only. It is not a
promotion receipt because the candidate worktree was dirty and the case is
synthetic CPU evidence.
