# JAX Best-Practices Baseline (Reviewer Reference)

**Target runtime:** JAX 0.10.0, jaxlib 0.10.0, NumPy 2.x, Python 3.11
**Repository under review:** `/Users/suhjungdae/code/columbia/simsopt-jax`
**Date:** 2026-05-16
**Purpose:** Authoritative rule-set for cross-checking JAX modules in the simsopt-jax port. Every numbered rule below is sourced from a docs page that was actually fetched (URLs cited). Other agents may cite rules by section number (e.g. "violates JBP-3.2").

This document is intentionally normative. It is the contract every JAX module in this worktree must satisfy unless an explicit, file-local waiver exists.

---

## How to read this document

- Sections are numbered `JBP-N` where N is the section number below.
- Each rule has: **Rule**, **Rationale**, **Typical violation**, **How to detect in review**, **Source**.
- "Source" cites the docs.jax.dev URL that was fetched during baseline assembly. If a rule cites a verbatim quote, quotes are kept under 125 characters per source.

---

## JBP-1. Purity

### JBP-1.1 — All inputs must enter via parameters, all outputs must exit via return

- **Rule.** A JAX-traced function must obtain every input through its argument list and return every result via its return value. No reads of module globals, no writes to globals, no `self.cache`-style mutation inside a jitted region.
- **Rationale.** The docs phrase this directly: "all the input data is passed through the function parameters, all the results are output through the function results. A pure function will always return the same result if invoked with the same inputs." Tracing happens once per (shape, dtype, static-arg) signature; subsequent calls hit the cached XLA. Anything that lives in Python scope at trace time is baked in. Anything that runs as a Python side-effect (print, file write, list.append) fires only during the trace, never during cached execution.
- **Typical violation.** A `@jax.jit`'d method that reads `self._cached_thing` from instance state, or appends to a module-level diagnostics list, or relies on `np.random.rand()` for randomness.
- **How to detect.** Grep for `global `, attribute writes inside `@jit`'d functions, calls to `print(...)`, `logging.*`, `open(...)`, `np.random.*`, `time.*`, and any `self.<attr> = ...` inside a `@jit`'d method body. Also flag closures that read mutable Python lists/dicts inside the traced function body.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Pure functions"); reinforced by https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html (section "Pure Functions: No Side Effects, Deterministic, No Mutation").

### JBP-1.2 — No Python-level side effects inside traced code (print, logging, file I/O, time)

- **Rule.** `print`, `logging.*`, `pdb.set_trace`, file I/O, and `time.*` inside a `jit`/`vmap`/`grad`/`scan`/`cond` body are forbidden. Use `jax.debug.print` or `jax.debug.callback` instead.
- **Rationale.** "Subsequent runs with parameters of same type and shape may not show the side-effect. This is because JAX now invokes a cached compilation of the function" (gotchas page). Side effects in traced code execute only on the first trace, never on subsequent cached executions, producing intermittent and confusing behavior that disappears once the cache is hit.
- **Typical violation.** `print("solving...")` inside `_solve_step` that is invoked inside a `jax.lax.scan` body; `logger.info(f"residual={r}")` inside an objective wrapper.
- **How to detect.** Search every `_jax.py` file for `print(`, `logging.`, `logger.`, `warnings.warn(` inside functions that are decorated by `@jax.jit` or passed as bodies to `jax.lax.{cond,scan,while_loop,fori_loop}`.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Pure functions").

### JBP-1.3 — Determinism

- **Rule.** A traced function must yield identical outputs for identical inputs (modulo XLA non-determinism on GPU which is governed at the platform layer). No reliance on Python `id()`, dict iteration order pre-3.7, `time.perf_counter`, `np.random.default_rng()`, or any hidden state.
- **Rationale.** JIT cache assumes determinism — non-deterministic behavior breaks cache reuse, complicates parity testing, and undermines reproducibility of acceptance-lane runs.
- **Typical violation.** Branching on `time.time() % 2`, or seeding NumPy RNG implicitly inside a traced function.
- **How to detect.** Grep for `time.`, `random.`, `np.random.`, `id(` inside JAX modules; verify pseudorandomness goes through `jax.random.split` and an explicit `key` arg.
- **Source.** https://docs.jax.dev/en/latest/random-numbers.html ("Pseudorandom numbers" section); reinforced by https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html (PRNG section).

---

## JBP-2. JIT discipline

### JBP-2.1 — Prefer `static_argnames` over `static_argnums`

- **Rule.** Use `static_argnames=("...","...")` instead of positional `static_argnums=(...)` when marking arguments as static. Static args drive recompilation per unique value, so name them so the contract is greppable.
- **Rationale.** "If we specify `static_argnums`, then the cached code will be used only for the same values of arguments labelled as static. If any of them change, recompilation occurs." (jit-compilation docs). Named markers survive argument reordering and prevent silent miscompilation.
- **Typical violation.** `@partial(jit, static_argnums=(0, 2, 5))` with no comment; reviewer cannot tell which arguments are static without counting.
- **How to detect.** Grep `static_argnums` in JAX modules; prefer `static_argnames`. Verify each static arg's Python value space is small and bounded.
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html (sections on "JIT and caching", "static_argnums").

### JBP-2.2 — Cache-explosion guard: bound the set of static-arg values

- **Rule.** If `static_argnames` includes a value that can take many concrete Python values across one session (e.g. a float threshold, an array shape derived at runtime), the cache will grow without bound. Restrict static args to a small enumerable set (algorithm flags, integer modes).
- **Rationale.** "If there are many values, then your program might spend more time compiling than it would have executing ops one-by-one." (jit docs).
- **Typical violation.** Marking a tolerance `tol` as static, where `tol` is a Python float chosen anew per call. Marking an array shape that is determined by an outer optimizer iteration.
- **How to detect.** For every static arg, ask: "what is the universe of values this can take across one run?" If the answer is more than ~5, it should be dynamic.
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html (caching cost section).

### JBP-2.3 — Define jitted functions at module scope, not in closures

- **Rule.** `jax.jit(f)` should be applied once, at module load. Do not call `jax.jit(...)` inside a hot loop. Do not use `partial(jit, ...)` inside loops where the resulting function identity changes each call.
- **Rationale.** "Don't do this! each time the partial returns a function with different hash." The gotcha is dramatic — measured 1.28 ms vs 366 ms for cached-vs-uncached partials over 20 iterations (jit-compilation docs).
- **Typical violation.** `for _ in range(n_iter): grad_fn = jax.jit(lambda x: ...); g = grad_fn(x)` — recompiles every step.
- **How to detect.** Grep for `jax.jit(` and `jit(...)` inside `def` bodies, especially inside loops. Allow inside `__init__` / one-time setup but not in step functions.
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html.

### JBP-2.4 — Marking class methods as JIT

- **Rule.** A `@jax.jit` decorator cannot be applied to a method directly because `self` is not a JAX array. Three accepted patterns: (a) external helper `_calc(...)` with `static_argnums=(0,)` for the flag; (b) mark `self` static via `static_argnums=(0,)` and provide `__hash__` and `__eq__` on the class; (c) register the class as a pytree via `jax.tree_util.register_pytree_node_class`.
- **Rationale.** The gotchas page enumerates all three strategies. Pytree registration is the most flexible and is the recommended pattern for SIMSOPT-style `Optimizable` adapters that hold arrays and need to compose with `vmap`/`grad`.
- **Typical violation.** `@jax.jit\n    def calc(self, y): ...` (gives `TypeError`).
- **How to detect.** Grep for `@jax.jit` or `@partial(jit, ...)` on `def <name>(self, ...)`; verify one of the three strategies is in place.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Using `jax.jit` with class methods").

### JBP-2.5 — Tracer leaks

- **Rule.** Tracer objects must never escape the jitted region. If a `Tracer` ends up stored on `self`, returned to a non-traced caller, or compared to a Python int, it constitutes a "tracer leak" and indicates a mixing-of-levels bug.
- **Rationale.** "Impure functions are dangerous because under JAX transformations they are likely not to behave as intended; they might fail silently, or produce surprising downstream errors like leaked Tracers." Detect with `jax.check_tracer_leaks()` context.
- **Typical violation.** Caching a JAX array inside an `__init__` while the array is actually a tracer; storing intermediate trace artifacts on the adapter.
- **How to detect.** Run tests under `with jax.check_tracer_leaks(): ...` for adapter classes that hold mutable state.
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html (section on tracer leaks); https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html.

### JBP-2.6 — Concretization errors

- **Rule.** Code that tries to read the concrete Python value of a tracer (e.g. `int(x)`, `bool(x)`, `if x > 0:`) raises `ConcretizationTypeError` or `TracerBoolConversionError`. These errors are not transient — they indicate the function must be rewritten to use `jax.lax.cond` or to mark the offending argument as static.
- **Rationale.** "Traced values within JIT, like x and n here, can only affect control flow via their static attributes: such as shape or dtype, and not via their values." (jit-compilation docs).
- **Typical violation.** `if booz_surf.res["success"]:` inside a jitted helper, where `success` is a JAX scalar. Fix: `jax.lax.cond(success, ..., ...)` or convert to Python bool outside jit.
- **How to detect.** Grep for `bool(`, `int(`, `float(` inside jitted regions; also `if <var>` where `<var>` is the output of a JAX op.
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html (sections "Traced values", "ConcretizationTypeError").

### JBP-2.7 — Boundary scalar conversions

- **Rule.** JAX scalars exiting a traced region into SciPy / NumPy / pure-Python consumers must be converted: `int(result.nit)`, `bool(result.success)`, `float(result.fun)`. Do this at the boundary, not inside the traced body.
- **Rationale.** SciPy's `OptimizeResult` and many NumPy reductions expect Python scalars; storing JAX scalars there can leak tracers if the wrapping function is later traced.
- **Typical violation.** Returning `{"iter": result.nit}` where `result.nit` is a `jnp.int32`.
- **How to detect.** Grep result-dict assembly for JAX-typed scalar values; confirm `int(.)`/`bool(.)`/`float(.)` casts. (See CLAUDE.md "JAX scalar boundary conversions".)
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (out-of-bounds / concretization sections).

---

## JBP-3. Control flow

### JBP-3.1 — Python `if`/`while` on tracers is forbidden

- **Rule.** `if x > 0:` and `while x < tol:` where `x` is a tracer raise `TracerBoolConversionError`. Use `jax.lax.cond` / `jax.lax.while_loop` / `jax.lax.scan` / `jax.lax.fori_loop` instead.
- **Rationale.** "Python's `if`, `else`, and logical operators (`and`, `or`) depend on runtime values and cause `TracerBoolConversionError`" (control-flow docs). The condition evaluates to a tracer (abstract), and Python cannot coerce it to a concrete branch.
- **Typical violation.** Pythonic guard `if jnp.any(jnp.isnan(x)): ...` inside a jitted body.
- **How to detect.** Grep for `if jnp.`, `if jax.`, `while jnp.`, `while jax.` patterns inside jitted code.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "Python Control Flow Under JIT").

### JBP-3.2 — `jax.lax.cond` — both branches must return matching pytree structure

- **Rule.** `jax.lax.cond(pred, true_fun, false_fun, operand)` requires `true_fun(operand)` and `false_fun(operand)` to return values with identical pytree structure, identical leaf shapes, and identical leaf dtypes.
- **Rationale.** "Both branches must return values with identical structure." Mismatched structure produces `TypeError` at trace time.
- **Typical violation.** `true_fun` returns `(x, y)`; `false_fun` returns just `x`. Or `true_fun` returns `jnp.float32` and `false_fun` returns `jnp.float64`.
- **How to detect.** Read both branches; trace-check via a unit test with each branch alone.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "lax.cond").

### JBP-3.3 — `jax.lax.while_loop` — reverse-mode autodiff is NOT supported

- **Rule.** `jax.lax.while_loop` is forward-mode-differentiable only. If you need a gradient through a loop with a data-dependent termination condition, you must use `jax.lax.scan` (with a fixed iteration count + early-stop via masking), or use `jax.custom_vjp` to implement the gradient via the implicit-function theorem.
- **Rationale.** "Critical limitation: fwd-mode-differentiable only — reverse-mode autodiff is not supported through while_loop. Use scan if you need gradient computation." (control-flow docs).
- **Typical violation.** Implementing a Newton solver inside `jax.lax.while_loop` and then trying to `grad` outer code that depends on the solver output without a `custom_vjp` adjoint.
- **How to detect.** For every `jax.lax.while_loop`, locate the smallest enclosing `grad`/`jacrev`/`vjp` and verify either (a) a `custom_vjp` wraps the loop, or (b) the loop is downstream of `jax.lax.stop_gradient`.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "lax.while_loop").

### JBP-3.4 — `jax.lax.fori_loop` — reverse-mode only for static bounds

- **Rule.** `jax.lax.fori_loop(start, stop, body, init_val)` supports reverse-mode autodiff only when `start` and `stop` are static integers. If either is a traced value, fori_loop lowers to `while_loop` internally and reverse-mode breaks.
- **Rationale.** "Supports reverse-mode autodiff only when loop bounds are static constants. If bounds are traced values, the implementation falls back to `while_loop`, losing reverse-mode support." (control-flow docs).
- **Typical violation.** `jax.lax.fori_loop(0, n_iter, body, x0)` where `n_iter` is a tracer; outer `grad` then silently produces NaNs or errors.
- **How to detect.** Inspect call sites of `fori_loop`; confirm `start` and `stop` are Python ints / static.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "lax.fori_loop").

### JBP-3.5 — Prefer `jax.lax.scan` for loops with carry and gradients

- **Rule.** When you need a carry-state loop with reverse-mode autodiff (e.g. accumulating a residual through iterations), use `jax.lax.scan(body, init, xs)`. It is the only primitive that supports reverse-mode through a structured loop natively.
- **Rationale.** "Preferred for loops with carry state when gradients are needed. Differentiable in both forward and reverse modes." (control-flow docs).
- **Typical violation.** Implementing an unrolled Python loop inside a `grad`-target objective; either compiles slowly or doesn't compile at all when iteration count is large.
- **How to detect.** Grep for `for i in range(N):` inside `@jit` regions where `N > ~16`; suggest `scan` replacement.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "lax.scan").

### JBP-3.6 — Logical operators: `&` / `|` / `~` and `jnp.logical_*`, never `and`/`or`/`not`

- **Rule.** Inside a traced region, combine boolean arrays with element-wise operators: `(a > 0) & (b < 1)`, `jnp.logical_and(a, b)`, `jnp.where(c, x, y)`. Python `and`/`or`/`not` short-circuit on truthiness and fail on non-scalar arrays.
- **Rationale.** "Python's `and`/`or` operators short-circuit and fail under JIT on non-scalar arrays." (control-flow docs).
- **Typical violation.** `if (x > 0) and (y < 1): ...` where x, y are JAX arrays.
- **How to detect.** Grep ` and `, ` or `, ` not ` inside JAX module bodies; flag any use on JAX-typed operands.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "Logical Operators").

### JBP-3.7 — Fixed-length Python loops compile fine

- **Rule.** A Python `for i in range(N):` with constant `N` known at trace time compiles correctly — it unrolls. This is acceptable for small `N` (≤ ~16) where unrolling produces readable jaxprs and small XLA programs. For large `N`, prefer `scan` / `fori_loop`.
- **Rationale.** "Fixed-length loops: `for i in range(3)` compiles because the iteration count is known at trace time" (control-flow docs).
- **How to detect.** Measure XLA compile time and program size; unrolled loops can blow up compile time non-linearly.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "What Works").

---

## JBP-4. Closures and tracer leaks

### JBP-4.1 — Closing over a JAX array freezes that array's identity

- **Rule.** A jitted function that closes over an array `A` (e.g. captures it via lexical scope rather than passing it as an argument) treats `A` as a compile-time constant. If `A` is mutated externally (e.g. via `field.set_points(new_pts)`), the jitted function continues to use the original `A`.
- **Rationale.** Tracing serializes the captured array into the XLA HLO. Subsequent calls to the cached function reuse the baked value.
- **Typical violation.** `SquaredFluxJAX` capturing `gamma`, `normal`, `target` arrays at construction and the caller later does `surface.set_points(new_pts)` — the JAX wrapper still uses the old gamma. (This is exactly the SIMSOPT-JAX policy: "Do not call `field.set_points()` after constructing `SquaredFluxJAX`.")
- **How to detect.** For each jitted closure, list the arrays it captures (closure variables). Confirm via tests that the captured arrays are not externally mutated, or rebuild the closure after mutation.
- **Source.** Inferred from JIT caching semantics (https://docs.jax.dev/en/latest/jit-compilation.html section "JIT and caching") and codified in this repo's CLAUDE.md ("JIT closure strategy").

### JBP-4.2 — Closing over Python scalars inlines them

- **Rule.** Closing over a Python `int`/`float`/`bool` inlines the value as an XLA constant. Changing the Python value later does NOT update the compiled function — a new trace is required.
- **Rationale.** Same as above: tracing freezes Python-side constants.
- **Typical violation.** `def make_obj(tol): @jit\n  def f(x): return jnp.where(x > tol, ...);  return f` — closing over `tol`. Later calling `f` with the original closure does not see the new `tol`. Fix: pass `tol` as a (possibly static) argument.
- **How to detect.** Closure variables that are Python scalars should be either (a) constant for the lifetime of the program, (b) passed as static_argnames args, or (c) passed as regular dynamic args.

### JBP-4.3 — Tracer escape

- **Rule.** Storing a tracer on `self`, returning a tracer from a non-traced function, or threading a tracer through Python collections (lists/dicts/sets) breaks composition. Use `jax.check_tracer_leaks` in tests.
- **Rationale.** "errors like leaked Tracers" (jit-compilation docs).
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html.

---

## JBP-5. PyTrees

### JBP-5.1 — A pytree is any nested container of arrays/leaves; built-ins are list/tuple/dict/None

- **Rule.** JAX treats `list`, `tuple`, `dict`, and `None` as internal nodes; anything else (numbers, arrays, strings, custom classes not registered) is a leaf. `None` is a node with zero children, not a leaf, unless overridden by `is_leaf=`.
- **Rationale.** "A pytree is a container-like structure built out of container-like Python objects — 'leaf' pytrees and/or more pytrees." (pytrees page).
- **Source.** https://docs.jax.dev/en/latest/pytrees.html.

### JBP-5.2 — Custom classes: `register_pytree_node_class` vs `register_pytree_node`

- **Rule.** Use `@jax.tree_util.register_pytree_node_class` as a class decorator when you control the class definition. Use `jax.tree_util.register_pytree_node(cls, flatten, unflatten)` (functional API) when registering an external class. Both require a `tree_flatten` returning `(children, aux_data)` and a `tree_unflatten(aux_data, children)`.
- **Rationale.** "`register_pytree_node` is functional API ... `register_pytree_node_class` is decorator-based approach applied directly to classes. More concise for straightforward custom types." (pytrees page).
- **Typical violation.** Mixing the two APIs on the same class; forgetting that `aux_data` must be hashable (used as part of jit cache key).
- **How to detect.** Grep for `register_pytree_node`; confirm `aux_data` is composed of hashable, immutable values (tuples, frozensets, ints).
- **Source.** https://docs.jax.dev/en/latest/pytrees.html (section "Custom Pytree Registration"); reinforced by https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (Strategy 3 for class methods).

### JBP-5.3 — Static vs dynamic fields in custom pytrees

- **Rule.** Numerical data (arrays) goes in `children`. Configuration (shape, axis order, mode flags, hash-able config objects) goes in `aux_data` and is static. Changing `aux_data` triggers a jit recompile; changing `children` (same shapes) does not.
- **Rationale.** "Static fields cause recompilation when their values change. Only mark truly static configuration as such; numerical data should be dynamic to avoid excessive recompilation overhead." (pytrees page).
- **Typical violation.** Putting a shape parameter `n_phi` in `children` (now a tracer); or putting an array in `aux_data` (now hashed every call).
- **How to detect.** For each registered pytree, confirm `children` are arrays and `aux_data` is a tuple/frozenset of hashable primitives.
- **Source.** https://docs.jax.dev/en/latest/pytrees.html (section "Static vs Dynamic Fields").

### JBP-5.4 — Tree mapping requires matching structure

- **Rule.** `jax.tree.map(f, tree1, tree2)` requires identical pytree structures. Mismatch (different list length, different dict keys, different nesting depth) raises an error at runtime.
- **Rationale.** "Multi-argument `tree.map` requires identical structures — same list lengths, dict keys, nesting depth." (pytrees page).
- **Typical violation.** Mapping over `(gradient_dict, surface_state_dict)` where one has an extra key.
- **How to detect.** Cross-check pytree structures with `jax.tree_util.tree_structure(...)` before mapping; assert equality in tests.
- **Source.** https://docs.jax.dev/en/latest/pytrees.html (section "Common Gotchas").

### JBP-5.5 — Use `jax.tree.map` / `jax.tree.leaves` / `jax.tree.structure` (modern aliases)

- **Rule.** `jax.tree.map` is the preferred convenience alias of `jax.tree_util.tree_map` in modern JAX. Use it.
- **Source.** https://docs.jax.dev/en/latest/pytrees.html (section "Modern Convenience Namespace").

---

## JBP-6. Autodiff

### JBP-6.1 — `jax.grad` requires scalar output

- **Rule.** `jax.grad(f)` requires `f` to return a single scalar. For vector output use `jax.jacrev` / `jax.jacfwd` / `jax.vjp` / `jax.jvp`.
- **Rationale.** "Output must be scalar-valued" (autodiff cookbook).
- **Typical violation.** `jax.grad(lambda x: jnp.array([f1(x), f2(x)]))` — must use `value_and_grad` or `jacrev` instead.
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "jax.grad").

### JBP-6.2 — `value_and_grad` saves a forward pass

- **Rule.** Use `jax.value_and_grad(f)` rather than calling `f(x)` and `jax.grad(f)(x)` separately — the latter forward-passes twice.
- **Rationale.** "Avoids redundant computation versus calling the function and `grad` separately." (autodiff cookbook).
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "jax.value_and_grad").

### JBP-6.3 — `has_aux=True` for auxiliary outputs

- **Rule.** When the function returns `(loss, aux)`, decorate with `grad(f, has_aux=True)` or `value_and_grad(f, has_aux=True)`. The aux pytree is passed through unmodified.
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "jax.grad" → has_aux parameter).

### JBP-6.4 — `jacfwd` for tall Jacobians, `jacrev` for wide

- **Rule.** Use `jax.jacfwd` when the output dimension is larger than the input dimension (tall: e.g. mapping few DOFs to many residuals). Use `jax.jacrev` when input dim is larger (wide: many DOFs, few outputs, like an objective). For Hessians: `jacfwd(jacrev(f))` is canonical.
- **Rationale.** "jacfwd preferred for tall Jacobians ... jacrev preferred for wide Jacobians ... Hessian: `jacfwd(jacrev(f))` is typically most efficient." (autodiff cookbook).
- **Typical violation.** Using `jacrev` for a tall residual (computing `nresid × ndof` columns one at a time via VJP).
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "jax.jacfwd vs jax.jacrev").

### JBP-6.5 — `jax.vjp` returns `(value, vjp_fun)` — use it for VJP-based composition

- **Rule.** `y, vjp_fun = jax.vjp(f, x); cot = vjp_fun(g)` is the foundation for `grad`, `jacrev`, and custom adjoint paths. The `vjp_fun` accepts cotangents in the shape of `y` and returns cotangents in the shape of `x`.
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "jax.vjp and jax.jvp").

### JBP-6.6 — Hessian-vector products: forward-over-reverse

- **Rule.** `hvp(f, primals, tangents) = jax.jvp(jax.grad(f), primals, tangents)[1]` is the recommended HVP form. Forward-over-reverse is more memory-efficient than reverse-over-forward for typical objective functions.
- **Rationale.** "Forward-over-reverse is more efficient than reverse-over-forward." (autodiff cookbook).
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "Hessian-vector products").

### JBP-6.7 — `jax.custom_vjp` for implicit differentiation / non-differentiable inner solves

- **Rule.** When an inner iterative solver (Newton, BFGS, fixed-point) sits inside a function whose outer gradient is needed, wrap the solver in `@jax.custom_vjp` and define the gradient via the implicit-function theorem (IFT) rather than backpropagating through the solver iterations.
- **Rationale.** "For iterative solvers using while_loop, custom_vjp avoids differentiating through iterations by instead 'exploiting the mathematical structure' via the implicit function theorem." (custom-derivative-rules page).
- **Typical violation.** Backproping through a `jax.lax.while_loop` Newton solver (forbidden by JBP-3.3) or unrolling 100 BFGS iterations to be `jacrev`'d.
- **How to detect.** For every JAX module exposing a "solve then differentiate" pattern (BoozerSurfaceJAX, IotasJAX, NonQuasiSymmetricRatioJAX in this repo), confirm there is a `custom_vjp` with an `fwd` that returns residuals capturing the converged state, and a `bwd` that runs an adjoint linear solve.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html (sections "When to Use It", "Example: Implicit function differentiation of iterative implementations").

### JBP-6.8 — `custom_vjp` signature contract

- **Rule.** A `custom_vjp` wraps `f(x, y, ...)`. You must define `f_fwd(x, y, ...) -> (primal_out, residuals)` and `f_bwd(residuals, g) -> (x_bar, y_bar, ...)`. The `f_bwd` return tuple must have arity equal to the number of differentiable primal args; non-diff args (declared via `nondiff_argnums`) appear first in both `f_fwd` and `f_bwd` signatures.
- **Rationale.** "The `_bwd` function's return value must be a sequence with length matching the number of primal function arguments." (custom-derivative-rules page).
- **Typical violations.** (1) Returning a single `x_bar` from `f_bwd` for a two-arg primal. (2) Forgetting to wrap nondiff args. (3) Computing residuals lazily — `f_fwd` must explicitly save every value `f_bwd` will read.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html (section "jax.custom_vjp").

### JBP-6.9 — `custom_vjp` and forward-mode autodiff are incompatible

- **Rule.** `jax.jvp` / `jax.jacfwd` applied to a function decorated with `@jax.custom_vjp` raises an error. If both modes are needed, use `@jax.custom_jvp` instead and rely on transposition for VJP.
- **Rationale.** "Forward-mode autodiff cannot be used on the `jax.custom_vjp` function and will raise an error." (custom-derivative-rules page).
- **Source.** https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html (section "jax.custom_vjp" → restrictions).

### JBP-6.10 — `custom_jvp` requires linear tangent outputs

- **Rule.** A `@jax.custom_jvp` rule's tangent_out must be a linear function of `tangents`. If it is not (e.g. you call `jnp.sin(tangent_x)`), JAX cannot transpose to reverse mode and the rule produces wrong gradients.
- **Rationale.** "The tangent outputs should be a linear function of the tangent inputs." (custom-derivative-rules page).
- **Source.** https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html (section "jax.custom_jvp").

### JBP-6.11 — `stop_gradient` semantics

- **Rule.** `jax.lax.stop_gradient(x)` returns `x` in forward pass but yields a zero cotangent in backward pass. Use it (a) to detach an iterate from the autodiff graph at fixed-point convergence, (b) for straight-through estimator, (c) for RL policy-gradient. It is NOT a replacement for `custom_vjp` when you actually want a non-trivial adjoint.
- **Rationale.** "`stop_gradient` prevents *any* gradient computation; `custom_vjp` lets you compute custom gradients instead." (custom-derivative-rules page summary).
- **Typical violation.** Wrapping a Newton iterate in `stop_gradient` thinking it implements IFT — it does not; it just zeroes the gradient.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html (closing comparison section).

### JBP-6.12 — `jax.checkpoint` / `jax.remat` to trade memory for compute

- **Rule.** When a backward pass would store too many activations (typical in long `scan` loops or deep networks), wrap the forward function in `jax.checkpoint` to rematerialize activations on the backward pass. Use `jax.checkpoint_policies` to pick which intermediates to save.
- **Rationale.** "an alternative evaluation strategy is for some of the linearization points to be recomputed (i.e. rematerialized) rather than stored. This approach can reduce memory usage at the cost of increased computation." (checkpoint page).
- **Typical violation.** OOM on a long-trajectory `scan` adjoint that could be rescued by `jax.checkpoint(body)`.
- **Source.** https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html.

### JBP-6.13 — `check_grads` in tests

- **Rule.** Validate custom autodiff (custom_vjp, custom_jvp, implicit-diff adjoints) against finite differences via `jax.test_util.check_grads(f, args, order=1)` and `order=2` where relevant. Set `rtol` to match the parity-ladder lane.
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "Checking against numerical differences").

---

## JBP-7. Vectorization (`vmap`)

### JBP-7.1 — Prefer `vmap` over Python `for`

- **Rule.** Replace `for i in range(N): out[i] = f(x[i])` with `out = jax.vmap(f)(x)`. The vmap'd version executes in a single XLA kernel.
- **Source.** https://docs.jax.dev/en/latest/automatic-vectorization.html (section "Anti-patterns").

### JBP-7.2 — `in_axes` / `out_axes` control batched and broadcast dimensions

- **Rule.** Use `in_axes=(0, None)` to vmap argument 0 along its leading axis while broadcasting argument 1. Use `out_axes=0` (default) for batched leading-axis outputs.
- **Source.** https://docs.jax.dev/en/latest/automatic-vectorization.html (section "(a) Signature and Parameters").

### JBP-7.3 — Compose `vmap` outside `grad` for per-example gradients

- **Rule.** Per-example gradients = `jax.vmap(jax.grad(loss))(batch)`. Wrapping `grad` outside `vmap` (i.e. `grad(vmap(loss))`) computes a single scalar gradient over the whole batch, not per-example.
- **Source.** https://docs.jax.dev/en/latest/automatic-vectorization.html (section "(c) Composition with jit and grad").

### JBP-7.4 — `pmap` is deprecated; use `jit` + sharding instead

- **Rule.** Do not introduce new `jax.pmap` call sites. Use `jax.jit` with sharded inputs (NamedSharding + PartitionSpec) for multi-device parallelism.
- **Rationale.** "The modern JAX approach uses 'jit + sharded inputs' replacing deprecated `pmap`." (parallel page). Per the JAX 0.10.0 changelog (2026-04-16): "The `jax_pmap_shmap_merge` config state has been removed. `jax.pmap` now always wraps `jax.jit(jax.shard_map)`."
- **Typical violation.** Adding `@jax.pmap` decorators for multi-GPU code.
- **Source.** https://docs.jax.dev/en/latest/parallel.html (section "Single-Controller Model"); https://github.com/jax-ml/jax/blob/main/CHANGELOG.md (0.10.0 breaking changes).

---

## JBP-8. Sharding and multi-device

### JBP-8.1 — Single-controller model: `jit` + sharded inputs

- **Rule.** For a multi-GPU stellarator solve, build a `jax.sharding.NamedSharding(mesh, PartitionSpec(...))`, `device_put` the inputs, then `jax.jit(f)` runs across the mesh automatically.
- **Source.** https://docs.jax.dev/en/latest/parallel.html (section "Single-Controller Model").

### JBP-8.2 — Mesh, PartitionSpec, NamedSharding

- **Rule.** A `Mesh` names device-grid axes (e.g. `('X', 'Y')`). A `PartitionSpec` (aliased `jax.P`) maps array axes to mesh axes: `P('X', None)` shards axis 0 over 'X' and replicates axis 1. A `NamedSharding(mesh, spec)` is the sharding object you put on an array via `jax.device_put`.
- **Source.** https://docs.jax.dev/en/latest/parallel.html (sections "Mesh", "NamedSharding", "PartitionSpec").

### JBP-8.3 — `with_sharding_constraint` to nudge the compiler

- **Rule.** Inside a jitted function, `jax.lax.with_sharding_constraint(z, jax.P('X', None))` requests a specific intermediate sharding. Use sparingly; over-constraint can cause unnecessary all-gathers.
- **Source.** https://docs.jax.dev/en/latest/parallel.html (section "Sharding Constraints").

### JBP-8.4 — Explicit-sharding mode requires explicit `out_sharding`

- **Rule.** Under explicit-sharding meshes, a contraction like `jnp.dot(x, y)` with ambiguous output sharding raises rather than guessing. Pass `out_sharding=jax.P('X', None)` to disambiguate.
- **Rationale.** "When output sharding is ambiguous, JAX errors rather than guessing." (parallel page).
- **Source.** https://docs.jax.dev/en/latest/parallel.html (section "Error-First Philosophy").

### JBP-8.5 — `shard_map` for manual, collective-style code

- **Rule.** When you need MPI-style per-shard code with explicit collectives (`psum_scatter`, `all_gather`), use `jax.shard_map`. Use this for hand-tuned routines; otherwise prefer the automatic sharding path.
- **Source.** https://docs.jax.dev/en/latest/parallel.html (section "Manual Mode: Per-Device View").

### JBP-8.6 — Multi-process: initialize before any device touch

- **Rule.** `jax.distributed.initialize(...)` must run before any of `jax.devices()`, `jax.local_devices()`, or any computation. Calling it after device discovery is an error.
- **Rationale.** "jax.distributed.initialize() must be called before running jax.devices(), jax.local_devices(), or running any computations on devices." (multi-process page).
- **Source.** https://docs.jax.dev/en/latest/multi_process.html.

### JBP-8.7 — Identical script per process

- **Rule.** "All processes (usually) run the same Python script ... except for array creation." Branching on `jax.process_index()` inside collective operations causes deadlocks.
- **Source.** https://docs.jax.dev/en/latest/multi_process.html.

### JBP-8.8 — JAX 0.10.0 sharding-related breakage

- **Rule.** Verify (a) `PartitionSpec == tuple` comparisons have been removed (PartitionSpec equality is no longer compatible with tuples); (b) `jax.sharding.PmapSharding` is no longer accessible; (c) `jax.device_put_sharded` and `jax.device_put_replicated` have been removed — use `jax.device_put(array, NamedSharding(...))`.
- **Source.** https://github.com/jax-ml/jax/blob/main/CHANGELOG.md (jax 0.10.0, 2026-04-16, "Breaking Changes").

---

## JBP-9. Precision and dtype

### JBP-9.1 — `jax_enable_x64` set at the entrypoint, before any JAX call

- **Rule.** Set `jax.config.update("jax_enable_x64", True)` (or env `JAX_ENABLE_X64=True`) at the program entrypoint, before any `jnp.array` / device discovery / jit. Setting it later silently fails: arrays already created are still float32.
- **Rationale.** "JAX by default enforces single-precision ... only works before first JAX call" (gotchas page).
- **Typical violation.** A test fixture that enables x64 inside the test body after the JAX module has imported its own constants at float32.
- **How to detect.** Grep for `jax_enable_x64` and `JAX_ENABLE_X64`; verify it is set in `__init__.py`, conftest, or the top-level entrypoint script.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Double (64bit) precision").

### JBP-9.2 — Make dtypes explicit on every array allocation

- **Rule.** Use `jnp.zeros((n,), dtype=jnp.float64)` and `jnp.array(x, dtype=jnp.float64)`. Avoid relying on Python promotion rules. The simsopt-jax repo uses NumPy 2.x which has stricter promotion than NumPy 1.x.
- **Rationale.** Default-dtype reliance leads to silent f32 demotion under x64-disabled regimes; combined with NumPy 2.x type promotion changes, this can produce subtle parity-lane failures.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Miscellaneous divergences from NumPy", subsection on type promotion).

### JBP-9.3 — Subnormal flush-to-zero

- **Rule.** Some XLA backends flush denormalized floats to zero. Algorithms that depend on subnormals (very small steps in line search, near-singular matrix scaling) need explicit guards.
- **Rationale.** "Subnormal Floats: Uses flush-to-zero on some backends" (gotchas page).
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Miscellaneous divergences from NumPy").

### JBP-9.4 — Casting via `dtype=` not via `.astype(...)` chained on tracers

- **Rule.** Prefer `jnp.array(x, dtype=jnp.float64)` and `op(x, dtype=jnp.float64)` to make dtype intent explicit in the jaxpr. `.astype` is fine but easier to miss in review.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section on type promotion).

---

## JBP-10. Random numbers

### JBP-10.1 — Use `jax.random.key`, not `jax.random.PRNGKey`

- **Rule.** New code should call `jax.random.key(seed)` which produces a typed key array (e.g. `dtype=key<fry>`). `jax.random.PRNGKey` still works for back-compat but is the legacy form.
- **Source.** https://docs.jax.dev/en/latest/random-numbers.html (section "Current Preferred Approach").

### JBP-10.2 — Never reuse a key

- **Rule.** Each call to a `jax.random.*` sampler consumes a key; reusing the same key produces identical samples. Use `key, sub = jax.random.split(key)` (two-way) or `keys = jax.random.split(key, num=N)` (N-way) for independent draws.
- **Rationale.** "never reuse keys (unless you want identical outputs). Reusing the same state will cause sadness and monotony." (random-numbers page).
- **Typical violation.** A loop drawing samples with `key = jax.random.key(0)` once and not splitting per iteration.
- **How to detect.** Grep for `jax.random.normal`, `jax.random.uniform`, etc. and verify each call has a distinct key (either freshly split or threaded through a carry).
- **Source.** https://docs.jax.dev/en/latest/random-numbers.html.

### JBP-10.3 — Splitting is cheap

- **Rule.** `jax.random.split` is essentially arithmetic on the key bits. It is not a computational concern. Split liberally rather than fearing the cost.
- **Source.** https://docs.jax.dev/en/latest/random-numbers.html.

### JBP-10.4 — No global RNG state

- **Rule.** JAX has no equivalent of `np.random.seed()`. Always plumb a `key` argument through any function that draws random samples. Adapter classes that need randomness should accept a `key` in their interface.
- **Source.** https://docs.jax.dev/en/latest/random-numbers.html.

### JBP-10.5 — `jax.random.fold_in` for derived keys

- **Rule.** When you need a key derived from a counter or step index without consuming a splitting allocation, use `jax.random.fold_in(key, step_index)`. Useful inside `scan` bodies.
- **Source.** https://docs.jax.dev/en/latest/random-numbers.html (section "fold_in for Derived Keys").

### JBP-10.6 — Numerical-value compatibility caveat

- **Rule.** "The _exact_ values of numerical operations are not guaranteed to be stable across JAX releases." (api-compatibility page). Acceptance lanes must therefore pin a single jaxlib version and capture seeds explicitly.
- **Source.** https://docs.jax.dev/en/latest/api_compatibility.html (section "Numerical Compatibility").

---

## JBP-11. NumPy / JAX interop

### JBP-11.1 — Pass arrays, not Python lists

- **Rule.** `jnp.sum([1, 2, 3])` raises (in newer JAX) or implicitly converts (in older JAX) — either way it is wrong. Explicitly `jnp.array([1, 2, 3])` first.
- **Rationale.** "in tracing/JIT, each list element becomes a separate JAX variable, causing hidden performance degradation." (gotchas page).
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Non-array inputs").

### JBP-11.2 — `np.asarray(jnp.array(...))` materializes to host

- **Rule.** `np.asarray(x)` on a `jax.Array` forces a device-to-host transfer. Use only at module boundaries (e.g. handing data to SciPy). Never inside a jitted region.
- **Source.** https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html (interop section).

### JBP-11.3 — `jax.device_put` and `jax.device_get`

- **Rule.** Use `jax.device_put(x, device_or_sharding)` to move data to a specific device or sharding. Use `jax.device_get(x)` for an explicit, synchronous host pull. These are the canonical primitives — avoid implicit transfers.
- **Source.** https://docs.jax.dev/en/latest/parallel.html (section "device_put for Resharding").

### JBP-11.4 — Async dispatch + `block_until_ready()` for benchmarking

- **Rule.** When timing JAX code, call `.block_until_ready()` on the result. Otherwise, `%time` returns only the dispatch latency, not the actual compute time.
- **Rationale.** "To measure the true cost of the operation we must either read the value on the host ... or use the `block_until_ready()` method." (async-dispatch page).
- **Typical violation.** Benchmarks in `benchmarks/` that print microsecond timings for ops that secretly take milliseconds because dispatch is async.
- **How to detect.** Every `time.perf_counter()` pair around a JAX call must bracket a `.block_until_ready()` on the final array.
- **Source.** https://docs.jax.dev/en/latest/async_dispatch.html.

### JBP-11.5 — JAX 0.10.0 NumPy interop changes

- **Rule.** Audit calls to `jnp.hstack/vstack/dstack/...` and `jnp.clip`. As of jax 0.10.0: "Stack functions (hstack, vstack, dstack, etc.) no longer accept non-ArrayLike inputs"; and "jax.numpy.clip deprecated keyword arguments removed". Confirm all such call sites pass arrays / array-likes only.
- **Source.** https://github.com/jax-ml/jax/blob/main/CHANGELOG.md (jax 0.10.0).

---

## JBP-12. Side effects in jit (callbacks)

### JBP-12.1 — `jax.debug.print` is the only safe print inside jit

- **Rule.** Python `print(x)` inside a jitted body prints the *tracer* (only at trace time, not at runtime). For runtime values, use `jax.debug.print("x = {}", x)` — it routes via host callback and prints the actual value each time the compiled function runs.
- **Rationale.** "Standard Python `print()` fires only during tracing ... `jax.debug.print()` fires at runtime." (debugging page).
- **Source.** https://docs.jax.dev/en/latest/debugging/index.html (section "Interactive inspection with jax.debug").

### JBP-12.2 — `jax.pure_callback` for pure host functions

- **Rule.** Use `jax.pure_callback(fn, result_shape_dtype, *args)` to call a pure Python/NumPy/SciPy function from inside a traced region. It supports `jit` and `vmap`. It does NOT support `grad` unless you wrap with `@custom_jvp`. The compiler is free to elide it if its output is unused.
- **Rationale.** "appropriate for pure functions" (external-callbacks page); compiler may eliminate unused callbacks.
- **Source.** https://docs.jax.dev/en/latest/external-callbacks.html.

### JBP-12.3 — `jax.experimental.io_callback` for ordered side effects

- **Rule.** When you need a side effect (e.g. checkpoint write, telemetry emit) and want guaranteed execution, use `jax.experimental.io_callback(fn, result_shape, *args, ordered=True)`. Note: `ordered=True` is incompatible with `vmap`.
- **Source.** https://docs.jax.dev/en/latest/external-callbacks.html.

### JBP-12.4 — `jax.debug.callback` for debug-only with full transformation compatibility

- **Rule.** `jax.debug.callback(fn, *args)` returns nothing and is the only callback compatible with `grad`. Use it for inspecting gradient flow at specific points.
- **Source.** https://docs.jax.dev/en/latest/external-callbacks.html.

---

## JBP-13. Compile cache

### JBP-13.1 — Enable the persistent cache via `jax.config.update("jax_compilation_cache_dir", ...)` BEFORE the first compile

- **Rule.** Set the cache directory once, at the entrypoint, before any `jit` call. Use a shared filesystem (NFS) or GCS for distributed runs.
- **Source.** https://docs.jax.dev/en/latest/persistent_compilation_cache.html (section "Configuration Setup").

### JBP-13.2 — Cache key includes HLO, jaxlib version, XLA flags, device topology

- **Rule.** The cache key incorporates: the non-optimized HLO, jaxlib version, XLA compile flags, device count and topology, and a custom hook. Shape and dtype changes invalidate the cache. Plan for cache misses across hardware/version boundaries.
- **Rationale.** "Shape and dtype changes invalidate the cache — different tensor dimensions or data types produce new keys." (cache page).
- **Source.** https://docs.jax.dev/en/latest/persistent_compilation_cache.html (section "Cache Key Sensitivity").

### JBP-13.3 — Caching thresholds

- **Rule.** `jax_persistent_cache_min_compile_time_secs` (default 1.0s) governs whether a compile is worth caching. For Stage 2 / single-stage objectives where compile takes 5-60 s, the defaults are fine. For fast inner solves, consider lowering.
- **Source.** https://docs.jax.dev/en/latest/persistent_compilation_cache.html (section "Caching Thresholds").

### JBP-13.4 — Cache does not work with host callbacks or custom_partitioning

- **Rule.** "The persistent cache doesn't work with function that have host callbacks". A function that uses `jax.pure_callback` / `jax.experimental.io_callback` / `jax.debug.callback` will be recompiled every run.
- **Source.** https://docs.jax.dev/en/latest/persistent_compilation_cache.html (section "Pitfalls").

### JBP-13.5 — Multi-process: rank-0 writes, others read

- **Rule.** In a multi-process job, only rank 0 writes the cache. Other processes read it. If the cache is on rank-0-local disk, other ranks recompile every run. Use NFS/GCS for shared caches.
- **Source.** https://docs.jax.dev/en/latest/persistent_compilation_cache.html (section "Multi-Process Considerations").

---

## JBP-14. Concrete vs abstract — `ConcretizationTypeError` patterns

### JBP-14.1 — Branching on tracer-derived booleans

- **Rule.** `if jnp.any(jnp.isnan(x)): ...` raises `TracerBoolConversionError`. Refactor with `jnp.where` masking or `jax.lax.cond`.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html.

### JBP-14.2 — `.item()` / `.tolist()` inside jit

- **Rule.** `.item()` and `.tolist()` force concretization and fail on tracers. Reserve them for code that runs outside any JAX transformation.
- **Source.** Derived from concretization rules across https://docs.jax.dev/en/latest/jit-compilation.html and gotchas pages.

### JBP-14.3 — Using `len(array)` on a 1D tracer

- **Rule.** `len(x)` on a 1D JAX array returns the shape — this is concrete (shape is static) and works under jit. But `len(x)` on a list of tracers does not. Be aware which `len` you are calling.

### JBP-14.4 — Concrete inputs at the boundary

- **Rule.** When implementing a `custom_vjp`, the residuals saved in `fwd` and consumed in `bwd` cross transformation boundaries. Confirm that anything used for control flow inside `fwd` or `bwd` is either Python-static or wrapped with `jax.lax.cond`.

---

## JBP-15. `jax.lax.while_loop` reverse-mode (deepdive)

### JBP-15.1 — Reverse-mode through `while_loop` is unsupported — period

- **Rule.** There is no built-in workaround. If you need a gradient through a data-dependent loop, you have three options:
  1. Convert to `jax.lax.scan` with a fixed upper-bound iteration count and a `done` mask.
  2. Wrap the loop in `jax.custom_vjp` and define the gradient analytically (typically via IFT for fixed-point iterations).
  3. Use `optimistix` / `lineax` / `equinox` libraries that provide checkpointed loops with custom adjoint rules.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "lax.while_loop").

### JBP-15.2 — `fori_loop` lowers to `while_loop` when bounds are tracers

- **Rule.** When loop bounds are not static integers, `fori_loop` internally lowers to `while_loop` and inherits its no-reverse-mode restriction.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html (section "lax.fori_loop").

---

## JBP-16. Memory management

### JBP-16.1 — `donate_argnums` to reuse input buffers

- **Rule.** When a jitted function consumes a large array `x` and produces `y` of the same shape/dtype, declare `donate_argnums=(0,)` (positional only — does not work with keyword args). XLA reuses the buffer, halving peak memory.
- **Rationale.** "you can specify that you want the corresponding input buffer to be donated to hold an output. This will reduce the memory required" (buffer-donation page).
- **Source.** https://docs.jax.dev/en/latest/buffer_donation.html.

### JBP-16.2 — Donated buffers must not be used afterward

- **Rule.** After `jax.jit(f, donate_argnums=(0,))(x)` returns, `x` is invalid. Touching it raises `RuntimeError`. Use this only in functional, immutable update patterns where the caller does not need the old value.
- **Source.** https://docs.jax.dev/en/latest/buffer_donation.html (section "Common Gotchas").

### JBP-16.3 — Donation requires positional args

- **Rule.** "this currently does not work when calling your function with key-word arguments" — calling the jitted function with kwargs disables donation silently.
- **Source.** https://docs.jax.dev/en/latest/buffer_donation.html.

### JBP-16.4 — Unused-donation `UserWarning`

- **Rule.** "Some donated buffers were not usable" warnings indicate the function's output shape/dtype does not match the donated input; either fix the shapes or drop the donation.
- **Source.** https://docs.jax.dev/en/latest/buffer_donation.html.

---

## JBP-17. Numerical equivalence pitfalls

### JBP-17.1 — The "double `where`" trick for NaN-safe gradients

- **Rule.** A `jnp.where(cond, valid, fallback)` masks the value but BOTH branches' gradients are still computed. If `valid` is `sqrt(x)` and `cond` is `x > 0`, the gradient of `sqrt` at the masked locations is `+inf` or `NaN`, and the resulting cotangent is NaN. Fix: mask the input to the unsafe branch as well, so the unsafe op is never applied to its unsafe argument even in the "dead" computation:

```python
safe = jnp.where(x > 0, x, 1.0)         # always positive
result = jnp.where(x > 0, jnp.sqrt(safe), 0.0)
```

- **Rationale.** "The fix involves protecting both branches by applying the mask again internally" (gotchas / autodiff cross-reference).
- **Typical violation.** `safe_norm = jnp.sqrt(jnp.sum(x**2))` at `x = 0` — gradient is `0/0` → NaN.
- **How to detect.** Grep for `jnp.sqrt`, `jnp.log`, `1.0 / `, `jnp.where(.*safe.*` patterns in autodiff-targeted code and verify each potentially-singular op is masked at the input level, not only at the output.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Debugging NaNs and Infs"); pattern documented in https://docs.jax.dev/en/latest/faq.html ("ensure that there is a jnp.where inside the partially-defined function").

### JBP-17.2 — Out-of-bounds indexing is silent

- **Rule.** `jnp.arange(10)[11]` returns `9`, not an error. Updates `.at[11].set(...)` are silently dropped. Use `mode='fill'` for explicit out-of-bounds NaN, or assert in tests.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Out-of-bounds indexing").

### JBP-17.3 — Boolean indexing is forbidden in jit

- **Rule.** `x[mask]` where `mask` is a tracer raises `NonConcreteBooleanIndexError` under jit. Use `jnp.where(mask, x, 0).sum()` for masked reductions, or compress upstream with a static shape.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Dynamic shapes").

### JBP-17.4 — In-place updates via `.at[]`

- **Rule.** `x[i] = v` is illegal. Use `x.at[i].set(v)` / `.at[...].add(v)` / `.at[...].max(v)` / etc. These return a new array.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 In-place updates").

---

## JBP-18. API stability and deprecations relevant to JAX 0.10.0

### JBP-18.1 — Public API surface

- **Rule.** Stable public modules: `jax`, `jax.dlpack`, `jax.image`, `jax.lax`, `jax.nn`, `jax.numpy`, `jax.ops`, `jax.profiler`, `jax.random`, `jax.scipy`, `jax.tree`, `jax.tree_util`, `jax.test_util`. Avoid importing from `jax.core`, `jax.interpreters`, `jax.lib`, `jax.util`, `jax.experimental`, `jax.example_libraries`, or anything underscore-prefixed.
- **Rationale.** "Internal/Private APIs are not covered by stability guarantees ... `jax.experimental` — Experimental—may change without warning." (api-compatibility page).
- **Source.** https://docs.jax.dev/en/latest/api_compatibility.html.

### JBP-18.2 — 3-month deprecation timeline

- **Rule.** When JAX deprecates an API, it gives at least 3 months' notice via `DeprecationWarning` and changelog. Watch the warnings in CI; do not suppress them globally.
- **Source.** https://docs.jax.dev/en/latest/api_compatibility.html (section "Deprecation Policy").

### JBP-18.3 — JAX 0.10.0 breaking changes (2026-04-16)

- **Rule.** Audit and verify against these breaking changes:
  1. **PartitionSpec equality**: PartitionSpec no longer equates to tuple — fix `spec == (...)` patterns.
  2. **`jax.core.ShapedArray.vma` removed**: use `manual_axis_type.varying`.
  3. **CPU device naming**: `cpu:0`, `cpu:1`, ... (not `TFRT_CPU_0`). Update device-name-based tests.
  4. **`jax.pmap` mandatory implementation**: `jax_pmap_shmap_merge` config removed; `pmap` always wraps `jit(shard_map)`.
  5. **`jax.device_put_sharded` / `jax.device_put_replicated` removed**: use `jax.device_put(array, NamedSharding(...))`.
  6. **`jax.sharding.PmapSharding` removed**.
  7. **`jax.numpy.clip` deprecated kwargs removed**.
  8. **Stack functions (`hstack/vstack/dstack/...`) require ArrayLike**.
  9. **SciPy minimum version: 1.14**.
  10. **Batched linear solvers (`cho_solve`/`lu_solve`/`solve_triangular`) deprecate batched 1D solves with `b.ndim > 1`**.
- **Source.** https://github.com/jax-ml/jax/blob/main/CHANGELOG.md (jax 0.10.0, 2026-04-16).

### JBP-18.4 — `jax.tree_util` is stable; `jax.tree` is the modern alias

- **Rule.** Use `jax.tree.map`, `jax.tree.leaves`, `jax.tree.structure`. Avoid `jax.tree_util.tree_*` in new code but accept it in legacy.
- **Source.** https://docs.jax.dev/en/latest/pytrees.html (section "Modern Convenience Namespace").

### JBP-18.5 — `jax.Array` is the canonical array type

- **Rule.** All JAX arrays are `jax.Array` instances. There is no remaining `DeviceArray` / `ShardedDeviceArray` split. Reviewer-flag any isinstance checks against legacy names.
- **Source.** https://docs.jax.dev/en/latest/api_compatibility.html.

### JBP-18.6 — Exact-value reproducibility is not guaranteed across releases

- **Rule.** "The exact values of numerical operations are not guaranteed to be stable across JAX releases." Acceptance lanes must pin jaxlib (this repo: jaxlib 0.10.0) and treat byte-identity gates as version-locked.
- **Source.** https://docs.jax.dev/en/latest/api_compatibility.html (section "Numerical Compatibility").

---

## JBP-19. Anti-patterns

### JBP-19.1 — Do not pass Python `list` / `tuple` to `jnp.sum` / `jnp.array` aggregators

- **Rule.** Convert first: `jnp.sum(jnp.array(xs))`.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (section "🔪 Non-array inputs").

### JBP-19.2 — Do not call `.item()` / `int(x)` / `float(x)` inside jit

- **Rule.** All three force concretization and fail on tracers.
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html (Concretization section).

### JBP-19.3 — Do not branch Python `if` on tracers

- **Rule.** Use `jax.lax.cond` or restructure to use `jnp.where`.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html.

### JBP-19.4 — Do not call `np.asarray()` on a tracer

- **Rule.** Calling NumPy conversion on a tracer fails. Convert at boundaries only.
- **Source.** https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html (interop section).

### JBP-19.5 — Do not accumulate via `jnp.append` in a Python `for` loop

- **Rule.** `jnp.append` allocates a new array every call. Use `jnp.stack` once on a pre-built list, or `jax.lax.scan` with carry, or preallocate with `jnp.zeros((N, ...))` and `.at[i].set(...)`.
- **Source.** https://docs.jax.dev/en/latest/control-flow.html and https://docs.jax.dev/en/latest/automatic-vectorization.html (anti-patterns).

### JBP-19.6 — Do not mix `jit` and Python control flow on tracers

- **Rule.** See JBP-3.1.

### JBP-19.7 — Do not register a class as a pytree AND mark it `static` in jit

- **Rule.** Pick one. If the class is a pytree (`children` = arrays), its arrays flow through jit as dynamic data. If the class is `static_argnums`, the whole class is hashed/frozen and `__hash__`/`__eq__` must be defined.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (class-methods section).

### JBP-19.8 — Do not use `pmap` for new code

- **Rule.** See JBP-7.4.

### JBP-19.9 — Do not store tracers on `self`

- **Rule.** See JBP-2.5 / JBP-4.3.

### JBP-19.10 — Do not skip `block_until_ready()` in benchmarks

- **Rule.** See JBP-11.4.

---

## JBP-20. Testing JAX code

### JBP-20.1 — `jax.test_util.check_grads` to validate custom autodiff

- **Rule.** For any function with `custom_vjp` / `custom_jvp` / implicit-diff, run `jax.test_util.check_grads(f, args, order=1, modes=('rev', 'fwd'))` and `order=2` if a Hessian path exists. Tolerance must match the parity-ladder lane (`rtol=1e-6` for FD, tighter for direct-kernel).
- **Source.** https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html (section "Checking against numerical differences").

### JBP-20.2 — `jax.tree_util.tree_all_close` (or `jax.tree.map` + `jnp.allclose`) for nested results

- **Rule.** When comparing two pytrees (e.g. JAX result dict vs CPU result dict), use `jax.tree.map(lambda a, b: jnp.allclose(a, b, rtol=R, atol=A), tree_a, tree_b)` then reduce.
- **Source.** https://docs.jax.dev/en/latest/pytrees.html.

### JBP-20.3 — Set `jax_enable_x64` before any JAX call in conftest

- **Rule.** Repo's `conftest.py` (or top-level test setup) must call `jax.config.update("jax_enable_x64", True)` before any JAX module imports.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (x64 section).

### JBP-20.4 — Use `jax.check_tracer_leaks()` for adapter-class tests

- **Rule.** Tests that exercise `BoozerSurfaceJAX`, `SquaredFluxJAX`, `BiotSavartJAX`, or any other adapter class that holds caches should run at least one parameterization under `with jax.check_tracer_leaks(): ...`.
- **Source.** https://docs.jax.dev/en/latest/jit-compilation.html.

### JBP-20.5 — Independent oracle per parity assertion

- **Rule (repo-specific, complements JAX docs).** New `test_*_jax_*.py` files must cite an independent oracle for every parity assertion (C++ symbol, closed-form expression, finite-difference, pinned dataset). Re-export `is`-identity, JAX-vs-JAX, and "host wrapper that routes through JAX" comparisons are tautologies and are forbidden. (See `tests/REVIEWER_ORACLE_LINT.md`.)
- **Source.** This repository's CLAUDE.md and `tests/REVIEWER_ORACLE_LINT.md` (project-specific layer on top of JAX baseline).

### JBP-20.6 — `JAX_DEBUG_NANS=True` in CI

- **Rule.** Run at least one CI job with `JAX_DEBUG_NANS=True` (and ideally `JAX_DEBUG_INFS=True`) to surface silent NaN propagation.
- **Source.** https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html (NaN section); https://docs.jax.dev/en/latest/debugging/index.html.

### JBP-20.7 — `jax.disable_jit()` for stepwise debug

- **Rule.** When a jitted function misbehaves, wrap the test in `with jax.disable_jit(): ...` to run Python-eagerly and use `pdb`/`print` normally. Tracer-related errors that disappear under disable_jit are tracing bugs (not numerical bugs).
- **Source.** https://docs.jax.dev/en/latest/debugging/index.html (section "JAX_DISABLE_JIT").

### JBP-20.8 — Pin jaxlib version in lockfile

- **Rule.** Because "exact values of numerical operations are not guaranteed to be stable across JAX releases" (JBP-18.6), pin jaxlib in `requirements.txt` / `pyproject.toml` (this repo: 0.10.0). Byte-identity gates and parity lanes are bound to this pin.
- **Source.** https://docs.jax.dev/en/latest/api_compatibility.html.

---

## Reviewer Checklist (one-liners)

The following one-line rules are the minimum that every JAX module in this repo must satisfy. Cite by section number (e.g. "violates JBP-3.3") when filing review comments.

- [JBP-1.1] Inputs in args, outputs in return; no global reads/writes in traced code.
- [JBP-1.2] No `print` / `logging` / `time.*` / file I/O inside `@jit` or under `vmap`/`grad`/`scan`/`cond`.
- [JBP-1.3] Deterministic: no `time.time()`, no `np.random.*`, no Python `id()`-based branching.
- [JBP-2.1] Use `static_argnames` not `static_argnums`.
- [JBP-2.2] Static-arg value space must be small and bounded (no float thresholds as static).
- [JBP-2.3] `jit` at module scope, not inside loops.
- [JBP-2.4] Methods are jitted via pytree-registration, helper function, or `static_argnums=(0,)` with `__hash__`/`__eq__` — pick one.
- [JBP-2.5] No tracers stored on `self`; tests guard with `jax.check_tracer_leaks()`.
- [JBP-2.6] No `int(.)/bool(.)/float(.)` / `if <tracer>:` inside jit.
- [JBP-2.7] Boundary scalar conversions at jit edges; JAX scalars cast to Python before SciPy/NumPy.
- [JBP-3.1] No Python `if`/`while` on tracers.
- [JBP-3.2] Both `cond` branches must return matching pytree (structure + shape + dtype).
- [JBP-3.3] No reverse-mode autodiff through `lax.while_loop`; use `scan` or `custom_vjp`.
- [JBP-3.4] `fori_loop` bounds must be static for reverse-mode autodiff.
- [JBP-3.5] Use `lax.scan` for loops with carry that need gradients.
- [JBP-3.6] Use `&`/`|`/`~` and `jnp.logical_*`, never `and`/`or`/`not` on JAX arrays.
- [JBP-3.7] Fixed-length Python `for` loops are fine for small `N` but blow up XLA compile for large `N`.
- [JBP-4.1] A jit closure freezes the captured array; rebuild after mutation.
- [JBP-4.2] A jit closure freezes the captured Python scalar; pass as static arg if it changes.
- [JBP-4.3] Tracers must not escape via `self`, lists, dicts, or return values from non-traced functions.
- [JBP-5.1] Pytree containers are list/tuple/dict/None; everything else is a leaf or registered.
- [JBP-5.2] Custom classes registered via `register_pytree_node_class` or `register_pytree_node`; `aux_data` must be hashable.
- [JBP-5.3] Arrays in `children`, configuration in `aux_data`.
- [JBP-5.4] `tree.map` requires identical pytree structures.
- [JBP-5.5] Use `jax.tree.map` (modern alias).
- [JBP-6.1] `grad` requires scalar output; use `jacrev`/`jacfwd`/`vjp`/`jvp` for vector output.
- [JBP-6.2] Use `value_and_grad` to avoid recomputing the forward pass.
- [JBP-6.3] `has_aux=True` for `(loss, aux)` returns.
- [JBP-6.4] `jacfwd` for tall jacobians (more outputs), `jacrev` for wide (more inputs); Hessian = `jacfwd(jacrev(f))`.
- [JBP-6.5] `vjp` returns `(value, vjp_fun)`; foundation for adjoints.
- [JBP-6.6] HVP = `jvp(grad(f), primals, tangents)[1]` (forward-over-reverse).
- [JBP-6.7] Inner iterative solvers need `custom_vjp` with implicit-function-theorem adjoint; never backprop through Newton/BFGS iterations directly.
- [JBP-6.8] `custom_vjp` `f_bwd` returns a tuple of arity = number of differentiable primal args; nondiff args first.
- [JBP-6.9] `custom_vjp` is incompatible with `jvp`/`jacfwd`; use `custom_jvp` if both modes are needed.
- [JBP-6.10] `custom_jvp` tangent output must be linear in tangent input.
- [JBP-6.11] `stop_gradient` zeros gradients; it is not a substitute for a `custom_vjp` adjoint.
- [JBP-6.12] Use `jax.checkpoint` for long `scan` loops to control adjoint memory.
- [JBP-6.13] Validate every custom adjoint with `jax.test_util.check_grads`.
- [JBP-7.1] Replace Python `for` over batch with `jax.vmap`.
- [JBP-7.2] Use `in_axes`/`out_axes` to control batching.
- [JBP-7.3] Per-example gradients: `vmap(grad(loss))`, not `grad(vmap(loss))`.
- [JBP-7.4] No new `pmap`; use `jit` + sharding.
- [JBP-8.1] Multi-GPU uses `jit` + `NamedSharding`.
- [JBP-8.2] `Mesh` names axes, `PartitionSpec` maps array axes to mesh axes, `NamedSharding` ties them together.
- [JBP-8.3] `with_sharding_constraint` only when needed.
- [JBP-8.4] Explicit-sharding mode: pass `out_sharding=...` to contractions.
- [JBP-8.5] Hand-tuned collective code uses `shard_map`.
- [JBP-8.6] `jax.distributed.initialize` before any device touch.
- [JBP-8.7] Same script on all processes; no `process_index`-conditional collectives.
- [JBP-8.8] Audit JAX-0.10.0-removed sharding APIs (`PmapSharding`, `device_put_sharded`, `device_put_replicated`, `PartitionSpec == tuple`).
- [JBP-9.1] `jax_enable_x64` at entrypoint, before any JAX call.
- [JBP-9.2] Explicit `dtype=` on array allocations.
- [JBP-9.3] No reliance on subnormal floats.
- [JBP-9.4] Make casts explicit via `dtype=`, not silent `.astype` chains.
- [JBP-10.1] `jax.random.key(seed)` (not legacy `PRNGKey`).
- [JBP-10.2] Never reuse a key; split per draw.
- [JBP-10.3] Splitting is cheap.
- [JBP-10.4] Thread `key` through function signatures; no global RNG.
- [JBP-10.5] `fold_in` for counter-derived keys inside `scan`.
- [JBP-10.6] Pin jaxlib for parity lanes — exact numerics not guaranteed across releases.
- [JBP-11.1] Pass JAX arrays, not Python lists.
- [JBP-11.2] `np.asarray(jax.Array)` at boundaries only.
- [JBP-11.3] Use `jax.device_put` / `device_get` for explicit transfers.
- [JBP-11.4] `block_until_ready()` in every benchmark.
- [JBP-11.5] Audit `jnp.hstack/vstack/dstack/clip` for the JAX 0.10.0 ArrayLike requirement.
- [JBP-12.1] Use `jax.debug.print` for runtime values inside jit.
- [JBP-12.2] `pure_callback` for pure host functions; supports `jit`/`vmap`; needs `custom_jvp` for `grad`.
- [JBP-12.3] `io_callback(ordered=True)` for guaranteed side effects; incompatible with `vmap`.
- [JBP-12.4] `debug.callback` for debug-only with full transformation compatibility.
- [JBP-13.1] Persistent cache dir set at entrypoint, before first jit.
- [JBP-13.2] Cache key invalidated by shape, dtype, jaxlib version, device topology, XLA flags.
- [JBP-13.3] `jax_persistent_cache_min_compile_time_secs` defaults are fine for stage-2 / single-stage.
- [JBP-13.4] Cache disabled for functions with host callbacks.
- [JBP-13.5] Multi-process: shared filesystem cache or rank-0 only writes.
- [JBP-14.1] Branching on `jnp.any` / `jnp.all` outputs inside jit fails — use `jnp.where` masking.
- [JBP-14.2] `.item()` / `.tolist()` outside jit only.
- [JBP-15.1] Reverse-mode through `while_loop` unsupported — restructure or `custom_vjp`.
- [JBP-15.2] `fori_loop` with traced bounds inherits `while_loop` restrictions.
- [JBP-16.1] `donate_argnums` for input-buffer reuse on large in-place-style updates.
- [JBP-16.2] Donated buffers are dead after the call.
- [JBP-16.3] Donation requires positional args, not kwargs.
- [JBP-16.4] Investigate `Some donated buffers were not usable` warnings.
- [JBP-17.1] Apply the "double `where`" pattern for NaN-safe gradients.
- [JBP-17.2] Out-of-bounds indexing is silent — guard explicitly.
- [JBP-17.3] No boolean indexing under jit; use `jnp.where` masking.
- [JBP-17.4] Updates via `.at[]`, never `x[i] = v`.
- [JBP-18.1] Import only public modules (`jax.numpy`, `jax.lax`, `jax.random`, `jax.tree`, ...). Avoid `jax.core`, `jax.interpreters`, `jax.lib`.
- [JBP-18.2] Treat `DeprecationWarning` as a follow-up item, not as noise.
- [JBP-18.3] Audit JAX 0.10.0 breaking changes (PartitionSpec equality, cpu naming, pmap implementation, device_put_sharded removal, NumPy alignments, SciPy 1.14 min, batched linear solver).
- [JBP-18.4] Use `jax.tree` over `jax.tree_util` in new code.
- [JBP-18.5] `jax.Array` is the canonical type; no `DeviceArray`/`ShardedDeviceArray` checks.
- [JBP-18.6] Pin jaxlib for byte-identity / parity lanes.
- [JBP-19.1] No Python lists into JAX aggregators.
- [JBP-19.2] No `.item()` / `int(.)` / `float(.)` under jit.
- [JBP-19.3] No `if <tracer>:` under jit.
- [JBP-19.4] No `np.asarray(<tracer>)`.
- [JBP-19.5] No `jnp.append` accumulator loops.
- [JBP-19.6] No mixing Python control flow with traced values.
- [JBP-19.7] Class is pytree XOR static — not both.
- [JBP-19.8] No new `pmap`.
- [JBP-19.9] No tracers on `self`.
- [JBP-19.10] No benchmark without `block_until_ready`.
- [JBP-20.1] `check_grads` for every custom autodiff path.
- [JBP-20.2] `tree.map` + `allclose` for nested-result comparison.
- [JBP-20.3] `jax_enable_x64` in conftest before any import that touches JAX.
- [JBP-20.4] `check_tracer_leaks()` for adapter-class tests.
- [JBP-20.5] Independent oracle for every parity assertion; no JAX-vs-JAX tautologies.
- [JBP-20.6] At least one CI job with `JAX_DEBUG_NANS=True`.
- [JBP-20.7] `disable_jit()` is the first-pass debug tool.
- [JBP-20.8] Pin jaxlib in `requirements.txt` / `pyproject.toml`.

---

## Sources (all fetched 2026-05-16)

- https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html
- https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
- https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html
- https://docs.jax.dev/en/latest/jit-compilation.html
- https://docs.jax.dev/en/latest/pytrees.html
- https://docs.jax.dev/en/latest/control-flow.html
- https://docs.jax.dev/en/latest/parallel.html
- https://docs.jax.dev/en/latest/multi_process.html
- https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html
- https://docs.jax.dev/en/latest/random-numbers.html
- https://docs.jax.dev/en/latest/api_compatibility.html
- https://docs.jax.dev/en/latest/persistent_compilation_cache.html
- https://docs.jax.dev/en/latest/jax-primitives.html
- https://docs.jax.dev/en/latest/debugging/index.html
- https://docs.jax.dev/en/latest/external-callbacks.html
- https://docs.jax.dev/en/latest/async_dispatch.html
- https://docs.jax.dev/en/latest/automatic-vectorization.html
- https://docs.jax.dev/en/latest/buffer_donation.html
- https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html
- https://docs.jax.dev/en/latest/faq.html
- https://github.com/jax-ml/jax/blob/main/CHANGELOG.md (JAX 0.10.0 entry, 2026-04-16)

Repo-internal cross-references (project layer on top of JAX baseline):

- `/Users/suhjungdae/code/columbia/simsopt-jax/CLAUDE.md`
- `/Users/suhjungdae/code/columbia/simsopt-jax/tests/REVIEWER_ORACLE_LINT.md`
