# Native-to-JAX Unit-Test Coverage Blindspot Pass

**Status:** Draft  
**Last updated:** 2026-07-29

## Question Investigated

What is still unknown behind this statement?

> Baseline: approximately 737 native test definitions across 70 files. Existing
> JAX tests cannot be subtracted directly because coverage is not one-to-one.
> The first inventory/classification phase will produce the defensible missing
> count.

## Verdict

The statement is directionally correct but its baseline is already stale and
its implied single “missing count” is underspecified.

- Local `upstream_check/master` at `377cf6651` contains 70 test files and **738**
  `unittest` test methods, not 737.
- Current remote upstream `master` is `4ad6fd99189b99d9722ad33aaeb5d30adc81680f`.
  It contains 71 test files and **753** test methods.
- The remote delta adds or changes 15 test methods across configuration,
  deprecation/framework, and logging behavior. Most of that delta is not a new
  GPU numerical kernel.
- The remote suite also contains 91 `self.subTest(...)` sites and 73 conditional
  skip sites. A test-method count therefore does not equal the number of
  behavioral cases that execute in every environment.
- No separate C++ test source files were found under conventional upstream test
  paths. The “native C++ tests” are predominantly Python `unittest` tests that
  exercise SIMSOPT and the compiled `simsoptpp` bindings directly or
  indirectly.

The defensible output is not one subtraction. It is a coverage ledger plus
separate counts for portable JAX equivalence, partial evidence, missing JAX
capability, hybrid boundaries, shared Python behavior, and native-only
behavior.

## Evidence Snapshot

| Evidence | Local snapshot | Current remote upstream |
|---|---:|---:|
| Commit | `377cf6651` | `4ad6fd99189b99d9722ad33aaeb5d30adc81680f` |
| Native test files | 70 | 71 |
| `test_` methods | 738 | 753 |
| `unittest.TestCase` classes | 133 | 135 |
| `self.subTest(...)` sites | 91 | 91 |
| Conditional skip sites | 73 | 73 |
| Conventional C++ test files | 0 | 0 |

Remote test-method definitions by domain:

| Domain | Definitions |
|---|---:|
| `tests/configs` | 12 |
| `tests/core` | 171 |
| `tests/field` | 181 |
| `tests/geo` | 226 |
| `tests/mhd` | 98 |
| `tests/objectives` | 15 |
| `tests/solve` | 17 |
| `tests/util` | 33 |
| **Total** | **753** |

These are source-structure counts, not coverage results. The current repository
also has 26 full example parity relationships and one unsupported VMEC-hybrid
certification relationship, but example evidence must not be counted as
complete unit-capability coverage.

## Prioritized Blindspots

### 1. The denominator is not defined

**What was hidden:** “737 tests” could mean source methods, collected node IDs,
`subTest` regimes, individual assertions, or scientific capabilities.

**Why it matters:** The upstream suite is `unittest`-based. One method can loop
over multiple curve types, symmetries, resolutions, labels, signs, or derivative
orders using `subTest`. Counting methods alone can mark a partially covered
parameter matrix as complete.

**Decision forced:** Use every source test method as the traceability
denominator, while recording each relevant `subTest` parameter regime and
observable in its capability record. Do not promise an “assertion count.”

### 2. “Missing” has several incompatible meanings

**What was hidden:** A behavior can have a JAX implementation but no native
oracle comparison, CPU evidence but no strict-GPU evidence, value parity but no
derivative parity, or a deliberately host-only owner.

**Why it matters:** Combining these into one number hides the engineering work
and can label host-only logging or VMEC execution as a missing GPU kernel.

**Decision forced:** Report at least:

- `jax_equivalent`
- `jax_partial`
- `jax_missing`
- `hybrid_boundary`
- `shared_python`
- `native_only`

Only `jax_partial` and `jax_missing` form the unresolved portable JAX backlog.

### 3. Native test coverage is not the same as JAX product scope

**What was hidden:** Native tests include deprecation decorators, logger
configuration, JSON/file behavior, plotting, MPI orchestration, external
solvers, mutable `Optimizable` graphs, and numerical kernels.

**Why it matters:** Porting all test mechanics literally would create GPU APIs
for responsibilities that should remain shared Python or explicit host
adapters.

**Decision forced:** Target supported public numerical behavior and required
JAX transformation contracts. Keep backend-independent Python behavior shared,
and classify external/host responsibilities explicitly.

### 4. The upstream baseline moves during the campaign

**What was hidden:** The inspected local upstream ref was already behind remote
upstream by 15 test methods.

**Why it matters:** A coverage percentage tied to a moving branch becomes stale
before implementation closes.

**Decision forced:** Pin one immutable upstream commit for each authority
campaign and run a separate drift check against moving upstream `master`.

### 5. Native output is not always a sufficient scientific oracle

**What was hidden:** Some native tests validate derivatives with Taylor or
finite-difference checks, analytic identities, symmetry, or invariants rather
than simply comparing output to `simsoptpp`.

**Why it matters:** Matching a native implementation can reproduce the same bug
or convention mistake in JAX.

**Decision forced:** Preserve independent analytic, finite-difference, Taylor,
symmetry, and conservation checks wherever the native test uses them. Native
CPU parity is necessary but not always sufficient.

### 6. One native method can mix several ownership boundaries

**What was hidden:** A single native test may construct Python objects, call a
compiled kernel, mutate shared state, serialize results, and invoke an external
runtime.

**Why it matters:** Assigning one disposition to the entire method either
overclaims JAX equivalence or discards portable numerical behavior.

**Decision forced:** Permit one native method to map to multiple capability
records. Each record owns one observable behavior and one execution boundary.

### 7. Existing JAX test volume does not prove semantic coverage

**What was hidden:** JAX has many device, transfer, regression, solver, and
example tests that do not share names with native tests.

**Why it matters:** Filename subtraction undercounts real parity, while raw JAX
test counts overstate breadth.

**Decision forced:** Reuse an existing JAX test only after reviewing its inputs,
observables, derivative order, parameter regimes, precision, and device lane.

### 8. GPU execution and GPU purity are separate facts

**What was hidden:** A test can run in a GPU process while performing some
computation in NumPy, native code, a host callback, or an adapter.

**Why it matters:** Such a test is valid hybrid evidence but not proof of a
JAX-native GPU implementation.

**Decision forced:** Require real device identity, FP64, strict transfer guards,
callback/import boundaries, and explicit hybrid labeling.

### 9. Skip behavior changes the observable denominator

**What was hidden:** Optional VMEC, SPEC, MPI, visualization, and platform
dependencies can skip behavior in ordinary collection environments.

**Why it matters:** A green suite can mean “not executed.”

**Decision forced:** Use AST inventory as the completeness owner, record
environment requirements, and collect/run external slices in declared authority
environments. Skips never count as parity evidence.

### 10. Solver equivalence requires a different acceptance contract

**What was hidden:** Native and JAX solvers can reach different but equivalent
minima, use different line searches, or report different evaluation counts.

**Why it matters:** Requiring exact iterates is usually wrong; comparing only
the final objective is too weak.

**Decision forced:** Compare initial values and derivatives exactly within
numerical tolerance, then compare final scientific success, feasibility,
objective/residual, status, and equivalent-minimum invariants. Report iteration
and evaluation differences separately.

### 11. Unit parity does not prove performance or memory safety

**What was hidden:** A numerically correct JAX port can materialize a full
Jacobian, JIT an oversized outer loop, or transfer repeatedly.

**Why it matters:** Correctness closure can still leave an unusable GPU
implementation.

**Decision forced:** Keep performance/memory as a separate representative gate:
native SIMSOPT/C++ CPU versus JAX GPU, matched inputs, compilation separated,
synchronized timing, RSS/VRAM, and scaling checks for risky kernels.

## Recommended Decisions

Unless maintainers choose otherwise, execution should use these defaults:

1. **Authority baseline:** current remote upstream commit
   `4ad6fd99189b99d9722ad33aaeb5d30adc81680f`, refreshed once more immediately
   before the inventory commit.
2. **Traceability denominator:** all 753 source test methods at that baseline.
3. **Behavior denominator:** capability records that explicitly preserve
   material `subTest` regimes and observables.
4. **Product boundary:** supported public numerical behavior, JAX
   transformation behavior, and explicit adapter contracts—not literal
   reimplementation of all native utilities.
5. **Coverage reporting:** separate disposition counts; do not publish one
   ambiguous “percent ported.”
6. **Completion:** zero `jax_partial` and `jax_missing` records among approved
   portable capabilities. Hybrid/shared/native-only remain visible rather than
   being relabeled as JAX-equivalent.
7. **Scientific authority:** native CPU plus independent invariants/derivative
   checks, JAX CPU, and strict JAX GPU where applicable.

## Still-Open Decisions

1. **Scope:** Should the target be all supported public numerical capabilities
   (recommended), or also private/internal native behavior with no intended JAX
   API?
2. **Reporting:** Should `hybrid_boundary` and `shared_python` appear as
   separately closed categories (recommended), or be included in a broader
   non-missing total?
3. **Constrained optimization:** Is a SIMSOPT-owned backend-neutral constrained
   solver in scope, or should this remain a named architectural blocker?
4. **Authority ownership:** Which JAX maintainer and native-domain maintainer
   approve non-JAX dispositions?
5. **GPU authority:** Which durable runner retains strict-GPU receipts and
   device/environment provenance?

## Rewritten Execution Prompt

> At a freshly verified immutable upstream SIMSOPT commit, inventory every
> native `unittest` test method and preserve all material `subTest` parameter
> regimes. Build a fail-closed ledger mapping each native method to
> behavior-oriented capability records classified as `jax_equivalent`,
> `jax_partial`, `jax_missing`, `hybrid_boundary`, `shared_python`, or
> `native_only`. Reuse existing JAX evidence only after verifying identical
> inputs, observables, derivative order, precision, failure semantics, and
> device lane. Report exact disposition counts before implementation. Then
> close approved portable gaps with replayable RED → GREEN → REFACTOR tests
> across native CPU, JAX CPU, and strict FP64 JAX GPU, while retaining explicit
> host/external boundaries and independent analytic or derivative oracles. Do
> not infer coverage from filenames, raw test counts, skips, example parity, or
> GPU process placement alone.
