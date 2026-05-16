# Upstream Clean Merge Strategy

Date: 2026-04-24

## Context

This repository is currently far diverged from upstream SIMSOPT.

- Upstream repository: `https://github.com/hiddenSymmetries/simsopt`
- Upstream branch: `master`
- Local branch analyzed: `surrogate-confinement-v2`
- Merge base used in the analysis: `539c0f98bd46d7b32eda14dc4ad3b197abae8281`
- Local commits ahead of merge base: 338
- Upstream commits ahead of merge base: 732
- Merge probe conflict files: 14
- Merge probe conflict hunks: 16

The current local branch mixes several categories of work:

1. Upstream SIMSOPT library changes.
2. Generic CWS and tensor-surface geometry extensions.
3. Wataru finite-current banana optimization workflows.
4. Generated artifacts and run outputs.
5. Build, CI, packaging, and export-surface drift.

Those categories should not be merged as one branch. The clean strategy is to start from upstream and rebuild only intentional local functionality as isolated, reviewable patches.

## User Decisions

These choices define the strategy.

- Work product: plan only.
- Finite-current Boozer approach: do not patch core `BoozerSurface`; use an adapter with `alpha = G + iota * I`.
- CWS and tensor-surface work: move all CWS and banana-specific code outside core.
- Wataru workflow location: project-layer package outside upstream SIMSOPT core.
- Artifact and build drift policy: drop generated artifacts and avoid build/CI drift unless strictly required.

## Design Principles

- Keep upstream SIMSOPT as the library engine.
- Keep Wataru/banana optimization as project code that uses SIMSOPT.
- Use SSOT for finite-current conventions.
- Avoid duplicate current handling in Stage 2 and single-stage workflows.
- Avoid patching core SIMSOPT for project-specific physics contracts.
- Keep generated outputs out of source control.
- Keep patches small enough to review and revert independently.

## Target Layout

```text
src/simsopt/
  Upstream-clean SIMSOPT core.

src/simsopt_surrogate/
  Project-specific code.

src/simsopt_surrogate/finite_current.py
  SSOT for finite-current Boozer convention:
  I = mu0 * plasma_current_A
  alpha = G + iota * I

src/simsopt_surrogate/wataru/
  Wataru finite-current workflow orchestration.

src/simsopt_surrogate/banana_opt/
  Banana optimization contracts, scoring, replay, and search logic.

src/simsopt_surrogate/cws/
  CWS/banana-specific surface-curve logic if not accepted as generic SIMSOPT core.

tests/simsopt_surrogate/
  Project-layer tests with small fixtures only.

runs/
  Ignored generated run outputs.

artifacts/
  Ignored generated reports and plots, unless explicitly promoted as small fixtures.
```

## Finite-Current Contract

Wataru finite-current workflows use:

```text
I = mu0 * plasma_current_A
```

with no extra `2*pi` factor.

The current local core changed the Boozer residual from:

```text
G * B - |B|^2 * (x_phi + iota * x_theta)
```

to:

```text
(G + iota * I) * B - |B|^2 * (x_phi + iota * x_theta)
```

The clean upstream-compatible implementation should not add `I` to core `BoozerSurface`. Instead, project code should pass:

```text
alpha = G + iota * I
```

as the upstream `G` argument.

Project code must name this value `alpha` or `boozer_alpha`, not physical `G`.

```python
MU_0 = 4 * np.pi * 1e-7
boozer_I = MU_0 * plasma_current_A
boozer_alpha = physical_G + iota * boozer_I

res = boozer_surface.run_code(iota, G=boozer_alpha)

physical_G = res["G"] - res["iota"] * boozer_I
```

This gives the finite-current residual using upstream SIMSOPT's vacuum residual machinery while keeping the physical current convention in the project layer.

## Patch Stack

### 1. Upstream Base

Create a clean integration branch from upstream:

```bash
git fetch upstream_check master
git switch -c surrogate-clean-upstream upstream_check/master
```

Do not merge the current mixed branch wholesale.

### 2. Project Package Skeleton

Add only the local project package:

```text
src/simsopt_surrogate/
tests/simsopt_surrogate/
```

No changes to `src/simsopt` in this patch.

### 3. Finite-Current Adapter

Add one module that owns:

- `plasma_current_A -> boozer_I`
- `physical_G, iota, boozer_I -> boozer_alpha`
- `boozer_alpha, iota, boozer_I -> physical_G`

Acceptance checks:

- `plasma_current_A = 0` gives `boozer_alpha == physical_G`.
- Nonzero current uses `boozer_alpha = physical_G + iota * boozer_I`.
- Wataru Stage 2 and single-stage use the same conversion function.

### 4. Wataru Workflow Project Layer

Move workflow behavior out of `examples/` and into project modules.

Thin scripts can remain as entrypoints, but they should not contain the current convention or duplicated objective construction.

Required project modules:

- current contract module
- Stage 2 field/coil construction module
- single-stage Boozer/objective construction module
- artifact path and run-output policy module

### 5. CWS and Banana Geometry

Since the selected strategy is to move all CWS/banana-specific code outside core, do not reapply CWS classes directly into `src/simsopt` as part of the clean merge.

If later work proves that CWS requires generic core surface derivative APIs, split that into a separate proposal:

```text
feat: add arbitrary-point surface derivative evaluation
```

That patch must be generic, not Wataru-specific, and should be limited to `src/simsoptpp` surface APIs and focused tests.

### 6. Tests

Add project-layer tests for:

- zero-current equivalence to upstream vacuum Boozer behavior
- nonzero-current alpha mapping
- Wataru Stage 2 to single-stage current consistency
- no duplicate current convention in workflow modules
- generated outputs are not required for unit tests

Use tiny fixtures only. Do not commit production run outputs as fixtures.

### 7. Drop Artifacts

Do not reapply generated files from:

```text
examples/3_Advanced/optimization_cws_*/
artifacts/comparisons/
*.vtu
*.vts
large *.nc run files
generated biot_savart_opt*.json
jac_log*.dat
VMEC run-output directories
```

If an equilibrium file is needed for tests, create or select a minimal fixture and document why it is committed.

## Conflict Policy

Use upstream as the base for:

```text
pyproject.toml
CMakeLists.txt
ci/test.yml
src/simsopt/**
src/simsoptpp/**
```

Then reapply local behavior only in the project package unless a generic library patch is separately justified.

### Known Conflict Files

The prior merge probe found conflicts in:

```text
docs/source/installation.rst
pyproject.toml
src/simsopt/configs/__init__.py
src/simsopt/field/__init__.py
src/simsopt/field/coil.py
src/simsopt/geo/boozersurface.py
src/simsopt/geo/curve.py
src/simsopt/geo/curveobjectives.py
src/simsopt/util/__init__.py
tests/field/test_biotsavart.py
tests/field/test_fieldline.py
tests/geo/test_boozersurface.py
tests/geo/test_curve.py
tests/geo/test_surface_objectives.py
```

Resolution stance:

- Prefer upstream for packaging, public exports, and generic library behavior.
- Do not keep local `BoozerSurface(..., I=...)` in core.
- Do not keep local broad `__init__.py` rewrites unless a specific import failure remains on the clean branch.
- Move Wataru finite-current behavior into the project adapter.
- Move CWS/banana-specific code outside core under the selected strategy.
- Port tests only after their corresponding project-layer code exists.

## What Not To Do

- Do not run `git merge upstream_check/master` on the current branch and hand-resolve all conflicts in place.
- Do not take all local conflict hunks.
- Do not take all upstream conflict hunks without preserving Wataru finite-current behavior.
- Do not keep generated run outputs in the clean source tree.
- Do not mix build/CI changes with physics changes.
- Do not duplicate the current convention in multiple scripts.

## Acceptance Criteria

The clean merge plan is successful when:

- Upstream SIMSOPT core remains close to upstream.
- Wataru finite-current behavior is preserved through the alpha adapter.
- `PROXY_CURRENT_KA = 0` is equivalent to upstream vacuum Boozer semantics.
- Nonzero current uses `alpha = G + iota * I`.
- `I = mu0 * plasma_current_A` is defined in exactly one project-layer place.
- CWS and banana-specific workflow code do not live in `src/simsopt`.
- No generated artifacts are committed.
- Tests are organized by feature and use small fixtures only.
- Each patch in the stack can be reviewed independently.

## Recommended Next Step

Create the upstream-based branch and implement only the project package skeleton plus finite-current adapter first. Do not port CWS or workflow code until the finite-current SSOT tests pass.
