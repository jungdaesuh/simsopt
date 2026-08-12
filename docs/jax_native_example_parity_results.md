# Native SIMSOPT vs JAX Example Parity Results

This file is generated from one aggregate parity `summary.json`; do not edit the numerical table manually.

- Run ID: `20260726T225943Z-09dfdc3e`
- Verdict: **pass**
- Evidence class: **authoritative** (clean checkout)
- Repository commit: `799c656e186642bdb7e296e46ad1c6cd61277839`
- Lanes: `native-cpu`, `jax-cpu`, `jax-gpu`
- Aggregate artifact: `.artifacts/jax-example-parity/20260726T225943Z-09dfdc3e`

This authoritative run is source-bound to the named clean committed checkout and may promote only the classifications and bounded scale reported below.

| JAX example | Native source | Classification | Scale | Oracle | Verdict | Comparisons |
|---|---|---|---|---|---:|---:|
| traceable-least-squares | `1_Simple/just_a_quadratic.py` | full | bounded | native_python_scipy | pass | 24/24 |
| curve-length-optimization | `1_Simple/minimize_curve_length.py` | full | bounded | native_python_scipy | pass | 15/15 |
| surface-geometry-optimization | `1_Simple/surf_vol_area.py` | reduced | bounded | native_python_scipy | pass | 36/36 |
| coil-flux-optimization | `1_Simple/stage_two_optimization_minimal.py` | reduced | bounded | native_python_scipy | pass | 24/24 |
| qfm-surface-optimization | `1_Simple/qfm.py` | reduced | bounded | native_python_scipy | pass | 42/42 |
| permanent-magnet-optimization | `1_Simple/permanent_magnet_simple.py` | reduced | bounded | native_simsoptpp | pass | 21/21 |
| wireframe-optimization | `2_Intermediate/wireframe_rcls_basic.py` | reduced | bounded | native_python_scipy | pass | 30/30 |
| coil-force-and-finite-build | `2_Intermediate/strain_optimization.py` | reduced | bounded | native_simsoptpp | pass | 36/36 |

`full` describes declared scientific workflow-stage coverage, while `bounded` describes execution scale. `reduced` evidence must not be reported as full native-example equivalence. Oracle kinds distinguish native Python/SciPy, loaded `simsoptpp`, analytic, and external references.
