# Grouped Field Distribution Probe (2026-05-18)

## Fixture

- File: `benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json`
- Lane: local CPU probe with `jax_enable_x64=True`
- Evaluated helper: `simsopt.jax_core.biotsavart.grouped_biot_savart_B`

## Group Distribution

| group | gammas shape | gammadashs shape | currents shape | coil share |
| --- | --- | --- | --- | --- |
| TF | `(20, 15, 3)` | `(20, 15, 3)` | `(20,)` | 20 / 30 |
| banana | `(10, 128, 3)` | `(10, 128, 3)` | `(10,)` | 10 / 30 |

The largest group covers 66.7% of coils, below the N25 80% threshold for a
homogeneous common path. This fixture is the heterogeneous regime: padding the
15-point TF coils up to the 128-point banana quadrature would add avoidable
work.

## Decision

Keep the per-group accumulation model. The source change makes the legacy
`_grouped_field` helper a JIT boundary keyed by the static field function and
group count, so heterogeneous fixtures reuse a stable compiled shape without
padding groups together.

## Local Timing Probe

The timing probe compared the old-equivalent Python group loop against the new
JIT-keyed grouped helper on 64 synthetic evaluation points using the fixture
coil arrays.

| path | median steady-state time |
| --- | ---: |
| manual Python group loop | 0.000482 s |
| JIT-keyed grouped helper | 0.000383 s |

Observed local CPU ratio: 1.259x manual-loop time over JIT-keyed helper time.

## Command

```bash
PYTHONPATH=/Users/suhjungdae/code/columbia/simsopt-jax/src .conda/jax/bin/python - <<'PY'
import json
import statistics
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from simsopt.jax_core.biotsavart import biot_savart_B, grouped_biot_savart_B

fixture = Path("benchmarks/fixtures/single_stage_seed_iota15/single_stage_jax_runtime_spec.json")
payload = json.loads(fixture.read_text())
groups = payload["field"]["coil_set"]["fields"]["groups"]["items"]
coil_arrays = []
for group in groups:
    fields = group["fields"]
    coil_arrays.append(
        (
            jnp.asarray(fields["gammas"]["data"], dtype=jnp.float64),
            jnp.asarray(fields["gammadashs"]["data"], dtype=jnp.float64),
            jnp.asarray(fields["currents"]["data"], dtype=jnp.float64),
        )
    )
coil_arrays = tuple(coil_arrays)

rng = np.random.default_rng(25)
points = jnp.asarray(rng.normal(size=(64, 3)), dtype=jnp.float64)
points = points.at[:, 0].add(1.2)

def manual_group_loop(eval_points, arrays):
    gammas, gammadashs, currents = arrays[0]
    result = biot_savart_B(eval_points, gammas, gammadashs, currents)
    for gammas, gammadashs, currents in arrays[1:]:
        result = result + biot_savart_B(eval_points, gammas, gammadashs, currents)
    return result

def bench(fn, *, warmups=2, reps=7):
    for _ in range(warmups):
        fn(points, coil_arrays).block_until_ready()
    times = []
    for _ in range(reps):
        start = time.perf_counter()
        fn(points, coil_arrays).block_until_ready()
        times.append(time.perf_counter() - start)
    return times

manual = bench(manual_group_loop)
jitted = bench(grouped_biot_savart_B)
np.testing.assert_allclose(
    np.asarray(manual_group_loop(points, coil_arrays)),
    np.asarray(grouped_biot_savart_B(points, coil_arrays)),
    rtol=1e-12,
    atol=1e-14,
)
print("manual_median_s", f"{statistics.median(manual):.6f}")
print("jitted_median_s", f"{statistics.median(jitted):.6f}")
print(
    "speed_ratio_manual_over_jitted",
    f"{statistics.median(manual) / statistics.median(jitted):.3f}",
)
PY
```
