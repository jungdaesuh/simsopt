"""Full-precision GSCO multistep capture, JAX lane.

Replicates the exact body of examples/jax/3_Advanced/wireframe_gsco_multistep.py
solve() at native_default (module functions, same arguments), then saves the
final segment-currents vector and full-precision stage objectives. The repo is
imported read-only; nothing is modified.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed")
EXAMPLE = REPO / "examples" / "jax" / "3_Advanced" / "wireframe_gsco_multistep.py"
OUT = Path(sys.argv[1])
LEG = sys.argv[2]

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

spec = importlib.util.spec_from_file_location("gsco_jax_example", EXAMPLE)
module = importlib.util.module_from_spec(spec)
sys.modules["gsco_jax_example"] = module
spec.loader.exec_module(module)

scale = "native_default"
max_steps = module.NATIVE_ITERATIONS

t_solve0 = time.monotonic()
wireframe, plasma, external_field, poloidal_current = module._build_problem(scale)
response, target = module.bnorm_obj_matrices_jax(
    wireframe,
    plasma,
    ext_field=external_field,
    area_weighted=True,
    verbose=False,
)
loops = np.asarray(wireframe.get_cell_key(), dtype=np.int32)
free_loops = np.asarray(wireframe.get_free_cells(form="logical"), dtype=np.int32)
segments = np.asarray(wireframe.segments, dtype=np.int32)
connections = np.asarray(wireframe.connected_segments, dtype=np.int32)
base_constrained = np.zeros(wireframe.n_segments, dtype=np.bool_)
base_constrained[np.asarray(wireframe.constrained_segments(), dtype=np.intp)] = True
initial_fraction = 0.2
current_scale = abs(poloidal_current)
initial_default_current = initial_fraction * current_scale
response_device = jax.device_put(np.asarray(response, dtype=np.float64))
target_device = jax.device_put(np.asarray(target, dtype=np.float64))
loops_device = jax.device_put(loops)
neighbors_device = jax.device_put(
    np.asarray(wireframe.get_cell_neighbors(), dtype=np.int32)
)
params = module.WireframeGSCOLiveParams(
    A=response_device,
    loops=loops_device,
    free_loops=jax.device_put(free_loops),
    segments=jax.device_put(segments),
    connections=jax.device_put(connections),
    default_current=jax.device_put(
        np.asarray(initial_default_current, dtype=np.float64)
    ),
    max_current=jax.device_put(
        np.asarray(1.1 * initial_default_current, dtype=np.float64)
    ),
    lambda_s=jax.device_put(np.asarray(1.0e-7, dtype=np.float64)),
    tol=jax.device_put(np.asarray(0.001 * initial_default_current, dtype=np.float64)),
    max_loop_count=1,
    no_crossing=True,
    no_new_coils=False,
    match_current=False,
)
initial_currents = jax.device_put(np.zeros((wireframe.n_segments,), dtype=np.float64))
result = module.wireframe_gsco_multistep_loop_jax(
    params,
    target_device,
    initial_currents,
    jax.device_put(np.zeros((loops.shape[0],), dtype=np.int32)),
    loops_device,
    neighbors_device,
    jax.device_put(base_constrained),
    max_iter_per_step=max_steps,
    max_outer_steps=12,
    initial_current_fraction=initial_fraction,
    current_scale=current_scale,
    min_coil_size=20,
    final_max_current=1.1 * initial_default_current,
)
stage_count = int(jax.device_get(result.stage_count))
all_stage_objectives = np.asarray(
    jax.device_get(result.stage_objectives), dtype=np.float64
)
stage_objectives = all_stage_objectives[:stage_count]
solution = np.asarray(jax.device_get(result.x), dtype=np.float64).ravel()
initial_normal_error = float(jax.device_get(jnp.linalg.norm(target_device)))
final_normal_error = float(np.sqrt(2.0 * stage_objectives[-1]))
maximum_current = float(np.max(np.abs(solution)))
nonfinal_steps = int(jax.device_get(result.nonfinal_steps))
t_solve = time.monotonic() - t_solve0

# Full-precision final objective recomputed from the host copies of A, b, x —
# same formula as the native example's f_B_post.
A_host = np.asarray(response, dtype=np.float64)
b_host = np.asarray(target, dtype=np.float64).reshape(-1)
f_b_final = 0.5 * float(np.sum((A_host @ solution - b_host) ** 2))

sha = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
dirty = subprocess.run(
    ["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True
).stdout.strip()

np.save(OUT / f"{LEG}.currents.npy", solution)
np.save(OUT / f"{LEG}.stage_objectives.npy", stage_objectives)
record = {
    "label": "diagnostic-not-certifying",
    "leg": LEG,
    "device": str(jax.devices()[0]),
    "git_head": sha,
    "git_dirty_files": dirty.splitlines(),
    "wall_build_plus_solve_s": round(t_solve, 3),
    "stage_count": stage_count,
    "nonfinal_steps": nonfinal_steps,
    "iterations": nonfinal_steps * max_steps,
    "stage_objectives_hex": [np.float64(v).tobytes().hex() for v in stage_objectives],
    "stage_objectives": [float(v) for v in stage_objectives],
    "initial_normal_error": initial_normal_error,
    "final_normal_error": final_normal_error,
    "maximum_current": maximum_current,
    "f_b_final_fullprec": f_b_final,
    "f_b_final_hex": np.float64(f_b_final).tobytes().hex(),
    "nonzero_segments": int(np.count_nonzero(solution)),
}
(OUT / f"{LEG}.meta.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
