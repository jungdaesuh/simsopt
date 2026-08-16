"""Full-precision GSCO multistep capture, native lane.

Executes the unmodified examples/3_Advanced/wireframe_gsco_multistep.py via
runpy and saves the final wf.currents plus a full-precision final objective
recomputed with the example's own post-processing formula. Repo untouched.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed")
EXAMPLE = REPO / "examples" / "3_Advanced" / "wireframe_gsco_multistep.py"
OUT = Path(sys.argv[1])
LEG = sys.argv[2]

workdir = OUT / f"{LEG}.cwd"
workdir.mkdir(parents=True, exist_ok=True)
os.chdir(workdir)

t0 = time.monotonic()
namespace = runpy.run_path(str(EXAMPLE), run_name="__main__")
wall = time.monotonic() - t0

wf = namespace["wf"]
res = namespace["res"]
currents = np.asarray(wf.currents, dtype=np.float64).ravel()
x_post = currents.reshape((-1, 1))
f_b_final = 0.5 * float(np.sum((res["Amat"] @ x_post - res["bvec"]) ** 2))

sha = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()

np.save(OUT / f"{LEG}.currents.npy", currents)
record = {
    "label": "diagnostic-not-certifying",
    "leg": LEG,
    "git_head": sha,
    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    "wall_full_script_s": round(wall, 3),
    "n_steps": int(namespace["n_step"]),
    "f_b_final_fullprec": f_b_final,
    "f_b_final_hex": np.float64(f_b_final).tobytes().hex(),
    "maximum_current": float(np.max(np.abs(np.asarray(res["x"])))),
    "nonzero_segments": int(np.count_nonzero(currents)),
}
(OUT / f"{LEG}.meta.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
