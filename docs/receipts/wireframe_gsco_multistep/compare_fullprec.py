"""Full-precision comparison for the GSCO multistep parity captures.

Usage: python compare_fullprec.py <native.currents.npy> <jax.currents.npy>
Writes the comparison verdict as JSON to stdout.
"""

from __future__ import annotations

import json
import sys

import numpy as np

native = np.load(sys.argv[1])
jax_lane = np.load(sys.argv[2])
assert native.shape == jax_lane.shape, (native.shape, jax_lane.shape)
diff = np.abs(native - jax_lane)
support = native != 0
relative = np.zeros_like(native)
relative[support] = diff[support] / np.abs(native[support])
print(
    json.dumps(
        {
            "shape": list(native.shape),
            "support_identical": bool(np.array_equal(native != 0, jax_lane != 0)),
            "bitwise_identical": bool(np.array_equal(native, jax_lane)),
            "allclose_rtol1e-6_atol1e-7": bool(
                np.allclose(jax_lane, native, rtol=1e-6, atol=1e-7)
            ),
            "max_abs_diff_A": float(diff.max()),
            "max_rel_diff_on_support": float(relative.max()) if support.any() else None,
            "n_differing_entries": int(np.count_nonzero(native != jax_lane)),
            "native_nonzero": int(np.count_nonzero(native)),
            "jax_nonzero": int(np.count_nonzero(jax_lane)),
            "unique_currents_native": [float(v) for v in np.unique(native[support])],
            "unique_currents_jax": [
                float(v) for v in np.unique(jax_lane[jax_lane != 0])
            ],
        },
        indent=2,
    )
)
