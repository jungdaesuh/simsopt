"""Chartered discriminator: native vs JAX outer gradient, component-wise."""
import json, sys
import numpy as np
sys.path.insert(0, "/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed")
sys.path.insert(0, "/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/src")
from benchmarks.nested_ls_outer_native_child import (
    InnerWarmStart, NativeOuterRun, build_native_outer_objective,
    build_vessel_surface, load_lane_vessel_coordinates, load_outer_optimizer_policy,
)
from simsopt_jax_adapters.geo.flat675.manifest import load_flat675_input_manifest, load_flat675_vessel_template
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE, DEFAULT_F3_B37_NATIVE_LANE, DEFAULT_FLAT675_BUNDLE_ROOT,
    load_archived_nested_ls_pair, load_flat675_lane_blocks,
)
coils, surface_block, lane_meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
seed_iota, seed_g = (float(v) for v in lane_meta["inner_state"])
native, jax_boozer, _t = load_archived_nested_ls_pair(coil_coordinates=coils, surface_coordinates=surface_block)
del _t, jax_boozer
manifest = load_flat675_input_manifest(DEFAULT_FLAT675_BUNDLE_ROOT)
vessel = build_vessel_surface(load_flat675_vessel_template(DEFAULT_FLAT675_BUNDLE_ROOT),
                              load_lane_vessel_coordinates(DEFAULT_F3_B37_GPU_LANE))
objective = build_native_outer_objective(native, objective_policy=manifest.objective_policy, vessel=vessel)
policy = load_outer_optimizer_policy(DEFAULT_F3_B37_NATIVE_LANE, budget=1, maxcor=10)
start = np.asarray(native.biotsavart.x, dtype=np.float64)
run = NativeOuterRun(objective, seed=InnerWarmStart(
    surface_dofs=np.asarray(native.surface.get_dofs(), dtype=np.float64), iota=seed_iota, G=seed_g),
    rejection_value_offset=policy.rejection_value_offset,
    rejection_distance_scale=policy.rejection_distance_scale)
value, grad = run(start)
red = json.load(open("/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/docs/receipts/evidence/nested_ls_outer_fd0_20260823.amendment2-red.json"))
gj = np.asarray(red["probe"]["outer_gradient"], dtype=np.float64)
gn = np.asarray(grad, dtype=np.float64)
print("native J:", repr(value))
print("idx | native | jax | abs diff | rel diff")
for i in range(11):
    d = abs(gn[i]-gj[i]); r = d/max(abs(gn[i]), 1e-30)
    print(f"{i} | {gn[i]:+.9f} | {gj[i]:+.9f} | {d:.3e} | {r:.3e}")
print("max rel diff:", max(abs(gn[i]-gj[i])/max(abs(gn[i]),1e-30) for i in range(11)))
