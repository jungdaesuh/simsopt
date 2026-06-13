# Single-stage 11-vs-51 production test matrix

- Matrix id: `single_stage_11_51_2026-06-13`
- Source SHA: `42cbc2e964fd378251f383d4c6616e15c827de89`
- Seed: `benchmarks/fixtures/single_stage_seed_iota15/biot_savart_opt.json`
- Budgets: outer L-BFGS 1500, Boozer BFGS 1500, Boozer Newton 50, polish `run`
- Inner-LS naming: quasinewton=`quasi-newton`, LM=`lm`, LS=`lm-minpack`

## Formulation/backend coupling

- `11` reduced (coils only, surface solved) ⇒ `scipy-jax` (host SciPy).
- `51` full-space (coils+surface free, residual penalty) ⇒ `ondevice` (on-device JAX).
- The cpp/CPU reference is always 51; the 11-dim JAX lane is compared against a dim-mismatched 51 reference.

## Cells

| id | dim | backend | platform | inner-LS | tier | status | ref dim match | runnable now |
|---|---|---|---|---|---|---|---|---|
| `ss_11_scipyjax_cpu_quasinewton_mpol2` | 11 | scipy-jax | cpu | quasinewton=`quasi-newton` | mpol2 | core | NO (51 vs 11) | yes |
| `ss_11_scipyjax_cpu_quasinewton_mpol10` | 11 | scipy-jax | cpu | quasinewton=`quasi-newton` | mpol10 | core | NO (51 vs 11) | no (donor) |
| `ss_11_scipyjax_gpu_quasinewton_mpol2` | 11 | scipy-jax | gpu | quasinewton=`quasi-newton` | mpol2 | core | NO (51 vs 11) | yes |
| `ss_11_scipyjax_gpu_quasinewton_mpol10` | 11 | scipy-jax | gpu | quasinewton=`quasi-newton` | mpol10 | core | NO (51 vs 11) | no (donor) |
| `ss_11_scipyjax_cpu_LM_mpol2` | 11 | scipy-jax | cpu | LM=`lm` | mpol2 | extended | NO (51 vs 11) | yes |
| `ss_11_scipyjax_cpu_LM_mpol10` | 11 | scipy-jax | cpu | LM=`lm` | mpol10 | extended | NO (51 vs 11) | no (donor) |
| `ss_11_scipyjax_gpu_LM_mpol2` | 11 | scipy-jax | gpu | LM=`lm` | mpol2 | extended | NO (51 vs 11) | yes |
| `ss_11_scipyjax_gpu_LM_mpol10` | 11 | scipy-jax | gpu | LM=`lm` | mpol10 | extended | NO (51 vs 11) | no (donor) |
| `ss_11_scipyjax_cpu_LS_mpol2` | 11 | scipy-jax | cpu | LS=`lm-minpack` | mpol2 | extended | NO (51 vs 11) | yes |
| `ss_11_scipyjax_cpu_LS_mpol10` | 11 | scipy-jax | cpu | LS=`lm-minpack` | mpol10 | extended | NO (51 vs 11) | no (donor) |
| `ss_11_scipyjax_gpu_LS_mpol2` | 11 | scipy-jax | gpu | LS=`lm-minpack` | mpol2 | extended | NO (51 vs 11) | yes |
| `ss_11_scipyjax_gpu_LS_mpol10` | 11 | scipy-jax | gpu | LS=`lm-minpack` | mpol10 | extended | NO (51 vs 11) | no (donor) |
| `ss_51_ondevice_cpu_quasinewton_mpol2` | 51 | ondevice | cpu | quasinewton=`quasi-newton` | mpol2 | core | yes | yes |
| `ss_51_ondevice_cpu_quasinewton_mpol10` | 51 | ondevice | cpu | quasinewton=`quasi-newton` | mpol10 | core | yes | no (donor) |
| `ss_51_ondevice_gpu_quasinewton_mpol2` | 51 | ondevice | gpu | quasinewton=`quasi-newton` | mpol2 | core | yes | yes |
| `ss_51_ondevice_gpu_quasinewton_mpol10` | 51 | ondevice | gpu | quasinewton=`quasi-newton` | mpol10 | core | yes | no (donor) |
| `ss_51_ondevice_cpu_LM_mpol2` | 51 | ondevice | cpu | LM=`lm` | mpol2 | core | yes | yes |
| `ss_51_ondevice_cpu_LM_mpol10` | 51 | ondevice | cpu | LM=`lm` | mpol10 | core | yes | no (donor) |
| `ss_51_ondevice_gpu_LM_mpol2` | 51 | ondevice | gpu | LM=`lm` | mpol2 | core | yes | yes |
| `ss_51_ondevice_gpu_LM_mpol10` | 51 | ondevice | gpu | LM=`lm` | mpol10 | core | yes | no (donor) |
| `ss_51_ondevice_cpu_LS_mpol2` | 51 | ondevice | cpu | LS=`lm-minpack` | mpol2 | core | yes | yes |
| `ss_51_ondevice_cpu_LS_mpol10` | 51 | ondevice | cpu | LS=`lm-minpack` | mpol10 | core | yes | no (donor) |
| `ss_51_ondevice_gpu_LS_mpol2` | 51 | ondevice | gpu | LS=`lm-minpack` | mpol2 | core | yes | yes |
| `ss_51_ondevice_gpu_LS_mpol10` | 51 | ondevice | gpu | LS=`lm-minpack` | mpol10 | core | yes | no (donor) |

Total cells: 24 (16 core, 8 extended).

## Notes

- 11=scipy-jax (reduced), 51=ondevice (full); coupling is not overridable by any flag.
- Every run also yields the cpp/CPU reference at 51 full-space; there is no coil-only native reference in this harness, so the 11-dim JAX lane has a dim-mismatched cpp reference.
- optax-lbfgs and optimistix-lbfgs are intentionally excluded.
- lm/lm-minpack export BOOZER_LEAST_SQUARES_ALGORITHM to BOTH the target and reference children; the cpp reference uses the native C++ Boozer solver and is expected to ignore it, but the first lm/lm-minpack run must confirm the reference child tolerates it.

## Tier detail

- `mpol2`: mpol=2 ntor=2 nphi=31 ntheta=16, binds_budget=False, runnable_now=True. smoke resolution: loose tolerances (gtol 1e-2), budgets do not bind, optimizers stop in ~5-219 steps; cheap diagnostics tier
- `mpol10`: mpol=10 ntor=10 nphi=64 ntheta=32, binds_budget=True, runnable_now=False. production resolution: tolerances tighten (gtol 1e-7), 1500/50 budgets bind; requires --warm-start-run-dir from the donor build (2->4->6->8->10 ladder); smaller than mpol12 so the ondevice-51 dense-Newton graph is more likely to fit, though it may still need polish-policy skip
