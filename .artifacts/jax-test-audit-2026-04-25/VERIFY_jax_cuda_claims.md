# VERIFY_jax_cuda_claims.md

Date: 2026-04-25
Working directory: `/Users/suhjungdae/code/columbia/simsopt-jax`
Probed environment: `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python` with `PYTHONNOUSERSITE=1`.

---

## Claim 1 — jaxlib 0.9.2 has NO stable CUDA-version attrs

### Raw probe output (verbatim)

Command:
```
PYTHONNOUSERSITE=1 .conda/jax-0.9.2/bin/python -c "<probe script from prompt>"
```

Output:
```
jaxlib version: 0.9.2
has cuda_versions: False
jaxlib attrs containing cuda: []
jax version: 0.9.2
jax attrs containing cuda: []
jax.lib attrs containing cuda: []
no jax_cuda12_pjrt: No module named 'jax_cuda12_pjrt'
no jaxlib.cuda: No module named 'jaxlib.cuda'
runtime version (if any):
   xla_client.runtime_version: absent
```

Direct evaluation of the audit's earlier suggestion:
```
jaxlib.cuda_versions on 0.9.2: None
would the audit field be useless?: True
```

i.e. `getattr(jaxlib, "cuda_versions", None)` evaluates to `None` on the
0.9.2 lane. Including `jaxlib_cuda_versions` as a required provenance
field on this lane records nothing useful.

### What jax 0.9.2 DOES expose (CUDA-relevant)

Probe (CPU-only macOS host, no nvidia-smi available; collected on the same
0.9.2 interpreter):
```
jax.devices(): [CpuDevice(id=0)]
  device: TFRT_CPU_0 | str: TFRT_CPU_0 | platform: cpu | device_kind: cpu
jax.default_backend(): cpu
```

- `jax.devices()` returns a list whose `repr` is e.g. `[CpuDevice(id=0)]`
  on CPU. On a CUDA host the canonical repr is `[CudaDevice(id=0), ...]`
  with `platform == "cuda"` and `device_kind` reporting the GPU model
  (e.g. `"NVIDIA H100 80GB HBM3"`). The platform string is the SSOT
  identifier for the active backend.
- `jax.default_backend()` returns the canonical backend label
  (`"cpu"`, `"gpu"`, `"tpu"`). On a CUDA host with `JAX_PLATFORMS=cuda`
  it returns `"gpu"`.
- There is no `nvcc_version` / `cuda_version` exposed by jax/jaxlib 0.9.2,
  no `jax_cuda12_pjrt`, no `jaxlib.cuda` submodule, and
  `jaxlib.xla_client.runtime_version` is absent on this build. (Newer
  jaxlib releases shipped via the pip plugin model do expose
  `jaxlib.cuda_versions`, which is precisely why the audit assumed it
  was available — it is not on the 0.9.2 conda build we ship.)
- `nvidia-smi --query-gpu=driver_version,name --format=csv` is the
  appropriate cross-source for CUDA driver + GPU SKU on the production
  GPU host. It is **not** available on the local macOS dev machine
  (`nvidia-smi not found`), which is itself a useful signal: the audit
  field cannot be filled by the Python process alone — it must be
  captured by the launcher on the GPU host.

### Verdict: CORRECT

jaxlib 0.9.2 does not expose a stable `cuda_versions` attribute (or any
other CUDA-version attribute reachable from `jaxlib`/`jax`/`jax.lib`).
The only runtime-visible CUDA-relevant facts on 0.9.2 are the device
list + platform string surfaced by `jax.devices()` /
`jax.default_backend()`.

### Implication for the corrected plan

- **Drop** `jaxlib_cuda_versions` from required provenance fields on the
  0.9.2 lane. On 0.9.2 it will always serialize as `None`, which is
  noise.
- **Keep** the SSOT provenance fields already produced by
  `build_provenance`: `jax`, `jaxlib`, `backend`, `devices`,
  `backend_mode`, `backend_strict`, transfer-guard state, compilation
  cache, sharding metadata.
- **Add** explicit CUDA-environment fields captured by the **launcher**
  (not by Python at runtime): `nvidia_smi_driver_version`,
  `nvidia_smi_gpu_name`, `cuda_visible_devices`, `cuda_force_ptx_jit`,
  `cuda_disable_ptx_jit`, plus `jax.default_backend()` and the resolved
  `jax.devices()` repr.
- If/when we move to a newer jaxlib that ships `jaxlib.cuda_versions`,
  reintroduce the field as **optional/best-effort**, never required.

---

## Claim 2 — `CUDA_FORCE_PTX_JIT` and `CUDA_DISABLE_PTX_JIT` are the meaningful PTX/cubin knobs

### NVIDIA documentation (verbatim, via WebFetch +  WebSearch on docs.nvidia.com)

Source: NVIDIA CUDA Programming Guide, Section 5.2 "CUDA Environment
Variables":
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html

`CUDA_FORCE_PTX_JIT`:
> "These variables instruct the CUDA driver to ignore any CUBIN embedded
> in an application and perform Just-In-Time (JIT) compilation of the
> embedded PTX code instead. […] Forcing JIT compilation increases an
> application's load time during initial execution. This variable is
> useful for testing compatibility and validating that PTX code is
> embedded in an application and that its Just-In-Time compilation is
> functioning properly, which ensures forward compatibility with future
> architectures."

Documented values: `1` (force PTX JIT), `0` (default).

`CUDA_DISABLE_PTX_JIT`:
> "The environment variables disable the Just-In-Time (JIT) compilation
> of embedded PTX code and use the compatible CUBIN embedded in an
> application. A kernel will fail to load if it does not have embedded
> binary code, or if the embedded binary was compiled for an
> incompatible architecture. These environment variables can be used to
> validate that an application has compatible CUBIN code generated for
> each kernel."

Documented values: `1` (disable PTX JIT), `0` (default).

`CUDA_VISIBLE_DEVICES`:
> "The environment variable controls which GPU devices are visible to a
> CUDA application and in what order they are enumerated."
> Documented values: comma-separated GPU indices/UUIDs/MIG instances;
> unset → all GPUs visible; empty string → no GPUs visible.

### Verdict: CORRECT

- Both `CUDA_FORCE_PTX_JIT` and `CUDA_DISABLE_PTX_JIT` exist and are
  documented by NVIDIA in the official CUDA Programming Guide.
- They directly control PTX→cubin (SASS) JIT behavior, which is the
  exact knob relevant to cubin-vs-driver mismatch incidents (e.g. the
  Runpod `jaxlib cubin v12.9 vs system nvlink` episode logged in
  `project_runpod_cuda_block.md`).
- `CUDA_VISIBLE_DEVICES` controls **device visibility/enumeration**, not
  PTX/cubin compatibility. Recording only `CUDA_VISIBLE_DEVICES` does
  not prove a kernel was loaded from cubin vs JIT-compiled from PTX, nor
  whether PTX JIT is enabled/disabled.

### Implication for the corrected plan

- The proof bundle / launcher provenance must record **all three**:
  `CUDA_VISIBLE_DEVICES`, `CUDA_FORCE_PTX_JIT`, `CUDA_DISABLE_PTX_JIT`.
- For a true PTX/cubin compatibility proof on a given driver:
  - One canary run with `CUDA_DISABLE_PTX_JIT=1` confirms that
    embedded CUBIN is sufficient for the active GPU/driver pair (no
    silent PTX JIT fallback). This is the "production cubin path"
    proof.
  - One canary run with `CUDA_FORCE_PTX_JIT=1` confirms that the PTX
    fallback path works (forward compatibility validation).
- Recording `CUDA_VISIBLE_DEVICES` alone is **necessary but not
  sufficient** for proving cubin/PTX compatibility.

---

## Claim 3 — JAX donation: input becomes invalid/deleted after the call

### JAX documentation (verbatim, via WebFetch)

Source: https://docs.jax.dev/en/latest/buffer_donation.html

> "you need to guarantee to XLA that you will not use the donated input
> buffer after calling the donating function."

Demonstrated failure mode (verbatim from the docs):
> "RuntimeError: Invalid argument: CopyToHostAsync() called on invalid
> buffer"

Buffer-aliasing language (verbatim):
> "if it matches the shape and element type of one of the outputs, you
> can specify that you want the corresponding input buffer to be donated
> to hold an output."
> "The result has the same shape and type as `y`, so it will share its
> buffer."

i.e. aliasing happens **conditionally on shape/dtype match**, and is
described as a hint to XLA, not as a guaranteed user-visible buffer
identity contract.

Source: https://docs.jax.dev/en/latest/_autosummary/jax.jit.html

`donate_argnums` (verbatim):
> "optional, collection of integers to specify which positional argument
> buffers can be overwritten by the computation and marked deleted in
> the caller."

> "You should not reuse buffers that you donate to a computation; JAX
> will raise an error if you try to."

> "In some cases XLA can make use of donated buffers to reduce the
> amount of memory needed to perform a computation, for example
> recycling one of your input buffers to store a result."

So `donate_argnums` is documented as taking **integers indexing positional
arguments** (i.e. positional indices, even though the keyword is
keyword-only on `jit`).

### Verdict: CORRECT

The JAX documentation explicitly says:
1. donated buffers can be **overwritten and marked deleted in the caller**;
2. reusing a donated buffer raises an error (the documented behavior is
   `RuntimeError: Invalid argument: ... called on invalid buffer`);
3. `donate_argnums` indexes positional arguments by integer index;
4. output buffer aliasing with the donated input is described as an
   XLA optimization opportunity, **not a user-visible guarantee** — JAX
   only guarantees you cannot keep using the donated input.

The audit's recommendation to assert `donated.is_deleted() is True` AND
`jnp.asarray(donated)` raises `RuntimeError` after the donating call is
the **documented contract**, not implementation-defined behavior. It is
the right gate to enforce in tests.

### Implication for the corrected plan

- Donation tests should assert the documented contract:
  `donated_array.is_deleted()` and that re-reading the donated buffer
  raises `RuntimeError`. Do **not** assert that the output shares the
  same buffer as the donated input — that is an XLA hint, not a JAX
  guarantee.
- Treat `donate_argnums` as positional-integer indices in the test
  harness. The keyword name itself is keyword-only on `jit`, but its
  value is a collection of positional arg indices.

---

## Claim 4 — `JAX_PLATFORMS=cuda,cpu` SSOT for callback compatibility

### Codebase evidence

`grep -rn 'JAX_PLATFORMS' src/ benchmarks/ tests/` confirms `cuda,cpu`
is treated as the canonical CUDA-mode platform string in three places:

- `src/simsopt/backend/runtime.py:601-610`
  ```python
  def _runtime_jax_platforms_value(platform: str) -> str:
      if platform == "cuda":
          return "cuda,cpu"
      return _runtime_jax_platform_value(platform)
  ```
- `benchmarks/validation_ladder_common.py:263-264`
  ```python
  jax_platforms = "cuda,cpu" if platform == "cuda" else platform
  env["JAX_PLATFORMS"] = jax_platforms
  ```
- `tests/subprocess/import_smoke_cases.py:574` —
  `case_entrypoint_runtime_helper_promotes_cuda_to_cuda_cpu_for_callback_flags`
  asserts that requesting `JAX_PLATFORMS=cuda` together with a
  `--diagnostic-callbacks` flag is **promoted to `cuda,cpu`** by the
  entrypoint runtime helper. The case name itself documents the
  rationale: callback-flag launches require both backends.

### JAX documentation (verbatim, via WebFetch on docs.jax.dev)

`JAX_PLATFORMS` / `jax_platforms` (Configuration Options page):
> "Comma-separated list of platform names specifying which platforms
> jax should initialize."
> "The first platform in the list will be the default platform."
> "For example, config.jax_platforms=cpu,tpu means that CPU and TPU
> backends will be initialized, and the CPU backend will be used unless
> otherwise specified. If TPU initialization fails, it will raise an
> exception."

i.e. listing `cuda,cpu` means: initialize the CUDA backend (default,
because it is first), **and also** initialize the CPU backend so it is
available for use.

External callbacks (https://docs.jax.dev/en/latest/external-callbacks.html):
> "When running on accelerators like GPU or TPU, this data movement and
> host synchronization can lead to significant overhead each time `jv`
> is called."
> "However, if you are running JAX on a single CPU (where the 'host'
> and 'device' are on the same hardware), JAX will generally do this
> data transfer in a fast, zero-copy fashion."

`jax.pure_callback` (per WebSearch summary of docs):
> "pure_callback passes JAX arrays placed on a local CPU as input, and
> should also return JAX arrays on CPU."

The official docs do not contain a single-sentence "callbacks **require**
the CPU backend to be initialized when running on GPU" line, but they
do document that callback inputs/outputs are CPU-placed, and the JAX
launcher logic in this repo treats `cuda,cpu` as the SSOT precisely to
guarantee the CPU backend is available for those host-side callback
hops without a second runtime initialization round-trip.

### Verdict: PARTIALLY CORRECT

- The codebase invariant — "the SSOT CUDA-mode platform string is
  `cuda,cpu`, and a launcher with `--diagnostic-callbacks` (or other
  callback-bearing flags) must promote `cuda` → `cuda,cpu`" — is real,
  enforced by tests, and consistent with how `jax.pure_callback` /
  `jax.experimental.io_callback` use CPU-placed buffers.
- The official JAX docs **do not** explicitly state "callbacks require
  the CPU backend to be initialized." They document that callbacks are
  CPU-placed and that `JAX_PLATFORMS` initializes the listed backends
  in order. The conclusion that "CUDA-mode launches that use callbacks
  must list `cuda,cpu`" is a defensible SSOT decision in this codebase
  but is not a documented JAX hard requirement.
- Therefore: keep the SSOT, keep the runtime guard call as the
  authoritative check, but document the rationale as "this repo's
  policy to keep callback host hops within an initialized CPU backend"
  rather than "JAX requires it."

### Implication for the corrected plan

- Keep `JAX_PLATFORMS=cuda,cpu` as the SSOT CUDA-mode platform string.
- Every production GPU probe must call the runtime guard
  (`_runtime_jax_platforms_value("cuda") == "cuda,cpu"` and
  `jax.default_backend() == "gpu"`) rather than re-deriving the value.
- In the plan rationale, cite `_runtime_jax_platforms_value`,
  `validation_ladder_common.py`, and the
  `case_entrypoint_runtime_helper_promotes_cuda_to_cuda_cpu_for_callback_flags`
  smoke case as the SSOT enforcement points, and cite the JAX
  Configuration Options page as the documented semantics for
  comma-separated `JAX_PLATFORMS`. Do **not** claim JAX requires `cpu`
  to be in the platform list for callbacks — that is a repo-side
  policy, not a documented JAX requirement.

---

## Summary of verdicts

| Claim | Verdict |
|---|---|
| 1. jaxlib 0.9.2 has no stable CUDA-version attrs | CORRECT |
| 2. `CUDA_FORCE_PTX_JIT` / `CUDA_DISABLE_PTX_JIT` are the PTX/cubin knobs | CORRECT |
| 3. JAX donation: input becomes invalid/deleted after the call | CORRECT |
| 4. `JAX_PLATFORMS=cuda,cpu` SSOT for callback compatibility | PARTIALLY CORRECT (codebase SSOT real and enforced; not a documented JAX hard requirement) |

## Sources

- [JAX buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)
- [jax.jit (donate_argnums)](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html)
- [JAX Configuration Options (jax_platforms)](https://docs.jax.dev/en/latest/config_options.html)
- [JAX External Callbacks](https://docs.jax.dev/en/latest/external-callbacks.html)
- [NVIDIA CUDA Programming Guide — Environment Variables (5.2)](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
- Local probes against `/Users/suhjungdae/code/columbia/simsopt-jax/.conda/jax-0.9.2/bin/python` with `PYTHONNOUSERSITE=1` (jaxlib 0.9.2, jax 0.9.2, CPU-only macOS host)
- Codebase: `src/simsopt/backend/runtime.py:601-610`, `benchmarks/validation_ladder_common.py:263-264`, `tests/subprocess/import_smoke_cases.py:574-594`
