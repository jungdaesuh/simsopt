JAX GPU Environment Setup
=========================

This page documents how to set up and run the JAX-accelerated code path
in simsopt on GPU-equipped nodes.

Supported Hardware
------------------

+----------+-------------------------------------+-------------------------------+
| Tier     | Hardware                            | Purpose                       |
+==========+=====================================+===============================+
| Primary  | NVIDIA A100 / H100 (FP64-capable)   | Performance benchmarks        |
+----------+-------------------------------------+-------------------------------+
| Minimum  | NVIDIA V100 (FP64-capable)          | Lowest acceptable reference   |
+----------+-------------------------------------+-------------------------------+
| Smoke    | Consumer NVIDIA (RTX 3090/4090)     | Functional correctness only   |
+----------+-------------------------------------+-------------------------------+
| Always   | CPU                                 | Reference oracle, CI, dev     |
+----------+-------------------------------------+-------------------------------+

The Boozer and field kernels are FP64-heavy scientific workloads.  Consumer
GPUs have severely reduced FP64 throughput and are not representative of
the target use case.

Dependencies
------------

Base install (CPU development/testing)::

    conda env create -f envs/jax-0.9.2.yml
    conda activate jax-0.9.2

The environment recipe already performs the editable CPU-side
``simsopt[JAX,dev]`` install, so no extra ``pip install`` step is needed for
the public CPU development lane.

GPU install (CUDA 12)::

    # From the repo root:
    pip install -e ".[JAX_GPU,dev]"

    # or for a non-editable runtime:
    pip install "simsopt[JAX_GPU]"

Both the public and private optimizer lanes are Python ``3.11+`` with JAX /
jaxlib ``0.9.2``. The trusted reference backend remains
``optimizer_backend="scipy"``; ``ondevice`` / ``hybrid`` still require the
separate private-optimizer validation track.

Verify the install::

    python -c "import jax; print(jax.devices())"
    # Should show: [CudaDevice(id=0)]

Environment Variables
---------------------

The preferred runtime selector is the explicit backend mode:

.. list-table::
   :header-rows: 1

   * - Variable
     - Values
     - Default
     - Purpose
   * - ``SIMSOPT_BACKEND_MODE``
     - ``native_cpu``, ``jax_cpu_parity``, ``jax_gpu_parity``, ``jax_gpu_fast``
     - ``native_cpu``
     - Select the runtime contract directly
   * - ``SIMSOPT_BACKEND_STRICT``
     - truthy / falsy env values
     - falsy
     - Record strict fallback policy for callsites that honor strict-mode guards

The older env pair remains supported for compatibility.

**Code-path backend**:

.. list-table::
   :header-rows: 1

   * - Variable
     - Values
     - Default
     - Purpose
   * - ``SIMSOPT_BACKEND``
     - ``cpu``, ``jax``
     - ``cpu``
     - Select simsoptpp (CPU) or JAX code path
   * - ``STAGE2_BACKEND``
     - ``cpu``, ``jax``
     - ``cpu``
     - Legacy alias for ``SIMSOPT_BACKEND``

**JAX device platform**:

.. list-table::
   :header-rows: 1

   * - Variable
     - Values
     - Default
     - Purpose
   * - ``SIMSOPT_JAX_PLATFORM``
     - ``cpu``, ``cuda``
     - ``cpu``
     - JAX XLA device selection
   * - ``SIMSOPT_JAX_BACKEND``
     - ``cpu``, ``cuda``
     - ``cpu``
     - Legacy alias for ``SIMSOPT_JAX_PLATFORM``

Example: run Stage 2 on the GPU parity lane::

    SIMSOPT_BACKEND_MODE=jax_gpu_parity \
        python banana_coil_solver.py

Legacy-compatible example::

    SIMSOPT_BACKEND=jax SIMSOPT_JAX_PLATFORM=cuda \
        python banana_coil_solver.py

Programmatic access::

    from simsopt import config as simsopt_config
    from simsopt.backend import (
        get_backend,
        get_backend_mode,
        get_backend_policy,
        get_chunk_tuning,
        is_jax_backend,
    )

    simsopt_config.set_backend("jax_gpu_parity", strict=True)
    policy = get_backend_policy()
    chunk_tuning = get_chunk_tuning()

    assert get_backend_mode() == "jax_gpu_parity"
    assert policy.chunk_policy == "stable_default"
    assert chunk_tuning.point_chunk_size >= 0
    assert policy.tolerance_tier == "parity"
    if is_jax_backend():
        from simsopt.field import BiotSavartJAX
        ...

Call ``set_backend(...)`` before importing JAX-heavy simsopt subpackages so the
requested runtime mode is visible to those imports.

Mode-owned policy
-----------------

The mode contract now owns the baseline numerical-policy labels that should
appear in provenance and validation output.

.. list-table::
   :header-rows: 1

   * - Mode
     - X64
     - Chunk policy
     - Tolerance tier
     - Compilation cache policy
   * - ``native_cpu``
     - required by the current import-time scientific runtime contract
     - ``host_reference``
     - ``cpu_reference``
     - ``not_applicable``
   * - ``jax_cpu_parity``
     - required
     - ``stable_default``
     - ``parity``
     - ``optional_persistent``
   * - ``jax_gpu_parity``
     - required
     - ``stable_default``
     - ``parity``
     - ``optional_persistent``
   * - ``jax_gpu_fast``
     - currently still required
     - ``performance_tuned``
     - ``fast``
     - ``optional_persistent``

These labels do not mean every kernel already implements the final chunked
parity/fast architecture. They define the runtime contract and provenance
surface while the remaining kernel work is still in progress.

Chunk autotuning
----------------

The runtime now resolves one effective chunk-tuning contract for the active
mode:

- explicit env overrides still win for coil/quadrature/pairwise chunk sizes
- otherwise, JAX CUDA modes try to bucket chunk sizes from available GPU VRAM
- if VRAM cannot be detected, the checked-in mode defaults remain the fallback

The VRAM probe looks at ``SIMSOPT_JAX_GPU_MEMORY_TOTAL_MB`` first and then
falls back to ``nvidia-smi``. Set ``SIMSOPT_JAX_CHUNK_AUTOTUNE=0`` to disable
autotuning and force the checked-in mode defaults.

GPU Node Quick-Start
--------------------

1. **Load CUDA toolkit** (cluster-specific)::

       module load cuda/12.x  # or equivalent

2. **Create environment**::

       conda env create -f envs/jax-0.9.2.yml
       conda activate jax-0.9.2

3. **Upgrade the editable install to the CUDA extra set**::

       cd simsopt-jax
       pip install -e ".[JAX_GPU,dev]"

4. **Verify GPU access**::

       python -c "
       import jax
       print('Devices:', jax.devices())
       x = jax.numpy.ones(1000)
       print('Sum:', float(x.sum()))
       "

5. **Run a benchmark**::

       SIMSOPT_JAX_PLATFORM=cuda python benchmarks/jax_feasibility_spike.py

Hugging Face Jobs
-----------------

For repeatable A100 / H200 proof runs on Hugging Face Jobs, the repo now
ships a dedicated launch path under ``benchmarks/hf_jobs``.

This is still proof infrastructure, not evidence that full-GPU productization
is complete. Routine GPU CI and warm production-scale workflow proof remain
separate ship gates.

1. **Build and publish the reusable runtime image once**::

       docker build -f benchmarks/hf_jobs/production_gpu_proof.Dockerfile \
         -t <registry>/simsopt-jax:cuda12-jax092 .
       docker push <registry>/simsopt-jax:cuda12-jax092

   The image bakes the heavy system and Python dependency stack, including
   ``jax[cuda12]==0.9.2``. Runtime jobs still clone and build the exact target
   repo SHA so the proof remains commit-accurate.

2. **Launch the proof jobs**::

       SIMSOPT_HF_GPU_IMAGE=<registry>/simsopt-jax:cuda12-jax092 \
         python benchmarks/hf_jobs/launch_production_gpu_proof.py

   By default this launches both ``a100-large`` and ``h200`` jobs, pins the
   current ``fork`` remote SHA, prints a local preflight summary, and runs the
   authoritative Tier 2 / Tier 3 entrypoints:

   - ``benchmarks/stage2_e2e_comparison.py``
   - ``benchmarks/single_stage_init_parity.py``

   The default ``maxiter=20`` Stage 2 path is now portability smoke only:
   it runs the cold/warm Stage 2 probes without forcing an explicit end-state
   geometry override. Single-stage cold/warm probes still run even if an earlier
   rung fails, and the harness returns one aggregate exit code after collecting
   all rung evidence.

   To add a dedicated Stage 2 reproducibility rung with an explicit geometry
   gate, pass both a longer Stage 2 budget and ``--geometry-rel-tol``::

       SIMSOPT_HF_GPU_IMAGE=<registry>/simsopt-jax:cuda12-jax092 \
         python benchmarks/hf_jobs/launch_production_gpu_proof.py \
         --stage2-maxiter 60 --geometry-rel-tol 1e-6

3. **Fallback mode**::

       python benchmarks/hf_jobs/launch_production_gpu_proof.py \
         --image python:3.11-bookworm --bootstrap-mode always

   This keeps the old ad hoc bootstrap behavior for environments where the
   reusable image has not been published yet.

Troubleshooting
---------------

**"No GPU/TPU found"**
  Check ``nvidia-smi`` output.  Ensure ``jax[cuda12]`` is installed
  (the CUDA extras are on the ``jax`` package, not ``jaxlib``).  Verify CUDA driver version is compatible with
  the installed ``jaxlib`` (JAX docs have a compatibility table).

**Slow first call**
  Expected.  JAX compiles XLA kernels on first invocation.  Subsequent
  calls on the same array shapes are fast (cached).  Use
  ``jax.clear_caches()`` and ``JAX_EXPLAIN_CACHE_MISSES=1`` to debug
  unexpected recompilations.

**Out of memory**
  Large Jacobian/Hessian objects can exhaust GPU memory.  Reduce grid
  resolution (``mpol``, ``ntor``) or use ``XLA_PYTHON_CLIENT_PREALLOCATE=false``
  to disable JAX's default 75% memory pre-allocation.

**FP64 precision on consumer GPUs**
  Consumer GPUs (RTX series) have 1/32 or 1/64 FP64 throughput.
  Performance will be poor; use only for correctness validation.

Version Upgrade Policy
----------------------

JAX version upgrades are explicit maintenance work.  Each upgrade requires:

1. Re-run all parity tests.
2. Re-benchmark steady-state and compile-time overhead.
3. Verify no unexpected recompilation from changed XLA semantics.
