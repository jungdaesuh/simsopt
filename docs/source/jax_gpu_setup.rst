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

    conda env create -f envs/columbia-jax-0.9.2.yml
    conda activate columbia-jax-0.9.2
    pip install -e ".[JAX]"

GPU install (CUDA 12)::

    pip install "simsopt[JAX_GPU]"

    # or manually:
    pip install "jax[cuda12]==0.9.2"

Both the public and private optimizer lanes are Python ``3.11+`` with JAX /
jaxlib ``0.9.2``. The trusted reference backend remains
``optimizer_backend="scipy"``; ``ondevice`` / ``hybrid`` still require the
separate private-optimizer validation track.

Verify the install::

    python -c "import jax; print(jax.devices())"
    # Should show: [CudaDevice(id=0)]

Environment Variables
---------------------

Two orthogonal settings control the JAX lane:

**Code-path backend** (which implementation to use):

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

**JAX device platform** (which hardware JAX targets):

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

Example: run Stage 2 on GPU::

    SIMSOPT_BACKEND=jax SIMSOPT_JAX_PLATFORM=cuda \
        python banana_coil_solver.py

Programmatic access::

    from simsopt.backend import get_backend, is_jax_backend, get_jax_platform

    if is_jax_backend():
        from simsopt.field import BiotSavartJAX
        ...

GPU Node Quick-Start
--------------------

1. **Load CUDA toolkit** (cluster-specific)::

       module load cuda/12.x  # or equivalent

2. **Create environment**::

       conda env create -f envs/columbia-jax-0.9.2.yml
       conda activate columbia-jax-0.9.2
       pip install "jax[cuda12]==0.9.2"

3. **Install simsopt**::

       cd simsopt-jax
       pip install -e ".[JAX_GPU]"

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
   current ``fork`` remote SHA, and runs the authoritative Tier 2 / Tier 3
   entrypoints:

   - ``benchmarks/stage2_e2e_comparison.py``
   - ``benchmarks/single_stage_init_parity.py``

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
