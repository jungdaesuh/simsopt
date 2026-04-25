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
``SIMSOPT_BACKEND_MODE=native_cpu`` with ``optimizer_backend="scipy"``.
All JAX backend modes now require the on-device optimizer lane at the
high-level Stage 2 / single-stage / Boozer contracts, so ``backend="jax"``
cannot silently route back through the host SciPy loops. The remaining SciPy
adapter is a CPU/reference-only oracle path.

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

Repo-local bootstrap
--------------------

When developing in a workspace that also contains sibling ``simsopt*``
repositories or older editable installs, plain ``import simsopt`` /
``import simsoptpp`` can resolve to the wrong checkout or to a stale
compiled extension in ``site-packages``. This is especially easy to miss
when the Python package comes from this repo but ``simsoptpp`` still comes
from some other environment.

Use the repo bootstrap helper in scripts, examples, and ad hoc probes::

    from repo_bootstrap import bootstrap_local_simsopt

    bootstrap_local_simsopt("src")

``bootstrap_local_simsopt("src")`` does two things:

- strips editable import finders that hijack ``simsopt`` imports
- if a matching local ``build/*/simsoptpp*.so`` exists for the active
  interpreter, loads that repo-local extension explicitly before importing
  ``simsopt``

This is the supported way to keep this repo self-contained at runtime
without relying on whichever ``simsoptpp`` binary happens to be first on
the ambient Python path.

To refresh the compiled extension for the active interpreter from the repo
root::

    python -m pip wheel --no-deps . -w .artifacts/wheels

Verify that the active process is really using the local extension::

    python - <<'PY'
    from repo_bootstrap import bootstrap_local_simsopt
    bootstrap_local_simsopt("src")
    import simsoptpp
    print(simsoptpp.__file__)
    PY

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
         python benchmarks/hf_jobs/launch_production_gpu_proof.py \
           --single-stage-warm-start-run-dir <tracked-single-stage-run-dir>

   By default this launches both ``a100-large`` and ``h200`` jobs, pins the
   current ``fork`` remote SHA, prints a local preflight summary, and runs the
   authoritative Tier 2 / Tier 3 entrypoints:

   - ``benchmarks/stage2_e2e_comparison.py``
   - ``benchmarks/single_stage_init_parity.py``

   Single-stage proof jobs require a seed path that exists inside the cloned
   target repo at the requested SHA. Pass a repo-relative warm-start directory
   containing ``surf_opt.json``, ``results.json``, and ``biot_savart_opt.json``,
   or pass a repo-relative runtime seed spec. Host-local absolute paths are
   rejected before launch.

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

**Wrong repo or wrong ``simsoptpp`` binary loaded**
  If ``simsopt`` or ``simsoptpp`` resolves to ``site-packages`` or a sibling
  repo checkout, call ``bootstrap_local_simsopt("src")`` before importing
  simsopt-heavy modules. Rebuild the local extension with
  ``python -m pip wheel --no-deps . -w .artifacts/wheels`` if no matching
  local ``build/*/simsoptpp*.so`` exists for the active interpreter.

**``libomp.dylib`` missing on macOS**
  A repo-local virtualenv can still fail to import ``simsoptpp`` if the
  extension was linked against OpenMP but the runtime loader cannot find
  ``libomp.dylib``. In that case, either provide a working OpenMP runtime
  for that environment or use an interpreter whose local repo build already
  loads successfully.

**FP64 precision on consumer GPUs**
  Consumer GPUs (RTX series) have 1/32 or 1/64 FP64 throughput.
  Performance will be poor; use only for correctness validation.

Runpod Operational Notes
------------------------

The following points were validated on **April 17, 2026** during real
Runpod RTX 4090 proof runs with exact ``jax==0.9.2`` / ``jaxlib==0.9.2``.
These are setup lessons, not theoretical guidance.

**Use a Linux-built environment on the pod**
  Do not copy a local macOS ``.conda`` or virtualenv into Runpod and try to
  execute it there.  The copied interpreter can fail immediately with
  ``Exec format error`` because the binaries are built for the wrong OS/ABI.
  Build the Python environment on the pod itself.

**Exact JAX 0.9.2 on the stock CUDA 12.4 Runpod image needed CUDA toolkit 12.9**
  On ``runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04``, the exact
  ``jax[cuda12]==0.9.2`` wheel stack only became reliable after installing
  ``cuda-toolkit-12-9`` and repointing ``/usr/local/cuda`` to
  ``/usr/local/cuda-12.9``.  Without that upgrade, prior proof attempts hit
  CUDA userspace/toolchain mismatches.

**Keep the bundled-wheel CUDA mode**
  For the pip-installed JAX CUDA wheel path, keep
  ``SIMSOPT_JAX_CUDA_LIBRARY_MODE=bundled``.  That preserves the intended
  wheel-provided NVIDIA library resolution path.  Do not reintroduce older
  behavior that unsets ``LD_LIBRARY_PATH`` in the proof harness.

**Editable installs still need version metadata**
  This repo uses ``setuptools_scm`` for versioning.  If you sync a source tree
  onto the pod without ``.git``, editable install can fail because the build
  backend cannot resolve version metadata.  Either:

  1. clone the repo on the pod with real git metadata, or
  2. create a local git snapshot on the pod before installing.

**Single-stage seed paths are a package contract, not just one JSON file**
  ``single_stage_banana_example.py`` does not only read
  ``biot_savart_opt.json``.  It also expects the sibling ``results.json`` next
  to that Stage 2 seed path.  If you copy only ``biot_savart_opt.json`` onto
  the pod, single-stage startup can fail with ``FileNotFoundError`` for the
  missing adjacent ``results.json``.

**Current single-stage smoke donors must satisfy the TF-current contract**
  The older single-stage benchmark seed family includes ``TF_CURRENT_A=100000``.
  Current single-stage heads enforce ``TF_CURRENT_A <= 80000``, so real Runpod
  single-stage validation should use a strict-80k donor package instead of the
  legacy 100 kA fixture.

**macOS-to-Linux archive syncs can still leak Apple metadata**
  When syncing from macOS to Runpod, tar streams can emit repeated
  ``LIBARCHIVE.xattr.com.apple.*`` warnings on the Linux extractor if the
  source archive carries Apple xattrs.  Use a metadata-free archive path when
  pushing repo snapshots to the pod.

Version Upgrade Policy
----------------------

JAX version upgrades are explicit maintenance work.  Each upgrade requires:

1. Re-run all parity tests.
2. Re-benchmark steady-state and compile-time overhead.
3. Verify no unexpected recompilation from changed XLA semantics.
