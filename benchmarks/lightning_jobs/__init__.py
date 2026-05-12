"""Lightning AI launchers for the production CUDA proof.

The HF Jobs launcher in :mod:`benchmarks.hf_jobs.launch_production_gpu_proof`
is the SSOT for the proof contract (preflight, run-proof argv, command body).
The Lightning launcher in this package builds the same proof command and
submits it through the ``lightning_sdk`` API, mounting the
``simsopt-jax-parity-proofs`` data connection so artifacts land outside the
ephemeral job container.
"""
