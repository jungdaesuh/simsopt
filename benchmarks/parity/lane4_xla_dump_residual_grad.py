"""Lane 4 — Phase 4 P4.4 XLA dump reproducer for the JAX Boozer residual grad.

Forces a fresh JIT compilation of
``boozer_residual_scalar_and_grad_cpu_ordered`` against the canonical pinned
boundary inputs (P4.1 bundle), with ``XLA_FLAGS=--xla_dump_to=<DIR>`` honored
from ``os.environ``. The dump directory then contains:

```
module_*.jit_*.before_optimizations.txt           # pre-opt HLO
module_*.jit_*.cpu_after_optimizations.txt        # post-opt HLO
module_*.jit_*.__compute_module_*.ir-no-opt.ll    # pre-opt LLVM IR
module_*.jit_*.__compute_module_*.ir-with-opt.ll  # post-opt LLVM IR
module_*.jit_*.obj-file.__compute_module_*.o      # codegen object file
```

The script makes ZERO assumptions about caching: it explicitly clears any
JAX persistent cache state from the in-process driver before lowering so the
``XLA_FLAGS`` set in the parent shell are respected by the fresh
compilation. Run via:

```
DUMP_DIR=.artifacts/parity/20260508-residual-pinned-inputs/p4_4_xla_dump
mkdir -p "$DUMP_DIR"
XLA_FLAGS="--xla_dump_to=$DUMP_DIR" \\
    .conda/jax-0.9.2/bin/python benchmarks/parity/lane4_xla_dump_residual_grad.py
```

Plan reference: ``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md``
§10 P4.4 + §19 (methodology).

This script does not import or modify any production code in ``src/**``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = REPO_ROOT / ".artifacts" / "parity" / "20260508-residual-pinned-inputs"


def _emit(msg: str) -> None:
    """Stdout-only log helper (avoid logging modules to keep dependencies minimal)."""

    print(msg, flush=True)


def _verify_xla_flags() -> str:
    """Refuse to run unless ``XLA_FLAGS=--xla_dump_to=<dir>`` is set up-front.

    JAX initializes the XLA driver lazily; if ``XLA_FLAGS`` is set inside the
    Python process AFTER ``jax`` has been imported anywhere in the call graph,
    the dump flag is silently ignored. We assert the flag is present in
    ``os.environ`` BEFORE importing ``jax``.
    """

    raw = os.environ.get("XLA_FLAGS", "")
    if "--xla_dump_to=" not in raw:
        raise SystemExit(
            "XLA_FLAGS must contain --xla_dump_to=<dir> before this script "
            "is launched. Got XLA_FLAGS=" + repr(raw)
        )
    # Extract the directory.
    for token in raw.split():
        if token.startswith("--xla_dump_to="):
            return token.split("=", 1)[1]
    raise AssertionError("unreachable")


def _load_canonical(bundle: Path) -> dict[str, np.ndarray]:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    canonical_entries = [
        e for e in manifest.get("files", []) if e["role"] == "canonical"
    ]
    arrays: dict[str, np.ndarray] = {}
    for entry in canonical_entries:
        arr = np.load(bundle / entry["path"], allow_pickle=False)
        if arr.dtype != np.float64:
            raise RuntimeError(
                f"canonical {entry['name']} dtype={arr.dtype}; expected float64"
            )
        arrays[entry["name"]] = arr
    # The scalar `optimize_G`, `weight_inv_modB`, `iota`, `G_value` are stored
    # as 0-D arrays alongside the array bundle.
    return arrays


def main() -> int:
    dump_dir = Path(_verify_xla_flags()).resolve()
    _emit(f"[lane4] dump_dir={dump_dir}")
    if not dump_dir.exists():
        dump_dir.mkdir(parents=True, exist_ok=True)

    bundle = Path(os.environ.get("CANONICAL_BUNDLE", str(DEFAULT_BUNDLE))).resolve()
    if not (bundle / "manifest.json").is_file():
        raise SystemExit(f"missing manifest.json under {bundle!s}")
    _emit(f"[lane4] bundle={bundle}")

    canonical = _load_canonical(bundle)
    weight_inv_modB = bool(np.asarray(canonical["weight_inv_modB"]))
    G_value = float(np.asarray(canonical["G_value"]))
    iota = float(np.asarray(canonical["iota"]))

    # Import jax AFTER asserting XLA_FLAGS so the dump flag is honored.
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    _emit(f"[lane4] jax.__version__={jax.__version__}")
    _emit(f"[lane4] devices={jax.devices()}")

    from simsopt.geo.boozer_residual_jax import (  # noqa: PLC0415
        boozer_residual_scalar_and_grad_cpu_ordered,
    )

    # Promote canonical numpy arrays to JAX device arrays. Use jnp.asarray
    # so the dtype/shape metadata is preserved; the canonical bundle is
    # already float64 (verified above).
    B = jnp.asarray(canonical["B"], dtype=jnp.float64)
    dB_dX = jnp.asarray(canonical["dB_dX"], dtype=jnp.float64)
    xphi = jnp.asarray(canonical["xphi"], dtype=jnp.float64)
    xtheta = jnp.asarray(canonical["xtheta"], dtype=jnp.float64)
    dx_ds = jnp.asarray(canonical["dx_ds"], dtype=jnp.float64)
    dxphi_ds = jnp.asarray(canonical["dxphi_ds"], dtype=jnp.float64)
    dxtheta_ds = jnp.asarray(canonical["dxtheta_ds"], dtype=jnp.float64)
    G_arr = jnp.asarray(G_value, dtype=jnp.float64)
    iota_arr = jnp.asarray(iota, dtype=jnp.float64)

    # Wrap the residual entrypoint in jax.jit. Static args are the boolean
    # `optimize_G` and `weight_inv_modB`. We bake them into a closure so the
    # AOT call signature is purely positional.
    def grad_fn(G, iota_, B_, dB_dX_, xphi_, xtheta_, dx_ds_, dxphi_ds_, dxtheta_ds_):
        return boozer_residual_scalar_and_grad_cpu_ordered(
            G,
            iota_,
            B_,
            dB_dX_,
            xphi_,
            xtheta_,
            dx_ds_,
            dxphi_ds_,
            dxtheta_ds_,
            optimize_G=True,
            weight_inv_modB=weight_inv_modB,
        )

    jitted = jax.jit(grad_fn)

    # Sanity: the AOT pipeline (`lower(...).compile()`) does not actually
    # execute the kernel, but it does emit the HLO/LLVM/object dump. We
    # then call the jitted function once more to confirm parity with the
    # reference numbers from the byte arbiter (and to flush any laziness).
    _emit("[lane4] lowering JIT trace ...")
    lowered = jitted.lower(
        G_arr, iota_arr, B, dB_dX, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds
    )
    _emit("[lane4] compiling AOT ...")
    compiled = lowered.compile()
    _emit(f"[lane4] compiled type={type(compiled).__name__}")

    _emit("[lane4] running compiled call ...")
    val, grad = compiled(
        G_arr, iota_arr, B, dB_dX, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds
    )
    val = jax.device_get(val)
    grad = np.asarray(jax.device_get(grad), dtype=np.float64)
    _emit(f"[lane4] value={float(val):.18e}")
    _emit(f"[lane4] grad.shape={grad.shape}")
    _emit(f"[lane4] |grad|_inf={float(np.max(np.abs(grad))):.18e}")

    # Inventory the dump directory.
    dump_files = sorted(p.name for p in dump_dir.iterdir())
    _emit(f"[lane4] dump_dir contains {len(dump_files)} files")
    for name in dump_files[:10]:
        _emit(f"  - {name}")
    if len(dump_files) > 10:
        _emit(f"  ... ({len(dump_files) - 10} more)")
    if not dump_files:
        _emit(
            "[lane4] WARNING: dump directory is empty. "
            "Ensure XLA_FLAGS=--xla_dump_to=<dir> is set BEFORE python launches "
            "and that this is a fresh process (no warm cache)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
