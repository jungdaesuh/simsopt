"""
Conftest for integration tests that need both simsoptpp (CPU reference)
and our JAX modules.

When running against a conda env that has simsopt installed as a
scikit-build editable install (e.g. ``.conda/jax``), the
ScikitBuildRedirectingFinder intercepts all ``simsopt.*`` imports.
This conftest patches the finder to include the top-level JAX modules
so they can be imported alongside the CPU reference.
"""

import os
import sys


def _patch_meta_path_finder():
    """Add simsopt-jax source modules to the scikit-build editable finder."""
    src_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src")
    )

    new_modules = {
        "simsopt_jax.field.biotsavart": os.path.join(
            src_root, "simsopt_jax", "field", "biotsavart.py"
        ),
        "simsopt_jax_adapters.field.biotsavart_backend": os.path.join(
            src_root,
            "simsopt_jax_adapters",
            "field",
            "biotsavart_backend.py",
        ),
        "simsopt_jax.geo.surface_fourier": os.path.join(
            src_root, "simsopt_jax", "geo", "surface_fourier.py"
        ),
        "simsopt_jax.geo.boozer_residual": os.path.join(
            src_root, "simsopt_jax", "geo", "boozer_residual.py"
        ),
        "simsopt_jax_adapters.objectives.flux": os.path.join(
            src_root, "simsopt_jax_adapters", "objectives", "flux.py"
        ),
        "simsopt_jax.objectives.integral_bdotn": os.path.join(
            src_root, "simsopt_jax", "objectives", "integral_bdotn.py"
        ),
        "simsopt_jax.geo.label_constraints": os.path.join(
            src_root, "simsopt_jax", "geo", "label_constraints.py"
        ),
        "simsopt_jax.geo.optimizers.optimizer": os.path.join(
            src_root, "simsopt_jax", "geo", "optimizers", "optimizer.py"
        ),
        "simsopt_jax_adapters.geo.boozer_surface": os.path.join(
            src_root, "simsopt_jax_adapters", "geo", "boozer_surface.py"
        ),
        "simsopt_jax_adapters.geo.surface_objectives": os.path.join(
            src_root, "simsopt_jax_adapters", "geo", "surface_objectives.py"
        ),
    }

    for finder in sys.meta_path:
        if hasattr(finder, "known_source_files"):
            finder.known_source_files.update(new_modules)
            # Also update submodule_search_locations so the parent
            # packages know about the new files.
            for mod, filepath in new_modules.items():
                parent = ".".join(mod.split(".")[:-1])
                if parent and parent in finder.submodule_search_locations:
                    finder.submodule_search_locations[parent].add(
                        os.path.dirname(filepath)
                    )
            return True
    return False


# Patch on conftest load (before any test imports).
SCIKIT_BUILD_EDITABLE_FINDER_PATCHED = _patch_meta_path_finder()


def pytest_report_header(config):
    del config
    if SCIKIT_BUILD_EDITABLE_FINDER_PATCHED:
        return "simsopt-jax integration: patched scikit-build editable finder"
    return (
        "simsopt-jax integration: no scikit-build editable finder detected; "
        "using normal import path and per-module simsoptpp gates"
    )
