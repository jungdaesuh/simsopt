"""
Conftest for integration tests that need both simsoptpp (CPU reference)
and our JAX modules.

When running against a conda env that has simsopt installed as a
scikit-build editable install (e.g. candidate-fixed), the
ScikitBuildRedirectingFinder intercepts all ``simsopt.*`` imports.
This conftest patches the finder to include our new JAX modules
so they can be imported alongside the CPU reference.
"""

import os
import sys


def _patch_meta_path_finder():
    """Add simsopt-jax source modules to the scikit-build editable finder."""
    jax_src = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "simsopt")
    )

    new_modules = {
        "simsopt.field.biotsavart_jax": os.path.join(
            jax_src, "field", "biotsavart_jax.py"
        ),
        "simsopt.field.biotsavart_jax_backend": os.path.join(
            jax_src, "field", "biotsavart_jax_backend.py"
        ),
        "simsopt.geo.surface_fourier_jax": os.path.join(
            jax_src, "geo", "surface_fourier_jax.py"
        ),
        "simsopt.geo.boozer_residual_jax": os.path.join(
            jax_src, "geo", "boozer_residual_jax.py"
        ),
        "simsopt.objectives.fluxobjective_jax": os.path.join(
            jax_src, "objectives", "fluxobjective_jax.py"
        ),
        "simsopt.objectives.integral_bdotn_jax": os.path.join(
            jax_src, "objectives", "integral_bdotn_jax.py"
        ),
        "simsopt.geo.label_constraints_jax": os.path.join(
            jax_src, "geo", "label_constraints_jax.py"
        ),
        "simsopt.geo.optimizer_jax": os.path.join(jax_src, "geo", "optimizer_jax.py"),
        "simsopt.geo.boozersurface_jax": os.path.join(
            jax_src, "geo", "boozersurface_jax.py"
        ),
        "simsopt.geo.surfaceobjectives_jax": os.path.join(
            jax_src, "geo", "surfaceobjectives_jax.py"
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
_patch_meta_path_finder()
