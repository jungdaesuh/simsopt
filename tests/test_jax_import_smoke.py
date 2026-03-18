"""
Import smoke tests for the JAX code path.

These tests verify that JAX modules can be imported through the real
``simsopt`` package entrypoints (not via ``importlib.util`` bypass).
They run in the no-simsoptpp environment to catch import-chain regressions.
"""


def test_import_package_root():
    """simsopt package imports without simsoptpp."""
    import simsopt

    assert hasattr(simsopt, "__version__")


def test_import_biotsavart_jax():
    """BiotSavartJAX is importable through the real package entrypoint."""
    from simsopt.field import BiotSavartJAX

    assert BiotSavartJAX is not None


def test_import_squaredflux_jax():
    """SquaredFluxJAX is importable through the real package entrypoint."""
    from simsopt.objectives import SquaredFluxJAX

    assert SquaredFluxJAX is not None


def test_import_boozersurface_jax():
    """BoozerSurfaceJAX is importable through the real package entrypoint."""
    from simsopt.geo import BoozerSurfaceJAX

    assert BoozerSurfaceJAX is not None


def test_import_core_optimizable():
    """Optimizable base class imports without simsoptpp."""
    from simsopt._core.optimizable import Optimizable

    assert Optimizable is not None


def test_jax_classes_inherit_optimizable():
    """JAX adapter classes use the real Optimizable metaclass."""
    from simsopt._core.optimizable import Optimizable
    from simsopt.field import BiotSavartJAX
    from simsopt.objectives import SquaredFluxJAX

    assert issubclass(BiotSavartJAX, Optimizable)
    assert issubclass(SquaredFluxJAX, Optimizable)


def test_import_pure_jax_modules():
    """Pure JAX compute modules (M1) import through the package."""
    from simsopt.field.biotsavart_jax import biot_savart_B
    from simsopt.geo.surface_fourier_jax import stellsym_scatter_indices
    from simsopt.geo.boozer_residual_jax import boozer_residual_scalar
    from simsopt.objectives.integral_bdotn_jax import integral_BdotN

    assert callable(biot_savart_B)
    assert callable(stellsym_scatter_indices)
    assert callable(boozer_residual_scalar)
    assert callable(integral_BdotN)


def test_m5_classes_require_simsoptpp():
    """M5 single-stage wrappers need SurfaceXYZTensorFourier (CPU class).

    BoozerResidualJAX, IotasJAX, NonQuasiSymmetricRatioJAX use CPU surface
    objects at the boundary (M0 adapter pattern). Without simsoptpp they
    are not importable via the package entrypoint. This is expected.
    """
    import simsopt.geo

    m5_names = ["BoozerResidualJAX", "IotasJAX", "NonQuasiSymmetricRatioJAX"]
    try:
        from simsoptpp import Curve  # noqa: F401

        has_simsoptpp = True
    except (ImportError, AttributeError):
        has_simsoptpp = False

    for name in m5_names:
        available = hasattr(simsopt.geo, name)
        if has_simsoptpp:
            assert available, f"{name} should be available with simsoptpp"
        else:
            assert not available, f"{name} should NOT be available without simsoptpp"
