import weakref
from dataclasses import dataclass

import numpy as np
from sympy import Symbol, lambdify, exp

from .._core.json import GSONable
from .._core.util import RealArray

import simsoptpp as sopp
from simsopt.geo.curve import Curve

__all__ = ['GaussianSampler', 'PerturbationSample', 'CurvePerturbed']


@dataclass
class GaussianSampler(GSONable):
    r"""
    Generate a periodic gaussian process on the interval [0, 1] on a given list of quadrature points.
    The process has standard deviation ``sigma`` a correlation length scale ``length_scale``.
    Large values of ``length_scale`` correspond to smooth processes, small values result in highly oscillatory
    functions.
    Also has the ability to sample the derivatives of the function.

    We consider the kernel

    .. math::

        \kappa(d) = \sigma^2 \exp(-d^2/l^2)

    and then consider a Gaussian process with covariance

    .. math::

        Cov(X(s), X(t)) = \sum_{i=-\infty}^\infty \sigma^2 \exp(-(s-t+i)^2/l^2)

    the sum is used to make the kernel periodic and in practice the infinite sum is truncated.

    Args:
        points: the quadrature points along which the perturbation should be computed.
        sigma: standard deviation of the underlying gaussian process
               (measure for the magnitude of the perturbation).
        length_scale: length scale of the underlying gaussian process
                      (measure for the smoothness of the perturbation).
        n_derivs: number of derivatives of the gaussian process to sample.
    """

    points: RealArray
    sigma: float
    length_scale: float
    n_derivs: int = 1

    def __post_init__(self):
        xs = self.points
        n = len(xs)
        cov_mat = np.zeros((n*(self.n_derivs+1), n*(self.n_derivs+1)))

        def kernel(x, y):
            return sum((self.sigma**2)*exp(-(x-y+i)**2/(self.length_scale**2)) for i in range(-5, 6))

        XX, YY = np.meshgrid(xs, xs, indexing='ij')
        x = Symbol("x")
        y = Symbol("y")
        f = kernel(x, y)
        for ii in range(self.n_derivs+1):
            for jj in range(self.n_derivs+1):
                if ii + jj == 0:
                    lam = lambdify((x, y), f, "numpy")
                else:
                    lam = lambdify((x, y), f.diff(*(ii * [x] + jj * [y])), "numpy")
                cov_mat[(ii*n):((ii+1)*n), (jj*n):((jj+1)*n)] = lam(XX, YY)

        # we need to compute the sqrt of the covariance matrix. we used to do this using scipy.linalg.sqrtm,
        # but it seems sometime between scipy 1.11.1 and 1.11.2 that function broke/changed behaviour.
        # So we use a LDLT decomposition instead. See als https://github.com/hiddenSymmetries/simsopt/issues/349
        # from scipy.linalg import sqrtm, ldl
        # self.L = np.real(sqrtm(cov_mat))
        from scipy.linalg import ldl
        lu, d, _ = ldl(cov_mat)
        self.L = lu @ np.sqrt(np.maximum(d, 0))

    def draw_sample(self, randomgen=None):
        """
        Returns a list of ``n_derivs+1`` arrays of size ``(len(points), 3)``, containing the
        perturbation and the derivatives.
        """
        n = len(self.points)
        n_derivs = self.n_derivs
        if randomgen is None:
            randomgen = np.random.Generator(np.random.PCG64DXSM())
        z = randomgen.standard_normal(size=(n*(n_derivs+1), 3))
        curve_and_derivs = self.L@z
        return [curve_and_derivs[(i*n):((i+1)*n), :] for i in range(n_derivs+1)]


class PerturbationSample(GSONable):
    """
    This class represents a single sample of a perturbation.  The point of
    having a dedicated class for this is so that we can apply the same
    perturbation to multipe curves (e.g. in the case of multifilament
    approximations to finite build coils).
    The main way to interact with this class is via the overloaded ``__getitem__``
    (i.e. ``[ ]`` indexing).
    For example::

        sample = PerturbationSample(...)
        g = sample[0] # get the values of the perturbation
        gd = sample[1] # get the first derivative of the perturbation

    Every :obj:`CurvePerturbed` built on a sample is registered with it, so
    :meth:`resample` refreshes all curves that share the sample and every
    object that depends on them (e.g. coils and Biot-Savart fields).
    The registry holds weak references and is never serialized or copied:
    pickling or copying a sample yields one with no registered curves, and
    curves rebuilt on it register again.
    """

    def __init__(self, sampler, randomgen=None, sample=None):
        """
        Hold one draw of ``sampler``, or the given pre-drawn ``sample``.

        Args:
            sampler: the :obj:`GaussianSampler` that draws the perturbation.
            randomgen: optional ``numpy.random.Generator`` used for every draw;
                a fresh unseeded generator is used for each draw if ``None``.
            sample: optional pre-drawn perturbation, a list of
                ``sampler.n_derivs + 1`` arrays of shape ``(len(sampler.points), 3)``
                holding the perturbation (in metres) and its derivatives with
                respect to the curve parameter: the k-th entry is the k-th
                derivative, in metres per (unit parameter)^k.
                A new sample is drawn if ``None``.
        """
        self.sampler = sampler
        self.randomgen = randomgen   # If not None, most likely fail with serialization
        self._consumers = []
        if sample:
            self._sample = sample
        else:
            self.resample()

    def __getstate__(self):
        """
        Return the picklable state, without the registry of curves.

        Weak references cannot be pickled, and the curves of a copied sample
        are not consumers of the copy.

        Returns:
            A shallow copy of ``__dict__`` without the consumer registry.
        """
        state = self.__dict__.copy()
        del state['_consumers']
        return state

    def __setstate__(self, state):
        """
        Restore the state written by :meth:`__getstate__` with an empty registry.

        Args:
            state: the dictionary returned by :meth:`__getstate__`.
        """
        self.__dict__.update(state)
        self._consumers = []

    def add_consumer(self, curve):
        """
        Register a curve whose geometry depends on this sample.

        Only a weak reference is kept, so registering does not extend the
        lifetime of ``curve``. Curves are compared by identity, so another curve
        that compares equal to ``curve`` stays registered and registering the
        same curve twice has no further effect.

        Args:
            curve: a :obj:`CurvePerturbed` (or any ``Optimizable``) to notify
                whenever :meth:`resample` draws a new sample.
        """
        if not any(ref() is curve for ref in self._consumers):
            self._consumers.append(weakref.ref(curve))

    def remove_consumer(self, curve):
        """
        Stop notifying ``curve`` when :meth:`resample` draws a new sample.

        Curves are compared by identity, so another curve that compares equal
        to ``curve`` stays registered. References to consumers that no longer
        exist are dropped as well.

        Args:
            curve: a consumer previously passed to :meth:`add_consumer`; a curve
                that is not registered is ignored.
        """
        self._consumers = [ref for ref in self._consumers
                           if (consumer := ref()) is not None and consumer is not curve]

    def resample(self):
        """
        Draw a new sample and invalidate every registered consumer.

        Each live consumer receives ``set_recompute_flag()``, which clears its
        cached geometry and propagates to its dependents, and
        ``_notify_non_dof_change()``, which clears results that a bell leaves in
        place when no DOF is free (e.g. the native cache of a fully fixed field).
        References to consumers that no longer exist are dropped.

        A new draw rings dependents exactly as a DOF change does, so
        :obj:`~simsopt.geo.surfaceobjectives.ToroidalFlux` and
        :obj:`~simsopt.geo.surfaceobjectives.QfmResidual` move a field they
        share to their surface points at once; another reader of that field
        must call ``set_points`` again after the draw.

        Call this method on every rank that holds the graph. As after a DOF
        change, the next access to a dependent
        :obj:`~simsopt.objectives.utilities.MPIOptimizable` calls ``allgather``,
        so a draw on only some of the ranks hangs.
        """
        self._sample = self.sampler.draw_sample(self.randomgen)
        live = [c for c in (ref() for ref in self._consumers) if c is not None]
        self._consumers = [weakref.ref(c) for c in live]
        for consumer in live:
            consumer.set_recompute_flag()
            consumer._notify_non_dof_change()

    def __getitem__(self, deriv):
        """
        Get the perturbation (if ``deriv=0``) or its ``deriv``-th derivative.
        """
        assert isinstance(deriv, int)
        if deriv >= len(self._sample):
            raise ValueError("""
The sample on has {len(self._sample)-1} derivatives.
Adjust the `n_derivs` parameter of the sampler to access higher derivatives.
""")
        return self._sample[deriv]


class CurvePerturbed(sopp.Curve, Curve):

    """A perturbed curve."""

    def __init__(self, curve, sample):
        r"""
        Perturb a underlying :mod:`simsopt.geo.curve.Curve` object by drawing a perturbation from a
        :obj:`GaussianSampler`.

        Comment:
        Doing anything involving randomness in a reproducible way requires care.
        Even more so, when doing things in parallel.
        Let's say we have a list of :mod:`simsopt.geo.curve.Curve` objects ``curves`` that represent a stellarator,
        and now we want to consider ``N`` perturbed stellarators. Let's also say we have multiple MPI ranks.
        To avoid the same thing happening on the different MPI ranks, we could pick a different seed on each rank.
        However, then we get different results depending on the number of MPI ranks that we run on. Not ideal.
        Instead, we should pick a new seed for each :math:`1\le i\le N`. e.g.

        .. code-block:: python

            from np.random import SeedSequence, PCG64DXSM, Generator
            import numpy as np
            curves = ...
            sigma = 0.01
            length_scale = 0.2
            sampler = GaussianSampler(curves[0].quadpoints, sigma, length_scale, n_derivs=1)
            globalseed = 1
            N = 10 # number of perturbed stellarators
            seeds = SeedSequence(globalseed).spawn(N)
            idx_start, idx_end = split_range_between_mpi_rank(N) # e.g. [0, 5) on rank 0, [5, 10) on rank 1
            perturbed_curves = [] # this will be a List[List[Curve]], with perturbed_curves[i] containing the perturbed curves for the i-th stellarator
            for i in range(idx_start, idx_end):
                rg = Generator(PCG64DXSM(seeds_sys[j]))
                stell = []
                for c in curves:
                    pert = PerturbationSample(sampler_systematic, randomgen=rg)
                    stell.append(CurvePerturbed(c, pert))
                perturbed_curves.append(stell)

        Args:
            curve: the underlying :obj:`~simsopt.geo.curve.Curve`.
            sample: the :obj:`PerturbationSample` added to ``curve``. It must be a
                :obj:`PerturbationSample`, because the curve registers with it to
                follow its :meth:`PerturbationSample.resample`. To apply a fixed,
                pre-drawn deviation, wrap it as
                ``PerturbationSample(sampler, sample=[d0, d1, ...])``.
        """
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        self._sample = sample
        sample.add_consumer(self)

    @property
    def sample(self):
        """
        The :obj:`PerturbationSample` added to the underlying curve.

        Assigning a sample invalidates the curve and its dependents (e.g. coils
        and fields) at once. An assignment or a new draw rings dependents exactly
        as a DOF change does, so
        :obj:`~simsopt.geo.surfaceobjectives.ToroidalFlux` and
        :obj:`~simsopt.geo.surfaceobjectives.QfmResidual` move a field they share
        to their surface points, and both must happen on every rank that holds
        the graph (see :meth:`PerturbationSample.resample`).
        """
        return self._sample

    @sample.setter
    def sample(self, sample):
        """
        Use ``sample`` as the perturbation of this curve.

        The curve leaves its previous sample and registers with ``sample``, so
        a later ``sample.resample()`` refreshes it and a resample of the
        previous sample no longer does.

        Args:
            sample: the new :obj:`PerturbationSample`.
        """
        self._sample.remove_consumer(self)
        self._sample = sample
        sample.add_consumer(self)
        self.set_recompute_flag()
        self._notify_non_dof_change()

    def resample(self):
        """
        Draw a new perturbation for the shared :obj:`PerturbationSample`.

        Equivalent to ``self.sample.resample()``: every curve built on the same
        sample, and everything that depends on those curves, is invalidated.
        """
        self.sample.resample()

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        assert quadpoints.shape[0] == self.curve.quadpoints.shape[0]
        assert np.linalg.norm(quadpoints - self.curve.quadpoints) < 1e-15
        gamma[:] = self.curve.gamma() + self.sample[0]

    def gammadash_impl(self, gammadash):
        gammadash[:] = self.curve.gammadash() + self.sample[1]

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = self.curve.gammadashdash() + self.sample[2]

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = self.curve.gammadashdashdash() + self.sample[3]

    def dgamma_by_dcoeff_vjp(self, v):
        return self.curve.dgamma_by_dcoeff_vjp(v)

    def dgammadash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadash_by_dcoeff_vjp(v)

    def dgammadashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdash_by_dcoeff_vjp(v)

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        return self.curve.dgammadashdashdash_by_dcoeff_vjp(v)
