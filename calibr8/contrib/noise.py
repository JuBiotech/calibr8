import numpy
import scipy.stats

from ..core import DistributionMixin
from ..utils import HAS_PYMC, pm


class NormalNoise(DistributionMixin):
    """Normal noise, predicted in terms of mean and standard deviation."""

    scipy_dist = scipy.stats.norm
    pymc_dist = pm.Normal if HAS_PYMC else None

    @staticmethod
    def to_scipy(*params):
        return dict(loc=params[0], scale=params[1])

    @staticmethod
    def to_pymc(*params):
        return dict(mu=params[0], sigma=params[1])


class LaplaceNoise(DistributionMixin):
    """Normal noise, predicted in terms of mean and scale."""

    scipy_dist = scipy.stats.laplace
    pymc_dist = pm.Laplace if HAS_PYMC else None

    @staticmethod
    def to_scipy(*params):
        return dict(loc=params[0], scale=params[1])

    @staticmethod
    def to_pymc(*params):
        return dict(mu=params[0], b=params[1])


class LogNormalNoise(DistributionMixin):
    """Log-Normal noise, predicted in logarithmic mean and standard deviation.
    ⚠ This corresponds to the NumPy/Aesara/PyMC parametrization!
    """

    scipy_dist = scipy.stats.lognorm
    pymc_dist = pm.Lognormal if HAS_PYMC else None

    @staticmethod
    def to_scipy(*params):
        # SciPy wants linear scale mean and log scale standard deviation!
        return dict(scale=numpy.exp(params[0]), s=params[1])

    @staticmethod
    def to_pymc(*params):
        return dict(mu=params[0], sigma=params[1])


class StudentTNoise(DistributionMixin):
    """Student-t noise, predicted in terms of mean, scale and degree of freedom."""

    scipy_dist = scipy.stats.t
    pymc_dist = pm.StudentT if HAS_PYMC else None

    @staticmethod
    def to_scipy(*params):
        return dict(loc=params[0], scale=params[1], df=params[2])

    @staticmethod
    def to_pymc(*params):
        return dict(mu=params[0], sigma=params[1], nu=params[2])


class PoissonNoise(DistributionMixin):
    """Poisson noise, predicted in terms of mean."""

    scipy_dist = scipy.stats.poisson
    pymc_dist = pm.Poisson if HAS_PYMC else None

    @staticmethod
    def to_scipy(*params):
        return dict(mu=params[0])

    @staticmethod
    def to_pymc(*params):
        return dict(mu=params[0])
