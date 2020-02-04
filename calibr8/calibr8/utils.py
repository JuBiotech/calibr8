import collections
import numpy
import scipy.stats

try:
    import theano.tensor as tt
    HAS_THEANO = True
except ModuleNotFoundError:
    HAS_THEANO = False

try:
    import pymc3
    HAS_PYMC3 = True
except ModuleNotFoundError:
    HAS_PYMC3 = False


def istensor(input:object):
    """"Convenience function to test whether an input is a TensorVariable
        or if an input array or list contains TensorVariables.
    
    Args:
        input: object to be tested
    
    Return: 
        result(bool): Indicates if the object is or in any instance contains a TensorVariable.
    """
    if not HAS_THEANO:
        return False
    elif isinstance(input, str):
        return False
    elif isinstance(input, tt.TensorVariable):
        return True
    elif isinstance(input, dict):
        for element in input.values():
            if istensor(element):
                return True  
    elif isinstance(input, collections.Iterable):
        if len(input)>1:
            for element in input:
                if istensor(element):
                    return True
    return False


class ImportWarner:
    """Mock for an uninstalled package, raises `ImportError` when used."""
    __all__ = []

    def __init__(self, module_name):
        self.module_name = module_name

    def __getattr__(self, attr):
        raise ImportError(
            f'{self.module_name} is not installed. In order to use this function try:\npip install {self.module_name}'
        )


def plot_norm_band(ax, independent, mu, scale):
    """Helper function for plotting the 68, 90 and 95 % likelihood-bands of a Normal distribution.
    
    Args:
        ax (matplotlib.Axes): subplot object to plot into
        independent (array-like): x-values for the plot
        mu (array-like): mu parameter of the Normal distribution
        scale (array-like): scale parameter of the Normal distribution

    Returns:
        artists (list of matplotlib.Artist): the created artists (1x Line2D, 3x PolyCollection)
    """
    artists = ax.plot(independent, mu, color='green')
    for q in reversed([97.5, 95, 84]):
        percent = q - (100 - q)
        artists.append(ax.fill_between(independent,
            # by using the Percent Point Function (PPF), which is the inverse of the CDF,
            # the visualization will show symmetric intervals of <percent> probability
            scipy.stats.norm.ppf(1-q/100, loc=mu, scale=scale),
            scipy.stats.norm.ppf(q/100, loc=mu, scale=scale),
            alpha=.15, color='green', label=f'{percent:.1f} % likelihood band'
        ))
    return artists


def plot_t_band(ax, independent, mu, scale, df):
    """Helper function for plotting the 68, 90 and 95 % likelihood-bands of a t-distribution.
    
    Args:
        ax (matplotlib.Axes): subplot object to plot into
        independent (array-like): x-values for the plot
        mu (array-like): mu parameter of the t-distribution
        scale (array-like): scale parameter of the t-distribution
        df (array-like): density parameter of the t-distribution

    Returns:
        artists (list of matplotlib.Artist): the created artists (1x Line2D, 3x PolyCollection)
    """
    artists = ax.plot(independent, mu, color='green')
    for q in reversed([97.5, 95, 84]):
        percent = q - (100 - q)
        artists.append(ax.fill_between(independent,
            # by using the Percent Point Function (PPF), which is the inverse of the CDF,
            # the visualization will show symmetric intervals of <percent> probability
            scipy.stats.t.ppf(1-q/100, loc=mu, scale=scale, df=df),
            scipy.stats.t.ppf(q/100, loc=mu, scale=scale, df=df),
            alpha=.15, color='green', label=f'{percent:.1f} % likelihood band'
        ))
    return artists


class CompatibilityException(Exception):
    pass


class MajorMismatchException(CompatibilityException):
    pass


class MinorMismatchException(CompatibilityException):
    pass


class PatchMismatchException(CompatibilityException):
    pass


class BuildMismatchException(CompatibilityException):
    pass


def assert_version_match(vA:str, vB:str):
    """Compares two version numbers and raises exceptions that indicate where they missmatch.

    Args:
        vA (str): first version number
        vB (str): second version number

    Raises:
        MajorMismatchException: difference on the first level
        MinorMismatchException: difference on the second level
        PatchMismatchException: difference on the third level
        BuildMismatchException: difference on the fourth level
    """
    level_exceptions = (
        MajorMismatchException,
        MinorMismatchException,
        PatchMismatchException,
        BuildMismatchException
    )
    versions_A = vA.split('.')
    versions_B = vB.split('.')
    for ex, a, b in zip(level_exceptions, versions_A, versions_B):
        if int(a) != int(b):
            raise ex(f'{vA} != {vB}')
    return


def guess_asymmetric_logistic_theta(X, Y):
    """Creates an initial guess for the parameter vector of an `asymmetric_logistic` function.
    
    Args:
        X (array-like): independent values of the data points
        Y (array-like): dependent values (observations)
        
    Returns:
        [L_L, L_U, I_X, k, v] (list): guess of the `asymmetric_logistic` parameters
    """
    X = numpy.array(X)
    Y = numpy.array(Y)
    if not X.shape == Y.shape and len(X.shape) == 1:
        raise ValueError('X and Y must have the same 1-dimensional shape.')
    L_L = numpy.min(Y)
    L_U = numpy.max(Y) + numpy.ptp(Y)
    I_X = (min(X) + max(X)) / 2
    k, _ = numpy.polyfit(X, Y, deg=1) / 10
    v = 1
    return [L_L, L_U, I_X, k, v]
