"""
This module implements helper functions for a variety of tasks, including
imports, timestamp parsing and plotting.
"""
import datetime
from  collections.abc import Iterable
import matplotlib
from matplotlib import pyplot
import numpy
import scipy.stats
import typing

try:
    import theano
    import theano.tensor as tt
    HAS_THEANO = True
except ModuleNotFoundError:
    HAS_THEANO = False

try:
    import pymc3
    HAS_PYMC3 = True
except ModuleNotFoundError:
    HAS_PYMC3 = False


def parse_datetime(s: typing.Optional[str]) -> typing.Optional[datetime.datetime]:
    """ Parses a timezone-aware datetime formatted like 2020-08-05T13:37:00Z.

    Returns
    -------
    result : optional, datetime
        may be None when the input was None
    """
    if s is None:
        return None
    return datetime.datetime.strptime(s.replace('Z', '+0000'), '%Y-%m-%dT%H:%M:%S%z')


def format_datetime(dt: typing.Optional[datetime.datetime]) -> typing.Optional[str]:
    """ Formats a datetime like 2020-08-05T13:37:00Z.

    Returns
    -------
    result : optional, str
        may be None when the input was None
    """
    if dt is None:
        return None
    return dt.astimezone(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S%z').replace('+0000', 'Z')


def istensor(input:object) -> bool:
    """"Convenience function to test whether an input is a TensorVariable
        or if an input array or list contains TensorVariables.
    
    Parameters
    ----------
    input : object
        an object shat shall be analyzed
    
    Return
    ------
    result : bool
        Indicates if the object is or contains a TensorVariable.
    """
    if not HAS_THEANO:
        return False
    elif isinstance(input, str):
        return False
    elif isinstance(input, (tt.TensorVariable, tt.TensorConstant, theano.gof.graph.Variable)):
        return True
    elif isinstance(input, dict):
        for element in input.values():
            if istensor(element):
                return True  
    elif isinstance(input, Iterable):
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
    
    Parameters
    ----------
    ax : matplotlib.Axes
        subplot object to plot into
    independent : array-like
        x-values for the plot
    mu : array-like
        mu parameter of the Normal distribution
    scale : array-like
        scale parameter of the Normal distribution

    Returns
    -------
    artists : list of matplotlib.Artist
        the created artists (1x Line2D, 6x PolyCollection (alternating plot & legend))
    """
    artists = ax.plot(independent, mu, color='green')
    for q, c in zip([97.5, 95, 84], ['#d9ecd9', '#b8dbb8', '#9ccd9c']):
        percent = q - (100 - q)
        artists.append(ax.fill_between(independent,
            # by using the Percent Point Function (PPF), which is the inverse of the CDF,
            # the visualization will show symmetric intervals of <percent> probability
            scipy.stats.norm.ppf(1-q/100, loc=mu, scale=scale),
            scipy.stats.norm.ppf(q/100, loc=mu, scale=scale),
            alpha=.15, color='green'
        ))
        artists.append(ax.fill_between(
            [], [], [],
            color=c, label=f'{percent:.1f} % likelihood band'
        ))
    return artists


def plot_t_band(ax, independent, mu, scale, df):
    """Helper function for plotting the 68, 90 and 95 % likelihood-bands of a t-distribution.
    
    Parameters
    ----------
    ax : matplotlib.Axes
        subplot object to plot into
    independent : array-like
        x-values for the plot
    mu : array-like
        mu parameter of the t-distribution
    scale : array-like
        scale parameter of the t-distribution
    df : array-like
        density parameter of the t-distribution

    Returns
    -------
    artists : list of matplotlib.Artist
        the created artists (1x Line2D, 6x PolyCollection (alternating plot & legend))
    """
    artists = ax.plot(independent, mu, color='green')
    for q, c in zip([97.5, 95, 84], ['#d9ecd9', '#b8dbb8', '#9ccd9c']):
        percent = q - (100 - q)
        artists.append(ax.fill_between(independent,
            # by using the Percent Point Function (PPF), which is the inverse of the CDF,
            # the visualization will show symmetric intervals of <percent> probability
            scipy.stats.t.ppf(1-q/100, loc=mu, scale=scale, df=df),
            scipy.stats.t.ppf(q/100, loc=mu, scale=scale, df=df),
            alpha=.15, color='green'
        ))
        artists.append(ax.fill_between(
            [], [], [],
            color=c, label=f'{percent:.1f} % likelihood band'
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

    Parameters
    ----------
    vA : str
        first version number
    vB : str
        second version number

    Raises
    ------
    MajorMismatchException
        difference on the first level
    MinorMismatchException
        difference on the second level
    PatchMismatchException
        difference on the third level
    BuildMismatchException
        difference on the fourth level
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


def guess_asymmetric_logistic_theta(X, Y) -> typing.List[float]:
    """Creates an initial guess for the parameter vector of an `asymmetric_logistic` function.
    
    Parameters
    ----------
    X : array-like
        independent values of the data points
    Y : array-like
        dependent values (observations)
        
    Returns
    -------
    guess : list
        guess of the `asymmetric_logistic` parameters [L_L, L_U, I_x, S, c]
    """
    X = numpy.array(X)
    Y = numpy.array(Y)
    if not X.shape == Y.shape and len(X.shape) == 1:
        raise ValueError('X and Y must have the same 1-dimensional shape.')
    L_L = numpy.min(Y)
    L_U = numpy.max(Y) + numpy.ptp(Y)
    I_x = (min(X) + max(X)) / 2
    S, _ = numpy.polyfit(X, Y, deg=1)
    c = -1
    return [L_L, L_U, I_x, S, c]


def guess_asymmetric_logistic_bounds(X, Y, *, half_open=True) -> typing.List[typing.Tuple[float, float]]:
    """Creates bounds for the parameter vector of an `asymmetric_logistic` function.
    
    Parameters
    ----------
    X : array-like
        independent values of the data points
    Y : array-like
        dependent values (observations)
    half_open : bool
        sets whether the half-open bounds are allowed (e.g. for L_L and L_U)    

    Returns
    -------
    bounds : list
        bounds for the `asymmetric_logistic` parameters
    """
    X = numpy.array(X)
    Y = numpy.array(Y)
    if not X.shape == Y.shape and len(X.shape) == 1:
        raise ValueError('X and Y must have the same 1-dimensional shape.')
    slope, _ = numpy.polyfit(X, Y, deg=1)
    bounds = [
        # L_L
        (-numpy.inf if half_open else min(Y) - numpy.ptp(Y)*100, numpy.median(Y)),
        # L_U
        (numpy.median(Y), numpy.inf if half_open else max(Y) + numpy.ptp(Y)*100),
        # I_x
        (min(X) - 3 * numpy.ptp(X), max(X) + 3 * numpy.ptp(X)),
        # S
        (0, 10 * slope) if slope > 0 else (10 * slope, 0),
        # c
        (-5, 5)
    ]
    return bounds


def plot_model(
    model, *,
    fig:typing.Optional[matplotlib.figure.Figure] = None,
    axs:typing.Optional[typing.Sequence[matplotlib.axes.Axes]] = None,
):
    """Makes a plot of the model with its data.

    Parameters
    -----------
    model : ErrorModel
        A fitted error model with data.
        The predict_dependent method should return a tuple where the mean is the first entry.
    fig : optional, matplotlib.figure.Figure
        An existing figure (to be used in combination with [axs] argument).
    axs : optional, [matplotlib.axes.Axes]
        matplotlib subplots to use instead of creating new ones.

    Returns
    -------
    fig : matplotlib.figure.Figure
        the (created) figure
    axs : [matplotlib.axes.Axes]
        subplots of x-linear, x-log and residuals (xlog)
    """
    X = model.cal_independent
    Y = model.cal_dependent

    fig = None
    if axs is None:
        fig, axs = pyplot.subplots(ncols=3, figsize=(14,6), dpi=120)
    left, right, residuals = axs

    X_pred = numpy.exp(numpy.linspace(numpy.log(min(X)), numpy.log(max(X)), 1000))
    Y_pred = model.predict_dependent(X_pred)
    plot_t_band(left, X_pred, *Y_pred)
    plot_t_band(right, X_pred, *Y_pred)
    plot_t_band(residuals, X_pred, numpy.repeat(0, len(X_pred)), *Y_pred[1:])

    left.scatter(X, Y)
    right.scatter(X, Y)
    residuals.scatter(X, Y - model.predict_dependent(X)[0])

    left.set_ylabel(model.dependent_key)
    left.set_xlabel(model.independent_key)
    right.set_xlabel(model.independent_key)
    right.set_ylabel(model.dependent_key)
    if all(X > 0):
        right.set_xscale('log')
        right.set_xlim(numpy.min(X)*0.9, numpy.max(X)*1.1)
        residuals.set_xscale('log')
        residuals.set_xlim(numpy.min(X)*0.9, numpy.max(X)*1.1)
    residuals.set_xlabel(model.independent_key)
    residuals.set_ylabel('residuals')
    return fig, axs
