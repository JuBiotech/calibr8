"""
This module implements helper functions for a variety of tasks, including
imports, timestamp parsing and plotting.
"""
import datetime
import typing
import warnings
from collections.abc import Iterable
from typing import Optional, Sequence, Tuple

import matplotlib
import numpy
import scipy.stats
from matplotlib import pyplot


class ImportWarner:
    """Mock for an uninstalled package, raises `ImportError` when used."""

    __all__ = []

    def __init__(self, module_name):
        self.module_name = module_name

    def __getattr__(self, attr):
        raise ImportError(
            f"{self.module_name} is not installed. In order to use this function try:\npip install {self.module_name}"
        )


try:
    # Aesara
    import aesara
    from aesara.graph.basic import Variable

    HAS_TENSORS = True
except ModuleNotFoundError:
    # Aesara is not available
    try:
        # Theano-PyMC 1.1.2
        import theano
        from theano.graph.basic import Variable

        HAS_TENSORS = True
    except ModuleNotFoundError:
        HAS_TENSORS = False

tensor_types = (Variable,) if HAS_TENSORS else ()


try:
    try:
        import pymc3 as pm
    except ModuleNotFoundError:
        import pymc as pm
    HAS_PYMC = True
except ModuleNotFoundError:
    HAS_PYMC = False
    pm = ImportWarner("pymc")


def parse_datetime(s: typing.Optional[str]) -> typing.Optional[datetime.datetime]:
    """Parses a timezone-aware datetime formatted like 2020-08-05T13:37:00Z.

    Returns
    -------
    result : optional, datetime
        may be None when the input was None
    """
    if s is None:
        return None
    return datetime.datetime.strptime(s.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")


def format_datetime(dt: typing.Optional[datetime.datetime]) -> typing.Optional[str]:
    """Formats a datetime like 2020-08-05T13:37:00Z.

    Returns
    -------
    result : optional, str
        may be None when the input was None
    """
    if dt is None:
        return None
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z").replace("+0000", "Z")


def istensor(input: object) -> bool:
    """ "Convenience function to test whether an input is a TensorVariable
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
    if not HAS_TENSORS:
        return False
    elif isinstance(input, str):
        return False
    elif isinstance(input, tensor_types):
        return True
    elif isinstance(input, dict):
        for element in input.values():
            if istensor(element):
                return True
    elif isinstance(input, Iterable):
        if len(input) > 1:
            for element in input:
                if istensor(element):
                    return True
    return False


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
    warnings.warn(
        "`plot_norm_band` is substituted by a more general `plot_continuous_band` function."
        "It will be removed in a future release.",
        DeprecationWarning,
    )
    artists = ax.plot(independent, mu, color="green")
    for q, c in zip([97.5, 95, 84], ["#d9ecd9", "#b8dbb8", "#9ccd9c"]):
        percent = q - (100 - q)
        artists.append(
            ax.fill_between(
                independent,
                # by using the Percent Point Function (PPF), which is the inverse of the CDF,
                # the visualization will show symmetric intervals of <percent> probability
                scipy.stats.norm.ppf(1 - q / 100, loc=mu, scale=scale),
                scipy.stats.norm.ppf(q / 100, loc=mu, scale=scale),
                alpha=0.15,
                color="green",
            )
        )
        artists.append(ax.fill_between([], [], [], color=c, label=f"{percent:.1f} % likelihood band"))
    return artists


def plot_t_band(ax, independent, mu, scale, df, *, residual_type: typing.Optional[str] = None):
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
    residual_type : str, optional
        One of { None, "absolute", "relative" }.
        Specifies if bands are for no, absolute or relative residuals.

    Returns
    -------
    artists : list of matplotlib.Artist
        the created artists (1x Line2D, 6x PolyCollection (alternating plot & legend))
    """
    warnings.warn(
        "`plot_t_band` is substituted by a more general `plot_continuous_band` function."
        "It will be removed in a future release.",
        DeprecationWarning,
    )
    if residual_type:
        artists = ax.plot(independent, numpy.repeat(0, len(independent)), color="green")
    else:
        artists = ax.plot(independent, mu, color="green")
    for q, c in zip([97.5, 95, 84], ["#d9ecd9", "#b8dbb8", "#9ccd9c"]):
        percent = q - (100 - q)

        if residual_type == "absolute":
            mu = numpy.repeat(0, len(independent))
        lower = scipy.stats.t.ppf(1 - q / 100, loc=mu, scale=scale, df=df)
        upper = scipy.stats.t.ppf(q / 100, loc=mu, scale=scale, df=df)

        if residual_type == "relative":
            lower = (lower - mu) / mu
            upper = (upper - mu) / mu

        elif residual_type == "absolute" or residual_type is None:
            pass
        else:
            raise Exception(f'Only "relative" or "absolute" residuals supported. You passed {residual_type}')

        artists.append(
            ax.fill_between(
                independent,
                # by using the Percent Point Function (PPF), which is the inverse of the CDF,
                # the visualization will show symmetric intervals of <percent> probability
                lower,
                upper,
                alpha=0.15,
                color="green",
            )
        )
        artists.append(ax.fill_between([], [], [], color=c, label=f"{percent:.1f} % likelihood band"))
    return artists


def plot_continuous_band(ax, independent, model, residual_type: typing.Optional[str] = None):
    """Helper function for plotting the 68, 90 and 95 % likelihood-bands of a univariate distribution.

    Parameters
    ----------
    ax : matplotlib.Axes
        subplot object to plot into
    independent : array-like
        x-values for the plot
    model : CalibrationModel
        A fitted calibration model with data.
        The predict_dependent method should return a tuple where the mean is the first entry.
    residual_type : str, optional
        One of { None, "absolute", "relative" }.
        Specifies if bands are for no, absolute or relative residuals.

    Returns
    -------
    artists : list of matplotlib.Artist
        the created artists (1x Line2D, 6x PolyCollection (alternating plot & legend))
    """
    if not hasattr(model.scipy_dist, "ppf"):
        raise ValueError(
            "Only Scipy distributions with a ppf method can be used for the continuous likelihood bands."
        )
    params = model.predict_dependent(independent)
    median = model.scipy_dist.ppf(0.5, **model.to_scipy(*params))
    if residual_type:
        artists = ax.plot(independent, numpy.repeat(0, len(independent)), color="green")
    else:
        artists = ax.plot(independent, median, color="green")
    for q, c in zip([97.5, 95, 84], ["#d9ecd9", "#b8dbb8", "#9ccd9c"]):
        percent = q - (100 - q)

        if residual_type:
            lower = model.scipy_dist.ppf(1 - q / 100, **model.to_scipy(*params)) - median
            upper = model.scipy_dist.ppf(q / 100, **model.to_scipy(*params)) - median

            if residual_type == "relative":
                lower = (lower) / median
                upper = (upper) / median

        elif residual_type is None:
            lower = model.scipy_dist.ppf(1 - q / 100, **model.to_scipy(*params))
            upper = model.scipy_dist.ppf(q / 100, **model.to_scipy(*params))
        else:
            raise Exception(f'Only "relative" or "absolute" residuals supported. You passed {residual_type}')

        artists.append(
            ax.fill_between(
                independent,
                # by using the Percent Point Function (PPF), which is the inverse of the CDF,
                # the visualization will show symmetric intervals of <percent> probability
                lower,
                upper,
                alpha=0.15,
                color="green",
            )
        )
        artists.append(ax.fill_between([], [], [], color=c, label=f"{percent:.1f} % likelihood band"))
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


def assert_version_match(vA: str, vB: str):
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
        BuildMismatchException,
    )
    versions_A = vA.split(".")
    versions_B = vB.split(".")
    for ex, a, b in zip(level_exceptions, versions_A, versions_B):
        if int(a) != int(b):
            raise ex(f"{vA} != {vB}")
    return


def plot_model(
    model,
    *,
    fig: Optional[matplotlib.figure.Figure] = None,
    axs: Optional[Sequence[matplotlib.axes.Axes]] = None,
    residual_type="absolute",
    band_xlim: Tuple[Optional[float], Optional[float]] = (None, None),
):
    """Makes a plot of the model with its data.

    Parameters
    -----------
    model : CalibrationModel
        A fitted calibration model with data.
        The predict_dependent method should return a tuple where the mean is the first entry.
    fig : optional, matplotlib.figure.Figure
        An existing figure (to be used in combination with [axs] argument).
    axs : optional, [matplotlib.axes.Axes]
        matplotlib subplots to use instead of creating new ones.
    residual_type : optional, str
        Specifies if residuals are plotted absolutely or relatively.
    band_xlim : tuple
        Optional overrides for the minimum/maximum x coordinates
        of the likelihood bands in the left and center plots.
        Either entry can be `None` or a float.
        Defaults to the min/max of the calibration data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        the (created) figure
    axs : [matplotlib.axes.Axes]
        subplots of x-linear, x-log and residuals (xlog)
    """
    X = model.cal_independent
    Y = model.cal_dependent

    logscale = all(X > 0)
    xmin = min(X) if band_xlim[0] is None else band_xlim[0]
    xmax = max(X) if band_xlim[1] is None else band_xlim[1]
    if logscale:
        xband = numpy.exp(numpy.linspace(numpy.log(xmin or 1e-5), numpy.log(xmax), 300))
    else:
        xband = numpy.linspace(xmin, xmax, 300)

    fig = None
    if axs is None:
        # Create a figure where the two left subplots share an axis and the rightmost is independent.
        # The gridspecs are configured to make the margins work out nicely.
        fig = pyplot.figure(figsize=(12, 3.65), dpi=120)
        gs1 = fig.add_gridspec(1, 3, wspace=0.05, width_ratios=[1.125, 1.125, 1.5])
        gs2 = fig.add_gridspec(1, 3, wspace=0.5, width_ratios=[1.125, 1.125, 1.5])
        axs = []
        axs.append(fig.add_subplot(gs1[0, 0]))
        axs.append(fig.add_subplot(gs1[0, 1], sharey=axs[0]))
        pyplot.setp(axs[1].get_yticklabels(), visible=False)
        axs.append(fig.add_subplot(gs2[0, 2]))

    # ======= Left =======
    # Untransformed, outer range
    ax = axs[0]
    if hasattr(model.scipy_dist, "ppf"):
        plot_continuous_band(
            ax,
            xband,
            model,
            residual_type=None,
        )
    ax.scatter(X, Y)
    ax.set(
        ylabel=model.dependent_key,
        xlabel=model.independent_key,
    )

    # ======= Center =======
    # Transformed if possible, outer range
    ax = axs[1]
    if hasattr(model.scipy_dist, "ppf"):
        plot_continuous_band(
            ax,
            xband,
            model,
            residual_type=None,
        )
    ax.scatter(X, Y)
    ax.set(
        xlabel=model.independent_key,
        xscale="log" if logscale else "linear",
    )

    # ======= Center =======
    # Transformed if possible, data range
    ax = axs[2]
    if logscale:
        xresiduals = numpy.exp(numpy.linspace(numpy.log(min(X)), numpy.log(max(X)), 300))
    else:
        xresiduals = numpy.linspace(min(X), max(X), 300)
    if hasattr(model.scipy_dist, "ppf"):
        plot_continuous_band(
            ax,
            xresiduals,
            model,
            residual_type=residual_type,
        )

    if residual_type == "relative":
        ax.scatter(X, (Y - model.predict_dependent(X)[0]) / model.predict_dependent(X)[0])
    elif residual_type == "absolute":
        ax.scatter(X, Y - model.predict_dependent(X)[0])
    else:
        raise ValueError('Residual type must be "absolute" or "relative".')
    maxlim = max(numpy.abs(ax.get_ylim()))
    ax.set(
        ylabel=f"{residual_type} residuals",
        ylim=(-maxlim, maxlim),
        xlabel=model.independent_key,
        xscale="log" if logscale else "linear",
    )

    # Automatically change the xtick formatter on log-scaled subplots
    # if the data does not scale more than one order of magnitude.
    # This avoids ugly overlaps with the scientific notation.
    if logscale and numpy.ptp(numpy.log(X)) < 1:
        axs[1].xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        axs[2].xaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    return fig, axs


def check_scale_degree(scale_degree: int) -> int:
    """
    Evaluates user input for the scale degree and raises warning/error if the value is unexpected.
    Returns the user input.

    Parameters
    ----------
    scale_degree : int
        Degree of scale as set by the user.

    Returns
    -------
    scale_degree : int
        Degree of scale as set by the user.

    Raises
    ------
    ValueError
        scale_degree is None or negative
    UserWarning
        scale_degree is unexpectedly high
    """
    if scale_degree is None or scale_degree < 0:
        raise ValueError("Scale/sigma degree should be a natural number!")
    if scale_degree >= 2:
        warnings.warn("Scale/sigma degree >= 2 is quite unusual. Consider a lower value.", UserWarning)
    return scale_degree
