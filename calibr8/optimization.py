"""
The optimization module implements convenience functions for maximum
likelihood estimation of calibration model parameters.
"""
import logging
import typing

import numpy
import scipy.optimize

from . import core

_log = logging.getLogger("calibr8.optimization")


def _mask_and_warn_inf_or_nan(x: numpy.ndarray, y: numpy.ndarray, on: typing.Optional[str] = None):
    """Filters `x` and `y` such that only finite elements remain.

    Parameters
    ----------
    x : ndarray
        1-dimensional array of values, same shape as y
    y : ndarray
        1-dimensional array of values, same shape as x
    on : [None, "x", "y"]
        May be passed to filter on only x, or y instead of both (None, default).

    Returns
    -------
    x : array
    y : array
    """
    xdims = numpy.ndim(x)
    if xdims == 1:
        mask_x = numpy.isfinite(x)
    elif xdims == 2:
        mask_x = ~numpy.any(numpy.ma.masked_invalid(x).mask, axis=1)
    else:
        raise ValueError(f"The independent values are {numpy.ndim(x)}-dimensional. That's not supported.")
    mask_y = ~numpy.ma.masked_invalid(y).mask
    if on == "y":
        mask = mask_y
    elif on == "x":
        mask = mask_x
    else:
        mask = numpy.logical_and(mask_x, mask_y)
    if numpy.any(~mask):
        _log.warning("%d elements in x and y where dropped because they were inf or nan.", sum(~mask))
    return x[mask], y[mask]


def _warn_hit_bounds(theta, bounds, theta_names) -> bool:
    """Helper function that logs a warning for every parameter that hits a bound.

    Parameters
    ----------
    theta : array-like
        parameters
    bounds : list of (lb, ub)
        bounds
    theta_names : tuple
        corresponding parameter names

    Returns
    -------
    bound_hit : bool
        is True if at least one parameter was close to a bound
    """
    bound_hit = False
    for (ip, p), (lb, ub) in zip(enumerate(theta), bounds):
        pname = f"{ip+1}" if not theta_names else theta_names[ip]
        if numpy.isclose(p, lb):
            _log.warn(f"Parameter {pname} ({p}) is close to its lower bound ({lb}).")
            bound_hit = True
        if numpy.isclose(p, ub):
            _log.warn(f"Parameter {pname} ({p}) is close to its upper bound ({ub}).")
            bound_hit = True
    return bound_hit


def fit_scipy(
    model: core.CalibrationModel,
    *,
    independent: numpy.ndarray,
    dependent: numpy.ndarray,
    theta_guess: list,
    theta_bounds: list,
    minimize_kwargs: dict = None,
):
    """Function to fit the calibration model with observed data.

    Parameters
    ----------
    model : calibr8.CalibrationModel
        the calibration model to fit (inplace)
    independent : array-like
        desired values of the independent variable or measured values of the same
    dependent : array-like
        observations of dependent variable
    theta_guess : array-like
        initial guess for parameters describing the PDF of the dependent variable
    theta_bounds : array-like
        bounds to fit the parameters
    minimize_kwargs : dict
        keyword-arguments for scipy.optimize.minimize

    Returns
    -------
    theta : array-like
        best found parameter vector
    history : list
        history of the optimization
    """
    n_theta = len(model.theta_names)
    if len(theta_guess) != n_theta:
        raise ValueError(
            f"The length of theta_guess ({len(theta_guess)}) does not match the number of model parameters ({n_theta})."
        )
    if len(theta_bounds) != n_theta:
        raise ValueError(
            f"The length of theta_bounds ({len(theta_bounds)}) does not match the number of model parameters ({n_theta})."
        )

    if not minimize_kwargs:
        minimize_kwargs = {}

    independent_finite, dependent_finite = _mask_and_warn_inf_or_nan(independent, dependent)

    history = []
    fit = scipy.optimize.minimize(
        model.objective(independent=independent_finite, dependent=dependent_finite, minimize=True),
        x0=theta_guess,
        bounds=theta_bounds,
        callback=lambda x: history.append(x),
        **minimize_kwargs,
    )

    # check for fit success
    if theta_bounds:
        bound_hit = _warn_hit_bounds(fit.x, theta_bounds, model.theta_names)

    if not fit.success or bound_hit:
        _log.warning(f"Fit of {type(model).__name__} has failed:")
        _log.warning(fit)
    model.theta_bounds = theta_bounds
    model.theta_guess = theta_guess
    model.theta_fitted = fit.x
    model.cal_independent = numpy.array(independent)
    model.cal_dependent = numpy.array(dependent)
    return fit.x, history


def fit_scipy_global(
    model: core.CalibrationModel,
    *,
    independent: numpy.ndarray,
    dependent: numpy.ndarray,
    theta_bounds: list,
    method: str = None,
    maxiter: int = 5000,
    minimizer_kwargs: dict = None,
):
    """Function to fit the calibration model with observed data using global optimization.

    Parameters
    ----------
    model : calibr8.CalibrationModel
        the calibration model to fit (inplace)
    independent : array-like
        desired values of the independent variable or measured values of the same
    dependent : array-like
        observations of dependent variable
    theta_bounds : array-like
        bounds to fit the parameters
    method: str, optional
        Type of solver. Must be one of the following:
            - ``"dual_annealing"``
        If not given, defaults to ``"dual_annealing"``.
    maxiter: int, optional
        Maximum number of iterations of the dual_annealing solver.
        If not given, defaults to 5000.
    minimize_kwargs : dict
        keyword-arguments for scipy.optimize.minimize

    Returns
    -------
    theta : array-like
        best found parameter vector
    history : list
        history of the optimization, containing best parameter,
        objective value, and solver status

    Raises
    ------
    ValueError
        user input does not match number of parameters
    ValueError
        user specifies not supported optimization method
    Warning
        fit failed or bounds constrain optimization
    """

    n_theta = len(model.theta_names)
    if len(theta_bounds) != n_theta:
        raise ValueError(
            f"The length of theta_bounds ({len(theta_bounds)}) "
            "does not match the number of model parameters ({n_theta})."
        )

    if not minimizer_kwargs:
        minimizer_kwargs = {}

    independent_finite, dependent_finite = _mask_and_warn_inf_or_nan(independent, dependent)

    if method is None:
        method = "dual_annealing"
    if method.lower() != "dual_annealing":
        raise ValueError(
            f"`{method.lower()}` is not supported. The supported global optimization "
            "solver method is `dual_annealing`."
        )

    history = []

    fit = scipy.optimize.dual_annealing(
        model.objective(
            independent=independent_finite,
            dependent=dependent_finite,
            minimize=True,
        ),
        bounds=theta_bounds,
        callback=lambda x, f, context: history.append((x, f, context)),
        maxiter=maxiter,
    )

    # check for fit success
    if theta_bounds:
        bound_hit = _warn_hit_bounds(fit.x, theta_bounds, model.theta_names)

    if not fit.success or bound_hit:
        _log.warning(f"Fit of {type(model).__name__} has failed:")
        _log.warning(fit)
    model.theta_bounds = theta_bounds
    model.theta_fitted = fit.x
    model.cal_independent = numpy.array(independent)
    model.cal_dependent = numpy.array(dependent)
    return fit.x, history
