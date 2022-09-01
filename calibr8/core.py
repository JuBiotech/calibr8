"""
This module contains type definitions that generalize across all applications.

Also, it implements a variety of modeling functions such as polynomials,
or (asymmetric) logistic functions and their corresponding inverse functions.
"""
import datetime
import inspect
import json
import logging
import os
import typing
import warnings
from typing import Union

import numpy
import scipy

from . import utils
from .utils import pm

__version__ = "6.6.0"
_log = logging.getLogger("calibr8")


class InferenceResult:
    """Generic base type of independent value inference results."""


class ContinuousInference(InferenceResult):
    """Describes properties common for inference with continuous independent variables."""

    def __init__(
        self,
        eti_x: numpy.ndarray,
        eti_pdf: numpy.ndarray,
        eti_prob: float,
        hdi_x: numpy.ndarray,
        hdi_pdf: numpy.ndarray,
        hdi_prob: float,
    ) -> None:
        """The result of a numeric infer_independent operation.

        Parameters
        ----------
        eti_x : array
            Values of the independent variable in [eti_lower, eti_upper]
        eti_pdf : array
            Values of the posterior pdf at positions [eti_x]
        eti_prob : float
            Probability mass in the ETI
        hdi_x : array
            Values of the independent variable in [hdi_lower, hdi_upper]
        hdi_pdf : array
            Values of the posterior pdf at positions [hdi_x]
        hdi_prob : float
            Probability mass in the HDI
        """
        self._eti_x = eti_x
        self._eti_pdf = eti_pdf
        self._eti_prob = eti_prob
        self._hdi_x = hdi_x
        self._hdi_pdf = hdi_pdf
        self._hdi_prob = hdi_prob

    @property
    def eti_x(self) -> numpy.ndarray:
        """Values of the independent variable in the interval [eti_lower, eti_upper]"""
        return self._eti_x

    @property
    def eti_pdf(self) -> numpy.ndarray:
        """Values of the posterior probability density at the positions `eti_x`."""
        return self._eti_pdf

    @property
    def eti_lower(self) -> Union[float, numpy.ndarray]:
        """Lower bound of the ETI. This is the first value in `eti_x`."""
        return self._eti_x[..., 0]

    @property
    def eti_upper(self) -> Union[float, numpy.ndarray]:
        """Upper bound of the ETI. This is the last value in `eti_x`."""
        return self._eti_x[..., -1]

    @property
    def eti_width(self) -> Union[float, numpy.ndarray]:
        """Width of the ETI."""
        return self.eti_upper - self.eti_lower

    @property
    def eti_prob(self) -> float:
        """Probability mass of the given equal tailed interval."""
        return self._eti_prob

    @property
    def hdi_x(self) -> numpy.ndarray:
        """Values of the independent variable in the interval [hdi_lower, hdi_upper]"""
        return self._hdi_x

    @property
    def hdi_pdf(self) -> numpy.ndarray:
        """Values of the posterior probability density at the positions `hdi_x`."""
        return self._hdi_pdf

    @property
    def hdi_lower(self) -> Union[float, numpy.ndarray]:
        """Lower bound of the HDI. This is the first value in `hdi_x`."""
        return self._hdi_x[..., 0]

    @property
    def hdi_upper(self) -> Union[float, numpy.ndarray]:
        """Upper bound of the HDI. This is the last value in `hdi_x`."""
        return self._hdi_x[..., -1]

    @property
    def hdi_width(self) -> Union[float, numpy.ndarray]:
        """Width of the HDI."""
        return self.hdi_upper - self.hdi_lower

    @property
    def hdi_prob(self) -> float:
        """Probability mass of the given highest density interval."""
        return self._hdi_prob


class ContinuousUnivariateInference(ContinuousInference):
    """The result of a numeric infer_independent operation with a univariate model."""

    def __init__(
        self,
        median: float,
        **kwargs,
    ) -> None:
        """The result of a numeric infer_independent operation.

        Parameters
        ----------
        median : float
            x-value of the posterior median
        **kwargs : dict
            Forwarded to InferenceResult base constructor.
        """
        self._median = median
        super().__init__(**kwargs)

    @property
    def median(self) -> float:
        """Median of the posterior distribution. 50 % of the probability mass on either side."""
        return self._median

    def __repr__(self) -> str:
        result = (
            str(type(self))
            + f"\n    ETI ({numpy.round(self.eti_prob, 3) * 100:.1f} %): [{numpy.round(self.eti_lower, 4)}, {numpy.round(self.eti_upper, 4)}] Δ={round(self.eti_width, 4)}"
            + f"\n    HDI ({numpy.round(self.hdi_prob, 3) * 100:.1f} %): [{numpy.round(self.hdi_lower, 4)}, {numpy.round(self.hdi_upper, 4)}] Δ={round(self.hdi_width, 4)}"
        )
        return result


class ContinuousMultivariateInference(ContinuousInference):
    """The result of a multivariate independent variable inference."""


def _interval_prob(x_cdf: numpy.ndarray, cdf: numpy.ndarray, a: float, b: float):
    """Calculates the probability in the interval [a, b]."""
    ia = numpy.argmin(numpy.abs(x_cdf - a))
    ib = numpy.argmin(numpy.abs(x_cdf - b))
    return cdf[ib] - cdf[ia]


def _get_eti(x_cdf: numpy.ndarray, cdf: numpy.ndarray, ci_prob: float) -> typing.Tuple[float, float]:
    """Find the equal tailed interval (ETI) corresponding to a certain credible interval probability level.

    Parameters
    ----------
    x_cdf : numpy.ndarray
        Coordinates where the cumulative density function was evaluated
    cdf : numpy.ndarray
        Values of the cumulative density function at `x_cdf`
    ci_prob : float
        Desired probability level

    Returns
    -------
    eti_lower : float
        Lower bound of the ETI
    eti_upper : float
        Upper bound of the ETI
    """
    i_lower = numpy.argmin(numpy.abs(cdf - (1 - ci_prob) / 2))
    i_upper = numpy.argmin(numpy.abs(cdf - (1 + ci_prob) / 2))
    eti_lower = x_cdf[i_lower]
    eti_upper = x_cdf[i_upper]
    return eti_lower, eti_upper


def _get_hdi(
    x_cdf: numpy.ndarray,
    cdf: numpy.ndarray,
    ci_prob: float,
    guess_lower: float,
    guess_upper: float,
    *,
    history: typing.Optional[typing.DefaultDict[str, typing.List]] = None,
) -> typing.Tuple[float, float]:
    """Find the highest density interval (HDI) corresponding to a certain credible interval probability level.

    Parameters
    ----------
    x_cdf : numpy.ndarray
        Coordinates where the cumulative density function was evaluated
    cdf : numpy.ndarray
        Values of the cumulative density function at `x_cdf`
    ci_prob : float
        Desired probability level
    guess_lower : float
        Initial guess for the lower bound of the HDI
    guess_upper : float
        Initial guess for the upper bound of the HDI
    history : defaultdict of list, optional
        A defaultdict(list) may be passed to capture intermediate parameter and loss values
        during the optimization. Helps to understand, diagnose and test.

    Returns
    -------
    hdi_lower : float
        Lower bound of the HDI
    hdi_upper : float
        Upper bound of the HDI
    """

    def hdi_objective(x):
        a, d = x
        b = a + d

        prob = _interval_prob(x_cdf, cdf, a, b)
        delta_prob = numpy.abs(prob - ci_prob)

        if prob < ci_prob:
            # do not allow shrinking below the desired level
            L_prob = numpy.inf
            L_delta = 0
        else:
            # above the desired level penalize the interval width
            L_prob = 0
            L_delta = d

        L = L_prob + L_delta

        if history is not None:
            history["prob"].append(prob)
            history["delta_prob"].append(delta_prob)
            history["a"].append(a)
            history["b"].append(b)
            history["d"].append(d)
            history["L_prob"].append(L_prob)
            history["L_delta"].append(L_delta)
            history["L"].append(L)
        return L

    initial_guess = [guess_lower, guess_upper - guess_lower]
    if not numpy.isfinite(hdi_objective(initial_guess)):
        # Bad initial guess. Reset to the outer limimts.
        initial_guess = [x_cdf[0], x_cdf[-1] - x_cdf[0]]

    fit = scipy.optimize.fmin(
        hdi_objective,
        # parametrize as b=a+d
        x0=initial_guess,
        xtol=numpy.ptp(x_cdf) / len(x_cdf),
        disp=False,
    )
    hdi_lower, hdi_width = fit
    hdi_upper = hdi_lower + hdi_width
    return hdi_lower, hdi_upper


class DistributionMixin:
    """Maps the values returned by `CalibrationModel.predict_dependent`
    to a SciPy distribution and its parameters, and optionally also to
    a PyMC distribution and its parameters.
    """

    scipy_dist = None
    pymc_dist = None

    def to_scipy(*args) -> dict:
        raise NotImplementedError("This model does not implement a mapping to SciPy distribution parameters.")

    def to_pymc(*args) -> dict:
        raise NotImplementedError("This model does not implement a mapping to PyMC distribution parameters.")


def _inherits_noisemodel(cls):
    """Determines if cls is a sub-type of DistributionMixin that's not DistributionMixin or a calibration model."""
    for m in cls.__mro__:
        if (
            issubclass(m, DistributionMixin)
            and m is not DistributionMixin
            and not issubclass(m, CalibrationModel)
        ):
            return True
    return False


class CalibrationModel(DistributionMixin):
    """A parent class providing the general structure of a calibration model."""

    def __init__(
        self,
        independent_key: str,
        dependent_key: str,
        *,
        theta_names: typing.Tuple[str],
        ndim: int,
    ):
        """Creates a CalibrationModel object.

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        theta_names : optional, tuple of str
            names of the model parameters
        ndim : int
            Number of independent dimensions in the model.
        """
        if not _inherits_noisemodel(type(self)):
            warnings.warn(
                "This model does not implement a noise model yet."
                "\nAdd a noise model mixin to your class definition. For example:"
                "\n`class MyModel(CalibrationModel, LaplaceNoise)`",
                DeprecationWarning,
                stacklevel=2,
            )
        # make sure that the inheriting type has no required constructor (kw)args
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(
            type(self).__init__
        )
        n_defaults = 0 if not defaults else len(defaults)
        n_kwonlyargs = 0 if not kwonlyargs else len(kwonlyargs)
        n_kwonlydefaults = 0 if not kwonlydefaults else len(kwonlydefaults)
        if (len(args) - 1 > n_defaults) or (n_kwonlyargs > n_kwonlydefaults):
            raise TypeError("The constructor must not have any required (kw)arguments.")

        # underlying private attributes
        self.__theta_timestamp = None
        self.__theta_fitted = None

        # public attributes/properties
        self.ndim = ndim
        self.independent_key = independent_key
        self.dependent_key = dependent_key
        self.theta_names = theta_names
        self.theta_bounds = None
        self.theta_guess = None
        self.theta_fitted = None
        self.cal_independent: numpy.ndarray = None
        self.cal_dependent: numpy.ndarray = None
        super().__init__()

    @property
    def theta_fitted(self) -> typing.Optional[typing.Tuple[float]]:
        """The parameter vector that describes the fitted model."""
        return self.__theta_fitted

    @theta_fitted.setter
    def theta_fitted(self, value: typing.Optional[typing.Sequence[float]]):
        if value is not None:
            if numpy.shape(value) != numpy.shape(self.theta_names):
                raise ValueError(
                    f"The number of parameters ({len(value)}) "
                    f"does not match the number of parameter names ({len(self.theta_names)})."
                )
            self.__theta_fitted = tuple(value)
            self.__theta_timestamp = (
                datetime.datetime.now().astimezone(datetime.timezone.utc).replace(microsecond=0)
            )
        else:
            self.__theta_fitted = None
            self.__theta_timestamp = None

    @property
    def theta_timestamp(self) -> typing.Optional[datetime.datetime]:
        """The timestamp when `theta_fitted` was set."""
        return self.__theta_timestamp

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters of a probability distribution which characterises
           the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            numeric or symbolic independent variable
        theta : optional, array-like
            parameters of functions that model the parameters of the dependent variable distribution
            (defaults to self.theta_fitted)

        Returns
        -------
        parameters : array-like
            parameters characterizing the dependent variable distribution for given [x]
        """
        raise NotImplementedError(
            "The predict_dependent function should be implemented by the inheriting class."
        )

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
        theta : optional, array-like
            parameters of functions that model the parameters of the dependent variable distribution
            (defaults to self.theta_fitted)

        Returns
        -------
        x : array-like
            predicted independent values given the observations
        """
        raise NotImplementedError(
            "The predict_independent function should be implemented by the inheriting class."
        )

    def infer_independent(
        self,
        y: Union[int, float, numpy.ndarray],
        *,
        lower,
        upper,
    ) -> InferenceResult:
        """Infer the posterior distribution of the independent variable given the observations of the dependent variable.
        A Uniform (lower,upper) prior is applied.

        Parameters
        ----------
        y : int, float, array
            one or more observations at the same x
        lower : float
            lower limit for uniform distribution of prior
        upper : float
            upper limit for uniform distribution of prior

        Returns
        -------
        posterior : InferenceResult
            Result of the independent variable inference.
        """
        raise NotImplementedError(
            f"This calibration model does not implement an .infer_independent() method."
        )

    def loglikelihood(
        self,
        *,
        y,
        x,
        name: str = None,
        replicate_id: str = None,
        dependent_key: str = None,
        theta=None,
        **dist_kwargs,
    ):
        """Loglikelihood of observation (dependent variable) given the independent variable.

        If both x and y are 1D-vectors, they must have the same length and the likelihood will be evaluated elementwise.

        For a 2-dimensional `x`, the implementation *should* broadcast and return a result that has
        the same length as the first dimension of `x`.

        Parameters
        ----------
        y : scalar or array-like
            observed measurements (dependent variable)
        x : scalar, array-like or TensorVariable
            assumed independent variable
        name : str
            Name for the likelihood variable in a PyMC model (tensor mode).
            Previously this was `f'{replicate_id}.{dependent_key}'`.
        replicate_id : optional, str
            Deprecated; pass the `name` kwarg instead.
        dependent_key : optional, str
            Deprecated; pass the `name` kwarg instead.
        theta : optional, array-like
            Parameters for the calibration model to use instead of `theta_fitted`.
            The vector must have the correct length, but can have numeric and or symbolic entries.
            Use this kwarg to run MCMC on calibration model parameters.
        **dist_kwargs : dict
            Additional keyword arguments are forwarded to the PyMC distribution.
            Most prominent example: `dims`.

        Returns
        -------
        L : float or TensorVariable
            sum of log-likelihoods
        """
        if theta is None:
            if self.theta_fitted is None:
                raise Exception("No parameter vector was provided and the model is not fitted with data yet.")
            theta = self.theta_fitted

        if not isinstance(x, (list, numpy.ndarray, float, int)) and not utils.istensor(x):
            raise ValueError(
                f"Input x must be a scalar, TensorVariable or an array-like object, but not {type(x)}"
            )
        if not isinstance(y, (list, numpy.ndarray, float, int)) and not utils.istensor(x):
            raise ValueError(f"Input y must be a scalar or an array-like object, but not {type(y)}")

        params = self.predict_dependent(x, theta=theta)
        if utils.istensor(x) or utils.istensor(theta):
            if pm.Model.get_context(error_if_none=False) is not None:
                if replicate_id and dependent_key:
                    warnings.warn(
                        "The `replicate_id` and `dependent_key` parameters are deprecated. Use `name` instead.",
                        DeprecationWarning,
                    )
                    name = f"{replicate_id}.{dependent_key}"
                if not name:
                    raise ValueError("A `name` must be specified for the PyMC likelihood.")
                rv = self.pymc_dist(name, **self.to_pymc(*params), observed=y, **dist_kwargs or {})
            else:
                rv = self.pymc_dist.dist(**self.to_pymc(*params), **dist_kwargs or {})
            # The API to get log-likelihood tensors differs between PyMC versions
            if pm.__version__[0] == "3":
                if isinstance(rv, pm.model.ObservedRV):
                    return rv.logpt.sum()
                elif isinstance(rv, pm.Distribution):
                    return rv.logp(y).sum()
            else:
                return pm.joint_logpt(rv, y, sum=True)
        else:
            logp = None
            if hasattr(self.scipy_dist, "logpdf"):
                logp = self.scipy_dist.logpdf
            elif hasattr(self.scipy_dist, "logpmf"):
                logp = self.scipy_dist.logpmf
            else:
                raise NotImplementedError("No logpdf or logpmf methods found on {self.scipy_dist}.")
            return logp(y, **self.to_scipy(*params)).sum(axis=-1)

    def likelihood(self, *, y, x, theta=None, scan_x: bool = False):
        """Likelihood of observation (dependent variable) given the independent variable.

        Relies on the `loglikelihood` method.

        Parameters
        ----------
        y : scalar or array-like
            observed measurements (dependent variable)
        x : scalar, array-like or TensorVariable
            assumed independent variable
        theta : optional, array-like
            model parameters
        scan_x : bool
            When set to True, the method evaluates `likelihood(xi, y) for all xi in x`

        Returns
        -------
        L : float or TensorVariable
            sum of likelihoods
        """
        if scan_x:
            try:
                # Try to pass `x` as a column vector to benefit from broadcasting
                # if that's implemented by the underlying model.
                result = numpy.exp(self.loglikelihood(y=y, x=x[..., None], theta=theta))
                if not numpy.shape(result) == numpy.shape(x):
                    raise ValueError("The underlying model does not seem to implement broadcasting.")
                return result
            except:
                return numpy.exp([self.loglikelihood(y=y, x=xi, theta=theta) for xi in x])
        return numpy.exp(self.loglikelihood(y=y, x=x, theta=theta))

    def objective(self, independent, dependent, minimize=True) -> typing.Callable:
        """Creates an objective function for fitting to data.

        Parameters
        ----------
        independent : array-like
            numeric or symbolic values of the independent variable
        dependent : array-like
            observations of dependent variable
        minimize : bool
            switches between creation of a minimization (True) or maximization (False) objective function

        Returns
        -------
        objective : callable
            takes a numeric or symbolic parameter vector and returns the
            (negative) log-likelihood
        """

        def objective(x):
            L = self.loglikelihood(x=independent, y=dependent, theta=x)
            if minimize:
                return -L
            else:
                return L

        return objective

    def save(self, filepath: os.PathLike):
        """Save key properties of the calibration model to a JSON file.

        Parameters
        ----------
        filepath : path-like
            path to the output file
        """
        data = dict(
            calibr8_version=__version__,
            model_type=".".join([self.__module__, self.__class__.__qualname__]),
            theta_names=tuple(self.theta_names),
            theta_bounds=tuple(self.theta_bounds),
            theta_guess=tuple(self.theta_guess),
            theta_fitted=self.theta_fitted,
            theta_timestamp=utils.format_datetime(self.theta_timestamp),
            independent_key=self.independent_key,
            dependent_key=self.dependent_key,
            cal_independent=self.cal_independent.tolist() if self.cal_independent is not None else None,
            cal_dependent=self.cal_dependent.tolist() if self.cal_dependent is not None else None,
        )
        with open(filepath, "w") as jfile:
            json.dump(data, jfile, indent=4)
        return

    @classmethod
    def load(cls, filepath: os.PathLike):
        """Instantiates a model from a JSON file of key properties.

        Parameters
        ----------
        filepath : path-like
            path to the input file

        Raises
        ------
        MajorMismatchException
            when the major calibr8 version is different
        CompatibilityException
            when the model type does not match with the savefile

        Returns
        -------
        calibrationmodel : CalibrationModel
            the instantiated calibration model
        """
        with open(filepath, "r") as jfile:
            data = json.load(jfile)

        # check compatibility
        try:
            utils.assert_version_match(data["calibr8_version"], __version__)
        except (utils.BuildMismatchException, utils.PatchMismatchException, utils.MinorMismatchException):
            pass

        # create model instance
        cls_type = f"{cls.__module__}.{cls.__name__}"
        json_type = data["model_type"]
        if json_type != cls_type:
            raise utils.CompatibilityException(
                f"The model type from the JSON file ({json_type}) does not match this class ({cls_type})."
            )
        obj = cls()

        # assign essential attributes
        obj.independent_key = data["independent_key"]
        obj.dependent_key = data["dependent_key"]
        obj.theta_names = data["theta_names"]

        # assign additional attributes (check keys for backwards compatibility)
        obj.theta_bounds = tuple(map(tuple, data["theta_bounds"])) if "theta_bounds" in data else None
        obj.theta_guess = tuple(data["theta_guess"]) if "theta_guess" in data else None
        obj.__theta_fitted = tuple(data["theta_fitted"]) if "theta_fitted" in data else None
        obj.__theta_timestamp = utils.parse_datetime(data.get("theta_timestamp", None))
        obj.cal_independent = numpy.array(data["cal_independent"]) if "cal_independent" in data else None
        obj.cal_dependent = numpy.array(data["cal_dependent"]) if "cal_dependent" in data else None
        return obj


class ContinuousUnivariateModel(CalibrationModel):
    def __init__(self, independent_key: str, dependent_key: str, *, theta_names: typing.Tuple[str]):
        super().__init__(independent_key, dependent_key, theta_names=theta_names, ndim=1)

    def infer_independent(
        self,
        y: Union[int, float, numpy.ndarray],
        *,
        lower: float,
        upper: float,
        steps: int = 300,
        ci_prob: float = 1,
    ) -> ContinuousUnivariateInference:
        """Infer the posterior distribution of the independent variable given the observations of the dependent variable.
        The calculation is done numerically by integrating the likelihood in a certain interval [upper,lower].
        This is identical to the posterior with a Uniform (lower,upper) prior.

        Parameters
        ----------
        y : int, float, array
            One or more observations at the same x.
        lower : float
            Lower limit for uniform distribution of prior.
        upper : float
            Upper limit for uniform distribution of prior.
        steps : int
            Steps between lower and upper or steps between the percentiles (default 300).
        ci_prob : float
            Probability level for ETI and HDI credible intervals.
            If 1 (default), the complete interval [upper,lower] will be returned,
            else the PDFs will be trimmed to the according probability interval;
            float must be in the interval (0,1]

        Returns
        -------
        posterior : ContinuousUnivariateInference
            Result of the numeric posterior calculation.
        """
        y = numpy.atleast_1d(y)

        likelihood_integral, _ = scipy.integrate.quad(
            func=lambda x: self.likelihood(x=x, y=y),
            # by restricting the integral into the interval [a,b], the resulting PDF is
            # identical to the posterior with a Uniform(a, b) prior.
            # 1. prior probability is constant in [a,b]
            # 2. prior probability is 0 outside of [a,b]
            # > numerical integral is only computed in [a,b], but because of 1. and 2., it's
            #   identical to the integral over [-∞,+∞]
            a=lower,
            b=upper,
        )

        # high resolution x-coordinates for integration
        # the first integration is just to find the peak
        x_integrate = numpy.linspace(lower, upper, 10_000)
        area = scipy.integrate.cumtrapz(
            self.likelihood(x=x_integrate, y=y, scan_x=True), x_integrate, initial=0
        )
        cdf = area / area[-1]

        # now we find a high-resolution CDF for (1-shrink) of the probability mass
        shrink = 0.00001
        xfrom, xto = _get_eti(x_integrate, cdf, 1 - shrink)
        x_integrate = numpy.linspace(xfrom, xto, 100_000)
        area = scipy.integrate.cumtrapz(
            self.likelihood(x=x_integrate, y=y, scan_x=True), x_integrate, initial=0
        )
        cdf = (area / area[-1]) * (1 - shrink) + shrink / 2

        # TODO: create a smart x-vector from the CDF with varying stepsize

        if ci_prob != 1:
            if not isinstance(ci_prob, (int, float)) or not (0 < ci_prob <= 1):
                raise ValueError(
                    f"Unexpected `ci_prob` value of {ci_prob}. Expected float in interval (0, 1]."
                )

            # determine the interval bounds from the high-resolution CDF
            eti_lower, eti_upper = _get_eti(x_integrate, cdf, ci_prob)
            hdi_lower, hdi_upper = _get_hdi(x_integrate, cdf, ci_prob, eti_lower, eti_upper, history=None)

            eti_x = numpy.linspace(eti_lower, eti_upper, steps)
            hdi_x = numpy.linspace(hdi_lower, hdi_upper, steps)
            eti_pdf = self.likelihood(x=eti_x, y=y, scan_x=True) / likelihood_integral
            hdi_pdf = self.likelihood(x=hdi_x, y=y, scan_x=True) / likelihood_integral
            eti_prob = _interval_prob(x_integrate, cdf, eti_lower, eti_upper)
            hdi_prob = _interval_prob(x_integrate, cdf, hdi_lower, hdi_upper)
        else:
            x = numpy.linspace(lower, upper, steps)
            eti_x = hdi_x = x
            eti_pdf = hdi_pdf = self.likelihood(x=x, y=y, scan_x=True) / likelihood_integral
            eti_prob = hdi_prob = 1

        median = x_integrate[numpy.argmin(numpy.abs(cdf - 0.5))]

        return ContinuousUnivariateInference(
            median,
            eti_x=eti_x,
            eti_pdf=eti_pdf,
            eti_prob=eti_prob,
            hdi_x=hdi_x,
            hdi_pdf=hdi_pdf,
            hdi_prob=hdi_prob,
        )


class ContinuousMultivariateModel(CalibrationModel):
    def __init__(
        self, independent_key: str, dependent_key: str, *, theta_names: typing.Tuple[str], ndim: int
    ):
        super().__init__(independent_key, dependent_key, theta_names=theta_names, ndim=ndim)

    def infer_independent(
        self, y: Union[int, float, numpy.ndarray], *, lower, upper
    ) -> ContinuousMultivariateInference:
        return super().infer_independent(y, lower=lower, upper=upper)


def logistic(x, theta):
    """4-parameter logistic model.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        parameters of the logistic model
            I_x: x-value at inflection point
            I_y: y-value at inflection point
            Lmax: maximum value
            s: slope at the inflection point

    Returns
    -------
    y : array-like
        dependent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    x = numpy.array(x)
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2 * s / (Lmax - I_y) * (x - I_x)))
    return y


def inverse_logistic(y, theta):
    """Inverse 4-parameter logistic model.

    Parameters
    ----------
    y : array-like
            dependent variables
    theta : array-like
        parameters of the logistic model
            I_x: x-value at inflection point
            I_y: y-value at inflection point
            Lmax: maximum value
            s: slope at the inflection point

    Returns
    -------
    x : array-like
        independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.array(y)
    x = I_x - ((Lmax - I_y) / (2 * s)) * numpy.log((2 * (Lmax - I_y) / (y + Lmax - 2 * I_y)) - 1)
    return x


def asymmetric_logistic(x, theta):
    """5-parameter asymmetric logistic model.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        parameters of the logistic model
            L_L: lower asymptote
            L_U: upper asymptote
            I_x: x-value at inflection point
            S: slope at the inflection point
            c: symmetry parameter (0 is symmetric)

    Returns
    -------
    y : array-like
        dependent variable
    """
    L_L, L_U, I_x, S, c = theta[:5]
    # common subexpressions
    s0 = numpy.exp(c) + 1
    s1 = numpy.exp(-c)
    s2 = s0 ** (s0 * s1)
    # re-scale the inflection point slope with the interval
    s3 = S / (L_U - L_L)

    x = numpy.array(x)
    y = (numpy.exp(s2 * (s3 * (I_x - x) + c / s2)) + 1) ** -s1
    return L_L + (L_U - L_L) * y


def inverse_asymmetric_logistic(y, theta):
    """Inverse 5-parameter asymmetric logistic model.

    Parameters
    ----------
    y : array-like
        dependent variable
    theta : array-like
        parameters of the logistic model
            L_L: lower asymptote
            L_U: upper asymptote
            I_x: x-value at inflection point
            S: slope at the inflection point
            c: symmetry parameter (0 is symmetric)

    Returns
    -------
    x : array-like
        independent variable
    """
    L_L, L_U, I_x, S, c = theta[:5]
    # re-scale the inflection point slope with the interval
    s = S / (L_U - L_L)

    # re-scale into the interval [0, 1]
    y = numpy.array(y)
    y = (y - L_L) / (L_U - L_L)

    x0 = numpy.exp(c)
    x1 = x0 + 1
    x2 = -c
    x3 = numpy.exp(x2)
    x4 = I_x * s * x1**x3

    return -(x1 ** (-x1 * x3) * numpy.log(((1 / y) ** x0 - 1) * numpy.exp(-x0 * x4 + x2 - x4))) / s


def xlog_asymmetric_logistic(x, theta):
    """5-parameter asymmetric logistic model on log10 independent value.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        parameters of the logistic model
            L_L: lower asymptote
            L_U: upper asymptote
            log_I_x: log10(x)-value at x-logarithmic inflection point
            S: slope at the inflection point (Δy/Δlog10(x))
            c: symmetry parameter (0 is symmetric)

    Returns
    -------
    y : array-like
        dependent variable
    """
    L_L, L_U, log_I_x, S, c = theta[:5]
    # common subexpressions
    s0 = numpy.exp(c) + 1
    s1 = numpy.exp(-c)
    s2 = s0 ** (s0 * s1)
    # re-scale the inflection point slope with the interval
    s3 = S / (L_U - L_L)

    x = numpy.array(x)
    y = (numpy.exp(s2 * (s3 * (log_I_x - numpy.log10(x)) + c / s2)) + 1) ** -s1
    return L_L + (L_U - L_L) * y


def inverse_xlog_asymmetric_logistic(y, theta):
    """Inverse 5-parameter asymmetric logistic model on log10 independent value.

    Parameters
    ----------
    y : array-like
        dependent variable
    theta : array-like
        parameters of the logistic model
            L_L: lower asymptote
            L_U: upper asymptote
            log_I_x: log10(x)-value at x-logarithmic inflection point
            S: slope at the inflection point (Δy/Δlog10(x))
            c: symmetry parameter (0 is symmetric)

    Returns
    -------
    x : array-like
        independent variable
    """
    L_L, L_U, log_I_x, S, c = theta[:5]
    # re-scale the inflection point slope with the interval
    s = S / (L_U - L_L)

    # re-scale into the interval [0, 1]
    y = numpy.array(y)
    y = (y - L_L) / (L_U - L_L)

    x0 = numpy.exp(c)
    x1 = x0 + 1
    x2 = -c
    x3 = numpy.exp(x2)
    x4 = log_I_x * s * x1**x3

    x_hat = -(x1 ** (-x1 * x3) * numpy.log(((1 / y) ** x0 - 1) * numpy.exp(-x0 * x4 + x2 - x4))) / s
    return 10**x_hat


def log_log_logistic(x, theta):
    """4-parameter log-log logistic model.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        parameters of the log-log logistic model
            I_x: inflection point (ln(x))
            I_y: inflection point (ln(y))
            Lmax: logarithmic maximum value
            s: slope at the inflection point

    Returns
    -------
    y : array-like
        dependent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    x = numpy.log(x)
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2 * s / (Lmax - I_y) * (x - I_x)))
    return numpy.exp(y)


def inverse_log_log_logistic(y, theta):
    """4-parameter log-log logistic model.

    Parameters
    ----------
    y : array-like
        dependent variable
    theta : array-like
        parameters of the logistic model
            I_x: x-value at inflection point (ln(x))
            I_y: y-value at inflection point (ln(y))
            Lmax: maximum value in log space
            s: slope at the inflection point

    Returns
    -------
    x : array-like
        independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.log(y)
    x = I_x - ((Lmax - I_y) / (2 * s)) * numpy.log((2 * (Lmax - I_y) / (y + Lmax - 2 * I_y)) - 1)
    return numpy.exp(x)


def xlog_logistic(x, theta):
    """4-parameter x-log logistic model.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        parameters of the log-log logistic model
            I_x: inflection point (ln(x))
            I_y: inflection point (y)
            Lmax: maximum value
            s: slope at the inflection point

    Returns
    -------
    y : array-like
        dependent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    x = numpy.log(x)
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2 * s / (Lmax - I_y) * (x - I_x)))
    return y


def inverse_xlog_logistic(y, theta):
    """Inverse 4-parameter x-log logistic model.

    Parameters
    ----------
    y : array-like
        dependent variable
    theta : array-like
        parameters of the logistic model
            I_x: x-value at inflection point (ln(x))
            I_y: y-value at inflection point
            Lmax: maximum value
            s: slope at the inflection point

    Returns
    -------
    x : array-like
        independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.array(y)
    x = I_x - ((Lmax - I_y) / (2 * s)) * numpy.log((2 * (Lmax - I_y) / (y + Lmax - 2 * I_y)) - 1)
    return numpy.exp(x)


def ylog_logistic(x, theta):
    """4-parameter y-log logistic model.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        parameters of the log-log logistic model
            I_x: inflection point (x)
            I_y: inflection point (ln(y))
            Lmax: maximum value in log sapce
            s: slope at the inflection point

    Returns
    -------
    y : array-like
        dependent variables
    """
    I_x, I_y, Lmax, s = theta[:4]
    x = numpy.array(x)
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2 * s / (Lmax - I_y) * (x - I_x)))
    return numpy.exp(y)


def inverse_ylog_logistic(y, theta):
    """Inverse 4-parameter y-log logistic model.

    Parameters
    ----------
    y : array-like
        dependent variable
    theta : array-like
        parameters of the logistic model
            I_x: x-value at inflection point
            I_y: y-value at inflection point (ln(y))
            Lmax: maximum value in log space
            s: slope at the inflection point

    Returns
    -------
    x : array-like
        independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.log(y)
    x = I_x - ((Lmax - I_y) / (2 * s)) * numpy.log((2 * (Lmax - I_y) / (y + Lmax - 2 * I_y)) - 1)
    return x


def polynomial(x, theta):
    """Variable-degree polynomical model.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        polynomial coefficients (lowest degree first)

    Returns
    -------
    y : array-like
        dependent variable
    """
    # Numpy's polynomial function wants to get the highest degree first
    return numpy.polyval(theta[::-1], x)


def exponential(x, theta):
    """3-parameter exponential model.

    Parameters
    ----------
    x : array-like
        independent variable
    theta : array-like
        Parameters of the exponential model:
        - I: y-axis intercept
        - L: asymptotic limit
        - k: kinetic rate

    Returns
    -------
    y : array-like
        dependent variable
    """
    I, L, k = theta[:3]
    return (L - I) * (1 - numpy.exp(-k * x)) + I


def inverse_exponential(y, theta):
    """Inverse of 3-parameter exponential model.

    Parameters
    ----------
    y : array-like
        dependent variable
    theta : array-like
        Parameters of the exponential model:
        - I: y-axis intercept
        - L: asymptotic limit
        - k: kinetic rate

    Returns
    -------
    x : array-like
        independent variable
    """
    I, L, k = theta[:3]
    return -1 / k * numpy.log(1 - (y - I) / (L - I))
