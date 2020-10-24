"""
This module contains type definitions that generalize across all applications.

Also, it implements a variety of modeling functions such as polynomials,
or (asymmetric) logistic functions and their corresponding inverse functions.
"""
import abc
import datetime
import inspect
import json
import logging
import numpy
import os
import scipy.optimize
import typing

from . import utils


__version__ = '5.0.0'
_log = logging.getLogger('calibr8')


class NumericPosterior(typing.NamedTuple):
    """ The result of a numeric infer_independent operation.

    Attributes
    ----------
    x_dense : array
        values of the independent variable in the percentiles or in [lower, upper]
    pdf : array
        values of the posterior pdf at positions [x]
    median : float
        x-value of the posterior median
    hdi_prob: float
        highest density interval probability (0,1]
    lower : float
        x-value at the lower bound of the hdi
    upper : float
        x-value at the upper bound of the hdi
    """
    x_dense: numpy.ndarray
    pdf: numpy.ndarray
    median: float
    hdi_prob: float
    lower: float
    upper: float

    def __repr__(self) -> str:
        result = str(type(self))
        for name, value in zip(self._fields, self):
            result += f"\n    .{name}:\t{type(value).__name__}"
        return result


class ErrorModel:
    """A parent class providing the general structure of an error model."""
    
    def __init__(self, independent_key:str, dependent_key:str, *, theta_names:typing.Tuple[str]):
        """Creates an ErrorModel object.

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        theta_names : optional, tuple of str
            names of the model parameters
        """
        # make sure that the inheriting type has no required constructor (kw)args
        args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(type(self).__init__)
        n_defaults = 0 if not defaults else len(defaults)
        n_kwonlyargs = 0 if not kwonlyargs else len(kwonlyargs)
        n_kwonlydefaults = 0 if not kwonlydefaults else len(kwonlydefaults)
        if (len(args) - 1 > n_defaults) or (n_kwonlyargs > n_kwonlydefaults):
            raise TypeError('The constructor must not have any required (kw)arguments.')

        # underlying private attributes
        self.__theta_timestamp = None
        self.__theta_fitted = None

        # public attributes/properties
        self.independent_key = independent_key
        self.dependent_key = dependent_key
        self.theta_names = theta_names
        self.theta_bounds = None
        self.theta_guess = None
        self.theta_fitted = None
        self.cal_independent:numpy.ndarray = None
        self.cal_dependent:numpy.ndarray = None
        super().__init__()

    @property
    def theta_fitted(self) -> typing.Optional[typing.Tuple[float]]:
        """ The parameter vector that describes the fitted model.
        """
        return self.__theta_fitted

    @theta_fitted.setter
    def theta_fitted(self, value: typing.Optional[typing.Sequence[float]]):
        if value is not None:
            self.__theta_fitted = tuple(value)
            self.__theta_timestamp = datetime.datetime.utcnow().astimezone(datetime.timezone.utc).replace(microsecond=0)
        else:
            self.__theta_fitted = None
            self.__theta_timestamp = None

    @property
    def theta_timestamp(self) -> typing.Optional[datetime.datetime]:
        """ The timestamp when `theta_fitted` was set.
        """
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
        raise NotImplementedError('The predict_dependent function should be implemented by the inheriting class.')
    
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
        raise NotImplementedError('The predict_independent function should be implemented by the inheriting class.')

    def infer_independent(
        self, y:typing.Union[int,float,numpy.ndarray], *, 
        lower:float, upper:float, steps:int=300, 
        hdi_prob:float=1
    ) -> NumericPosterior:
        """Infer the posterior distribution of the independent variable given the observations of the dependent variable.

        The calculation is done numerically by integrating the likelihood in a certain interval [upper, lower].
        This is identical to the posterior with a Uniform (lower, upper) prior. If a hdi_prob < 1 is specified,
        the interval of the PDF will be shortened.

        Parameters
        ----------
        y : int, float, array
            one or more observations at the same x
        lower : float
            lower limit for uniform distribution of prior
        upper : float
            upper limit for uniform distribution of prior
        steps : int
            steps between lower and upper or steps between the percentiles (default 300)
        hdi_prob : float
            if 1 (default), the complete interval [upper,lower] will be returned, 
            else pdf will be trimmed to the according probability interval; 
            float must be in the interval (0,1]
                                
        Returns
        -------
        posterior : NumericPosterior
            the result of the numeric posterior calculation
        """  
        raise NotImplementedError('The infer_independent function should be implemented by the inheriting class.')

    def loglikelihood(self, *, y, x, theta=None):
        """Loglikelihood of observation (dependent variable) given the independent variable

        Parameters
        ----------
        y : array-like
            observed measurements (dependent variable)
        x : array-like or TensorVariable
            assumed independent variable
        replicate_id : optional, str
            unique identifier for replicate (necessary for pymc3 likelihood)
        dependent_key : optional, str
            key of the dependent variable (necessary for pymc3 likelihood)
        theta : optional, array-like
            model parameters

        Returns
        -------
        L : float or TensorVariable
            sum of log-likelihoods
        """
        raise NotImplementedError('The loglikelihood function should be implemented by the inheriting class.')

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
        """Save key properties of the error model to a JSON file.

        Parameters
        ----------
        filepath : path-like
            path to the output file
        """
        data = dict(
            calibr8_version=__version__,
            model_type='.'.join([self.__module__, self.__class__.__qualname__]),
            theta_names=tuple(self.theta_names),
            theta_bounds=tuple(self.theta_bounds),
            theta_guess=tuple(self.theta_guess),
            theta_fitted=self.theta_fitted,
            theta_timestamp=utils.format_datetime(self.theta_timestamp),
            independent_key=self.independent_key,
            dependent_key=self.dependent_key,
            cal_independent=tuple(self.cal_independent) if self.cal_independent is not None else None,
            cal_dependent=tuple(self.cal_dependent) if self.cal_dependent is not None else None,
        )
        with open(filepath, 'w') as jfile:
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
        errormodel : ErrorModel
            the instantiated error model
        """
        with open(filepath, 'r') as jfile:
            data = json.load(jfile)
        
        # check compatibility
        try:
            utils.assert_version_match(data['calibr8_version'], __version__)
        except (utils.BuildMismatchException, utils.PatchMismatchException, utils.MinorMismatchException):
            pass

        # create model instance
        cls_type = f'{cls.__module__}.{cls.__name__}'
        json_type = data['model_type']
        if json_type != cls_type:
            raise utils.CompatibilityException(f'The model type from the JSON file ({json_type}) does not match this class ({cls_type}).')
        obj = cls()

        # assign essential attributes
        obj.independent_key = data['independent_key']
        obj.dependent_key = data['dependent_key']
        obj.theta_names = data['theta_names']

        # assign additional attributes (check keys for backwards compatibility)
        obj.theta_bounds = tuple(map(tuple, data['theta_bounds'])) if 'theta_bounds' in data else None
        obj.theta_guess = tuple(data['theta_guess']) if 'theta_guess' in data else None
        obj.__theta_fitted = tuple(data['theta_fitted']) if 'theta_fitted' in data else None
        obj.__theta_timestamp = utils.parse_datetime(data.get('theta_timestamp', None))
        obj.cal_independent = numpy.array(data['cal_independent']) if 'cal_independent' in data else None
        obj.cal_dependent = numpy.array(data['cal_dependent']) if 'cal_dependent' in data else None
        return obj


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
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))
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
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
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
    return L_L + (L_U-L_L) * y


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
    x4 = I_x*s*x1**x3
    
    return - (x1**(-x1*x3) * numpy.log( ((1/y)**x0 - 1) * numpy.exp(-x0*x4+x2-x4) ) ) / s


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
    return L_L + (L_U-L_L) * y


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

    x_hat = - (x1**(-x1*x3) * numpy.log( ((1/y)**x0 - 1) * numpy.exp(-x0*x4+x2-x4) ) ) / s
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
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))
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
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
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
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))
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
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
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
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))
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
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
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
