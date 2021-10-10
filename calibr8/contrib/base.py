"""
This module implements generic, reusable calibration models that can be subclassed to
implement custom calibration models.
"""
import numpy
import scipy
import typing
import warnings

from .. import core
from .. import utils

try:
    try:
        import pymc3 as pm
    except ModuleNotFoundError:
        import pymc as pm
except ModuleNotFoundError:
    pm = utils.ImportWarner('pymc3')


class BaseModelT(core.CalibrationModel):
    def loglikelihood(self, *, y, x, name: str=None, replicate_id: str=None, dependent_key: str=None, theta=None, **dist_kwargs):
        """ Loglikelihood of observation (dependent variable) given the independent variable.

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
            Name for the likelihood variable in a PyMC3 model (tensor mode).
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
            Additional keyword arguments are forwarded to the `pm.StudentT` distribution.
            Most prominent example: `dims`.

        Returns
        -------
        L : float or TensorVariable
            sum of log-likelihoods
        """
        if theta is None:
            if self.theta_fitted is None:
                raise Exception('No parameter vector was provided and the model is not fitted with data yet.')
            theta = self.theta_fitted

        if not isinstance(x, (list, numpy.ndarray, float, int)) and not utils.istensor(x):
            raise ValueError(
                f'Input x must be a scalar, TensorVariable or an array-like object, but not {type(x)}'
            )
        if not isinstance(y, (list, numpy.ndarray, float, int)) and not utils.istensor(x):
            raise ValueError(
                f'Input y must be a scalar or an array-like object, but not {type(y)}'
            )

        mu, scale, df = self.predict_dependent(x, theta=theta)
        if utils.istensor(x) or utils.istensor(theta):
            if pm.Model.get_context(error_if_none=False) is not None:
                if replicate_id and dependent_key:
                    warnings.warn(
                        "The `replicate_id` and `dependent_key` parameters are deprecated. Use `name` instead.",
                        DeprecationWarning
                    )
                    name = f'{replicate_id}.{dependent_key}'
                if not name:
                    raise ValueError("A `name` must be specified for the PyMC likelihood.")
                rv = pm.StudentT(
                    name,
                    mu=mu,
                    sigma=scale,
                    nu=df,
                    observed=y,
                    **dist_kwargs or {}
                )
            else:
                rv = pm.StudentT.dist(
                    mu=mu,
                    sigma=scale,
                    nu=df,
                    **dist_kwargs or {}
                )
            # The API to get log-likelihood tensors differs between PyMC versions
            if pm.__version__[0] == "3":
                if isinstance(rv, pm.model.ObservedRV):
                    return rv.logpt.sum()
                elif isinstance(rv, pm.Distribution):
                    return rv.logp(y).sum()
            else:
                return pm.logpt(rv, y, sum=True)
        else:
            # If `x` is given as a column vector, this model can broadcast automatically.
            # This gives considerable performance benefits for the `likelihood(..., scan_x=True)`
            # case which is relevant for `infer_independent`.
            return numpy.sum(scipy.stats.t.logpdf(x=y, loc=mu, scale=scale, df=df), axis=-1)


class BasePolynomialModelT(BaseModelT):
    def __init__(
        self, *,
        independent_key: str, dependent_key: str,
        mu_degree: int, scale_degree: int=0,
        theta_names: typing.Optional[typing.Tuple[str]]=None,
    ):
        """ Template for a model with polynomial trend (mu) and scale (as a function of mu).

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        mu_degree : int
            degree of the polynomial model describing the trend (mu)
        scale_degree : optional, int
            degree of the polynomial model describing the scale as a function of mu
        theta_names : optional, tuple of str
            may be used to set the names of the model parameters
        """
        if mu_degree == 0:
            raise Exception('0-degree (constant) mu calibration models are useless.')
        self.mu_degree = mu_degree
        self.scale_degree = scale_degree
        if theta_names is None:
            theta_names = tuple(
                f'mu_{d}'
                for d in range(mu_degree + 1)
            ) + tuple(
                f'scale_{d}'
                for d in range(scale_degree + 1)
            ) + ('df',)
        super().__init__(independent_key=independent_key, dependent_key=dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a student-t-distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            values of the independent variable
        theta : optional, array-like
            parameter vector of the calibration model:
                [mu_degree] parameters for mu (lowest degree first)
                [scale_degree] parameters for scale (lowest degree first)
                1 parameter for degree of freedom

        Returns
        -------
        mu : array-like
            values for the mu parameter of a student-t-distribution describing the dependent variable
        scale : array-like or float
            values for the scale parameter of a student-t-distribution describing the dependent variable
        df : float
            degree of freedom of student-t-distribution
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.polynomial(x, theta=theta[:self.mu_degree+1])
        if self.scale_degree == 0:
            scale = theta[-2]
        else:
            scale = core.polynomial(mu, theta=theta[self.mu_degree+1:self.mu_degree+1 + self.scale_degree+1])
        df = theta[-1]
        return mu, scale, df

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
        theta : optional, array-like
            parameter vector of the calibration model:
                [mu_degree] parameters for mu (lowest degree first)
                [scale_degree] parameters for scale (lowest degree first)
                1 parameter for degree of freedom

        Returns
        -------
        x : array-like
            predicted independent values given the observations
        """
        if theta is None:
            theta = self.theta_fitted
        if self.mu_degree > 1:
            raise NotImplementedError('Inverse prediction of higher order polynomials are not implemented.')      
        a, b = theta[:2]
        return (y - a) / b


class BaseAsymmetricLogisticT(BaseModelT):
    def __init__(
        self, *,
        independent_key:str, dependent_key:str,
        scale_degree:int=0,
        theta_names: typing.Optional[typing.Tuple[str]]=None,
    ):
        """ Template for a model with asymmetric logistic trend (mu) and polynomial scale (as a function of mu).

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        scale_degree : optional, int
            degree of the polynomial model describing the scale as a function of mu
        theta_names : optional, tuple of str
            may be used to set the names of the model parameters
        """
        self.scale_degree = scale_degree
        if theta_names is None:
            theta_names = tuple('L_L,L_U,I_x,S,c'.split(',')) + tuple(
                f'scale_{d}'
                for d in range(scale_degree + 1)
            ) + ('df',)
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a student-t-distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            values of the independent variable
        theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of asymmetric logistic model for mu
                [scale_degree] parameters for scale (lowest degree first)
                1 parameter for degree of freedom

        Returns
        -------
        mu : array-like
            values for the mu parameter of a student-t-distribution describing the dependent variable
        scale : array-like or float
            values for the scale parameter of a student-t-distribution describing the dependent variable
        df : float
            degree of freedom of student-t-distribution
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.asymmetric_logistic(x, theta[:5])
        if self.scale_degree == 0:
            scale = theta[-2]
        else:
            scale = core.polynomial(mu, theta[5:-1])
        df = theta[-1]
        return mu, scale, df

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
         theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of asymmetric logistic model for mu
                [scale_degree] parameters for scale (lowest degree first)
                1 parameter for degree of freedom

        Returns
        -------
        x : array-like
            predicted independent values given the observations
        """
        if theta is None:
            theta = self.theta_fitted
        return core.inverse_asymmetric_logistic(y, theta[:5])


class BaseLogIndependentAsymmetricLogisticT(BaseModelT):
    def __init__(
        self, *,
        independent_key:str, dependent_key:str,
        scale_degree:int=0,
        theta_names: typing.Optional[typing.Tuple[str]]=None,
    ):
        """ Template for a model with asymmetric logistic trend (mu) and polynomial scale (as a function of mu).

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        scale_degree : optional, int
            degree of the polynomial model describing the scale as a function of mu
        theta_names : optional, tuple of str
            may be used to set the names of the model parameters
        """
        self.scale_degree = scale_degree
        if theta_names is None:
            theta_names = tuple('L_L,L_U,log_I_x,S,c'.split(',')) + tuple(
                f'scale_{d}'
                for d in range(scale_degree + 1)
            ) + ('df',)
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a student-t-distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            values of the independent variable
        theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of log-independent asymmetric logistic model for mu
                [scale_degree] parameters for scale (lowest degree first)
                1 parameter for degree of freedom

        Returns
        -------
        mu : array-like
            values for the mu parameter of a student-t-distribution describing the dependent variable
        scale : array-like or float
            values for the scale parameter of a student-t-distribution describing the dependent variable
        df : float
            degree of freedom of student-t-distribution
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.xlog_asymmetric_logistic(x, theta[:5])
        if self.scale_degree == 0:
            scale = theta[-2]
        else:
            scale = core.polynomial(mu, theta[5:-1])
        df = theta[-1]
        return mu, scale, df

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
        theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of log-independent asymmetric logistic model for mu
                [scale_degree] parameters for scale (lowest degree first)
                1 parameter for degree of freedom

        Returns
        -------
        x : array-like
            predicted independent values given the observations
        """
        if theta is None:
            theta = self.theta_fitted
        return core.inverse_xlog_asymmetric_logistic(y, theta[:5])
