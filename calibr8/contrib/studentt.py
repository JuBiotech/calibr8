"""
This module implements reusable calibration models
with Students-t distributions for the dependent variable.
"""
import typing
import warnings

from .. import core, utils
from . import noise


class BaseModelT(core.ContinuousUnivariateModel, noise.StudentTNoise):
    def __init__(self, independent_key: str, dependent_key: str, *, theta_names: typing.Tuple[str]):
        warnings.warn(
            "The`BaseModelT` class is deprecated. "
            "Inherit `core.ContinuousUnivariateModel, noise.StudentTNoise` directly.",
            FutureWarning,
        )
        super().__init__(
            independent_key=independent_key, dependent_key=dependent_key, theta_names=theta_names
        )


class BasePolynomialModelT(core.ContinuousUnivariateModel, noise.StudentTNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        mu_degree: int,
        scale_degree: int = 0,
        theta_names: typing.Optional[typing.Tuple[str]] = None,
    ):
        """Template for a model with polynomial trend (mu) and scale (as a function of mu)
        with a Student-t distributed observation noise.

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
            raise ValueError("0-degree (constant) mu calibration models are useless.")
        self.mu_degree = mu_degree
        self.scale_degree = utils.check_scale_degree(scale_degree)
        if theta_names is None:
            theta_names = (
                tuple(f"mu_{d}" for d in range(mu_degree + 1))
                + tuple(f"scale_{d}" for d in range(scale_degree + 1))
                + ("df",)
            )
        super().__init__(
            independent_key=independent_key, dependent_key=dependent_key, theta_names=theta_names
        )

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a Student-t distribution which
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
            values for the mu parameter of a Student-t distribution describing the dependent variable
        scale : array-like or float
            values for the scale parameter of a Student-t distribution describing the dependent variable
        df : float
            degree of freedom of Student-t distribution
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.polynomial(x, theta=theta[: self.mu_degree + 1])
        if self.scale_degree == 0:
            scale = theta[-2]
        else:
            scale = core.polynomial(
                mu, theta=theta[self.mu_degree + 1 : self.mu_degree + 1 + self.scale_degree + 1]
            )
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
            raise NotImplementedError("Inverse prediction of higher order polynomials are not implemented.")
        a, b = theta[:2]
        return (y - a) / b


class BaseAsymmetricLogisticT(core.ContinuousUnivariateModel, noise.StudentTNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        scale_degree: int = 0,
        theta_names: typing.Optional[typing.Tuple[str]] = None,
    ):
        """Template for a model with asymmetric logistic trend (mu) and polynomial scale (as a function of mu).

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
        self.scale_degree = utils.check_scale_degree(scale_degree)
        if theta_names is None:
            theta_names = (
                tuple("L_L,L_U,I_x,S,c".split(","))
                + tuple(f"scale_{d}" for d in range(scale_degree + 1))
                + ("df",)
            )
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a Student-t distribution which
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
            values for the mu parameter of a Student-t distribution describing the dependent variable
        scale : array-like or float
            values for the scale parameter of a Student-t distribution describing the dependent variable
        df : float
            degree of freedom of Student-t distribution
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


class BaseLogIndependentAsymmetricLogisticT(core.ContinuousUnivariateModel, noise.StudentTNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        scale_degree: int = 0,
        theta_names: typing.Optional[typing.Tuple[str]] = None,
    ):
        """Template for a model with asymmetric logistic trend (mu) and polynomial scale (as a function of mu).

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
        self.scale_degree = utils.check_scale_degree(scale_degree)
        if theta_names is None:
            theta_names = (
                tuple("L_L,L_U,log_I_x,S,c".split(","))
                + tuple(f"scale_{d}" for d in range(scale_degree + 1))
                + ("df",)
            )
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a Student-t distribution which
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
            values for the mu parameter of a Student-t distribution describing the dependent variable
        scale : array-like or float
            values for the scale parameter of a Student-t distribution describing the dependent variable
        df : float
            degree of freedom of Student-t distribution
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


class BaseExponentialModelT(core.ContinuousUnivariateModel, noise.StudentTNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        scale_degree: int = 0,
        fixed_intercept: typing.Optional[float] = None,
        theta_names: typing.Optional[typing.Tuple[str]] = None,
    ):
        """Template for a model with exponential trend (mu) and polynomial scale (as a function of mu).

        Parameters
        ----------
        independent_key : str
            Name of the independent variable.
        dependent_key : str
            Name of the dependent variable.
        scale_degree : optional, int
            Degree of the polynomial model describing the scale as a function of mu.
            ⚠ Attention: for scale_degree > 0, ensure that scale is always positive!
        fixed_intercept : optional, float
            If set, the y-axis intercept will be fixed to this value.
            Otherwise the intercept becomes a free parameter.
        theta_names : optional, tuple of str
            May be used to set the names of the model parameters.
        """
        self.scale_degree = utils.check_scale_degree(scale_degree)
        self.fixed_intercept = fixed_intercept
        if theta_names is None:
            if fixed_intercept is not None:
                theta_names = ("L", "k")
            else:
                theta_names = ("I", "L", "k")
            theta_names += tuple(f"scale_{d}" for d in range(scale_degree + 1)) + ("df",)
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and scale of a Student-t distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            Values of the independent variable.
        theta : optional, array-like
            Parameter vector of the calibration model.
            Depending on the ``fixed_intercept`` setting these are
            [I, L, k] or [L, k] parameters of exponential model for mu.
            Followed by parameters for the model for scale (lowest degree first).
            Followed by the degree of freedom parameter.

        Returns
        -------
        mu : array-like
            Values for the mu parameter of a Student-t distribution describing the dependent variable.
        scale : array-like or float
            Values for the scale parameter of a Student-t distribution describing the dependent variable.
        df : float
            Degree of freedom of Student-t distribution.
        """
        if theta is None:
            theta = self.theta_fitted

        if self.fixed_intercept is None:
            theta_mu = (theta[0], theta[1], theta[2])
            theta_scale = theta[3:-1]
        else:
            theta_mu = (self.fixed_intercept, theta[0], theta[1])
            theta_scale = theta[2:-1]
        assert len(theta_scale) == self.scale_degree + 1
        mu = core.exponential(x, theta_mu)
        if self.scale_degree == 0:
            scale = theta[-2]
        else:
            scale = core.polynomial(mu, theta_scale)
        df = theta[-1]
        return mu, scale, df

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            Observations
        theta : optional, array-like
            Parameter vector of the calibration model.
            Depending on the ``fixed_intercept`` setting these are
            [I, L, k] or [L, k] parameters of exponential model for mu.
            Followed by parameters for the model for scale (lowest degree first).
            Followed by the degree of freedom parameter.

        Returns
        -------
        x : array-like
            Predicted independent values given the observations.
        """
        if theta is None:
            theta = self.theta_fitted

        if self.fixed_intercept is None:
            theta_mu = (theta[0], theta[1], theta[2])
        else:
            theta_mu = (self.fixed_intercept, theta[0], theta[1])

        return core.inverse_exponential(y, theta_mu)
