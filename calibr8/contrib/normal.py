"""
This module implements reusable calibration models
with Normal distributions for the dependent variable.
"""
from typing import Optional, Tuple

from .. import core, utils
from . import noise


class BasePolynomialModelN(core.ContinuousUnivariateModel, noise.NormalNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        mu_degree: int,
        sigma_degree: int = 0,
        theta_names: Optional[Tuple[str]] = None,
    ):
        """Template for a model with polynomial trend (mu) and sigma (as a function of mu)
        with a normally distributed observation noise.

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        mu_degree : int
            degree of the polynomial model describing the trend (mu)
        sigma_degree : optional, int
            degree of the polynomial model describing the sigma as a function of mu
        theta_names : optional, tuple of str
            may be used to set the names of the model parameters
        """
        if mu_degree == 0:
            raise ValueError("0-degree (constant) mu calibration models are useless.")
        self.mu_degree = mu_degree
        self.sigma_degree = utils.check_scale_degree(sigma_degree)
        if theta_names is None:
            theta_names = tuple(f"mu_{d}" for d in range(mu_degree + 1)) + tuple(
                f"sigma_{d}" for d in range(sigma_degree + 1)
            )
        super().__init__(
            independent_key=independent_key, dependent_key=dependent_key, theta_names=theta_names
        )

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and sigma of a normal distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            values of the independent variable
        theta : optional, array-like
            parameter vector of the calibration model:
                [mu_degree] parameters for mu (lowest degree first)
                [sigma_degree] parameters for sigma (lowest degree first)

        Returns
        -------
        mu : array-like
            values for the mu parameter of a normal distribution describing the dependent variable
        sigma : array-like or float
            values for the sigma parameter of a normal distribution describing the dependent variable
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.polynomial(x, theta=theta[: self.mu_degree + 1])
        if self.sigma_degree == 0:
            sigma = theta[-1]
        else:
            sigma = core.polynomial(mu, theta=theta[self.mu_degree + 1 :])
        return mu, sigma

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
        theta : optional, array-like
            parameter vector of the calibration model:
                [mu_degree] parameters for mu (lowest degree first)
                [sigma_degree] parameters for sigma (lowest degree first)

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


class BaseAsymmetricLogisticN(core.ContinuousUnivariateModel, noise.NormalNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        sigma_degree: int = 0,
        theta_names: Optional[Tuple[str]] = None,
    ):
        """Template for a model with asymmetric logistic trend (mu) and polynomial sigma (as a function of mu).

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        sigma_degree : optional, int
            degree of the polynomial model describing the sigma as a function of mu
        theta_names : optional, tuple of str
            may be used to set the names of the model parameters
        """
        self.sigma_degree = utils.check_scale_degree(sigma_degree)
        if theta_names is None:
            theta_names = tuple("L_L,L_U,I_x,S,c".split(",")) + tuple(
                f"sigma_{d}" for d in range(sigma_degree + 1)
            )
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and sigma of a normal distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            values of the independent variable
        theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of asymmetric logistic model for mu
                [sigma_degree] parameters for sigma (lowest degree first)

        Returns
        -------
        mu : array-like
            values for the mu parameter of a normal distribution describing the dependent variable
        sigma : array-like or float
            values for the sigma parameter of a normal distribution describing the dependent variable
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.asymmetric_logistic(x, theta[:5])
        if self.sigma_degree == 0:
            sigma = theta[-1]
        else:
            sigma = core.polynomial(mu, theta[5:])
        return mu, sigma

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
        theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of asymmetric logistic model for mu
                [sigma_degree] parameters for sigma (lowest degree first)

        Returns
        -------
        x : array-like
            predicted independent values given the observations
        """
        if theta is None:
            theta = self.theta_fitted
        return core.inverse_asymmetric_logistic(y, theta[:5])


class BaseLogIndependentAsymmetricLogisticN(core.ContinuousUnivariateModel, noise.NormalNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        sigma_degree: int = 0,
        theta_names: Optional[Tuple[str]] = None,
    ):
        """Template for a model with asymmetric logistic trend (mu) and polynomial sigma (as a function of mu).

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        sigma_degree : optional, int
            degree of the polynomial model describing the sigma as a function of mu
        theta_names : optional, tuple of str
            may be used to set the names of the model parameters
        """
        self.sigma_degree = utils.check_scale_degree(sigma_degree)
        if theta_names is None:
            theta_names = tuple("L_L,L_U,log_I_x,S,c".split(",")) + tuple(
                f"sigma_{d}" for d in range(sigma_degree + 1)
            )
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and sigma of a normal distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            values of the independent variable
        theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of log-independent asymmetric logistic model for mu
                [sigma_degree] parameters for sigma (lowest degree first)

        Returns
        -------
        mu : array-like
            values for the mu parameter of a normal distribution describing the dependent variable
        sigma : array-like or float
            values for the sigma parameter of a normal distribution describing the dependent variable
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.xlog_asymmetric_logistic(x, theta[:5])
        if self.sigma_degree == 0:
            sigma = theta[-1]
        else:
            sigma = core.polynomial(mu, theta[5:])
        return mu, sigma

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
        theta : optional, array-like
            parameter vector of the calibration model:
                5 parameters of log-independent asymmetric logistic model for mu
                [sigma_degree] parameters for sigma (lowest degree first)

        Returns
        -------
        x : array-like
            predicted independent values given the observations
        """
        if theta is None:
            theta = self.theta_fitted
        return core.inverse_xlog_asymmetric_logistic(y, theta[:5])


class BaseExponentialModelN(core.ContinuousUnivariateModel, noise.NormalNoise):
    def __init__(
        self,
        *,
        independent_key: str,
        dependent_key: str,
        sigma_degree: int = 0,
        fixed_intercept: Optional[float] = None,
        theta_names: Optional[Tuple[str]] = None,
    ):
        """Template for a model with exponential trend (mu) and polynomial sigma (as a function of mu).

        Parameters
        ----------
        independent_key : str
            name of the independent variable
        dependent_key : str
            name of the dependent variable
        sigma_degree : optional, int
            degree of the polynomial model describing the sigma as a function of mu
            ⚠ Attention: for sigma_degree > 0, ensure that sigma is always positive!
        fixed_intercept : optional, float
            If set, the y-axis intercept will be fixed to this value.
            Otherwise the intercept becomes a free parameter.
        theta_names : optional, tuple of str
            may be used to set the names of the model parameters
        """
        self.sigma_degree = utils.check_scale_degree(sigma_degree)
        self.fixed_intercept = fixed_intercept
        if theta_names is None:
            if fixed_intercept is not None:
                theta_names = ("L", "k")
            else:
                theta_names = ("I", "L", "k")
            theta_names += tuple(f"sigma_{d}" for d in range(sigma_degree + 1))
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and sigma of a normal distribution which
        characterizes the dependent variable given values of the independent variable.

        Parameters
        ----------
        x : array-like
            values of the independent variable
        theta : optional, array-like
            Parameter vector of the calibration model.
            Depending on the ``fixed_intercept`` setting these are
            [I, L, k] or [L, k] parameters of exponential model for mu.
            Followed by parameters for the model for sigma (lowest degree first).

        Returns
        -------
        mu : array-like
            values for the mu parameter of a normal distribution describing the dependent variable
        sigma : array-like or float
            values for the sigma parameter of a normal distribution describing the dependent variable
        """
        if theta is None:
            theta = self.theta_fitted

        if self.fixed_intercept is None:
            theta_mu = (theta[0], theta[1], theta[2])
            theta_sigma = theta[3:]
        else:
            theta_mu = (self.fixed_intercept, theta[0], theta[1])
            theta_sigma = theta[2:]

        mu = core.exponential(x, theta_mu)
        if self.sigma_degree == 0:
            sigma = theta[-1]
        else:
            sigma = core.polynomial(mu, theta_sigma)
        return mu, sigma

    def predict_independent(self, y, *, theta=None):
        """Predict the independent variable using the inverse trend model.

        Parameters
        ----------
        y : array-like
            observations
        theta : optional, array-like
            Parameter vector of the calibration model.
            Depending on the ``fixed_intercept`` setting these are
            [I, L, k] or [L, k] parameters of exponential model for mu.
            Followed by parameters for the model for sigma (lowest degree first).

        Returns
        -------
        x : array-like
            predicted independent values given the observations
        """
        if theta is None:
            theta = self.theta_fitted

        if self.fixed_intercept is None:
            theta_mu = (theta[0], theta[1], theta[2])
        else:
            theta_mu = (self.fixed_intercept, theta[0], theta[1])
        return core.inverse_exponential(y, theta_mu)
