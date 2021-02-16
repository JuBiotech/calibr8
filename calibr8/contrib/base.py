"""
This module implements generic, reusable calibration models that can be subclassed to
implement custom calibration models.
"""
from collections import defaultdict
import numpy
import scipy
import typing

from .. import core
from .. import utils 

try:
    import theano
except ModuleNotFoundError:
    theano = utils.ImportWarner('theano')
try:
    import pymc3 as pm
except ModuleNotFoundError:
    pm = utils.ImportWarner('pymc3')


def _interval_prob(x_cdf: numpy.ndarray, cdf: numpy.ndarray, a: float, b: float):
    """Calculates the probability in the interval [a, b]."""
    ia = numpy.argmin(numpy.abs(x_cdf - a))
    ib = numpy.argmin(numpy.abs(x_cdf - b))
    return (cdf[ib] - cdf[ia])


def _get_eti(
    x_cdf: numpy.ndarray,
    cdf: numpy.ndarray,
    ci_prob: float
) -> typing.Tuple[float, float]:
    """ Find the equal tailed interval (ETI) corresponding to a certain credible interval probability level.

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
    history: typing.Optional[typing.DefaultDict[str, typing.List]]=None
) -> typing.Tuple[float]:
    """ Find the highest density interval (HDI) corresponding to a certain credible interval probability level.

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

    fit = scipy.optimize.fmin(
        hdi_objective,
        # parametrize as b=a+d
        x0=[guess_lower, guess_upper - guess_lower],
        xtol=numpy.ptp(x_cdf) / len(x_cdf),
        disp=False
    )
    hdi_lower, hdi_width = fit
    hdi_upper = hdi_lower + hdi_width
    return hdi_lower, hdi_upper

class BaseModelT(core.CalibrationModel):
    def loglikelihood(self, *, y, x, replicate_id: str=None, dependent_key: str=None, theta=None):
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
        if theta is None:
            if self.theta_fitted is None:
                raise Exception('No parameter vector was provided and the model is not fitted with data yet.')
            theta = self.theta_fitted
        mu, scale, df = self.predict_dependent(x, theta=theta)
        if utils.istensor(x) or utils.istensor(theta):
            if pm.Model.get_context(error_if_none=False) is not None:
                if not replicate_id:
                    raise ValueError(f'A replicate_id is required in tensor-mode.')
                if not dependent_key:
                    raise ValueError(f'A dependent_key is required in tensor-mode.')
                L = pm.StudentT(
                    f'{replicate_id}.{dependent_key}',
                    mu=mu,
                    sigma=scale,
                    nu=df,
                    observed=y
                )
            else:
                # TODO: broadcasting behaviour differs between numpy/theano API of loglikelihood function
                L = pm.StudentT.dist(
                    mu=mu,
                    sigma=scale,
                    nu=df,
                ).logp(y).sum()
            return L
        elif isinstance(x, (list, numpy.ndarray)):
            # using t-distributed noise in the non-transformed space
            loglikelihoods = scipy.stats.t.logpdf(x=y, loc=mu, scale=scale, df=df)
            return numpy.sum(loglikelihoods)
        else:
            raise Exception('Input x must either be a TensorVariable or an array-like object.')

    def infer_independent(
        self, y:typing.Union[int,float,numpy.ndarray], *, 
        lower:float, upper:float, steps:int=300, 
        ci_prob:float=1
    ) -> core.NumericPosterior:
        """Infer the posterior distribution of the independent variable given the observations of the dependent variable.
        The calculation is done numerically by integrating the likelihood in a certain interval [upper,lower]. 
        This is identical to the posterior with a Uniform (lower,upper) prior. If precentiles are provided, the interval of
        the PDF will be shortened.

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
        ci_prob : float
            The probability for equal tailed interval (ETI) and highest density interval (HDI).
            if 1 (default), the complete interval [upper,lower] will be returned, 
            else pdf will be trimmed to the according probability interval; 
            float must be in the interval (0,1]
                                
        Returns
        -------
        posterior : NumericPosterior
            the result of the numeric posterior calculation
        """  
        y = numpy.atleast_1d(y)

        def likelihood(x, y):
            loc, scale, df = self.predict_dependent(x)
            # get log-probs for all observations
            logpdfs = [
                scipy.stats.t.logpdf(y_, loc=loc, scale=scale, df=df)
                for y_ in y
            ]
            # sum them and exp them (numerically better than numpy.prod of pdfs)
            return numpy.exp(numpy.sum(logpdfs, axis=0))

        # high resolution x-coordinates for integration

        likelihood_integral, _ = scipy.integrate.quad(
            func=likelihood,
            # by restricting the integral into the interval [a,b], the resulting PDF is
            # identical to the posterior with a Uniform(a, b) prior.
            # 1. prior probability is constant in [a,b]
            # 2. prior probability is 0 outside of [a,b]
            # > numerical integral is only computed in [a,b], but because of 1. and 2., it's
            #   identical to the integral over [-∞,+∞]
                a=lower, b=upper,
                args=(y,)
            )

        # the first integration is just to find the peak
        x_integrate = numpy.linspace(lower, upper, 10_000)
        area = scipy.integrate.cumtrapz(likelihood(x_integrate, y), x_integrate, initial=0)
        cdf = area / area[-1]

        # now we find a high-resolution CDF for (1-shrink) of the probability mass
        shrink = 0.00001
        xfrom, xto = _get_eti(x_integrate, cdf, 1 - shrink)
        x_integrate = numpy.linspace(xfrom, xto, 100_000)
        area = scipy.integrate.cumtrapz(likelihood(x_integrate, y), x_integrate, initial=0)
        cdf = (area / area[-1]) * (1 - shrink) + shrink / 2

        # TODO: create a smart x-vector from the CDF with varying stepsize

        if ci_prob != 1:
            if not (0 < ci_prob <= 1):
                raise ValueError(f'Unexpected `ci_prob` value of {ci_prob}. Expected float in interval (0, 1].')

            # determine the interval bounds from the high-resolution CDF
            eti_lower, eti_upper = _get_eti(x_integrate, cdf, ci_prob)
            hdi_lower, hdi_upper = _get_hdi(x_integrate, cdf, ci_prob, eti_lower, eti_upper, history=None)

            eti_x = numpy.linspace(eti_lower, eti_upper, steps)
            hdi_x = numpy.linspace(hdi_lower, hdi_upper, steps)
            eti_pdf = likelihood(eti_x, y) / likelihood_integral
            hdi_pdf = likelihood(hdi_x, y) / likelihood_integral
            eti_prob = _interval_prob(x_integrate, cdf, eti_lower, eti_upper)
            hdi_prob = _interval_prob(x_integrate, cdf, hdi_lower, hdi_upper)
        else:
            x = numpy.linspace(lower, upper, steps)
            eti_x = hdi_x = x
            eti_pdf = hdi_pdf = likelihood(x, y) / likelihood_integral
            eti_prob = hdi_prob = 1

        median = x_integrate[numpy.argmin(numpy.abs(cdf - 0.5))]

        return core.NumericPosterior(
            median,
            eti_x, eti_pdf, eti_prob,
            hdi_x, hdi_pdf, hdi_prob,
        )


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
