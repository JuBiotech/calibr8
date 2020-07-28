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


class BaseModelT(core.ErrorModel):
    def loglikelihood(self, *, y,  x, replicate_id=None, dependent_key=None, theta=None):
        """Loglikelihood of observation (dependent variable) given the independent variable

        Args:
            y (array): observed measurements (dependent variable)
            x (array or TensorVariable): assumed independent variable
            replicate_id(str): unique identifier for replicate (necessary for pymc3 likelihood)
            dependent_key(str): key of the dependent variable (necessary for pymc3 likelihood)
            theta: model parameters

        Returns:
            L (float or TensorVariable): sum of log-likelihoods
        """
        if theta is None:
            if self.theta_fitted is None:
                raise Exception('No parameter vector was provided and the model is not fitted with data yet.')
            theta = self.theta_fitted
        mu, scale, df = self.predict_dependent(x, theta=theta)
        if utils.istensor(x) or utils.istensor(theta):
            if not replicate_id:
                raise  ValueError(f'A replicate_id is required in tensor-mode.')
            if not dependent_key:
                raise  ValueError(f'A dependent_key is required in tensor-mode.')
            L = pm.StudentT(
                f'{replicate_id}.{dependent_key}',
                mu=mu,
                sigma=scale,
                nu=df,
                observed=y
            )
            return L
        elif isinstance(x, (list, numpy.ndarray)):
            # using t-distributed error in the non-transformed space
            loglikelihoods = scipy.stats.t.logpdf(x=y, loc=mu, scale=scale, df=df)
            return numpy.sum(loglikelihoods)
        else:
            raise Exception('Input x must either be a TensorVariable or an array-like object.')

    def infer_independent(self, y:typing.Union[int,float,numpy.ndarray], *, lower:float, upper:float, steps:int=300, percentiles:typing.Optional[typing.Tuple[float,float]]=None) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """Infer the posterior distribution of the independent variable given the observations of the dependent variable.
           The calculation is done numerically by integrating the likelihood in a certain interval [upper,lower]. 
           This is identical to the posterior with a Uniform (lower,upper) prior. If precentiles are provided, the interval of
           the PDF will be shortened.

        Args:
            y (int, float, array):  one or more obersevations at the same x
            lower (float):          lower limit for uniform distribution of prior
            upper (float):          upper limit for uniform distribution of prior
            steps (int):            steps between lower and upper or steps between the percentiles
            percentiles:
                    if tying.tuple[float,float]:
                                    pdf will be trimmed to the given percentiles; floats must be between 0 and 1
                    if None:
                                    the complete interval [upper,lower] will be retruned

          
        Returns:
            x:      values of the independent variable in the percentiles or in [lower, upper]
            pdf:    probability of the posterior distribution of the inferred independent variable
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
        if percentiles is not None:
            try:
                if len(percentiles) != 2:
                    raise Exception('Both a lower and upper percentile of the PDF must be provided to trim the PDF.')
            except:
                raise Exception(f'Wrong type of input. It should be a tuple, but is {type(percentiles)}')
            if not (percentiles[0] >= 0 and percentiles[1] <= 1 and percentiles[0] < percentiles[1]):
                raise Exception(f'The tuple of percentiles must be floats between 0 and 1 and sorted by value, but are {percentiles[0], percentiles[1]}.')

            x_integrate = numpy.linspace(lower, upper, 100_000)
            area_by_x = scipy.integrate.cumtrapz(likelihood(x_integrate, y), x_integrate, initial=0)
            prob_by_x = area_by_x / area_by_x[-1]
            i_025 = numpy.argmax(prob_by_x > percentiles[0])
            i_975 = numpy.argmax(prob_by_x > percentiles[1])
            x_dense = numpy.linspace(
                x_integrate[i_025],
                x_integrate[i_975-1],
                steps
            )
            pdf = likelihood(x_dense, y) / likelihood_integral
            return (x_dense, pdf)
        
        else:
            x_integrate = numpy.linspace(lower, upper, steps)
            pdf = likelihood(x_integrate, y) / likelihood_integral
            return (x_integrate, pdf)


class BasePolynomialModelT(BaseModelT):
    def __init__(self, *, independent_key:str, dependent_key:str, mu_degree:int, scale_degree:int=0, theta_names=None):
        if mu_degree == 0:
            raise Exception('0-degree (constant) mu error models are useless.')
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
        """Predicts the parameters mu and scale of a student-t-distribution which characterises the dependent variable
           given values of the independent variable.

        Args:
            x (array): values of the independent variable
            theta (array): parameter vector of the error model:
                [mu_degree] parameters for mu (lowest degree first)
                [scale_degree]  parameters for scale (lowest degree first)
                1 parameter for degree of freedom

        Returns:
            mu (array): values for the mu parameter of a student-t-distribution describing the dependent variable
            scale (scalar): values for the scale parameter of a student-t-distribution describing the dependent variable
            df (scalar): degree of freedom of student-t-distribution
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
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y (array): observations
            theta (array): parameter vector of the error model

        Returns:
            mu (array): predicted independent values given the observations
        """
        if theta is None:
            theta = self.theta_fitted
        if self.mu_degree > 1:
            raise NotImplementedError('Inverse prediction of higher order polynomials are not implemented.')        
        a, b = theta[:2]
        return (y - a) / b


class BaseAsymmetricLogisticT(BaseModelT):
    def __init__(self, *, independent_key:str, dependent_key:str, scale_degree:int=0, theta_names=None):
        self.scale_degree = scale_degree
        if theta_names is None:
            theta_names = tuple('L_L,L_U,I_x,S,c'.split(',')) + tuple(
                f'scale_{d}'
                for d in range(scale_degree + 1)
            ) + ('df',)
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """
        Predicts the parameters mu, scale, df of a student-t-distribution which characterises
        the dependent variable given values of the independent variable.

        Args:
            x (array): values of the independent variable
            theta (array): parameter vector of the error model:
                5 parameters of asymmetric logistic model (mu)
                scale_degree parameters of for scale
                1 parameter for degree of freedom

        Returns:
            mu (array): values for the mu parameter of a student-t-distribution describing the dependent variable
            scale (scalar): values for the scale parameter of a student-t-distribution describing the dependent variable
            df (scalar): degree of freedom of student-t-distribution
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
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y (array): observed measurements (dependent variable)
            theta (array): parameter vector of the error model

        Returns:
            x (array): most likely values (independent variable)
        """
        if theta is None:
            theta = self.theta_fitted
        return core.inverse_asymmetric_logistic(y, theta[:5])


class BaseLogIndependentAsymmetricLogisticT(BaseModelT):
    def __init__(self, *, independent_key:str, dependent_key:str, scale_degree:int=0, theta_names=None):
        self.scale_degree = scale_degree
        if theta_names is None:
            theta_names = tuple('L_L,L_U,log_I_x,S,c'.split(',')) + tuple(
                f'scale_{d}'
                for d in range(scale_degree + 1)
            ) + ('df',)
        super().__init__(independent_key, dependent_key, theta_names=theta_names)

    def predict_dependent(self, x, *, theta=None):
        """
        Predicts the parameters mu, scale, df of a student-t-distribution which characterises
        the dependent variable given values of the independent variable.

        Args:
            x (array): values of the independent variable
            theta (array): parameter vector of the error model:
                5 parameters of log-independent asymmetric logistic model (mu)
                scale_degree parameters of for scale
                1 parameter for degree of freedom

        Returns:
            mu (array): values for the mu parameter of a student-t-distribution describing the dependent variable
            scale (scalar): values for the scale parameter of a student-t-distribution describing the dependent variable
            df (scalar): degree of freedom of student-t-distribution
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
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y (array): observed measurements (dependent variable)
            theta (array): parameter vector of the error model

        Returns:
            x (array): most likely values (independent variable)
        """
        if theta is None:
            theta = self.theta_fitted
        return core.inverse_xlog_asymmetric_logistic(y, theta[:5])
