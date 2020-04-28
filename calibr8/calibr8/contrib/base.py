import numpy  
import scipy

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
        mu, sigma, df = self.predict_dependent(x, theta=theta)
        if utils.istensor(x) or utils.istensor(theta):
            if not replicate_id:
                raise  ValueError(f'A replicate_id is required in tensor-mode.')
            if not dependent_key:
                raise  ValueError(f'A dependent_key is required in tensor-mode.')
            L = pm.StudentT(
                f'{replicate_id}.{dependent_key}',
                mu=mu,
                sigma=sigma,
                nu=df,
                observed=y
            )
            return L
        elif isinstance(x, (list, numpy.ndarray)):
            # using t-distributed error in the non-transformed space
            loglikelihoods = scipy.stats.t.logpdf(x=y, loc=mu, scale=sigma, df=df)
            return numpy.sum(loglikelihoods)
        else:
            raise Exception('Input x must either be a TensorVariable or an array-like object.')

    def infer_independent(self, y, *, lower, upper, draws=1000):
        """Infer the posterior distribution of the independent variable given the observations of one point of the dependent variable.
        
        Args:
            y (array): observed measurements
            lower (float): lower limit for uniform distribution of prior
            upper (float): lower limit for uniform distribution of prior
            draws (int): number of samples to draw (handed to pymc3.sample)
        
        Returns:
            trace: trace of the posterior distribution of inferred independent variable
        """ 
        theta = self.theta_fitted
        with pm.Model():
            prior = pm.Uniform(self.independent_key, lower=lower, upper=upper, shape=(1,))
            mu, scale, df = self.predict_dependent(prior, theta=theta)
            pm.StudentT('likelihood', nu=df, mu=mu, sigma=scale, observed=y, shape=(1,))
            trace = pm.sample(draws, cores=1)
        return trace


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
        """Predicts the parameters mu and sigma of a student-t-distribution which characterises the dependent variable
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
        a, b = self.theta_fitted[:2]
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
        return core.inverse_asymmetric_logistic(y, self.theta_fitted[:5])
