import abc
import numpy
import scipy.optimize
import sys

HAVE_PYMC3 = False
HAVE_THEANO = False

try:
    import pymc3 as pm
    HAVE_PYMC3 = True

except ModuleNotFoundError:  # pymc3 is optional, throw exception when used
    class _ImportWarnerPyMC3:
        __all__ = []

        def __init__(self, attr):
            self.attr = attr

        def __call__(self, *args, **kwargs):
            raise ImportError(
                "PyMC3 is not installed. In order to use this function:\npip install pymc3"
            )

    class _PyMC3:
        def __getattr__(self, attr):
            return _ImportWarnerPyMC3(attr)
    
    pm = _PyMC3()

try:
    import theano
    HAVE_THEANO = True
except ModuleNotFoundError:  # theano is optional, throw exception when used

    class _ImportWarnerTheano:
        __all__ = []

        def __init__(self, attr):
            self.attr = attr

        def __call__(self, *args, **kwargs):
            raise ImportError(
                "Theano is not installed. In order to use this function:\npip install theano"
            )

    class _Theano:
        def __getattr__(self, attr):
            return _ImportWarnerTheano(attr)
    
    theano = _Theano()

from .. import core 
from .. import utils


class BiomassErrorModel(core.ErrorModel):
    def __init__(self, independent_key:str, dependent_key:str):
        """ A class for modeling the error of backscatter measurements of biomass.

        Args:
            independent: biomass (cell dry weight)
            dependent: backscatter measurements
        """
        super().__init__(independent_key, dependent_key)
        self.student_df=1
          
    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters mu and sigma of a student-t-distribution which characterises the dependent variable (backscatter)
        given values of the independent variable (CDW).

        Args:
            x (array): independent variable (CDW)
            theta: parameters of asymmetric_logistic (mu) and and polynomial functions (scale)

        Returns:
            mu,sigma (array): values for mu and sigma charcterising the student-t-distributions describing the dependent variable (backscatter)
            df: degree of freedom of student-t-distribution (always set to 1)
        """
        if theta is None:
            theta = self.theta_fitted
        mu = core.asymmetric_logistic(x, theta[:5])
        sigma = core.polynomial(mu, theta[5:])
        df = self.student_df
        return mu, sigma, df

    def predict_independent(self, y):
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y (array): observed backscatter measurements (dependent variable)

        Returns:
            biomass (array): most likely biomass values (independent variable)
        """
        x = core.inverse_asymmetric_logistic(y, self.theta_fitted)
        return x
        
    def theano_asymmetric_logistic(self, x, theta):
        """5-parameter logistic model of the expected measurement outcome, given a true independent variable.
    
        Args:
            x (theano.TensorVariable): symbolic independent variable
            theta (array): parameters of the logistic model
                L_L: lower asymptote
                L_U: upper asymptote
                I_x: x-value at inflection point
                k: growth rate
                v: symmetry parameter
        
        Returns:
            y (theano.TensorVariable): symbolic expected measurement outcome
        """
        L_L, L_U, I_x, k, v = theta[:5]
        y = L_L + (L_U-L_L)/(theano.tensor.power((1+theano.tensor.exp(-k*(x-I_x))),1/v))
        return y


    def infer_independent(self, y, *, cdw_lower=0, cdw_upper=17, draws=1000):
        """Infer the posterior distribution of the independent variable given the observations of one point of the dependent variable.
        
        Args:
            y (array): observed backscatter measurements
            cdw_lower (int): lower limit for uniform distribution of cdw prior
            cdw_upper (int): upper limit for uniform distribution of cdw prior
            draws (int): number of samples to draw (handed to pymc3.sample)
        
        Returns:
            trace: trace of the posterior distribution of inferred biomass concentration
        """ 
        theta = self.theta_fitted
        with pm.Model() as model:
            cdw = pm.Uniform('CDW', lower=cdw_lower, upper=cdw_upper, shape=(1,))
            mu = self.theano_asymmetric_logistic(cdw, theta[:5])
            sd = core.polynomial(cdw, theta[5:])
            ll = pm.StudentT('likelihood', nu=self.student_df, mu=mu, sd=sd, observed=y, shape=(1,))
            trace = pm.sample(draws)
        return trace
        
    def loglikelihood(self, *, y,  x, replicate_id=None, dependent_key=None, theta=None):
        """Loglikelihood of observation (dependent variable) given the independent variable

        Args:
            y (array): observed backscatter measurements (dependent variable)
            x (array or TensorVariable): assumed independent variable
            replicate_id(str): unique identifier for replicate (necessary for pymc3 likelihood)
            dependent_key(str): key of the dependent variable (necessary for pymc3 likelihood)
            theta: parameters of asymmetric_logistic (mu) and and polynomial functions (scale)

        Returns:
            L (float or TensorVariable): sum of log-likelihoods
        """
        if theta is None:
            if self.theta_fitted is None:
                raise Exception('No parameter vector was provided and the model is not fitted with data yet.')
            theta = self.theta_fitted
        mu, sigma, df = self.predict_dependent(x, theta=theta)
        if HAVE_THEANO and isinstance(x, theano.tensor.TensorVariable):
            L = pm.StudentT(
                f'{replicate_id}.{dependent_key}',
                mu=mu,
                sd=sigma,
                nu=df,
                observed=y
            )
            return L
        elif isinstance(x, (list, numpy.ndarray)):
            # using t-distributed error in the non-transformed space
            likelihoods = scipy.stats.t.pdf(x=y, loc=mu, scale=sigma, df=df)
            loglikelihoods = numpy.log(likelihoods)
            ll = numpy.sum(loglikelihoods)
            return ll
        else:
            raise Exception('Input x must either be a TensorVariable or an array-like object.')
