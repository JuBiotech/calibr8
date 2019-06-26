import abc
import numpy
import scipy.optimize
import sys
try:
    import pymc3 as pm
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

from .. core import ErrorModel, log_log_logistic, polynomial, inverse_log_log_logistic


class BiomassErrorModel(ErrorModel):
    def __init__(self, independent_key:str, dependent_key:str):
        """ A class for modeling the error of backscatter measurements of biomass.

        Args:
            independent: biomass (cell dry weight)
            dependent: backscatter measurements
        """
        super().__init__(independent_key, dependent_key)
        self.student_df=1
          
    def predict_dependent(self, y_hat, *, theta=None):
        """Predicts the parameters mu and sigma of a student-t-distribution which characterises the dependent variable (backscatter) given values of the independent variable (CDW).

        Args:
            y_hat (array): values of the independent variable
            theta: parameters describing the logistic function of mu and the polynomial function of sigma (default to self.theta_fitted)

        Returns:
            mu,sigma (array): values for mu and sigma charcterising the student-t-distributions describing the dependent variable (backscatter)
            df: degree of freedom of student-t-distribution (always set to 1)
        """
        if theta is None:
            theta = self.theta_fitted
        mu = log_log_logistic(y_hat, theta[:4])
        sigma = polynomial(y_hat,theta[4:])
        df=self.student_df
        return mu, sigma, df

    def predict_independent(self, y_obs):
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y_obs (array): observed backscatter measurements (dependent variable)

        Returns:
            biomass (array): most likely biomass values (independent variable)
        """
        y_hat = inverse_log_log_logistic(y_obs, self.theta_fitted)
        return y_hat
        
    def theano_logistic(self, y_hat, theta_log):
        """Log-log logistic model of the expected measurement outcomes, given a true independent variable.
        
        Arguments:
            y_hat (array): realizations of the independent variable
            theta_log (array): parameters of the log-log logistic model
            I_x: inflection point (ln(x))
            I_y: inflection point (ln(y))
            Lmax: maximum value in log sapce
            s: log-log slope
        """
        # IMPORTANT: Outside of this function, it is irrelevant that the correlation is modeled in log-log space.
        # Since the logistic function is assumed for logarithmic backscatter in dependency of logarithmic NTU,
        # the interpretation of (I_x, I_y, Lmax and s) is in terms of log-space.
        I_x, I_y, Lmax = theta_log[:3]
        s = theta_log[3:]

        # For the same reason, y_hat (the x-axis) must be transformed into log-space.
        y_hat = theano.tensor.log(y_hat)
        y_val = 2.0 * I_y - Lmax + (2.0 * (Lmax - I_y)) / (1.0 + theano.tensor.exp(-4.0*s * (y_hat - I_x)))

        # The logistic model predicts a log-transformed y_val, but outside of this
        # function, the non-log value is expected.
        return theano.tensor.exp(y_val)

    def infer_independent(self, y_obs, *, btm_lower=0, btm_upper=17, draws=1000):
        """Infer the posterior distribution of the independent variable given the observations of one point of the dependent variable.
        
        Args:
            y_obs (array): observed OD measurements
            btm_lower (int): lower limit for uniform distribution of cdw prior
            btm_upper (int): upper limit for uniform distribution of cdw prior
            student_df (int): df of student-t-likelihood (default: 1)
            draws (int): number of samples to draw (handed to pymc3.sample)
        
        Returns:
            trace: trace of the posterior distribution of inferred biomass concentration
        """ 
        theta = self.theta_fitted
        with pm.Model() as model:
            btm = pm.Uniform('BTM', lower=btm_lower, upper=btm_upper, shape=(1,))
            mu = self.theano_logistic(btm, theta[:4])
            sd = polynomial(btm, theta[4:])
            ll = pm.StudentT('likelihood', nu=self.student_df, mu=mu, sd=sd, observed=y_obs, shape=(1,))
            trace = pm.sample(draws)
        return trace
        
    def loglikelihood(self, *, y_obs,  y_hat, theta=None):
        """Loglikelihood of observation (dependent variable) given the independent variable

        Args:
            y_obs (array): observed backscatter measurements (dependent variable)
            y_hat (array): predicted values of independent variable
            theta: parameters describing the logistic function of mu and the polynomial function of sigma (to be fitted with data)

        """
        if theta is None:
            if self.theta_fitted is None:
                raise Exception('No parameter vector was provided and the model is not fitted with data yet.')
            theta = self.theta_fitted
        mu, sigma, df = self.predict_dependent(y_hat, theta=theta)
        # using t-distributed error in the non-transformed space
        likelihoods = scipy.stats.t.pdf(x=y_obs, loc=mu, scale=sigma, df=df)
        loglikelihoods = numpy.log(likelihoods)
        ll = numpy.sum(loglikelihoods)
        return ll
    
    def fit(self, dependent, independent, *, theta_guessed, bounds):
        """Function to fit the error model with observed data. The attribute theta_fitted is overwritten after the fit.

        Args:
            dependent (array): observations of dependent variable
            independent (array): desired values of the independent variable or measured values of the same
            theta_guessed: initial guess for parameters describing the logistic function of mu and the polynomial function of sigma
            bounds: bounds to fit the parameters

        Returns:
            fit: Fitting result of scipy.optimize.minimize
        """
        def sum_negative_loglikelihood(theta):
            return(-self.loglikelihood(y_obs=dependent, y_hat=independent, theta=theta))
        fit = scipy.optimize.minimize(sum_negative_loglikelihood, theta_guessed, bounds=bounds)
        self.theta_fitted = fit.x
        return fit
    
    
