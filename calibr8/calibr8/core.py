import abc
import numpy
import scipy.optimize


class ErrorModel(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, independent_key:str, dependent_key:str):
        """ A parent class providing the general structure of an error model.

        Args:
            independent_key: key of predicted Timeseries (independent variable of the error model)
            dependent_key: key of observed Timeseries (dependent variable of the error model)
        """
        self.independent_key = independent_key
        self.dependent_key = dependent_key
        self.theta_fitted = None
        super().__init__()
    
    @abc.abstractmethod
    def predict_dependent(self, y_hat, *, theta=None):
        """ Predicts the parameters of a probability distribution which characterises 
            the dependent variable given values of the independent variable.

        Args:
            y_hat (array): values of the independent variable
            theta: parameters of functions describing the mode and standard deviation of the PDF

        Returns:
            mu,sigma (array): values for mu and sigma characterising a PDF describing the dependent variable
        """
        raise NotImplementedError('The predict_dependent function should be implemented by the inheriting class.')
    
    @abc.abstractmethod
    def predict_independent(self, y_hat):
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y_obs (array): observed measurements

        Returns:
            mu (array): predicted mode of the independent variable
        """
        raise NotImplementedError('The predict_independent function should be implemented by the inheriting class.')

    @abc.abstractmethod
    def infer_independent(self, y_obs):
        """Infer the posterior distribution of the independent variable given the observations of one point of the dependent variable.
        Args:
            y_obs (array): observed measurements
        
        Returns:
            trace: trace of the posterior distribution of the inferred independent variable
        """  
        raise NotImplementedError('The infer_independent function should be implemented by the inheriting class.')
            
    def loglikelihood(self, *, y_obs,  y_hat, theta=None):
        """Loglikelihood of observations (dependent variable) given the independent variable

        Args:
            y_obs (array): observed backscatter measurements (dependent variable)
            y_hat (array): predicted values of independent variable
            theta: parameters describing the logistic function of mu and the polynomial function of sigma (to be fitted with data, otherwise theta=self.theta_fitted)
        
        Return:
            Sum of loglikelihoods

        """
        raise NotImplementedError('The loglikelihood function should be implemented by the inheriting class.')
        
    def fit(self, dependent, independent, *, theta_guessed, bounds=None):
        """Function to fit the error model with observed data. The attribute theta_fitted is overwritten after the fit.

        Args:
            dependent (array): observations of dependent variable
            independent (array): desired values of the independent variable or measured values of the same
            theta_guessed: initial guess for parameters describing the mode and standard deviation of a PDF of the dependent variable
            bounds: bounds to fit the parameters

        Returns:
            fit: Fitting result of scipy.optimize.minimize
        """
        def sum_negative_loglikelihood(theta):
            return(-self.loglikelihood(y_obs=dependent, y_hat=independent, theta=theta))
        fit = scipy.optimize.minimize(sum_negative_loglikelihood, theta_guessed, bounds=bounds)
        self.theta_fitted = fit.x
        return fit


def logistic(y_hat, theta):
        """Logistic model of the expected measurement outcome, given a true independent variable.
        
        Args:
            y_hat (array): realizations of the independent variable
            theta (array): parameters of the logistic model
            I_x: x-value at inflection point
            I_y: y-value at inflection point
            Lmax: maximum value
            s: slope at the inflection point
        
        Returns:
            y_val: expected measurement outcome
        """
        I_x, I_y, Lmax, s = theta[:4]
        y_hat = numpy.array(y_hat)      
        y_val = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (y_hat - I_x)))
             
        return y_val


def inverse_logistic(y_obs, theta):
    """Inverse logistic model returning the predicted independent variable given the measurement.
    
    Args:
        y_obs (array): measured values
        theta (array): parameters of the logistic model
        I_x: x-value at inflection point
        I_y: y-value at inflection point
        Lmax: maximum value
        s: slope at the inflection point
    
    Returns:
        y_hat: predicted value of the independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y_val = numpy.array(y_obs)
    y_hat = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y_val+Lmax-2*I_y))-1)

    return y_hat


def log_log_logistic(y_hat, theta_log):
    """Log-log logistic model of the expected measurement outcome, given a true independent variable.
    
    Args:
        y_hat (array): realizations of the independent variable
        theta_log (array): parameters of the log-log logistic model
        I_x: inflection point (ln(x))
        I_y: inflection point (ln(y))
        Lmax: logarithmic maximum value
        s: slope at the inflection point
    
    Returns:
        y_obs: expected measurement outcome
    """
    I_x, I_y, Lmax, s = theta_log[:4]
    y_hat = numpy.log(y_hat)    
    y_val = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (y_hat - I_x)))
            
    return numpy.exp(y_val)


def inverse_log_log_logistic(y_obs, theta_log):
    """Inverse logistic model returning the predicted independent variable given the measurement.
        
    Args:
        y_obs (array): measured values
        theta (array): parameters of the logistic model
        I_x: x-value at inflection point (ln(x))
        I_y: y-value at inflection point (ln(y))
        Lmax: maximum value in log space
        s: slope at the inflection point
    
    Returns:
        y_hat: predicted value of the independent variable
    """
    I_x, I_y, Lmax, s = theta_log[:4]
    y_val = numpy.log(y_obs)
    y_hat = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y_val+Lmax-2*I_y))-1)
    
    return numpy.exp(y_hat)


def xlog_logistic(y_hat, theta_log):
    """Log-log logistic model of the expected measurement outcomes, given a true independent variable.
    
    Args:
        y_hat (array): realizations of the independent variable
        theta_log (array): parameters of the log-log logistic model
        I_x: inflection point (ln(x))
        I_y: inflection point (y)
        Lmax: maximum value
        s: slope at the inflection point
    
    Returns:
        y_obs: expected measurement outcome
    """
    I_x, I_y, Lmax, s = theta_log[:4]
    y_hat = numpy.log(y_hat)    
    y_val = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (y_hat - I_x)))
            
    return y_val


def inverse_xlog_logistic(y_obs, theta_log):
    """Inverse logistic model returning the predicted independent variable given the measurement.
        
    Args:
        y_obs (array): measured values
        theta (array): parameters of the logistic model
        I_x: x-value at inflection point (ln(x))
        I_y: y-value at inflection point
        Lmax: maximum value
        s: slope at the inflection point
    
    Returns:
        y_hat: predicted value of the independent variable
    """
    I_x, I_y, Lmax, s = theta_log[:4]
    y_val = y_obs
    y_hat = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y_val+Lmax-2*I_y))-1)
    
    return numpy.exp(y_hat)


def ylog_logistic(y_hat, theta_log):
        """Log-log logistic model of the expected measurement outcomes, given a true independent variable.
        
        Args:
            y_hat (array): realizations of the independent variable
            theta_log (array): parameters of the log-log logistic model
            I_x: inflection point (x)
            I_y: inflection point (ln(y))
            Lmax: maximum value in log sapce
            s: slope at the inflection point
        
        Returns:
            y_obs: expected measurement outcome
        """
        I_x, I_y, Lmax, s = theta_log[:4]
        y_hat = numpy.array(y_hat) 
        y_val = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (y_hat - I_x)))
             
        return numpy.exp(y_val)


def inverse_ylog_logistic(y_obs, theta_log):
    """Inverse logistic model returning the predicted independent variable given the measurement.
        
    Args:
        y_obs (array): measured values
        theta (array): parameters of the logistic model
        I_x: x-value at inflection point
        I_y: y-value at inflection point (ln(y))
        Lmax: maximum value in log space
        s: slope at the inflection point
    
    Returns:
        y_hat: predicted value of the independent variable
    """
    I_x, I_y, Lmax, s = theta_log[:4]
    y_val = numpy.log(y_obs)
    y_hat = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y_val+Lmax-2*I_y))-1)
    
    return y_hat


def polynomial(y_hat, theta_pol):
    # Numpy's polynomial function wants to get the highest degree first
    return numpy.polyval(theta_pol[::-1], y_hat)
    