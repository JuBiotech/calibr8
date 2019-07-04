import abc
import numpy
import scipy.optimize


class ErrorModel(object):
    """A parent class providing the general structure of an error model."""
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, independent_key:str, dependent_key:str):
        """Creates an ErrorModel object.

        Args:
            independent_key: key of predicted Timeseries (independent variable of the error model)
            dependent_key: key of observed Timeseries (dependent variable of the error model)
        """
        self.independent_key = independent_key
        self.dependent_key = dependent_key
        self.theta_fitted = None
        super().__init__()
    
    @abc.abstractmethod
    def predict_dependent(self, x, *, theta=None):
        """Predicts the parameters of a probability distribution which characterises 
           the dependent variable given values of the independent variable.

        Args:
            x (array): independent variable
            theta: parameters of functions describing the mode and standard deviation of the PDF

        Returns:
           parameters (array): parameters characterising a distribution for the dependent variable
        """
        raise NotImplementedError('The predict_dependent function should be implemented by the inheriting class.')
    
    @abc.abstractmethod
    def predict_independent(self, y):
        """Predict the most likely value of the independent variable using the calibrated error model in inverse direction.

        Args:
            y (array): realizations of the dependent variable

        Returns:
            x (array): predicted mode of the independent variable
        """
        raise NotImplementedError('The predict_independent function should be implemented by the inheriting class.')

    @abc.abstractmethod
    def infer_independent(self, y):
        """Infer the posterior distribution of the independent variable given the observations of one point of the dependent variable.

        Args:
            y (array): realizations of the dependent variable
        
        Returns:
            trace: PyMC3 trace of the posterior distribution of the inferred independent variable
        """  
        raise NotImplementedError('The infer_independent function should be implemented by the inheriting class.')
            
    def loglikelihood(self, *, y,  x, theta=None):
        """Loglikelihood of dependent variable realizations given assumed independent variables.

        Args:
            y (array): realizations of the dependent variable
            x (array): assumptions of the independent variable
            theta: model parameters (defaults to self.theta_fitted)
        
        Return:
            L (float): sum of loglikelihoods
        """
        raise NotImplementedError('The loglikelihood function should be implemented by the inheriting class.')
        
    def fit(self, *, independent, dependent, theta_guessed, bounds=None):
        """Function to fit the error model with observed data. The attribute theta_fitted is overwritten after the fit.

        Args:
            independent (array): desired values of the independent variable or measured values of the same
            dependent (array): observations of dependent variable
            theta_guessed: initial guess for parameters describing the mode and standard deviation of a PDF of the dependent variable
            bounds: bounds to fit the parameters

        Returns:
            fit: Fitting result of scipy.optimize.minimize
        """
        def sum_negative_loglikelihood(theta):
            return(-self.loglikelihood(x=independent, y=dependent, theta=theta))
        fit = scipy.optimize.minimize(sum_negative_loglikelihood, theta_guessed, bounds=bounds)
        self.theta_fitted = fit.x
        return fit


def logistic(x, theta):
        """4-parameter logistic model.
        
        Args:
            x (array): independent variable
            theta (array): parameters of the logistic model
                I_x: x-value at inflection point
                I_y: y-value at inflection point
                Lmax: maximum value
                s: slope at the inflection point
        
        Returns:
            y (array): dependent variable
        """
        I_x, I_y, Lmax, s = theta[:4]
        x = numpy.array(x)      
        y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))  
        return y


def inverse_logistic(y, theta):
    """Inverse 4-parameter logistic model.
    
    Args:
        y (array): dependent variables
        theta (array): parameters of the logistic model
            I_x: x-value at inflection point
            I_y: y-value at inflection point
            Lmax: maximum value
            s: slope at the inflection point
    
    Returns:
        x (array): independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.array(y)
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
    return x


def asymmetric_logistic(x, theta):
    """5-parameter asymmetric logistic model.
    
    Args:
        x (array): independent variable
        theta (array): parameters of the logistic model
            L_L: lower asymptote
            L_U: upper asymptote
            I_x: x-value at inflection point (v=1)
            k: growth rate
            v: symmetry parameter
    
    Returns:
        y (array): dependent variable
    """
    L_L, L_U, I_x, k, v = theta[:5]
    y = L_L + (L_U-L_L)/(numpy.power((1+numpy.exp(-k*(x-I_x))),1/v))
    return y


def inverse_asymmetric_logistic(y, theta):
    """Inverse 5-parameter asymmetric logistic model.
        
    Args:
        y (array): dependent variable
        theta (array): parameters of the logistic model
            L_L: lower asymptote
            L_U: upper asymptote
            I_x: x-value at inflection point (v=1)
            k: growth rate
            v: symmetry parameter
    
    Returns:
        x (array): independent variable
    """
    L_L, L_U, I_x, k, v = theta[:5]
    y = numpy.array(y)
    x = I_x-k*numpy.log((numpy.power((L_U-L_L)/(y-L_L), v))-1)    
    return x


def log_log_logistic(x, theta):
    """4-parameter log-log logistic model.
    
    Args:
        x (array): independent variable
        theta (array): parameters of the log-log logistic model
            I_x: inflection point (ln(x))
            I_y: inflection point (ln(y))
            Lmax: logarithmic maximum value
            s: slope at the inflection point
    
    Returns:
        y (array): dependent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    x = numpy.log(x)    
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))   
    return numpy.exp(y)


def inverse_log_log_logistic(y, theta):
    """4-parameter log-log logistic model.
        
    Args:
        y (array): dependent variable
        theta (array): parameters of the logistic model
            I_x: x-value at inflection point (ln(x))
            I_y: y-value at inflection point (ln(y))
            Lmax: maximum value in log space
            s: slope at the inflection point
    
    Returns:
        x (array): independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.log(y)
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
    return numpy.exp(x)


def xlog_logistic(x, theta):
    """4-parameter x-log logistic model.
    
    Args:
        x (array): independent variable
        theta (array): parameters of the log-log logistic model
            I_x: inflection point (ln(x))
            I_y: inflection point (y)
            Lmax: maximum value
            s: slope at the inflection point
    
    Returns:
        y (array): dependent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    x = numpy.log(x)    
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))     
    return y


def inverse_xlog_logistic(y, theta):
    """Inverse 4-parameter x-log logistic model.
        
    Args:
        y (array): dependent variable
        theta (array): parameters of the logistic model
            I_x: x-value at inflection point (ln(x))
            I_y: y-value at inflection point
            Lmax: maximum value
            s: slope at the inflection point
    
    Returns:
        x (array): independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.array(y)
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
    return numpy.exp(x)


def ylog_logistic(x, theta):
    """4-parameter y-log logistic model.
    
    Args:
        x (array): independent variable
        theta (array): parameters of the log-log logistic model
            I_x: inflection point (x)
            I_y: inflection point (ln(y))
            Lmax: maximum value in log sapce
            s: slope at the inflection point
    
    Returns:
        y (array): dependent variables
    """
    I_x, I_y, Lmax, s = theta[:4]
    x = numpy.array(x) 
    y = 2 * I_y - Lmax + (2 * (Lmax - I_y)) / (1 + numpy.exp(-2*s/(Lmax - I_y) * (x - I_x)))
    return numpy.exp(y)


def inverse_ylog_logistic(y, theta):
    """Inverse 4-parameter y-log logistic model.
        
    Args:
        y (array): dependent variable
        theta (array): parameters of the logistic model
            I_x: x-value at inflection point
            I_y: y-value at inflection point (ln(y))
            Lmax: maximum value in log space
            s: slope at the inflection point
    
    Returns:
        x (array): independent variable
    """
    I_x, I_y, Lmax, s = theta[:4]
    y = numpy.log(y)
    x = I_x-((Lmax-I_y)/(2*s))*numpy.log((2*(Lmax-I_y)/(y+Lmax-2*I_y))-1)
    return x


def polynomial(x, theta):
    """Variable-degree polynomical model.

    Args:
        x (array): independent variable
        theta (array): polynomial coefficients (lowest degree first)

    Returns:
        y (array) dependent variable
    """
    # Numpy's polynomial function wants to get the highest degree first
    return numpy.polyval(theta[::-1], x)
