from . core import ErrorModel, asymmetric_logistic, inverse_asymmetric_logistic, inverse_logistic, inverse_log_log_logistic, inverse_xlog_logistic, inverse_ylog_logistic, logistic, log_log_logistic, polynomial, xlog_logistic, ylog_logistic
from . contrib import BiomassErrorModel, BaseGlucoseErrorModel, LinearGlucoseErrorModel, LogisticGlucoseErrorModel
from . utils import to_did

__version__ = '3.0.0'
