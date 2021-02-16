from .contrib.base import (
    BaseAsymmetricLogisticT,
    BaseLogIndependentAsymmetricLogisticT,
    BaseModelT,
    BasePolynomialModelT,
)
from .core import (
    CalibrationModel,
    NumericPosterior,
    __version__,
    asymmetric_logistic,
    inverse_asymmetric_logistic,
    inverse_log_log_logistic,
    inverse_logistic,
    inverse_xlog_asymmetric_logistic,
    inverse_xlog_logistic,
    inverse_ylog_logistic,
    log_log_logistic,
    logistic,
    polynomial,
    xlog_asymmetric_logistic,
    xlog_logistic,
    ylog_logistic,
)
from .optimization import fit_pygmo, fit_scipy
from .utils import (
    BuildMismatchException,
    CompatibilityException,
    MajorMismatchException,
    MinorMismatchException,
    PatchMismatchException,
    istensor,
    plot_model,
    plot_norm_band,
    plot_t_band,
)


class ErrorModel(CalibrationModel):
    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "The `ErrorModel` class was renamed to `CalibrationModel`. It will be removed in a future release.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
