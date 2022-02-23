
from .contrib.noise import (
    NormalNoise,
    LaplaceNoise,
    LogNormalNoise,
    StudentTNoise,
    PoissonNoise,
)
from .contrib.normal import (
    BaseAsymmetricLogisticN,
    BaseLogIndependentAsymmetricLogisticN,
    BasePolynomialModelN,
    BaseExponentialModelN,
)
from .contrib.studentt import (
    BaseAsymmetricLogisticT,
    BaseLogIndependentAsymmetricLogisticT,
    BaseModelT,
    BasePolynomialModelT,
    BaseExponentialModelT,
)
from .core import (
    CalibrationModel,
    DistributionMixin,
    ContinuousMultivariateModel,
    ContinuousUnivariateModel,
    InferenceResult,
    ContinuousMultivariateInference,
    ContinuousUnivariateInference,
    __version__,
    asymmetric_logistic,
    exponential,
    inverse_asymmetric_logistic,
    inverse_exponential,
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
    HAS_TENSORS,
    HAS_PYMC,
    istensor,
    plot_model,
    plot_t_band,
    plot_norm_band,
    plot_continuous_band,
)


class ErrorModel(CalibrationModel):
    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "The `ErrorModel` class was renamed to `CalibrationModel`."
            " It will be removed in a future release.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class NumericPosterior(ContinuousUnivariateInference):
    """Deprecated alias for ContinuousUnivariateInference"""
    def __init__(self, *args, **kwargs) -> None:
        import warnings

        warnings.warn(
            "The `NumericPosterior` class was renamed to `ContinuousUnivariateInference`."
            " It will be removed in a future release.",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
