from .contrib.noise import (
    LaplaceNoise,
    LogNormalNoise,
    NormalNoise,
    PoissonNoise,
    StudentTNoise,
)
from .contrib.normal import (
    BaseAsymmetricLogisticN,
    BaseExponentialModelN,
    BaseLogIndependentAsymmetricLogisticN,
    BasePolynomialModelN,
)
from .contrib.studentt import (
    BaseAsymmetricLogisticT,
    BaseExponentialModelT,
    BaseLogIndependentAsymmetricLogisticT,
    BaseModelT,
    BasePolynomialModelT,
)
from .core import (
    CalibrationModel,
    ContinuousMultivariateInference,
    ContinuousMultivariateModel,
    ContinuousUnivariateInference,
    ContinuousUnivariateModel,
    DistributionMixin,
    InferenceResult,
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
from .optimization import fit_pygmo, fit_scipy, fit_scipy_global
from .utils import (
    HAS_PYMC,
    HAS_TENSORS,
    BuildMismatchException,
    CompatibilityException,
    MajorMismatchException,
    MinorMismatchException,
    PatchMismatchException,
    istensor,
    plot_continuous_band,
    plot_model,
    plot_norm_band,
    plot_t_band,
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
