# Inheritance and composition of `calibr8` models
Models in `calibr8` are implemented as classes, inheriting from the `CalibrationModel` interface* either directly or via configureable `calibr8.BaseXYZ` model classes.
Generalization across different distributions in the noise model is provided with mixins**.

The reasoning for this design choice is explained in the following sections.

<sup>*[Further reading on informal interfaces in Python](https://realpython.com/python-interface/).</sup></br>
<sup>**[Further reading on mixins](https://realpython.com/inheritance-composition-python/#mixing-features-with-mixin-classes).</sup></br>

## Inheritance of model classes
The choice of the inheritance architecture is based on an analysis of two important aspects of a calibration model:
1. What kind of independent variable is the model working with? (continuous/discrete)
2. How many dimensions does the independent variable have? (univariate/multivariate)

Note that the `(log)likelihood` of a model is agnostic to the dependent variable being continuous/discrete, and therefore we do not have to consider it here.

The following tables summarizes the long-form matrix of the various combinations and corresponding implementations.
Note that since MCMC can be used for any of them, the "inference strategy" column denotes the computationally simplest generic strategy.

| ndim | independent variable | integration/inference strategy | `InferenceResult` (properties) | model subclass |
|----------------|----------------------|--------------------|--------------------------------|----------------|
| 1 | continuous | `trapz` | `ContinuousUnivariateInference` (ETI, HDI, median) | `ContinuousUnivariateModel` |
| >1 | continuous | MCMC | `ContinuousMultivariateInference` (idata, ETI, HDI) | `ContinuousMultivariateModel` |
| 1 | discrete | summation & division of likelihoods | *, `DiscreteUnivariateInference` (probability vector, mode) | *, `DiscreteUnivariateModel` |
| >1 | discrete | summation & division of likelihoods | *, `DiscreteAnyvariateInference` (probability ndarray) | *, `DiscreteMultivariateModel` |
| >1 | **mix of continuous & discrete 🤯 | MCMC | *, `MixedMultivariateInference` (idata) | *, `MixedMultivariateModel` |

<sup>*Not implemented.</sup></br>
<sup>**Needs custom SciPy and PyMC distribution.</sup>

## Composition of distribution mixins
The `CalibrationModel`, and specifically it's `loglikelihood` implementation can generalize across distributions (noise models) for the dependent variable based on a mapping of predicted distribution parameters (`predict_dependent`) to the corresponding SciPy, and optionally a PyMC distribution.

This mapping is managed via class attributes and staticmethods and provided by subclassing a `DistributionMixin`.
For example, a user-implemented model may subclass the `calibr8.LogNormalNoise` and implement a `predict_dependent` method to return a tuple of the two distribution parameters.

```python
class LogNormalNoise(DistributionMixin):
    """Log-Normal noise, predicted in logarithmic mean and standard deviation.
    ⚠ This corresponds to the NumPy/Aesara/PyMC parametrization!
    """
    scipy_dist = scipy.stats.lognorm
    pymc_dist = pm.Lognormal if HAS_PYMC else None

    @staticmethod
    def to_scipy(*params):
        # SciPy wants linear scale mean and log scale standard deviation!
        return dict(scale=numpy.exp(params[0]), s=params[1])

    @staticmethod
    def to_pymc(*params):
        return dict(mu=params[0], sigma=params[1])
```

## Localization of implementations
Most of the source code—and certainly the most intricate code—of a `calibr8` model is located in the classes provided by the library.
For most applications the user may directly use a common model class from `calibr8`, or implement their own class with nothing more than the `__init__` method.

If the desired model structure is not already provided by `calibr8.BaseXYZ` types, a custom `predict_dependent` can be implemented by subclassing from a model subclass.
Examples are the use of uncommon noise models, or multivariate calibration models.

| name ╲ class | `CalibrationModel` | model subclass | `CustomModel` | `BaseXYZ` | `UserModel` |
|--------------|--------------------|----------------|---------------|-----------|-------------|
| implemented by | `calibr8` | `calibr8` | user | `calibr8` | user
| inherits from | `object` | `CalibrationModel` | model subclass | model subclass | `BaseXYZ` |
| adds mixin | `DistributionMixin` |  | user-specified | `NormalNoise` or `StudentTNoise` | |
| generalizes for | all models | ndim and type of</br>independent variable | | common model structures</br> (e.g. univariate polynomial)
| `__init__` | ✔ | ✔ | ✔ | (✔) | ✔ |
| `theta_*` | ✔ | ← | ← | ← | ← |
| `save` | ✔ | ← | ← | ← | ← |
| `load` | ✔ | ← | ← | ← | ← |
| `objective` | ✔ | ← | ← | ← | ← |
| `loglikelihood` | ✔ | ← | ← | ← | ← |
| `likelihood` | ✔ | ← | ← | ← | ← |
| `infer_independent` | ❌ | ✔ | ← | ← | ← |
| `predict_dependent` | ❌ | ❌ | ✔ | ✔ | ← |
| `predict_independent` | ❌ | ❌ | (✔) | (✔) | ← |


Symbols
* ❌ `NotImplementedError` at this level
* ✔ implemented at this class/mixin level
* (✔) necessity/feasability depends on the type of model
* ← inherits the implementation
