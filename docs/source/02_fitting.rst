How to estimate calibration model parameters
--------------------------------------------

After implementing a custom calibration model class, the next step is to estimate
the parameters of the calibration model by maximum likelihood.

In addition to measurement results obtained from measurement standards, this
optimization step requires an initial guess and bounds for the calibration model parameters.

The `calibr8.optimization <calibr8_optimization.html>`__ module implements convenience functions such as
`fit_scipy <calibr8_optimization.html#calibr8.optimization.fit_scipy>`__ that use the common
API of all calibration models.

.. code-block:: python
    :linenos:

    cmodel = MyCustomCalibrationModel()

    cal_independent = ...
    cal_dependent = ...

    theta_fit, history = calibr8.fit_scipy(
        cmodel,
        independent=cal_independent,
        dependent=cal_dependent,
        theta_guess=[
            # NOTE: this example assumes a linear model with
            #       constant, normally distributed noise
            0,    # intercept
            1,    # slope
            0.2,  # standard deviation
        ],
        theta_guess=[
            (-1, 1),       # intercept
            (0, 3),        # slope
            (0.001, 0.5),  # standard deviation (must be >= 0)
        ]
    )

In some cases finding a good initial guess and parameter bounds can be surprisingly hard.
That's why `fit_scipy <calibr8_optimization.html#calibr8.optimization.fit_scipy>`__ returns
not only the estimated parameter vector ``theta_fitted``, but also the ``history`` of parameter
vectors that were tested during optimization.
Looking into the path that the optimizer took through the parameter space sometimes helps to
diagnose bad convergence.

If you can't get the `scipy.optimize.minimize`-based optimization to work, you can take out
one of the "big guns":

* Via the `fit_pygmo <calibr8_optimization.html#calibr8.optimization.fit_pygmo>`__ convenience
  function, you can fit the calibration model with `PyGMO2 <https://esa.github.io/pygmo2>`_.
* You can set up your own optimization using the `loglikelihood <calibr8_core.html#calibr8.core.CalibrationModel.loglikelihood>`__ method.
* Or you could run MCMC sampling by creating a PyMC3 model of your calibration model.
  For this you can simply pass a list of PyMC3 random variables (priors) as the ``theta`` argument
  of the `loglikelihood <calibr8_core.html#calibr8.core.CalibrationModel.loglikelihood>`__ method.

Good luck!