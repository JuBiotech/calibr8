Applying calibration models: Useful features
--------------------------------------------

Saving & Loading
^^^^^^^^^^^^^^^^

After fitting, calibration models can be saved to JSON files with :code:`calibrationmodel.save("my_calibration.json")`.
This facilitates their re-use across different notebooks and analysis sessions.

To load a calibration model, you'll need to reference the ``class`` definition of the model:

.. code-block:: python

    cmodel = MyCustomCalibrationModel.load("my_calibration.json")

The loading routine checks the name of the calibration model class to prevent accidentally instantiating
models from the wrong parameter sets.

Numerical Inference
^^^^^^^^^^^^^^^^^^^
A fitted calibration model may be used to convert from one or many measurement observations into
a prediction about the independent variable.
While the `predict_independent <calibr8_core.html#calibr8.core.CalibrationModel.predict_independent>`__
gives a point estimate, we recommend to use
`infer_independent <calibr8_core.html#calibr8.core.CalibrationModel.infer_independent>`__.

`infer_independent <calibr8_core.html#calibr8.core.CalibrationModel.infer_independent>`__ takes one or
more measurement observations as the input and returns a
`NumericPosterior <calibr8_core.html#calibr8.core.NumericPosterior>`__
that describes the uncertainty about the independent variable.

