Applying error models: Useful features
--------------------------------------

Saving & Loading
^^^^^^^^^^^^^^^^

After fitting, error models can be saved to JSON files with :code:`errormodel.save("my_errormodel.json")`.
This facilitates their re-use across different notebooks and analysis sessions.

To load an error model, you'll need to reference the ``class`` definition of the model:

.. code-block:: python

    errormodel = MyCustomErrorModel.load("my_errormodel.json")

The loading routine checks the name of the error model class to prevent accidentally instantiating
models from the wrong parameter sets.

Numerical Inference
^^^^^^^^^^^^^^^^^^^
A fitted error model may be used to convert from one or many measurement observations into
a prediction about the independent variable.
While the `predict_independent <calibr8_core.html#calibr8.core.ErrorModel.predict_independent>`__
gives a point estimate, we recommend to use
`infer_independent <calibr8_core.html#calibr8.core.ErrorModel.infer_independent>`__.

`infer_independent <calibr8_core.html#calibr8.core.ErrorModel.infer_independent>`__ takes one or
more measurement observations as the input and returns a
`NumericPosterior <calibr8_core.html#calibr8.core.NumericPosterior>`__
that describes the uncertainty about the independent variable.

