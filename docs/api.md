# API

## Model

Marginal-likelihood estimation is not supported. The inherited
`get_marginal_ll()` method raises `NotImplementedError`. Use `get_elbo()` or
`get_reconstruction_error()` for their respective evaluation metrics; neither
is a marginal-likelihood estimate.

```{eval-rst}
.. module:: sccoral.model
.. currentmodule:: sccoral

.. autosummary::
    :toctree: generated

    model.SCCORAL
```

## Module

```{eval-rst}
.. module:: sccoral.module
.. currentmodule:: sccoral

.. autosummary::
    :toctree: generated

    module.MODULE

```

## Training plan

```{eval-rst}
.. module:: sccoral.train
.. currentmodule:: sccoral

.. autosummary::
   :toctree: generated

   train.ScCoralTrainingPlan
```

## Datasets

```{eval-rst}
.. module:: sccoral.data
.. currentmodule:: sccoral

.. autosummary::
    :toctree: generated

    data.splatter_simulation

```
