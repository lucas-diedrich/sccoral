# API

## Model

Marginal-likelihood estimation is not supported. The inherited
`get_marginal_ll()` method raises `NotImplementedError`. Use `get_elbo()` or
`get_reconstruction_error()` for their respective evaluation metrics; neither
is a marginal-likelihood estimate.

Each `train()` call records its progress in `model.training_status_`, which is
retained by `save()` and `load()`. It includes whether pretraining was enabled
and completed, the zero-based `unfreeze_epoch`, completed pretraining and joint
epoch counts, and a `termination_reason`. With pretraining disabled, the
pretraining requirement is considered complete and `unfreeze_epoch` is `None`.
Partial epochs stopped by a step limit are not counted as completed epochs.

Automatic overall early stopping begins only after the count encoder is unfrozen.
Explicit user callbacks can still request a stop. A warning is emitted if no
joint-training epoch completes; inspect the status before using such a model.
The record describes the latest training attempt, rather than cumulative history
or a guarantee of convergence. Older saved models may not contain this attribute.

```python
model.train(training_status_path="results/training-status.json")
print(model.training_status_)
```

Termination reasons include `max_epochs`, `max_steps`, `early_stopping`,
`stop_requested`, `interrupted`, and `failed` (`completed` is a fallback for other
normal exits). The optional JSON file is also written when fitting raises an
exception. Hard process termination cannot guarantee a final record. When
pretraining is enabled, `max_epochs` includes both training phases, so allow
enough epochs for the joint phase.

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
