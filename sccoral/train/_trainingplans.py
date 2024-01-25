from collections.abc import Iterable
from typing import Callable

import optax
import torch
from scvi.train import TrainingPlan

axOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


class ScCoralTrainingPlan(TrainingPlan):
    """Implement custom pretraining procedure for sccoral

    Inherits from `scvi.train.TrainingPlan` and adds
    custom properties `is_pretrained` and `pretraining
    """

    def __init__(self, module, **kwargs):
        super().__init__(
            module,
            # optimizer,
            # optimizer_creator,
            # lr,
            # weight_decay,
            # eps,
            # n_steps_kl_warmup,
            # n_epochs_kl_warmup,
            # reduce_lr_on_plateau,
            # lr_factor,
            # lr_patience,
            # lr_scheduler_metric,
            # lr_min,
            # max_kl_weight,
            # min_kl_weight,
            # **loss_kwargs,
        )

        self._is_pretrained = False
        self._pretraining_early_stopping_condition = False

    @property
    def is_pretrained(self):
        """Indicate if model is pretrained

        This is used by the `PretrainingFreezeWeights` callback
        to decide whether to freeze the weights of the decoder
        """
        return self._is_pretrained

    @is_pretrained.setter
    def is_pretrained(self, value: bool):
        self._is_pretrained = value

    @property
    def pretraining_early_stopping_condition(self):
        """Indicate if pretraining early stopping condition is met

        Used by callback to decide whether to unfreeze encoder weights.
        """
        return self._pretraining_early_stopping_condition

    @pretraining_early_stopping_condition.setter
    def pretraining_early_stopping_condition(self, value: bool):
        self._pretraining_early_stopping_condition = value
