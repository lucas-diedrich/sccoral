from collections.abc import Iterable
from typing import Callable, Literal, Optional

import torch
from scvi.train import TrainingPlan

# Changes after scvi 1.0.4
try:
    from scvi.autotune import Tunable
except ImportError:
    from scvi._types import Tunable


TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


class ScCoralTrainingPlan(TrainingPlan):
    """Implement custom pretraining procedure for sccoral

    Inherits from `scvi.train.TrainingPlan` and adds
    custom properties `is_pretrained` and `pretraining_early_stopping_condition`
    """

    def __init__(
        self,
        module,
        optimizer: Tunable[Literal["Adam", "AdamW", "Custom"]] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Tunable[float] = 1e-3,
        weight_decay: Tunable[float] = 1e-6,
        eps: float = 0.01,
        n_steps_kl_warmup: Tunable[int] = None,
        n_epochs_kl_warmup: Tunable[int] = 400,
        reduce_lr_on_plateau: bool = False,
        lr_factor: Tunable[float] = 0.6,
        lr_patience: Tunable[int] = 30,
        lr_threshold: Tunable[float] = 0.0,
        lr_scheduler_metric: Tunable[
            Literal["elbo_validation", "reconstruction_loss_validation", "kl_local_validation"]
        ] = "elbo_validation",
        lr_min: float = 0,
        max_kl_weight: Tunable[float] = 1.0,
        min_kl_weight: Tunable[float] = 0.0,
        **loss_kwargs,
    ):
        super().__init__(
            module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            max_kl_weight=max_kl_weight,
            min_kl_weight=min_kl_weight,
            **loss_kwargs,
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

    @is_pretrained.getter
    def is_pretrained(self):
        return self._is_pretrained

    @property
    def pretraining_early_stopping_condition(self):
        """Indicate if pretraining early stopping condition is met

        Used by callback to decide whether to unfreeze encoder weights.
        """
        return self._pretraining_early_stopping_condition

    @pretraining_early_stopping_condition.setter
    def pretraining_early_stopping_condition(self, value: bool):
        self._pretraining_early_stopping_condition = value
