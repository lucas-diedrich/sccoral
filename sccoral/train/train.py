from collections.abc import Iterable
from typing import Callable

import optax
import torch
from scvi.train import TrainingPlan

axOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]


class ScCoralTrainingPlan(TrainingPlan):
    """Implement custom pretraining procedure for sccoral"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._is_pretrained = False
        self._pretraining_early_stopping_condition = False

    @property
    def is_pretrained(self):
        """If model is pretrained"""
        return self._is_pretrained

    @is_pretrained.setter
    def is_pretrained(self, value: bool):
        self._is_pretrained = value

    @property
    def pretraining_early_stopping_condition(self):
        """If pretraining should stop"""
        return self._pretraining_early_stopping_condition

    @pretraining_early_stopping_condition.setter
    def pretraining_early_stopping_condition(self, value: bool):
        self._pretraining_early_stopping_condition = value
