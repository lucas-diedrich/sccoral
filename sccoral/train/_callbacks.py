import logging
from typing import Literal

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BaseFinetuning, EarlyStopping
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class EarlyStoppingCheck(EarlyStopping):
    """Check if early stopping condition is met but do not interrupt training

    Modified `lightning.pytorch.callbacks.EarlyStopping` class that
    instead of sending early stopping signal to `Trainer` sets the
    parameter `pretraining_early_stopping_condition` in `TrainingPlan`
    to true

    Parameters
    ----------
    monitor
        Which loss to monitor.
    min_delta
        Definition of converging loss
    patience
        Number of consequetive epochs to wait until we send a stopping signal
    mode
        Look for maximum or minimum
    check_on_train
        Whether to check on training epoch end or validation epoch end. Defaults
        to training epoch
    **kwargs
        Other arguments passed to `lightning.pytorch.callbacks.EarlyStopping`
    """

    def __init__(
        self,
        monitor="",
        min_delta: float = 0.0,
        patience: int = 5,
        mode: Literal["max", "min"] = "min",
        check_on_train: bool = True,
        **kwargs,
    ):
        super().__init__(monitor, min_delta, patience, mode, **kwargs)

        self.check_on_train = check_on_train

        self.state = {}

    def _run_early_stopping_check(self, trainer: Trainer, pl_module: LightningModule):
        """Overwrite method that stops trainer"""
        pass

    def _check_stopping(self, trainer: Trainer, pl_module: LightningModule):
        logs = trainer.callback_metrics
        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # Write if model should stop
        pl_module.pretraining_early_stopping_condition = should_stop

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        if not self.check_on_train:
            return
        self._check_stopping(trainer, pl_module)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule):
        if self.check_on_train:
            return
        self._check_stopping(trainer, pl_module)


class PretrainingFreezeWeights(BaseFinetuning):
    """Freeze weights of parts of the module until pretraining ends

    Parameters
    ----------
    submodule
        For which part of the model we would like to freeze the weights during pretraining
    n_pretraining_epochs
        Maximal number of pretraining epochs
    early_stopping
        Whether to use `EarlyStoppingCheck` as additional stopping metric
    train_batch_norm:
        Whether to freeze batch norm layers (defaults to False)
    **kwargs
        Other keyword arguments passed to `lightning.pytorch.callbacks.BaseFinetuning`
    """

    def __init__(
        self,
        submodule: str = "z_encoder",
        n_pretraining_epochs: int = 500,
        early_stopping=True,
        lr: float = 1e-3,
        train_batch_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_pretraining_epochs = n_pretraining_epochs
        self.early_stopping = early_stopping
        self.submodule = submodule
        self.lr = lr
        self.freeze_batch_norm = train_batch_norm

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        module = getattr(pl_module.module, self.submodule)
        self.freeze(module, train_bn=self.train_batch_norm)

    def finetune_function(self, pl_module: LightningModule, epoch: int, optimizer: Optimizer) -> None:
        if pl_module.is_pretrained:  # skip if pretraining is finished
            return
        early_stopping_condition = False
        if self.early_stopping:
            early_stopping_condition = pl_module.pretraining_early_stopping_condition
        if (epoch == self.n_pretraining_epochs) or early_stopping_condition:
            self.unfreeze_and_add_param_group(
                modules=getattr(pl_module.module, self.submodule),
                optimizer=optimizer,
                lr=self.lr,
                initial_denom_lr=1,
                train_bn=True,
            )
            pl_module.is_pretrained = True

            logger.info(
                f"Unfreeze weights - Epoch {self.n_pretraining_epochs} - Early stopping: {early_stopping_condition}"
            )
