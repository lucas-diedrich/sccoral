import logging
from typing import Literal

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BaseFinetuning, Callback, EarlyStopping
from torch.optim.optimizer import Optimizer

logger = logging.getLogger(__name__)


class JointTrainingEarlyStopping(EarlyStopping):
    """Evaluate overall early stopping only after the count encoder is unfrozen."""

    def _run_early_stopping_check(self, trainer: Trainer):
        pl_module = trainer.lightning_module
        if not pl_module.is_pretrained:
            return
        already_stopping = trainer.should_stop
        super()._run_early_stopping_check(trainer)
        if trainer.should_stop and not already_stopping:
            pl_module.training_status["termination_reason"] = "early_stopping"


class TrainingStatus(Callback):
    """Record fully completed training epochs in the model's saved status dictionary."""

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self._batches_completed = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batches_completed += 1

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        # A max_steps limit can end an epoch after only a subset of its batches.
        if self._batches_completed < trainer.num_training_batches:
            return
        status = pl_module.training_status
        status["epochs_completed"] += 1
        key = "joint_epochs_completed" if pl_module.is_pretrained else "pretraining_epochs_completed"
        status[key] += 1

    def on_exception(self, trainer, pl_module, exception):
        pl_module.training_status["termination_reason"] = (
            "interrupted" if isinstance(exception, KeyboardInterrupt) else "failed"
        )


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
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, mode=mode, **kwargs)

        self.check_on_train = check_on_train

    def _run_early_stopping_check(self, trainer: Trainer, pl_module: LightningModule):
        """Overwrite method that stops trainer"""
        pass

    def _check_stopping(self, trainer: Trainer, pl_module: LightningModule):
        if pl_module.is_pretrained or trainer.sanity_checking:
            return
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
        self.train_batch_norm = train_batch_norm

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
            pl_module.training_status["pretraining_completed"] = True
            pl_module.training_status["unfreeze_epoch"] = epoch
            logger.info("Count encoder unfrozen at epoch %s; joint training begins.", epoch)
