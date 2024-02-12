# Imports
import logging
from collections.abc import Iterable
from typing import Any, Literal

import anndata as ad
import pandas as pd
import torch
from scvi import REGISTRY_KEYS

# Changes after scvi 1.0.4
try:
    from scvi.autotune import Tunable, TunableMixin
except ImportError:
    from scvi._types import Tunable, TunableMixin

from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField, NumericalJointObsField
from scvi.dataloaders import DataSplitter
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass
from scvi.train import TrainRunner
from torch import inference_mode

from sccoral.module import MODULE
from sccoral.train import ScCoralTrainingPlan
from sccoral.train import _callbacks as tcb

logger = logging.getLogger(__name__)


class SCCORAL(BaseModelClass, TunableMixin):
    """Single-cell COvariate-informed Regularized variational Autoencoder with Linear Decoder

    Parameters
    ----------
    adata
        Registered AnnData object
    n_latent
        Number of latent dimensions, approximated dimensionality of dataset
    alpha_l1
        Regularization strength in decoder
    n_hidden
        Number of hidden layers in encoder
    n_layers
        Number of layers in encoder neural network (see LSCVI)
    dropout_rate
        Dropout rate for neural networks (see LSCVI)
    dispersion
        Whether dispersion parameters of genes are fit on the level of
        1) datasets 2) batches 3) cells (not implemented: labels)
    log_variational
        Whether to log(x+1) counts x during encoding
    latent_distribution
        Prior on latent space
    gene_likelihood
        One of (see scVI/LSCVI)

            * ``nb`` - Negative binomial distribution
            * ``zinb`` - Zero inflated negative binomial distribution
            * ``poisson`` - Poisson distribution
    use_batch_norm
        Batch norm in encoder/decoder
    use_layer_norm
        Layer norm in encoder
    **model_kwargs
        Keyword arguments for :class:`~sccoral.module._module`


    Examples
    --------
    >>> adata = sccoral.data.simulation_dataset()
    >>> sccoral.model.setup_anndata(adata,
                                    categorical_covariate='categorical_covariate',
                                    continuous_covariate='continuous_covariate'
                                    )
    >>> m = sccoral.model(adata, n_latent=7)
    >>> m.train()
    >>> representation = m.get_latent_representation()  # pd.DataFrame cells x n_latent
    >>> loadings = m.get_loadings()  # pd.DataFrame genes x n_latent
    >>> r2 = m.get_explained_variance_per_factor()  # pd.DataFrame 1 x n_latent

    Notes
    -----
    Upcoming documentation
    1. :doc:

    References
    ----------
    :cite:p:`Svensson20`.
    """

    _module_cls = MODULE
    _data_splitter_cls = DataSplitter
    # scvi.train.TrainingPlan with additional class attributes for pretraining
    _training_plan_cls = ScCoralTrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: ad.AnnData,
        n_latent: int = 10,
        alpha_l1: Tunable[float] = 1000,
        n_hidden: Tunable[int] = 128,
        n_layers: Tunable[int] = 1,
        dropout_rate: Tunable[float] = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-cell"] = "gene",  # TODO gene-label
        log_variational: bool = True,
        latent_distribution: Literal["normal", "ln"] = "normal",
        gene_likelihood: Tunable[Literal["nb", "zinb", "poisson"]] = "nb",
        use_batch_norm: Literal["encoder", "decoder", "both", "none"] = "both",
        use_layer_norm: bool = False,
        use_observed_lib_size: bool = True,
        **vae_kwargs,
    ) -> None:
        super().__init__(adata)

        n_input = self.summary_stats.n_vars

        # BATCH
        n_batch = self.summary_stats.n_batch

        # CATEGORICAL COVARIATES
        # None if not categorical_covariate is passed
        names_categorical = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).get("field_keys")
        n_level_categorical = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).get("n_cats_per_key")

        categorical_mapping = (
            dict(zip(names_categorical, n_level_categorical)) if names_categorical is not None else None
        )

        # CONTINUOUS COVARIATES
        continuous_names = self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY).get("columns")

        # TODO
        (library_log_means, library_log_vars) = _init_library_size(self.adata_manager, n_batch)

        # SETUP MODULE
        self.module = self._module_cls(
            n_input=n_input,
            categorical_mapping=categorical_mapping,
            continuous_names=continuous_names,
            alpha_l1=alpha_l1,
            n_batch=n_batch,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            dispersion=dispersion,
            log_variational=log_variational,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_observed_lib_size=use_observed_lib_size,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **vae_kwargs,
        )

        self.init_params_ = self._get_init_params(locals())

        self._model_summary_string = f"""scCORAL
            n_latent: {n_latent}
            alpha_l1: {alpha_l1}
            n_hidden: {n_hidden}
            n_layers: {n_layers}
            dropout_rate: {dropout_rate}
            """

    def get_loadings(self, set_column_names: bool = True) -> pd.DataFrame:
        """Extract linear weights of decoder

        Parameters
        ----------
        set_column_names
            Whether to set the column names to covariate names

        Returns
        -------
        Pandas DataFrame
            `n_genes` x `n_latent`
        """
        if not self.is_trained_:
            raise RuntimeError("Train model first")

        var_names = self.adata.var_names
        column_names = None
        if set_column_names:
            column_names = [
                # Free factors
                *list(range(self.module.n_latent)),
                *self.module.categorical_names,
                *self.module.continuous_names,
            ]

        loadings = pd.DataFrame(self.module.get_loadings(), index=var_names, columns=column_names)

        return loadings

    @inference_mode()
    def get_latent_representation(
        self,
        adata: None | ad.AnnData = None,
        set_column_names: bool = True,
        indices: Iterable = None,
        batch_size: int = None,
    ) -> pd.DataFrame:
        """Get latent representation of cells in anndata object

        Parameters
        ----------
        adata
            AnnData object to embed. If `None` use stored `anndata.AnnData`
        set_column_names
            Whether to set the column names to covariate names
        indices
            Indices of cells to retrieve
        batch_size
            Batch size during inference.

        Returns
        -------
        Pandas DataFrame
            `n_cells` x `n_latent`
        """
        if not self.is_trained_:
            raise RuntimeError("Train model first")

        if adata is None:
            adata = self.adata
        adata = self._validate_anndata(adata)

        # Instantiate data loader
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latent = []

        # Inference
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)

            z = outputs["z"]

            latent.append(z.cpu())

        column_names = None
        if set_column_names:
            column_names = [
                # Free factors
                *list(range(self.module.n_latent)),
                *self.module.categorical_names,
                *self.module.continuous_names,
            ]

        return pd.DataFrame(torch.cat(latent).detach().numpy(), index=adata.obs_names, columns=column_names)

    @inference_mode()
    def get_explained_variance_per_factor(
        self, adata: None | ad.AnnData, set_column_names: bool = False
    ) -> pd.DataFrame:
        """Compute explained variance per factor

        Parameters
        ----------
        adata
            AnnData object to embed. If `None` use stored `anndata.AnnData`
        set_column_names
            Whether to set the column names to covariate names

        Returns
        -------
        Pandas DataFrame
            `1` x `n_latent`
        """
        raise NotImplementedError

    @classmethod
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        batch_key: None | str = None,
        # labels_key: None | str = None,
        categorical_covariates: None | str | Iterable[str] = None,
        continuous_covariates: None | str | Iterable[str] = None,
        layer: None | str = None,
        **kwargs,
    ):
        if isinstance(categorical_covariates, str):
            categorical_covariates = [categorical_covariates]
        if isinstance(continuous_covariates, str):
            continuous_covariates = [continuous_covariates]
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            # LabelsField(REGISTRY_KEYS.LABELS_KEY),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariates),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariates),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 500,
        pretraining: bool = True,
        use_gpu: bool = True,
        accelerator: None | Literal["cpu", "gpu", "auto"] = "auto",
        devices="auto",
        validation_size: None | float = 0.1,
        batch_size: int = 128,
        early_stopping: bool = True,
        # TODO refactor into pretraining_kwargs
        pretraining_max_epochs: int = 500,
        pretraining_early_stopping: bool = True,
        pretraining_early_stopping_metric: Tunable[
            None | Literal["reconstruction_loss_train", "train_loss_epoch", "elbo_train"]
        ] = "reconstruction_loss_train",
        pretraining_min_delta: float = 0.0,
        pretraining_early_stopping_patience: Tunable[int] = 5,
        plan_kwargs: None | dict[str, Any] = None,
        trainer_kwargs: None | dict[str, Any] = None,
        **kwargs,
    ) -> None:
        """Train sccoral model

        Training is split into pretraining (only training on covariates, frozen z_encoder weights)
        and training (unfrozen weights).
        Same training procedure as for scVI/LSCVI except for pretraining.

        Parameters
        ----------
        max_epochs
            Maximum epochs during training
        max_pretraining_epochs
            Maximum epochs during pretraining. If `None`, same as max_epochs
        accelerator
            cpu/gpu/auto: auto automatically detects available devices
        devices
            If `auto`, automatically detects available devices
        validation_size
            Size of validation split (0-1). Rest is train split
        batch_size
            Size of minibatches during training
        early_stopping
            Enable early stopping during training
        pretraining
            Whether to conduct pretraining
        pretraining_max_epochs
            Maximum number of epochs for pretraining to continue.
        pretraining_early_stopping
            Enable early stopping during pretraining
        plan_kwargs
            Training keyword arguments passed to `sccoral.train.TrainingPlan`
        trainer_kwargs
            Additional keyword arguments passed to `scvi.train.TrainRunner`
        kwargs
            Not passed.

        Returns
        -------
        Training runner (scvi-tools wrapper of pytorch lightning trainer.)

        """
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        lr = plan_kwargs["lr"] if "lr" in plan_kwargs else 0.001

        trainer_kwargs = trainer_kwargs if isinstance(trainer_kwargs, dict) else {}
        trainer_kwargs["early_stopping"] = (
            early_stopping if "early_stopping" not in trainer_kwargs.keys() else trainer_kwargs["early_stopping"]
        )

        # Data splitter (default)
        assert validation_size < 1 and validation_size >= 0, "validation_size must in interval [0-1)"
        train_size = 1 - validation_size
        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )

        # IMPLEMENT PRETRAINING
        if pretraining:
            check_pretraining_stop_callback = tcb.EarlyStoppingCheck(
                monitor=pretraining_early_stopping_metric,
                min_delta=pretraining_min_delta,
                patience=pretraining_early_stopping_patience,
                mode="min",
                check_on_train=True,
            )
            pretraing_freeze_callback = tcb.PretrainingFreezeWeights(
                submodule="z_encoder",
                n_pretraining_epochs=pretraining_max_epochs,
                early_stopping=pretraining_early_stopping,
                lr=lr,
            )

            if "callbacks" not in trainer_kwargs:
                trainer_kwargs["callbacks"] = []

            trainer_kwargs["callbacks"] += [check_pretraining_stop_callback, pretraing_freeze_callback]

        # PRETRAINING
        # TRAINING
        # PASSED TO pl.Trainer
        training_plan = self._training_plan_cls(module=self.module, **plan_kwargs)

        # Should be left as is
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )

        return runner()
