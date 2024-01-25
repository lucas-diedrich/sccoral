# Imports
import logging
from collections.abc import Iterable
from typing import Any, Literal

import anndata as ad
import pandas as pd
from scvi import REGISTRY_KEYS
from scvi.autotune import Tunable, TunableMixin
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField, NumericalJointObsField
from scvi.dataloaders import DataSplitter
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass

# from scvi.train import TrainingPlan
from scvi.train import TrainRunner
from scvi.utils import setup_anndata_dsp
from torch import inference_mode

from sccoral.module import MODULE
from sccoral.train import ScCoralTrainingPlan
from sccoral.train import _callbacks as tcb

logger = logging.getLogger(__name__)


class SCCORAL(TunableMixin, BaseModelClass):
    """Single-cell COvariate-informed Regularized variational Autoencoder with Linear Decoder

    Parameters
    ----------
    adata
        Registered AnnData object
    n_latent
        Number of latent dimensions
    n_layers
        Number of layers in encoder neural network (see LSCVI)
    dropout_rate
        Dropout rate for neural networks (see LSCVI)
    dispersion
        #TODO
    gene_likelihood
        One of (see scVI/LSCVI)

            * ``nb`` - Negative binomial distribution
            * ``zinb`` - Zero inflated negative binomial distribution
            * ``poisson`` - Poisson distribution

    latent_distribution
        # TODO

    **model_kwargs
        Keyword arguments for :class:`~sccoral.module._module`


    Examples
    --------
    >>> adata = sccoral.data.simulation_dataset()
    >>> sccoral.model.setup_anndata(adata,
                                    n_latent=7,
                                    categorical_covariate='categorical_covariate',
                                    continuous_covariate='continuous_covariate'
                                    )
    >>> m = sccoral.model(adata)
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
    _data_splitter_class = DataSplitter
    # scvi.train.TrainingPlan with additional class attributes for pretraining
    _training_plan_class = ScCoralTrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
        self,
        adata: ad.AnnData,
        n_latent: int = 10,
        alpha_l1: Tunable[float] = 0.1,
        n_hidden: Tunable[int] = 128,
        n_layers: Tunable[int] = 1,
        dropout_rate: Tunable[float] = 0.1,
        #  TODO dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Tunable[Literal["nb", "zinb", "poisson"]] = "nb",
        # TODO latent_distribution: Literal["normal", "lognormal"] = "normal",
        **model_kwargs,
    ) -> None:
        super().__init__(adata)

        # BATCH
        n_batch = self.summary_stats.nbatch()

        # CATEGORICAL COVARIATES
        # None if not categorical_covariate is passed
        names_categorical = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).get("field_keys")
        n_level_categorical = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).get("n_cats_per_key")

        category_mapping = dict(zip(names_categorical, n_level_categorical)) if names_categorical is not None else None
        # CONTINUOUS COVARIATES
        n_continous = self.summary_stats.n_extra_continuous_covs

        names_continous = self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY).get("field_keys")

        # TODO
        (library_log_means, library_log_vars) = _init_library_size(self.adata_manager, n_batch)

        # SETUP MODULE
        self.module = self._module_cls(**model_kwargs)

        # TODO Necessary?
        self.n_latent = n_latent

        self.init_params_ = self._get_init_params(locals())

        self._model_summary_string = f"""scCORAL
            n_latent: {n_latent}
            alpha_l1: {alpha_l1}
            n_hidden: {n_hidden}
            n_layers: {n_layers}
            dropout_rate: {dropout_rate}
            """

    def get_loadings(self, set_column_names: bool = False) -> pd.DataFrame:
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
        raise NotImplementedError

    @inference_mode()
    def get_latent_representation(self, adata: None | ad.AnnData, set_columns_names: bool = False) -> pd.DataFrame:
        """Get latent representation of cells in anndata object

        Parameters
        ----------
        adata
            AnnData object to embed. If `None` use stored `anndata.AnnData`
        set_column_names
            Whether to set the column names to covariate names

        Returns
        -------
        Pandas DataFrame
            `n_cells` x `n_latent`
        """
        raise NotImplementedError

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
        raise NotImplementedError()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        batch_key: None | str = None,
        # labels_key: None | str = None,
        cat_keys: None | Iterable[str] = None,
        cont_keys: None | Iterable[str] = None,
        layer: None | str = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_layer)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            # LabelsField(REGISTRY_KEYS.LABELS_KEY),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, cat_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, cont_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 500,
        pretraining: bool = True,
        use_gpu: None | bool = None,
        accelerator: None | Literal["cpu", "gpu", "auto"] = "auto",
        devices="auto",
        train_size: None | float = 0.9,
        batch_size: int = 128,
        early_stopping: bool = True,
        # TODO refactor into pretraining_kwargs
        pretraining_max_epochs: int = 500,
        pretraining_early_stopping: bool = True,
        pretraining_early_stopping_metric: Tunable[None | Literal["reconstruction_loss"]] = None,
        pretraining_min_delta: float = 0.0,
        pretraining_early_stopping_patience: Tunable[int] = 5,
        plan_kwargs: None | dict[str, Any] = None,
        trainer_kwargs: None | dict[str, Any] = None,
    ) -> None:
        """Train sccoral model

        Training is split into pretraining (only training on covariates, frozen z_encoder weights)
        and training (unfrozen weights)

        max_epochs
            Maximum epochs during training
        max_pretraining_epochs
            Maximum epochs during pretraining. If `None`, same as max_epochs
        use_gpu
            Whether to use gpu. If `None` automatically detects gpu
        accelerator
            cpu/gpu/auto: auto automatically detects available devices
        devices
            If `auto`, automatically detects available devices
        train_size
            Size of train split (0-1). Rest is validation split
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
        """
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
            )

            trainer_kwargs["callbacks"] += [check_pretraining_stop_callback, pretraing_freeze_callback]

        # PRETRAINING
        # TRAINING
        # PASSED TO pl.Trainer
        training_plan = self._training_plan(module=self.module, **plan_kwargs)

        assert (train_size <= 1) & (train_size > 0)
        validation_size = 1 - train_size

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

        # Should be left as is
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )

        return runner()
