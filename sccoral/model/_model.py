# Imports
import logging
from collections.abc import Iterable
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scvi import REGISTRY_KEYS

# Changes after scvi 1.0.4
try:
    from scvi.autotune import Tunable, TunableMixin
except ImportError:
    from scvi._types import Tunable, TunableMixin

from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField, NumericalJointObsField
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass, VAEMixin
from scvi.train import TrainRunner
from torch import inference_mode

from sccoral.module import MODULE
from sccoral.tl._stats import _pcr
from sccoral.train import ScCoralDataSplitter, ScCoralTrainingPlan
from sccoral.train import _callbacks as tcb

logger = logging.getLogger(__name__)


class SCCORAL(BaseModelClass, TunableMixin, VAEMixin):
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
        1) datasets ("gene") 2) batches ("gene-batch")
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
    >>> adata = sccoral.data.synthetic_data()
    >>> sccoral.SCCORAL.setup_anndata(adata,
                                      categorical_covariates='categorical_covariate',
                                      continuous_covariates='continuous_covariate'
                                      )
    >>> m = sccoral.SCCORAL(adata, n_latent=7)
    >>> m.train()
    >>> representation = m.get_latent_representation()  # pd.DataFrame cells x n_latent
    >>> loadings = m.get_loadings()  # pd.DataFrame genes x n_latent
    >>> ev = m.get_explained_variance_per_factor()  # pd.DataFrame 1 x n_latent

    References
    ----------
    :cite:p:`Svensson20`.
    """

    _module_cls = MODULE
    _data_splitter_cls = ScCoralDataSplitter
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
        dispersion: Literal["gene", "gene-batch"] = "gene",
        log_variational: bool = True,
        latent_distribution: Literal["normal", "ln"] = "ln",
        gene_likelihood: Tunable[Literal["nb", "zinb", "poisson"]] = "nb",
        use_batch_norm: Literal["encoder", "decoder", "both", "none"] = "both",
        use_layer_norm: Literal["encoder", "none"] = "none",
        use_observed_lib_size: bool = False,
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
            dict(zip(names_categorical, n_level_categorical, strict=False)) if names_categorical is not None else None
        )

        # CONTINUOUS COVARIATES
        continuous_names = self.adata_manager.get_state_registry(REGISTRY_KEYS.CONT_COVS_KEY).get("columns")

        # Library size priors are only needed when the library size is inferred
        # (use_observed_lib_size=False); skip the computation otherwise.
        if not use_observed_lib_size:
            (library_log_means, library_log_vars) = _init_library_size(self.adata_manager, n_batch)
        else:
            library_log_means = library_log_vars = None

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
            categorical_names = self.module.categorical_names if self.module.categorical_names is not None else []
            continuous_names = self.module.continuous_names if self.module.continuous_names is not None else []

            column_names = [
                # Free factors
                *list(range(self.module.n_latent)),
                *categorical_names,
                *continuous_names,
            ]

        loadings = pd.DataFrame(self.module.get_loadings(), index=var_names, columns=column_names)

        return loadings

    @inference_mode()
    def get_latent_representation(
        self,
        adata: ad.AnnData | None = None,
        indices: Iterable[int] | None = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: int | None = None,
        return_dist: bool = False,
        set_column_names: bool = True,
        suffix: str | None = None,
    ) -> pd.DataFrame | tuple[np.ndarray, np.ndarray]:
        """Get latent representation of cells in anndata object

        Parameters
        ----------
        adata
            AnnData object to embed. If `None` use stored `anndata.AnnData`
        indices
            Indices of cells to retrieve (see scvi-tools)
        give_mean
            Whether to give the full distribution or mean of distribution. Defaults to mean
            See scvi-tools
        mc_samples
            For distributions with no closed analytical solution - how many samples to draw (see scvi-tools)
        batch_size
            Batch size during inference.
        return_dist
            Whether to return single-measurement values (False) or parameters of the distribution (True)
            See scvi-tools
        set_column_names
            Whether to set the column names to covariate names (defaults to True)
        suffix
            Whether to add a suffix (e.g. `__factor`) so that columns in dataframe are better distinguishable
            from metadata info. Per default, no suffix is added.

        Returns
        -------
        Pandas DataFrame
            `n_cells` x `n_latent`
        """
        res = super().get_latent_representation(adata, indices, give_mean, mc_samples, batch_size, return_dist)

        if adata is None:
            adata = self.adata

        if return_dist:
            return res

        else:
            column_names = None
            if set_column_names:
                categorical_names = self.module.categorical_names if self.module.categorical_names is not None else []
                continuous_names = self.module.continuous_names if self.module.continuous_names is not None else []

                column_names = [
                    # Free factors
                    *list(range(self.module.n_latent)),
                    *categorical_names,
                    *continuous_names,
                ]
            if suffix is not None and column_names is not None:
                column_names = [f"{col}{suffix}" for col in column_names]

            return pd.DataFrame(res, index=adata.obs_names, columns=column_names)

    @inference_mode()
    def get_explained_variance_per_factor(
        self, adata: ad.AnnData | None = None, set_column_names: bool = True, run_pca: bool = False
    ) -> pd.DataFrame:
        """Compute the explained variance per factor via principal component regression

        Each latent factor is treated as a covariate and regressed against the principal
        components of the data (see :func:`sccoral.tl.stats.principal_component_regression`).
        For factor ``k`` the value is the fraction of the data's total (PCA) variance that
        is linearly explained by that factor:

            ``EV_k = sum_pc R2(factor_k -> PC_pc) * variance_ratio_pc / sum_pc variance_ratio_pc``

        Factors are scored independently, so the values lie in `[0, 1]` but do **not**
        sum to `1` (correlated factors can each explain overlapping variance).

        Parameters
        ----------
        adata
            AnnData object to embed. If `None` use stored `anndata.AnnData`. Must contain
            a precomputed PCA in ``obsm['X_pca']`` / ``uns['pca']['variance_ratio']``
            (unless `run_pca` is set).
        set_column_names
            Whether to set the column names to covariate names
        run_pca
            Whether to run PCA with default parameters if it is not found in `adata`
            (otherwise raises `ValueError`).

        Returns
        -------
        Pandas DataFrame
            `1` x `n_latent + n_categorical + n_continuous`. Columns follow the same
            layout as :meth:`get_loadings` / :meth:`get_latent_representation` (free
            factors, then categorical, then continuous covariates). Values lie in `[0, 1]`.

        Raises
        ------
        ValueError
            If `X_pca` is not found in `adata.obsm` and `run_pca` is `False`.
        """
        if not self.is_trained_:
            raise RuntimeError("Train model first")

        if adata is None:
            adata = self.adata

        # PCA is required to decompose the data's variance (mirrors
        # sccoral.tl.stats.principal_component_regression).
        if "X_pca" not in adata.obsm:
            if not run_pca:
                raise ValueError("Run PCA first")
            logger.warning("X_pca not found. Run PCA with default parameters")
            sc.pp.pca(adata)

        X_pca = adata.obsm["X_pca"].T
        variance_ratio = adata.uns["pca"]["variance_ratio"]

        # Latent representation (cells x factors); each factor is a covariate for PCR.
        # `_pcr` already clips each value into [0, 1].
        z = self.get_latent_representation(adata, set_column_names=False).to_numpy()
        explained_variance = np.array([_pcr(z[:, [k]], X_pca, variance_ratio) for k in range(z.shape[1])])

        column_names = None
        if set_column_names:
            categorical_names = self.module.categorical_names if self.module.categorical_names is not None else []
            continuous_names = self.module.continuous_names if self.module.continuous_names is not None else []

            column_names = [
                # Free factors
                *list(range(self.module.n_latent)),
                *categorical_names,
                *continuous_names,
            ]

        return pd.DataFrame(explained_variance[np.newaxis, :], index=["explained_variance"], columns=column_names)

    @classmethod
    def setup_anndata(
        cls,
        adata: ad.AnnData,
        batch_key: None | str = None,
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
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariates),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariates),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: int = 2000,
        pretraining: Tunable[bool] = True,
        accelerator: None | Literal["cpu", "gpu", "auto"] = "auto",
        devices="auto",
        validation_size: None | float = 0.1,
        batch_size: int = 128,
        early_stopping: Tunable[bool] = True,
        pretraining_max_epochs: Tunable[int] = 500,
        pretraining_early_stopping: Tunable[bool] = True,
        pretraining_early_stopping_metric: Tunable[
            Literal[
                "reconstruction_loss_validation",
                "elbo_validation",
                "reconstruction_loss_train",
                "train_loss_epoch",
                "elbo_train",
            ]
        ] = "reconstruction_loss_validation",
        pretraining_min_delta: Tunable[float] = 0.0,
        pretraining_early_stopping_patience: Tunable[int] = 5,
        plan_kwargs: None | dict[str, Any] = None,
        trainer_kwargs: None | dict[str, Any] = None,
        **kwargs,
    ) -> Any:
        """Train sccoral model

        Training is split into pretraining (only training on covariates, frozen z_encoder weights)
        and training (unfrozen weights).
        Same training procedure as for scVI/LSCVI except for pretraining.

        Parameters
        ----------
        max_epochs
            Maximum epochs during training
        pretraining
            Whether to conduct pretraining
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
        pretraining_max_epochs
            Maximum number of epochs for pretraining to continue.
        pretraining_early_stopping
            Enable early stopping during pretraining
        pretraining_early_stopping_metric
            Metric monitored for pretraining early stopping. Metrics ending in
            `_validation` require a validation split; without one they fall back to
            the corresponding `_train` metric.
        pretraining_min_delta
            Minimum change in the monitored metric to qualify as an improvement.
        pretraining_early_stopping_patience
            Number of checks with no improvement before pretraining early stopping triggers.
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
        if validation_size is None or not (0 <= validation_size < 1):
            raise ValueError("validation_size must be a float in the interval [0, 1)")

        # Early stopping monitors a validation metric, which requires a validation split.
        # With validation_size=0 there is no val dataloader, so scvi/Lightning would request
        # one and crash on the `None` returned by `val_dataloader`. Disable it in that case.
        if validation_size == 0 and trainer_kwargs["early_stopping"]:
            logger.warning(
                "`validation_size=0` leaves no validation split, but `early_stopping=True` "
                "monitors a validation metric. Disabling early stopping for this run."
            )
            trainer_kwargs["early_stopping"] = False

        train_size = 1 - validation_size
        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )

        # IMPLEMENT PRETRAINING
        if pretraining:
            # Validation metrics require a validation split. If none is requested,
            # fall back to the equivalent training metric so early stopping still works.
            if validation_size == 0 and pretraining_early_stopping_metric.endswith("_validation"):
                fallback_metric = pretraining_early_stopping_metric.replace("_validation", "_train")
                logger.warning(
                    f"`validation_size=0` but `pretraining_early_stopping_metric` is "
                    f"'{pretraining_early_stopping_metric}', which is unavailable without a "
                    f"validation split. Falling back to '{fallback_metric}'."
                )
                pretraining_early_stopping_metric = fallback_metric

            # Validation metrics are only available at validation epoch end, so
            # check there; training metrics are checked at training epoch end.
            check_on_train = not pretraining_early_stopping_metric.endswith("_validation")

            # scvi's TrainRunner only enables validation when early_stopping/checkpointing
            # are active. If we're monitoring a validation metric for pretraining, we must
            # ensure validation runs every epoch — otherwise the metric is never logged.
            if not check_on_train and "check_val_every_n_epoch" not in trainer_kwargs:
                trainer_kwargs["check_val_every_n_epoch"] = 1
            check_pretraining_stop_callback = tcb.EarlyStoppingCheck(
                monitor=pretraining_early_stopping_metric,
                min_delta=pretraining_min_delta,
                patience=pretraining_early_stopping_patience,
                mode="min",
                check_on_train=check_on_train,
            )
            pretraing_freeze_callback = tcb.PretrainingFreezeWeights(
                submodule="z_encoder",
                n_pretraining_epochs=pretraining_max_epochs,
                early_stopping=pretraining_early_stopping,
                lr=lr,
                train_batch_norm=False,
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
