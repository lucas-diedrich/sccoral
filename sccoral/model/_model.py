# Imports
import logging
from collections.abc import Iterable
from typing import Any, Literal

import anndata as ad
import pandas as pd
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField, NumericalJointObsField
from scvi.model._utils import _init_library_size
from scvi.model.base import BaseModelClass
from scvi.utils import setup_anndata_dsp
from torch import inference_mode

from sccoral.module import MODULE

logger = logging.getLogger(__name__)


class SCCORAL(BaseModelClass):
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

    def __init__(
        self,
        adata: ad.AnnData,
        n_latent: int = 10,
        alpha_l1: float = 0.1,
        n_hidden: int = 128,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        #  TODO dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["nb", "zinb", "poisson"] = "nb",
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
        pass

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
        pass

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
        pass

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
        max_pretraining_epochs: None | int = 500,
        accelerator: None | Literal["cpu", "gpu"] = None,
        validation_size: None | float = None,
        plan_kwargs: None | dict[str, Any] = None,
        batch_size: int = 128,
        early_stopping_pretraining: bool = True,
        early_stopping: bool = True,
        **trainer_kwargs: Any,
    ) -> None:
        # TODO
        pass
