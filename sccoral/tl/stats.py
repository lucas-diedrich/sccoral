import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

logger = logging.Logger(__name__)
logger.setLevel("warn")


def principal_component_regression(
    adata: ad.AnnData, covariate: str, transform_categorical: bool = True, raise_error: bool = True
) -> float:
    """Run principal component regression

    Parameters
    ----------
    adata
        AnnData Object
    covariate
        Covariate of interest for which we run PCR.
        Categorical covariates are automatically transformed into
        one hot encodings (must have `pd.CategoricalDtype`)
    transform_categorical
        Whether to transform categorical covariates
    raise_error
        Whether to raise an error if there is not PCA in the anndata
        (otherwise, run default parameters)

    Returns
    -------
    Explained variance by covariate in first 50 components (float)

    Raises
    ------
    ValueError
        If `X_pca` not in `adata.obsm`
    """
    # Validate that PCA was run on data
    # Run PCA
    if "X_pca" not in adata.obsm:
        if raise_error:
            raise ValueError("Run PCA first")
        logger.warning("X_pca not found. Run PCA with default parameters")
        sc.pp.highly_variable_genes(adata)
        sc.pp.pca(adata)

    X_pca = adata.obsm["X_pca"].T
    explained_variance = adata.uns["pca"]["variance_ratio"]

    x = adata.obs[covariate].to_numpy().reshape(-1, 1)
    if isinstance(adata.obs[covariate].dtype, pd.CategoricalDtype):
        x = OneHotEncoder(sparse_output=False, drop="if_binary").fit_transform(x)

    covariate_explained_variance = _pcr(x, X_pca, explained_variance)

    return covariate_explained_variance


# TODO Write test
def _pcr(x, X_pca, explained_variance) -> float:
    """Run PCR"""
    scores = []
    for y in X_pca:
        scores.append(LinearRegression().fit(x, y).score(x, y))

    covariate_explained_variance = np.sum(np.array(scores) * explained_variance)
    total_explained_variance = np.sum(explained_variance)

    return covariate_explained_variance / total_explained_variance
