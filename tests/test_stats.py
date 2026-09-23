import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from sccoral.tl._stats import _pcr, principal_component_regression
from scvi.data import synthetic_iid


def test_pcr_weighted_r2():
    """`_pcr` returns the variance-ratio-weighted R2 of the covariate across PCs."""
    # Two orthogonal, zero-mean principal components
    pc0 = np.array([1.0, -1.0, 1.0, -1.0])
    pc1 = np.array([1.0, 1.0, -1.0, -1.0])
    X_pca = np.vstack([pc0, pc1])  # (n_pcs, n_cells), as principal_component_regression passes it
    variance_ratio = np.array([0.6, 0.4])

    # x == pc0 -> R2(pc0)=1, R2(pc1)=0 -> (1*0.6 + 0*0.4) / (0.6+0.4) = 0.6
    result = _pcr(pc0.reshape(-1, 1), X_pca, variance_ratio)
    assert isinstance(result, float)
    assert np.isclose(result, 0.6)


def test_pcr_constant_covariate_is_non_negative():
    """A constant covariate explains no variance and must never return a negative value."""
    rng = np.random.default_rng(0)
    X_pca = rng.normal(size=(5, 50))  # 5 PCs x 50 cells
    variance_ratio = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
    x = np.ones((50, 1))  # constant covariate

    result = _pcr(x, X_pca, variance_ratio)
    assert result == 0.0


@pytest.fixture(scope="module")
def pca_adata():
    """Synthetic AnnData with a precomputed PCA and a categorical covariate."""
    adata = synthetic_iid(batch_size=100, n_genes=50, n_proteins=0, n_regions=0, n_batches=1, n_labels=2)
    adata.obs["categorical_covariate"] = pd.Categorical(np.random.choice(["A", "B"], size=adata.n_obs))
    sc.pp.pca(adata)
    return adata


def test_principal_component_regression_in_unit_interval(pca_adata):
    ev = principal_component_regression(pca_adata, "categorical_covariate", transform_categorical=True)
    assert isinstance(ev, float)
    assert 0.0 <= ev <= 1.0


def test_principal_component_regression_honors_transform_categorical(pca_adata):
    """`transform_categorical=False` must skip the one-hot encoding of categoricals.

    Without the OHE the covariate stays as strings, so the linear regression can no
    longer be fit -- confirming the flag actually gates the transform.
    """
    with pytest.raises(ValueError):
        principal_component_regression(pca_adata, "categorical_covariate", transform_categorical=False)


def test_principal_component_regression_requires_pca():
    """Without a precomputed PCA (and run_pca=False) it must raise."""
    adata = synthetic_iid(batch_size=50, n_genes=50, n_proteins=0, n_regions=0, n_batches=1, n_labels=2)
    adata.obs["categorical_covariate"] = pd.Categorical(np.random.choice(["A", "B"], size=adata.n_obs))

    with pytest.raises(ValueError, match="Run PCA first"):
        principal_component_regression(adata, "categorical_covariate")
