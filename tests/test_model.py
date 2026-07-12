import numpy as np
import pytest
import sccoral
import torch
from sccoral.model import SCCORAL
from scvi.data import synthetic_iid


def test_package_has_version():
    assert sccoral.__version__ is not None


@pytest.fixture(scope="module")
def adata():
    """Create synthetic adata with categorical+continuous covariate"""
    adata = synthetic_iid(batch_size=200, n_genes=100, n_proteins=0, n_regions=0, n_batches=2, n_labels=2)

    adata.obs["categorical_covariate"] = np.random.choice(["A", "B"], size=adata.n_obs, replace=True)
    adata.obs["continuous_covariate"] = 1

    return adata


@pytest.mark.parametrize(["cat_name", "con_name"], [["categorical_covariate", None], ["continuous_covariate", None]])
def test_setup_adata(adata, cat_name, con_name):
    SCCORAL.setup_anndata(adata, categorical_covariates=cat_name, continuous_covariates=con_name)


@pytest.mark.parametrize(["cat_name", "con_name"], [["categorical_covariate", None], ["continuous_covariate", None]])
def test_setup_model(adata, cat_name, con_name):
    SCCORAL.setup_anndata(adata, categorical_covariates=cat_name, continuous_covariates=con_name)

    SCCORAL(adata, n_latent=5)


def test_train(adata):
    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )
    model = SCCORAL(adata, n_latent=5)
    model.train(pretraining=False, max_epochs=10)


def test_train_no_validation_split(adata):
    """validation_size=0 must not crash even with the default early_stopping=True."""
    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )
    model = SCCORAL(adata, n_latent=5)
    # early_stopping defaults to True; with no validation split it should be auto-disabled
    model.train(max_epochs=2, accelerator="cpu", validation_size=0)


@pytest.mark.parametrize(["max_pretraining_epochs", "requires_grad"], [[10, True], [21, False]])
def test_pretraining_max_epochs(adata, max_pretraining_epochs, requires_grad):
    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )
    model = SCCORAL(adata, n_latent=5)

    model.train(
        max_epochs=20,
        pretraining=True,
        accelerator="cpu",
        early_stopping=False,
        pretraining_max_epochs=max_pretraining_epochs,
        pretraining_early_stopping=False,
    )
    assert all((param.requires_grad == requires_grad for _, param in model.module.z_encoder.named_parameters()))


@pytest.mark.parametrize(["pretraining_early_stopping", "requires_grad"], [[True, True], [False, False]])
def test_pretraining_early_stopping(adata, pretraining_early_stopping, requires_grad):
    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )
    model = SCCORAL(adata, n_latent=5)

    model.train(
        max_epochs=20,
        pretraining=True,
        accelerator="cpu",
        early_stopping=False,
        pretraining_early_stopping=pretraining_early_stopping,
        pretraining_max_epochs=21,
        pretraining_min_delta=np.inf,
        pretraining_early_stopping_patience=1,
    )
    assert all((param.requires_grad == requires_grad for _, param in model.module.z_encoder.named_parameters()))


def test_train_singleton_final_batch():
    """n_cells == k*batch_size + 1 must not crash BatchNorm (singleton last batch)."""
    batch_size = 128
    n_obs = 2 * batch_size + 1  # 257 -> last train batch would be a single cell

    adata = synthetic_iid(batch_size=n_obs, n_genes=50, n_proteins=0, n_regions=0, n_batches=1, n_labels=2)
    adata.obs["categorical_covariate"] = np.random.choice(["A", "B"], size=adata.n_obs, replace=True)
    adata.obs["continuous_covariate"] = 1

    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )
    model = SCCORAL(adata, n_latent=5)
    # validation_size=0 keeps all cells in the train split so the remainder is exactly 1.
    # early_stopping/pretraining are disabled to isolate the train-loader BatchNorm path.
    model.train(
        max_epochs=2,
        accelerator="cpu",
        validation_size=0,
        batch_size=batch_size,
        early_stopping=False,
        pretraining=False,
    )


@pytest.fixture(scope="module", params=["normal", "ln"])
def basic_train(adata, request):
    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )
    model = SCCORAL(adata, n_latent=5, latent_distribution=request.param)
    model.train(max_epochs=20, accelerator="cpu")

    return model


def test_loadings(basic_train):
    loadings = basic_train.get_loadings()

    assert loadings.shape == (100, 7)
    assert "categorical_covariate" in loadings.columns
    assert "continuous_covariate" in loadings.columns


def test_representation(basic_train):
    representation = basic_train.get_latent_representation()
    assert representation.shape == (400, 7)
    assert "categorical_covariate" in representation.columns
    assert "continuous_covariate" in representation.columns


def test_representation_suffix(basic_train):
    representation = basic_train.get_latent_representation(suffix="__factor")
    assert "categorical_covariate__factor" in representation.columns
    assert "continuous_covariate__factor" in representation.columns


def test_explained_variance_per_factor(basic_train):
    # PCR-based metric requires a precomputed PCA on the data
    ev = basic_train.get_explained_variance_per_factor(run_pca=True)

    # n_latent (5) + categorical (1) + continuous (1) factors
    assert ev.shape == (1, 7)
    assert "categorical_covariate" in ev.columns
    assert "continuous_covariate" in ev.columns
    # PCR explained variance per factor lies in [0, 1] (does not sum to 1)
    values = ev.to_numpy().ravel()
    assert ((values >= 0) & (values <= 1)).all()


def test_explained_variance_per_factor_requires_pca(basic_train):
    """Without a precomputed PCA (and run_pca=False) the method must raise."""
    adata = synthetic_iid(batch_size=50, n_genes=100, n_proteins=0, n_regions=0, n_batches=1, n_labels=2)
    adata.obs["categorical_covariate"] = np.random.choice(["A", "B"], size=adata.n_obs, replace=True)
    adata.obs["continuous_covariate"] = 1
    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )

    with pytest.raises(ValueError, match="Run PCA first"):
        basic_train.get_explained_variance_per_factor(adata)


def test_marginal_ll(basic_train):
    """`get_marginal_ll` must return one finite value per cell (batch dim preserved)."""
    per_cell = basic_train.get_marginal_ll(n_mc_samples=10, return_mean=False)
    assert per_cell.shape == (400,)  # n_cells
    assert torch.isfinite(per_cell).all()

    mean_ll = basic_train.get_marginal_ll(n_mc_samples=10, return_mean=True)
    assert isinstance(mean_ll, float)
    assert np.isfinite(mean_ll)


def test_get_reconstruction_error(basic_train):
    # Setup new anndata
    adata = synthetic_iid(batch_size=50, n_genes=100, n_proteins=0, n_regions=0, n_batches=1, n_labels=2)
    adata.obs["categorical_covariate"] = np.random.choice(["A", "B"], size=adata.n_obs, replace=True)
    adata.obs["continuous_covariate"] = 1

    SCCORAL.setup_anndata(
        adata, categorical_covariates="categorical_covariate", continuous_covariates="continuous_covariate"
    )

    error = basic_train.get_reconstruction_error(adata)
    assert isinstance(error["reconstruction_loss"], float)
