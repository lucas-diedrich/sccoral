import anndata as ad
import numpy as np
import pytest
import torch
from sccoral.model import SCCORAL


@pytest.fixture(params=["normal", "ln"])
def posterior_model(request):
    """Use a known narrow posterior to check activities without fitting a model."""
    rng = np.random.default_rng(42)
    adata = ad.AnnData(rng.poisson(3, (12, 10)).astype(np.float32))
    adata.obs["category"] = ["A", "B"] * 6
    adata.obs["continuous"] = np.linspace(0, 1, 12)
    SCCORAL.setup_anndata(adata, categorical_covariates="category", continuous_covariates="continuous")
    model = SCCORAL(adata, n_latent=3, latent_distribution=request.param, use_batch_norm="none")
    model.module.eval()
    # Bypass the trained guard only for this controlled posterior fixture.
    model.is_trained_ = True
    with torch.no_grad():
        encoder = model.module.z_encoder
        encoder.mean_encoder.weight.zero_()
        encoder.mean_encoder.bias.zero_()
        encoder.var_encoder.weight.zero_()
        encoder.var_encoder.bias.fill_(-20)
        for encoder in (model.module.encoder_category, model.module.encoder_continuous):
            encoder.mean.weight.zero_()
            encoder.mean.bias.zero_()
            encoder.var.weight.zero_()
            encoder.var.bias.fill_(-20)
    return model


def test_posterior_mean_matches_training_activities(posterior_model):
    model = posterior_model
    torch.manual_seed(42)
    values = model.get_latent_representation(mc_samples=256, batch_size=5).to_numpy()
    if model.module.latent_distribution == "ln":
        # Symmetry gives E[softmax(a_free)] = 1/U and E[sigmoid(a_cov)] = 1/2.
        np.testing.assert_allclose(values[:, :3], 1 / 3, atol=0.002)
        np.testing.assert_allclose(values[:, 3:], 0.5, atol=0.002)
        np.testing.assert_allclose(values[:, :3].sum(-1), 1, atol=1e-6)
        assert np.all(values.sum(-1) > 1.9)
    else:
        np.testing.assert_array_equal(values, 0)


@pytest.mark.parametrize("n_samples", [1, 512])
def test_inference_uses_same_latent_blocks(posterior_model, n_samples):
    model = posterior_model
    tensors = next(iter(model._make_data_loader(model.adata, batch_size=12)))
    torch.manual_seed(42)
    with torch.no_grad():
        output = model.module.inference(**model.module._get_inference_input(tensors), n_samples=n_samples)
    values = output["z"].numpy()
    if model.module.latent_distribution == "ln":
        np.testing.assert_allclose(values[..., :3].sum(-1), 1, atol=1e-6)
        np.testing.assert_allclose(values[..., 3:], 0.5, atol=0.02)
    else:
        np.testing.assert_allclose(values, 0, atol=0.05)


def test_covariate_shift_does_not_renormalize_other_factors(posterior_model):
    model = posterior_model
    torch.manual_seed(42)
    before = model.get_latent_representation(mc_samples=256)
    with torch.no_grad():
        model.module.encoder_category.mean.bias.fill_(2)
    torch.manual_seed(42)
    after = model.get_latent_representation(mc_samples=256)
    np.testing.assert_array_equal(before.iloc[:, [0, 1, 2, 4]], after.iloc[:, [0, 1, 2, 4]])
    assert (after["category"] > before["category"]).all()


def test_distribution_sample_and_cell_indices(posterior_model):
    model = posterior_model
    with torch.no_grad():
        model.module.encoder_continuous.mean.weight.fill_(2)
    mean, variance = model.get_latent_representation(return_dist=True, batch_size=5)
    np.testing.assert_allclose(mean[:, -1], 2 * model.adata.obs["continuous"], atol=1e-6)
    np.testing.assert_allclose(variance, 1e-4, atol=1e-7)
    for indices in ([8, 2, 8], list(reversed(range(12)))):
        subset_mean, subset_variance = model.get_latent_representation(indices=indices, return_dist=True)
        np.testing.assert_allclose(subset_mean, mean[indices])
        np.testing.assert_allclose(subset_variance, variance[indices])
        result = model.get_latent_representation(indices=iter(indices), mc_samples=32, suffix="__factor")
        assert result.index.tolist() == model.adata.obs_names[indices].tolist()
        assert result.columns[-1] == "continuous__factor"
    torch.manual_seed(42)
    sample = model.get_latent_representation(give_mean=False)
    torch.manual_seed(42)
    tensors = next(iter(model._make_data_loader(model.adata)))
    with torch.no_grad():
        expected = model.module.inference(**model.module._get_inference_input(tensors))["z"]
    np.testing.assert_array_equal(sample.to_numpy(), expected.numpy())


@pytest.mark.parametrize("covariates", ["none", "categorical", "continuous"])
def test_extraction_with_missing_covariate_blocks(covariates):
    adata = ad.AnnData(np.full((6, 8), 3, dtype=np.float32))
    adata.obs["category"] = ["A", "B"] * 3
    adata.obs["continuous"] = np.linspace(0, 1, 6)
    SCCORAL.setup_anndata(
        adata,
        categorical_covariates="category" if covariates == "categorical" else None,
        continuous_covariates="continuous" if covariates == "continuous" else None,
    )
    model = SCCORAL(adata, n_latent=3, use_batch_norm="none")
    model.module.eval()
    model.is_trained_ = True
    values = model.get_latent_representation(mc_samples=32).to_numpy()
    np.testing.assert_allclose(values[:, :3].sum(-1), 1, atol=1e-6)
    assert values.shape[1] == 3 + (covariates != "none")
    if covariates != "none":
        assert ((values[:, 3] > 0) & (values[:, 3] < 1)).all()
