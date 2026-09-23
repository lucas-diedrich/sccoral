import anndata as ad
import numpy as np
import pytest
import sccoral


@pytest.fixture
def adata():
    data = ad.AnnData(np.zeros((2, 8)))
    data.uns["pca"] = {"variance_ratio": np.array([0.5, 0.25, 0.1, 0.03, 0.025, 0.02, 0.015, 0.01])}
    return data


def test_public_selector_returns_component_count_without_mutation(adata):
    before = adata.copy()
    options = {"S": 1.0}
    result = sccoral.tl.select_dimensionality(adata, kneed_kws=options)
    # This curve's elbow is at PC 3; return a count, not the zero-based index 2.
    assert isinstance(result, int)
    assert result == 3
    assert options == {"S": 1.0}
    assert set(adata.uns) == set(before.uns)
    np.testing.assert_array_equal(adata.uns["pca"]["variance_ratio"], before.uns["pca"]["variance_ratio"])
    np.testing.assert_array_equal(adata.X, before.X)
    assert not adata.obsm


def test_log_and_custom_pca_key(adata):
    adata.uns["custom_pca"] = adata.uns.pop("pca")
    before = adata.uns["custom_pca"]["variance_ratio"].copy()
    assert sccoral.tl.select_dimensionality(adata, pca_key="custom_pca", log=True) == 4
    np.testing.assert_array_equal(adata.uns["custom_pca"]["variance_ratio"], before)


@pytest.mark.parametrize("pca", [None, {}, {"variance": [0.5, 0.3, 0.1]}])
def test_missing_pca(pca):
    data = ad.AnnData()
    if pca is not None:
        data.uns["pca"] = pca
    with pytest.raises(ValueError, match="Run PCA first"):
        sccoral.tl.select_dimensionality(data)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([], "at least three"),
        ([0.5, 0.2], "at least three"),
        ([[0.5, 0.2, 0.1]], "1-D"),
        ([0.5, np.nan, 0.1], "finite"),
        ([np.inf, 0.2, 0.1], "finite"),
        ([0.5, 0.2, -0.1], "nonnegative"),
        ([0.2, 0.5, 0.1], "descending"),
        ([0.2, 0.2, 0.2], "constant"),
        ([0, 0, 0], "constant"),
        (["bad", "data", "here"], "numeric"),
    ],
)
def test_invalid_variance_ratios(adata, values, message):
    adata.uns["pca"]["variance_ratio"] = values
    with pytest.raises(ValueError, match=message):
        sccoral.tl.select_dimensionality(adata)


def test_log_rejects_zero(adata):
    adata.uns["pca"]["variance_ratio"][-1] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        sccoral.tl.select_dimensionality(adata, log=True)
    assert sccoral.tl.select_dimensionality(adata) == 3


def test_no_elbow(adata):
    adata.uns["pca"]["variance_ratio"] = np.array([8, 7, 6, 5, 4, 3, 2, 1]) / 36
    with pytest.raises(ValueError, match="No PCA elbow detected"):
        sccoral.tl.select_dimensionality(adata)


@pytest.mark.parametrize("key", ["x", "y", "curve", "direction"])
def test_reserved_options(adata, key):
    with pytest.raises(ValueError, match="Reserved KneeLocator options"):
        sccoral.tl.select_dimensionality(adata, kneed_kws={key: None})


def test_sensitivity_is_forwarded(adata):
    with pytest.raises(ValueError, match="No PCA elbow detected"):
        sccoral.tl.select_dimensionality(adata, kneed_kws={"S": 100})
