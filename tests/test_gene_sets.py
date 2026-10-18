import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import sccoral
from scipy import sparse


def test_signed_knees_and_metadata():
    tail = np.array([0.5, 0.25, 0.1, 0.03, 0.025, 0.02, 0.015, 0.01])
    weights = pd.DataFrame({"factor": np.r_[tail, -tail, 0]}, index=[f"g{i}" for i in range(17)])
    original = weights.copy()
    result = sccoral.tl.extract_gene_sets(weights, min_genes=1)
    assert result.genes["factor"] == {"positive": ["g0", "g1", "g2"], "negative": ["g8", "g9", "g10"]}
    assert result.selection.loc[("factor", "positive"), "knee"] == 3
    assert not result.selection["fallback_used"].any()
    pd.testing.assert_frame_equal(weights, original)
    clamped = sccoral.tl.extract_gene_sets(weights, min_genes=5, max_genes=6)
    assert clamped.selection["selected"].tolist() == [5, 5]
    assert clamped.selection["size_limited"].all()


def test_fallback_empty_tail_and_ties():
    weights = pd.DataFrame({7: [-1.0] * 8}, index=list("abcdefgh"))
    result = sccoral.tl.extract_gene_sets(weights, min_genes=1, fallback_n=3)
    assert result.genes[7] == {"positive": [], "negative": list("abc")}
    assert result.selection.loc[(7, "negative"), "fallback_used"]
    assert not result.selection.loc[(7, "positive"), "fallback_used"]
    assert result.selection["knee"].isna().all()


@pytest.mark.parametrize("values", [[0, 0, 0], [1, np.nan, -1], [1, np.inf, -1]])
def test_invalid_loadings(values):
    with pytest.raises(ValueError):
        sccoral.tl.extract_gene_sets(pd.DataFrame({"a": values}, index=list("abc")))


@pytest.mark.parametrize(
    "kwargs", [{"min_genes": 0}, {"max_genes": 2, "min_genes": 3}, {"fallback_n": -1}, {"sensitivity": 0}]
)
def test_invalid_selection_options(kwargs):
    with pytest.raises(ValueError):
        sccoral.tl.extract_gene_sets(pd.DataFrame({"a": [1, 2, 3]}, index=list("abc")), **kwargs)


def test_duplicate_gene_names():
    with pytest.raises(ValueError, match="unique"):
        sccoral.tl.extract_gene_sets(pd.DataFrame({"a": [1, 2]}, index=["x", "x"]))


@pytest.fixture
def signature():
    return sccoral.tl.extract_gene_sets(
        pd.DataFrame({"stim": [3, 2, 1, -3, -2, -1]}, index=[f"g{i}" for i in range(6)])
    )


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def expression(request):
    rng = np.random.default_rng(42)
    values = np.log1p(rng.poisson(3, size=(20, 100)).astype(float))
    return ad.AnnData(
        sparse.csr_matrix(values) if request.param else values,
        obs=pd.DataFrame(index=[f"cell{i}" for i in range(20)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(100)]),
    )


def test_score_matches_scanpy_and_preserves_adata(expression, signature):
    before = expression.copy()
    expected = expression.copy()
    sc.tl.score_genes(
        expected, signature.genes["stim"]["positive"], score_name="pos", use_raw=False, n_bins=5, random_state=9
    )
    sc.tl.score_genes(
        expected, signature.genes["stim"]["negative"], score_name="neg", use_raw=False, n_bins=5, random_state=9
    )
    result = sccoral.tl.score_gene_sets(expression, signature, n_bins=5, random_state=9)
    np.testing.assert_allclose(result.scores["stim"], expected.obs.pos - expected.obs.neg)
    assert result.scores.index.equals(expression.obs_names)
    assert result.coverage["scored"].all()
    pd.testing.assert_frame_equal(expression.obs, before.obs)
    if sparse.issparse(expression.X):
        assert (expression.X != before.X).nnz == 0
    else:
        np.testing.assert_array_equal(expression.X, before.X)
    repeated = sccoral.tl.score_gene_sets(expression, signature, n_bins=5, random_state=9)
    pd.testing.assert_frame_equal(result.scores, repeated.scores)


def test_raw_and_layer_sources(expression, signature):
    expected = sccoral.tl.score_gene_sets(expression, signature, n_bins=5).scores
    expression.layers["log"] = expression.X.copy()
    expression.raw = expression.copy()
    expression.X = np.zeros(expression.shape)
    pd.testing.assert_frame_equal(
        sccoral.tl.score_gene_sets(expression, signature, layer="log", n_bins=5).scores, expected
    )
    subset = expression[:, 6:].copy()
    pd.testing.assert_frame_equal(
        sccoral.tl.score_gene_sets(subset, signature, use_raw=True, n_bins=5).scores, expected
    )
    with pytest.raises(ValueError, match="together"):
        sccoral.tl.score_gene_sets(expression, signature, use_raw=True, layer="log")


def test_missing_genes_and_incomplete_tail(expression, signature):
    subset = expression[:, [g for g in expression.var_names if g != "g5"]].copy()
    with pytest.raises(ValueError, match="absent"):
        sccoral.tl.score_gene_sets(subset, signature, missing_genes="raise")
    with pytest.warns(UserWarning, match="absent"), pytest.raises(ValueError, match="only 2"):
        sccoral.tl.score_gene_sets(subset, signature)
    with pytest.warns(UserWarning):
        result = sccoral.tl.score_gene_sets(subset, signature, incomplete="zero", n_bins=5)
    assert not result.coverage.loc[("stim", "negative"), "scored"]
    assert result.coverage.loc[("stim", "negative"), "missing"] == ["g5"]
    sc.tl.score_genes(subset, signature.genes["stim"]["positive"], use_raw=False, n_bins=5)
    np.testing.assert_allclose(result.scores["stim"], subset.obs.score)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="No scoreable tail"):
            sccoral.tl.score_gene_sets(expression[:, 6:], signature, incomplete="zero")


def test_negative_only_signature(expression):
    signature = sccoral.tl.extract_gene_sets(pd.DataFrame({"negative": [-3, -2, -1]}, index=["g0", "g1", "g2"]))
    result = sccoral.tl.score_gene_sets(expression, signature, n_bins=5)
    sc.tl.score_genes(expression, ["g0", "g1", "g2"], use_raw=False, n_bins=5)
    np.testing.assert_allclose(result.scores["negative"], -expression.obs.score)


def test_orientation_reference_alignment_and_transfer(expression, signature):
    scores = pd.DataFrame({"stim": [3.0, 4.0, 1.0, 2.0], "free": [4.0, 3.0, 2.0, 1.0]}, index=list("abcd"))
    covariates = pd.DataFrame({"stim": [0, 0, 1, 1]}, index=list("abcd"))
    signs = sccoral.tl.get_score_orientation(scores, covariates.iloc[::-1])
    assert signs.to_dict() == {"stim": -1, "free": 1}
    unflipped = sccoral.tl.score_gene_sets(expression, signature, n_bins=5)
    flipped = sccoral.tl.score_gene_sets(expression, signature, n_bins=5, orientation=signs[["stim"]])
    np.testing.assert_allclose(flipped.scores, -unflipped.scores)
    assert flipped.orientation["stim"] == -1
    with pytest.raises(ValueError, match="Unknown orientation"):
        sccoral.tl.score_gene_sets(expression, signature, orientation=signs)


@pytest.mark.parametrize("groups", [[0, 0, 0, 0], [0, 1, np.nan, 1], [0, 1, 2, 1]])
def test_invalid_orientation_groups(groups):
    with pytest.raises(ValueError, match="both 0 and 1"):
        sccoral.tl.get_score_orientation(pd.DataFrame({"a": [1, 2, 3, 4]}), pd.DataFrame({"a": groups}))


def test_orientation_tie_and_misaligned_cells():
    scores = pd.DataFrame({"a": [1, 1]}, index=["c1", "c2"])
    with pytest.raises(ValueError, match="Equal group means"):
        sccoral.tl.get_score_orientation(scores, pd.DataFrame({"a": [0, 1]}, index=scores.index))
    with pytest.raises(ValueError, match="indices must match"):
        sccoral.tl.get_score_orientation(scores, pd.DataFrame({"a": [0, 1]}, index=["x", "y"]))
