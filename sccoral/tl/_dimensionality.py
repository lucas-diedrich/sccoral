from typing import Any

import anndata as ad
import numpy as np
from kneed import KneeLocator


def select_dimensionality(
    adata: ad.AnnData,
    *,
    log: bool = False,
    pca_key: str = "pca",
    kneed_kws: dict[str, Any] | None = None,
) -> int:
    """Estimate a component count from the elbow of PCA variance ratios.

    Parameters
    ----------
    adata
        AnnData with PCA variance ratios already computed, for example with
        ``scanpy.pp.pca``. Neither PCA nor preprocessing is performed here.
    log
        Detect the elbow after applying log10 to the variance ratios. This
        requires strictly positive ratios and can change the selected count.
    pca_key
        Key in ``adata.uns`` containing ``"variance_ratio"``.
    kneed_kws
        Additional ``kneed.KneeLocator`` options, such as ``{"S": 2.0}``.
        ``x``, ``y``, ``curve``, and ``direction`` are reserved. The curve is
        always convex and decreasing. The supplied dictionary is not modified.

    Returns
    -------
    int
        Number of components through the detected elbow, counting from one.

    Raises
    ------
    ValueError
        If PCA variance ratios are missing, invalid, constant, contain fewer
        than three components, or have no detected elbow; or if reserved
        options are supplied.

    Notes
    -----
    This is a PCA-elbow heuristic for an initial choice of dimensionality,
    not an estimate of the optimal scCoral latent dimension. It does not
    evaluate model fit or account for model covariates. Results depend on
    preprocessing and the number of PCs computed. AnnData is not modified.

    Examples
    --------
    >>> import scanpy as sc
    >>> import sccoral
    >>> sc.pp.pca(adata)
    >>> n_latent = sccoral.tl.select_dimensionality(adata)
    """
    options = dict(kneed_kws or {})
    reserved = {"x", "y", "curve", "direction"}.intersection(options)
    if reserved:
        raise ValueError(f"Reserved KneeLocator options cannot be supplied: {', '.join(sorted(reserved))}.")

    try:
        raw_values = adata.uns[pca_key]["variance_ratio"]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError(
            f"Missing PCA variance ratios at adata.uns[{pca_key!r}]['variance_ratio']. Run PCA first."
        ) from exc
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("PCA variance ratios must be numeric.") from exc
    if values.ndim != 1 or values.size < 3:
        raise ValueError("PCA variance ratios must be a 1-D array with at least three components.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("PCA variance ratios must be finite and nonnegative.")
    if np.any(np.diff(values) > 0):
        raise ValueError("PCA variance ratios must be in descending (non-increasing) order.")
    if values[0] == values[-1]:
        raise ValueError("PCA variance ratios are constant; no elbow can be detected.")
    if log and np.any(values <= 0):
        raise ValueError("PCA variance ratios must be strictly positive when log=True.")

    locator = KneeLocator(
        np.arange(1, values.size + 1),
        np.log10(values) if log else values,
        curve="convex",
        direction="decreasing",
        **options,
    )
    if locator.knee is None:
        raise ValueError(
            "No PCA elbow detected. Inspect the variance-ratio curve and choose dimensionality explicitly."
        )
    return int(locator.knee)
