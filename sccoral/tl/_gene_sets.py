"""Extract and score signed gene signatures without modifying fitted models."""

import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from kneed import KneeLocator


@dataclass
class FactorGeneSets:
    """Signed gene lists and a per-factor, per-tail selection table.

    ``genes[factor]`` contains ``positive`` and ``negative`` gene lists.
    ``selection`` records the available genes, detected knee (one-based),
    selected count, and whether the fallback or size limits affected selection.
    """

    genes: dict[Hashable, dict[str, list[str]]]
    selection: pd.DataFrame


@dataclass
class GeneSetScores:
    """Cell-by-factor scores, gene coverage, and applied orientation multipliers.

    ``coverage`` is indexed by factor and tail and records requested, retained,
    and missing genes, plus whether that tail was actually scored. Scores are
    expression-matched gene-set scores, not inferred model latent activities.
    """

    scores: pd.DataFrame
    coverage: pd.DataFrame
    orientation: pd.Series


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def extract_gene_sets(
    loadings: pd.DataFrame,
    *,
    sensitivity: float = 1.0,
    min_genes: int = 10,
    max_genes: int = 300,
    fallback_n: int = 50,
) -> FactorGeneSets:
    """Extract positive and negative gene sets from factor loading elbows.

    Parameters
    ----------
    loadings
        Finite numeric gene-by-factor DataFrame, e.g. ``model.get_loadings()``.
        Gene names must be unique strings and factor labels must be unique.
    sensitivity
        Positive Kneedle sensitivity (``S``).
    min_genes, max_genes
        Lower and upper limits on each nonempty selected tail, capped by the
        number of available genes. Defaults reproduce the analysis helper.
    fallback_n
        Number selected before size limits when no knee is detected, the tail
        has fewer than five genes, or all tail magnitudes are identical.

    Returns
    -------
    FactorGeneSets
        Gene lists and selection metadata. Each tail is sorted independently
        by decreasing absolute loading; ties preserve input order. The knee
        gene is included. Zero loadings are excluded; an absent tail is empty.

    Notes
    -----
    Kneedle uses a convex, decreasing curve with ``online=False``. A fallback
    is a rank cutoff, not a detected elbow; inspect ``selection`` to distinguish
    them. No fitting, factor assignment, or sign orientation is performed.
    """
    for name, value in (("min_genes", min_genes), ("max_genes", max_genes), ("fallback_n", fallback_n)):
        _positive_integer(value, name)
    if min_genes > max_genes:
        raise ValueError("min_genes must not exceed max_genes.")
    if not np.isfinite(sensitivity) or sensitivity <= 0:
        raise ValueError("sensitivity must be finite and positive.")
    if not isinstance(loadings, pd.DataFrame) or loadings.empty:
        raise ValueError("loadings must be a nonempty gene-by-factor DataFrame.")
    if not loadings.index.is_unique or not loadings.columns.is_unique:
        raise ValueError("Gene names and factor labels must be unique.")
    if not all(isinstance(g, str) for g in loadings.index):
        raise ValueError("Gene names must be strings.")
    if np.iscomplexobj(loadings.to_numpy()):
        raise ValueError("Loadings must be real and finite.")
    values = loadings.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Loadings must be real and finite; remove missing values explicitly.")
    genes, rows = {}, []
    for j, factor in enumerate(loadings.columns):
        v = pd.Series(values[:, j], index=loadings.index)
        if not (v != 0).any():
            raise ValueError(f"Factor {factor!r} has only zero loadings.")
        genes[factor] = {}
        for tail, magnitudes in (("positive", v[v > 0]), ("negative", -v[v < 0])):
            ranked = magnitudes.sort_values(ascending=False, kind="stable")
            n = len(ranked)
            knee = None
            if n >= 5 and ranked.iloc[0] != ranked.iloc[-1]:
                knee = KneeLocator(
                    np.arange(1, n + 1),
                    ranked.to_numpy(),
                    S=sensitivity,
                    curve="convex",
                    direction="decreasing",
                    online=False,
                ).knee
            knee = None if knee is None else int(knee)
            proposed = knee if knee is not None else min(fallback_n, n)
            count = min(max(proposed, min_genes), max_genes, n) if n else 0
            genes[factor][tail] = ranked.index[:count].tolist()
            rows.append(
                {
                    "factor": factor,
                    "tail": tail,
                    "available": n,
                    "knee": knee,
                    "selected": count,
                    "fallback_used": bool(n and knee is None),
                    "size_limited": count != proposed,
                }
            )
    selection = pd.DataFrame(rows).set_index(["factor", "tail"])
    selection["knee"] = selection["knee"].astype("Int64")
    return FactorGeneSets(genes, selection)


def _orientation_multipliers(columns, orientation):
    result = pd.Series(1, index=columns, dtype=int, name="orientation")
    if orientation is not None:
        if isinstance(orientation, pd.Series) and not orientation.index.is_unique:
            raise ValueError("Orientation labels must be unique.")
        for factor, sign in orientation.items():
            if factor not in result.index:
                raise ValueError(f"Unknown orientation factor: {factor!r}.")
            if sign not in (-1, 1):
                raise ValueError("Orientation multipliers must be +1 or -1.")
            result.loc[factor] = sign
    return result


def score_gene_sets(
    adata: ad.AnnData,
    gene_sets: FactorGeneSets,
    *,
    use_raw: bool = False,
    layer: str | None = None,
    min_genes: int = 3,
    missing_genes: str = "warn",
    incomplete: str = "raise",
    orientation: Mapping | pd.Series | None = None,
    ctrl_size: int = 50,
    n_bins: int = 25,
    gene_pool: Sequence[str] | None = None,
    random_state: int = 0,
) -> GeneSetScores:
    """Score each signed signature as positive minus negative Scanpy scores.

    Parameters
    ----------
    adata
        Target cells with log-normalized expression and matching gene names.
        This function does not modify AnnData or normalize its expression.
    gene_sets
        Output of :func:`extract_gene_sets`.
    use_raw, layer
        Expression source: ``X`` by default, ``raw.X`` if ``use_raw=True``,
        or the named layer. Raw and layer cannot be selected together.
    min_genes
        Minimum retained genes required to score each originally nonempty tail.
    missing_genes
        ``"warn"`` drops absent genes with a warning; ``"raise"`` rejects them.
        Missing gene names are always recorded in the coverage table.
    incomplete
        ``"raise"`` rejects a nonempty tail with too few retained genes.
        ``"zero"`` explicitly permits omitting it, with a warning. A tail empty
        at extraction contributes zero; negative-only signatures are supported.
        At least one tail per factor must remain scoreable.
    orientation
        Optional mapping from factor labels to +1/-1, learned on reference
        scores with :func:`get_score_orientation`. Unspecified factors keep +1.
        The same mapping should be reused across target datasets.
    ctrl_size, n_bins, gene_pool, random_state
        Passed to ``scanpy.tl.score_genes`` for each tail. Set the random seed
        and expression source consistently for reproducible comparisons.

    Returns
    -------
    GeneSetScores
        Scores aligned to ``adata.obs_names``, tail coverage, and orientation.
        These scores summarize signatures; their absolute scales need not be
        comparable across datasets with different preprocessing or gene pools.
    """
    _positive_integer(min_genes, "min_genes")
    if missing_genes not in ("warn", "raise") or incomplete not in ("raise", "zero"):
        raise ValueError("Use missing_genes='warn'/'raise' and incomplete='raise'/'zero'.")
    if not isinstance(gene_sets, FactorGeneSets) or not gene_sets.genes:
        raise ValueError("gene_sets must be a nonempty FactorGeneSets object.")
    if use_raw and layer is not None:
        raise ValueError("use_raw and layer cannot be selected together.")
    if use_raw:
        if adata.raw is None:
            raise ValueError("use_raw=True requires adata.raw.")
        expression, names = adata.raw.X, adata.raw.var_names
    else:
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Layer {layer!r} is missing.")
        expression = adata.X if layer is None else adata.layers[layer]
        names = adata.var_names
    if expression is None or not names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Expression must be present and cell and gene names must be unique.")
    multipliers = _orientation_multipliers(pd.Index(list(gene_sets.genes)), orientation)
    # Share expression read-only with a small scratch AnnData. Scanpy only writes
    # scores into its obs; no full matrix copy or temporary columns on adata.
    scratch = ad.AnnData(
        X=expression, obs=pd.DataFrame(index=adata.obs_names.copy()), var=pd.DataFrame(index=names.copy())
    )
    rows, retained = [], {}
    for factor, tails in gene_sets.genes.items():
        if set(tails) != {"positive", "negative"}:
            raise ValueError("Each factor must contain positive and negative gene lists.")
        if set(tails["positive"]) & set(tails["negative"]):
            raise ValueError(f"Factor {factor!r} has overlapping positive and negative genes.")
        for tail, requested in tails.items():
            if len(set(requested)) != len(requested):
                raise ValueError(f"Duplicate genes in {factor!r}/{tail}.")
            present = [g for g in requested if g in names]
            absent = [g for g in requested if g not in names]
            if absent:
                message = f"{factor!r}/{tail}: {len(absent)} genes absent from the selected expression source."
                if missing_genes == "raise":
                    raise ValueError(message)
                warnings.warn(message, UserWarning, stacklevel=2)
            scored = len(present) >= min_genes
            if requested and not scored:
                message = f"{factor!r}/{tail}: only {len(present)} genes retained; require {min_genes}."
                if incomplete == "raise":
                    raise ValueError(message)
                warnings.warn(message + " This tail contributes zero.", UserWarning, stacklevel=2)
            retained[factor, tail] = present if scored else []
            rows.append(
                {
                    "factor": factor,
                    "tail": tail,
                    "requested": len(requested),
                    "retained": len(present),
                    "missing": absent,
                    "scored": scored,
                }
            )
        if not any(retained[factor, tail] for tail in ("positive", "negative")):
            raise ValueError(f"No scoreable tail remains for factor {factor!r}.")
    scores = pd.DataFrame(index=adata.obs_names.copy())
    for factor in gene_sets.genes:
        total = np.zeros(adata.n_obs)
        for tail, sign in (("positive", 1), ("negative", -1)):
            selected = retained[factor, tail]
            if selected:
                sc.tl.score_genes(
                    scratch,
                    selected,
                    score_name="_score",
                    use_raw=False,
                    ctrl_size=ctrl_size,
                    n_bins=n_bins,
                    gene_pool=gene_pool,
                    random_state=random_state,
                )
                total += sign * scratch.obs["_score"].to_numpy()
        scores[factor] = total * multipliers.loc[factor]
    return GeneSetScores(scores, pd.DataFrame(rows).set_index(["factor", "tail"]), multipliers)


def get_score_orientation(reference_scores: pd.DataFrame, reference_covariates: pd.DataFrame) -> pd.Series:
    """Learn score signs from binary covariates in a reference dataset.

    Parameters
    ----------
    reference_scores
        Finite cell-by-factor gene-set scores, e.g. ``score_gene_sets(...).scores``.
    reference_covariates
        Binary 0/1 covariates with columns named for the factors to orient.
        Cell indices must match the scores (order may differ). Only included
        columns are oriented; all other factors, including free factors, keep +1.

    Returns
    -------
    pandas.Series
        +1/-1 multipliers for all factors. A sign of -1 means reference mean
        activity was lower in group 1 than group 0. Ties, missing values, and
        absent groups raise ValueError. Reuse the mapping across datasets.

    Notes
    -----
    This orients derived scores for reporting. It does not negate model latents
    or decoder weights, and does not assert a sign symmetry of the fitted model.
    """
    if not reference_scores.index.is_unique or not reference_covariates.index.is_unique:
        raise ValueError("Reference cell indices must be unique.")
    if not reference_scores.columns.is_unique or not reference_covariates.columns.is_unique:
        raise ValueError("Reference columns must be unique.")
    if (
        len(reference_scores) != len(reference_covariates)
        or not reference_scores.index.isin(reference_covariates.index).all()
    ):
        raise ValueError("Reference score and covariate cell indices must match.")
    if not np.isfinite(reference_scores.to_numpy(dtype=float)).all():
        raise ValueError("Reference scores must be finite.")
    covariates = reference_covariates.reindex(reference_scores.index)
    result = _orientation_multipliers(reference_scores.columns, None)
    for factor in covariates:
        if factor not in reference_scores:
            raise ValueError(f"No scores for covariate-associated factor {factor!r}.")
        groups = covariates[factor]
        if groups.isna().any() or set(groups.unique()) != {0, 1}:
            raise ValueError(f"Covariate {factor!r} must contain both 0 and 1, without missing values.")
        score = reference_scores[factor]
        difference = score[groups == 1].mean() - score[groups == 0].mean()
        if difference == 0:
            raise ValueError(f"Equal group means for {factor!r}; orientation is undefined.")
        result.loc[factor] = 1 if difference > 0 else -1
    return result
