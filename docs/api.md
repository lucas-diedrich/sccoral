# API

## Tools

`sccoral.tl.select_dimensionality` estimates a component count from the elbow
of precomputed PCA variance ratios. Run PCA on your chosen preprocessed data
first; the selector does not modify AnnData or perform preprocessing.

```python
import scanpy as sc
import sccoral

sc.pp.pca(adata)
n_latent = sccoral.tl.select_dimensionality(adata)
# After configuring AnnData with SCCORAL.setup_anndata for your model:
model = sccoral.model.SCCORAL(adata, n_latent=n_latent)
```

This is a starting heuristic, not an optimal model dimension: it does not
evaluate model fit or account for covariates. The result depends on
preprocessing and how many PCs were computed. `log=True` uses log10 variance
ratios and requires strictly positive values. Use `pca_key` for a custom PCA
entry in `adata.uns`, and `kneed_kws={"S": 2.0}` to adjust knee sensitivity.
Missing or invalid PCA results and curves without a detected elbow raise
`ValueError`; inspect the scree plot and choose a dimension explicitly when
no elbow is found. The returned integer is a component count starting at one,
not the zero-based index returned by the earlier analysis helper.

```{eval-rst}
.. module:: sccoral.tl
.. currentmodule:: sccoral

.. autosummary::
    :toctree: generated

    tl.select_dimensionality
    tl.extract_gene_sets
    tl.score_gene_sets
    tl.get_score_orientation
    tl.FactorGeneSets
    tl.GeneSetScores
```

### Signed factor gene sets

Convert fitted loadings into reusable positive and negative gene lists:

```python
gene_sets = sccoral.tl.extract_gene_sets(model.get_loadings())
print(gene_sets.genes)       # factor -> positive/negative gene lists
print(gene_sets.selection)   # detected knee, fallback, and final size per tail

# adata_target.X must contain log-normalized expression.
result = sccoral.tl.score_gene_sets(adata_target, gene_sets)
scores = result.scores      # cells x factors; AnnData is not modified
print(result.coverage)      # retained/missing genes and whether each tail scored
```

Each tail is ranked by absolute loading and selected independently using
Kneedle (`S=1`, convex/decreasing, offline). The knee gene is included. By
default, each nonempty set is limited to 10–300 genes, capped by availability.
When a tail has fewer than five genes, has constant magnitudes, or has no
detected knee, selection falls back to 50 genes before applying those limits.
Use `min_genes`, `max_genes`, `fallback_n`, and `sensitivity` to adjust this
behavior. The selection table distinguishes detected knees from fallbacks.
Zero loadings are excluded; nonfinite loadings and all-zero factors are rejected.

Scoring calls `scanpy.tl.score_genes` separately for the two tails and returns
positive minus negative scores. Choose the expression source explicitly with
`layer="log_normalized"` or `use_raw=True` when appropriate; otherwise `X` is
used even if `.raw` exists. Control sampling is reproducible with
`random_state=0` by default; `ctrl_size`, `n_bins`, and `gene_pool` are configurable.
Absent genes produce a warning and are recorded in the coverage table;
`missing_genes="raise"` instead rejects them. Each originally nonempty tail
must retain at least three genes. `incomplete="zero"` explicitly allows a
smaller tail to contribute zero, with a warning. An originally empty tail
contributes zero naturally, including for negative-only signatures. At least
one scoreable tail must remain per factor.

These are derived gene-set activity scores, not the fitted model's latent
activities. Absolute values can depend on preprocessing, gene coverage, and
the control-gene pool, so they are not automatically calibrated across datasets.

To orient covariate-associated signatures, learn a sign mapping **once on the
reference dataset** and reuse it on target datasets:

```python
reference = sccoral.tl.score_gene_sets(adata_reference, gene_sets)
# Columns must match the corresponding factor labels and contain binary 0/1.
covariates = adata_reference.obs[["stimulation"]]
orientation = sccoral.tl.get_score_orientation(reference.scores, covariates)
oriented_reference = reference.scores.mul(orientation, axis="columns")
target = sccoral.tl.score_gene_sets(
    adata_target, gene_sets, orientation=orientation,
)
```

Listed factors are oriented to have higher mean reference scores in group 1
than group 0. Unlisted factors, including free factors, retain their original
orientation. Both groups must be present, and ties or missing values raise an
error. This is a reporting convention for derived scores; it does not change
decoder weights or latent activities and does not imply that negating a
logistic-normal model factor preserves the fitted model.

## Model

Marginal-likelihood estimation is not supported. The inherited
`get_marginal_ll()` method raises `NotImplementedError`. Use `get_elbo()` or
`get_reconstruction_error()` for their respective evaluation metrics; neither
is a marginal-likelihood estimate.

Each `train()` call records its progress in `model.training_status_`, which is
retained by `save()` and `load()`. It includes whether pretraining was enabled
and completed, the zero-based `unfreeze_epoch`, completed pretraining and joint
epoch counts, and a `termination_reason`. With pretraining disabled, the
pretraining requirement is considered complete and `unfreeze_epoch` is `None`.
Partial epochs stopped by a step limit are not counted as completed epochs.

Automatic overall early stopping begins only after the count encoder is unfrozen.
Explicit user callbacks can still request a stop. A warning is emitted if no
joint-training epoch completes; inspect the status before using such a model.
The record describes the latest training attempt, rather than cumulative history
or a guarantee of convergence. Older saved models may not contain this attribute.

```python
model.train(training_status_path="results/training-status.json")
print(model.training_status_)
```

Termination reasons include `max_epochs`, `max_steps`, `early_stopping`,
`stop_requested`, `interrupted`, and `failed` (`completed` is a fallback for other
normal exits). The optional JSON file is also written when fitting raises an
exception. Hard process termination cannot guarantee a final record. When
pretraining is enabled, `max_epochs` includes both training phases, so allow
enough epochs for the joint phase.

```{eval-rst}
.. module:: sccoral.model
.. currentmodule:: sccoral

.. autosummary::
    :toctree: generated

    model.SCCORAL
```

## Module

```{eval-rst}
.. module:: sccoral.module
.. currentmodule:: sccoral

.. autosummary::
    :toctree: generated

    module.MODULE

```

## Training plan

```{eval-rst}
.. module:: sccoral.train
.. currentmodule:: sccoral

.. autosummary::
   :toctree: generated

   train.ScCoralTrainingPlan
```

## Datasets

```{eval-rst}
.. module:: sccoral.data
.. currentmodule:: sccoral

.. autosummary::
    :toctree: generated

    data.splatter_simulation

```
