# Changelog

All notable changes to this project will be documented in this file.The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Removed

- Removed the incorrect marginal-likelihood estimator. The inherited
  `get_marginal_ll()` entry point now raises `NotImplementedError`.
  Training, ELBO evaluation, and reconstruction-error evaluation remain available.

### Fixed

- Delay automatic overall early stopping until joint training begins and warn
  when a run ends without completing a joint-training epoch.
- Forward extra `train()` keyword arguments to the trainer, including
  `early_stopping_patience`. Reject duplicate direct/nested trainer arguments
  and preserve caller-owned trainer settings and callback lists.
- Preserve free-factor softmax and independent covariate sigmoid transformations in
  posterior mean extraction and multi-sample inference. Existing covariate-informed
  logistic-normal embeddings should be re-extracted from saved models, and analyses
  using those embeddings recomputed. Model weights and ordinary training are unchanged.
- Preserve cell identities when extracting a subset or reordered latent representation.

### Added

- Add `sccoral.tl.extract_gene_sets`, `score_gene_sets`, and
  `get_score_orientation` for signed factor signatures, with cutoff metadata,
  gene coverage reporting, and reusable reference-based score orientation.
- Add `sccoral.tl.select_dimensionality` to estimate a component count from
  the PCA variance-ratio elbow, with optional log transformation.
- Record the latest training attempt in `model.training_status_`, retained by
  model save/load. Optional `training_status_path` exports the same record as JSON.

#### March 2024

- Hyperparameter tuning

- Changed default latent distribution from `"normal"` to logit normal (`"ln"`)

#### January 2024

- `sccoral` in `sc-verse` cookiecutter template
- Added `sccoral.data` for project-specific data
- Added callbacks for pretraining procecure

**Tests**

- Added tests for model setup
- Added tests based on implemented tests for callbacks
- Added tests for misc. model features

**Model**

- Added custom `TrainingPlan` with pre-training procedure for covariates

**Documentation**

- Add readthedocs website

#### December 2023

- Model/Module/nn - Covariate informed VAE with linear decoder, model based on LSCVI
- Added l1-regularization term to linear decoder

### [TODO]

#### Project data

- Implement data projection

#### Jupyter Notebooks

- Jupyter Notebook - Demonstration on simulated data
- Jupyter Notebook - Demonstration on IFN stimulation - Kang et al, 2018 data

#### tl

- Find markers

#### Plotting (.pl)

- Implement plotting module

### Refactoring

<!-- #### Tools (.tl)  -->
