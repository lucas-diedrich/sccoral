# Changelog

All notable changes to this project will be documented in this file.The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

#### December 2023

-   Model/Module/nn - Covariate informed VAE with linear decoder, model based on LSCVI
-   Added l1-regularization term to linear decoder

#### January 2024

-   Added `sccoral.data` for project-specific data
-   Added callbacks for pretraining procecure

#### TODO

**Tests**

-   Add tests for model setup
-   Add tests based on implemented tests for callbacks

**Model**

-   Added custom `TrainingPlan` with pre-training procedure for covariates

**Jupyter Notebooks**

-   Jupyter Notebook - Demonstration on simulated data
-   Jupyter Notebook - Demonstration on IFN stimulation - Kang et al, 2018 data

**Documentation**

-   Add readthedocs website
