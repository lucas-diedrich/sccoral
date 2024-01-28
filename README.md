# scCoral

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]
![codecov][badge-codecov]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/lucas-diedrich/scCoral/test.yaml?branch=main
[link-tests]: https://github.com/lucas-diedrich/sccoral/actions/workflows/test.yaml/badge.svg
[badge-docs]: https://readthedocs.org/projects/sccoral/badge/?version=latest&style=default
[badge-codecov]: https://codecov.io/gh/lucas-diedrich/sccoral/graph/badge.svg?token=IXBPQ5SSXL

## Getting started

### Motivation

Increasing throughput in single-cell technologies enables researchers to create population-scale single-cell RNAseq datasets. An ongoing challenge in the analysis of this data is to link molecular features (i.e. single-cell gene expression) with patient-level covariates (e.g. disease severity, age, sex, ...). `sc-coral` aims to find an interpretable link between subject/sample-level features and gene expression by embedding cellular metadata and gene expression in the same latent space of a variational autoencoder architecture {cite:p}`lopez2019`. By leveraging and improving the network architecture of linear scVI {cite:p}`svensson2020`, we aim to find a direct and interpretable link between embedded covariates and gene expression.

<!-- Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api]. -->

### Installation

This repository is still under active development. If you want to install the package, please install it directly from `GitHub`.

<!--
You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge). -->

<!--
1) Install the latest release of `scCoral` from `PyPI <https://pypi.org/project/scCoral/>`_:

```bash
pip install scCoral
```
-->

Create a suitable conda environment with Python 3.9 or newer:

```{bash}
conda create -n scvi-env python=3.11
conda activate scvi-env
```

Install the latest development version:

```bash
pip install git+https://github.com/lucas-diedrich/sccoral.git@main
```

### Usage

```{python}
import sccoral

# Load data
adata = sccoral.data.splatter_simulation()

# Setup + train model with scvi-tools syntax
sccoral.model.setup_anndata(adata,
                            categorical_covariates='categorical_covariate',
                            continuous_covariates='continuous_covariates'
                            )
model = sccoral.model.SCCORAL(adata, n_latent=7)
model.train()

# Get latent representation of cells/factor usages
z = model.get_latent_representation()

# Get interpretable gene programs (factor loadings)
loadings = model.get_loadings()
```

## Release notes

This repository is still under active development. See the [changelog][changelog].

## Contact

For feedback, questions and help requests, you can reach out via the issue tracker. Feel free to contact us via [ldiedric@broadinstitute.org](mailto:ldiedric@broadinstitute.org)

<!-- in the [scverse discourse][scverse-discourse]. -->

If you found a bug, please also use the [issue tracker][issue-tracker].

## References

**Interpretable factor models of single-cell RNA-seq via variational autoencoders**
[^svensson2020] Valentine Svensson, Adam Gayoso, Nir Yosef, Lior Pachter, Bioinformatics, Volume 36, Issue 11, June 2020, Pages 3418–3421, https://doi.org/10.1093/bioinformatics/btaa169

**A Python library for probabilistic analysis of single-cell omics data**
[^lopez2019] Adam Gayoso, Romain Lopez, Galen Xing, Pierre Boyeau, Valeh Valiollah Pour Amiri, Justin Hong, Katherine Wu, Michael Jayasuriya, Edouard Mehlman, Maxime Langevin, Yining Liu, Jules Samaran, Gabriel Misrachi, Achille Nazaret, Oscar Clivio, Chenling Xu, Tal Ashuach, Mariano Gabitto, Mohammad Lotfollahi, Valentine Svensson, Eduardo da Veiga Beltrame, Vitalii Kleshchevnikov, Carlos Talavera-López, Lior Pachter, Fabian J. Theis, Aaron Streets, Michael I. Jordan, Jeffrey Regier & Nir Yosef
Nature Biotechnology 2022 Feb 07. https://doi.org/10.1038/s41587-021-01206-w.

**The scverse project provides a computational ecosystem for single-cell omics data analysis**
[^virshup2022] Isaac Virshup, Danila Bredikhin, Lukas Heumos, Giovanni Palla, Gregor Sturm, Adam Gayoso, Ilia Kats, Mikaela Koutrouli, Scverse Community, Bonnie Berger, Dana Pe’er, Aviv Regev, Sarah A. Teichmann, Francesca Finotello, F. Alexander Wolf, Nir Yosef, Oliver Stegle & Fabian J. Theis
Nat Biotechnol. 2022 Apr 10. https://doi.org/10.1038/s41587-023-01733-8.

## Citation

> None

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/lucas-diedrich/scCoral/issues
[changelog]: https://sccoral.readthedocs.io/en/latest/changelog.html
[link-docs]: https://sccoral.readthedocs.io
[link-api]: https://sccoral.readthedocs.io/en/latest/api.html
