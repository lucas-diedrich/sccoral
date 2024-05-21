#!/usr/bin/env python
import os
from urllib.request import urlretrieve

import anndata as ad
import numpy as np
from scvi.data import synthetic_iid


def synthetic_data(n_cat=2, n_con=10, save: str | None = None, seed: int = 42) -> ad.AnnData:
    """Generate synthetic data based on the scvi-tools synthetic data implementation

    Parameters
    ----------
    n_cat
        Number of levels in categorical covariate
    save:
        Whether to save adata

    Returns
    -------
    Annotated data matrix.

    References
    ----------
    [1] Gayoso, A. et al. A Python library for probabilistic analysis of single-cell omics data. Nat Biotechnol 40, 163–166 (2022).
    """
    np.random.seed(seed)
    adata = synthetic_iid(batch_size=50, n_genes=300, n_batches=2, n_labels=5)

    adata.obs["categorical_covariate"] = np.random.randint(0, n_cat, size=adata.n_obs)
    adata.obs["continuous_covariate"] = np.linspace(0, 1, adata.n_obs)

    if save is not None:
        adata.write_h5ad(save)

    return adata


def splatter_simulation(save_path: str = "data/", filename: str = "simulation.h5ad") -> ad.AnnData:
    """Load simulated count data

    Parameters
    ----------
    save_path
        Where to save the data
    filename
        Filename

    Returns
    -------
    Annotated data matrix.

    References
    ----------
    .. [1] Zappia, L., Phipson, B. & Oshlack, A. Splatter: simulation of single-cell RNA sequencing data. Genome Biol 18, 174 (2017).
    .. [2] Gerard, D. Data-based RNA-seq simulations by binomial thinning. BMC Bioinformatics 21, 206 (2020).
    """
    url = "https://www.dropbox.com/scl/fi/tf6qks693176jk61e2gdd/simulation.1.simplified.h5ad?rlkey=xe0b03y2baeg92h8bdjfi76f5&st=9ihmva4s&dl=1"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    path_to_file = os.path.join(save_path, filename)

    if not os.path.isfile(path_to_file):
        urlretrieve(url, path_to_file)
    return ad.read_h5ad(path_to_file)


def ifn_kang2018_cd4(save_path: str = "data/", filename: str = "ifn_kang2018.h5ad") -> ad.AnnData:
    """Load CD4+ T cells from Kang et al, 2018

    Parameters
    ----------
    save_path
        Where to save the data
    filename
        Filename

    Returns
    -------
    Annotated data matrix.
        IFN Kang et al, 2018.
        - obs (index: cell_id): sample_id, stimulation_condition (stim/ctrl)
        - var (index: gene_id)
        - varm
            - deg_log2fc: Ground truth differentially expressed genes
            - deg_pvals_adj: Ground truth adjusted pvals (wilcoxon)

    References
    ----------
    Kang et al, 2018

    .. [1] Kang, H. M. et al. Multiplexed droplet single-cell RNA-sequencing using natural genetic variation. Nat Biotechnol 36, 89–94 (2018).
    """
    pass
