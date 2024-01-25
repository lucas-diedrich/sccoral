#!/usr/bin/env python
import os
from urllib.request import urlretrieve

import anndata as ad


def splatter_simulation(save_path: str = "data/", filename: str = "simulation.h5ad") -> ad.AnnData:
    """Load simulated data

    Parameters
    ----------
    save_path
        Where to save the data
    filename
        Filename

    Returns
    -------
    Annotated data matrix.
        Simulated anndata object.
        - obs (index: cell_id): sample_id, categorical_covariate, continuous_covariate
        - var (index: gene_id)
        - osbm
            - factor_usage: Ground truth factor usage
        - varm
            - factor_loading: Ground truth factor loadings

    References
    ----------
    **Splatter**
    .. [1] Zappia, L., Phipson, B. & Oshlack, A. Splatter: simulation of single-cell RNA sequencing data. Genome Biol 18, 174 (2017).

    **Seqgendiff**
    .. [2] Gerard, D. Data-based RNA-seq simulations by binomial thinning. BMC Bioinformatics 21, 206 (2020).
    """
    url = "https://figshare.com/ndownloader/files/44184947?private_link=199140ec1dc329efcfbd"

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
