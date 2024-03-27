import logging
from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS

# Changes after scvi 1.0.4
try:
    from scvi.autotune import Tunable
except ImportError:
    from scvi._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, one_hot
from torch import logsumexp
from torch.distributions import Normal
from torch.distributions import kl_divergence as kld

from sccoral.nn import LinearDecoder, LinearEncoder

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)


class MODULE(BaseModuleClass):
    """scCoral model

    The model takes the base structure from :cite:p:`Svensson20`, with three important
    modifications

        1. Addition of l1-regularization to linear decoder, tunable with `alpha_l1` parameter
        2. Direct encoding of covariates in latent space. The original instead incorporates them into the neural network
        3. Modified training procedure to account for covariate information.

    Overall, this model aims to find interpretable links between macroscopic covariates and
    molecular gene expression by embedding the covariates directly in the latent space.

    **Limitations** The same limitations as for LSCVI apply.

    Parameters
    ----------
    n_input
        Number of input genes
    categorical_mapping
        Dictionary mapping category names and number of categories per category
    continuous_names
        Iterable listing the continuous category names
    alpha_l1
        Regularization parameter
    n_batch
        Number of batches
    # n_labels NOT IMPLEMENTED
    n_hidden
        Number of nodes per hidden layer on encoder site
    n_latent
        Number of latent dimensions
    n_layers
        Number of layers on encoder site
    latent_distribution
        Whether the latent distribution is normal (scVI) or lognormal (as suggested by LSCVI).
        As the original authors found that the log(data+1) latent distribution is less powerful,
        we use `normal` per default.
    dispersion
        Fit dispersion parameters on a per-gene, per-gene/individual batch, per-gene/individual cell
        basis
    log_variational
        Logarithmized variance for increased stability
    use_batch_norm
        Whether to use batch norm in encoder
    use_layer_norm
        Whether to use layer norm in encoder.
    library_log_means, library_log_vars
        Library parameters
    **vae_kwargs
        Keyword arguments passed to Encoder
    """

    def __init__(
        self,
        n_input: int,
        categorical_mapping: None | dict[str, int],
        continuous_names: None | Iterable,
        alpha_l1: Tunable[float] = 0,
        n_batch: int = 0,
        # n_labels: int = 0,  # TODO gene-labels not implemented
        n_hidden: Tunable[int] = 128,
        n_latent: int = 10,
        n_layers: Tunable[int] = 1,
        dropout_rate: Tunable[float] = 0.1,
        gene_likelihood: Tunable[Literal["nb", "zinb", "poisson"]] = "nb",  # as LSCVI
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",  # as LSCVI
        dispersion: Tunable[Literal["gene", "gene-batch", "gene-cell"]] = "gene",  # TODO gene-labels not implemented
        log_variational: bool = True,  # as LSCVI
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "none"]] = "none",
        use_observed_lib_size: Tunable[bool] = True,  # TODO LSCVI overwrites this flag and uses False
        library_log_means: None | np.ndarray = None,
        library_log_vars: None | np.ndarray = None,
        **vae_kwargs,
    ):
        super().__init__()

        self.n_input = n_input
        self.n_latent = n_latent

        n_cat = len(categorical_mapping) if categorical_mapping is not None else 0
        n_con = len(continuous_names) if continuous_names is not None else 0

        self.categorical_names = categorical_mapping.keys() if categorical_mapping is not None else None
        self.continuous_names = continuous_names

        self.categorical_mapping = categorical_mapping

        self.alpha_l1 = alpha_l1

        self.n_batch = n_batch
        # self.n_labels = n_labels # TODO gene-labels not implemented
        self.latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution

        self.dispersion = dispersion

        # self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_observed_lib_size

        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, " "must provide library_log_means and library_log_vars."
                )
            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        # elif self.dispersion == "gene-label": # TODO gene-label not implemented
        #     self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        self.use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        self.use_layer_norm_encoder = use_layer_norm == "encoder"

        self.use_batch_norm = use_batch_norm
        use_layer_norm = use_layer_norm

        # SETUP Neural nets
        # Setup latent space as follows:
        # dim 0...n_latent-1: free factors
        # dim n_latnet...n_latent+n_cat-1: categorical factors
        # dim n_latent_n_cat..n_latent+n_cat_n_con: continuous factors

        # Counts
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=False,
            **vae_kwargs,
        )

        # Library
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=True,
        )

        # NOVELTY
        # Categorical embedding
        categorical_encoder_collection = {}
        if categorical_mapping is not None:
            for (cat_name, n_levels), dim in zip(categorical_mapping.items(), range(n_latent, n_latent + n_cat)):
                # Binary covariates are encoded as 1-dimensional OHE
                if n_levels == 2:
                    n_levels = 1

                name = f"encoder_{cat_name}"
                model = LinearEncoder(n_levels, 1, distribution=latent_distribution, mean_bias=True, var_bias=True)

                # Register encoder in class
                setattr(self, name, model)
                # Store position dictionary
                categorical_encoder_collection[cat_name] = {"covariate": cat_name, "name": name, "dim": dim}

        self.categorical_encoder_collection = categorical_encoder_collection

        # NOVELTY
        # Continous embedding
        continous_encoder_collection = {}
        if continuous_names is not None:
            for con_name, dim in zip(continuous_names, range(n_latent + n_cat, n_latent + n_cat + n_con)):
                name = f"encoder_{con_name}"
                model = LinearEncoder(1, 1, latent_distribution=latent_distribution)

                # Register encoder in class
                setattr(self, name, model)
                # Store
                continous_encoder_collection[con_name] = {"covariate": con_name, "name": name, "dim": dim}
        self.continous_encoder_collection = continous_encoder_collection

        # Linear Decoder (as in LSCVI)
        self.decoder = LinearDecoder(
            n_input=n_latent + n_cat + n_con,
            n_output=n_input,
            n_cat_list=None,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=False,
            bias=False,
        )

    def _get_inference_input(self, tensors):
        """Get counts, batch indices, categorical/continuous covariates

        Categorical covariates are split into a dict of One-hot-encodings.
        Continuous covariates are split into a dict of continuous vectors
        """
        x = tensors[REGISTRY_KEYS.X_KEY]

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        # Categorical covariates to individual, one-hot-encoded
        # tensors
        # For binary covariates, encode as 1-dim OHE, for
        # N>2 categorical covariates as N-dim OHE
        categorical_key = REGISTRY_KEYS.CAT_COVS_KEY
        categorical_covariates_ohe = None
        if categorical_key in tensors.keys():
            categorical_covariates_ohe = {}
            categorical_covariates = torch.split(tensors[categorical_key], split_size_or_sections=1, dim=1)
            for xi, (cat_name, n_level) in zip(categorical_covariates, self.categorical_mapping.items()):
                # TODO
                if n_level == 2:
                    categorical_covariates_ohe[cat_name] = xi.to(dtype=torch.float32, device=self.device)
                else:
                    categorical_covariates_ohe[cat_name] = one_hot(index=xi, n_cat=n_level).to(
                        dtype=torch.float32, device=self.device
                    )

        # Continuous covariates to individual tensors
        # dict[covariate_name: tensor[continuous_covariate]]
        continous_key = REGISTRY_KEYS.CONT_COVS_KEY
        continuous_covs_split = None
        if continous_key in tensors.keys():
            continuous_covariates = torch.split(tensors[continous_key], split_size_or_sections=1, dim=1)
            continuous_covs_split = {
                k: v.to(device=self.device) for k, v in zip(self.continuous_names, continuous_covariates)
            }

        input_dict = {
            "x": x,
            "batch_index": batch_index,
            "continuous_covariates": continuous_covs_split,
            "categorical_covariates": categorical_covariates_ohe,
        }
        return input_dict

    def inference(self, x, batch_index, continuous_covariates, categorical_covariates, n_samples=1):
        x_ = x

        if self.log_variational:
            x_ = torch.log(1 + x)

        if self.use_observed_lib_size:
            ql = None
            library = torch.log(x.sum(1)).unsqueeze(1)
        else:
            ql, library = self.l_encoder(x_, batch_index)

        # CATEGORICAL COVARIATES (NOVELTY)
        if categorical_covariates is not None:
            mean_ca, var_ca, latent_ca = [], [], []
            for category_name, ohe in categorical_covariates.items():
                encoder = getattr(self, f"encoder_{category_name}")
                mean_ca_i, var_ca_i, latent_ca_i = encoder(ohe.to(device=self.device))

                mean_ca.append(mean_ca_i)
                var_ca.append(var_ca_i)
                latent_ca.append(latent_ca_i)
        else:
            mean_ca, var_ca, latent_ca = (
                [torch.tensor([], device=x.device)],
                [torch.tensor([], device=x.device)],
                [torch.tensor([], device=x.device)],
            )

        # CONTINUOUS COVARIATES (NOVELTY)
        if continuous_covariates is not None:
            mean_cc, var_cc, latent_cc = [], [], []
            for con_name, vector in continuous_covariates.items():
                encoder = getattr(self, f"encoder_{con_name}")
                mean_cc_i, var_cc_i, latent_cc_i = encoder(vector)

                mean_cc.append(mean_cc_i)
                var_cc.append(var_cc_i)
                latent_cc.append(latent_cc_i)
        else:
            mean_cc, var_cc, latent_cc = (
                torch.tensor([], device=x.device),
                torch.tensor([], device=x.device),
                torch.tensor([], device=x.device),
            )

        # count encoding
        (mean_counts, var_counts, latent_counts) = self.z_encoder(x_, batch_index)

        # concatenate latent representations of counts,
        # categorical covariates and continuous covariates
        # to latent representation of the cell
        mean_z = torch.cat([mean_counts, *mean_ca, *mean_cc], dim=1)
        var_z = torch.cat([var_counts, *var_ca, *var_cc], dim=1)
        z = torch.cat([latent_counts, *latent_ca, *latent_cc], dim=1)

        qz = Normal(loc=mean_z, scale=torch.sqrt(var_z))

        if n_samples > 1:
            # Sample n samples from normal distribution
            # if lognormal, transform z
            z_untransformed = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(z_untransformed)

            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))

        outputs = {"z": z, "qz": qz, "ql": ql, "library": library}

        return outputs

    def _get_generative_input(self, tensors, inference_outputs):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        z = inference_outputs["z"]
        library = inference_outputs["library"]

        input_dict = {"z": z, "library": library, "batch_index": batch_index}

        return input_dict

    def _compute_local_library_params(self, batch_index):
        """Computes local library parameters. (Only used when used when use_observed_lib_size == False)

        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(one_hot(batch_index, n_batch), self.library_log_means)
        local_library_log_vars = F.linear(one_hot(batch_index, n_batch), self.library_log_vars)
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        # transform_batch=None NOT IMPLEMENTED IN LSCVI
    ):
        decoder_input = z

        # Linear decoder (Linear layer)
        # px_scale: normalized gene expression (relative to library size)
        # px_r: Inverse dispersion of negative binomial
        # px_rate = torch.exp(library)*scale Unnormalized gene expression
        # px_dropout: For ZINB model, dropout rate/rate of zero inflation,
        # not recommended

        # TODO Double check, add batch_id as covariate
        # TODO library to size factor
        px_scale, px_r, px_rate, px_dropout = self.decoder(dispersion="", z=decoder_input, library=library)

        # TODO gene-label not implemented
        # gene-cell: do nothing
        # if self.dispersion == 'gene-label':
        #     px_r = F.linear(
        #         one_hot(y, self.n_labels), self.px_r
        #     )
        if self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        # Counts prediction
        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout, scale=px_scale)
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        else:
            raise ValueError(f"gene_likelihood {self.gene_likelihood} not defined")

        # Library size predictions
        if self.use_observed_lib_size:
            pl = None
        else:
            (local_library_log_means, local_library_log_vars) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())

        # Prior distribution
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        generative_outputs = {"px": px, "pz": pz, "pl": pl}
        return generative_outputs

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Loss

        Loss is the same as in LSCVI except for additional regularization
        """
        x = tensors[REGISTRY_KEYS.X_KEY]

        # KL prior/latent distribution
        kl_divergence_z = kld(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=-1)

        if not self.use_observed_lib_size:
            kl_divergence_l = kld(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        weighted_kl_local = kl_weight * kl_divergence_z + kl_divergence_l

        # Reconstruction loss
        # Sum over log likelihood of data
        reconstruction_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        # L1 norm
        weights = self.decoder.factor_loading.fc_layers[0][0].weight
        l1_loss = F.l1_loss(weights, torch.zeros_like(weights, device=weights.device))

        loss = torch.mean(reconstruction_loss + weighted_kl_local + self.alpha_l1 * l1_loss)

        kl_local = {"kl_divergence_l": kl_divergence_l, "kl_divergence_z": kl_divergence_z}

        return LossOutput(
            loss=loss, reconstruction_loss=reconstruction_loss, kl_local=kl_local, extra_metrics={"l1_loss": l1_loss}
        )

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples, return_mean=False, n_mc_samples_per_pass=1):
        """Implementation from scvi-tools"""
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = []
        if n_mc_samples_per_pass > n_mc_samples:
            logger.warn(
                "Number of chunks is larger than the total number of samples, setting it to the number of samples"
            )
            n_mc_samples_per_pass = n_mc_samples
        n_passes = int(np.ceil(n_mc_samples / n_mc_samples_per_pass))
        for _ in range(n_passes):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors, inference_kwargs={"n_samples": n_mc_samples_per_pass})
            qz = inference_outputs["qz"]
            ql = inference_outputs["ql"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities
            p_z = Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale)).log_prob(z).sum(dim=-1)
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = Normal(local_library_log_means, local_library_log_vars.sqrt()).log_prob(library).sum(dim=-1)
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum.append(log_prob_sum)
        to_sum = torch.cat(to_sum, dim=0)
        batch_log_lkl = logsumexp(to_sum, dim=0) - np.log(n_mc_samples)
        if return_mean:
            batch_log_lkl = torch.mean(batch_log_lkl).item()
        else:
            batch_log_lkl = batch_log_lkl.cpu()
        return batch_log_lkl

    @torch.inference_mode()
    def get_loadings(self) -> np.ndarray:
        """Implementation from LSCVI"""
        if self.use_batch_norm_decoder:
            w = self.decoder.factor_loading.fc_layers[0][0].weight
            bn = self.decoder.factor_loading.fc_layers[0][1]
            sigma = torch.sqrt(bn.running_var + bn.eps)
            gamma = bn.weight
            b = gamma / sigma
            b_identity = torch.diag(b)
            loadings = torch.matmul(b_identity, w)
        else:
            loadings = self.decoder.factor_loading.fc_layers[0][0].weight
        loadings = loadings.detach().cpu().numpy()
        # TODO double check
        # if self.n_batch > 1:
        # loadings = loadings[:, : -self.n_batch]

        return loadings
