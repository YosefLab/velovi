# -*- coding: utf-8 -*-
"""Main module."""
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers
from torch import nn as nn
from torch.distributions import Categorical, Dirichlet, MixtureSameFamily, Normal
from torch.distributions import kl_divergence as kl

from ._constants import REGISTRY_KEYS

torch.backends.cudnn.benchmark = True


class DecoderVELOVI(nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_ouput = n_output
        self.rho_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
            # activation_fn=torch.nn.ELU,
        )

        self.tau_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
            # activatnion_fn=torch.nn.ELU,
        )

        self.pi_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
            # activation_fn=torch.nn.ELU,
        )

        # categorical pi
        # 4 states
        self.px_pi_decoder = nn.Linear(n_hidden, 4 * n_output)

        # rho for induction
        self.px_rho_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

        # tau for repression
        self.px_tau_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())

    def forward(self, z: torch.Tensor, *cat_list: int):
        """
        The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        z :
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        rho_first = self.rho_first_decoder(z, *cat_list)
        # tau_first = self.tau_first_decoder(z, *cat_list)

        px_rho = self.px_rho_decoder(rho_first)
        px_tau = self.px_tau_decoder(rho_first)
        # cells by genes by 4
        pi_first = self.pi_first_decoder(z, *cat_list)
        px_pi = nn.Softplus()(
            torch.reshape(self.px_pi_decoder(pi_first), (z.shape[0], self.n_ouput, 4))
        )

        return px_pi, px_rho, px_tau


# VAE model
class VELOVAE(BaseModuleClass):
    """
    Variational auto-encoder model.

    This is an implementation of the scVI model descibed in [Lopez18]_

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    latent_distribution
        One of

        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    use_layer_norm
        Whether to use layer norm in layers
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        true_time_switch: Optional[np.ndarray] = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        log_variational: bool = True,
        latent_distribution: str = "normal",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_observed_lib_size: bool = True,
        var_activation: Optional[Callable] = torch.nn.Softplus(),
        model_steady_states: bool = True,
        log_gamma_init: Optional[np.ndarray] = None,
        log_alpha_init: Optional[np.ndarray] = None,
        switch_spliced: Optional[np.ndarray] = None,
        switch_unspliced: Optional[np.ndarray] = None,
        induction_gene_mask: Optional[np.ndarray] = None,
        t_max: float = 20,
        penalty_scale: float = 0.2,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        self.n_input = n_input
        self.model_steady_states = model_steady_states
        self.t_max = t_max
        self.penalty_scale = penalty_scale

        if induction_gene_mask is not None:
            self.register_buffer(
                "induction_gene_mask",
                torch.from_numpy(induction_gene_mask).type(torch.BoolTensor),
            )

        if switch_spliced is not None:
            self.register_buffer("switch_spliced", torch.from_numpy(switch_spliced))
        else:
            self.switch_spliced = None
        if switch_unspliced is not None:
            self.register_buffer("switch_unspliced", torch.from_numpy(switch_unspliced))
        else:
            self.switch_unspliced = None

        n_genes = n_input * 2

        # switching time
        self.switch_time_unconstr = torch.nn.Parameter(7 + 0.5 * torch.randn(n_input))
        if true_time_switch is not None:
            self.register_buffer("true_time_switch", torch.from_numpy(true_time_switch))
        else:
            self.true_time_switch = None

        # degradation
        if log_gamma_init is None:
            self.gamma_mean_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input))
        else:
            self.gamma_mean_unconstr = torch.nn.Parameter(
                torch.from_numpy(log_gamma_init)
            )

        self.gamma_mean_prior_unconstr = torch.nn.Parameter(-1 * torch.ones(1))
        self.gamma_var_prior_unconstr = torch.nn.Parameter(3 * torch.ones(1))

        # splicing
        # first samples around 1
        self.beta_mean_unconstr = torch.nn.Parameter(0.5 * torch.ones(n_input))

        self.beta_mean_prior_unconstr = torch.nn.Parameter(0 * torch.ones(1))
        self.beta_var_prior_unconstr = torch.nn.Parameter(3 * torch.ones(1))

        # transcription
        if log_alpha_init is None:
            self.alpha_unconstr = torch.nn.Parameter(0 * torch.ones(n_input))
        else:
            self.alpha_unconstr = torch.nn.Parameter(torch.from_numpy(log_alpha_init))
        # self.alpha_unconstr.requires_grad = False

        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_genes, 4))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_genes
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent
        self.decoder = DecoderVELOVI(
            n_input_decoder,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            activation_fn=torch.nn.ReLU,
        )

    def set_globals(self, train: bool = True):

        if train is not True:
            self.beta_mean_unconstr.requires_grad = False
            self.beta_mean_prior_unconstr.requires_grad = False
            self.beta_var_prior_unconstr.requires_grad = False
            self.alpha_unconstr.requires_grad = False
            self.gamma_mean_unconstr.requires_grad = False
            self.gamma_mean_prior_unconstr.requires_grad = False
            self.gamma_var_prior_unconstr.requires_grad = False
            # self.switch_time_unconstr.requires_grad = False
        else:
            self.beta_mean_unconstr.requires_grad = True
            self.beta_mean_prior_unconstr.requires_grad = True
            self.beta_var_prior_unconstr.requires_grad = True
            self.alpha_unconstr.requires_grad = True
            self.gamma_mean_unconstr.requires_grad = True
            self.gamma_mean_prior_unconstr.requires_grad = True
            self.gamma_var_prior_unconstr.requires_grad = True
            # self.switch_time_unconstr.requires_grad = True

    def _get_inference_input(self, tensors):
        spliced = tensors[REGISTRY_KEYS.S_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        input_dict = dict(
            spliced=spliced,
            unspliced=unspliced,
        )
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]

        input_dict = {
            "z": z,
        }
        return input_dict

    @auto_move_data
    def inference(
        self,
        spliced,
        unspliced,
        n_samples=1,
    ):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        spliced_ = spliced
        unspliced_ = unspliced
        if self.log_variational:
            spliced_ = torch.log(0.01 + spliced)
            unspliced_ = torch.log(0.01 + unspliced)

        encoder_input = torch.cat((spliced_, unspliced_), dim=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

        # globals
        # degradation
        gamma = torch.clamp(F.softplus(self.gamma_mean_unconstr), 0, 50)
        # splicing
        beta = torch.clamp(F.softplus(self.beta_mean_unconstr), 0, 50)
        # transcription
        alpha = torch.clamp(F.softplus(self.alpha_unconstr), 0, 1500)

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, gamma=gamma, beta=beta, alpha=alpha)
        return outputs

    @auto_move_data
    def generative(
        self,
        z,
    ):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha, px_rho, px_tau = self.decoder(
            decoder_input,
        )
        px_pi = Dirichlet(px_pi_alpha).rsample()
        # px_pi = px_pi_alpha / px_pi_alpha.sum(dim=-1, keepdim=True)

        # additional constraint
        # px_tau = 1 - px_rho

        scale_unconstr = self.scale_unconstr
        scale = F.softplus(scale_unconstr)

        return dict(
            px_pi=px_pi,
            px_rho=px_rho,
            px_tau=px_tau,
            scale=scale,
            px_pi_alpha=px_pi_alpha,
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced = tensors[REGISTRY_KEYS.S_KEY]
        unspliced = tensors[REGISTRY_KEYS.U_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]
        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]
        scale = generative_outputs["scale"]
        px_rho = generative_outputs["px_rho"]
        px_tau = generative_outputs["px_tau"]

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)

        reconst_loss_s, reconst_loss_u, end_penalty = self.get_reconstruction_loss(
            spliced,
            unspliced,
            px_pi,
            px_rho,
            px_tau,
            scale,
            gamma,
            beta,
            alpha,
        )
        reconst_loss = reconst_loss_u.sum(dim=-1) + reconst_loss_s.sum(dim=-1)

        if hasattr(self, "induction_gene_mask"):
            concentration = px_pi_alpha[..., ~self.induction_gene_mask, :]
            kl_pi = kl(
                Dirichlet(concentration),
                Dirichlet(0.1 * torch.ones_like(concentration)),
            ).sum(dim=-1)
            kl_pi += kl(
                Dirichlet(px_pi_alpha[..., self.induction_gene_mask, :]),
                Dirichlet(
                    torch.tensor([5.0, 1.0, 0.1, 5.0], device=concentration.device)
                ),
            ).sum(dim=-1)
        else:
            kl_pi = kl(
                Dirichlet(px_pi_alpha), Dirichlet(0.25 * torch.ones_like(px_pi))
            ).sum(dim=-1)
        # kl_pi = 0

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = torch.mean(reconst_loss + weighted_kl_local)

        # combine local and global
        global_ll = (
            -Normal(
                self.gamma_mean_prior_unconstr,
                F.softplus(self.gamma_var_prior_unconstr).sqrt(),
            )
            .log_prob(self.gamma_mean_unconstr)
            .sum()
        )
        global_ll += (
            -Normal(
                self.beta_mean_prior_unconstr,
                F.softplus(self.beta_var_prior_unconstr).sqrt(),
            )
            .log_prob(self.beta_mean_unconstr)
            .sum()
        )

        # global_loss = global_ll
        global_loss = 0
        # global_penalty = ((beta / gamma) - 1).pow(2).sum()
        # global_loss += global_penalty
        loss = (
            local_loss
            + self.penalty_scale * (1 - kl_weight) * end_penalty
            + (1 / n_obs) * kl_weight * (global_loss)
        )

        loss_recoder = LossRecorder(loss, reconst_loss, kl_local, global_loss)

        return loss_recoder

    @auto_move_data
    def get_reconstruction_loss(
        self,
        spliced,
        unspliced,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        beta,
        alpha,
    ) -> torch.Tensor:

        t_s = torch.clamp(F.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # induction
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, beta, gamma, t_s * px_rho
        )
        mean_u_ind_steady = torch.clamp(
            (alpha / beta).expand(n_cells, self.n_input), 0, 1000
        )
        mean_s_ind_steady = torch.clamp(
            (alpha / gamma).expand(n_cells, self.n_input), 0, 1000
        )
        scale_u = scale[: self.n_input, :].expand(n_cells, self.n_input, 4).sqrt()

        # repression
        u_0, s_0 = self._get_induction_unspliced_spliced(alpha, beta, gamma, t_s)
        u_0 = torch.clamp(u_0, 0, 1000)
        s_0 = torch.clamp(s_0, 0, 1000)
        # mean_u_ind_steady = u_0.expand(n_cells, self.n_input)
        # mean_s_ind_steady = s_0.expand(n_cells, self.n_input)

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = torch.zeros_like(mean_u_ind)
        mean_s_rep_steady = torch.zeros_like(mean_u_ind)
        scale_s = scale[self.n_input :, :].expand(n_cells, self.n_input, 4).sqrt()

        # end time penalty
        mean_u_rep_end, mean_s_rep_end = self._get_repression_unspliced_spliced(
            u_0, s_0, beta, gamma, (self.t_max - t_s)
        )
        # make time t=20 close to 0
        # end_penalty = mean_u_rep_end.pow(2).sum() + mean_s_rep_end.pow(2).sum()
        # end_penalty = 0
        # make switch time close to steady state
        end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
            (s_0 - self.switch_spliced).pow(2)
        ).sum()

        # make steady state close to switch time
        # end_penalty = ((mean_u_ind_steady[0] - self.switch_unspliced).pow(2)).sum() + (
        #     (mean_s_ind_steady[0] - self.switch_spliced).pow(2)
        # ).sum()

        # unspliced
        mean_u = torch.stack(
            (
                mean_u_ind,
                mean_u_ind_steady,
                mean_u_rep,
                mean_u_rep_steady,
            ),
            dim=2,
        )
        scale_u = torch.stack(
            (
                scale_u[..., 0],
                scale_u[..., 0],
                scale_u[..., 0],
                0.1 * scale_u[..., 0],
            ),
            dim=2,
        )
        dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)

        # spliced
        mean_s = torch.stack(
            (mean_s_ind, mean_s_ind_steady, mean_s_rep, mean_s_rep_steady),
            dim=2,
        )
        scale_s = torch.stack(
            (
                scale_s[..., 0],
                scale_s[..., 0],
                scale_s[..., 0],
                0.1 * scale_s[..., 0],
            ),
            dim=2,
        )
        dist_s = Normal(torch.clamp(mean_s, 0, 3000), scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)
        reconst_loss_s = -mixture_dist_s.log_prob(spliced)

        return reconst_loss_s, reconst_loss_u, end_penalty

    def _get_induction_unspliced_spliced(self, alpha, beta, gamma, t):
        unspliced = (alpha / beta) * (1 - torch.exp(-beta * t))
        spliced = (alpha / gamma) * (1 - torch.exp(-gamma * t)) + (
            alpha / (gamma - beta)
        ) * (torch.exp(-gamma * t) - torch.exp(-beta * t))
        return unspliced, spliced

    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t):
        unspliced = torch.exp(-beta * t) * u_0
        spliced = s_0 * torch.exp(-gamma * t) - (beta * u_0 / (gamma - beta)) * (
            torch.exp(-gamma * t) - torch.exp(-beta * t)
        )
        return unspliced, spliced

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        raise NotImplementedError
