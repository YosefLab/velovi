from typing import Callable, Iterable, Optional

import numpy as np
from scvi._compat import Literal
from scvi.module.base import JaxBaseModuleClass, LossRecorder
from flax import linen as nn
from numpyro.distributions import (
    Categorical,
    Dirichlet,
    MixtureSameFamily,
    Normal,
    kl_divergence as kl,
)
import jax.numpy as jnp

from ._constants import REGISTRY_KEYS


class _MLP(nn.Module):
    """Fully connected layers with dropout and batchnorm."""

    n_layers: int
    n_hidden: int
    dropout_rate: float
    batch_norm: bool = True
    layer_norm: bool = True
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, x, training: Optional[bool] = None):
        training = nn.merge_param("training", self.training, training)
        is_eval = not training
        for _ in range(self.n_layers):
            x = nn.Dense(features=self.n_hidden)(x)
            if self.batch_norm:
                x = nn.BatchNorm(
                    use_running_average=is_eval, momentum=0.99, epsilon=0.001
                )(x)
            if self.layer_norm:
                x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate)(x)
        return x


class _Encoder(nn.Module):
    """Encoder for the velocity model."""

    n_latent: int
    n_layers: int
    n_hidden: int
    dropout_rate: float
    batch_norm: bool = True
    layer_norm: bool = True
    var_eps: float = 1e-4
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, x, training: Optional[bool] = None):
        training = nn.merge_param("training", self.training, training)
        x = _MLP(
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            layer_norm=self.layer_norm,
            training=training,
        )(x)
        mu = nn.Dense(features=self.n_latent)(x)
        var = (
            nn.Sequential(nn.Dense(features=self.n_latent), nn.softplus)(x)
            + self.var_eps
        )
        latent = Normal(mu, var).rsample()
        return mu, var, latent


class _DecoderVELOVI(nn.Module):
    """
    Decodes data from latent space.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_output
        The dimensionality of the output (data space)
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    n_output: int
    n_layers: int = 1
    n_hidden: int = 128
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    dropout_rate: float = 0.0
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: Optional[bool] = None):
        rho_first = self._MLP(
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            batch_norm=self.use_batch_norm,
            layer_norm=self.use_layer_norm,
            training=training,
        )(z)

        px_rho = nn.Sequential(nn.Dense(features=self.n_output), nn.sigmoid)(rho_first)
        px_tau = nn.Sequential(nn.Dense(features=self.n_output), nn.sigmoid)(rho_first)

        # cells by genes by 4
        pi_first = self._MLP(
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            batch_norm=self.use_batch_norm,
            layer_norm=self.use_layer_norm,
            training=training,
        )(z)
        px_pi = nn.Sequential(
            nn.DenseGeneral(features=(self.n_output, 4)), nn.softplus
        )(pi_first)

        return px_pi, px_rho, px_tau


class JaxVELOVAE(JaxBaseModuleClass):
    """
    Variational auto-encoder model.

    Parameters
    ----------
    n_input
        Number of input genes
    switch_spliced_init
        Initial value for the switch spliced parameter
    switch_unspliced_init
        Initial value for the switch unspliced parameter
    time_dep_transcription_rate
        Whether to use time-dependent transcription rate
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    use_layer_norm
        Whether to use layer norm in layers
    """

    n_input: int
    switch_spliced_init: np.ndarray
    switch_unspliced_init: np.ndarray
    time_dep_transcription_rate: bool = False
    n_hidden: int = 128
    n_latent: int = 10
    n_layers: int = 1
    dropout_rate: float = 0.1
    use_batch_norm: Literal["encoder" "decoder" "none" "both"] = "both"
    use_layer_norm: Literal["encoder" "decoder" "none" "both"] = "both"
    gamma_unconstr_init: Optional[np.ndarray] = None
    alpha_unconstr_init: Optional[np.ndarray] = None
    alpha_1_unconstr_init: Optional[np.ndarray] = None
    lambda_alpha_unconstr_init: Optional[np.ndarray] = None
    t_max: float = 20
    penalty_scale: float = 0.2
    dirichlet_concentration: float = 0.25
    latent_distribution: str = "normal"

    def setup(self):
        use_batch_norm = self.use_batch_norm
        use_layer_norm = self.use_layer_norm
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.switch_spliced = self.variable(
            "switch_spliced", lambda: jnp.from_numpy(self.switch_spliced)
        ).value
        self.switch_unspliced = self.variable(
            "switch_unspliced", lambda: jnp.from_numpy(self.switch_unspliced)
        ).value

        self.n_genes = self.n_input * 2

        # switching time
        self.switch_time_unconstr = self.param(
            "switch_time_unconstr",
            lambda rng, shape: 7 + 0.5 * jnp.randn(rng, shape),
            (self.n_input,),
        )

        # degradation
        if self.gamma_unconstr_init is None:
            self.gamma_mean_unconstr = self.param(
                "gamma_mean_unconstr",
                lambda shape: -1 * jnp.ones(shape),
                (self.n_input,),
            )
        else:
            self.gamma_mean_unconstr = self.variable(
                "gamma_mean_unconstr", lambda: jnp.from_numpy(self.gamma_unconstr_init)
            ).value

        # splicing
        # first samples around 1
        self.beta_mean_unconstr = self.param(
            "beta_mean_unconstr", lambda shape: 0.5 * jnp.ones(shape), (self.n_input,)
        )

        # transcription
        if self.alpha_unconstr_init is None:
            self.alpha_unconstr_init = self.param(
                "alpha_unconstr_init",
                lambda shape: 0 * jnp.ones(shape),
                (self.n_input,),
            )
        else:
            self.alpha_unconstr_init = self.variable(
                "alpha_unconstr_init", lambda: jnp.from_numpy(self.alpha_unconstr_init)
            ).value

        if self.alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = self.variable(
                "alpha_1_unconstr", lambda shape: 0 * jnp.ones(shape), (self.n_input,)
            ).value
        else:
            self.alpha_1_unconstr = self.parameter(
                "alpha_1_unconstr", lambda: jnp.from_numpy(self.alpha_1_unconstr_init)
            )
        if self.lambda_alpha_unconstr_init is None:
            self.lambda_alpha_unconstr = self.variable(
                "lambda_alpha_unconstr", lambda shape: 0 * jnp.ones(shape), (self.n_input,)
            ).value
        else:
            self.lambda_alpha_unconstr = self.parameter(
                "lambda_alpha_unconstr", lambda: jnp.from_numpy(self.lambda_alpha_unconstr_init)
            )
        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = self.param(
            "scale_unconstr", lambda shape: -1 * jnp.ones(shape), (self.n_genes, 4)
        )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        self.z_encoder = _Encoder(
            n_latent=self.n_latent,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            batch_norm=use_batch_norm_encoder,
            layer_norm=use_layer_norm_encoder,
        )
        self.decoder = _DecoderVELOVI(
            n_output=self.n_input,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            batch_norm=use_batch_norm_decoder,
            layer_norm=use_layer_norm_decoder,
        )

    def _get_inference_input(self, ndarrays):
        spliced = ndarrays[REGISTRY_KEYS.X_KEY]
        unspliced = ndarrays[REGISTRY_KEYS.U_KEY]

        input_dict = dict(
            spliced=spliced,
            unspliced=unspliced,
        )
        return input_dict

    def _get_generative_input(self, ndarrays, inference_outputs):
        z = inference_outputs["z"]
        gamma = inference_outputs["gamma"]
        beta = inference_outputs["beta"]
        alpha = inference_outputs["alpha"]
        alpha_1 = inference_outputs["alpha_1"]
        lambda_alpha = inference_outputs["lambda_alpha"]

        input_dict = {
            "z": z,
            "gamma": gamma,
            "beta": beta,
            "alpha": alpha,
            "alpha_1": alpha_1,
            "lambda_alpha": lambda_alpha,
        }
        return input_dict

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
        encoder_input = jnp.cat((spliced_, unspliced_), axis=-1)

        qz_m, qz_v, z = self.z_encoder(encoder_input)
        qz = Normal(qz_m, qz_v)

        if n_samples > 1:
            z = qz.rsample((n_samples,))

        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            gamma=gamma,
            beta=beta,
            alpha=alpha,
            alpha_1=alpha_1,
            lambda_alpha=lambda_alpha,
        )
        return outputs

    def _get_rates(self):
        # globals
        # degradation
        gamma = jnp.clip(nn.softplus(self.gamma_mean_unconstr), 0, 50)
        # splicing
        beta = jnp.clip(nn.softplus(self.beta_mean_unconstr), 0, 50)
        # transcription
        alpha = jnp.clip(nn.softplus(self.alpha_unconstr), 0, 50)
        if self.time_dep_transcription_rate:
            alpha_1 = jnp.clip(nn.softplus(self.alpha_1_unconstr), 0, 50)
            lambda_alpha = jnp.clip(nn.softplus(self.lambda_alpha_unconstr), 0, 50)
        else:
            alpha_1 = self.alpha_1_unconstr
            lambda_alpha = self.lambda_alpha_unconstr

        return gamma, beta, alpha, alpha_1, lambda_alpha

    def generative(self, z, gamma, beta, alpha, alpha_1, lambda_alpha):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha, px_rho, px_tau = self.decoder(
            decoder_input,
        )
        px_pi = Dirichlet(px_pi_alpha).rsample()

        scale_unconstr = self.scale_unconstr
        scale = nn.softplus(scale_unconstr)

        mixture_dist_s, mixture_dist_u, end_penalty = self.get_px(
            px_pi,
            px_rho,
            px_tau,
            scale,
            gamma,
            beta,
            alpha,
            alpha_1,
            lambda_alpha,
        )

        return dict(
            px_pi=px_pi,
            px_rho=px_rho,
            px_tau=px_tau,
            scale=scale,
            px_pi_alpha=px_pi_alpha,
            mixture_dist_u=mixture_dist_u,
            mixture_dist_s=mixture_dist_s,
            end_penalty=end_penalty,
        )

    def loss(
        self,
        ndarrays,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
        n_obs: float = 1.0,
    ):
        spliced = ndarrays[REGISTRY_KEYS.X_KEY]
        unspliced = ndarrays[REGISTRY_KEYS.U_KEY]

        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        kl_divergence_z = kl(Normal(qz_m, jnp.sqrt(qz_v)), Normal(0, 1)).sum(axis=1)

        reconst_loss_s = -mixture_dist_s.log_prob(spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)

        reconst_loss = reconst_loss_u.sum(axis=-1) + reconst_loss_s.sum(axis=-1)

        kl_pi = kl(
            Dirichlet(px_pi_alpha),
            Dirichlet(self.dirichlet_concentration * jnp.ones_like(px_pi)),
        ).sum(axis=-1)

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = jnp.mean(reconst_loss + weighted_kl_local)

        # combine local and global
        global_loss = 0
        loss = (
            local_loss
            + self.penalty_scale * (1 - kl_weight) * end_penalty
            + (1 / n_obs) * kl_weight * (global_loss)
        )

        loss_recoder = LossRecorder(
            loss, reconst_loss, kl_local, jnp.ndarray(global_loss)
        )

        return loss_recoder

    def get_px(
        self,
        px_pi,
        px_rho,
        px_tau,
        scale,
        gamma,
        beta,
        alpha,
        alpha_1,
        lambda_alpha,
    ) -> jnp.ndarray:

        t_s = jnp.clip(nn.softplus(self.switch_time_unconstr), 0, self.t_max)

        n_cells = px_pi.shape[0]

        # component dist
        comp_dist = Categorical(probs=px_pi)

        # induction
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s * px_rho
        )

        if self.time_dep_transcription_rate:
            mean_u_ind_steady = jnp.resize((alpha_1 / beta), (n_cells, self.n_input))
            mean_s_ind_steady = jnp.resize((alpha_1 / gamma), (n_cells, self.n_input))
        else:
            mean_u_ind_steady = jnp.resize((alpha / beta), (n_cells, self.n_input))
            mean_s_ind_steady = jnp.resize((alpha / gamma), (n_cells, self.n_input))
        scale_u = jnp.sqrt(
            jnp.resize(scale[: self.n_input, :], (n_cells, self.n_input, 4))
        )

        # repression
        u_0, s_0 = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s
        )

        tau = px_tau
        mean_u_rep, mean_s_rep = self._get_repression_unspliced_spliced(
            u_0,
            s_0,
            beta,
            gamma,
            (self.t_max - t_s) * tau,
        )
        mean_u_rep_steady = jnp.zeros_like(mean_u_ind)
        mean_s_rep_steady = jnp.zeros_like(mean_u_ind)
        scale_s = jnp.resize(
            scale[self.n_input :, :], (n_cells, self.n_input, 4)
        ).sqrt()

        end_penalty = ((u_0 - self.switch_unspliced).pow(2)).sum() + (
            (s_0 - self.switch_spliced).pow(2)
        ).sum()

        # unspliced
        mean_u = jnp.stack(
            (
                mean_u_ind,
                mean_u_ind_steady,
                mean_u_rep,
                mean_u_rep_steady,
            ),
            axis=2,
        )
        scale_u = jnp.stack(
            (
                scale_u[..., 0],
                scale_u[..., 0],
                scale_u[..., 0],
                0.1 * scale_u[..., 0],
            ),
            axis=2,
        )
        dist_u = Normal(mean_u, scale_u)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u)

        # spliced
        mean_s = jnp.stack(
            (mean_s_ind, mean_s_ind_steady, mean_s_rep, mean_s_rep_steady),
            axis=2,
        )
        scale_s = jnp.stack(
            (
                scale_s[..., 0],
                scale_s[..., 0],
                scale_s[..., 0],
                0.1 * scale_s[..., 0],
            ),
            axis=2,
        )
        dist_s = Normal(mean_s, scale_s)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s)

        return mixture_dist_s, mixture_dist_u, end_penalty

    def _get_induction_unspliced_spliced(
        self, alpha, alpha_1, lambda_alpha, beta, gamma, t, eps=1e-6
    ):
        if self.time_dep_transcription_rate:
            unspliced = alpha_1 / beta * (1 - jnp.exp(-beta * t)) - (
                alpha_1 - alpha
            ) / (beta - lambda_alpha) * (
                jnp.exp(-lambda_alpha * t) - jnp.exp(-beta * t)
            )

            spliced = (
                alpha_1 / gamma * (1 - jnp.exp(-gamma * t))
                + alpha_1
                / (gamma - beta + eps)
                * (jnp.exp(-gamma * t) - jnp.exp(-beta * t))
                - beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - lambda_alpha + eps)
                * (jnp.exp(-lambda_alpha * t) - jnp.exp(-gamma * t))
                + beta
                * (alpha_1 - alpha)
                / (beta - lambda_alpha + eps)
                / (gamma - beta + eps)
                * (jnp.exp(-beta * t) - jnp.exp(-gamma * t))
            )
        else:
            unspliced = (alpha / beta) * (1 - jnp.exp(-beta * t))
            spliced = (alpha / gamma) * (1 - jnp.exp(-gamma * t)) + (
                alpha / ((gamma - beta) + eps)
            ) * (jnp.exp(-gamma * t) - jnp.exp(-beta * t))

        return unspliced, spliced

    def _get_repression_unspliced_spliced(self, u_0, s_0, beta, gamma, t, eps=1e-6):
        unspliced = jnp.exp(-beta * t) * u_0
        spliced = s_0 * jnp.exp(-gamma * t) - (beta * u_0 / ((gamma - beta) + eps)) * (
            jnp.exp(-gamma * t) - jnp.exp(-beta * t)
        )
        return unspliced, spliced

    def sample(
        self,
    ) -> np.ndarray:
        """Not implemented."""
        raise NotImplementedError
