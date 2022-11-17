from json import encoder
from typing import Callable, Iterable, Optional

import numpy as np
from scvi._compat import Literal
from scvi.module.base import JaxBaseModuleClass, LossRecorder, flax_configure
from flax import linen as nn
import chex
from numpyro.distributions import (
    Categorical,
    Dirichlet as DirichletNP,
    MixtureSameFamily,
    Normal,
    kl_divergence as kl,
)
from tensorflow_probability.substrates import jax as tfp

import jax.numpy as jnp
import jax

from ._constants import REGISTRY_KEYS

_STATE_COLLECTION = "constants"
_LATENT_RNG_KEY = "latent"

Dirichlet = tfp.distributions.Dirichlet

from scvi.module._jaxvae import Dense
from functools import partial
from jax import custom_jvp

# def _get_max(max_, array):
#     max_ = jnp.maximum(max_, array)
#     return max_, 0


# @partial(custom_jvp, nondiff_argnums=(0,))
# def dirichlet_sample(key, alpha):
#     return Dirichlet(alpha + 1).sample(seed=key)


# @dirichlet_sample.defjvp
# def dirichlet_sample_jvp(key, primals, tangents):
#     (alpha,) = primals
#     (alpha_dot,) = tangents
#     dirichlet = partial(jax.random.dirichlet, key)
#     return jax.jvp(dirichlet, primals, tangents)


def softplus(x, threshold=20):
    """Softplus function that avoids overflow."""
    return jnp.where(x < threshold, jnp.log1p(jnp.exp(x)), x)


class _MLP(nn.Module):
    """Fully connected layers with dropout and batchnorm."""

    n_layers: int
    n_hidden: int
    dropout_rate: float
    batch_norm: bool = True
    layer_norm: bool = True
    training: Optional[bool] = None
    activation_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x, training: Optional[bool] = None):
        training = nn.merge_param("training", self.training, training)
        is_eval = not training
        for _ in range(self.n_layers):
            x = Dense(features=self.n_hidden)(x)
            if self.batch_norm:
                x = nn.BatchNorm(
                    use_running_average=is_eval, momentum=0.99, epsilon=0.001
                )(x)
            if self.layer_norm:
                x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
            x = self.activation_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=is_eval)(x)
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
        mu = Dense(features=self.n_latent)(x)
        var = softplus(nn.Dense(features=self.n_latent)(x)) + self.var_eps
        qz = Normal(mu, var, validate_args=False)
        z = qz.rsample(self.make_rng(_LATENT_RNG_KEY))
        return qz, z


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
    batch_norm
        Whether to use batch norm in layers
    layer_norm
        Whether to use layer norm in layers
    """

    n_output: int
    n_layers: int = 1
    n_hidden: int = 128
    batch_norm: bool = True
    layer_norm: bool = True
    dropout_rate: float = 0.0
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, z: jnp.ndarray, training: Optional[bool] = None):
        training = nn.merge_param("training", self.training, training)
        rho_first = _MLP(
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            layer_norm=self.layer_norm,
            training=training,
        )(z)

        px_rho = nn.sigmoid(Dense(features=self.n_output)(rho_first))
        px_tau = nn.sigmoid(Dense(features=self.n_output)(rho_first))

        # cells by genes by 4
        pi_first = _MLP(
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            layer_norm=self.layer_norm,
            training=training,
        )(z)
        px_pi = softplus(Dense(features=(self.n_output * 4))(pi_first))
        px_pi = jnp.reshape(px_pi, (-1, self.n_output, 4))

        return px_pi, px_rho, px_tau


@flax_configure
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
    training: bool = True

    def setup(self):
        use_batch_norm = self.use_batch_norm
        use_layer_norm = self.use_layer_norm
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.switch_spliced = self.variable(
            _STATE_COLLECTION,
            "switch_spliced",
            lambda: jnp.asarray(self.switch_spliced_init.copy()),
        )
        self.switch_unspliced = self.variable(
            _STATE_COLLECTION,
            "switch_unspliced",
            lambda: jnp.asarray(self.switch_unspliced_init.copy()),
        )

        self.n_genes = self.n_input * 2

        # switching time
        self.switch_time_unconstr = self.param(
            "switch_time_unconstr",
            lambda rng, shape: 7 + 0.5 * jax.random.normal(rng, shape),
            (self.n_input,),
        )

        # degradation
        if self.gamma_unconstr_init is None:
            self.gamma_mean_unconstr = self.param(
                "gamma_mean_unconstr",
                lambda _, shape: -1 * jnp.ones(shape),
                (self.n_input,),
            )
        else:
            self.gamma_mean_unconstr = self.param(
                "gamma_mean_unconstr",
                lambda _: jnp.asarray(self.gamma_unconstr_init),
            )

        # splicing
        # first samples around 1
        self.beta_mean_unconstr = self.param(
            "beta_mean_unconstr",
            lambda _, shape: 0.5 * jnp.ones(shape),
            (self.n_input,),
        )

        # transcription
        if self.alpha_unconstr_init is None:
            self.alpha_unconstr = self.param(
                "alpha_unconstr",
                lambda _, shape: 0 * jnp.ones(shape),
                (self.n_input,),
            )
        else:
            self.alpha_unconstr = self.param(
                "alpha_unconstr",
                lambda _: jnp.asarray(self.alpha_unconstr_init),
            )

        if self.alpha_1_unconstr_init is None:
            self.alpha_1_unconstr = 0
        else:
            self.alpha_1_unconstr = self.param(
                "alpha_1_unconstr", lambda _: jnp.asarray(self.alpha_1_unconstr_init)
            )
        if self.lambda_alpha_unconstr_init is None:
            self.lambda_alpha_unconstr = 0
        else:
            self.lambda_alpha_unconstr = self.param(
                "lambda_alpha_unconstr",
                lambda _: self.lambda_alpha_unconstr_init,
            )
        # likelihood dispersion
        # for now, with normal dist, this is just the variance
        self.scale_unconstr = self.param(
            "scale_unconstr", lambda _, shape: -1 * jnp.ones(shape), (self.n_genes,)
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

    @property
    def required_rngs(self):  # noqa: D102
        return ("params", "dropout", _LATENT_RNG_KEY)

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
        encoder_input = jnp.concatenate((spliced_, unspliced_), axis=-1)

        qz, z = self.z_encoder(encoder_input, training=self.training)

        if n_samples > 1:
            z = qz.rsample(self.make_rng(_LATENT_RNG_KEY), (n_samples,))

        gamma, beta, alpha, alpha_1, lambda_alpha = self._get_rates()

        outputs = dict(
            z=z,
            qz=qz,
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
        gamma = jnp.clip(softplus(self.gamma_mean_unconstr), 0, 50)
        # splicing
        beta = jnp.clip(softplus(self.beta_mean_unconstr), 0, 50)
        # transcription
        alpha = jnp.clip(softplus(self.alpha_unconstr), 0, 50)
        if self.time_dep_transcription_rate:
            alpha_1 = jnp.clip(softplus(self.alpha_1_unconstr), 0, 50)
            lambda_alpha = jnp.clip(softplus(self.lambda_alpha_unconstr), 0, 50)
        else:
            alpha_1 = self.alpha_1_unconstr
            lambda_alpha = self.lambda_alpha_unconstr

        return gamma, beta, alpha, alpha_1, lambda_alpha

    def _get_rates_from_params(self):
        # globals
        # degradation
        gamma = jnp.clip(softplus(self.params["gamma_mean_unconstr"]), 0, 50)
        # splicing
        beta = jnp.clip(softplus(self.params["beta_mean_unconstr"]), 0, 50)
        # transcription
        alpha = jnp.clip(softplus(self.params["alpha_unconstr"]), 0, 50)
        if self.time_dep_transcription_rate:
            alpha_1 = jnp.clip(softplus(self.params["alpha_1_unconstr"]), 0, 50)
            lambda_alpha = jnp.clip(
                softplus(self.params["lambda_alpha_unconstr"]), 0, 50
            )
        else:
            alpha_1 = jnp.array([0.0])
            lambda_alpha = jnp.array([0.0])

        return gamma, beta, alpha, alpha_1, lambda_alpha

    def generative(self, z, gamma, beta, alpha, alpha_1, lambda_alpha):
        """Runs the generative model."""
        decoder_input = z
        px_pi_alpha, px_rho, px_tau = self.decoder(
            decoder_input, training=self.training
        )
        # px_pi_alpha = jnp.clip(px_pi_alpha, 0, 5)
        px_pi = DirichletNP(px_pi_alpha).rsample(self.make_rng(_LATENT_RNG_KEY))
        # px_pi_unnormalized = jnp.square(
        #     Normal(2 * px_pi_alpha, jnp.sqrt(0.5 * jnp.ones_like(px_pi_alpha))).rsample(
        #         self.make_rng(_LATENT_RNG_KEY)
        #     )
        # )
        # px_pi = dirichlet_sample(self.make_rng(_LATENT_RNG_KEY), px_pi_alpha)
        # px_pi = jnp.nan_to_num(
        #     px_pi,
        #     nan=jnp.finfo(px_pi).tiny,
        #     posinf=1 - jnp.finfo(px_pi).eps,
        #     neginf=jnp.finfo(px_pi).tiny,
        # )
        # px_pi = px_pi_unnormalized / jnp.sum(px_pi_unnormalized, axis=-1, keepdims=True)
        # px_pi = jnp.clip(
        #     px_pi, a_min=jnp.finfo(px_pi).tiny, a_max=1 - jnp.finfo(px_pi).eps
        # )
        # px_pi = DirichletNP(px_pi_alpha).rsample(self.make_rng(_LATENT_RNG_KEY))
        # For now just do MAP estimation for pi
        # px_pi = px_pi_alpha / jnp.sum(px_pi_alpha, axis=-1, keepdims=True)

        scale_unconstr = self.scale_unconstr
        scale = softplus(scale_unconstr)

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
            mixture_dist_s=mixture_dist_s,
            mixture_dist_u=mixture_dist_u,
            end_penalty=end_penalty,
        )

    def loss(
        self,
        ndarrays,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        spliced = ndarrays[REGISTRY_KEYS.X_KEY]
        unspliced = ndarrays[REGISTRY_KEYS.U_KEY]

        qz = inference_outputs["qz"]

        px_pi = generative_outputs["px_pi"]
        px_pi_alpha = generative_outputs["px_pi_alpha"]

        end_penalty = generative_outputs["end_penalty"]
        mixture_dist_s = generative_outputs["mixture_dist_s"]
        mixture_dist_u = generative_outputs["mixture_dist_u"]

        kl_divergence_z = kl(qz, Normal(0, 1, validate_args=False)).sum(axis=1)

        reconst_loss_s = -mixture_dist_s.log_prob(spliced)
        reconst_loss_u = -mixture_dist_u.log_prob(unspliced)

        reconst_loss = reconst_loss_u.sum(axis=-1) + reconst_loss_s.sum(axis=-1)

        kl_pi = kl(
            DirichletNP(px_pi_alpha),
            DirichletNP(
                self.dirichlet_concentration
                * jnp.ones((px_pi.shape[-2], px_pi.shape[-1])),
            ),
        ).sum(axis=-1)
        # kl_pi = (
        #     Dirichlet(
        #         self.dirichlet_concentration
        #         * jnp.ones((px_pi.shape[-2], px_pi.shape[-1])),
        #         validate_args=False,
        #     )
        #     .log_prob(px_pi)
        #     .sum(axis=-1)
        #     * -1
        # )
        chex.assert_equal_shape([kl_pi, kl_divergence_z])

        # local loss
        kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * (kl_divergence_z) + kl_pi

        local_loss = jnp.mean(reconst_loss + weighted_kl_local)

        # combine local and global
        loss = local_loss + self.penalty_scale * (1 - kl_weight) * end_penalty

        loss_recoder = LossRecorder(loss, reconst_loss, kl_local)

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

        t_s = jnp.clip(softplus(self.switch_time_unconstr), 0, self.t_max)
        # component dist
        comp_dist = Categorical(probs=px_pi)

        # induction
        mean_u_ind, mean_s_ind = self._get_induction_unspliced_spliced(
            alpha, alpha_1, lambda_alpha, beta, gamma, t_s * px_rho
        )

        if self.time_dep_transcription_rate:
            mean_u_ind_steady = (alpha_1 / beta) * jnp.ones_like(mean_u_ind)
            mean_s_ind_steady = (alpha_1 / gamma) * jnp.ones_like(mean_u_ind)
        else:
            mean_u_ind_steady = (alpha / beta) * jnp.ones_like(mean_u_ind)
            mean_s_ind_steady = (alpha / gamma) * jnp.ones_like(mean_u_ind)
        scale_u = jnp.sqrt(scale[: self.n_input])

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
        scale_s = jnp.sqrt(scale[self.n_input :])

        end_penalty = (jnp.square(u_0 - self.switch_unspliced.value)).sum() + (
            jnp.square(s_0 - self.switch_spliced.value)
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
        # dims by 4
        scale_u = jnp.stack(
            (
                scale_u,
                scale_u,
                scale_u,
                0.1 * scale_u,
            ),
            axis=1,
        )
        dist_u = Normal(mean_u, scale_u, validate_args=False)
        mixture_dist_u = MixtureSameFamily(comp_dist, dist_u, validate_args=False)

        # spliced
        mean_s = jnp.stack(
            (mean_s_ind, mean_s_ind_steady, mean_s_rep, mean_s_rep_steady),
            axis=2,
        )
        # dims by 4
        scale_s = jnp.stack(
            (
                scale_s,
                scale_s,
                scale_s,
                0.1 * scale_s,
            ),
            axis=1,
        )
        # scale gets broadcasted
        dist_s = Normal(mean_s, scale_s, validate_args=False)
        mixture_dist_s = MixtureSameFamily(comp_dist, dist_s, validate_args=False)

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
