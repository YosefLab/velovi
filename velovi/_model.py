import logging
import warnings
from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from scipy.sparse import isspmatrix
from scvi._compat import Literal
from scvi._utils import _doc_params
from scvi.data import AnnDataManager
from scvi.data.fields import LayerField
from scvi.model._utils import scrna_raw_counts_properties
from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin
from scvi.model.base._utils import _de_core
from scvi.utils._docstrings import doc_differential_expression, setup_anndata_dsp
from sklearn.metrics.pairwise import cosine_similarity

from ._constants import REGISTRY_KEYS
from ._module import VELOVAE

logger = logging.getLogger(__name__)


class VELOVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """
    Velocity Variational Inference

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~velovi.VELOVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    **model_kwargs
        Keyword args for :class:`~velovi.VELOVAE`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        gamma_var_key: Optional[str] = None,
        induction_genes: Optional[Iterable[str]] = None,
        **model_kwargs,
    ):
        super().__init__(adata)

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        unspliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.U_KEY)

        sorted_unspliced = np.argsort(unspliced, axis=0)
        ind = int(adata.n_obs * 0.99)
        us_upper_ind = sorted_unspliced[ind:, :]

        us_upper = []
        ms_upper = []
        for i in range(len(us_upper_ind)):
            row = us_upper_ind[i]
            us_upper += [unspliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
            ms_upper += [spliced[row, np.arange(adata.n_vars)][np.newaxis, :]]
        us_upper = np.median(np.concatenate(us_upper, axis=0), axis=0)
        ms_upper = np.median(np.concatenate(ms_upper, axis=0), axis=0)

        y = torch.from_numpy(us_upper)
        log_alpha_init = torch.where(y > 20, y, y.expm1().log()).numpy()
        # log_alpha_init = log_alpha_init.mean(0)  # + log_alpha_init.std(0)
        if isspmatrix(log_alpha_init):
            log_alpha_init = log_alpha_init.A
        log_alpha_init = np.asarray(log_alpha_init).ravel()

        if gamma_var_key is not None:
            # log_gamma = np.log(np.exp(adata.var[gamma_var_key].values) - 1)
            y = torch.from_numpy(us_upper / ms_upper)
            log_gamma = torch.where(y > 20, y, y.expm1().log()).numpy()
        else:
            log_gamma = None

        # ms_upper = np.percentile(spliced, 99, axis=0)
        # us_upper = np.percentile(unspliced, 99, axis=0)

        module_class = VELOVAE
        if induction_genes is not None:
            induction_gene_mask = pd.Series(
                [False] * adata.n_vars, index=adata.var_names
            )
            induction_gene_mask.loc[induction_genes] = True
            induction_gene_mask = induction_gene_mask.values.astype(bool)
        else:
            induction_gene_mask = None

        self.module = module_class(
            n_input=self.summary_stats["n_vars"],
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            log_gamma_init=log_gamma,
            log_alpha_init=log_alpha_init,
            switch_spliced=ms_upper,
            switch_unspliced=us_upper,
            induction_gene_mask=induction_gene_mask,
            **model_kwargs,
        )
        self._model_summary_string = (
            "VELOVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
        )
        self.init_params_ = self._get_init_params(locals())

    @torch.no_grad()
    def get_state_assignment(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        hard_assignment: bool = False,
        n_samples: int = 20,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        states = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats

    @torch.no_grad()
    def get_state_entropy(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 20,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        states = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                output = torch.distributions.Dirichlet(
                    generative_outputs["px_pi_alpha"]
                ).entropy()
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        return states

    @torch.no_grad()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        time_statistic: Literal["mean", "max"] = "mean",
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                rep_steady_prob = pi[..., 3]
                # renormalize
                switch_time = F.softplus(self.module.switch_time_unconstr)

                # clamped = torch.ones_like(generative_outputs["px_rho"])
                # ind_time = torch.clamp(
                #     generative_outputs["px_rho"], 0 * clamped, switch_time * clamped
                # )

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.module.t_max - switch_time)
                )
                # rep_time = switch_time + torch.clamp(
                #     generative_outputs["px_tau"],
                #     0 * clamped,
                #     (self.module.t_max - switch_time) * clamped,
                # )

                # max
                # mask = rep_prob > ind_prob
                # ind_time.masked_scatter(mask, rep_time)
                # output = ind_time
                # expectation
                if time_statistic == "mean":
                    output = (
                        ind_prob * ind_time
                        + rep_prob * rep_time
                        + steady_prob * switch_time
                        + rep_steady_prob * self.module.t_max
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return times

    @torch.no_grad()
    def get_velocity(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal["spliced", "unspliced"] = "spliced",
        clip: bool = True,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        velos = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                alpha = inference_outputs["alpha"]
                beta = inference_outputs["beta"]
                gamma = inference_outputs["gamma"]
                tau = generative_outputs["px_tau"]
                rho = generative_outputs["px_rho"]

                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                switch_time = F.softplus(self.module.switch_time_unconstr)

                ind_time = switch_time * rho
                u_0, s_0 = self.module._get_induction_unspliced_spliced(
                    alpha, beta, gamma, switch_time
                )
                rep_time = (self.module.t_max - switch_time) * tau
                mean_u_rep, mean_s_rep = self.module._get_repression_unspliced_spliced(
                    u_0,
                    s_0,
                    beta,
                    gamma,
                    rep_time,
                )
                if velo_mode == "spliced":
                    velo_rep = beta * mean_u_rep - gamma * mean_s_rep
                else:
                    velo_rep = -beta * mean_u_rep
                mean_u_ind, mean_s_ind = self.module._get_induction_unspliced_spliced(
                    alpha, beta, gamma, ind_time
                )
                if velo_mode == "spliced":
                    velo_ind = beta * mean_u_ind - gamma * mean_s_ind
                else:
                    velo_ind = alpha - beta * mean_u_ind

                if velo_mode == "spliced":
                    # velo_steady = beta * u_0 - gamma * s_0
                    velo_steady = torch.zeros_like(velo_ind)
                else:
                    # velo_steady = alpha - beta * u_0
                    velo_steady = torch.zeros_like(velo_ind)

                # expectation
                if velo_statistic == "mean":
                    output = (
                        ind_prob * velo_ind
                        + rep_prob * velo_rep
                        + steady_prob * velo_steady
                    )
                # maximum
                else:
                    v = torch.stack(
                        [
                            velo_ind,
                            velo_steady.expand(velo_ind.shape),
                            velo_rep,
                            torch.zeros_like(velo_rep),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (v * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes
            velos.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                # mean over samples axis
                velos[-1] = np.mean(velos[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            velos = np.concatenate(velos, axis=-2)
        else:
            velos = np.concatenate(velos, axis=0)

        spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)

        if clip:
            velos = np.clip(velos, -spliced[indices], None)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                velos,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return velos

    @torch.no_grad()
    def get_expression_fit(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        fits_s = []
        fits_u = []
        for tensors in scdl:
            minibatch_samples_s = []
            minibatch_samples_u = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (mixture_dist_s, mixture_dist_u, _,) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    beta,
                    alpha,
                )
                fit_s = mixture_dist_s.mean
                fit_u = mixture_dist_u.mean

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            # samples by cells by genes
            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            # samples by cells by genes
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u

    @torch.no_grad()
    def get_gene_likelihood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""
        Returns the likelihood per gene. Higher is better.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        rls = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                spliced = tensors[REGISTRY_KEYS.X_KEY]
                unspliced = tensors[REGISTRY_KEYS.U_KEY]

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (mixture_dist_s, mixture_dist_u, _,) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    beta,
                    alpha,
                )
                reconst_loss_s = -mixture_dist_s.log_prob(spliced)
                reconst_loss_u = -mixture_dist_u.log_prob(unspliced)
                output = -(reconst_loss_s + reconst_loss_u)
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            rls.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rls[-1] = np.mean(rls[-1], axis=0)

        rls = np.concatenate(rls, axis=0)
        return rls

    @torch.no_grad()
    def get_rates(self, mean: bool = True):

        gamma, beta, alpha = self.module._get_rates()

        return {
            "beta": beta.cpu().numpy(),
            "gamma": gamma.cpu().numpy(),
            "alpha": alpha.cpu().numpy(),
        }

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        spliced_layer: str,
        unspliced_layer: str,
        **kwargs,
    ) -> Optional[AnnData]:
        """
        %(summary)s.
        Parameters
        ----------
        %(param_adata)s
        spliced_layer
            Layer in adata with spliced normalized expression
        unspliced_layer
            Layer in adata with unspliced normalized expression

        Returns
        -------
        %(returns)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, spliced_layer, is_count_data=False),
            LayerField(REGISTRY_KEYS.U_KEY, unspliced_layer, is_count_data=False),
        ]
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @torch.no_grad()
    @_doc_params(
        doc_differential_expression=doc_differential_expression,
    )
    def differential_velocity(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla", "change"] = "vanilla",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        A unified method for differential velocity analysis.

        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.

        Parameters
        ----------
        {doc_differential_expression}
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        def model_fn(adata, **kwargs):
            if "transform_batch" in kwargs.keys():
                kwargs.pop("transform_batch")
            return partial(
                self.get_velocity,
                batch_size=batch_size,
                n_samples=1,
                return_numpy=True,
                clip=False,
            )(adata, **kwargs)

        col_names = adata.var_names

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            scrna_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            **kwargs,
        )

        return result

    @torch.no_grad()
    @_doc_params(
        doc_differential_expression=doc_differential_expression,
    )
    def differential_time(
        self,
        adata: Optional[AnnData] = None,
        groupby: Optional[str] = None,
        group1: Optional[Iterable[str]] = None,
        group2: Optional[str] = None,
        idx1: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        idx2: Optional[Union[Sequence[int], Sequence[bool], str]] = None,
        mode: Literal["vanilla", "change"] = "vanilla",
        delta: float = 0.25,
        batch_size: Optional[int] = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Optional[Iterable[str]] = None,
        batchid2: Optional[Iterable[str]] = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""
        A unified method for differential velocity analysis.

        Implements `"vanilla"` DE [Lopez18]_ and `"change"` mode DE [Boyeau19]_.

        Parameters
        ----------
        {doc_differential_expression}
        **kwargs
            Keyword args for :meth:`scvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        adata = self._validate_anndata(adata)

        def model_fn(adata, **kwargs):
            if "transform_batch" in kwargs.keys():
                kwargs.pop("transform_batch")
            return partial(
                self.get_latent_time,
                batch_size=batch_size,
                n_samples=1,
                return_numpy=True,
            )(adata, **kwargs)

        col_names = adata.var_names

        result = _de_core(
            self.get_anndata_manager(adata, required=True),
            model_fn,
            groupby,
            group1,
            group2,
            idx1,
            idx2,
            all_stats,
            scrna_raw_counts_properties,
            col_names,
            mode,
            batchid1,
            batchid2,
            delta,
            batch_correction,
            fdr_target,
            silent,
            **kwargs,
        )

        return result

    @torch.no_grad()
    def differential_transition(
        self,
        groupby: str,
        group1: str,
        group2: str,
        adata: Optional[AnnData] = None,
        batch_size: Optional[int] = None,
        n_samples: Optional[int] = 5000,
        **kwargs,
    ) -> pd.DataFrame:
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        if not isinstance(group1, str):
            raise ValueError("Group 1 must be a string")

        cell_idx1 = (adata.obs[groupby] == group1).to_numpy().ravel()
        if group2 is None:
            cell_idx2 = ~cell_idx1
        else:
            cell_idx2 = (adata.obs[groupby] == group2).to_numpy().ravel()

        indices1 = np.random.choice(
            np.asarray(np.where(cell_idx1)[0].ravel()), n_samples
        )
        indices2 = np.random.choice(
            np.asarray(np.where(cell_idx2)[0].ravel()), n_samples
        )

        velo1 = self.get_velocity(
            adata,
            return_numpy=True,
            indices=indices1,
            n_samples=1,
            batch_size=batch_size,
        )
        velo1 = velo1 - velo1.mean(1)[:, np.newaxis]
        velo2 = self.get_velocity(
            adata,
            return_numpy=True,
            indices=indices2,
            n_samples=1,
            batch_size=batch_size,
        )
        velo2 = velo2 - velo2.mean(1)[:, np.newaxis]

        spliced = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        delta12 = spliced[indices2] - spliced[indices1]
        delta12 = delta12 - delta12.mean(1)[:, np.newaxis]

        delta21 = spliced[indices1] - spliced[indices2]
        delta21 = delta21 - delta21.mean(1)[:, np.newaxis]

        # TODO: Make more efficient
        correlation12 = np.diagonal(cosine_similarity(velo1, delta12))
        correlation21 = np.diagonal(cosine_similarity(velo2, delta21))

        return correlation12, correlation21
