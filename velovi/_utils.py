from typing import Optional

import numpy as np
import scvelo as scv
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler


def preprocess_data(
    adata: AnnData,
    spliced_layer: Optional[str] = "Ms",
    unspliced_layer: Optional[str] = "Mu",
) -> AnnData:
    """Preprocess data.

    This function removes poorly detected genes and minmax scales the data.

    Parameters
    ----------
    adata
        Annotated data matrix.
    spliced_layer
        Name of the spliced layer.
    unspliced_layer
        Name of the unspliced layer.

    Returns
    -------
    Preprocessed adata.
    """
    scaler = MinMaxScaler()
    adata.layers[spliced_layer] = scaler.fit_transform(adata.layers[spliced_layer])

    scaler = MinMaxScaler()
    adata.layers[unspliced_layer] = scaler.fit_transform(adata.layers[unspliced_layer])

    scv.tl.velocity(adata, mode="deterministic")

    adata = adata[
        :, np.logical_and(adata.var.velocity_r2 > 0, adata.var.velocity_gamma > 0)
    ].copy()
    adata = adata[:, adata.var.velocity_genes].copy()

    return adata
