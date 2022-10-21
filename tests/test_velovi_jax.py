import scvelo as scv
from scvi.data import synthetic_iid

from velovi import JaxVELOVI


def test_velovi_jax():
    n_latent = 5
    adata = synthetic_iid()
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = adata.X.copy()
    JaxVELOVI.setup_anndata(adata, unspliced_layer="unspliced", spliced_layer="spliced")
    model = JaxVELOVI(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    model.get_rates()
