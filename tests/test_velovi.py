from scvi.data import synthetic_iid

from velovi import VELOVI


def test_velovi():
    n_latent = 5
    adata = synthetic_iid()
    adata.layers["spliced"] = adata.X.copy()
    adata.layers["unspliced"] = adata.X.copy()
    VELOVI.setup_anndata(adata, unspliced_layer="unspliced", spliced_layer="spliced")
    model = VELOVI(adata, n_latent=n_latent)
    model.train(1, check_val_every_n_epoch=1, train_size=0.5)
    z = model.get_latent_representation()
    model.get_velocity_from_latent(z)
    model.get_velocity()
    model.get_latent_time()
    model.get_state_assignment()
    model.differential_velocity(groupby="labels")
    model.differential_transition(groupby="labels", group1="label_0", group2="label_1")
    model.get_expression_fit()
    model.get_directional_uncertainty()
    model.get_permutation_scores(labels_key="labels")

    model = VELOVI(adata, n_latent=n_latent, linear_decoder=True)
    model.train(1, train_size=0.5)
    model.get_loadings()
    model.get_variance_explained(labels_key="labels")

    model.history

    # tests __repr__
    print(model)
