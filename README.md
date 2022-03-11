# velovi

[![Stars](https://img.shields.io/github/stars/YosefLab/velovi?logo=GitHub&color=yellow)](https://github.com/YosefLab/scvi-tools/stargazers)
[![Documentation Status](https://readthedocs.org/projects/velovi/badge/?version=latest)](https://velovi.readthedocs.io/en/stable/?badge=stable)
![Build Status](https://github.com/YosefLab/velovi/workflows/velovi/badge.svg)
[![codecov](https://codecov.io/gh/YosefLab/velovi/branch/main/graph/badge.svg?token=BGI9Z8R11R)](https://codecov.io/gh/YosefLab/velovi)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

## Installation

`pip install -e .[dev,docs,tutorials]` after git clone.

## Usage

```
from velovi import VELOVI

VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
model = VELOVI(adata)
# current training params
model.train(
    800,
    plan_kwargs=dict(
        lr=1e-2, weight_decay=2e-4, optimizer="AdamW"
    ),
    check_val_every_n_epoch=1,
    batch_size=256,
    gradient_clip_val=10,
)
model.get_velocity()
model.get_latent_time()
model.get_rates()
```
