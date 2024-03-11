# velovi

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/yoseflab/velovi/test.yml?branch=main
[link-tests]: https://github.com/yoseflab/velovi/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/velovi

🚧 :warning: This package is no longer being actively developed or maintained. Please use the
[scvi-tools](https://github.com/scverse/scvi-tools) package instead. See this
[thread](https://github.com/scverse/scvi-tools/issues/2610) for more details. :warning: 🚧

Variational inference for RNA velocity. This is an experimental repo for the veloVI model. Installation instructions and tutorials are in the docs. Over the next few months veloVI will move into [scvelo](https://scvelo.org/).

## Getting started

Please refer to the [documentation][link-docs].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several alternative options to install velovi:

<!--
1) Install the latest release of `velovi` from `PyPI <https://pypi.org/project/velovi/>`_:

```bash
pip install velovi
```
-->

1. Install the latest release on PyPI:

```bash
pip install velovi
```

2. Install the latest development version:

```bash
pip install git+https://github.com/yoseflab/velovi.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

```
@article{gayoso2022deep,
  title={Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells},
  author={Gayoso, Adam and Weiler, Philipp and Lotfollahi, Mohammad and Klein, Dominik and Hong, Justin and Streets, Aaron M and Theis, Fabian J and Yosef, Nir},
  journal={bioRxiv},
  pages={2022--08},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
```

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/yoseflab/velovi/issues
[changelog]: https://velovi.readthedocs.io/latest/changelog.html
[link-docs]: https://velovi.readthedocs.io
[link-api]: https://velovi.readthedocs.io/latest/api.html
