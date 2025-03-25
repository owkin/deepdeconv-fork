<img src="https://github.com/scverse/scvi-tools/blob/main/docs/_static/scvi-tools-horizontal.svg?raw=true" width="400" alt="scvi-tools">

[![Stars](https://img.shields.io/github/stars/scverse/scvi-tools?logo=GitHub&color=yellow)](https://github.com/YosefLab/scvi-tools/stargazers)
[![PyPI](https://img.shields.io/pypi/v/scvi-tools.svg)](https://pypi.org/project/scvi-tools)
[![Documentation Status](https://readthedocs.org/projects/scvi/badge/?version=latest)](https://scvi.readthedocs.io/en/stable/?badge=stable)
![Build
Status](https://github.com/scverse/scvi-tools/workflows/scvi-tools/badge.svg)
[![Coverage](https://codecov.io/gh/scverse/scvi-tools/branch/master/graph/badge.svg)](https://codecov.io/gh/YosefLab/scvi-tools)
[![Code
Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Downloads](https://pepy.tech/badge/scvi-tools)](https://pepy.tech/project/scvi-tools)
[![Project chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://scverse.zulipchat.com/)

[scvi-tools](https://scvi-tools.org/) (single-cell variational inference
tools) is a package for probabilistic modeling and analysis of single-cell omics
data, built on top of [PyTorch](https://pytorch.org) and
[AnnData](https://anndata.readthedocs.io/en/latest/).

# Analysis of single-cell omics data

scvi-tools is composed of models that perform many analysis tasks across single- or multi-omics:

-   Dimensionality reduction
-   Data integration
-   Automated annotation
-   Factor analysis
-   Doublet detection
-   Spatial deconvolution
-   and more!

In the [user guide](https://docs.scvi-tools.org/en/stable/user_guide/index.html), we provide an overview of each model.
All model implementations have a high-level API that interacts with
[scanpy](http://scanpy.readthedocs.io/) and includes standard save/load functions, GPU acceleration, etc.

# Rapid development of novel probabilistic models

scvi-tools contains the building blocks to develop and deploy novel probablistic
models. These building blocks are powered by popular probabilistic and
machine learning frameworks such as [PyTorch
Lightning](https://www.pytorchlightning.ai/) and
[Pyro](https://pyro.ai/). For an overview of how the scvi-tools package
is structured, you may refer to [this](https://docs.scvi-tools.org/en/stable/user_guide/background/codebase_overview.html) page.

We recommend checking out the [skeleton
repository](https://github.com/scverse/simple-scvi) as a
starting point for developing and deploying new models with scvi-tools.

# Basic installation [Abstra]
Clone repository

```
git clone https://github.com/owkin/deepdeconv-fork # https
git clone git@github.com:owkin/deepdeconv-fork.git # ssh
```

Create an environment and install scvi-tools locally

```
conda create -n deepdeconv python=3.9
conda activate deepdeconv
cd deepdeconv-fork
# install library in editable mode
pip install -e ".[dev,docs,tutorials]"
# Install additional requirements
pip install -r requirements.txt
```

To confirm that scvi-tools was succesfully installed

```
pip show scvi-tools
```

Create an ipykernel so you can use your environment with a Jupyter notebook


```
`python -m ipykernel install --user --name=deepdeconv`
```

Create a branch for local development

```
git checkout -b {your-branch-name}
```

I you want to use a GPU, make sure to create a workspace with a GPU in *Abstra*.
Please be sure to install a version of [PyTorch](https://pytorch.org/) that is compatible with your GPU (if applicable).

# MixUpVI and deconvolution benchmark

MixUpVI is a model derived from scVI that creates a latent space in which the assumptions of NNLS are inherently met, thus allowing for latent deconvolution. Different resources are:
- `scvi.model._mixupvi.py`: The MixUpVI model.
- `run_mixupvi.py`: The script to train and/or tune the MixUpVI model.
- `run_benchmark.py`: The deconvolution benchmarking pipeline, where one can compare any added deconvolution model on bulk / simulated pseudobulk tasks.

To run the benchmark, one should:
- Create their own `config.yaml` config file within the `benchmark_configs` folder.
- Every possible config parameter is explained within the `@dataclass` of the `RunBenchmarkConfig` inside `run_benchmark_config_dataclass.py` in which value compatibilities of the different provided arguments are checked.
- To run a benchmark with the created config, run inside a terminal `python /home/owkin/deepdeconv-fork/run_benchmark.py --config /home/owkin/deepdeconv-fork/benchmark_configs.config.yaml`.
- If inside the config, `save: True`, then every output of the experiment will be saved inside `/home/owkin/project/run_benchmark_experiments/{your_experiment_name}`. Otherwise, only final plots will be saved there.

# Resources

-   Tutorials, API reference, and installation guides are available in
    the [documentation](https://docs.scvi-tools.org/).
-   For discussion of usage, check out our
    [forum](https://discourse.scvi-tools.org).
-   Please use the [issues](https://github.com/scverse/scvi-tools/issues) to submit bug reports.
-   If you\'d like to contribute, check out our [contributing
    guide](https://docs.scvi-tools.org/en/stable/contributing/index.html).
-   If you find a model useful for your research, please consider citing
    the corresponding publication (linked above).

# Reference

If you use `scvi-tools` in your work, please cite

> **A Python library for probabilistic analysis of single-cell omics data**
>
> Adam Gayoso, Romain Lopez, Galen Xing, Pierre Boyeau, Valeh Valiollah Pour Amiri, Justin Hong, Katherine Wu, Michael Jayasuriya, Edouard Mehlman, Maxime Langevin, Yining Liu, Jules Samaran, Gabriel Misrachi, Achille Nazaret, Oscar Clivio, Chenling Xu, Tal Ashuach, Mariano Gabitto, Mohammad Lotfollahi, Valentine Svensson, Eduardo da Veiga Beltrame, Vitalii Kleshchevnikov, Carlos Talavera-López, Lior Pachter, Fabian J. Theis, Aaron Streets, Michael I. Jordan, Jeffrey Regier & Nir Yosef
>
> _Nature Biotechnology_ 2022 Feb 07. doi: [10.1038/s41587-021-01206-w](https://doi.org/10.1038/s41587-021-01206-w).

along with the publicaton describing the model used.

You can cite the scverse publication as follows:

> **The scverse project provides a computational ecosystem for single-cell omics data analysis**
>
> Isaac Virshup, Danila Bredikhin, Lukas Heumos, Giovanni Palla, Gregor Sturm, Adam Gayoso, Ilia Kats, Mikaela Koutrouli, Scverse Community, Bonnie Berger, Dana Pe’er, Aviv Regev, Sarah A. Teichmann, Francesca Finotello, F. Alexander Wolf, Nir Yosef, Oliver Stegle & Fabian J. Theis
>
> _Nature Biotechnology_ 2022 Apr 10. doi: [10.1038/s41587-023-01733-8](https://doi.org/10.1038/s41587-023-01733-8).
