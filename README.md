[![Run tests](https://github.com/BayraktarLab/cell2fate/actions/workflows/run_tests.yml/badge.svg)](https://github.com/BayraktarLab/cell2fate/actions/workflows/run_tests.yml)  [![codecov](https://codecov.io/gh/AlexanderAivazidis/cell2fate/graph/badge.svg?token=CCJTK20MA7)](https://codecov.io/gh/AlexanderAivazidis/cell2fate)
[![Documentation Status](https://readthedocs.org/projects/cell2fate/badge/?version=latest)](https://cell2fate.readthedocs.io/en/latest/?badge=latest)

![alt text](https://github.com/BayraktarLab/cell2fate/blob/main/cell2fate_diagram.png?raw=true)

## Usage and Tutorials

Please find our documentation and tutorials [here](https://cell2fate.readthedocs.io/en/latest/).

## Publication figures

Results from all datasets in the [cell2fate preprint](https://www.biorxiv.org/content/10.1101/2023.08.03.551650v1.full.pdf) can be reproduced with [these noteobooks](https://github.com/AlexanderAivazidis/cell2fate_notebooks).

## Installation

We suggest using a separate conda environment for installing cell2fate.

Create a conda environment and install the `cell2fate` package.

```bash
conda create -y -n cell2fate_env python=3.9

conda activate cell2fate_env
pip install git+https://github.com/BayraktarLab/cell2fate
```

To use this environment in a jupyter notebook, add a jupyter kernel for this environment:

```bash
conda activate cell2fate_env
pip install ipykernel
python -m ipykernel install --user --name=cell2fate_env --display-name='Environment (cell2fate_env)'
```

If you do not have conda please install Miniconda first:

```bash
cd /path/to/software
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# use prefix /path/to/software/miniconda3
```

Before installing cell2fate and it's dependencies, it could be necessary to make sure that you are creating a fully isolated conda environment by telling python to NOT use user site for installing packages, ideally by adding this line to your `~/.bashrc` file , but this would also work during a terminal session:

```bash
export PYTHONNOUSERSITE="someletters"
```
