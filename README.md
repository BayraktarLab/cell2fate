### cell2fate

## Usage and Tutorials

The currently recommended workflow using the currently best performing model is in this tutorial [here](https://github.com/AlexanderAivazidis/cell2fate/blob/main/notebooks/cell2fate_tutorial_pancreas.ipynb). \
Benchmarking of all models is done in a seperate repository: https://github.com/AlexanderAivazidis/fate_benchmarking

## Installation

We suggest using a separate conda environment for installing cell2fate.

Create a conda environment and install the `cell2fate` package.

```bash
conda create -y -n cell2fate_env python=3.9

conda activate cell2fate_env
pip install git+https://github.com/AlexanderAivazidis/cell2fate
```

cell2fate also needs cell2location installed:

```bash
pip install git+https://github.com/BayraktarLab/cell2location.git#egg=cell2location[tutorials]
```

Finally, to use this environment in a jupyter notebook, add a jupyter kernel for this environment:

```bash
conda activate cell2fate_env
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
