### cell2fate

![alt text](https://github.com/BayraktarLab/cell2fate/blob/main/cell2fate_diagram.png?raw=true)

## Usage and Tutorials

The standard recommended workflow for using cell2fate can be found in this this tutorial [here](https://github.com/BayraktarLab/cell2fate/blob/main/notebooks/publication_figures/cell2fate_PancreasWithCC.ipynb).

To use the cell2fate + cell2location workflow: <br />
1.) Run cell2fate and save modules, as shown in [this notebook](https://github.com/BayraktarLab/cell2fate/blob/main/notebooks/publication_figures/cell2fate_HumanDevelopingBrain.ipynb). <br />
2.) Install cell2location, as explained [here](https://github.com/BayraktarLab/cell2location). <br />
3.) Run cell2location, with the cell2fate modules as input, as shown in [this notebook](https://github.com/BayraktarLab/cell2fate/blob/main/notebooks/publication_figures/cell2location_HumanDevelopingBrain.ipynb). <br />

## Publication figures

Results from all datasets in the [cell2fate preprint](https://www.biorxiv.org/content/10.1101/2023.08.03.551650v1.full.pdf) can be reproduced with [these noteobooks](https://github.com/BayraktarLab/cell2fate/blob/main/notebooks/publication_figures/).

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
