### cell2fate

cell2fate models differentiation based on single-cell data. At this point we can infer latent time, transcription/splicing/degredation rates and RNAvelocity within one lineage of cells. We will add new versions soon that infer: \
1.) multiple lineages \
2.) modules of genes that are activated together (i.e. change rates simultaneously) \
3.) the effect transcription factors have on module activation probabilities.

## Usage and Tutorials

A tutorial notebook is available here:

## Installation

We suggest using a separate conda environment for installing cell2fate.

Create a conda environment and install the `cell2fate` package

```bash
conda create -y -n cell2fate_env python=3.9

conda activate cell2fate_env
pip install git+https://github.com/AlexanderAivazidis/cell2fate
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
