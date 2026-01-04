# FiRE: Finite-range Embeddings

This repo implements Variational Monte Carlo (VMC) with a wavefunction ansatz using finite-range embeddings, as described in https://arxiv.org/abs/2504.06087.

## Installation

We recommand installation using the [uv package manager](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync --frozen     # create a .venv and install all dependencies
. .venv/bin/activate # activate the virtual environment
```

Installation may take ~1 min, because we depend on JAX with CUDA.
We have tested the code with jax==0.4.29 and jax==0.5.3 (default version obtained via uv sync), but other versions may work as well.

## Running the via CLI
To run the code, create a config.yaml file and run the code with

```bash
sparse-wf-run config.yaml
```

This will merge the config options provided in config.yaml with the default options provided in config/default.yaml and optimize the wavefunction.
The code supports multi-GPU usage either via a single process with access to all GPUs or one process per GPU. Multi-node usage may require SLURM.
All output will be written to stdout and we additionally support logging to W&B.
A wavefunction optimization with FiRE should only take a few minutes for very small molecules such as H2, but may take several GPU hours for larger molecules.
To compute properties for the molelcules investigated in the paper, set 'molecule_args.database_args.hash' to the corresponding hash in data/geometries.json.

## SLURM dispatch

To run hyperparameter sweeps and dispatch computations to SLURM you can use
```bash
sparse-wf-setup -i config.yaml -p model_args.embedding.new.cutoff 3 5 7
```
This will merge config/default.yaml, the input config.yaml and and overrides for any parameter sweep specified with -p. In this example it will create 3 separate runs, with the cutoff radius set to 3, 5, and 7 respectively.
Dispatch to SLURM may require adding details of your SLURM cluster to ``src/sparse_wf/setup_calculations.py`` and adding a SLURM job template to ``config/slurm_templates``

## Config options

All config options and their default values can be found in ``default/config.yaml``.
Of particular interest are:

- ``molecule_args.database_args.name``: Specify the molecule to be calculated by its name in the database of molecular geometries data/geometries.json. Alternatively you can also specify the molecule by its unique hash or comment.
- ``molecule_args.method``: Switch to "from_str" to specify the molecule as a psycf molecule string (with units in bohr) via the ``molecule_args.from_str.atom option
``
- ``model_args.embedding.new``: All options related to the finite-range embedding architecture

## Citing FiRE
FiRE is joint work by Michael Scherbela and Nicholas Gao. Please cite it as

```bibtex
@misc{scherbela2025accurateabinitioneuralnetworksolutions,
      title={Accurate Ab-initio Neural-network Solutions to Large-Scale Electronic Structure Problems},
      author={Michael Scherbela and Nicholas Gao and Philipp Grohs and Stephan Günnemann},
      year={2025},
      eprint={2504.06087},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2504.06087},
}
```













