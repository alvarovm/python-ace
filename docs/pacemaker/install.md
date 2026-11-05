# Installation

Create a separate environment; for example:

```bash
mamba create -n ace python=3.12 pip
mamba activate ace
```

You can use `conda` instead of `mamba`.

## Install pacemaker

```bash
pip install --upgrade pip
pip install --upgrade python-ace
```

Wheels are available for Linux, Apple Silicon, and Intel Macs.

## Add the PyTorch fitting backend

Fitting (`backend: {evaluator: torchpot}`) uses PyTorch through the separate
`torchpotential` package. Install PyTorch first (pick the build matching your
platform and CUDA driver on <https://pytorch.org>), then `torchpotential`:

```bash
pip install torch
pip install ./TorchPotential     # path of the torchpotential sources
```

All fitting computations are done in double precision. The `pyace` evaluator
(`backend: {evaluator: pyace}`) does not need PyTorch.

## Check the installation

```bash
pip check
pacemaker --help
```

On Linux, GPU fitting requires a compatible NVIDIA driver and a CUDA build of
PyTorch (`python -c "import torch; print(torch.cuda.is_available())"`). To use
the CPU, set `CUDA_VISIBLE_DEVICES=-1` (or `backend: {gpu_config: {gpu_ind: -1}}`)
before starting pacemaker.

Continue with the [quick start](quickstart.md) to prepare data and run a fit.
