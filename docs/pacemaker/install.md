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
platform and CUDA/XPU driver on <https://pytorch.org>), then `torchpotential`:

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

On Linux, GPU fitting requires either a compatible NVIDIA driver and CUDA-enabled PyTorch, or an Intel driver
and XPU-enabled PyTorch. The GPU must support double precision (FP64). Check the relevant backend:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import torch; print('XPU:', hasattr(torch, 'xpu') and torch.xpu.is_available()); print('XPU count:', torch.xpu.device_count() if hasattr(torch, 'xpu') else 0)"
```

For multi-GPU fitting, install compatible versions of **both** repositories. In this fork the multi-device/XPU
implementation lives on `distributed` in python-ace and TorchPotential; installing only one side is insufficient.
After selecting the matching source branches, install from their parent directory into the same environment:

```bash
pip install ./python-ace --force-reinstall --no-deps
pip install ./TorchPotential --force-reinstall --no-deps
```

These source-update commands assume the runtime dependencies above are already installed. Use a non-editable
python-ace install: its C extensions must be built and importable. On Aurora, load the site's frameworks module
and activate the environment first; see [Running on XPU](runxpu.md).

To force CPU, unset `TORCHPOTENTIAL_DEVICES` and set `TORCHPOTENTIAL_DEVICE=cpu`. A YAML
`backend: {gpu_config: {gpu_ind: -1}}` also selects CPU unless a device environment variable overrides it.
`CUDA_VISIBLE_DEVICES=-1` alone does not disable XPU.

Continue with the [quick start](quickstart.md) to prepare data and run a fit, or the
[multi-GPU tutorial](multi_gpu.md) to spread training batches over several devices on one node.
