# Tutorial: fitting on several GPUs

This tutorial runs a `pacemaker` fit on several GPUs of one machine, checks that the result is the same as on a single
GPU, and measures the speed-up. It supports NVIDIA CUDA and Intel XPU through the `torchpot` evaluator;
the `pyace` evaluator is CPU-only. This is **single-node** fitting, not multi-node DDP/MPI training.

## Quick setup for an existing input

Merge the following into your `input.yaml`, preserving the other fit settings:

```yaml
fit:
  optimizer: BFGS
backend:
  evaluator: torchpot
  batch_size: 10
  num_threads: 4
  gpu_config: {gpu_ind: [0, 1], mem_limit: 0}
```

Unset `TORCHPOTENTIAL_DEVICE` and `TORCHPOTENTIAL_DEVICES` to use these YAML indices, then run `pacemaker input.yaml`.
The first available backend is used: CUDA, then XPU. To choose the backend explicitly, run **one** of:

```bash
TORCHPOTENTIAL_DEVICES=cuda:0,cuda:1 pacemaker input.yaml  # NVIDIA
TORCHPOTENTIAL_DEVICES=xpu:0,xpu:1 pacemaker input.yaml    # Intel
```

Start one `pacemaker` process on the allocated node; it launches its own workers. Do not start one copy per GPU
with `mpirun`, `torchrun` or a multi-task scheduler command. Both packages must have compatible multi-device support;
see [Installation](install.md) and [Running on XPU](runxpu.md).

**How it works.** The training set is cut into batches (`backend::batch_size` structures each), exactly as for a
single GPU. `pacemaker` keeps the optimizer (scipy `BFGS`, `L-BFGS-B`, ...) in the main process, starts one worker
process for every additional GPU and gives every GPU its share of the batches. At every evaluation of the loss only the
coefficients, the summed gradient and the residuals are exchanged. The batches are the same as in a single-GPU fit, so
loss, regularisation and metrics agree with it up to rounding errors.

## 0. When is it worth it?

Evaluating one batch launches a fixed number of GPU kernels (about 9000), whatever the batch size. Below roughly
30 000 atoms per batch the evaluation is limited by these launches and not by the GPU in the CUDA measurements.
The threshold depends on the hardware; measure XPU scaling on your own dataset. Therefore:

 - several GPUs help for **large training sets** (hundreds of thousands of atoms or more): increase the work per GPU
   and use at least **4 batches per GPU**;
 - for small training sets one GPU (or even the CPU) is as fast or faster: the processes and the exchange per
   evaluation cost more than they save.

## 1. Requirements

 - Linux (the worker processes use inherited sockets; Windows is not supported for multi-GPU fits).
 - Compatible multi-device versions of python-ace and `torchpotential` >= 0.2.0 installed in the same environment.
   For this fork's XPU-enabled implementation, use `distributed` in both repositories.
 - A CUDA or XPU build of PyTorch that supports the target GPUs and double-precision computation.
 - Two or more NVIDIA GPUs or FP64-capable Intel XPUs visible to the process. For NVIDIA, check:

```bash
nvidia-smi -L
python -c "import torch, torchpotential; print(torch.cuda.device_count(), 'GPUs, torchpotential', torchpotential.__version__)"
```

GPU indices in the configuration are the indices *seen by the process*: with `CUDA_VISIBLE_DEVICES=2,3` the two
GPUs are `0` and `1`.

### Intel XPU (Aurora)

Activate the environment described in [Running on XPU](runxpu.md), then check the visible devices:

```bash
source env.sh
python -c "import torch; print(torch.xpu.is_available(), torch.xpu.device_count())"
```

`gpu_config: {gpu_ind: [0, 2]}` selects two *distinct physical* XPUs when CUDA is unavailable. To explicitly select
XPU even on a host with CUDA, or override the YAML device list:

```bash
TORCHPOTENTIAL_DEVICES=xpu:0,xpu:2 TORCHPOTENTIAL_NUM_THREADS=4 pacemaker input.yaml
```

On Aurora with `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, the visible indices refer to GPU tiles (two per physical card,
enumerated consecutively); a node can expose 12 XPU devices across 6 physical GPUs. **`gpu_ind: [0, 1]` is *not*
two distinct GPUs - it is both tiles of the same card**; use every other index (`[0, 2, 4, ...]`) to spread a fit
over that many distinct physical GPUs, or list every tile to use all of them with 2 worker processes per card. The
log warns (`... double up on ... physical GPU(s) ... not selected at all ...`) when some devices double up on a
card while another visible card is left completely idle. Use only indices allocated to your job. Launch **one**
`pacemaker` process on the node: it starts its own workers, so do not launch a copy per tile with MPI. This is
single-node data parallelism, not multi-node fitting. The
[12-device verification test](runxpu.md#verify_that_all_12_devices_execute_work) checks tensor placement and
actual batch evaluation on every tile of a fully allocated Aurora node.

## 2. Prepare the data

`pacemaker` reads a pickled pandas dataframe with the columns `ase_atoms`, `energy` (or `energy_corrected`) and
`forces` (see [Input file](inputfile.md)). If your reference data are in an extxyz file (or any other format readable
by ASE), convert it with this script (`extxyz2df.py`):

```python
import sys
import pandas as pd
from ase.io import read

frames = read(sys.argv[1], index=":")
df = pd.DataFrame({
    "ase_atoms": [at.copy() for at in frames],
    "energy": [at.get_potential_energy() for at in frames],
    "forces": [at.get_forces() for at in frames],
})
df["energy_corrected"] = df["energy"]      # or use `data: {reference_energy: ...}` in input.yaml
df.to_pickle(sys.argv[2], compression="gzip", protocol=4)
print(len(df), "structures,", sum(map(len, frames)), "atoms")
```

```bash
python extxyz2df.py my_data.extxyz data.pckl.gzip
```

The example below uses 1000 ethanol structures (9000 atoms) converted in this way.

## 3. Write the input file

`input.yaml` is a normal `pacemaker` input. Only the `backend` section is specific to multi-GPU fits:

```yaml
seed: 42

potential:
  deltaSplineBins: 0.001
  elements: ['C', 'H', 'O']
  embeddings:
    ALL: {npot: 'FinnisSinclairShiftedScaled', fs_parameters: [1, 1, 1, 0.5], ndensity: 2}
  bonds:
    ALL: {radbase: SBessel, radparameters: [5.25], rcut: 5.0, dcut: 0.01, NameOfCutoffFunction: cos}
  functions:
    number_of_functions_per_element: 50
    UNARY:   {nradmax_by_orders: [15, 6, 4, 3, 2, 2], lmax_by_orders: [0, 3, 3, 2, 2, 1]}
    BINARY:  {nradmax_by_orders: [15, 6, 3, 2, 2, 1], lmax_by_orders: [0, 3, 2, 1, 1, 0]}
    TERNARY: {nradmax_by_orders: [15, 3, 3, 2, 1],    lmax_by_orders: [0, 2, 2, 1, 1]}
    ALL:     {nradmax_by_orders: [15, 3, 2, 1, 1],    lmax_by_orders: [0, 2, 2, 1, 1]}

data:
  filename: data.pckl.gzip

fit:
  loss: {kappa: 0.3, L1_coeffs: 1e-8, L2_coeffs: 1e-8}
  optimizer: BFGS            # BFGS, L-BFGS-B, CG or dual_annealing
  maxiter: 10

backend:
  evaluator: torchpot
  batch_size: 50             # structures per batch
  num_threads: 4             # CPU threads *per process*
  gpu_config: {gpu_ind: [0, 1, 2, 3]}   # one process per entry
```

Settings that matter:

 - **`gpu_config: {gpu_ind: [...]}`**: the GPUs to use. The first entry is used by the main process (it also runs the
   optimizer and the metrics), every other entry starts a worker. `mem_limit` (MB) caps each process's CUDA/XPU
   allocator, including the main process. `0` means no cap; it does not reserve memory. With repeated indices,
   each process gets its own cap, so their combined allocation can exceed it. A single integer keeps the usual
   single-GPU fit. The test set is evaluated on the first device only; leave extra memory there.
 - **`batch_size`**: choose it so that there are **at least 4 batches per GPU**, i.e.
   `batch_size <= n_structures / (4 * n_gpus)`, and as large as the GPU memory allows. `pacemaker` prints
   `Batches per device: [...]` and warns when there are too few batches. The batches are not split or merged, so the
   fit does not depend on the number of GPUs.
 - **`num_threads`**: every process uses its own CPU threads. By default the cores are divided by the number of GPUs;
   under a batch system (Slurm, ...) set it explicitly, because the number of cores allowed to the job is not detected.
   `TORCHPOTENTIAL_NUM_THREADS` overrides the setting.
 - **`optimizer`**: `BFGS`, `L-BFGS-B`, `CG` and `dual_annealing` work. `Adam`, `SGD`, `trf`, `dogbox` and `lm` are
    rejected with several GPUs.

### Device-selection precedence

| Priority | Setting | Example |
|---|---|---|
| 1 | `TORCHPOTENTIAL_DEVICES` | `xpu:0,xpu:1` or `cuda:0,cuda:1` |
| 2 | `TORCHPOTENTIAL_DEVICE` | `cpu`, `xpu:0` or `cuda:0` (one process) |
| 3 | `backend.gpu_config.gpu_ind` | `[0, 1]`, `0`, or `-1` for CPU |
| 4 | Default | One device: `cuda:0`, else `xpu:0`, else CPU |

The plural environment variable wins even if the singular one is also set. Environment selection changes the device
list, not the YAML `mem_limit`: the cap still applies to every process on the selected CUDA/XPU devices.
An explicit GPU-index list fails if no accelerator is available; it does not silently become several CPU processes.

Check selection before starting a fit, without allocating a potential:

```bash
python - <<'PY'
from torchpotential.utils.utilities import get_devices
print([str(d) for d in get_devices({"gpu_ind": [0, 1]})])
PY
```

Use the same indices as your input file. This check honors both device environment variables, so it also catches
an inherited `TORCHPOTENTIAL_DEVICE=cpu` that would override a YAML GPU list.

## 4. Run

```bash
pacemaker input.yaml
```

At the start of the fit the log shows the devices and the share of every one:

```
INFO - Fitting on 4 devices: ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']
INFO - Started 3 worker process(es) on devices ['cuda:1', 'cuda:2', 'cuda:3']
INFO - Batch size: 50
INFO - Batches per device: [5, 5, 5, 5]
```

In a second terminal `nvidia-smi` shows one `python` process on every listed GPU while the fit runs. The rest of the
output (iterations, metrics, `output_potential.yaml`, ...) is the same as for a single GPU. The worker processes are
stopped when `pacemaker` ends, also after an error.
On Intel GPUs the device log uses `xpu:0`, `xpu:1`, etc.; use `xpu-smi` rather than `nvidia-smi` to monitor hardware.

## 5. Check the result against a single GPU

A multi-GPU fit should reproduce a single-GPU fit up to floating-point rounding. Keep the seed, optimizer, batch size
and training data identical. Use separate working directories: `pacemaker` writes metrics and intermediate potentials
there even if you customize the final output filename.

For the `input.yaml` / `data.pckl.gzip` example above, compare two XPUs with one XPU:

```bash
for mode in single multi; do
    mkdir -p "runs/$mode"
    cp input.yaml data.pckl.gzip "runs/$mode/"
done
(
    cd runs/single
    TORCHPOTENTIAL_DEVICES=xpu:0 pacemaker input.yaml
)
(
    cd runs/multi
    TORCHPOTENTIAL_DEVICES=xpu:0,xpu:2 pacemaker input.yaml   # xpu:0,xpu:1 would be the same physical card
)
grep -E 'Iteration:?[[:space:]]+#10[[:space:]]' runs/single/log.txt runs/multi/log.txt
```

Replace `xpu` with `cuda` for NVIDIA. Copy any other input files, or use absolute paths, if your setup differs from
this example. The example runs 10 iterations; adjust the `grep` pattern if your optimizer stops earlier.
Compare losses and validation errors rather than requiring identical coefficient files: summation order differs,
and BFGS may amplify small rounding differences over many iterations. Use the same hardware for scaling measurements.

## 6. Try it without several GPUs

 - **Several processes on one GPU:** repeat the index, `gpu_ind: [0, 0]`. Useful to test the set-up; whether it is
   faster depends on the workload and hardware. This does not combine the memory of multiple GPUs.
 - **CPU only:** `TORCHPOTENTIAL_DEVICES=cpu,cpu pacemaker input.yaml` runs two CPU processes. The environment variable
    `TORCHPOTENTIAL_DEVICES` (comma-separated, e.g. `cuda:0,cuda:1` or `xpu:0,xpu:1`) overrides `gpu_config` and
    `TORCHPOTENTIAL_DEVICE`.

## 7. Measure the scaling

`benchmarks/bench_fit.py` (in the python-ace repository) runs a fixed number of iterations through `pacemaker` and
records the time per evaluation, the environment and the final loss in a `result.json`:

```bash
python benchmarks/bench_fit.py --dataset my_data.extxyz --batch-size 250 --iterations 20 --gpus 0
python benchmarks/bench_fit.py --dataset my_data.extxyz --batch-size 250 --iterations 20 --gpus 0,1
python benchmarks/bench_fit.py --dataset my_data.extxyz --batch-size 250 --iterations 20 --gpus 0,1,2,3 --repeats 3
```

`--gpus` uses CUDA when available, otherwise XPU. `--gpus all` discovers all visible devices on that backend;
use it only if all of them belong to your allocation. The benchmark removes `TORCHPOTENTIAL_DEVICE` and
`TORCHPOTENTIAL_DEVICES` from its fit subprocess, so those variables cannot override `--gpus`.

Compare `eval_seconds.median` (time of one evaluation of the whole training set) between the runs, and check that
`final_metrics.loss` is the same. The speed-up of N GPUs is `eval(1 GPU) / eval(N GPUs)`. Keep the total number of
batches the same (same `--batch-size`) when comparing. See `benchmarks/README.md` for all options.

## 8. Python API

Without `pacemaker`, the same can be done with `torchpotential`:

```python
from torchpotential.distributed import DevicePool
from torchpotential.fit import FitTorchPotential
from torchpotential.potentials.ace import ACE
from torchpotential.torchpot import TorchPotential
from torchpotential.utils.device import set_num_threads
from torchpotential.utils.utilities import get_devices, init_gpu_config

# bbasis_config: pyace BBasisConfiguration; loss_specs: loss dictionary (see the TorchPotential README and tests);
# df: fitting dataframe with the `tp_atoms` column
config = init_gpu_config({"gpu_ind": [0, 1], "mem_limit": 0})
devices = get_devices(config)                 # CUDA/XPU indices, or an environment override
num_threads = set_num_threads(4)              # honors TORCHPOTENTIAL_NUM_THREADS
pool = DevicePool(devices, num_threads=num_threads, mem_limit=config["mem_limit"]) if len(devices) > 1 else None
try:
    potential = ACE(bbasis_config, device=devices[0])
    fitter = FitTorchPotential(TorchPotential(potential, loss_specs=loss_specs), pool=pool)
    fitter.fit(df, optimizer="BFGS", batch_size=250, niter=50)
finally:
    if pool is not None:
        pool.close()
```

`init_gpu_config` initializes rank 0 and its memory cap; `DevicePool` applies the same cap in the workers. The
`finally` block stops workers if the fit fails. `pacemaker` handles this lifecycle automatically.

## Example: Slurm job (one node, four GPUs)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# input.yaml: gpu_config: {gpu_ind: [0, 1, 2, 3]}, num_threads: 4
unset TORCHPOTENTIAL_DEVICE TORCHPOTENTIAL_DEVICES
pacemaker input.yaml
```

`gpu_ind` refers to the GPUs allocated to the job (they are numbered from 0). Set `num_threads` to
`cpus-per-task / number of GPUs`.
For an existing Aurora allocation, follow the [XPU launch example](runxpu.md#launch_inside_an_aurora_allocation)
instead of the Slurm directives. If your site requires a launcher, launch one main process with access to all
allocated devices and enough CPU cores for its workers, not one rank per GPU.

## Troubleshooting

 - **`Only N batches for M devices` warning / no speed-up:** decrease `batch_size` (at least 4 batches per GPU), and
   check section 0: small training sets do not profit from more GPUs.
 - **Some physical GPUs stay idle (low `xpu-smi` memory/utilisation) although `Batches per device` looks balanced:**
   on a multi-tile Intel XPU node (`ZE_FLAT_DEVICE_HIERARCHY=FLAT`), each physical card is exposed as two
   *consecutive* indices (`xpu:0`/`xpu:1` are one card, `xpu:2`/`xpu:3` the next, ...); a list such as
   `[0, 1, 2, 3, 4, 5]` only spans 3 physical GPUs, not 6. Pick one tile per card (every other index) to spread the
   fit over that many distinct GPUs; using every tile of every selected card (e.g. all 12 on a fully allocated
   node) is fine too. The log warns (`... double up on ... physical GPU(s) ... not selected at all ...`) when some
   devices double up on a card while another card visible to the process is left completely unused.
 - **Out of memory:** with `batch_size_reduction: True` (default) the batch size is reduced automatically and the fit
   restarts from the last potential, like for a single GPU. The memory of the main-process GPU is the largest, because
   it also holds the test set; set `mem_limit` or a smaller `batch_size` if needed.
 - **`worker rank k died` / `failed during evaluation`:** a worker crashed or raised an error; its message and traceback
   are printed above. The fit stops; no process is left running. `TORCHPOTENTIAL_WORKER_TIMEOUT=<seconds>` bounds the
   waiting time for a worker that is alive but not answering.
  - **Only one device is used although `gpu_ind` is a list:** the environment variable `TORCHPOTENTIAL_DEVICE` is set
    (it forces a single device unless `TORCHPOTENTIAL_DEVICES` overrides it); unset it or supply the plural list.
    GPU indices must be smaller than the visible device count. Clear both variables to rely on YAML.
  - **CPU was requested but GPUs are still used:** an inherited `TORCHPOTENTIAL_DEVICES` overrides
    `TORCHPOTENTIAL_DEVICE=cpu` and `gpu_ind: -1`. Unset it. `CUDA_VISIBLE_DEVICES=-1` does not hide XPUs.
  - **CUDA is selected when XPU was intended:** numeric YAML indices prefer CUDA; use an explicit
    `TORCHPOTENTIAL_DEVICES=xpu:0,xpu:2` list.
 - **`NotImplementedError: Optimizer ... is not supported with several devices`:** use `BFGS`, `L-BFGS-B`, `CG` or
   `dual_annealing`.
 - **The CPU is overloaded:** reduce `num_threads`; every process uses that many threads.
