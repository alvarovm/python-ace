# Tutorial: fitting on several GPUs

This tutorial runs a `pacemaker` fit on several GPUs of one machine, checks that the result is the same as on a single
GPU, and measures the speed-up. It uses the `torchpot` evaluator; the `pyace` evaluator is CPU-only.

**How it works.** The training set is cut into batches (`backend::batch_size` structures each), exactly as for a
single GPU. `pacemaker` keeps the optimizer (scipy `BFGS`, `L-BFGS-B`, ...) in the main process, starts one worker
process for every additional GPU and gives every GPU its share of the batches. At every evaluation of the loss only the
coefficients, the summed gradient and the residuals are exchanged. The batches are the same as in a single-GPU fit, so
loss, regularisation and metrics agree with it up to rounding errors.

## 0. When is it worth it?

Evaluating one batch launches a fixed number of GPU kernels (about 9000), whatever the batch size. Below roughly
30 000 atoms per batch the evaluation is limited by these launches and not by the GPU. Therefore:

 - several GPUs help for **large training sets** (hundreds of thousands of atoms or more): increase the work per GPU
   and use at least **4 batches per GPU**;
 - for small training sets one GPU (or even the CPU) is as fast or faster: the processes and the exchange per
   evaluation cost more than they save.

## 1. Requirements

 - Linux (the worker processes use inherited sockets; Windows is not supported for multi-GPU fits).
 - `pyace` (python-ace) and `torchpotential` >= 0.2.0 installed in the same environment, with a CUDA build of PyTorch.
 - Two or more NVIDIA GPUs visible to the process. Check:

```bash
nvidia-smi -L
python -c "import torch, torchpotential; print(torch.cuda.device_count(), 'GPUs, torchpotential', torchpotential.__version__)"
```

GPU indices in the configuration are the indices *seen by the process*: with `CUDA_VISIBLE_DEVICES=2,3` the two
GPUs are `0` and `1`.

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
   optimizer and the metrics), every other entry starts a worker. `mem_limit` (MB) applies to every GPU. A single
   integer keeps the usual single-GPU fit.
 - **`batch_size`**: choose it so that there are **at least 4 batches per GPU**, i.e.
   `batch_size <= n_structures / (4 * n_gpus)`, and as large as the GPU memory allows. `pacemaker` prints
   `Batches per device: [...]` and warns when there are too few batches. The batches are not split or merged, so the
   fit does not depend on the number of GPUs.
 - **`num_threads`**: every process uses its own CPU threads. By default the cores are divided by the number of GPUs;
   under a batch system (Slurm, ...) set it explicitly, because the number of cores allowed to the job is not detected.
   `TORCHPOTENTIAL_NUM_THREADS` overrides the setting.
 - **`optimizer`**: `BFGS`, `L-BFGS-B`, `CG` and `dual_annealing` work. `Adam`, `SGD`, `trf`, `dogbox` and `lm` are
   rejected with several GPUs.

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

## 5. Check the result against a single GPU

A multi-GPU fit must give the same loss as a single-GPU fit of the same input. Run the fit twice and compare the
iteration lines of `log.txt` (or `metrics.txt`):

```bash
pacemaker input.yaml -l log_multi.txt -o output_multi.yaml
sed 's/gpu_config: .*/gpu_config: {gpu_ind: 0}/' input.yaml > input_1gpu.yaml
pacemaker input_1gpu.yaml -l log_1gpu.txt -o output_1gpu.yaml
grep "Iteration   #10 " log_multi.txt log_1gpu.txt
```

The losses agree to rounding errors (the test suite requires a relative agreement of 1e-9; BFGS may amplify
rounding differences slightly over many iterations). Example: a fit of the ethanol set, 10 iterations, `batch_size: 50`:

| `gpu_ind` | loss after 10 iterations | `Fitting took` |
|---|---|---|
| `0` | 1.2619e+00 | 64.5 s |
| `[0, 0]` (two processes on the same GPU) | 1.2619e+00 | 40.6 s |

(Measured on a shared machine with one GPU; only the equality of the loss is meaningful. Two processes on one GPU
are faster here because the evaluation is limited by kernel launches, see section 0.)

## 6. Try it without several GPUs

 - **Several processes on one GPU:** repeat the index, `gpu_ind: [0, 0]`. Useful to test the set-up, and it speeds up
   fits that are limited by kernel launches.
 - **CPU only:** `TORCHPOTENTIAL_DEVICES=cpu,cpu pacemaker input.yaml` runs two CPU processes. The environment variable
   `TORCHPOTENTIAL_DEVICES` (comma-separated, e.g. `cuda:0,cuda:1`) overrides `gpu_config`.

## 7. Measure the scaling

`benchmarks/bench_fit.py` (in the python-ace repository) runs a fixed number of iterations through `pacemaker` and
records the time per evaluation, the environment and the final loss in a `result.json`:

```bash
python benchmarks/bench_fit.py --dataset my_data.extxyz --batch-size 250 --iterations 20 --gpus 0
python benchmarks/bench_fit.py --dataset my_data.extxyz --batch-size 250 --iterations 20 --gpus 0,1
python benchmarks/bench_fit.py --dataset my_data.extxyz --batch-size 250 --iterations 20 --gpus 0,1,2,3 --repeats 3
```

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
from torchpotential.utils.utilities import get_devices

# bbasis_config: pyace BBasisConfiguration; loss_specs: loss dictionary (see the TorchPotential README and tests);
# df: fitting dataframe with the `tp_atoms` column
devices = get_devices({"gpu_ind": [0, 1]})
pool = DevicePool(devices, num_threads=4)               # starts the workers
fitter = FitTorchPotential(TorchPotential(ACE(bbasis_config, device=devices[0]), loss_specs=loss_specs), pool=pool)
fitter.fit(df, optimizer="BFGS", batch_size=250, niter=50)
pool.close()
```

## Example: Slurm job (one node, four GPUs)

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

# input.yaml: gpu_config: {gpu_ind: [0, 1, 2, 3]}, num_threads: 4
pacemaker input.yaml
```

`gpu_ind` refers to the GPUs allocated to the job (they are numbered from 0). Set `num_threads` to
`cpus-per-task / number of GPUs`.

## Troubleshooting

 - **`Only N batches for M devices` warning / no speed-up:** decrease `batch_size` (at least 4 batches per GPU), and
   check section 0: small training sets do not profit from more GPUs.
 - **Out of memory:** with `batch_size_reduction: True` (default) the batch size is reduced automatically and the fit
   restarts from the last potential, like for a single GPU. The memory of the main-process GPU is the largest, because
   it also holds the test set; set `mem_limit` or a smaller `batch_size` if needed.
 - **`worker rank k died` / `failed during evaluation`:** a worker crashed or raised an error; its message and traceback
   are printed above. The fit stops; no process is left running. `TORCHPOTENTIAL_WORKER_TIMEOUT=<seconds>` bounds the
   waiting time for a worker that is alive but not answering.
 - **Only one device is used although `gpu_ind` is a list:** the environment variable `TORCHPOTENTIAL_DEVICE` is set
   (it forces a single device); unset it. `gpu_ind` larger than the number of visible GPUs is an error.
 - **`NotImplementedError: Optimizer ... is not supported with several devices`:** use `BFGS`, `L-BFGS-B`, `CG` or
   `dual_annealing`.
 - **The CPU is overloaded:** reduce `num_threads`; every process uses that many threads.
