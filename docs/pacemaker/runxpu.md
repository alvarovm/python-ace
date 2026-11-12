# Running pacemaker on Intel XPU

The `torchpot` backend supports single- and multi-XPU fits on FP64-capable Intel GPUs, including Intel Data Center
GPU Max 1550. CUDA is preferred during auto-detection, followed by XPU and then CPU.

## Environment on Aurora

The workspace's `env.sh` loads the site PyTorch build and activates the virtual environment:

```bash
module add frameworks/2026.1.0
source ~/soft/ace/myace/bin/activate
```

Use a virtual environment created with `--system-site-packages` so the module's XPU-enabled PyTorch remains visible.
Install matching `distributed` versions of **both** python-ace and TorchPotential; the worker implementation lives
in TorchPotential, not python-ace:

```bash
source env.sh
pip install ./python-ace --force-reinstall --no-deps
pip install ./TorchPotential --force-reinstall --no-deps
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
```

On an allocated Aurora node, `frameworks/2026.1.0` exposes 12 XPU devices with
`ZE_FLAT_DEVICE_HIERARCHY=FLAT` (two tiles per physical GPU). Check device visibility and FP64 support rather than
assuming a count:

```bash
python -c "import torch; print(torch.xpu.get_device_properties(0)); print(torch.ones(4, dtype=torch.float64, device='xpu:0').sum())"
```

**Tile pairing pitfall:** torch enumerates the two tiles of each physical card as *consecutive* indices
(`xpu:0`/`xpu:1` are one card, `xpu:2`/`xpu:3` the next, ...). A device list such as `[0, 1, 2, 3, 4, 5]` looks
like 6 GPUs but is really both tiles of only the first 3 physical cards - the other physical GPUs are never
selected and stay idle. To spread a fit over *N* distinct physical GPUs, pick one tile per card, e.g.
`[0, 2, 4, 6, 8, 10]` for 6 GPUs; to use every tile of every selected card (2 worker processes per card), select
them all, e.g. `[0, 1, 2, ..., 11]`. `get_devices` logs a warning when some devices double up on a physical card
while another card visible to the process is left completely unused (not when every visible tile is selected).

## Select one or several XPUs

The device selection order is:

1. `TORCHPOTENTIAL_DEVICES`, a comma-separated device list.
2. `TORCHPOTENTIAL_DEVICE`, a single device.
3. `backend.gpu_config.gpu_ind`, an integer or list of non-negative indices; `-1` means CPU.
4. Auto-detection: `cuda:0`, then `xpu:0`, then CPU. Auto-detection alone uses only one GPU.

For two *distinct physical* XPUs on a host without CUDA (one tile per card, indices 0 and 2):

```yaml
backend:
  evaluator: torchpot
  batch_size: 10
  num_threads: 4
  gpu_config: {gpu_ind: [0, 2], mem_limit: 32000}
```

Or override device selection through the environment:

```bash
TORCHPOTENTIAL_DEVICES=xpu:0,xpu:2 pacemaker input.yaml
```

The log should report `Fitting on 2 devices: ['xpu:0', 'xpu:2']` and `Batches per device: [...]`.
Launch one `pacemaker` process on the allocated node; it creates one worker for every additional device. No MPI,
DDP or distributed communication backend is needed. This implementation does not span multiple nodes.

`mem_limit` is in MB and applies to each process's XPU caching allocator, including the main process and when an
environment variable selects the devices. It is not a reservation of physical GPU memory. Worker XPU memory is
released after a fit and during out-of-memory retries. Repeating an index runs multiple processes on the same tile.

For a single XPU use `gpu_ind: 0` or `TORCHPOTENTIAL_DEVICE=xpu:0`. To force CPU, unset
`TORCHPOTENTIAL_DEVICES` and set `TORCHPOTENTIAL_DEVICE=cpu`; `CUDA_VISIBLE_DEVICES=-1` alone does not disable XPU.

## Launch inside an Aurora allocation

Run the following on an allocated compute node, not a login node. `env.sh` is specific to this workspace;
adjust the environment and fit-directory paths if you installed elsewhere:

```bash
cd ~/soft/ace
source env.sh
cd /path/to/fit-directory   # contains input.yaml and the referenced training data
TORCHPOTENTIAL_DEVICES=xpu:0,xpu:2 \
TORCHPOTENTIAL_NUM_THREADS=4 \
pacemaker input.yaml
```

The explicit device list overrides an inherited `TORCHPOTENTIAL_DEVICE` as well as the YAML indices. CPU thread
limits apply **per process**: two devices with four threads each need capacity for at least eight fitting threads,
plus any other work in the job. No `mpirun` or `torchrun` is needed. If a site launcher is required to enter the
compute node, launch one main process with access to all selected devices and enough CPU cores for its workers.

### Use all 12 visible tiles

Only use the following when **all 12 tiles are allocated to your job**. Replace the two-device `gpu_config` in
the backend section with:

```yaml
backend:
  evaluator: torchpot
  batch_size: 10
  num_threads: 4
  gpu_config:
    gpu_ind: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    mem_limit: 32000
```

Unset device environment overrides before relying on that YAML:

```bash
unset TORCHPOTENTIAL_DEVICE TORCHPOTENTIAL_DEVICES
pacemaker input.yaml
```

On an XPU-only node the startup log should list 12 `xpu` devices and 11 workers. With `batch_size: 10`,
480 structures give 48 batches (four per tile on average); choose the batch size for your dataset rather than
assuming this example is optimal. A small Cu-I run is useful for correctness checks but is not a scaling benchmark.
Using all tiles does not combine their memory into one allocation: each process must fit its potential and batches
within its own limit. See the [multi-GPU tutorial](multi_gpu.md) for a single-device comparison and scaling commands.

## Verify that all 12 devices execute work

Inside an allocation that owns all 12 XPUs, run the opt-in node test from the TorchPotential checkout:

```bash
cd ~/soft/ace
source env.sh
cd TorchPotential
TORCHPOTENTIAL_RUN_12_XPU_TEST=1 \
TORCHPOTENTIAL_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/test_xpu_node.py -v -s
```

This is a correctness test, not a performance benchmark. It performs a short BFGS fit on 48 equal-cost synthetic
batches, using one main process and 11 workers, with exactly four batches per XPU. It checks:

- 12 distinct process IDs and assignments `xpu:0` through `xpu:11`, with no idle workers;
- actual model, input-batch, loss, gradient, energy and force tensor devices inside **each** process, plus nonzero
  XPU allocator usage, rather than only the configured device list;
- a nonempty loss/gradient contribution from every rank and each training batch evaluated exactly once;
- fit results, merged loss, gradients and residuals against a CPU reference, and worker shutdown after the test.

With `-s`, pytest prints a row for each rank showing its PID, device, batch count, allocated memory and output
tensor devices. The test uses a test-only device-auditing entry point around the production worker loop; it does
not replace the fit computation or the XPU kernels. Audit JSON files are saved in pytest's temporary directory.
It skips during normal test runs unless `TORCHPOTENTIAL_RUN_12_XPU_TEST=1` is set; once explicitly enabled,
fewer than 12 visible XPUs is a failure, not a skip.

## Performance and troubleshooting

- Use only device indices visible to and allocated to your job. If XPU is unavailable, check the module, drivers,
  `ONEAPI_DEVICE_SELECTOR` and `ZE_FLAT_DEVICE_HIERARCHY`.
- **Some physical GPUs stay idle (low `xpu-smi` memory/utilisation) although enough devices were requested:**
  almost always the tile-pairing pitfall above - consecutive indices on a `ZE_FLAT_DEVICE_HIERARCHY=FLAT` node
  are tiles of the same card. Check the log for `... double up on ... physical GPU(s) ... not selected at all
  ...` and switch to one tile per card (every other index).
- Limit CPU threads per process with `num_threads` or `TORCHPOTENTIAL_NUM_THREADS` (the environment wins).
- Use enough batches to keep each tile busy; at least four per device is a useful starting point. Small examples
  such as Cu-I validate correctness but need not get faster with several GPUs.
- `BFGS`, `L-BFGS-B`, `CG` and `dual_annealing` support several devices. Other optimizers are rejected.
- With `batch_size_reduction: true` (default), XPU out-of-memory errors in the main process or a worker trigger
  the usual smaller-batch retry. Reduce `batch_size` if the next attempt also runs out of memory.
- `TORCHPOTENTIAL_WORKER_TIMEOUT` sets a worker response timeout in seconds.

See the [multi-GPU tutorial](multi_gpu.md) for data preparation, CPU comparison, benchmarking and the Python API.
