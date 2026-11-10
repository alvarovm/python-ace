# Running pacemaker on Intel XPU (GPU)

This guide describes how to run `pacemaker` with the PyTorch backend (`backend: {evaluator: torchpot}`) on Intel Data Center GPUs (XPU) — e.g. Intel Data Center GPU Max 1550 on Aurora (ALCF). It was validated with `frameworks/2026.1.0` (`torch 2.13.0a0+gitcf30153`, `torch.xpu.is_available()==True`, 12 XPU devices per node) and `examples/Cu-I/input.yaml`.

Related code: `TorchPotential/src/torchpotential/utils/device.py:13` (`get_device()`), `TorchPotential/src/torchpotential/utils/utilities.py:93` (`init_gpu_config()`), `src/pyace/fitadapter.py:178` (`_release_gpu_memory()`).

## 1. Prerequisites

* Intel GPU + Level-Zero driver (`/dev/dri/renderD*` visible, `sycl-ls` shows `level_zero:gpu` devices).
* OneAPI runtime (`oneapi/release/2026.1.0` on Aurora, sets `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`).
* PyTorch with XPU support. On Aurora use the `frameworks` module (includes `torch.xpu`):

```bash
module add frameworks/2026.1.0
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.device_count())"
# expect: 2.13.0a0+... True 12
```

For a local build see https://pytorch.org (Intel XPU wheel) or Intel Extension for PyTorch (IPEX). Verify:

```bash
python -c "import torch; assert torch.xpu.is_available(); a=torch.randn(10,dtype=torch.float64,device='xpu:0'); print(a.sum())"
```

* `python-ace` + `TorchPotential` installed into a venv that inherits system site-packages (so frameworks `torch` is visible):

```bash
python -m venv --system-site-packages ~/soft/ace/myace
source ~/soft/ace/myace/bin/activate
pip install ./python-ace          # or pip install python-ace
pip install ./TorchPotential
pacemaker --help
```

The repository provides `env.sh` for Aurora:

```bash
# env.sh
module add frameworks/2026.1.0
source ~/soft/ace/myace/bin/activate
```

```bash
source env.sh
```

## 2. Device selection

`torchpotential` selects the compute device in order:

1. `TORCHPOTENTIAL_DEVICE` env var (highest priority) — any `torch.device` string, e.g. `cpu`, `xpu:0`, `xpu:1`, `cuda:0`.
2. `backend: gpu_config: {gpu_ind: N}` from `input.yaml` — `N<0` → CPU, `N>=0` → first available accelerator (`cuda` if present else `xpu`). Memory cap `mem_limit` (MB) is optional.
3. Auto-detect — `cuda:0` if `torch.cuda.is_available()`, else `xpu:0` if `torch.xpu.is_available()`, else `cpu`.

The logic lives in `device.py:get_device()` and `utilities.py:init_gpu_config()`. Before the XPU patch auto-detect ignored XPU and fell back to CPU.

### Recommended settings

Default `input.yaml` needs no change — on an XPU node it now auto-selects `xpu:0`:

```yaml
backend:
  evaluator: torchpot
  batch_size: 100
  display_step: 50
```

Explicit XPU device:

```yaml
backend:
  evaluator: torchpot
  batch_size: 100
  display_step: 50
  gpu_config: {gpu_ind: 0}          # 0..11 on Aurora (6 tiles ×2)
  # gpu_config: {gpu_ind: 0, mem_limit: 32000}  # cap in MB
  # num_threads: 4                  # optional, see §4
```

Or via environment (overrides YAML):

```bash
export TORCHPOTENTIAL_DEVICE=xpu:0   # or xpu:1, cpu
export TORCHPOTENTIAL_DEVICE=cpu     # force CPU
```

Multi-GPU is not yet parallelized — one rank uses one XPU tile. For MPI launches bind one rank per tile and set `TORCHPOTENTIAL_DEVICE` per rank.

Check which device will be used without fitting:

```bash
python -c "
import os; os.environ.pop('TORCHPOTENTIAL_DEVICE',None)
import torchpotential.utils.device as d; d._DEVICE=None
from torchpotential.utils.utilities import init_gpu_config
print(init_gpu_config(None), d.get_device())
print(init_gpu_config({'gpu_ind':1}), d.get_device())
"
```

## 3. Running the Cu-I example

```bash
source env.sh
cd examples/Cu-I
pacemaker input.yaml
```

`input.yaml` (Cu, `rcut:7`, `FinnisSinclairShiftedScaled`, `nradmax_by_orders:[15,3,2,2,1]`, `maxiter:1000`, `ladder_step:15`) fits 105 structures (368 atoms) through the ladder scheme. On Aurora:

* XPU (`xpu:0` auto): ~70 µs/atom/eval in early ladders, ~2.5× faster than CPU (~170 µs/atom/eval). Full run completes, `output_potential.yaml` + `log.txt` + `report/` produced. Loss trajectories match CPU within 1e-12 initially; final coefficients differ by <4e-3 due to floating-point reduction order — numerically equivalent.
* CPU: `TORCHPOTENTIAL_DEVICE=cpu pacemaker input.yaml`
* Dry run (no fit): `pacemaker --dry-run input.yaml` or `pacemaker -dr input.yaml`

Quick test (short fit) used for CI:

```bash
mkdir /tmp/xpu_test && cd /tmp/xpu_test
cp $REPO/examples/Cu-I/Cu_df1_A1_A2_A3_EV_elast_phon.pckl.gzip .
cat > input.yaml <<'EOF'
seed: 42
metadata: {comment: quick XPU test}
potential:
  deltaSplineBins: 0.001
  elements: [Cu]
  embeddings: {ALL: {npot: FinnisSinclairShiftedScaled, fs_parameters: [1,1,1,0.5], ndensity: 2}}
  bonds: {ALL: {radbase: ChebExpCos, radparameters: [5.25], rcut: 7, dcut: 0.01, NameOfCutoffFunction: cos}}
  functions: {UNARY: {nradmax_by_orders: [15,3,2,2,1], lmax_by_orders: [0,2,2,1,1]}}
data: {filename: Cu_df1_A1_A2_A3_EV_elast_phon.pckl.gzip}
fit:
  loss: {kappa: 0.35, L1_coeffs: 1e-8, L2_coeffs: 1e-8, w0_rad: 1e-8, w1_rad: 1e-8, w2_rad: 1e-8}
  optimizer: BFGS
  maxiter: 20
  ladder_step: 15
  ladder_type: body_order
backend: {evaluator: torchpot, batch_size: 10, display_step: 5}
EOF
pacemaker input.yaml                          # XPU auto
TORCHPOTENTIAL_DEVICE=cpu pacemaker input.yaml # CPU
```

Expected: both exit 0, create `output_potential.yaml` (14 KB), `Avg. time` reported in log.

## 4. Performance tips

* `backend: batch_size: 10..100` — larger reduces kernel launches but uses more HBM. On OOM `fitadapter.py` automatically retries with `batch_size /= 1.618` (set `batch_size_reduction: true`).
* `backend: num_threads: 4` or `TORCHPOTENTIAL_NUM_THREADS=4` — PyTorch CPU threads for small tensors; default (all cores) is slow on oversubscribed nodes.
* `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` is default in frameworks; to expose all tiles keep `ZE_FLAT_DEVICE_HIERARCHY=FLAT` (frameworks default). Verify with `xpu-smi discovery` and `torch.xpu.device_count()`.
* Double precision is used throughout; XPU FP64 is required (Max 1550 has it). `has_fp64` in `torch.xpu.get_device_properties(0)`.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Falls back to CPU despite XPU | `python -c "import torch; print(torch.xpu.is_available())"` must be `True`. Ensure `module add frameworks/2026.1.0` and venv `--system-site-packages`. |
| `GPU ind 1 is requested, but there are only 1 GPUs` | Check `torch.xpu.device_count()` (12 on Aurora). `ONEAPI_DEVICE_SELECTOR` may filter devices. |
| `out of memory` | `device.py:is_oom_error()` now catches XPU OOM (string match). Lower `batch_size` or set `backend: {gpu_config: {mem_limit: 32000}}` and enable `batch_size_reduction`. |
| Slow `Time/eval` on XPU | Set `num_threads`/`TORCHPOTENTIAL_NUM_THREADS`, ensure `batch_size` not 1. For tiny datasets (e.g. Cu-I, 368 atoms) XPU overhead dominates; larger cells benefit more. |
| `pacemaker: command not found` | `source env.sh` or `source myace/bin/activate`; check `which pacemaker` points to `myace/bin/pacemaker`. |
| Need CPU | `export TORCHPOTENTIAL_DEVICE=cpu` or `backend: {gpu_config: {gpu_ind: -1}}` or `CUDA_VISIBLE_DEVICES=-1` (legacy). |

Verbose device log: `TORCHPOTENTIAL_DEVICE=xpu:0 pacemaker -v input.yaml` or inspect `log.txt` header.

## 6. References

* Installation (CUDA/XPU notes): `pacemaker/install.md:43`
* Quickstart & `CUDA_VISIBLE_DEVICES` selection: `pacemaker/quickstart.md:158`
* XPU implementation: `TorchPotential/src/torchpotential/utils/device.py`, `utilities.py`, `src/pyace/fitadapter.py`
