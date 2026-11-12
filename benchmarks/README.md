# Benchmarks

Performance benchmarks of `pacemaker` fits, mainly to measure how the `torchpot` backend scales over several GPUs
(`backend: {gpu_config: {gpu_ind: [0, 1, ...]}}`). They are not part of the test suite; datasets and results are
git-ignored (`datasets/` and `results/` only keep their `.gitignore`).

## bench_fit.py

End-to-end benchmark: a fixed number of BFGS / L-BFGS-B / CG iterations through the real `pacemaker` CLI.

```bash
# in the environment where pacemaker and torchpotential are installed (ace-torch)
python benchmarks/bench_fit.py --dataset benchmarks/datasets/mydata.extxyz --gpus 0 --batch-size 250 --iterations 20 --num-threads 4
python benchmarks/bench_fit.py --dataset benchmarks/datasets/mydata.extxyz --gpus 0,1 --batch-size 250 --iterations 20 --num-threads 4 --repeats 3
python benchmarks/bench_fit.py --dataset benchmarks/datasets/mydata.extxyz --gpus all --batch-size 250 --iterations 20 --num-threads 4
python benchmarks/bench_fit.py --dataset mydata.extxyz --gpus -1 --num-threads 8      # CPU
python benchmarks/bench_fit.py --dataset benchmarks/datasets/mydata.extxyz --gpus 0,1 --device xpu --batch-size 250 --iterations 20 --num-threads 4  # XPU
```

Dataset: any file readable by `ase.io.read` (extxyz, ...) with energies and forces, either attached as a calculator
result (the usual extxyz layout) or in `Atoms.info` / `Atoms.arrays` selected with `--energy-key` / `--forces-key`.
`--max-structures N` uses the first N structures, `--replicate K` repeats them K times to reach a larger problem
(throughput only: replicated data is not independent, and is flagged in `result.json`). Energies are used as they are
unless `--reference-energy` is given (passed to `data::reference_energy`).

What it does:

1. converts the dataset into the dataframe pickle that pacemaker reads, and writes `input.yaml` from
   `configs/benchmark.yaml` (`--template` replaces it; `potential.elements`, `data`, `fit` and `backend` are filled in);
2. runs `pacemaker` once per `--repeats` in `results/<label>_<gpus>_<time>/run_<k>/`;
3. parses the pacemaker log and writes `result.json`.

`--gpus` becomes `backend::gpu_config::gpu_ind`. A list runs one worker process per extra GPU; repeating an index
(`0,0`) runs several processes on one GPU, which already helps when evaluation is launch-bound. The environment variables
`TORCHPOTENTIAL_DEVICE(S)` are removed for the run so that `--gpus` decides. `batch_size_reduction` is off, so an
out-of-memory error fails the benchmark instead of silently changing the batch size.
`--device auto` (default) selects CUDA when available, otherwise Intel XPU. `--device xpu` (or `cuda`) forces the
accelerator even when both are visible: the benchmark sets `TORCHPOTENTIAL_DEVICES=xpu:0,xpu:1,...` for the run, and
stops if that device is not available. `--gpus all` discovers all visible devices on the chosen accelerator.
On Aurora, activate `env.sh` first and choose only devices allocated to the job; `all` is safe only when every
visible device is yours. GPU indices refer to visible tiles when `ZE_FLAT_DEVICE_HIERARCHY=FLAT`.
Run one benchmark driver, not one per GPU.

### result.json

- `dataset`: path, sha256, counts (structures, atoms, optionally pairs with `--count-pairs`), elements, replication;
- `potential`: hash of the potential section, `rcut`; `backend`: devices, ranks, batch size, threads, optimizer;
- `environment`: host, load average, NVIDIA utilization (`nvidia-smi`), torch / CUDA, XPU device names and memory,
  versions and git commits of pyace and torchpotential, thread/device environment variables;
  `busy_warnings` if the host or an NVIDIA GPU was loaded at the start (XPU utilization is not sampled);
- per run: return code, `wall_seconds` (whole process), `fit_seconds` (pacemaker's "Fitting took"), the number of
  iterations and evaluations, `eval_seconds` (median / mean / min / max / p90 of the per-evaluation time of the
  accepted iterates after the first `--warmup-evals`), `first_eval_seconds`, `seconds_per_iteration`,
  `seconds_per_evaluation_average`, `batches_per_device`, the last row of `metrics.txt` and every iteration record.

Reading the numbers: small batches can be limited by Python/kernel launches and process communication rather than
GPU throughput. Use at least 4 batches per GPU as a starting point, then measure on the target hardware.
Compare `final_metrics.loss` between runs with different device counts: it should agree to rounding (summation order).
Keep `--batch-size`, dataset, potential, iterations, optimizer and per-process thread count the same across runs;
compare `eval_seconds.median`, excluding startup/warm-up. Changing the batch size also changes the workload,
so it is not a pure device-count comparison. See the [multi-GPU tutorial](../docs/pacemaker/multi_gpu.md).
The machine is often shared: check `busy_warnings` and repeat runs (`--repeats`) before drawing conclusions.
