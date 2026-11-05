# Benchmarks

Performance benchmarks of `pacemaker` fits, mainly to measure how the `torchpot` backend scales over several GPUs
(`backend: {gpu_config: {gpu_ind: [0, 1, ...]}}`). They are not part of the test suite; datasets and results are
git-ignored (`datasets/` and `results/` only keep their `.gitignore`).

## bench_fit.py

End-to-end benchmark: a fixed number of BFGS / L-BFGS-B / CG iterations through the real `pacemaker` CLI.

```bash
# in the environment where pacemaker and torchpotential are installed (ace-torch)
python benchmarks/bench_fit.py --dataset benchmarks/datasets/mydata.extxyz --gpus 0 --batch-size 1000 --iterations 20
python benchmarks/bench_fit.py --dataset benchmarks/datasets/mydata.extxyz --gpus 0,1,2,3 --batch-size 250 --repeats 3
python benchmarks/bench_fit.py --dataset mydata.extxyz --gpus -1 --num-threads 8      # CPU
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

### result.json

- `dataset`: path, sha256, counts (structures, atoms, optionally pairs with `--count-pairs`), elements, replication;
- `potential`: hash of the potential section, `rcut`; `backend`: devices, ranks, batch size, threads, optimizer;
- `environment`: host, load average, GPUs (`nvidia-smi`), torch / CUDA, versions and git commits of pyace and
  torchpotential, `OMP_*` / `TORCHPOTENTIAL_*` variables; `busy_warnings` if the host or a GPU was loaded at the start;
- per run: return code, `wall_seconds` (whole process), `fit_seconds` (pacemaker's "Fitting took"), the number of
  iterations and evaluations, `eval_seconds` (median / mean / min / max / p90 of the per-evaluation time of the
  accepted iterates after the first `--warmup-evals`), `first_eval_seconds`, `seconds_per_iteration`,
  `seconds_per_evaluation_average`, `batches_per_device`, the last row of `metrics.txt` and every iteration record.

Reading the numbers: the evaluation of one batch launches ~9000 GPU kernels whatever its size, so below roughly
30000 atoms per batch it is limited by Python / kernel launches, not by the GPU; use at least 4 batches per GPU. Compare
`final_metrics.loss` between runs with different device counts: it must agree to rounding (summation order).
The machine is often shared: check `busy_warnings` and repeat runs (`--repeats`) before drawing conclusions.
