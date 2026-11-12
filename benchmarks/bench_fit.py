#!/usr/bin/env python
"""End-to-end fit benchmark: a fixed number of optimizer iterations through the real ``pacemaker`` CLI.

The dataset is any ASE-readable file (extxyz, ...). It is converted to the dataframe pickle that pacemaker reads, a
pacemaker ``input.yaml`` is generated from ``configs/benchmark.yaml`` and ``pacemaker`` is run once per repeat, so the
timings include exactly what users get (data preparation, backend setup, the optimizer, metric callbacks). Results,
logs and the converted dataset go to a new directory under ``benchmarks/results`` (git-ignored); ``result.json``
holds the environment, the dataset / potential hashes and the timings.

Example::

    python benchmarks/bench_fit.py --dataset data.extxyz --gpus 0 --batch-size 1000 --iterations 20
    python benchmarks/bench_fit.py --dataset data.extxyz --gpus 0,1,2,3 --batch-size 250 --repeats 3
    python benchmarks/bench_fit.py --dataset data.extxyz --gpus 0,1 --device xpu --batch-size 250 --iterations 20

Timings per evaluation are read from the pacemaker log ("Time/eval") for the accepted iterates; the first
``--warmup-evals`` of them are excluded from the statistics (device/index set-up happens there). ``fit_seconds`` is
pacemaker's own "Fitting took" and covers the whole optimisation, including line searches and callbacks.
"""
import argparse
import copy
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONFIGS_DIR, RESULTS_DIR, busy_warnings, environment_info, gpu_status,  # noqa: E402
                    sha256_file, sha256_json, write_json)

ACCELERATORS = ["cuda", "xpu"]

# options that make scipy run exactly --iterations iterations instead of stopping at its tolerances
FIXED_ITERATION_OPTIONS = {
    "BFGS": {"gtol": 0},
    "CG": {"gtol": 0},
    "L-BFGS-B": {"gtol": 0, "ftol": 0},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="ASE-readable file with energies and forces (e.g. extxyz)")
    p.add_argument("--dataset-format", default=None, help="format passed to ase.io.read (default: from extension)")
    p.add_argument("--energy-key", default=None,
                   help="Atoms.info key of the energy (default: attached calculator, else info['energy'])")
    p.add_argument("--forces-key", default=None,
                   help="Atoms.arrays key of the forces (default: attached calculator, else arrays['forces'])")
    p.add_argument("--reference-energy", default=None,
                   help="pacemaker data::reference_energy (number, 'auto', a preset name or a JSON dict); "
                        "default: the energy is used as it is")
    p.add_argument("--template", default=str(CONFIGS_DIR / "benchmark.yaml"),
                   help="pacemaker input template (default: configs/benchmark.yaml)")
    p.add_argument("--optimizer", default="BFGS", choices=sorted(FIXED_ITERATION_OPTIONS) + ["dual_annealing"])
    p.add_argument("--iterations", type=int, default=10, help="optimizer iterations (fit::maxiter)")
    p.add_argument("--batch-size", type=int, default=100, help="structures per batch (backend::batch_size)")
    p.add_argument("--gpus", default="0",
                   help="comma-separated GPU indices (several = multi-GPU fit), 'all', or -1 for the CPU")
    p.add_argument("--device", default="auto", choices=["auto"] + ACCELERATORS,
                   help="accelerator for --gpus: auto (CUDA when visible, else XPU), cuda or xpu")
    p.add_argument("--num-threads", type=int, default=None, help="CPU threads per process (also OMP_NUM_THREADS)")
    p.add_argument("--warmup-evals", type=int, default=1,
                   help="first evaluations excluded from the per-evaluation statistics")
    p.add_argument("--repeats", type=int, default=1, help="number of independent pacemaker runs")
    p.add_argument("--max-structures", type=int, default=None, help="use only the first N structures")
    p.add_argument("--replicate", type=int, default=1,
                   help="repeat the structures K times to scale the problem (throughput tests only: not new data)")
    p.add_argument("--count-pairs", action="store_true", help="also count neighbour pairs within the potential cutoff")
    p.add_argument("--label", default=None, help="name used in the output directory (default: dataset name)")
    p.add_argument("--output-dir", default=str(RESULTS_DIR), help="parent of the run directory")
    p.add_argument("--timeout", type=float, default=None, help="seconds allowed for one pacemaker run")
    p.add_argument("--prepare-only", action="store_true", help="write dataset and input.yaml, do not run pacemaker")
    p.add_argument("--allow-busy", action="store_true", help="do not warn about a busy host / GPU")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------------
def load_frames(path, fmt=None, max_structures=None, replicate=1):
    from ase.io import read

    frames = read(str(path), index=":", format=fmt)
    if not isinstance(frames, list):
        frames = [frames]
    n_read = len(frames)
    if max_structures is not None:
        frames = frames[:max_structures]
    if replicate < 1:
        raise ValueError("--replicate must be >= 1")
    used = len(frames)
    # deepcopy (not Atoms.copy) keeps the attached calculator that holds the energy and forces
    frames = [copy.deepcopy(at) for _ in range(replicate) for at in frames]
    return frames, n_read, used


def _energy(atoms, key, index):
    try:
        if key is not None:
            return float(atoms.info[key])
        if atoms.calc is not None:
            return float(atoms.get_potential_energy())
        return float(atoms.info["energy"])
    except Exception as exc:
        raise ValueError("structure #{}: cannot read the energy ({}: {}). Use --energy-key".format(
            index, type(exc).__name__, exc)) from exc


def _forces(atoms, key, index):
    try:
        if key is not None:
            f = atoms.arrays[key]
        elif atoms.calc is not None:
            f = atoms.get_forces()
        else:
            f = atoms.arrays["forces"]
    except Exception as exc:
        raise ValueError("structure #{}: cannot read the forces ({}: {}). Use --forces-key".format(
            index, type(exc).__name__, exc)) from exc
    f = np.asarray(f, dtype=float)
    if f.shape != (len(atoms), 3) or not np.all(np.isfinite(f)):
        raise ValueError("structure #{}: forces must be a finite [natoms, 3] array, got {}".format(index, f.shape))
    return f


def frames_to_dataframe(frames, energy_key=None, forces_key=None, name="benchmark", reference_energy=None):
    """Dataframe in the format read by pacemaker (``ase_atoms``, ``energy``, ``forces``, ``name``)."""
    import pandas as pd

    rows = []
    for i, atoms in enumerate(frames):
        e = _energy(atoms, energy_key, i)
        if not math.isfinite(e):
            raise ValueError("structure #{}: the energy is not finite".format(i))
        clean = atoms.copy()
        clean.calc = None
        rows.append(dict(ase_atoms=clean, energy=e, forces=_forces(atoms, forces_key, i), name="{}:{}".format(name, i)))
    df = pd.DataFrame(rows)
    if reference_energy is None:
        df["energy_corrected"] = df["energy"]
    return df


def count_pairs(frames, cutoff):
    from ase.neighborlist import neighbor_list

    return int(sum(len(neighbor_list("i", at, cutoff)) for at in frames))


# ----------------------------------------------------------------------------------------------------
# pacemaker input
# ----------------------------------------------------------------------------------------------------
def parse_gpus(text, device="auto"):
    """``(gpu_ind for the backend config, list of requested indices)``; ``-1`` is the CPU.

    ``device`` is ``auto`` (CUDA when visible, otherwise XPU), ``cuda`` or ``xpu``; an explicit device must be available.
    """
    text = str(text).strip().lower()
    if device != "auto" and device not in ACCELERATORS:
        raise ValueError("invalid --device {!r}".format(device))
    if device != "auto":
        import torch

        if not getattr(torch, device).is_available():
            raise ValueError("--device {}: {} is not available on this host".format(device, device.upper()))
    if text == "all":
        from torchpotential.utils.device import _available_accelerator
        import torch

        accelerator = device if device != "auto" else _available_accelerator()
        idx = list(range(getattr(torch, accelerator).device_count())) if accelerator else []
        if not idx:
            raise ValueError("--gpus all: no CUDA or XPU device is visible")
    else:
        idx = [int(t) for t in text.split(",") if t.strip() != ""]
    if not idx or (len(idx) > 1 and min(idx) < 0) or min(idx) < -1:
        raise ValueError("invalid --gpus {!r}".format(text))
    return (idx[0] if len(idx) == 1 else idx), idx


def build_input(template, elements, dataset_path, args, gpu_ind):
    inp = json.loads(json.dumps(template))  # plain deep copy
    inp.setdefault("potential", {})["elements"] = list(elements)
    data = {"filename": str(Path(dataset_path).resolve())}
    if args.reference_energy is not None:
        try:
            data["reference_energy"] = json.loads(args.reference_energy)
        except ValueError:
            data["reference_energy"] = args.reference_energy
    inp["data"] = data
    fit = inp.setdefault("fit", {})
    fit["optimizer"] = args.optimizer
    fit["maxiter"] = args.iterations
    if args.optimizer in FIXED_ITERATION_OPTIONS:
        fit["options"] = dict(FIXED_ITERATION_OPTIONS[args.optimizer])
    backend = inp.setdefault("backend", {})
    backend.update(evaluator="torchpot", batch_size=args.batch_size, display_step=10 ** 9,
                   batch_size_reduction=False, gpu_config={"gpu_ind": gpu_ind})
    if args.num_threads:
        backend["num_threads"] = args.num_threads
    return inp


def potential_cutoff(inp):
    """Largest ``rcut`` of the bonds section (None if it is not given explicitly)."""
    try:
        return max(float(b["rcut"]) for b in inp["potential"]["bonds"].values())
    except Exception:
        return None


# ----------------------------------------------------------------------------------------------------
# pacemaker output
# ----------------------------------------------------------------------------------------------------
_ITER = re.compile(r"#(\d+)\s+\((\d+) evals\):\s+Loss:\s+(\S+)")
_TIME = re.compile(r"Time/eval:\s+(\S+)\s+mcs/at")
_RMSE_E = re.compile(r"RMSE Energy\(low\):\s*(\S+)\s+\((\S+)\)")
_RMSE_F = re.compile(r"Forces\(low\):\s*(\S+)\s+\((\S+)\)")
_SHARDS = re.compile(r"Batches per device:\s*(\[[^\]]*\])")
_FIT_TOOK = re.compile(r"Fitting took\s+([\d.]+)\s+seconds")


def parse_log(log_text, n_atoms):
    """Per-iteration records of the accepted iterates: iteration, cumulative evaluations, loss, RMSEs and the
    time of the evaluation in seconds (pacemaker logs it in microseconds per atom)."""
    records = []
    for line in log_text.splitlines():
        m, t = _ITER.search(line), _TIME.search(line)
        if not (m and t):
            continue
        rec = dict(iteration=int(m.group(1)), evals=int(m.group(2)), loss=float(m.group(3)),
                   eval_seconds=float(t.group(1)) * 1e-6 * n_atoms)
        e, f = _RMSE_E.search(line), _RMSE_F.search(line)
        if e:
            rec["rmse_energy_meV_per_atom"] = float(e.group(1))
        if f:
            rec["rmse_forces_meV_per_A"] = float(f.group(1))
        records.append(rec)
    shards = None
    for line in log_text.splitlines():
        s = _SHARDS.search(line)
        if s:
            shards = json.loads(s.group(1))
    return records, shards


def parse_fit_seconds(stdout_text):
    values = [float(v) for v in _FIT_TOOK.findall(stdout_text)]
    return sum(values) if values else None


def last_metrics_row(path):
    """The last row of pacemaker's metrics.txt as a dict (column name -> float)."""
    try:
        lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
        names, values = lines[0].split(), lines[-1].split()
        return {n: float(v) for n, v in zip(names, values)}
    except Exception:
        return None


def stats(values):
    if not values:
        return None
    v = sorted(values)
    return dict(n=len(v), median=statistics.median(v), mean=statistics.fmean(v), min=v[0], max=v[-1],
                p90=v[min(len(v) - 1, int(math.ceil(0.9 * len(v))) - 1)],
                stdev=statistics.stdev(v) if len(v) > 1 else 0.0)


def summarize_run(records, fit_seconds, warmup_evals):
    times = [r["eval_seconds"] for r in records]
    kept = times[warmup_evals:] or times[-1:]
    iterations = max((r["iteration"] for r in records), default=0)
    evals = max((r["evals"] for r in records), default=0)
    out = dict(
        iterations_done=iterations,
        evaluations=evals,
        eval_seconds=stats(kept),
        first_eval_seconds=times[0] if times else None,
        fit_seconds=fit_seconds,
        seconds_per_iteration=(fit_seconds / iterations) if fit_seconds and iterations else None,
        seconds_per_evaluation_average=(fit_seconds / evals) if fit_seconds and evals else None,
    )
    if len(times) <= warmup_evals:
        out["note"] = "fewer logged evaluations than --warmup-evals + 1: statistics include the first evaluation"
    return out


# ----------------------------------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------------------------------
def run_pacemaker(run_dir, args, gpu_list):
    env = dict(os.environ)
    for var in ("TORCHPOTENTIAL_DEVICE", "TORCHPOTENTIAL_DEVICES"):
        env.pop(var, None)  # the backend config (--gpus) must decide
    if args.device != "auto" and gpu_list != [-1]:
        env["TORCHPOTENTIAL_DEVICES"] = ",".join("{}:{}".format(args.device, i) for i in gpu_list)
    if args.num_threads:
        env["OMP_NUM_THREADS"] = str(args.num_threads)
    code = ("import sys; from pyace.cli.pacemaker import main; sys.argv[0] = 'pacemaker'; "
            "main(['input.yaml', '-o', 'output_potential.yaml', '--no-predict'])")
    start = time.time()
    with open(run_dir / "stdout.txt", "w") as out:
        try:
            proc = subprocess.run([sys.executable, "-c", code], cwd=run_dir, env=env, stdout=out,
                                  stderr=subprocess.STDOUT, timeout=args.timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = "timeout"
    return returncode, time.time() - start


def main(argv=None):
    args = parse_args(argv)
    from ruamel.yaml import YAML

    dataset = Path(args.dataset)
    if not dataset.is_file():
        sys.exit("dataset not found: {}".format(dataset))
    gpu_ind, gpu_list = parse_gpus(args.gpus, args.device)

    yaml = YAML(typ="safe")
    yaml.default_flow_style = False
    with open(args.template) as f:
        template = yaml.load(f)

    frames, n_read, n_used = load_frames(dataset, args.dataset_format, args.max_structures, args.replicate)
    elements = sorted({s for at in frames for s in at.get_chemical_symbols()})
    n_atoms = int(sum(len(at) for at in frames))
    df = frames_to_dataframe(frames, args.energy_key, args.forces_key, name=dataset.stem,
                             reference_energy=args.reference_energy)
    inp = build_input(template, elements, "dataset.pckl.gzip", args, gpu_ind)

    label = args.label or dataset.stem
    gpus_tag = "cpu" if gpu_list == [-1] else "gpu" + "-".join(str(g) for g in gpu_list)
    if args.device != "auto" and gpu_list != [-1]:
        gpus_tag = args.device + "-".join(str(g) for g in gpu_list)
    base = Path(args.output_dir) / "{}_{}_{}".format(label, gpus_tag, time.strftime("%Y%m%d-%H%M%S"))
    base.mkdir(parents=True, exist_ok=False)
    dataset_pickle = base / "dataset.pckl.gzip"
    df.to_pickle(dataset_pickle, compression="gzip", protocol=4)
    inp["data"]["filename"] = str(dataset_pickle.resolve())
    with open(base / "input.yaml", "w") as f:
        yaml.dump(inp, f)

    cutoff = potential_cutoff(inp)
    result = dict(
        schema=1, label=label, command=sys.argv, args=vars(args), output_dir=str(base),
        dataset=dict(path=str(dataset.resolve()), sha256=sha256_file(dataset), format=args.dataset_format,
                     structures_in_file=n_read, structures_used=n_used, replicate=args.replicate,
                     structures=len(frames), atoms=n_atoms, mean_atoms=n_atoms / max(len(frames), 1),
                     elements=elements, pairs=count_pairs(frames, cutoff) if args.count_pairs and cutoff else None,
                     replicated_data_is_not_independent=args.replicate > 1),
        potential=dict(sha256=sha256_json(inp["potential"]), rcut=cutoff, template=str(args.template)),
        backend=dict(gpu_ind=gpu_ind, devices=gpu_list, device=args.device, ranks=len(gpu_list), batch_size=args.batch_size,
                     num_threads=args.num_threads, optimizer=args.optimizer, iterations=args.iterations,
                     warmup_evals=args.warmup_evals),
        environment=environment_info(),
        runs=[],
    )
    result["busy_warnings"] = [] if args.allow_busy else busy_warnings(result["environment"])
    for w in result["busy_warnings"]:
        print("WARNING: {} - timings may be unreliable".format(w), file=sys.stderr)
    write_json(base / "result.json", result)
    if args.prepare_only:
        print("Prepared {}".format(base))
        return 0

    failed = False
    for k in range(args.repeats):
        run_dir = base / "run_{}".format(k)
        run_dir.mkdir()
        (run_dir / "input.yaml").write_text((base / "input.yaml").read_text())
        gpus_before = gpu_status()
        returncode, wall = run_pacemaker(run_dir, args, gpu_list)
        run = dict(run_dir=str(run_dir), returncode=returncode, wall_seconds=wall, gpus_before=gpus_before,
                   gpus_after=gpu_status())
        stdout_text = (run_dir / "stdout.txt").read_text(errors="replace")
        log_path = run_dir / "log.txt"
        log_text = log_path.read_text(errors="replace") if log_path.is_file() else ""
        if returncode != 0:
            failed = True
            run["stdout_tail"] = stdout_text.splitlines()[-15:]
            print("run {} failed ({}); see {}".format(k, returncode, run_dir / "stdout.txt"), file=sys.stderr)
        else:
            records, shards = parse_log(log_text, n_atoms)
            run.update(summarize_run(records, parse_fit_seconds(stdout_text), args.warmup_evals))
            run["batches_per_device"] = shards
            run["final_metrics"] = last_metrics_row(run_dir / "metrics.txt")
            run["iterations"] = records
            if not records:
                run["note"] = "no per-iteration lines found in log.txt"
        result["runs"].append(run)
        write_json(base / "result.json", result)

    ok = [r for r in result["runs"] if r["returncode"] == 0 and r.get("eval_seconds")]
    if ok:
        result["summary"] = dict(
            runs=len(ok),
            median_eval_seconds=statistics.median(r["eval_seconds"]["median"] for r in ok),
            median_fit_seconds=statistics.median(r["fit_seconds"] for r in ok if r["fit_seconds"])
            if any(r["fit_seconds"] for r in ok) else None,
            median_wall_seconds=statistics.median(r["wall_seconds"] for r in ok),
        )
        s = result["summary"]
        print("{} | {} structures, {} atoms | devices {} | batch_size {}".format(
            label, len(frames), n_atoms, gpu_list, args.batch_size))
        print("eval: {:.4f} s (median)   fit: {} s   wall: {:.1f} s   -> {}".format(
            s["median_eval_seconds"],
            "{:.2f}".format(s["median_fit_seconds"]) if s["median_fit_seconds"] else "n/a",
            s["median_wall_seconds"], base / "result.json"))
    write_json(base / "result.json", result)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
