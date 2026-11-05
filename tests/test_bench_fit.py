"""Functional tests of benchmarks/bench_fit.py (input adapter, log parsing and a short end-to-end run)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import bench_fit  # noqa: E402


def make_frames(n=6, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n):
        at = bulk("Al", "fcc", a=4.05, cubic=True)
        at.rattle(0.05, seed=int(rng.integers(1000)))
        at.calc = SinglePointCalculator(at, energy=-3.0 * len(at) + 0.1 * rng.normal(),
                                        forces=0.1 * rng.normal(size=(len(at), 3)))
        frames.append(at)
    return frames


@pytest.fixture
def extxyz(tmp_path):
    path = tmp_path / "small.extxyz"
    write(str(path), make_frames(), format="extxyz")
    return path


def test_extxyz_is_converted_to_the_pacemaker_dataframe(extxyz):
    frames, n_read, n_used = bench_fit.load_frames(extxyz, max_structures=4, replicate=2)
    assert (n_read, n_used, len(frames)) == (6, 4, 8)
    df = bench_fit.frames_to_dataframe(frames, name="small")
    assert list(df.columns) == ["ase_atoms", "energy", "forces", "name", "energy_corrected"]
    assert len(df) == 8 and df["name"].is_unique
    assert all(f.shape == (len(a), 3) for f, a in zip(df["forces"], df["ase_atoms"]))
    assert all(a.calc is None for a in df["ase_atoms"])


def test_custom_keys_and_missing_data_are_reported(tmp_path):
    frames = make_frames(2)
    for at in frames:
        at.info["my_energy"] = -12.5
        at.arrays["my_forces"] = np.zeros((len(at), 3))
        at.calc = None
    df = bench_fit.frames_to_dataframe(frames, "my_energy", "my_forces")
    assert (df["energy"] == -12.5).all()
    with pytest.raises(ValueError, match="cannot read the energy"):
        bench_fit.frames_to_dataframe(frames)
    with pytest.raises(ValueError, match="cannot read the forces"):
        bench_fit.frames_to_dataframe(frames, "my_energy")


def test_gpus_option():
    assert bench_fit.parse_gpus("0") == (0, [0])
    assert bench_fit.parse_gpus("0, 1,2") == ([0, 1, 2], [0, 1, 2])
    assert bench_fit.parse_gpus("-1") == (-1, [-1])
    for bad in ("0,-1", "-2", ""):
        with pytest.raises(ValueError):
            bench_fit.parse_gpus(bad)


def test_log_parsing():
    log = "\n".join([
        "2026/11/04 11:00:00 I - Iteration: #0     (1 evals):  Loss: 0.500000 | RMSE Energy(low): 120.00 (110.00) meV/at"
        " | Forces(low): 400.00 (390.00) meV/A | Time/eval: 100.00 mcs/at",
        "2026/11/04 11:00:01 I - Iteration: #1     (3 evals):  Loss: 0.250000 | RMSE Energy(low): 80.50 (70.25) meV/at"
        " | Forces(low): 300.00 (290.00) meV/A | Time/eval:  50.00 mcs/at",
        "2026/11/04 11:00:01 I - Batches per device: [3, 2, 1]",
        "unrelated line",
    ])
    records, shards = bench_fit.parse_log(log, n_atoms=1000)
    assert shards == [3, 2, 1]
    assert [r["iteration"] for r in records] == [0, 1] and [r["evals"] for r in records] == [1, 3]
    assert records[1]["eval_seconds"] == pytest.approx(0.05)
    assert records[1]["rmse_forces_meV_per_A"] == 300.0
    summary = bench_fit.summarize_run(records, fit_seconds=2.0, warmup_evals=1)
    assert summary["eval_seconds"]["median"] == pytest.approx(0.05)
    assert summary["seconds_per_iteration"] == 2.0 and summary["evaluations"] == 3
    assert bench_fit.parse_fit_seconds("x\nFitting took   12.50 seconds\n") == 12.5


def test_generated_input_overrides_the_template(extxyz):
    from ruamel.yaml import YAML

    args = bench_fit.parse_args(["--dataset", str(extxyz), "--optimizer", "L-BFGS-B", "--iterations", "7",
                                 "--batch-size", "25", "--num-threads", "2"])
    template = YAML(typ="safe").load(open(bench_fit.CONFIGS_DIR / "benchmark.yaml"))
    inp = bench_fit.build_input(template, ["Al"], "data.pckl.gzip", args, [0, 1])
    assert inp["potential"]["elements"] == ["Al"]
    assert inp["fit"]["maxiter"] == 7 and inp["fit"]["options"] == {"gtol": 0, "ftol": 0}
    assert inp["backend"]["gpu_config"] == {"gpu_ind": [0, 1]} and inp["backend"]["batch_size"] == 25
    assert inp["backend"]["num_threads"] == 2 and inp["backend"]["batch_size_reduction"] is False
    assert template["potential"]["elements"] == []  # the template is not modified


@pytest.mark.torchpot
def test_short_end_to_end_run_writes_a_result(extxyz, tmp_path):
    out = tmp_path / "results"
    code = bench_fit.main(["--dataset", str(extxyz), "--gpus", "-1", "--iterations", "3", "--batch-size", "2",
                           "--num-threads", "1", "--output-dir", str(out), "--allow-busy", "--count-pairs"])
    assert code == 0
    (run_root,) = list(out.iterdir())
    result = json.loads((run_root / "result.json").read_text())
    assert result["dataset"]["structures"] == 6 and result["dataset"]["pairs"] > 0
    assert len(result["dataset"]["sha256"]) == 64 and len(result["potential"]["sha256"]) == 64
    (run,) = result["runs"]
    assert run["returncode"] == 0 and run["iterations_done"] >= 1 and run["evaluations"] >= run["iterations_done"]
    assert run["eval_seconds"]["median"] > 0 and run["fit_seconds"] > 0
    assert run["final_metrics"]["loss"] > 0
    assert result["backend"]["devices"] == [-1]
