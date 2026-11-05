"""Helpers shared by the benchmark scripts: environment capture, hashing, JSON output."""
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BENCH_DIR / "results"
CONFIGS_DIR = BENCH_DIR / "configs"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _run(cmd, cwd=None, timeout=30):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_revision(path):
    """Commit hash of the git repository containing ``path`` (``-dirty`` when there are tracked changes)."""
    path = Path(path)
    cwd = path if path.is_dir() else path.parent
    rev = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    if rev is None:
        return None
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd)
    return rev + ("-dirty" if dirty else "")


def gpu_status():
    """One dict per GPU from ``nvidia-smi`` (name, utilisation %, memory MiB); empty if unavailable."""
    out = _run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None  # "[N/A]" on unified-memory systems

        gpus.append(dict(index=int(parts[0]), name=parts[1], utilization_percent=num(parts[2]),
                         memory_used_mib=num(parts[3]), memory_total_mib=num(parts[4])))
    return gpus


def environment_info():
    info = dict(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        machine=platform.machine(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        load_average=list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        gpus=gpu_status(),
        env={k: v for k, v in os.environ.items()
             if k.startswith(("OMP_", "MKL_", "TORCHPOTENTIAL_", "CUDA_VISIBLE"))},
    )
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:
        info["torch"] = "unavailable: {}".format(exc)
    try:
        import pyace
        info["pyace"] = pyace.__version__
        info["pyace_git"] = git_revision(Path(pyace.__file__).resolve().parent)
    except Exception as exc:
        info["pyace"] = "unavailable: {}".format(exc)
    try:
        import torchpotential
        info["torchpotential"] = torchpotential.__version__
        info["torchpotential_git"] = git_revision(Path(torchpotential.__file__).resolve().parent)
    except Exception as exc:
        info["torchpotential"] = "unavailable: {}".format(exc)
    return info


def busy_warnings(info):
    """Reasons why timings on this machine may be unreliable (shared / loaded host or GPU)."""
    warnings = []
    load = info.get("load_average")
    cpus = info.get("cpu_count") or 1
    if load and load[0] > 0.5 * cpus:
        warnings.append("host load average {:.1f} on {} cores".format(load[0], cpus))
    for gpu in info.get("gpus", []):
        if (gpu.get("utilization_percent") or 0) > 20:
            warnings.append("GPU {} ({}) is {:.0f}% utilised by other work before the run".format(
                gpu["index"], gpu["name"], gpu["utilization_percent"]))
    return warnings


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")
