"""Run PACE fitting tests and verify that energy/force/gradient tensors are computed on the GPU."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCHPOTENTIAL_DEVICE", "cuda:0")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
import torchpotential
import pytest
from torchpotential.torchpot import TorchPotential

assert torch.cuda.is_available(), "CUDA is not available to torch"
assert torch.cuda.device_count() >= 1, "No visible GPU"
device = torch.device(os.environ["TORCHPOTENTIAL_DEVICE"])
assert device.type == "cuda", f"TORCHPOTENTIAL_DEVICE must select a CUDA device, got {device}"
product = torch.ones(128, 128, dtype=torch.float64, device=device) @ torch.ones(128, 128, dtype=torch.float64,
                                                                                  device=device)
assert product.device.type == "cuda", product.device
assert float(product[0, 0]) == 128.0
print("GPU_PREFLIGHT", json.dumps({
    "torch": torch.__version__, "cuda": torch.version.cuda, "torchpotential": torchpotential.__version__,
    "device": torch.cuda.get_device_name(device), "capability": torch.cuda.get_device_capability(device),
    "matmul_device": str(product.device),
}), flush=True)

original = TorchPotential.compute_grad
placements = []


def checked_compute_grad(self, batch):
    result = original(self, batch)
    devices = {name: str(value.device) for name, value in zip(("loss", "gradient", "energy", "forces"), result)}
    for name in ("gradient", "energy", "forces"):
        assert devices[name].startswith("cuda"), f"PACE {name} fell back to {devices[name]}"
    if not placements:
        print("PACE_NATIVE_PLACEMENT", json.dumps(devices), flush=True)
    placements.append(devices)
    return result


TorchPotential.compute_grad = checked_compute_grad
args = sys.argv[1:] or ["tests/", "--runtorchpot", "-m", "torchpot", "-q"]
status = pytest.main(args)
summary = {
    "pytest_exit_code": int(status), "fitting_calls_on_gpu": len(placements),
    "first_placement": placements[0] if placements else None,
    "peak_gpu_memory": torch.cuda.max_memory_allocated(device),
}
print("GPU_VALIDATION_RESULT", json.dumps(summary), flush=True)
assert placements, "No PACE fitting calls were exercised"
sys.exit(status)
