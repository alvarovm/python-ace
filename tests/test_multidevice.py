"""Fits on several devices (worker processes of torchpotential.distributed) must reproduce the single-device fit."""
import numpy as np
import pytest

from pyace.generalfit import GeneralACEFit

pytestmark = pytest.mark.torchpot

potential_config = {
    "deltaSplineBins": 0.001,
    "element": "Al",
    "fs_parameters": [1, 1, 1, 0.5],
    "npot": "FinnisSinclairShiftedScaled",
    "NameOfCutoffFunction": "cos",
    "nradmax_by_orders": [4, 2],
    "lmax_by_orders": [0, 1],
    "ndensity": 2,
    "rcut": 6.0,
    "dcut": 0.01,
    "radparameters": [5.25],
    "radbase": "ChebExpCos",
    "func_coefs_init": "random",
}
data_config = {"filename": "tests/representative_df.pckl.gzip"}
fit_config = {
    "optimizer": "BFGS",
    "maxiter": 6,
    "loss": {"kappa": 0.5, "L1_coeffs": 1e-6, "L2_coeffs": 1e-6},
}


def run_fit(optimizer="BFGS", batch_size=1, **backend):
    backend_config = {"evaluator": "torchpot", "batch_size": batch_size, "num_threads": 1}
    backend_config.update(backend)
    cfg = dict(fit_config, optimizer=optimizer)
    fitace = GeneralACEFit(potential_config=potential_config, data_config=data_config, fit_config=cfg,
                           backend_config=backend_config, seed=42)
    try:
        bbasis = fitace.fit()
        return np.array(bbasis.get_all_coeffs()), fitace
    finally:
        fitace.fit_backend.close()


@pytest.fixture
def cpu_serial(monkeypatch):
    monkeypatch.delenv("TORCHPOTENTIAL_DEVICES", raising=False)
    monkeypatch.setenv("TORCHPOTENTIAL_DEVICE", "cpu")


@pytest.mark.parametrize("optimizer", ["BFGS", "L-BFGS-B"])
def test_two_devices_reproduce_the_single_device_fit(optimizer, cpu_serial, monkeypatch):
    serial, _ = run_fit(optimizer)
    monkeypatch.setenv("TORCHPOTENTIAL_DEVICES", "cpu,cpu")
    dist, fitace = run_fit(optimizer)
    np.testing.assert_allclose(dist, serial, rtol=1e-6, atol=1e-8)


def test_workers_are_stopped_with_the_adapter(cpu_serial, monkeypatch):
    monkeypatch.setenv("TORCHPOTENTIAL_DEVICES", "cpu,cpu,cpu")
    fitace = GeneralACEFit(potential_config=potential_config, data_config=data_config,
                           fit_config=dict(fit_config, maxiter=2),
                           backend_config={"evaluator": "torchpot", "batch_size": 2, "num_threads": 1}, seed=42)
    fitace.fit()
    pool = fitace.fit_backend._pool
    assert pool is not None and pool.world_size == 3 and pool.active == []  # released after the fit
    procs = list(pool._procs)
    fitace.fit_backend.close()
    assert fitace.fit_backend._pool is None
    assert all(p.poll() is not None for p in procs)


def test_unsupported_optimizer_with_several_devices(cpu_serial, monkeypatch):
    monkeypatch.setenv("TORCHPOTENTIAL_DEVICES", "cpu,cpu")
    with pytest.raises(NotImplementedError):
        run_fit("trf")


def test_out_of_memory_on_a_worker_reduces_the_batch_size(cpu_serial, monkeypatch):
    from torchpotential.distributed import DevicePool, WorkerError

    monkeypatch.setenv("TORCHPOTENTIAL_DEVICES", "cpu,cpu")
    original = DevicePool.setup
    calls = []

    def failing_once(self, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise WorkerError("worker rank 1 failed during setup: OutOfMemoryError: CUDA out of memory")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DevicePool, "setup", failing_once)
    coefs, fitace = run_fit(batch_size=3)
    assert len(calls) == 2
    assert fitace.fit_backend.backend_config["batch_size"] == 1  # int(3 / 1.618)
    assert np.all(np.isfinite(coefs))
