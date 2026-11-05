# content of conftest.py

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runtorchpot", "--runtensorpot", action="store_true", default=False, dest="runtorchpot",
        help="run tests that need the PyTorch fitting backend (torchpotential); `--runtensorpot` is a deprecated alias"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "torchpot: mark test needs torchpotential (PyTorch backend) to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("runtorchpot"):
        return
    skip_torchpot = pytest.mark.skip(reason="need --runtorchpot option to run")
    for item in items:
        if "torchpot" in item.keywords:
            item.add_marker(skip_torchpot)
