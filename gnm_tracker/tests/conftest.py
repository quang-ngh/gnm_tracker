import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gnm_tracker.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def gnm_model(cfg):
    pytest.importorskip("torch")
    pytest.importorskip("gnm")
    from gnm_tracker.model import GnmModel

    return GnmModel(cfg, "cpu")
