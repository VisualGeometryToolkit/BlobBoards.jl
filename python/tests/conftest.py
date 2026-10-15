"""Shared fixtures: the Julia session, the example data, one detection run.

Run from the pixi py environment:

    pixi run -e py pytest python/tests

The GPU tests run when `blobboards.device()` resolves to "cuda" (pick the card
with CUDA_VISIBLE_DEVICES) and are skipped otherwise.
"""
import os

import numpy as np
import pytest

import blobboards


@pytest.fixture(scope="session")
def jl():
    blobboards._init()
    return blobboards._jl


@pytest.fixture(scope="session")
def board_paths():
    d = blobboards.example_data("boards")
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))


@pytest.fixture(scope="session")
def model_path():
    return blobboards.descriptor_network_path()


@pytest.fixture(scope="session")
def still_img():
    from PIL import Image
    path = os.path.join(blobboards.example_data("still"), "DSC02847.JPG")
    return np.asarray(Image.open(path).convert("L"))


@pytest.fixture(scope="session")
def gallery_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("gallery") / "gallery.jld2")


@pytest.fixture(scope="session")
def has_gpu():
    return blobboards.device() == "cuda"
