import numpy as np
import pytest

from scenesafe.embed.runtimes import BY_NAME, Runner


def cosine(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


@pytest.fixture(scope="module")
def pixels():
    return np.random.default_rng(0).standard_normal((5, 3, 224, 224)).astype(np.float32)


@pytest.fixture(scope="module")
def reference(pixels):
    return Runner(BY_NAME["pytorch-fp32-cpu"], batch=8).embed_all(pixels)


def test_onnx_export_matches_pytorch(pixels, reference):
    out = Runner(BY_NAME["onnxruntime-fp32-cpu"], batch=8).embed_all(pixels)
    assert out.shape == (5, 512)
    assert cosine(out, reference).min() > 0.9999


def test_fixed_batch_pads_and_trims(pixels, reference):
    """Core ML runs a fixed batch size; 5 frames at batch 4 = one full batch + one padded batch of 1."""
    out = Runner(BY_NAME["coreml-fp32-gpu"], batch=4).embed_all(pixels)
    assert out.shape == (5, 512)
    assert cosine(out, reference).min() > 0.999
