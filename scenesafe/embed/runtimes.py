"""One model (CLIP ViT-B/32 image encoder), several inference configurations: runtime x precision x device.

Every configuration takes the same input (normalised float32 pixels, N x 3 x 224 x 224, as a NumPy array) and
returns unnormalised 512-d embeddings as float32 NumPy, so they can be timed and compared like for like.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from scenesafe.common.results import REPO_ROOT
from scenesafe.embed.clip import ClipEmbedder

MODEL_DIR = REPO_ROOT / "data" / "models"
ONNX_FP32 = MODEL_DIR / "clip_vitb32_visual_fp32.onnx"
ort.set_default_logger_severity(3)


@dataclass
class Config:
    name: str        # e.g. "pytorch-fp16-mps"
    runtime: str     # pytorch | onnxruntime | onnxruntime-coreml
    precision: str   # fp32 | fp16 | int8-static | int8-dynamic
    device: str      # cpu | mps | coreml-gpu | coreml-ane
    lane: str        # results lane


CONFIGS = [
    Config("pytorch-fp32-cpu", "pytorch", "fp32", "cpu", "cpu-laptop"),
    Config("pytorch-fp32-mps", "pytorch", "fp32", "mps", "gpu-apple-m5"),   # the reference
    Config("pytorch-fp16-mps", "pytorch", "fp16", "mps", "gpu-apple-m5"),
    Config("onnxruntime-fp32-cpu", "onnxruntime", "fp32", "cpu", "cpu-laptop"),
    Config("onnxruntime-int8static-cpu", "onnxruntime", "int8-static", "cpu", "cpu-laptop"),
    Config("onnxruntime-int8dynamic-cpu", "onnxruntime", "int8-dynamic", "cpu", "cpu-laptop"),
    Config("coreml-fp32-gpu", "onnxruntime-coreml", "fp32", "coreml-gpu", "gpu-apple-m5"),
    Config("coreml-fp32-ane", "onnxruntime-coreml", "fp32", "coreml-ane", "ane-apple-m5"),
]
REFERENCE = "pytorch-fp32-mps"
BY_NAME = {c.name: c for c in CONFIGS}


def _visual(device: str) -> torch.nn.Module:
    return ClipEmbedder(device).model.visual.eval()


def export_onnx(path: Path, batch: int | None = None) -> Path:
    """Export the image encoder. batch=None gives a dynamic batch axis; Core ML needs a fixed one."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(batch or 4, 3, 224, 224)
    dynamic = None if batch else {"pixels": {0: "batch"}, "embedding": {0: "batch"}}
    torch.onnx.export(_visual("cpu"), (dummy,), str(path), input_names=["pixels"], output_names=["embedding"],
                      dynamic_axes=dynamic, opset_version=17, dynamo=False)
    return path


class CalibrationReader:
    """Feeds calibration batches to the ONNX Runtime static quantizer."""

    def __init__(self, pixels: np.ndarray, batch: int = 16):
        self.batches = iter([pixels[i:i + batch] for i in range(0, len(pixels), batch)])

    def get_next(self):
        b = next(self.batches, None)
        return None if b is None else {"pixels": b}


def quantize(precision: str, calibration: np.ndarray | None = None) -> Path:
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType, quantize_dynamic,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out = MODEL_DIR / f"clip_vitb32_visual_{precision.replace('-', '')}.onnx"
    if out.exists():
        return out
    src = export_onnx(ONNX_FP32)
    prep = MODEL_DIR / "clip_vitb32_visual_fp32_prep.onnx"
    if not prep.exists():
        quant_pre_process(str(src), str(prep), skip_symbolic_shape=True)  # symbolic inference fails on a CLIP Reshape
    if precision == "int8-dynamic":
        quantize_dynamic(str(prep), str(out), weight_type=QuantType.QInt8, per_channel=True)
    elif precision == "int8-static":
        if calibration is None:
            raise ValueError("int8-static needs calibration pixels")
        quantize_static(str(prep), str(out), CalibrationReader(calibration), quant_format=QuantFormat.QDQ,
                        per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        calibrate_method=CalibrationMethod.MinMax)
    else:
        raise ValueError(precision)
    return out


class Runner:
    """Loads one configuration; run(pixels) -> embeddings. Includes host<->device copies, as a real pipeline would."""

    def __init__(self, cfg: Config, batch: int, calibration: np.ndarray | None = None):
        self.cfg, self.batch = cfg, batch
        if cfg.runtime == "pytorch":
            self.dev = torch.device(cfg.device)
            self.model = _visual(cfg.device)
            self.dtype = torch.float16 if cfg.precision == "fp16" else torch.float32
            self.model = self.model.to(self.dtype)
        else:
            if cfg.runtime == "onnxruntime-coreml":
                path = export_onnx(MODEL_DIR / f"clip_vitb32_visual_fp32_b{batch}.onnx", batch)
                units = {"coreml-gpu": "CPUAndGPU", "coreml-ane": "CPUAndNeuralEngine"}[cfg.device]
                providers = [("CoreMLExecutionProvider", {"MLComputeUnits": units, "ModelFormat": "MLProgram"}),
                             "CPUExecutionProvider"]
            else:
                path = export_onnx(ONNX_FP32) if cfg.precision == "fp32" else quantize(cfg.precision, calibration)
                providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(path), providers=providers)

    def sync(self):
        if self.cfg.runtime == "pytorch" and self.dev.type == "mps":
            torch.mps.synchronize()

    @torch.inference_mode()
    def run(self, pixels: np.ndarray) -> np.ndarray:
        if self.cfg.runtime == "pytorch":
            x = torch.from_numpy(pixels).to(self.dev, self.dtype)
            return self.model(x).float().cpu().numpy()
        return self.session.run(None, {"pixels": pixels})[0]

    @torch.inference_mode()
    def transfer_only(self, pixels: np.ndarray) -> None:
        """Host -> device copy alone (PyTorch on MPS), to split transfer time from compute time."""
        if self.cfg.runtime == "pytorch" and self.dev.type != "cpu":
            torch.from_numpy(pixels).to(self.dev, self.dtype)
            self.sync()

    def embed_all(self, pixels: np.ndarray) -> np.ndarray:
        """Embeddings for any number of frames, in batches of self.batch (pads the last batch if the size is fixed)."""
        out = []
        for i in range(0, len(pixels), self.batch):
            b = pixels[i:i + self.batch]
            n = len(b)
            if self.cfg.runtime == "onnxruntime-coreml" and n < self.batch:
                b = np.concatenate([b, np.zeros((self.batch - n, *b.shape[1:]), b.dtype)])
            out.append(self.run(b)[:n])
        return np.concatenate(out)
