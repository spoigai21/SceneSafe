"""CLIP-style image embedder (OpenCLIP), PyTorch eager."""

import numpy as np
import open_clip
import torch
from PIL import Image

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class ClipEmbedder:
    def __init__(self, device: str, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED):
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(self.device)

    def preprocess_batch(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Resize, crop and normalise on the CPU (the naive path Phase 9 would fuse)."""
        return torch.stack([self.preprocess(Image.fromarray(f)) for f in frames])

    @torch.inference_mode()
    def embed(self, batch: torch.Tensor) -> torch.Tensor:
        out = self.model.encode_image(batch.to(self.device))
        out = out / out.norm(dim=-1, keepdim=True)
        return out.cpu()

    def sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()
