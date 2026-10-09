"""CLIP-style image embedder (OpenCLIP), PyTorch eager."""

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image

from scenesafe.embed.model import DEFAULT_MODEL, DEFAULT_PRETRAINED

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipEmbedder:
    def __init__(self, device: str, model_name: str = DEFAULT_MODEL, pretrained: str = DEFAULT_PRETRAINED):
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.eval().to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.size = 224

    def preprocess_batch(self, frames: list[np.ndarray]) -> torch.Tensor:
        """Resize, crop and normalise on the CPU (the naive path Phase 9 would fuse)."""
        return torch.stack([self.preprocess(Image.fromarray(f)) for f in frames])

    @torch.inference_mode()
    def preprocess_on_device(self, frames: np.ndarray) -> torch.Tensor:
        """Same steps as preprocess_batch (bicubic resize of the short side, centre crop, normalise), run on the device.

        Not bit-identical to the PIL path: on 30 frames, embedding cosine similarity to the PIL path was
        0.995 mean, 0.987 min. Use one path consistently within a comparison.
        """
        x = torch.from_numpy(frames).to(self.device).permute(0, 3, 1, 2).float() / 255
        h, w = x.shape[-2:]
        scale = self.size / min(h, w)
        nh, nw = round(h * scale), round(w * scale)
        x = F.interpolate(x, size=(nh, nw), mode="bicubic", antialias=True, align_corners=False).clamp(0, 1)
        top, left = (nh - self.size) // 2, (nw - self.size) // 2
        x = x[..., top:top + self.size, left:left + self.size]
        mean = torch.tensor(CLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(CLIP_STD, device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.inference_mode()
    def embed_text(self, texts: list[str]) -> torch.Tensor:
        out = self.model.encode_text(self.tokenizer(texts).to(self.device))
        return (out / out.norm(dim=-1, keepdim=True)).cpu()

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
