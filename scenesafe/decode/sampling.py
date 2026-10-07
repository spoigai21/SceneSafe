"""Sampling policies: which decoded frames are kept for the model.

The same Policy objects run in two modes, so the frames measured for speed are the frames measured
for quality:
  - streaming, over a live decoder (sample())
  - offline, over metadata stored from an every-frame decode (select_indices())
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from scenesafe.decode.decoder import iter_frames

SMALL_W, SMALL_H = 32, 18  # thumbnail used for scene-change detection


class Policy:
    name = "base"
    keyframes_only = False  # ask the decoder to skip non-keyframes
    needs_small = False     # needs a grayscale thumbnail of every decoded frame

    def reset(self):
        pass

    def keep(self, t: float, is_key: bool, small: np.ndarray | None) -> bool:
        raise NotImplementedError


class AllFrames(Policy):
    name = "all"

    def keep(self, t, is_key, small):
        return True


class Uniform(Policy):
    """Keep the first frame at or after each tick of 1/fps seconds."""

    def __init__(self, fps: float):
        self.fps, self.name = fps, f"uniform@{fps:g}"
        self.reset()

    def reset(self):
        self.next_t = None

    def keep(self, t, is_key, small):
        if self.next_t is None:
            self.next_t = t
        if t + 1e-9 < self.next_t:
            return False
        while self.next_t <= t + 1e-9:
            self.next_t += 1 / self.fps
        return True


class Keyframes(Policy):
    name = "keyframes"
    keyframes_only = True

    def keep(self, t, is_key, small):
        return is_key


class SceneChange(Policy):
    """Keep a frame when it differs from the last kept frame by more than `threshold`.

    Difference = mean absolute difference of 32x18 grayscale thumbnails, scaled to 0-1.
    Measured against the last *kept* frame (not the previous frame), so slow pans are sampled too.
    """
    needs_small = True

    def __init__(self, threshold: float):
        self.threshold, self.name = threshold, f"scene@{threshold:g}"
        self.reset()

    def reset(self):
        self.last = None

    def keep(self, t, is_key, small):
        s = small.astype(np.float32)
        if self.last is None or np.abs(s - self.last).mean() / 255 > self.threshold:
            self.last = s
            return True
        return False


def parse(spec: str) -> Policy:
    if spec == "all":
        return AllFrames()
    if spec == "keyframes":
        return Keyframes()
    kind, _, value = spec.partition("@")
    if kind == "uniform":
        return Uniform(float(value))
    if kind == "scene":
        return SceneChange(float(value))
    raise ValueError(f"unknown policy {spec!r}")


def thumbnail(frame) -> np.ndarray:
    return frame.reformat(width=SMALL_W, height=SMALL_H, format="gray").to_ndarray()


def sample(path: Path, policy: Policy, backend: str = "cpu", start_s: float = 0.0,
           duration_s: float | None = None, stats: dict | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Decode and yield (t, RGB uint8 array) for kept frames only. RGB conversion is paid only for kept frames.

    If `stats` is given, it is filled with decoded/kept counts.
    """
    policy.reset()
    decoded = kept = 0
    for d in iter_frames(path, backend, policy.keyframes_only, start_s, duration_s):
        decoded += 1
        small = thumbnail(d.frame) if policy.needs_small else None
        if policy.keep(d.t, d.is_key, small):
            kept += 1
            yield d.t, d.frame.to_ndarray(format="rgb24")
    if stats is not None:
        stats.update(decoded=decoded, kept=kept)


def select_indices(policy: Policy, times: np.ndarray, is_key: np.ndarray, smalls: np.ndarray) -> np.ndarray:
    """Offline mode: which of the stored every-frame entries the policy keeps."""
    policy.reset()
    return np.array([i for i in range(len(times))
                     if policy.keep(float(times[i]), bool(is_key[i]), smalls[i])], dtype=np.int64)
