"""Decode backends: cpu (ffmpeg), videotoolbox (Apple hardware), nvdec (NVIDIA hardware, GPU session only)."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
from av.codec.hwaccel import HWAccel

BACKENDS = ("cpu", "videotoolbox", "nvdec")
_HW_DEVICE = {"videotoolbox": "videotoolbox", "nvdec": "cuda"}


@dataclass
class Decoded:
    t: float          # presentation time in seconds
    is_key: bool      # self-contained frame (decodable without its neighbours)
    frame: av.VideoFrame


def iter_frames(path: Path, backend: str = "cpu", keyframes_only: bool = False,
                start_s: float = 0.0, duration_s: float | None = None) -> Iterator[Decoded]:
    """Yield decoded frames in [start_s, start_s + duration_s).

    keyframes_only tells the decoder to skip non-keyframes without decoding them, which is where
    the saving comes from: the skipped frames cost only the demux, not the decode.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}")
    hw = None if backend == "cpu" else HWAccel(device_type=_HW_DEVICE[backend], allow_software_fallback=False)
    end_s = float("inf") if duration_s is None else start_s + duration_s
    with av.open(str(path), hwaccel=hw) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if keyframes_only:
            stream.codec_context.skip_frame = "NONKEY"
        if start_s > 0:
            container.seek(int(start_s / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t < start_s:
                continue
            if t >= end_s:
                break
            yield Decoded(t, bool(frame.key_frame), frame)
