"""CPU decode backend (ffmpeg via PyAV)."""

from collections.abc import Iterator
from pathlib import Path

import av
import numpy as np


def probe(path: Path) -> dict:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return {
            "codec": stream.codec_context.name,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": float(stream.average_rate),
            "duration_s": float(container.duration / av.time_base),
        }


def decode_window(path: Path, start_s: float, duration_s: float, threads: str = "AUTO") -> Iterator[np.ndarray]:
    """Yield every frame in [start_s, start_s + duration_s) as an RGB uint8 array (H, W, 3).

    Includes the YUV -> RGB conversion, since the model needs RGB and that cost is part of decode.
    """
    end_s = start_s + duration_s
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = threads
        container.seek(int(start_s / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < start_s:
                continue
            if t >= end_s:
                break
            yield frame.to_ndarray(format="rgb24")
