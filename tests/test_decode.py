from fractions import Fraction

import av
import numpy as np
import pytest

from scenesafe.decode.cpu import decode_window, probe

FPS, SECONDS, W, H = 24, 4, 64, 48


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.time_base = Fraction(1, FPS)
        for i in range(FPS * SECONDS):
            img = np.full((H, W, 3), (i * 2) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def test_probe(clip):
    meta = probe(clip)
    assert (meta["width"], meta["height"]) == (W, H)
    assert meta["fps"] == pytest.approx(FPS)


def test_window_returns_exact_frame_count(clip):
    frames = list(decode_window(clip, start_s=1.0, duration_s=2.0))
    assert len(frames) == 2 * FPS
    assert frames[0].shape == (H, W, 3) and frames[0].dtype == np.uint8


def test_window_starts_at_requested_time(clip):
    first = next(decode_window(clip, start_s=1.0, duration_s=1.0))
    # frame 24 was painted with value 48; allow for lossy encoding
    assert abs(int(first.mean()) - 48) <= 3
