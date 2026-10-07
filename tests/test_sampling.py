from fractions import Fraction

import av
import numpy as np
import pytest

from scenesafe.decode.decoder import iter_frames
from scenesafe.decode.sampling import (SceneChange, Uniform, parse, sample, select_indices, thumbnail)

FPS, SECONDS, GOP, W, H = 24, 6, 12, 64, 48


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """6 s at 24 fps, a keyframe every 12 frames; the picture changes brightness sharply at 3 s."""
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=FPS, options={"g": str(GOP), "keyint_min": str(GOP),
                                                              "sc_threshold": "0"})
        stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
        stream.time_base = Fraction(1, FPS)
        for i in range(FPS * SECONDS):
            value = 40 if i < 3 * FPS else 200
            frame = av.VideoFrame.from_ndarray(np.full((H, W, 3), value, np.uint8), format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return path


def kept_times(clip, spec, **kw):
    return [t for t, _ in sample(clip, parse(spec), **kw)]


def test_all_keeps_every_frame(clip):
    assert len(kept_times(clip, "all")) == FPS * SECONDS


@pytest.mark.parametrize("fps", [0.5, 1, 2])
def test_uniform_rate(clip, fps):
    times = kept_times(clip, f"uniform@{fps:g}")
    assert len(times) == SECONDS * fps
    assert np.allclose(np.diff(times), 1 / fps, atol=1 / FPS)


def test_keyframes_skips_decoding(clip):
    stats = {}
    times = list(t for t, _ in sample(clip, parse("keyframes"), stats=stats))
    assert len(times) == FPS * SECONDS // GOP
    assert stats["decoded"] == len(times)  # non-keyframes never came out of the decoder


def test_scene_change_fires_on_cut(clip):
    times = kept_times(clip, "scene@0.2")
    assert times[0] == 0
    assert len(times) == 2 and abs(times[1] - 3.0) < 1e-6


def test_offline_selection_matches_streaming(clip):
    """The quality evaluation (offline) must keep exactly the frames the speed benchmark (streaming) keeps."""
    decoded = list(iter_frames(clip))
    times = np.array([d.t for d in decoded])
    is_key = np.array([d.is_key for d in decoded])
    smalls = np.stack([thumbnail(d.frame) for d in decoded])
    for spec in ["all", "uniform@1", "uniform@0.5", "keyframes", "scene@0.2"]:
        offline = times[select_indices(parse(spec), times, is_key, smalls)].tolist()
        assert offline == pytest.approx(kept_times(clip, spec)), spec


def test_window(clip):
    times = kept_times(clip, "all", start_s=2.0, duration_s=1.0)
    assert len(times) == FPS and times[0] == pytest.approx(2.0)


def test_policy_reset_between_runs():
    for policy in (Uniform(1), SceneChange(0.1)):
        small = np.zeros((18, 32), np.uint8)
        first = [policy.keep(t, False, small) for t in (0.0, 0.5, 1.0)]
        policy.reset()
        assert [policy.keep(t, False, small) for t in (0.0, 0.5, 1.0)] == first
