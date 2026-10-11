"""Phase 4f: speech-to-text for a film with Whisper (mlx-whisper, on the laptop GPU), cached in data/transcripts/.
Audio is decoded with PyAV to 16 kHz mono (no ffmpeg needed). Segments Whisper marks as probably not speech are kept
in the file but flagged, so later code can drop them."""

import json

import numpy as np

from scenesafe.common.results import REPO_ROOT
from scenesafe.corpus import video_path

TRANSCRIPT_DIR = REPO_ROOT / "data" / "transcripts"
WHISPER_MODEL = "mlx-community/whisper-large-v3-turbo"
RATE = 16_000
NO_SPEECH_MAX = 0.6          # a segment with no_speech_prob above this is treated as not speech (mlx-whisper reports
MIN_AVG_LOGPROB = -1.0       # 0 for every segment here, so in practice only the log-probability check applies)
LONG_SEGMENT = 10.0          # a segment this long (seconds) with fewer than MIN_WORDS_PER_SEC words is Whisper filling
MIN_WORDS_PER_SEC = 0.5      # music or silence with a stock phrase ("Thank you.", "The End"), not speech


def load_audio(path) -> np.ndarray:
    import av
    out = []
    with av.open(str(path)) as c:
        res = av.AudioResampler(format="flt", layout="mono", rate=RATE)
        for frame in c.decode(audio=0):
            for f in res.resample(frame):
                out.append(f.to_ndarray().reshape(-1))
        for f in res.resample(None):
            out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.float32)


def transcribe(title_id: str) -> dict:
    out = TRANSCRIPT_DIR / f"{title_id}.json"
    if out.exists():
        return json.loads(out.read_text())
    import mlx_whisper
    r = mlx_whisper.transcribe(load_audio(video_path(title_id)), path_or_hf_repo=WHISPER_MODEL, language="en",
                               condition_on_previous_text=False, verbose=None)
    segs = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip(),
             "no_speech_prob": round(s["no_speech_prob"], 3), "avg_logprob": round(s["avg_logprob"], 3)}
            for s in r["segments"]]
    data = {"title": title_id, "model": WHISPER_MODEL, "segments": segs}
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    return data


def is_speech(seg: dict) -> bool:
    dur = seg["end"] - seg["start"]
    sparse = dur >= LONG_SEGMENT and len(seg["text"].split()) / dur < MIN_WORDS_PER_SEC
    return (seg["no_speech_prob"] <= NO_SPEECH_MAX and seg["avg_logprob"] >= MIN_AVG_LOGPROB and bool(seg["text"])
            and not sparse)


def speech(segments: list[dict]) -> list[dict]:
    """The speech segments, with repetition loops collapsed: a segment whose text is the same as the one before it
    (Whisper stuck repeating a line, e.g. "I can." twenty times a second apart) is dropped."""
    out, prev = [], None
    for s in segments:
        if is_speech(s) and s["text"] != prev:
            out.append(s)
        prev = s["text"]
    return out


def scene_text(segments: list[dict], start: float, end: float) -> str:
    """Speech whose midpoint falls inside [start, end)."""
    return " ".join(s["text"] for s in speech(segments) if start <= (s["start"] + s["end"]) / 2 < end)
