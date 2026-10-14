"""Phase 6, NVIDIA lane: time the final pipeline on one rented NVIDIA GPU, on the sample the laptop was timed on
(gpu/sample.json, written by gpu/prepare.py). See gpu/RUNBOOK.md.

  smoke   every code path once on tiny inputs (one decode batch, 30 s of audio, one scene through every VLM call, one
          image, 8 card texts); about 2 minutes plus model loading. Exits non-zero if anything fails.
  time    3 runs of each stage (median and range), the same stages as scripts/run_phase6.py:
            - decode at 1 fps + CLIP ViT-B/32 fp16 (CPU decode, and NVDEC if this PyAV build has it)
            - audio decode (PyAV) and Whisper large-v3-turbo (faster-whisper, float16) on the same 10-minute windows
            - strip and grid images (CPU)
            - card text embedding (mpnet), all 4,922 cards per run, child process included as on the laptop
            - scene descriptions and the 5d safety passes (Qwen3-VL-4B-Instruct on vLLM, bf16), sequential (one request
              at a time, comparable to the laptop) and batched (all requests in one call)
          Writes results/phase6/gpu_gpu-nvidia-a10_seed0.json.

Differences from the laptop, recorded in the results: the VLM runs in bf16 on vLLM (laptop: MLX 4-bit), Whisper on
faster-whisper (laptop: mlx-whisper). So the comparison is hardware plus software, not hardware alone. A stage that fails
is recorded with its error and the run continues, so rented time is not lost.

--dry (laptop test only): CPU, 1-2 items per stage, Whisper "tiny", and a stand-in for vLLM that returns fixed text.
"""

import argparse
import gc
import json
import math
import os
import platform
import queue
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from scenesafe.common.results import Timer, load_avg, write_result     # noqa: E402

SAMPLE = ROOT / "gpu" / "sample.json"
LANE = "gpu-nvidia-a10"
PHASE = 6
VLM_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
WHISPER_MODEL = "large-v3-turbo"
CLIP_BATCH = 128
PACKAGES = ("torch", "vllm", "transformers", "faster-whisper", "ctranslate2", "open_clip_torch",
            "sentence-transformers", "av", "numpy", "pillow")


def stats(xs: list[float]) -> dict:
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "runs": xs}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_info() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                     "--format=csv,noheader"], text=True).strip()
        info["nvidia_smi"] = q
    except (OSError, subprocess.CalledProcessError) as e:
        info["nvidia_smi"] = f"unavailable: {e}"
    try:
        import torch
        info["torch_cuda"] = torch.version.cuda
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as e:                                                  # noqa: BLE001
        info["torch_cuda"] = f"unavailable: {e}"
    try:
        from av.codec.hwaccel import hwdevices_available
        info["pyav_hw_devices"] = list(hwdevices_available())        # "cuda" must be listed for NVDEC
    except Exception as e:                                                  # noqa: BLE001
        info["pyav_hw_devices"] = f"unavailable: {e}"
    info["packages"] = {}
    for p in PACKAGES:
        try:
            info["packages"][p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            info["packages"][p] = None
    return info


# ---------------------------------------------------------------- decode + CLIP

def frame_batches(path, backend: str, batch: int, limit: int | None = None):
    """Decode at 1 fps in a background thread (as scripts/embed_corpus.py does) and yield stacked RGB batches."""
    from scenesafe.decode.sampling import Uniform, sample
    q: queue.Queue = queue.Queue(maxsize=4)

    def produce():
        try:
            frames, n = [], 0
            for _, f in sample(path, Uniform(1.0), backend):
                frames.append(f)
                n += 1
                if len(frames) == batch:
                    q.put(np.stack(frames))
                    frames = []
                if limit and n >= limit:
                    break
            if frames:
                q.put(np.stack(frames))
        except Exception as e:                                              # noqa: BLE001
            q.put(e)
        q.put(None)

    threading.Thread(target=produce, daemon=True).start()
    while (item := q.get()) is not None:
        if isinstance(item, Exception):
            raise item
        yield item


class Clip:
    def __init__(self, device: str):
        import torch
        from scenesafe.embed.clip import ClipEmbedder
        self.torch, self.half = torch, device == "cuda"
        self.emb = ClipEmbedder(device)
        if self.half:
            self.emb.model.half()                          # the Phase 2 GPU choice: fp16

    def embed(self, path, backend: str, limit: int | None = None) -> int:
        n = 0
        with self.torch.inference_mode():
            for frames in frame_batches(path, backend, CLIP_BATCH if not limit else 8, limit):
                x = self.emb.preprocess_on_device(frames)
                e = self.emb.model.encode_image(x.half() if self.half else x).float()
                (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()          # .cpu() waits for the GPU
                n += len(frames)
        return n


# ---------------------------------------------------------------- Whisper

class Whisper:
    def __init__(self, device: str, model: str):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device=device, compute_type="float16" if device == "cuda" else "int8")

    def words(self, audio: np.ndarray) -> int:
        # greedy (beam_size=1) with the default temperature fallback, as mlx-whisper does on the laptop
        segs, _ = self.model.transcribe(audio, language="en", beam_size=1, condition_on_previous_text=False,
                                        vad_filter=False)
        return sum(len(s.text.split()) for s in segs)                     # the generator does the work


# ---------------------------------------------------------------- VLM (vLLM)

def p_yes(top: dict, yes_ids: list[int], no_ids: list[int]) -> float:
    """P(yes) from the first answer token, as scenesafe.safety.tags.verify_p computes it on the laptop: the softmax of
    yes vs no over the summed probabilities of "yes"/"Yes" and "no"/"No". `top` maps token id -> log-probability; vLLM
    is asked for exactly these 4 tokens (logprob_token_ids), so all are normally present. NaN if none is."""
    def lse(ids):
        xs = [top[i] for i in ids if i in top]
        if not xs:
            return -math.inf
        m = max(xs)
        return m + math.log(sum(math.exp(x - m) for x in xs))
    y, n = lse(yes_ids), lse(no_ids)
    if y == -math.inf and n == -math.inf:
        return math.nan
    if n == -math.inf:
        return 1.0
    if y == -math.inf:
        return 0.0
    return 1.0 / (1.0 + math.exp(-(y - n)))


def first_token_logprobs(output) -> dict:
    """{token id: logprob} for the first generated token of a vLLM CompletionOutput."""
    lp = output.logprobs[0]
    return {int(t): float(getattr(v, "logprob", v)) for t, v in lp.items()}


class FakeLLM:
    """Stands in for vllm.LLM in --dry runs on the laptop: same call shape, fixed answers."""

    def __init__(self, yes_id: int, no_id: int):
        self.yes_id, self.no_id = yes_id, no_id

    def generate(self, requests, params, use_tqdm=False):
        out = []
        for r in requests:
            assert "<|image_pad|>" in r["prompt"] and r["multi_modal_data"]["image"].size[0] > 0
            text = "weapons" if params.max_tokens > 3 else "yes"
            lps = [{self.yes_id: -0.2, self.no_id: -1.8}] if params.logprob_token_ids else None
            out.append(type("R", (), {"outputs": [type("O", (), {"text": text, "logprobs": lps})()]})())
        return out


class FakeParams:
    def __init__(self, **kw):
        self.max_tokens = kw.get("max_tokens", 16)
        self.logprob_token_ids = kw.get("logprob_token_ids")


class Vlm:
    """Qwen3-VL-4B-Instruct on vLLM. Prompts come from the same strings the laptop uses (scenesafe.match.cards and
    scenesafe.safety.tags), wrapped in the model's chat template with one image, as mlx-vlm does on the laptop."""

    def __init__(self, dry: bool):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(VLM_MODEL)
        tok = self.processor.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
        if dry:
            self.llm, self.Params = FakeLLM(self.yes[0], self.no[0]), FakeParams
        else:
            from vllm import LLM, SamplingParams
            # No prefix or image cache: runs 2 and 3 repeat the same prompts and images, and a cache would time
            # cache hits instead of the work. The laptop (mlx-vlm) has no such cache.
            self.llm = LLM(model=VLM_MODEL, dtype="bfloat16", max_model_len=8192, limit_mm_per_prompt={"image": 1},
                           gpu_memory_utilization=0.85, enable_prefix_caching=False, mm_processor_cache_gb=0,
                           seed=0)
            self.Params = SamplingParams

    def request(self, text: str, image_path: Path) -> dict:
        from PIL import Image
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "multi_modal_data": {"image": Image.open(image_path).convert("RGB")}}

    def run(self, jobs: list[tuple[str, Path]], params, batched: bool) -> list:
        """jobs: (prompt text, image path). Builds the requests (image load included, as on the laptop) and generates,
        all in one call (batched) or one request per call (sequential)."""
        if batched:
            return self.llm.generate([self.request(t, p) for t, p in jobs], params, use_tqdm=False)
        return [self.llm.generate([self.request(t, p)], params, use_tqdm=False)[0] for t, p in jobs]


def vlm_jobs(sample: dict, scenes: list[dict], score_ids: list[int]) -> dict:
    """The five kinds of VLM call in the final (5d) pipeline, with the laptop's prompts and token limits."""
    from scenesafe.match import cards as cd
    from scenesafe.safety import tags as tg
    p = lambda rel: ROOT / rel                                                  # noqa: E731
    dlg = lambda s: cd.cut(s["dialogue"]) or "(no speech)"                     # noqa: E731
    return {
        "card": ([(cd.PROMPT % dlg(s), p(s["strip"])) for s in scenes],
                 dict(temperature=0.0, max_tokens=cd.MAX_TOKENS, repetition_penalty=cd.REPETITION_PENALTY)),
        "strip_tags": ([(tg.PROMPT % dlg(s), p(s["strip"])) for s in scenes],
                       dict(temperature=0.0, max_tokens=tg.TAG_TOKENS)),
        "strip_check": ([(tg.VERIFY_PROMPT % (dlg(s), tg.TAGS[t]), p(s["strip"]))
                         for s in scenes for t in sample["strip_check_tags"]],
                        dict(temperature=0.0, max_tokens=tg.VERIFY_TOKENS)),
        "grid_tags": ([(tg.GRID_TAG_PROMPT % dlg(s), p(s["grid"])) for s in scenes],
                      dict(temperature=0.0, max_tokens=tg.TAG_TOKENS)),
        "grid_check": ([(tg.GRID_OBVIOUS_PROMPT % (dlg(s), tg.TAGS_OBVIOUS["weapons"]), p(s["grid"]))
                        for s in scenes],
                       dict(temperature=0.0, max_tokens=1, logprob_token_ids=score_ids)),
    }


PER_CHECK = ("strip_check", "grid_check")


def vlm_outputs(vlm: Vlm, kind: str, outs: list) -> list:
    if kind == "grid_check":
        return [p_yes(first_token_logprobs(o.outputs[0]), vlm.yes, vlm.no) for o in outs]
    return [o.outputs[0].text.strip() for o in outs]


# ---------------------------------------------------------------- text embedding

EMBED = """
import json, sys, numpy as np
from sentence_transformers import SentenceTransformer
texts = json.load(open(sys.argv[1]))
m = SentenceTransformer(sys.argv[3], device=sys.argv[4])
np.save(sys.argv[2], m.encode(texts, batch_size=128, normalize_embeddings=True).astype(np.float32))
"""


def embed_texts(texts: list[str], device: str, tmp: Path) -> int:
    """Same job as scenesafe.match.dialogue.embed_text (child process, model load included), on this device."""
    from scenesafe.match.dialogue import TEXT_MODEL
    key, out = tmp / "texts.json", tmp / "emb.npy"
    key.write_text(json.dumps(texts))
    subprocess.run([sys.executable, "-c", EMBED, str(key), str(out), TEXT_MODEL, device], check=True, cwd=ROOT)
    return len(np.load(out))


# ---------------------------------------------------------------- steps

class Errors:
    def __init__(self):
        self.items = {}

    def guard(self, name: str, fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except Exception as e:                                              # noqa: BLE001
            self.items[name] = f"{type(e).__name__}: {e}"
            log(f"FAILED {name}: {e}")
            traceback.print_exc()
            return None


def video_path(film: str) -> Path:
    from scenesafe.corpus import video_path as vp
    return vp(film)


def smoke(args, sample: dict):
    from scenesafe.match.display import frame_grid, frame_strip
    from scenesafe.match.transcribe import RATE, load_audio
    err, film = Errors(), sorted(sample["films"])[0]
    log(f"gpu: {json.dumps(gpu_info())}")
    if not args.dry:
        import torch
        assert torch.cuda.is_available(), "torch sees no CUDA device"
    clip = err.guard("clip_load", Clip, args.device)
    if clip:
        err.guard("decode_cpu_clip", lambda: log(f"decode cpu + CLIP: {clip.embed(video_path(film), 'cpu', 128)} frames"))
        if not args.dry:
            n = err.guard("decode_nvdec_clip", clip.embed, video_path(film), "nvdec", 128)
            log(f"decode nvdec + CLIP: {n} frames" if n else "nvdec unavailable (recorded, not fatal)")
            err.items.pop("decode_nvdec_clip", None)
        del clip
    audio = err.guard("audio_decode", load_audio, video_path(film))
    if audio is not None:
        a = int(sample["whisper_window_s"][0] * RATE)
        w = err.guard("whisper_load", Whisper, args.device, args.whisper_model)
        if w:
            err.guard("whisper", lambda: log(f"whisper: {w.words(audio[a:a + 30 * RATE])} words in 30 s"))
            del w
    with tempfile.TemporaryDirectory() as tmp:
        s = sample["image_scenes"][0]
        err.guard("strip_image", frame_strip, s["film"], s["start"], s["end"], Path(tmp) / "s.jpg")
        err.guard("grid_image", frame_grid, s["film"], s["start"], s["end"], Path(tmp) / "g.jpg")
        err.guard("text_embedding", lambda: log(f"text embedding: {embed_texts(sample['card_texts'][:8], args.device, Path(tmp))} vectors"))
    free_gpu()
    vlm = err.guard("vlm_load", Vlm, args.dry)
    if vlm:
        jobs = vlm_jobs(sample, sample["scenes"][:2], vlm.yes + vlm.no)
        for kind, (items, kw) in jobs.items():
            for batched in (False, True):
                outs = err.guard(f"vlm_{kind}_{'batched' if batched else 'sequential'}", vlm.run, items[:2],
                                 vlm.Params(**kw), batched)
                if outs:
                    log(f"{kind} ({'batched' if batched else 'sequential'}): {vlm_outputs(vlm, kind, outs)}")
        ps = [p for p in vlm_outputs(vlm, "grid_check", vlm.run(jobs["grid_check"][0][:2],
                                                               vlm.Params(**jobs["grid_check"][1]), True))]
        if not all(0.0 <= p <= 1.0 for p in ps):
            err.items["grid_check_p_yes"] = f"P(yes) not in [0, 1]: {ps}"
    if err.items:
        log(f"SMOKE FAILED: {json.dumps(err.items, indent=1)}")
        sys.exit(1)
    log("SMOKE OK")


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def time_step(args, sample: dict):
    from scenesafe.match.display import frame_grid, frame_strip
    from scenesafe.match.transcribe import RATE, load_audio
    err, runs = Errors(), (1 if args.dry else sample["runs"])
    films = sorted(sample["films"])
    scenes = sample["scenes"][:2] if args.dry else sample["scenes"]
    img_scenes = sample["image_scenes"][:2] if args.dry else sample["image_scenes"]
    texts = sample["card_texts"][:16] if args.dry else sample["card_texts"]
    limit = 16 if args.dry else None
    data, loads = {"errors": err.items}, []
    hours = sum(sample["films"][t]["duration_s"] for t in films) / 3600

    # decode + CLIP, CPU decode and NVDEC
    clip = err.guard("clip_load", Clip, args.device)
    if clip:
        clip.embed(video_path(films[0]), "cpu", 256)                         # warm-up (untimed)
        for backend in ("cpu", "nvdec"):
            if backend == "nvdec" and args.dry:
                continue
            xs, frames = [], 0
            for r in range(runs):
                with Timer() as t:
                    frames = err.guard(f"decode_{backend}_clip", lambda: sum(clip.embed(video_path(f), backend, limit)
                                                                             for f in films))
                if frames is None:
                    break
                xs.append(t.seconds / hours)
                log(f"run {r + 1}: decode ({backend}) + CLIP {xs[-1]:.1f} s/video-hour")
            if xs:
                data[f"decode_{backend}_clip_s_per_video_hour"] = stats(xs)
                data["frames_embedded"] = frames
        del clip
        free_gpu()

    # audio decode + Whisper
    audio = {}
    xs = []
    for r in range(runs):
        with Timer() as t:
            audio = err.guard("audio_decode", lambda: {f: load_audio(video_path(f)) for f in films}) or {}
        if not audio:
            break
        xs.append(t.seconds / hours)
        log(f"run {r + 1}: audio decode {xs[-1]:.1f} s/video-hour")
    if xs:
        data["audio_decode_s_per_video_hour"] = stats(xs)
    w = err.guard("whisper_load", Whisper, args.device, args.whisper_model) if audio else None
    if w:
        start, length = sample["whisper_window_s"]
        if args.dry:
            length = 10
        a, b = int(start * RATE), int((start + length) * RATE)
        w.words(audio[films[0]][a:a + 30 * RATE])                            # warm-up (untimed)
        xs, words = [], 0
        for r in range(runs):
            with Timer() as t:
                words = err.guard("whisper", lambda: sum(w.words(audio[f][a:b]) for f in films))
            if words is None:
                break
            xs.append(t.seconds / (len(films) * length / 3600))
            log(f"run {r + 1}: Whisper {xs[-1]:.0f} s per audio-hour ({words} words)")
        if xs:
            data["whisper_s_per_audio_hour"] = stats(xs)
            data["whisper_words"] = words
        del w
        free_gpu()
    audio = None

    # images (CPU) and text embedding
    rows = {"strip_image": [], "grid_image": [], "card_text_embedding": []}
    for r in range(runs):
        loads.append(load_avg())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for kind, fn in (("strip_image", frame_strip), ("grid_image", frame_grid)):
                with Timer() as t:
                    ok = err.guard(kind, lambda: [fn(s["film"], s["start"], s["end"], tmp / f"{i}_{kind}.jpg")
                                                  for i, s in enumerate(img_scenes)])
                if ok is not None:
                    rows[kind].append(t.seconds / len(img_scenes))
            with Timer() as t:
                n = err.guard("card_text_embedding", embed_texts, texts, args.device, tmp)
            if n:
                rows["card_text_embedding"].append(t.seconds / n)
        log(f"run {r + 1}: " + ", ".join(f"{k} {v[-1]:.3f}" for k, v in rows.items() if v))
    free_gpu()

    # VLM, sequential and batched
    vlm = err.guard("vlm_load", Vlm, args.dry)
    vlm_rows = {m: {k: [] for k in ("card", "strip_tags", "strip_check", "grid_tags", "grid_check")}
                for m in ("sequential", "batched")}
    outputs = {}
    if vlm:
        jobs = vlm_jobs(sample, scenes, vlm.yes + vlm.no)
        for kind, (items, kw) in jobs.items():                              # warm-up: one call of each kind
            err.guard(f"vlm_{kind}_warmup", vlm.run, items[:1], vlm.Params(**kw), False)
        for r in range(runs):
            for mode in ("sequential", "batched"):
                for kind, (items, kw) in jobs.items():
                    with Timer() as t:
                        outs = err.guard(f"vlm_{kind}_{mode}", vlm.run, items, vlm.Params(**kw), mode == "batched")
                    if outs is None:
                        continue
                    per = len(items) if kind in PER_CHECK else len(scenes)
                    vlm_rows[mode][kind].append(t.seconds / per)
                    if r == 0:
                        outputs.setdefault(mode, {})[kind] = vlm_outputs(vlm, kind, outs)
                log(f"run {r + 1} {mode}: " + ", ".join(f"{k} {v[-1]:.3f}" for k, v in vlm_rows[mode].items() if v))
    data["s_per_scene"] = {k: stats(v) for k, v in rows.items() if v}
    data["vlm"] = {m: {"s_per_scene": {k: stats(v) for k, v in rows_.items() if v and k not in PER_CHECK},
                       "s_per_check": {k: stats(v) for k, v in rows_.items() if v and k in PER_CHECK}}
                   for m, rows_ in vlm_rows.items()}
    data["vlm_outputs_run1"] = {"scenes": [s["name"] for s in scenes], **outputs}
    data["load_avg_per_run"] = loads
    data["gpu"] = gpu_info()
    config = {
        "runs": runs, "dry": args.dry, "films": films, "video_hours": hours,
        "n_scenes": len(scenes), "n_image_scenes": len(img_scenes),
        "image_sample": "60 scenes of the 3 shipped films, seed 0 (the laptop timed images on its 60 scenes from all "
                        "19 films; only 3 films are shipped)",
        "sampling": "uniform 1 fps", "clip": f"ViT-B/32 fp16, PyTorch CUDA, batch {CLIP_BATCH}",
        "decode": "PyAV/ffmpeg on the CPU in a background thread; and NVDEC (PyAV hwaccel cuda) if available",
        "whisper": f"{args.whisper_model} (faster-whisper / CTranslate2, float16, beam 1); laptop: mlx-whisper",
        "vlm": f"{VLM_MODEL} on vLLM, bf16, greedy, no prefix or image cache; laptop: MLX 4-bit (mlx-vlm)",
        "vlm_modes": "sequential: one request per call (comparable to the laptop); batched: all requests in one call",
        "text_model": "sentence-transformers all-mpnet-base-v2 on CUDA, child process per run (as on the laptop)",
        "excluded": "model loading and one warm-up call of each kind",
    }
    if args.dry:
        print(json.dumps({k: v for k, v in data.items() if k != "gpu"}, indent=1, default=str)[:3000])
        return
    path = write_result(PHASE, "gpu", LANE, sample["seed"], config, data)
    log(f"wrote {path}")
    if err.items:
        log(f"some stages failed (recorded in the results): {list(err.items)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["smoke", "time"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dry", action="store_true", help="laptop test: CPU, 1-2 items, Whisper tiny, fake vLLM")
    ap.add_argument("--whisper-model", default=None)
    args = ap.parse_args()
    if args.dry:
        args.device = "cpu"
    args.whisper_model = args.whisper_model or ("tiny" if args.dry else WHISPER_MODEL)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sample = json.loads(SAMPLE.read_text())
    (smoke if args.step == "smoke" else time_step)(args, sample)


if __name__ == "__main__":
    main()
