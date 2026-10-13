"""E3: AudioSet sound tags (AST, pretrained, no training) for each labelled scene: max over 10 s windows."""
import json, sys
from pathlib import Path
import numpy as np, av, torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification
R = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R))
from scenesafe.corpus import video_path
SP = Path(__file__).resolve().parents[2] / "data" / "phase5" / "dev5e"
times = json.load(open(SP / "scene_times.json"))
scenes = sorted({i["scene"] for p in ("results/phase5c/pool.json", "results/phase5d/pool.json")
                 for i in json.load(open(R / p))["items"]})
NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
fe = ASTFeatureExtractor.from_pretrained(NAME); model = ASTForAudioClassification.from_pretrained(NAME).eval()
labels = model.config.id2label
out_path = SP / "audio_dev.json"
done = json.load(open(out_path)) if out_path.exists() else {}
def audio(title, a, b, sr=16000):
    c = av.open(str(video_path(title))); s = c.streams.audio[0]
    rs = av.AudioResampler(format="flt", layout="mono", rate=sr)
    c.seek(int(max(a - 1, 0) / s.time_base), stream=s)
    buf = []
    for fr in c.decode(s):
        t = float(fr.pts * s.time_base)
        if t > b: break
        for r in rs.resample(fr):
            x = r.to_ndarray().reshape(-1)
            if t + len(x) / sr >= a: buf.append((t, x))
    c.close()
    if not buf: return np.zeros(sr, np.float32)
    x = np.concatenate([v for _, v in buf]); t0 = buf[0][0]
    return x[max(0, int((a - t0) * sr)): max(1, int((b - t0) * sr))]
for n, sc in enumerate(scenes, 1):
    if sc in done: continue
    title, a, b = times[sc]
    x = audio(title, a, b)
    wins = [x[i:i + 160000] for i in range(0, max(len(x), 1), 160000)][:12] or [x]
    with torch.no_grad():
        feats = fe([w for w in wins if len(w) > 1600] or [np.zeros(16000, np.float32)], sampling_rate=16000, return_tensors="pt")
        pr = torch.sigmoid(model(**feats).logits).max(0).values.numpy()
    top = np.argsort(-pr)[:15]
    done[sc] = {labels[int(k)]: round(float(pr[k]), 4) for k in top}
    keep = ["Gunshot, gunfire", "Machine gun", "Fusillade", "Artillery fire", "Cap gun", "Explosion", "Screaming",
            "Slap, smack", "Whack, thwack", "Smash, crash", "Thump, thud", "Shout", "Yell", "Battle cry", "Groan",
            "Glass", "Shatter", "Fire", "Crackle", "Siren", "Crying, sobbing", "Wail, moan", "Laughter", "Music",
            "Speech", "Silence"]
    lid = {v: int(k) for k, v in labels.items()}
    done[sc] |= {k: round(float(pr[lid[k]]), 4) for k in keep if k in lid}
    if n % 25 == 0 or n == len(scenes):
        json.dump(done, open(out_path, "w")); print(f"audio {n}/{len(scenes)}", flush=True)
json.dump(done, open(out_path, "w"))
