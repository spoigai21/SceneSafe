"""Phase 4g: a short written description ("scene card") of each scene, from a small vision-language model that looks
at the scene's 4-frame strip and reads its dialogue. Cached per film in data/cards/. Ads are matched to the card text
with the Phase 4f sentence model (scenesafe.match.dialogue.embed_text)."""

import json
import time

from scenesafe.common.results import REPO_ROOT

CARD_DIR = REPO_ROOT / "data" / "cards"
CARD_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
MAX_TOKENS = 90
REPETITION_PENALTY = 1.15     # without it the model repeats list items ("a man in a suit, a man in a suit, ...")
DIALOGUE_CHARS = 600          # dialogue given to the model, cut at a word
PROMPT = ("These are 4 frames from one scene of an old film, left to right in time. The scene's dialogue: \"%s\"\n"
          "Answer in this exact compact format, one line each:\n"
          "place: <where>\nvisible: <up to 6 different things, no repeats>\nactivity: <what people are doing>\n"
          "talk: <up to 3 specific topics of the dialogue, or none>\nmood: <one or two words>\n"
          "Only state what is clearly visible or said.")
MATCHED_FIELDS = ("place", "visible", "activity", "talk")     # mood is generated but not matched


def cut(text: str, n: int = DIALOGUE_CHARS) -> str:
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0]


def card_text(card: str) -> str:
    """The text matched against ads: the place, visible, activity and talk lines, without their keys; 'none' dropped."""
    out = []
    for line in card.splitlines():
        key, _, value = line.partition(":")
        value = value.strip()
        if key.strip().lower() in MATCHED_FIELDS and value and value.lower() != "none":
            out.append(value)
    return " ".join(out)


_LOADED = []


def _model():
    """Load the card model once per process (loading it per film kept old copies in memory)."""
    if not _LOADED:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
        _LOADED.extend([*load(CARD_MODEL), load_config(CARD_MODEL)])
    return _LOADED


def make_cards(title_id: str, scenes: list[tuple[str, str, str]]) -> dict:
    """scenes: (scene name, strip image path, dialogue). Returns {scene: {"card", "dialogue"}}, cached and resumable."""
    out = CARD_DIR / f"{title_id}.json"
    done = json.loads(out.read_text()) if out.exists() else {}
    todo = [s for s in scenes if s[0] not in done]
    if not todo:
        return done
    import mlx.core as mx
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    model, proc, cfg = _model()
    CARD_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for n, (name, img, dialogue) in enumerate(todo, 1):
        text = cut(dialogue)
        prompt = apply_chat_template(proc, cfg, PROMPT % (text or "(no speech)"), num_images=1)
        r = generate(model, proc, prompt, [str(img)], max_tokens=MAX_TOKENS, temperature=0.0,
                     repetition_penalty=REPETITION_PENALTY, verbose=False)
        done[name] = {"card": getattr(r, "text", r).strip(), "dialogue": text}
        mx.clear_cache()      # memory housekeeping only: without it the GPU cache grows and the laptop swaps
        if n % 25 == 0 or n == len(todo):
            out.write_text(json.dumps(done, indent=1) + "\n")
            print(f"  {title_id}: {n}/{len(todo)} ({(time.time() - t0) / n:.2f} s/scene)", flush=True)
    return done


# Phase 4h: the same model as a judge, re-ranking a shortlist. It sees the scene (4-frame strip and dialogue) and
# the ad's brief, literal guide line and theme line, and answers with one digit.
JUDGE_PROMPT = ("These are 4 frames from one scene of an old film, left to right in time. The scene's dialogue: \"%s\"\n"
                "An advertiser wants to show an ad next to a scene that suits it.\n"
                "Ad: %s\nLiteral fit: %s\nTheme: %s\n"
                "Answer with one digit only: 2 if the scene clearly shows the literal fit; 1 if it fits the theme, "
                "by what is shown or by what the conversation is about; 0 if neither.")
JUDGE_TOKENS = 3


def parse_digit(text: str) -> int:
    """The first 0, 1 or 2 in the answer; -1 if there is none (counted as 0 when ranking)."""
    for ch in text:
        if ch in "012":
            return int(ch)
    return -1


def judge(items: list[dict], cache_path) -> dict:
    """items: {"key", "img", "dialogue", "brief", "fits", "theme"}. Returns {key: digit}, cached and resumable."""
    done = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [it for it in items if it["key"] not in done]
    if todo:
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, proc, cfg = _model()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for n, it in enumerate(todo, 1):
            text = JUDGE_PROMPT % (cut(it["dialogue"]) or "(no speech)", it["brief"], it["fits"], it["theme"])
            r = generate(model, proc, apply_chat_template(proc, cfg, text, num_images=1), [str(it["img"])],
                         max_tokens=JUDGE_TOKENS, temperature=0.0, verbose=False)
            done[it["key"]] = {"answer": getattr(r, "text", r).strip(), "digit": parse_digit(getattr(r, "text", r))}
            mx.clear_cache()
            if n % 50 == 0 or n == len(todo):
                cache_path.write_text(json.dumps(done, indent=1) + "\n")
                print(f"  judge: {n}/{len(todo)} ({(time.time() - t0) / n:.2f} s/item)", flush=True)
    return {k: v["digit"] for k, v in done.items()}
