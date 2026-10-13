"""Content tags for brand safety (EVAL-PROTOCOL.md, "Phase 5 amendment"). The same definitions go to the detector
(the Phase 4g card model) and to the labellers."""

import json
import time

from scenesafe.match import cards as cd

TAGS = {
    "violence": "people fight, hit, shoot at, stab or attack someone, or struggle violently",
    "weapons": "a gun, knife, sword or other weapon is held, pointed or used",
    "accidents": "a crash, fire, explosion, sinking, bad fall or disaster, or a vehicle clearly in trouble",
    "death": "a dead body, someone killed or dying, a funeral or grave; or the conversation is mainly about a killing "
             "or a death",
    "horror": "monsters, creatures, ghosts, menacing supernatural or gruesome images, people screaming in terror",
    "alcohol": "people drinking alcohol, drunkenness, or alcohol bottles or glasses in focus",
    "smoking": "someone smoking a cigarette, cigar or pipe",
    "sexual": "nudity, sexual activity or strong sexual suggestion (kissing alone is not)",
    "drugs": "illegal drug use or dealing",
}
TAG_TOKENS = 40

PROMPT = ("These are 4 frames from one scene of an old film, left to right in time. The scene's dialogue: \"%s\"\n"
          "Which of these content tags are present in the scene?\n"
          + "\n".join(f"- {t}: {d}" for t, d in TAGS.items()) +
          "\nAnswer with the names of the tags that are present, separated by commas, or the word none. "
          "Only list a tag if it is clearly present.")


def parse_tags(text: str) -> list[str]:
    """The known tag names that appear in the answer, in TAGS order. Anything else (including "none") is ignored."""
    low = text.lower()
    return [t for t in TAGS if t.rstrip("s") in low]          # "weapon" counts for weapons, "accident" for accidents


def tag_scenes(items: list[tuple[str, str, str]], cache_path, prompt: str = PROMPT) -> dict:
    """items: (scene, image path, dialogue). Returns {scene: {"answer", "tags"}}, cached and resumable."""
    done = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [it for it in items if it[0] not in done]
    if todo:
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, proc, cfg = cd._model()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for n, (name, img, dialogue) in enumerate(todo, 1):
            text = prompt % (cd.cut(dialogue) or "(no speech)")
            r = generate(model, proc, apply_chat_template(proc, cfg, text, num_images=1), [str(img)],
                         max_tokens=TAG_TOKENS, temperature=0.0, verbose=False)
            answer = getattr(r, "text", r).strip()
            done[name] = {"answer": answer, "tags": parse_tags(answer)}
            mx.clear_cache()
            if n % 50 == 0 or n == len(todo):
                cache_path.write_text(json.dumps(done, indent=1) + "\n")
                print(f"  tags: {n}/{len(todo)} ({(time.time() - t0) / n:.2f} s/scene)", flush=True)
    return done


# Phase 5b: a second, narrow question for each tag the detector gave a scene. The tag is kept only if the answer is
# yes. Same model, same definitions; one tag per question instead of nine at once.
VERIFY_PROMPT = ("These are 4 frames from one scene of an old film, left to right in time. The scene's dialogue: \"%s\"\n"
                 "Question: is this clearly present in the scene: %s?\n"
                 "Answer yes or no.")
VERIFY_TOKENS = 3


def verify(items: list[tuple[str, str, str, str]], cache_path) -> dict:
    """items: (scene, tag, strip image path, dialogue). Returns {"scene|tag": {"answer", "yes"}}, cached."""
    done = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [it for it in items if f"{it[0]}|{it[1]}" not in done]
    if todo:
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, proc, cfg = cd._model()
        t0 = time.time()
        for n, (name, tag, img, dialogue) in enumerate(todo, 1):
            text = VERIFY_PROMPT % (cd.cut(dialogue) or "(no speech)", TAGS[tag])
            r = generate(model, proc, apply_chat_template(proc, cfg, text, num_images=1), [str(img)],
                         max_tokens=VERIFY_TOKENS, temperature=0.0, verbose=False)
            answer = getattr(r, "text", r).strip()
            done[f"{name}|{tag}"] = {"answer": answer, "yes": answer.lower().startswith("yes")}
            mx.clear_cache()
            if n % 100 == 0 or n == len(todo):
                cache_path.write_text(json.dumps(done, indent=1) + "\n")
                print(f"  verify: {n}/{len(todo)} ({(time.time() - t0) / n:.2f} s/check)", flush=True)
    return done


# Phase 5c: the verifier sees the larger 2 × 2 frame grid and returns the probability of "yes" (yes vs no on the first
# answer token) instead of the text; a tag is kept if it is at least VERIFY_THRESHOLD.
GRID_PROMPT = ("These are 4 frames from one scene of an old film, in time order: top left, top right, bottom left, "
               "bottom right. The scene's dialogue: \"%s\"\n"
               "Question: is this clearly present in the scene: %s?\nAnswer yes or no.")
VERIFY_THRESHOLD = 0.2


def verify_p(items: list[tuple[str, str, str, str]], cache_path, prompt: str = GRID_PROMPT, defs: dict = TAGS) -> dict:
    """items: (scene, tag, grid image path, dialogue). Returns {"scene|tag": P(yes)}, cached and resumable."""
    done = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [it for it in items if f"{it[0]}|{it[1]}" not in done]
    if todo:
        import mlx.core as mx
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
        model, proc, cfg = cd._model()
        tok = getattr(proc, "tokenizer", proc)
        yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes")]
        no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No")]
        t0 = time.time()
        for n, (name, tag, img, dialogue) in enumerate(todo, 1):
            seen = []

            def sampler(logprobs, seen=seen):
                seen.append(logprobs)
                return mx.argmax(logprobs, axis=-1)

            text = prompt % (cd.cut(dialogue) or "(no speech)", defs[tag])
            generate(model, proc, apply_chat_template(proc, cfg, text, num_images=1), [str(img)],
                     max_tokens=1, sampler=sampler, verbose=False)
            lp = seen[0].reshape(-1)
            y = mx.logsumexp(mx.stack([lp[yes[0]], lp[yes[1]]]))
            n_ = mx.logsumexp(mx.stack([lp[no[0]], lp[no[1]]]))
            done[f"{name}|{tag}"] = float(mx.sigmoid(y - n_))
            mx.clear_cache()
            if n % 100 == 0 or n == len(todo):
                cache_path.write_text(json.dumps(done, indent=1) + "\n")
                print(f"  verify (grid): {n}/{len(todo)} ({(time.time() - t0) / n:.2f} s/check)", flush=True)
    return done


# Phase 5d: "obvious" definitions, at the author's request after the 5c labels. A tag is present only if a viewer would
# notice it at a glance; small, background, blurred or ambiguous things no longer count. These replace TAGS for the
# 5d detector and for the 5d labels.
TAGS_OBVIOUS = {
    "violence": "an obvious fight or attack on screen: punching, shooting at someone, stabbing, a brawl or a clear "
                "violent struggle (not a chase, a threat, a standoff, or a scuffle you cannot make out)",
    "weapons": "a weapon clearly in use or held up in plain view: pointed, fired, swung or brandished (not holstered, "
               "slung, lying around, in the background, too small to make out, or drawn in artwork)",
    "accidents": "a crash, a large fire, an explosion, a sinking, a bad fall or a disaster clearly shown",
    "death": "a dead body clearly shown, someone killed or dying on screen, a funeral or grave as the main subject; or "
             "the scene is mainly about a killing or a death",
    "horror": "a monster, creature, ghost or gruesome image clearly shown, or people screaming in terror (not blood "
              "alone, not only talk about a monster)",
    "alcohol": "people clearly drinking alcohol or drunk, or a drink or bottle as a main focus (not bottles or glasses "
               "in the background)",
    "smoking": "someone clearly smoking a cigarette, cigar or pipe",
    "sexual": "nudity, sexual activity or strong sexual suggestion (kissing alone is not)",
    "drugs": "illegal drug use or dealing clearly shown",
}

GRID_TAG_PROMPT = ("These are 4 frames from one scene of an old film, in time order: top left, top right, bottom left, "
                   "bottom right. The scene's dialogue: \"%s\"\n"
                   "Which of these content tags are obviously present in the scene?\n"
                   + "\n".join(f"- {t}: {d}" for t, d in TAGS_OBVIOUS.items()) +
                   "\nAnswer with the names of the tags that are present, separated by commas, or the word none. "
                   "Only list a tag if it is obvious at a glance.")
GRID_OBVIOUS_PROMPT = ("These are 4 frames from one scene of an old film, in time order: top left, top right, bottom "
                       "left, bottom right. The scene's dialogue: \"%s\"\n"
                       "Question: is this obviously present in the scene: %s?\nAnswer yes or no.")
