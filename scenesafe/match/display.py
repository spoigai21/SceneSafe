"""Judging material: a 4-frame strip per scene, and labelling sheets for the primary labeller (no method shown)."""

from scenesafe.corpus import video_path

FRAMES_PER_SCENE = 4
THUMB = (192, 144)
PAIRS_PER_SHEET = 6


def frame_strip(title: str, start: float, end: float, out):
    """4 frames spaced evenly across the scene, side by side."""
    from PIL import Image
    from scenesafe.decode.decoder import iter_frames
    strip = Image.new("RGB", (THUMB[0] * FRAMES_PER_SCENE + 4 * (FRAMES_PER_SCENE - 1), THUMB[1]), "white")
    step = (end - start) / FRAMES_PER_SCENE
    for i in range(FRAMES_PER_SCENE):
        t = min(start + (i + 0.5) * step, max(start, end - 1.0))   # never past the last sampled second (BUGS.md #8)
        for f in iter_frames(video_path(title), start_s=t, duration_s=2.0):
            strip.paste(f.frame.to_image().resize(THUMB), (i * (THUMB[0] + 4), 0))
            break
    strip.save(out, quality=82)


def sheets(pool: list[dict], frames_dir, out_dir):
    """Labelling sheets for the primary labeller: PAIRS_PER_SHEET pairs each, in pool order, no method shown."""
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    w = THUMB[0] * FRAMES_PER_SCENE + 4 * (FRAMES_PER_SCENE - 1) + 16
    row_h = THUMB[1] + 30
    for s in range(0, len(pool), PAIRS_PER_SHEET):
        chunk = pool[s:s + PAIRS_PER_SHEET]
        sheet = Image.new("RGB", (w, row_h * len(chunk)), "white")
        draw = ImageDraw.Draw(sheet)
        for r, p in enumerate(chunk):
            y = r * row_h
            draw.text((8, y + 6), f"{p['pair']}  [{p['category']}]  {p['brief']}", fill="black")
            sheet.paste(Image.open(frames_dir / f"{p['scene']}.jpg"), (8, y + 24))
        sheet.save(out_dir / f"sheet_{s // PAIRS_PER_SHEET + 1:03d}.jpg", quality=82)
