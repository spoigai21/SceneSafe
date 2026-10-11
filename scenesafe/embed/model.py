"""The one model used throughout. Kept apart from clip.py so code that must not load torch (FAISS, which bundles
its own OpenMP runtime) can still record which model made the embeddings."""

DEFAULT_MODEL = "ViT-B-32"
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"

# Phase 4c: the one alternative model, chosen on practice films only (EVAL-PROTOCOL.md, "Phase 4c amendment").
ALT_MODEL = "ViT-B-16-SigLIP2"
ALT_PRETRAINED = "webli"
