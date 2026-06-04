#!/usr/bin/env python3
"""Pre-download + load the edu embedding model (validates EduEmbedder, warms HF cache)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data.edu_embed import DEFAULT_MODEL, EduEmbedder

t0 = time.monotonic()
print("warming", DEFAULT_MODEL, flush=True)
emb = EduEmbedder()
vec = emb.embed(["Dies ist ein deutscher Testsatz über Geschichte und Wissenschaft."])
print(f"WARM_OK dim={emb.dim} device={emb.device} shape={tuple(vec.shape)} "
      f"norm={float(vec.norm()):.3f} in {time.monotonic()-t0:.1f}s")
