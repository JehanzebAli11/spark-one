"""Model backends for the System One server. A backend answers the steps of one question
given a cacheable prefix (image + state); `PrefixCache` keeps prefix KV states in an LRU."""

from __future__ import annotations

import hashlib
import io
import threading
from collections import OrderedDict

import numpy as np


class PrefixCache:
    def __init__(self, max_entries: int = 256):
        self.max = max_entries
        self.d: OrderedDict = OrderedDict()
        self.hits = self.misses = 0
        self.lock = threading.Lock()

    @staticmethod
    def key(*parts) -> str:
        h = hashlib.sha256()
        for p in parts:
            h.update(p if isinstance(p, bytes) else str(p).encode())
            h.update(b"\x1f")
        return h.hexdigest()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                self.hits += 1
                return self.d[k]
            self.misses += 1
            return None

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.max:
                self.d.popitem(last=False)


class TinyBackend:
    """Serves a tiny checkpoint with prefix caching. Box steps decode greedily."""

    def __init__(self, ckpt_path: str, cache_size: int = 256, threads: int = 2):
        import torch

        from ..tiny.predict import load_model

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = load_model(ckpt_path)
        self.name = "spark-1-tiny"
        self.cache = PrefixCache(cache_size)

    def _image_tensor(self, img: bytes | None):
        if img is None:
            return None
        from PIL import Image

        c = self.model.c
        im = Image.open(io.BytesIO(img)).convert("RGB").resize((c.img_size, c.img_size), Image.BILINEAR)
        return self.torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].float() / 255.0

    def prefix(self, state: str, img: bytes | None, use_cache: bool = True):
        from ..prompt import state_prefix
        from ..tiny.tokenizer import BOS, IMG, encode_text

        k = PrefixCache.key(self.name, state, img or b"")
        hit = self.cache.get(k) if use_cache else None
        if hit is not None:
            return hit, True
        c = self.model.c
        ids = [BOS] + ([IMG] * c.n_patches if img else []) + encode_text(state_prefix(state))
        with self.torch.no_grad():
            _, past = self.model(self.torch.tensor([ids]), self._image_tensor(img))
        entry = (past, len(ids))
        if use_cache:
            self.cache.put(k, entry)
        return entry, False

    def answer_steps(self, prefix_entry, question: str, step_prefixes, step_labels):
        """-> list of per-step log-prob lists over each step's labels."""
        from ..tiny.tokenizer import ANS, encode_text, label_id

        t = self.torch
        past, off = prefix_entry
        ids = encode_text(question) + [ANS]
        out = []
        with t.no_grad():
            for k, labels in enumerate(step_labels):
                ids += encode_text(step_prefixes[k]) if step_prefixes[k] else []
                logits, past = self.model(t.tensor([ids]), None, past=past, offset=off)
                off += len(ids)
                sub = logits[0, -1, [label_id(l) for l in labels]]
                lp = t.log_softmax(sub, -1)
                out.append(lp.tolist())
                ids = [label_id(labels[int(lp.argmax())])]  # greedy feed for the next step
        return out
