"""Hugging Face path for Gemma 4 (student E2B; also works for text-only causal LMs).

One encoding is shared by LoRA training, prediction and the server so they cannot drift:
  chat template(system, user[image, state_prefix + question]) + assistant step prefixes
tokenized as  [prefix: everything up to the end of the state] + [suffix: question .. step].
The prefix is what the server caches.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch

from .prompt import SYSTEM, question_text, state_prefix
from .schema import Item
from .serve.backends import PrefixCache
from .steps import Step, item_steps

SPLIT = ""  # private-use marker where the prefix ends inside the rendered template


def load(model_id: str, adapter: str | None = None, dtype=None, device: str | None = None):
    from transformers import AutoProcessor, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        tok = getattr(processor, "tokenizer", processor)
    except Exception:
        processor, tok = None, AutoTokenizer.from_pretrained(model_id)
    model = None
    import transformers as T

    for cls_name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(T, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(model_id, dtype=dtype)
            break
        except (ValueError, KeyError):
            continue
    if model is None:
        raise RuntimeError(f"could not load {model_id}")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    return model.to(device), processor, tok


def label_token_ids(tok, labels) -> list[int]:
    out = []
    for l in labels:
        ids = tok.encode(l, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label {l!r} is {len(ids)} tokens for this tokenizer; pick single-token labels")
        out.append(ids[0])
    return out


def render(tok, state: str, question: str, has_image: bool) -> tuple[str, str]:
    content = ([{"type": "image"}] if has_image else []) + [{"type": "text", "text": state_prefix(state) + SPLIT + question}]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    before, after = s.split(SPLIT)
    return before, after


def encode_prefix(processor, tok, before: str, image, device):
    if image is not None:
        if processor is None:
            raise ValueError("image items need a multimodal processor")
        enc = processor(text=before, images=[image], return_tensors="pt", add_special_tokens=False)
    else:
        enc = tok(before, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}


@dataclass
class HFExample:
    prefix: dict  # processor outputs for the prefix (input_ids, attention_mask, pixel_values, ...)
    suffix_ids: list[int]
    step_pos: list[int]  # absolute positions predicting each step label
    step_label_ids: list[list[int]]
    steps: list[Step]


def encode_example(item: Item, root: Path, processor, tok, device, steps: list[Step] | None = None,
                   max_suffix: int = 1024) -> HFExample:
    from PIL import Image

    steps = steps or item_steps(item)
    q, sp = question_text(item.primitive, item.instructions, item.criteria)
    before, after = render(tok, item.state, q, item.image is not None)
    img = Image.open(root / item.image).convert("RGB") if item.image else None
    pre = encode_prefix(processor, tok, before, img, device)
    plen = pre["input_ids"].shape[1]
    ids = tok.encode(after, add_special_tokens=False)[:max_suffix]
    pos, labs = [], []
    for k, st in enumerate(steps):
        if sp[k]:
            ids += tok.encode(sp[k].lstrip() if k == 0 else sp[k], add_special_tokens=False)
        pos.append(plen + len(ids) - 1)
        labs.append(label_token_ids(tok, st.labels))
        if k < len(steps) - 1:
            ids.append(labs[-1][st.target])
    return HFExample(pre, ids, pos, labs, steps)


def forward_example(model, ex: HFExample, device):
    """Full forward of prefix + suffix; returns logits only at the step positions [S, V]."""
    pre = ex.prefix
    ids = torch.cat([pre["input_ids"], torch.tensor([ex.suffix_ids], device=device)], 1)
    kw = {k: v for k, v in pre.items() if k not in ("input_ids", "attention_mask")}
    if "token_type_ids" in kw:
        tt = kw["token_type_ids"]
        kw["token_type_ids"] = torch.cat([tt, torch.zeros(1, len(ex.suffix_ids), dtype=tt.dtype, device=device)], 1)
    if "mm_token_type_ids" in kw:
        tt = kw["mm_token_type_ids"]
        kw["mm_token_type_ids"] = torch.cat([tt, torch.zeros(1, len(ex.suffix_ids), dtype=tt.dtype, device=device)], 1)
    att = torch.ones_like(ids)
    keep = torch.tensor(ex.step_pos, device=device)
    try:
        out = model(input_ids=ids, attention_mask=att, logits_to_keep=keep, **kw)
        logits = out.logits[0]
        if logits.shape[0] != len(ex.step_pos):  # model ignored logits_to_keep
            logits = logits[keep]
    except TypeError:
        logits = model(input_ids=ids, attention_mask=att, **kw).logits[0][keep]
    return logits


class HFBackend:
    """Server/predict backend with prefix caching (deep-copied KV cache per question)."""

    def __init__(self, model_id: str, adapter: str | None = None, cache_size: int = 64,
                 use_prefix_cache: bool = True, model=None, processor=None, tok=None, dtype=None):
        if model is None:
            model, processor, tok = load(model_id, adapter, dtype=dtype)
        self.model, self.processor, self.tok = model.eval(), processor, tok
        self.device = next(model.parameters()).device
        self.name = f"spark-1:{Path(adapter).name if adapter else model_id}"
        self.cache = PrefixCache(cache_size)
        self.use_cache = use_prefix_cache

    @torch.no_grad()
    def prefix(self, state: str, img_bytes: bytes | None, use_cache: bool = True):
        import io

        from PIL import Image

        k = PrefixCache.key(self.name, state, img_bytes or b"")
        hit = self.cache.get(k) if (use_cache and self.use_cache) else None
        if hit is not None:
            return hit, True
        before, _ = render(self.tok, state, "", img_bytes is not None)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB") if img_bytes else None
        pre = encode_prefix(self.processor, self.tok, before, img, self.device)
        out = self.model(**pre, use_cache=True)
        entry = (pre, out.past_key_values, pre["input_ids"].shape[1])
        if use_cache and self.use_cache:
            self.cache.put(k, entry)
        return entry, False

    @torch.no_grad()
    def answer_steps(self, entry, question: str, step_prefixes, step_labels):
        pre, past, plen = entry
        _, after = render(self.tok, "", question, False)
        past = copy.deepcopy(past)
        ids = self.tok.encode(after, add_special_tokens=False)
        total = plen
        out = []
        for k, labels in enumerate(step_labels):
            if step_prefixes[k]:
                ids += self.tok.encode(step_prefixes[k].lstrip() if k == 0 else step_prefixes[k], add_special_tokens=False)
            x = torch.tensor([ids], device=self.device)
            att = torch.ones(1, total + len(ids), dtype=torch.long, device=self.device)
            o = self.model(input_ids=x, attention_mask=att, past_key_values=past, use_cache=True)
            past = o.past_key_values
            total += len(ids)
            lab_ids = label_token_ids(self.tok, labels)
            lp = torch.log_softmax(o.logits[0, -1, lab_ids].float(), -1)
            out.append(lp.tolist())
            ids = [lab_ids[int(lp.argmax())]]
        return out
