"""Question-bank framework: build items from generators with option shuffling, "none fits"
cases with companion Nouls, scene-grouped splits and provenance on every item."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

from . import __version__
from .schema import Item, stable_hash, write_jsonl

SPLIT_RATIOS = (("train", 0.8), ("val", 0.1), ("test", 0.1))
NONE_FITS_RATE = 0.12  # fraction of eligible choice items turned into "none fits"
COMPANION_POS_RATE = 0.12  # fraction of normal choice items that also get a companion Noul


def assign_split(group: str, salt: str = "spark-one-v1") -> str:
    u = (stable_hash(salt, group) % 10_000) / 10_000
    acc = 0.0
    for name, r in SPLIT_RATIOS:
        acc += r
        if u < acc:
            return name
    return SPLIT_RATIOS[-1][0]


def shuffle_options(item: Item, rng: random.Random) -> Item:
    """Randomly permute Choice options (other primitives are returned unchanged)."""
    if item.primitive != "choice":
        return item
    keys = list(item.criteria)
    rng.shuffle(keys)
    return replace(item, criteria={k: item.criteria[k] for k in keys})


def permute_options(item: Item, order: list[int]) -> Item:
    keys = list(item.criteria)
    return replace(item, criteria={keys[i]: item.criteria[keys[i]] for i in order})


def make_none_fits(item: Item) -> Item:
    """Remove the correct option so that no option fits. Needs >= 3 options."""
    assert item.primitive == "choice" and not item.none_fits and len(item.criteria) >= 3
    crit = {k: v for k, v in item.criteria.items() if k != item.answer}
    prov = dict(item.provenance, transforms=item.provenance.get("transforms", []) + ["none_fits"],
                removed_answer=item.answer)
    return replace(item, id=item.id + "-nf", criteria=crit, answer=None, none_fits=True,
                   provenance=prov)


def companion_noul(item: Item, lang: str) -> Item:
    """Noul asking whether any option fits; read the Choice only when this is high."""
    opts = "; ".join(item.criteria)
    if lang == "ur-roman":
        instr = f"Sawal: {item.instructions} Kya in mein se koi jawab sahi hai: {opts}?"
    else:
        instr = f"For the question \"{item.instructions}\", does any of these options fit: {opts}?"
    prov = dict(item.provenance, transforms=item.provenance.get("transforms", []) + ["companion_noul"],
                parent=item.id)
    return replace(item, id=item.id + "-any", task=item.task + "_anyfits", primitive="noul",
                   instructions=instr, criteria=None, answer=not item.none_fits,
                   none_fits=False, provenance=prov, extra={})


GeneratorFn = Callable[[random.Random, str], dict]


def build_task_items(name: str, fn: GeneratorFn, version: str, langs: Iterable[str], n: int,
                     seed: int, img_dir: Path | None, rel_img_prefix: str = "images") -> list[Item]:
    """Run one generator n times per language. fn(rng, lang) returns a scene dict:
    {"image": PIL.Image|None, "state": str, "params": dict,
     "questions": [{"q": short_name, "primitive", "instructions", "criteria", "answer", "extra"?}]}
    """
    items: list[Item] = []
    for lang in langs:
        for idx in range(n):
            scene_seed = stable_hash(name, lang, seed, idx)
            rng = random.Random(scene_seed)
            scene = fn(rng, lang)
            group = f"{name}-{lang}-{idx:06d}"
            split = assign_split(group)
            image_rel = None
            if scene.get("image") is not None:
                assert img_dir is not None
                image_rel = f"{rel_img_prefix}/{name}/{group}.png"
                path = img_dir / name / f"{group}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.png")
                scene["image"].save(tmp, optimize=False)
                os.replace(tmp, path)  # atomic: readers never see a half-written image
            for q in scene["questions"]:
                prov = {
                    "generator": name, "version": version, "seed": seed, "index": idx,
                    "scene_seed": scene_seed, "group": group, "label_source": "code",
                    "params": scene.get("params", {}),
                }
                it = Item(
                    id=f"{group}-{q['q']}", task=f"{name}.{q['q']}", primitive=q["primitive"],
                    instructions=q["instructions"], state=scene["state"], criteria=q["criteria"],
                    answer=q["answer"], lang=lang, image=image_rel, split=split, provenance=prov,
                    extra=q.get("extra", {}),
                )
                it = shuffle_options(it, rng)
                it.validate()
                if it.primitive == "choice" and len(it.criteria) >= 3 and q.get("allow_none_fits", True):
                    u = rng.random()
                    if u < NONE_FITS_RATE:
                        nf = make_none_fits(it)
                        nf.validate()
                        items += [nf, companion_noul(nf, lang)]
                        continue
                    if u < NONE_FITS_RATE + COMPANION_POS_RATE:
                        items.append(companion_noul(it, lang))
                items.append(it)
    return items


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_bank(items: list[Item], out_dir: Path, seed: int, n_per_task: int, tasks: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "items.jsonl"
    write_jsonl(path, (it.to_dict() for it in items))
    import PIL

    manifest = {
        "spark_one_version": __version__, "created_unix": int(time.time()), "seed": seed,
        "n_per_task_per_lang": n_per_task, "generators": tasks,
        "pillow": PIL.__version__, "python": platform.python_version(),
        "items_sha256": file_sha256(path), "n_items": len(items),
        "by_split": dict(Counter(it.split for it in items)),
        "by_primitive": dict(Counter(it.primitive for it in items)),
        "by_lang": dict(Counter(it.lang for it in items)),
        "by_modality": dict(Counter(it.modality for it in items)),
        "by_task": dict(sorted(Counter(it.task for it in items).items())),
        "none_fits": sum(it.none_fits for it in items),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
