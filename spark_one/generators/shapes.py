"""Shape scenes: counting, colours, positions, relations, sizes and Box targets.
Every label is computed from the scene parameters that drew the image."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from .common import (COLOURS, SHAPES, box_frac, draw_shape, number_options, shape_area,
                     shape_bbox)

VERSION = "1"
W = H = 256


def _scene(rng: random.Random, n_min=3, n_max=9):
    n = rng.randint(n_min, n_max)
    objs, tries = [], 0
    while len(objs) < n and tries < 2000:
        tries += 1
        r = rng.uniform(11, 34)
        cx, cy = rng.uniform(r + 4, W - r - 4), rng.uniform(r + 4, H - r - 4)
        if all((cx - o["cx"]) ** 2 + (cy - o["cy"]) ** 2 > (r + o["r"] + 6) ** 2 for o in objs):
            objs.append({"shape": rng.choice(list(SHAPES)), "colour": rng.choice(list(COLOURS)),
                         "cx": cx, "cy": cy, "r": r})
    bg = tuple(rng.randint(235, 255) for _ in range(3))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    for o in objs:
        draw_shape(d, o["shape"], o["cx"], o["cy"], o["r"], COLOURS[o["colour"]][0])
        o["bbox"] = shape_bbox(o["shape"], o["cx"], o["cy"], o["r"])
        o["area"] = shape_area(o["shape"], o["r"])
    return img, objs


def _desc(o, lang):
    if lang == "ur-roman":
        return f"{COLOURS[o['colour']][1]} {SHAPES[o['shape']][1]}"
    return f"{o['colour']} {o['shape']}"


def _unique(objs):
    from collections import Counter

    c = Counter((o["colour"], o["shape"]) for o in objs)
    return [o for o in objs if c[(o["colour"], o["shape"])] == 1]


def shapes_scene(rng: random.Random, lang: str) -> dict:
    img, objs = _scene(rng)
    ur = lang == "ur-roman"
    qs = []
    # counting
    shape = rng.choice(list(SHAPES))
    k = sum(o["shape"] == shape for o in objs)
    qs.append({"q": "count", "primitive": "choice",
               "instructions": (f"Tasveer mein kitne {SHAPES[shape][1]} hain?" if ur
                                else f"How many {SHAPES[shape][0]} are in the image?"),
               "criteria": number_options(k, rng), "answer": str(k)})
    # total count as a Score
    levels_en = ["1 to 3 shapes", "4 to 5 shapes", "6 to 7 shapes", "8 or more shapes"]
    levels_ur = ["1 se 3 shakalien", "4 se 5 shakalien", "6 se 7 shakalien", "8 ya zyada shakalien"]
    n = len(objs)
    lvl = 0 if n <= 3 else 1 if n <= 5 else 2 if n <= 7 else 3
    qs.append({"q": "total", "primitive": "score",
               "instructions": "Tasveer mein kul kitni shakalien hain?" if ur else "How many shapes are in the image in total?",
               "criteria": levels_ur if ur else levels_en, "answer": lvl})
    uniq = _unique(objs)
    if uniq:
        o = rng.choice(uniq)
        # colour of a shape whose type is unique
        same_type = [x for x in objs if x["shape"] == o["shape"]]
        if len(same_type) == 1:
            cols = [o["colour"]] + rng.sample([c for c in COLOURS if c != o["colour"]], 3)
            qs.append({"q": "colour", "primitive": "choice",
                       "instructions": (f"Tasveer mein {SHAPES[o['shape']][1]} ka rang kya hai?" if ur
                                        else f"What colour is the {o['shape']}?"),
                       "criteria": {(COLOURS[c][1] if ur else c): None for c in sorted(cols)},
                       "answer": COLOURS[o["colour"]][1] if ur else o["colour"]})
        # quadrant, only when the whole shape sits inside one quadrant
        l, t, r_, b = o["bbox"]
        quad = None
        m, cx_, cy_ = 8, W / 2, H / 2  # margin keeps the answer unambiguous by eye
        if r_ < cx_ - m and b < cy_ - m: quad = "top-left"
        elif l > cx_ + m and b < cy_ - m: quad = "top-right"
        elif r_ < cx_ - m and t > cy_ + m: quad = "bottom-left"
        elif l > cx_ + m and t > cy_ + m: quad = "bottom-right"
        if quad:
            names_ur = {"top-left": "ooper baen", "top-right": "ooper daen",
                        "bottom-left": "neeche baen", "bottom-right": "neeche daen"}
            qs.append({"q": "quadrant", "primitive": "choice",
                       "instructions": (f"{_desc(o, lang)} tasveer ke kis hisse mein hai?" if ur
                                        else f"In which part of the image is the {_desc(o, lang)}?"),
                       "criteria": {(names_ur[x] if ur else x): None for x in names_ur},
                       "answer": names_ur[quad] if ur else quad})
        # Box target
        qs.append({"q": "box", "primitive": "box", "instructions": f"the {_desc(o, 'en')}",
                   "criteria": None, "answer": box_frac(o["bbox"], W, H)})
    # left/right relation between two unique shapes that do not overlap horizontally
    pairs = [(a, b) for a in uniq for b in uniq if a is not b and a["bbox"][2] + 4 < b["bbox"][0]]
    if pairs:
        a, b = rng.choice(pairs)  # a is fully left of b
        if rng.random() < 0.5:
            a, b, truth = b, a, False
        else:
            truth = True
        if ur:
            ins = f"Kya {_desc(a, lang)} {_desc(b, lang)} ke baen taraf hai?"
            neg = f"Kya {_desc(a, lang)} {_desc(b, lang)} ke daen taraf hai?"
        else:
            ins = f"Is the {_desc(a, lang)} to the left of the {_desc(b, lang)}?"
            neg = f"Is the {_desc(a, lang)} to the right of the {_desc(b, lang)}?"
        qs.append({"q": "leftof", "primitive": "noul", "instructions": ins, "criteria": None,
                   "answer": truth, "extra": {"negated_instructions": neg}})
    # largest shape, only when it is clearly largest (area ratio >= 1.4)
    if len(uniq) >= 3:
        cand = rng.sample(uniq, min(4, len(uniq)))
        cand.sort(key=lambda o: -o["area"])
        if cand[0]["area"] >= 1.4 * cand[1]["area"]:
            qs.append({"q": "largest", "primitive": "choice",
                       "instructions": ("In mein se sab se bari shakal kaun si hai?" if ur
                                        else "Which of these shapes is the largest?"),
                       "criteria": {_desc(o, lang): None for o in sorted(cand, key=lambda o: _desc(o, lang))},
                       "answer": _desc(cand[0], lang)})
    params = {"objects": [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in o.items()
                           if k not in ("bbox",)} for o in objs]}
    return {"image": img, "state": "An image is attached." if not ur else "Tasveer saath hai.",
            "params": params, "questions": qs}
