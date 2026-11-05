"""Line and pie charts with exact labels (append-only addition to the bank)."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from .common import box_frac, font

VERSION = "1"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
SLICE_COLOURS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 160, 60),
                 "yellow": (235, 195, 20), "purple": (140, 60, 180)}
SLICE_UR = {"red": "laal", "blue": "neela", "green": "hara", "yellow": "peela", "purple": "jamni"}


def line_chart_scene(rng: random.Random, lang: str) -> dict:
    W, H = 400, 300
    ur = lang == "ur-roman"
    n = rng.randint(5, 8)
    cats = MONTHS[:n]
    vals = [rng.randint(1, 20)]
    for _ in range(n - 1):
        vals.append(max(1, min(20, vals[-1] + rng.choice([-6, -4, -3, -2, 2, 3, 4, 6]))))
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = font(12)
    x0, y0, x1, y1 = 40, 20, W - 20, H - 35
    sy = (y1 - y0) / 20
    for v in range(0, 21, 5):
        yy = y1 - v * sy
        d.line([x0, yy, x1, yy], fill=(225, 225, 225))
        d.text((x0 - 8 - f.getlength(str(v)), yy - 7), str(v), fill=(80, 80, 80), font=f)
    d.line([x0, y1, x1, y1], fill=(60, 60, 60))
    step = (x1 - x0) / (n - 1)
    pts = [(x0 + i * step, y1 - v * sy) for i, v in enumerate(vals)]
    d.line(pts, fill=(42, 120, 214), width=3)
    for (x, y), c in zip(pts, cats):
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(42, 120, 214))
        d.text((x - f.getlength(c) / 2, y1 + 8), c, fill=(40, 40, 40), font=f)
    qs = []
    i = rng.randrange(n - 1)
    up = vals[i + 1] > vals[i]
    a, b = cats[i], cats[i + 1]
    ins = f"Kya {a} se {b} tak line ooper gayi?" if ur else f"Does the line go up from {a} to {b}?"
    neg = f"Kya {a} se {b} tak line neeche gayi?" if ur else f"Does the line go down from {a} to {b}?"
    qs.append({"q": "up", "primitive": "noul", "instructions": ins, "criteria": None, "answer": up,
               "extra": {"negated_instructions": neg}})
    order = sorted(range(n), key=lambda k: -vals[k])
    if vals[order[0]] > vals[order[1]]:
        qs.append({"q": "peak", "primitive": "choice",
                   "instructions": "Kis mahine mein value sab se zyada thi?" if ur else "In which month was the value highest?",
                   "criteria": {c: None for c in cats}, "answer": cats[order[0]]})
    k = rng.randrange(n)
    x, y = pts[k]
    qs.append({"q": "box", "primitive": "box", "instructions": f"the data point for {cats[k]}", "criteria": None,
               "answer": box_frac((x - 6, y - 6, x + 6, y + 6), W, H)})
    return {"image": img, "state": "Line chart saath hai." if ur else "A line chart is attached.",
            "params": {"categories": cats, "values": vals}, "questions": qs}


def pie_chart_scene(rng: random.Random, lang: str) -> dict:
    W, H = 320, 260
    ur = lang == "ur-roman"
    names = rng.sample(list(SLICE_COLOURS), rng.randint(3, 5))
    while True:
        shares = [rng.randint(5, 60) for _ in names]
        tot = sum(shares)
        pcts = [s / tot for s in shares]
        srt = sorted(pcts, reverse=True)
        if srt[0] - srt[1] >= 0.08:
            break
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    cx, cy, r = W / 2, H / 2, 100
    start = rng.uniform(0, 360)
    for nme, p in zip(names, pcts):
        end = start + p * 360
        d.pieslice([cx - r, cy - r, cx + r, cy + r], start, end, fill=SLICE_COLOURS[nme], outline="white", width=2)
        start = end
    big = names[max(range(len(names)), key=lambda i: pcts[i])]
    disp = (lambda c: SLICE_UR[c]) if ur else (lambda c: c)
    qs = [{"q": "largest", "primitive": "choice",
           "instructions": "Kaun sa rang pie ka sab se bara hissa hai?" if ur else "Which colour is the largest slice of the pie?",
           "criteria": {disp(c): None for c in sorted(names)}, "answer": disp(big)}]
    c = rng.choice(names)
    p = pcts[names.index(c)]
    qs.append({"q": "half", "primitive": "noul",
               "instructions": (f"Kya {disp(c)} hissa aadhe pie se zyada hai?" if ur else f"Is the {c} slice more than half of the pie?"),
               "criteria": None, "answer": p > 0.5})
    if abs(p - 0.5) < 0.04:
        qs.pop()  # too close to call by eye
    qs.append({"q": "box", "primitive": "box", "instructions": "the pie chart", "criteria": None,
               "answer": box_frac((cx - r, cy - r, cx + r, cy + r), W, H)})
    return {"image": img, "state": "Pie chart saath hai." if ur else "A pie chart is attached.",
            "params": {"slices": dict(zip(names, [round(x, 4) for x in pcts]))}, "questions": qs}


GENERATORS = {"line_chart": line_chart_scene, "pie_chart": pie_chart_scene}
