"""Optical-illusion eval set with exact ground truth (stretch goal).

Each scene is drawn from exact lengths/radii. One third are physically equal; in the rest
the true difference often runs against the illusion, so answering from the illusion fails."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

VERSION = "1"


def _three_way(rng):
    return rng.choice(["equal", "a", "b"])


def muller_lyer(rng: random.Random, lang: str) -> dict:
    W, H = 320, 240
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    truth = _three_way(rng)
    base = rng.uniform(120, 170)
    ratio = rng.uniform(1.12, 1.25)
    la = base * ratio if truth == "a" else base
    lb = base * ratio if truth == "b" else base
    fins_out_top = rng.random() < 0.5  # arrow-tails make a line look longer
    for idx, (y, length) in enumerate(((75, la), (165, lb))):
        cx = W / 2
        x0, x1 = cx - length / 2, cx + length / 2
        d.line([x0, y, x1, y], fill="black", width=3)
        outward = fins_out_top if idx == 0 else not fins_out_top
        s = 18 if outward else -18
        for x, sign in ((x0, -1), (x1, 1)):
            for dy in (-14, 14):
                d.line([x, y, x + sign * s, y + dy], fill="black", width=3)
    ur = lang == "ur-roman"
    opts = ({"ooper wali": None, "neeche wali": None, "dono barabar": None} if ur else
            {"top line": None, "bottom line": None, "both are the same length": None})
    keys = list(opts)
    ans = keys[{"a": 0, "b": 1, "equal": 2}[truth]]
    q = {"q": "longer", "primitive": "choice", "allow_none_fits": False,
         "instructions": ("Kaun si seedhi lakeer lambi hai (teer ke nishaan chhor kar)?" if ur
                          else "Ignoring the arrow fins, which horizontal line is longer?"),
         "criteria": opts, "answer": ans}
    illusion_says = "a" if fins_out_top else "b"
    return {"image": img, "state": "Tasveer saath hai." if ur else "An image is attached.",
            "params": {"len_top": round(la, 2), "len_bottom": round(lb, 2), "truth": truth,
                       "illusion_favours": illusion_says},
            "questions": [q]}


def ebbinghaus(rng: random.Random, lang: str) -> dict:
    W, H = 360, 220
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    truth = _three_way(rng)
    r = rng.uniform(16, 22)
    ratio = rng.uniform(1.12, 1.25)
    ra = r * ratio if truth == "a" else r
    rb = r * ratio if truth == "b" else r
    big_left = rng.random() < 0.5  # big surrounding circles make the centre look smaller
    for cx, rc, big in ((90, ra, big_left), (270, rb, not big_left)):
        cy = H / 2
        d.ellipse([cx - rc, cy - rc, cx + rc, cy + rc], fill=(240, 120, 30))
        n, sr, dist = (6, 24, rc + 36) if big else (8, 8, rc + 14)
        for i in range(n):
            a = 2 * math.pi * i / n
            x, y = cx + dist * math.cos(a), cy + dist * math.sin(a)
            d.ellipse([x - sr, y - sr, x + sr, y + sr], fill=(90, 110, 200))
    ur = lang == "ur-roman"
    opts = ({"baen wala": None, "daen wala": None, "dono barabar": None} if ur else
            {"left one": None, "right one": None, "both are the same size": None})
    keys = list(opts)
    ans = keys[{"a": 0, "b": 1, "equal": 2}[truth]]
    q = {"q": "larger", "primitive": "choice", "allow_none_fits": False,
         "instructions": ("Darmiyan wale do narangi dairon mein se kaun sa bara hai?" if ur
                          else "Which of the two orange centre circles is larger?"),
         "criteria": opts, "answer": ans}
    return {"image": img, "state": "Tasveer saath hai." if ur else "An image is attached.",
            "params": {"r_left": round(ra, 2), "r_right": round(rb, 2), "truth": truth,
                       "illusion_favours": "b" if big_left else "a"},
            "questions": [q]}


def ponzo(rng: random.Random, lang: str) -> dict:
    """Two horizontal bars between converging rails; the upper bar tends to look longer."""
    W, H = 320, 260
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    apex_x, apex_y = W / 2, 10
    for bx in (20, W - 20):
        d.line([apex_x, apex_y, bx, H - 10], fill="black", width=3)
    truth = _three_way(rng)
    base = rng.uniform(50, 64)
    ratio = rng.uniform(1.12, 1.25)
    la = base * ratio if truth == "a" else base  # a = upper bar
    lb = base * ratio if truth == "b" else base
    for y, length in ((80, la), (200, lb)):
        d.line([W / 2 - length / 2, y, W / 2 + length / 2, y], fill=(200, 40, 40), width=5)
    ur = lang == "ur-roman"
    opts = ({"ooper wali": None, "neeche wali": None, "dono barabar": None} if ur else
            {"upper bar": None, "lower bar": None, "both are the same length": None})
    keys = list(opts)
    q = {"q": "longer", "primitive": "choice", "allow_none_fits": False,
         "instructions": ("Do laal lakeeron mein se kaun si lambi hai?" if ur else "Which of the two red bars is longer?"),
         "criteria": opts, "answer": keys[{"a": 0, "b": 1, "equal": 2}[truth]]}
    return {"image": img, "state": "Tasveer saath hai." if ur else "An image is attached.",
            "params": {"len_upper": round(la, 2), "len_lower": round(lb, 2), "truth": truth, "illusion_favours": "a"},
            "questions": [q]}


def delboeuf(rng: random.Random, lang: str) -> dict:
    """Two equal-or-different discs; a tight ring makes a disc look larger, a wide ring smaller."""
    W, H = 360, 220
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    truth = _three_way(rng)
    r = rng.uniform(24, 30)
    ratio = rng.uniform(1.12, 1.22)
    ra = r * ratio if truth == "a" else r
    rb = r * ratio if truth == "b" else r
    tight_left = rng.random() < 0.5
    for cx, rc, tight in ((90, ra, tight_left), (270, rb, not tight_left)):
        cy = H / 2
        d.ellipse([cx - rc, cy - rc, cx + rc, cy + rc], fill=(30, 30, 30))
        ring = rc + (6 if tight else 34)
        d.ellipse([cx - ring, cy - ring, cx + ring, cy + ring], outline=(30, 30, 30), width=2)
    ur = lang == "ur-roman"
    opts = ({"baen wala": None, "daen wala": None, "dono barabar": None} if ur else
            {"left disc": None, "right disc": None, "both are the same size": None})
    keys = list(opts)
    q = {"q": "larger", "primitive": "choice", "allow_none_fits": False,
         "instructions": ("Do kaale dairon (bharay hue) mein se kaun sa bara hai?" if ur
                          else "Which of the two filled black discs is larger?"),
         "criteria": opts, "answer": keys[{"a": 0, "b": 1, "equal": 2}[truth]]}
    return {"image": img, "state": "Tasveer saath hai." if ur else "An image is attached.",
            "params": {"r_left": round(ra, 2), "r_right": round(rb, 2), "truth": truth,
                       "illusion_favours": "a" if tight_left else "b"},
            "questions": [q]}


GENERATORS = {"muller_lyer": muller_lyer, "ebbinghaus": ebbinghaus, "ponzo": ponzo, "delboeuf": delboeuf}
