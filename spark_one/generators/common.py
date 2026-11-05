"""Shared drawing helpers and vocabularies for the procedural generators.

Fonts come from Pillow's bundled default font so renders are identical on any machine
with the same Pillow version (recorded in the bank manifest)."""

from __future__ import annotations

import math
import random
from functools import lru_cache

from PIL import ImageDraw, ImageFont

COLOURS = {  # name -> (rgb, roman urdu)
    "red": ((220, 40, 40), "laal"),
    "green": ((40, 160, 60), "hara"),
    "blue": ((40, 90, 220), "neela"),
    "yellow": ((235, 195, 20), "peela"),
    "purple": ((140, 60, 180), "jamni"),
    "orange": ((245, 130, 20), "narangi"),
}
SHAPES = {  # name -> (plural, roman urdu)
    "circle": ("circles", "daira"),
    "square": ("squares", "chokor"),
    "triangle": ("triangles", "tikon"),
    "star": ("stars", "sitara"),
}
NAMES = ["Ali", "Sara", "Ahmed", "Ayesha", "Bilal", "Fatima", "Hamza", "Zainab", "Usman",
         "Hira", "Omar", "Maryam", "Danish", "Sana", "Imran", "Noor"]


@lru_cache(maxsize=None)
def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def shape_points(shape: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "triangle":
        return [(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "star":
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rr = r if i % 2 == 0 else r * 0.45
            pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
        return pts
    raise ValueError(shape)


def polygon_area(pts) -> float:
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def shape_area(shape: str, r: float) -> float:
    if shape == "circle":
        return math.pi * r * r
    return polygon_area(shape_points(shape, 0, 0, r))


def shape_bbox(shape: str, cx: float, cy: float, r: float) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) in pixels."""
    if shape == "circle":
        return cx - r, cy - r, cx + r, cy + r
    pts = shape_points(shape, cx, cy, r)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def draw_shape(d: ImageDraw.ImageDraw, shape: str, cx: float, cy: float, r: float, rgb) -> None:
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    else:
        d.polygon(shape_points(shape, cx, cy, r), fill=rgb)


def box_frac(bbox_px, w: int, h: int) -> dict:
    left, top, right, bottom = bbox_px
    clip = lambda v: max(0.0, min(1.0, v))
    return {"top": clip(top / h), "left": clip(left / w), "bottom": clip(bottom / h),
            "right": clip(right / w)}


def number_options(true: int, rng: random.Random, n: int = 4, lo: int = 0) -> dict:
    """n distinct integer options around the true value, sorted ascending."""
    cands = {true}
    spread = 1
    while len(cands) < n:
        v = true + rng.randint(-spread - 1, spread + 1)
        if v >= lo:
            cands.add(v)
        spread += 1
    return {str(v): None for v in sorted(cands)}


def wrap(text: str, f, max_w: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if f.getlength(t) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
