"""Demo GIFs: (1) a tour of the procedural generators with their exact labels, (2) the tiny
student answering test items with probabilities (and Box predictions vs truth).

python scripts/make_demo_gif.py --ckpt runs/tiny/student_kd/final.pt --temps runs/tiny/student_kd/temps.json
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from spark_one.generators.common import font, wrap
from spark_one.schema import load_items

W, H = 820, 460
INK, INK2, BLUE, ORANGE, GREEN, MAG = (11, 11, 11), (82, 81, 78), (42, 120, 214), (235, 104, 52), (27, 175, 122), (232, 123, 164)


def frame(it, root, probs=None, pred_box=None, title=""):
    im = Image.new("RGB", (W, H), (252, 252, 251))
    d = ImageDraw.Draw(im)
    d.text((16, 12), title, fill=INK2, font=font(13))
    x0 = 16
    if it.image:
        src = Image.open(root / it.image).convert("RGB")
        s = min(380 / src.width, 380 / src.height)
        src = src.resize((int(src.width * s), int(src.height * s)))
        sd = ImageDraw.Draw(src)
        if it.primitive == "box":
            b = it.answer
            sd.rectangle([b["left"] * src.width, b["top"] * src.height, b["right"] * src.width, b["bottom"] * src.height], outline=GREEN, width=3)
            if pred_box:
                b = pred_box
                sd.rectangle([b["left"] * src.width, b["top"] * src.height, b["right"] * src.width, b["bottom"] * src.height], outline=MAG, width=3)
        im.paste(src, (16, 44))
        x0 = 16 + src.width + 20
    else:
        y = 44
        for line in wrap(it.state.replace("\n", " "), font(15), 360)[:14]:
            d.text((16, y), line, fill=INK, font=font(15))
            y += 22
        x0 = 400
    y = 44
    d.text((x0, y), f"{it.primitive.upper()}  ·  {it.lang}", fill=BLUE, font=font(13))
    y += 24
    for line in wrap(it.instructions, font(16), W - x0 - 16)[:4]:
        d.text((x0, y), line, fill=INK, font=font(16))
        y += 22
    y += 10
    if it.primitive == "choice":
        opts = list(it.criteria)
    elif it.primitive == "noul":
        opts = ["Yes", "No"]
    elif it.primitive == "score":
        opts = list(it.criteria)
    else:
        opts = []
    truth = None
    if it.primitive == "choice":
        truth = None if it.none_fits else opts.index(it.answer)
    elif it.primitive == "noul":
        truth = 0 if it.answer else 1
    elif it.primitive == "score":
        truth = it.answer
    bw = W - x0 - 190
    for i, o in enumerate(opts):
        d.text((x0, y + 2), o[:26], fill=INK, font=font(14))
        if probs is not None:
            p = probs[i]
            d.rectangle([x0 + 175, y + 4, x0 + 175 + bw, y + 18], fill=(236, 235, 232))
            d.rectangle([x0 + 175, y + 4, x0 + 175 + max(2, int(bw * p)), y + 18], fill=BLUE)
            d.text((x0 + 180 + bw, y + 3), f"{p:.2f}", fill=INK2, font=font(12))
        if i == truth:
            d.text((x0 - 14, y + 2), "✓" if False else "•", fill=GREEN, font=font(16))
        y += 26
    if it.none_fits:
        d.text((x0, y + 4), "No option fits (the correct one was removed).", fill=ORANGE, font=font(13))
    if it.primitive == "box":
        d.text((x0, y), "green = exact target", fill=GREEN, font=font(14))
        if pred_box:
            d.text((x0, y + 22), "pink = model prediction", fill=MAG, font=font(14))
    d.text((16, H - 24), "labels computed in code; green dot = correct answer", fill=INK2, font=font(11))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="data/generated/bank")
    ap.add_argument("--ckpt")
    ap.add_argument("--temps")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    root = Path(a.bank)
    items = [it for it in load_items(root / "items.jsonl") if it.split == "test"]
    rng = random.Random(4)
    tour, seen = [], set()
    rng.shuffle(items)
    for it in items:
        key = (it.task, it.lang)
        if it.task.split(".")[0] in {t.split(".")[0] for t, _ in seen} and len(tour) < 14:
            continue
        seen.add(key)
        tour.append(it)
        if len(tour) >= 14:
            break
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    frames = [frame(it, root, title="Spark-1 question bank: procedural items with exact labels") for it in tour]
    frames[0].save(out / "demo_generators.gif", save_all=True, append_images=frames[1:], duration=1800, loop=0)
    if a.ckpt:
        from spark_one.calibrate import apply_temperature
        from spark_one.steps import decode_box
        from spark_one.tiny.predict import load_model, predict

        temps = json.loads(Path(a.temps).read_text()) if a.temps else {}
        model = load_model(a.ckpt)
        res = predict(model, tour, root)
        frames = []
        for it, (lps, _) in zip(tour, res):
            t = temps.get(it.primitive, {}).get("temperature", 1.0)
            ps = [np.exp(apply_temperature(lp, t)) for lp in lps]
            if it.primitive == "box":
                frames.append(frame(it, root, pred_box=decode_box(ps), title="Tiny Spark-1 student (1M params, CPU): calibrated answers"))
            else:
                frames.append(frame(it, root, probs=list(ps[0]), title="Tiny Spark-1 student (1M params, CPU): calibrated answers"))
        frames[0].save(out / "demo_tiny_student.gif", save_all=True, append_images=frames[1:], duration=1800, loop=0)
    print("ok")


if __name__ == "__main__":
    main()
