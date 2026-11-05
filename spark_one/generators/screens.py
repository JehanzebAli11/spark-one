"""Rendered chat and app screenshots and bar charts, with exact labels and Box targets."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from .common import NAMES, box_frac, font, number_options, wrap

VERSION = "1"

MSG_EN = ["Are we still on for lunch?", "Running 10 minutes late", "Can you send the report?",
          "Happy birthday!", "Let's order pizza tonight", "The meeting moved to 3pm",
          "Did you see the cricket match?", "I'll call you after class", "Thanks a lot!",
          "Where should we meet?", "My train is delayed", "Sounds good to me",
          "Please bring the charger", "Who is coming to the wedding?", "See you tomorrow"]
MSG_UR = ["Kya hum lunch pe mil rahe hain?", "Main 10 minute late hun", "Report bhej do please",
          "Saalgirah mubarak!", "Aaj raat pizza mangwate hain", "Meeting 3 baje shift ho gayi",
          "Cricket match dekha?", "Class ke baad call karta hun", "Bohat shukriya!",
          "Kahan milna hai?", "Meri train late hai", "Theek hai, mujhe manzoor hai",
          "Charger saath le aana", "Shaadi pe kaun aa raha hai?", "Kal milte hain"]
KEYWORDS = {"pizza": ("pizza",), "train": ("train",), "charger": ("charger",),
            "meeting": ("meeting",), "cricket": ("cricket",), "lunch": ("lunch",)}


def chat_scene(rng: random.Random, lang: str) -> dict:
    W, H = 360, 600
    ur = lang == "ur-roman"
    pool = MSG_UR if ur else MSG_EN
    group = rng.random() < 0.5
    people = rng.sample(NAMES, 3 if group else 1)
    senders_pool = people + ["You"]
    img = Image.new("RGB", (W, H), (236, 229, 221))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=(7, 94, 84))
    title = (", ".join(people) if group else people[0])
    d.text((16, 18), title, fill="white", font=font(18))
    f, fs = font(15), font(11)
    y, msgs = 68, []
    n_target = rng.randint(3, 8)
    while len(msgs) < n_target:
        sender = rng.choice(senders_pool)
        text = rng.choice(pool)
        lines = wrap(text, f, 220)
        name_h = 16 if (group and sender != "You") else 0
        bh = 12 + 20 * len(lines) + name_h
        if y + bh + 10 > H - 50:
            break
        bw = max(f.getlength(l) for l in lines) + 24
        if group and sender != "You":
            bw = max(bw, fs.getlength(sender) + 24)
        x0 = W - 12 - bw if sender == "You" else 12
        rect = (x0, y, x0 + bw, y + bh)
        d.rounded_rectangle(rect, 10, fill=(220, 248, 198) if sender == "You" else (255, 255, 255))
        ty = y + 6
        if name_h:
            d.text((x0 + 12, ty), sender, fill=(200, 60, 60), font=fs)
            ty += name_h
        for l in lines:
            d.text((x0 + 12, ty), l, fill=(20, 20, 20), font=f)
            ty += 20
        msgs.append({"sender": sender, "text": text, "rect": rect})
        y += bh + 10
    d.rectangle([0, H - 44, W, H], fill=(245, 245, 245))
    d.text((16, H - 32), "Message" if not ur else "Paigham likhein", fill=(150, 150, 150), font=f)
    qs = []
    last = msgs[-1]["sender"]
    qs.append({"q": "last", "primitive": "choice",
               "instructions": "Aakhri paigham kis ne bheja?" if ur else "Who sent the last message?",
               "criteria": {s: ("the phone's owner (right side, green)" if s == "You" else None)
                            for s in senders_pool}, "answer": last, "allow_none_fits": len(senders_pool) >= 3})
    who = rng.choice(senders_pool)
    k = sum(m["sender"] == who for m in msgs)
    qs.append({"q": "count", "primitive": "choice",
               "instructions": ((f"{who} ne kitne paigham bheje?" if who != "You" else "Aap (phone ke maalik) ne kitne paigham bheje?") if ur
                                else (f"How many messages did {who} send?" if who != "You" else "How many messages did you (the phone's owner) send?")),
               "criteria": number_options(k, rng), "answer": str(k)})
    kw = rng.choice(list(KEYWORDS))
    present = any(kw in m["text"].lower() for m in msgs)
    qs.append({"q": "mention", "primitive": "noul",
               "instructions": (f"Kya chat mein kisi ne '{kw}' ka zikr kiya?" if ur
                                else f"Does any message in the chat mention '{kw}'?"),
               "criteria": None, "answer": present})
    texts = [m["text"] for m in msgs]
    uniq = [m for m in msgs if texts.count(m["text"]) == 1]
    if uniq:
        m = rng.choice(uniq)
        qs.append({"q": "box", "primitive": "box",
                   "instructions": f"the message bubble that says \"{m['text']}\"",
                   "criteria": None, "answer": box_frac(m["rect"], W, H)})
    params = {"messages": [{"sender": m["sender"], "text": m["text"]} for m in msgs], "group": group}
    return {"image": img, "state": "A chat screenshot is attached." if not ur else "Chat ka screenshot saath hai.",
            "params": params, "questions": qs}


SETTINGS = ["Wi-Fi", "Bluetooth", "Airplane mode", "Notifications", "Dark mode", "Location",
            "Auto-rotate", "Battery saver", "Mobile data", "Hotspot"]


def settings_scene(rng: random.Random, lang: str) -> dict:
    W, H = 360, 600
    ur = lang == "ur-roman"
    img = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=(33, 33, 33))
    d.text((16, 18), "Settings", fill="white", font=font(20))
    rows = rng.sample(SETTINGS, rng.randint(4, 8))
    states = {r: rng.random() < 0.5 for r in rows}
    f = font(16)
    y = 76
    toggles = {}
    for r in rows:
        d.text((20, y + 10), r, fill=(30, 30, 30), font=f)
        on = states[r]
        tx = W - 76
        rect = (tx, y + 8, tx + 52, y + 36)
        d.rounded_rectangle(rect, 14, fill=(52, 199, 89) if on else (200, 200, 200))
        kx = tx + 38 if on else tx + 14
        d.ellipse([kx - 12, y + 10, kx + 12, y + 34], fill="white")
        toggles[r] = rect
        d.line([16, y + 46, W - 16, y + 46], fill=(225, 225, 225))
        y += 52
    btn = (60, y + 30, W - 60, y + 76)
    label = rng.choice(["Save", "Apply", "Done"])
    d.rounded_rectangle(btn, 10, fill=(10, 132, 255))
    d.text(((W - f.getlength(label)) / 2, btn[1] + 14), label, fill="white", font=f)
    qs = []
    s = rng.choice(rows)
    if ur:
        ins, neg = f"Kya '{s}' on hai?", f"Kya '{s}' off hai?"
    else:
        ins, neg = f"Is '{s}' turned on?", f"Is '{s}' turned off?"
    qs.append({"q": "toggle", "primitive": "noul", "instructions": ins, "criteria": None,
               "answer": states[s], "extra": {"negated_instructions": neg}})
    k = sum(states.values())
    qs.append({"q": "count_on", "primitive": "choice",
               "instructions": "Kitni settings on hain?" if ur else "How many settings are turned on?",
               "criteria": number_options(k, rng), "answer": str(k)})
    t = rng.choice(rows)
    qs.append({"q": "box_toggle", "primitive": "box", "instructions": f"the toggle switch for '{t}'",
               "criteria": None, "answer": box_frac(toggles[t], W, H)})
    qs.append({"q": "box_button", "primitive": "box", "instructions": f"the '{label}' button",
               "criteria": None, "answer": box_frac(btn, W, H)})
    return {"image": img, "state": "A phone settings screenshot is attached." if not ur else "Phone settings ka screenshot saath hai.",
            "params": {"rows": rows, "states": states, "button": label}, "questions": qs}


PRODUCTS = ["Tea", "Rice", "Soap", "Bread", "Milk", "Eggs", "Juice", "Biscuits", "Butter", "Honey"]


def cart_scene(rng: random.Random, lang: str) -> dict:
    W, H = 360, 520
    ur = lang == "ur-roman"
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 56], fill=(255, 111, 0))
    d.text((16, 18), "Your cart", fill="white", font=font(20))
    f = font(16)
    names = rng.sample(PRODUCTS, rng.randint(2, 5))
    prices = rng.sample(range(50, 990, 10), len(names))
    y = 76
    for n, p in zip(names, prices):
        d.rounded_rectangle([16, y, W - 16, y + 50], 8, outline=(220, 220, 220), width=2)
        d.text((30, y + 16), n, fill=(30, 30, 30), font=f)
        t = f"Rs {p}"
        d.text((W - 30 - f.getlength(t), y + 16), t, fill=(30, 30, 30), font=f)
        y += 62
    total = sum(prices)
    qs = []
    top = names[prices.index(max(prices))]
    if len(names) >= 3:
        qs.append({"q": "priciest", "primitive": "choice",
                   "instructions": "Cart mein sab se mehngi cheez kaun si hai?" if ur else "Which item in the cart is the most expensive?",
                   "criteria": {n: None for n in sorted(names)}, "answer": top})
    opts = number_options(total // 10, rng)
    qs.append({"q": "total", "primitive": "choice",
               "instructions": "Cart ki kul qeemat kya hai?" if ur else "What is the total price of the cart?",
               "criteria": {f"Rs {int(k) * 10}": None for k in opts}, "answer": f"Rs {total}"})
    return {"image": img, "state": "A shopping cart screenshot is attached." if not ur else "Shopping cart ka screenshot saath hai.",
            "params": {"items": dict(zip(names, prices))}, "questions": qs}


CATS = {"months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
        "fruits": ["Apple", "Mango", "Grape", "Peach", "Plum", "Lime"],
        "cities": ["Lahore", "Karachi", "Quetta", "Multan", "Sialkot", "Peshawar"]}


def bar_chart_scene(rng: random.Random, lang: str) -> dict:
    W, H = 400, 300
    ur = lang == "ur-roman"
    kind = rng.choice(list(CATS))
    cats = CATS[kind][: rng.randint(3, 6)]
    vals = [rng.randint(1, 20) for _ in cats]
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    f = font(12)
    x0, y0, x1, y1 = 40, 20, W - 15, H - 35  # plot area
    scale = (y1 - y0) / 20
    for v in range(0, 21, 5):
        yy = y1 - v * scale
        d.line([x0, yy, x1, yy], fill=(225, 225, 225))
        d.text((x0 - 8 - f.getlength(str(v)), yy - 7), str(v), fill=(80, 80, 80), font=f)
    d.line([x0, y0, x0, y1], fill=(60, 60, 60))
    d.line([x0, y1, x1, y1], fill=(60, 60, 60))
    slot = (x1 - x0) / len(cats)
    colour = rng.choice([(70, 130, 180), (220, 120, 50), (90, 160, 90), (150, 90, 170)])
    rects = {}
    for i, (c, v) in enumerate(zip(cats, vals)):
        bx0 = x0 + i * slot + slot * 0.2
        bx1 = x0 + (i + 1) * slot - slot * 0.2
        rect = (bx0, y1 - v * scale, bx1, y1)
        d.rectangle(rect, fill=colour)
        rects[c] = rect
        d.text(((bx0 + bx1) / 2 - f.getlength(c) / 2, y1 + 8), c, fill=(40, 40, 40), font=f)
    qs = []
    order = sorted(range(len(cats)), key=lambda i: -vals[i])
    if vals[order[0]] - vals[order[1]] >= 2:
        qs.append({"q": "max", "primitive": "choice",
                   "instructions": "Kis bar ki value sab se zyada hai?" if ur else "Which bar has the highest value?",
                   "criteria": {c: None for c in cats}, "answer": cats[order[0]]})
    pairs = [(i, j) for i in range(len(cats)) for j in range(len(cats)) if i != j and abs(vals[i] - vals[j]) >= 2]
    if pairs:
        i, j = rng.choice(pairs)
        a, b = cats[i], cats[j]
        if ur:
            ins, neg = f"Kya {a} ki value {b} se zyada hai?", f"Kya {a} ki value {b} se kam hai?"
        else:
            ins, neg = f"Is the value for {a} higher than for {b}?", f"Is the value for {a} lower than for {b}?"
        qs.append({"q": "compare", "primitive": "noul", "instructions": ins, "criteria": None,
                   "answer": vals[i] > vals[j], "extra": {"negated_instructions": neg}})
    c = rng.choice(range(len(cats)))
    v = vals[c]
    qs.append({"q": "level", "primitive": "score",
               "instructions": (f"{cats[c]} ki value kitni hai?" if ur else f"What is the value of the {cats[c]} bar?"),
               "criteria": ["1 to 5", "6 to 10", "11 to 15", "16 to 20"], "answer": (v - 1) // 5})
    t = rng.choice(cats)
    qs.append({"q": "box", "primitive": "box", "instructions": f"the bar for {t}", "criteria": None,
               "answer": box_frac(rects[t], W, H)})
    return {"image": img, "state": "A bar chart is attached." if not ur else "Bar chart saath hai.",
            "params": {"categories": cats, "values": vals}, "questions": qs}


GENERATORS = {"chat": chat_scene, "settings": settings_scene, "cart": cart_scene,
              "bar_chart": bar_chart_scene}
