"""Text-only tasks in English and Roman Urdu with labels computed in code.

Parallel templates exist in both languages so the two slices measure the same skills."""

from __future__ import annotations

import datetime as dt
import random

from .common import NAMES, number_options

VERSION = "1"
WORDS = ["strawberry", "banana", "committee", "bookkeeper", "mississippi", "parallel",
         "assessment", "balloon", "coffee", "pepper", "success", "letter", "referee",
         "karachi", "lahore", "islamabad", "barabar", "khushboo", "intezaar", "mehmaan",
         "parrot", "tomorrow", "occurrence", "necessary", "accommodate", "address"]
ITEMS = {  # english, roman urdu plural
    "books": "kitaabein", "apples": "seb", "bananas": "kele", "pens": "qalam", "balls": "gendein",
    "mangoes": "aam", "eggs": "anday",
}
GROCERIES = {"potatoes": "aloo", "onions": "pyaaz", "tomatoes": "tamatar", "milk": "doodh",
             "eggs": "anday", "rice": "chawal", "flour": "aata", "sugar": "cheeni",
             "lentils": "daal", "tea": "chai"}
DAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAYS_UR = ["Peer", "Mangal", "Budh", "Jumeraat", "Jumma", "Hafta", "Itwaar"]
FILLER_EN = ["The weather was pleasant.", "Traffic was slow that day.", "Everyone had tea.",
             "The shop opened at nine.", "It was a busy week."]
FILLER_UR = ["Mausam acha tha.", "Us din traffic bohat tha.", "Sab ne chai pi.",
             "Dukaan nau baje khuli.", "Hafta kaafi masroof tha."]


def _text_scene(state, qs, params):
    return {"image": None, "state": state, "params": params, "questions": qs}


def count_letter(rng: random.Random, lang: str) -> dict:
    w = rng.choice(WORDS)
    ch = rng.choice(sorted(set(w)) + ["z", "q"] * (rng.random() < 0.1))
    k = w.count(ch)
    ins = (f"Lafz '{w}' mein harf '{ch}' kitni baar aata hai?" if lang == "ur-roman"
           else f"How many times does the letter '{ch}' appear in the word '{w}'?")
    q = {"q": "q", "primitive": "choice", "instructions": ins, "criteria": number_options(k, rng),
         "answer": str(k)}
    return _text_scene(f"Word: {w}", [q], {"word": w, "letter": ch})


def compare_people(rng: random.Random, lang: str) -> dict:
    a, b = rng.sample(NAMES, 2)
    item_en = rng.choice(list(ITEMS))
    x = rng.randint(1, 20)
    y = x if rng.random() < 0.2 else rng.choice([v for v in range(1, 21) if v != x])
    filler = rng.sample(FILLER_UR if lang == "ur-roman" else FILLER_EN, rng.randint(0, 2))
    if lang == "ur-roman":
        it = ITEMS[item_en]
        facts = [f"{a} ke paas {x} {it} hain.", f"{b} ke paas {y} {it} hain."]
        eq = "Dono ke paas barabar"
        ins = f"Kis ke paas zyada {it} hain?"
    else:
        facts = [f"{a} has {x} {item_en}.", f"{b} has {y} {item_en}."]
        eq = "They have the same number"
        ins = f"Who has more {item_en}?"
    sents = facts + filler
    rng.shuffle(sents)
    ans = a if x > y else b if y > x else eq
    q = {"q": "more", "primitive": "choice", "instructions": ins,
         "criteria": {a: None, b: None, eq: None}, "answer": ans, "allow_none_fits": False}
    # Noul with a negation twin (self-consistency)
    if x != y:
        if lang == "ur-roman":
            ins2, neg = f"Kya {a} ke paas {b} se zyada {it} hain?", f"Kya {a} ke paas {b} se kam {it} hain?"
        else:
            ins2, neg = f"Does {a} have more {item_en} than {b}?", f"Does {a} have fewer {item_en} than {b}?"
        n = {"q": "noul", "primitive": "noul", "instructions": ins2, "criteria": None,
             "answer": x > y, "extra": {"negated_instructions": neg}}
        qs = [q, n]
    else:
        qs = [q]
    return _text_scene(" ".join(sents), qs, {"a": a, "b": b, "x": x, "y": y, "item": item_en})


def day_after(rng: random.Random, lang: str) -> dict:
    days = DAYS_UR if lang == "ur-roman" else DAYS_EN
    i, k = rng.randrange(7), rng.randint(1, 10)
    ans = days[(i + k) % 7]
    opts = {ans} | set(rng.sample([d for d in days if d != ans], 3))
    if lang == "ur-roman":
        state, ins = f"Aaj {days[i]} hai.", f"{k} din baad kaun sa din hoga?"
    else:
        state, ins = f"Today is {days[i]}.", f"What day will it be {k} days from now?"
    q = {"q": "q", "primitive": "choice", "instructions": ins,
         "criteria": {d: None for d in days if d in opts}, "answer": ans}
    return _text_scene(state, [q], {"today": i, "k": k})


def count_list(rng: random.Random, lang: str) -> dict:
    keys = rng.sample(list(GROCERIES), rng.randint(1, 8))
    names = [GROCERIES[k] for k in keys] if lang == "ur-roman" else keys
    listed = ", ".join(names[:-1]) + (f" aur {names[-1]}" if lang == "ur-roman" else f" and {names[-1]}") if len(names) > 1 else names[0]
    k = len(keys)
    if lang == "ur-roman":
        state, ins = f"Main ne bazaar se {listed} kharide.", "Kitni mukhtalif cheezein kharidi gayin?"
    else:
        state, ins = f"I bought {listed} at the market.", "How many different things were bought?"
    q = {"q": "q", "primitive": "choice", "instructions": ins, "criteria": number_options(k, rng, lo=1),
         "answer": str(k)}
    return _text_scene(state, [q], {"items": keys})


def stock_score(rng: random.Random, lang: str) -> dict:
    item_en = rng.choice(list(ITEMS))
    n = rng.choice([0, rng.randint(1, 3), rng.randint(4, 10), rng.randint(11, 40)])
    lvl = 0 if n == 0 else 1 if n <= 3 else 2 if n <= 10 else 3
    if lang == "ur-roman":
        it = ITEMS[item_en]
        state = f"Dukaan mein {n} {it} bache hain." if n else f"Dukaan mein {it} khatam ho gaye hain."
        levels = ["0 (khatam)", "1 se 3 (bohat kam)", "4 se 10 (kuch)", "11 ya zyada (bohat)"]
        ins = f"Dukaan mein kitne {it} bache hain?"
    else:
        state = f"The shop has {n} {item_en} left." if n else f"The shop has run out of {item_en}."
        levels = ["0 (none)", "1 to 3 (very few)", "4 to 10 (some)", "11 or more (many)"]
        ins = f"How many {item_en} does the shop have left?"
    q = {"q": "q", "primitive": "score", "instructions": ins, "criteria": levels, "answer": lvl}
    return _text_scene(state, [q], {"n": n, "item": item_en})


def compare_decimals(rng: random.Random, lang: str) -> dict:
    # deliberately includes 9.11-vs-9.9 style traps
    a = round(rng.randint(1, 20) + rng.choice([rng.randint(1, 99) / 100, rng.randint(1, 9) / 10]), 2)
    b = a
    while b == a:
        b = round(int(a) + rng.choice([rng.randint(1, 99) / 100, rng.randint(1, 9) / 10]) + rng.choice([0, 0, 1, -1]), 2)
    if lang == "ur-roman":
        ins, neg = f"Kya {a} {b} se bara hai?", f"Kya {a} {b} se chhota hai?"
        state = f"Do numbers: {a} aur {b}."
    else:
        ins, neg = f"Is {a} greater than {b}?", f"Is {a} less than {b}?"
        state = f"Two numbers: {a} and {b}."
    q = {"q": "q", "primitive": "noul", "instructions": ins, "criteria": None, "answer": a > b,
         "extra": {"negated_instructions": neg}}
    return _text_scene(state, [q], {"a": a, "b": b})


def date_order(rng: random.Random, lang: str) -> dict:
    base = dt.date(2020, 1, 1)
    ds = set()
    while len(ds) < 3:
        ds.add(base + dt.timedelta(days=rng.randint(0, 2500)))
    ds = list(ds)
    fmts = ["%d %B %Y", "%Y-%m-%d", "%B %d, %Y", "%d/%m/%Y"]
    names = rng.sample(NAMES, 3)
    rows = [f"{n}: {d.strftime(rng.choice(fmts))}" for n, d in zip(names, ds)]
    earliest = names[min(range(3), key=lambda i: ds[i])]
    if lang == "ur-roman":
        state = "Har shakhs ki interview ki tareekh (din/mahina/saal jahan slash ho):\n" + "\n".join(rows)
        ins = "Kis ka interview sab se pehle hai?"
    else:
        state = "Interview date for each person (day/month/year where slashes are used):\n" + "\n".join(rows)
        ins = "Whose interview comes first?"
    q = {"q": "q", "primitive": "choice", "instructions": ins, "criteria": {n: None for n in names},
         "answer": earliest}
    return _text_scene(state, [q], {"dates": [d.isoformat() for d in ds]})


def arithmetic(rng: random.Random, lang: str) -> dict:
    a, b = rng.randint(2, 60), rng.randint(2, 60)
    price = rng.choice(["Rs", "$"])
    total = a + b
    if lang == "ur-roman":
        state = f"Ek cheez {price} {a} ki aur doosri {price} {b} ki hai."
        ins = "Dono ki kul qeemat kitni hai?"
    else:
        state = f"One item costs {price} {a} and another costs {price} {b}."
        ins = "What is the total cost of both?"
    opts = number_options(total, rng)
    q = {"q": "q", "primitive": "choice", "instructions": ins,
         "criteria": {f"{price} {k}": None for k in opts}, "answer": f"{price} {total}"}
    return _text_scene(state, [q], {"a": a, "b": b})


GENERATORS = {
    "count_letter": count_letter, "compare_people": compare_people, "day_after": day_after,
    "count_list": count_list, "stock_score": stock_score, "compare_decimals": compare_decimals,
    "date_order": date_order, "arithmetic": arithmetic,
}
