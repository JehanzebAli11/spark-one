import random
from collections import defaultdict

from spark_one.bank import assign_split, build_task_items, make_none_fits, shuffle_options
from spark_one.generators import ILLUSIONS, REGISTRY
from spark_one.schema import load_items


def test_all_items_valid_and_have_provenance(small_bank):
    items = load_items(small_bank / "items.jsonl")
    assert len(items) > 300
    for it in items:
        it.validate()
        assert it.provenance["label_source"] == "code"
        if it.image:
            assert (small_bank / it.image).exists()
    prims = {it.primitive for it in items}
    assert prims == {"choice", "noul", "score", "box"}
    assert {it.lang for it in items} == {"en", "ur-roman"}


def test_generation_is_deterministic(tmp_path):
    fn, ver, langs = REGISTRY["compare_people"]
    a = build_task_items("compare_people", fn, ver, langs, 5, 9, None)
    b = build_task_items("compare_people", fn, ver, langs, 5, 9, None)
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    fn, ver, langs = REGISTRY["shapes"]
    a = build_task_items("shapes", fn, ver, ("en",), 3, 9, tmp_path / "a")
    b = build_task_items("shapes", fn, ver, ("en",), 3, 9, tmp_path / "b")
    assert [x.to_dict() for x in a] == [x.to_dict() for x in b]
    for x in a:
        assert (tmp_path / "a" / x.image.split("/", 1)[1]).read_bytes() == (tmp_path / "b" / x.image.split("/", 1)[1]).read_bytes()


def test_splits_are_grouped_by_scene(small_bank):
    groups = defaultdict(set)
    for it in load_items(small_bank / "items.jsonl"):
        groups[it.provenance["group"]].add(it.split)
    assert all(len(s) == 1 for s in groups.values())
    counts = defaultdict(int)
    for i in range(5000):
        counts[assign_split(f"g{i}")] += 1
    assert 0.77 < counts["train"] / 5000 < 0.83


def test_none_fits_and_companion(small_bank):
    items = {it.id: it for it in load_items(small_bank / "items.jsonl")}
    nf = [it for it in items.values() if it.none_fits]
    assert nf
    for it in nf:
        assert it.answer is None and it.provenance["removed_answer"] not in it.criteria
        comp = items[it.id + "-any"]
        assert comp.primitive == "noul" and comp.answer is False
    pos = [it for it in items.values() if it.task.endswith("_anyfits") and it.answer is True]
    assert pos


def test_shuffle_keeps_answer():
    fn, ver, langs = REGISTRY["day_after"]
    it = build_task_items("day_after", fn, ver, ("en",), 1, 1, None)[0]
    rng = random.Random(0)
    for _ in range(10):
        s = shuffle_options(it, rng)
        assert set(s.criteria) == set(it.criteria) and s.answer == it.answer
    if not it.none_fits:
        nf = make_none_fits(it)
        assert nf.answer is None and len(nf.criteria) == len(it.criteria) - 1


def test_labels_match_params(small_bank):
    for it in load_items(small_bank / "items.jsonl"):
        p = it.provenance["params"]
        if it.task == "compare_people.more" and not it.none_fits:
            x, y = p["x"], p["y"]
            if x > y:
                assert it.answer == p["a"]
            elif y > x:
                assert it.answer == p["b"]
        if it.task == "shapes.count" and not it.none_fits:
            shape = [s for s in ("circle", "square", "triangle", "star") if s in it.instructions or
                     {"circle": "daira", "square": "chokor", "triangle": "tikon", "star": "sitara"}[s] in it.instructions][0]
            assert int(it.answer) == sum(o["shape"] == shape for o in p["objects"])
        if it.task == "bar_chart.level":
            assert 0 <= it.answer <= 3
        if it.task == "compare_decimals.q":
            assert it.answer == (p["a"] > p["b"])


def test_illusions_have_exact_truth(tmp_path):
    for name, (fn, ver, langs) in ILLUSIONS.items():
        its = build_task_items(name, fn, ver, langs, 12, 3, tmp_path)
        for it in its:
            p = it.provenance["params"]
            keys = list(it.criteria)
            vals = (p.get("len_top"), p.get("len_bottom")) if "len_top" in p else (p["r_left"], p["r_right"])
            if vals[0] == vals[1]:
                assert p["truth"] == "equal"
            assert p["truth"] in ("equal", "a", "b")
            assert it.answer in keys
