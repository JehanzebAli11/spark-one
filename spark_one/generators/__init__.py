"""Registry of procedural generators: name -> (function, version, languages)."""

from . import charts2, illusions, screens, shapes, text

LANGS = ("en", "ur-roman")

REGISTRY = {}
for _mod in (text, screens, charts2):
    for _name, _fn in _mod.GENERATORS.items():
        REGISTRY[_name] = (_fn, _mod.VERSION, LANGS)
REGISTRY["shapes"] = (shapes.shapes_scene, shapes.VERSION, LANGS)

# Eval-only stretch set: never used for training.
ILLUSIONS = {n: (f, illusions.VERSION, LANGS) for n, f in illusions.GENERATORS.items()}
