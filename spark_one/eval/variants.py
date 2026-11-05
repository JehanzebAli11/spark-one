"""Expand eval items into variants: base, option permutations (order consistency) and
negation twins for Nouls (self-consistency). Every backend predicts the same list."""

from __future__ import annotations

from dataclasses import replace

from ..bank import permute_options
from ..schema import Item


def expand(items: list[Item], perms: bool = True, negations: bool = True) -> list[tuple[str, str, list[int] | None, Item]]:
    """Returns (variant_id, variant, order, item). order maps presented position -> base position."""
    out = []
    for it in items:
        out.append((it.id, "base", None, it))
        if perms and it.primitive == "choice" and not it.none_fits and len(it.criteria) >= 3:
            n = len(it.criteria)
            for k, shift in ((1, 1), (2, 2)):
                order = [(i + shift) % n for i in range(n)]
                out.append((f"{it.id}~perm{k}", f"perm{k}", order, permute_options(it, order)))
        neg = it.extra.get("negated_instructions") if it.primitive == "noul" else None
        if negations and neg:
            out.append((f"{it.id}~neg", "neg", None,
                        replace(it, id=f"{it.id}~neg", instructions=neg, answer=not it.answer)))
    return out
