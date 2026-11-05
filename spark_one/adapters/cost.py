"""A hard cost cap shared by the eval adapters."""

from __future__ import annotations

import threading


class BudgetExceeded(RuntimeError):
    pass


class CostTracker:
    def __init__(self, max_usd: float):
        self.max_usd = float(max_usd)
        self.spent = 0.0
        self.calls = 0
        self.max_call = 0.0  # largest single-call cost seen; used as the next call's estimate
        self.lock = threading.Lock()

    def check(self, upcoming_estimate: float = 0.0) -> None:
        with self.lock:
            if self.spent + max(upcoming_estimate, self.max_call) > self.max_usd:
                raise BudgetExceeded(f"cost cap ${self.max_usd:.2f} reached (spent ${self.spent:.4f})")

    def add(self, usd: float) -> None:
        with self.lock:
            self.spent += float(usd)
            self.max_call = max(self.max_call, float(usd))
            self.calls += 1

    def summary(self) -> dict:
        return {"max_usd": self.max_usd, "spent_usd": round(self.spent, 6), "calls": self.calls}
