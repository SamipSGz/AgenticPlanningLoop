from __future__ import annotations

import os


class BudgetExceededError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Budget exceeded: {reason}")


class BudgetManager:
    def __init__(self) -> None:
        self.max_calls = int(os.environ.get("MAX_LLM_CALLS", "10"))
        self.max_cost = float(os.environ.get("MAX_COST_USD", "0.20"))
        self.price_per_1k_tokens = float(os.environ.get("MOCK_PRICE_PER_1K_TOKENS", "0.01"))
        self.calls_used = 0
        self.cost_used = 0.0

    def assert_can_call(self) -> None:
        if self.calls_used >= self.max_calls:
            raise BudgetExceededError(f"call limit reached ({self.calls_used}/{self.max_calls})")
        if self.cost_used >= self.max_cost:
            raise BudgetExceededError(f"cost limit reached (${self.cost_used:.4f}/${self.max_cost:.2f})")

    def record_call(self, input_tokens: int, output_tokens: int) -> float:
        self.calls_used += 1
        cost = ((input_tokens + output_tokens) / 1000) * self.price_per_1k_tokens
        self.cost_used += cost
        return cost

    def status(self) -> dict:
        return {
            "calls_used": self.calls_used,
            "calls_max": self.max_calls,
            "cost_used": round(self.cost_used, 6),
            "cost_max": self.max_cost,
        }
