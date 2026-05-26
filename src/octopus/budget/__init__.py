"""
Cognitive Budget System — cost-aware brain upgrade control.

Evaluates whether the expected gain of routing a task to a more
expensive brain justifies the additional token cost and latency.
Prevents budget overruns by enforcing monthly and per-task limits.
"""

from .cognitive_budget import CognitiveBudget, BudgetDecision

__all__ = ["CognitiveBudget", "BudgetDecision"]
