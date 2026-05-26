"""Real-time cost tracking and budget management for Octopus.

Tracks every API call's token usage and cost, persists records to disk,
and provides budget-aware statistics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from octopus.config import BudgetConfig


@dataclass
class CostRecord:
    """A single API call cost record."""

    timestamp: str  # ISO 8601
    api_name: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    task_id: Optional[str] = None

    @classmethod
    def now(cls, api_name: str, model: str, tokens_in: int, tokens_out: int, cost_usd: float, task_id: Optional[str] = None) -> "CostRecord":
        """Create a cost record with the current UTC timestamp."""
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            api_name=api_name,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            task_id=task_id,
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "api_name": self.api_name,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CostRecord":
        return cls(**d)


@dataclass
class CostStats:
    """Aggregated cost statistics."""

    total_cost_usd: float = 0.0
    total_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    monthly_budget: float = 0.0
    remaining_budget: float = 0.0
    budget_used_pct: float = 0.0
    by_api: dict[str, float] = field(default_factory=dict)
    by_date: dict[str, float] = field(default_factory=dict)
    by_task: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_cost_usd": self.total_cost_usd,
            "total_calls": self.total_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "monthly_budget": self.monthly_budget,
            "remaining_budget": self.remaining_budget,
            "budget_used_pct": self.budget_used_pct,
            "by_api": self.by_api,
            "by_date": self.by_date,
            "by_task": self.by_task,
        }


class CostTracker:
    """Tracks API call costs with persistence to workspace/costs.json.

    Features:
    - Per-call cost recording with token counts
    - Monthly budget tracking with warning thresholds
    - Multi-dimensional statistics (by API, by date, by task)
    - JSON persistence for crash recovery
    - Export to JSON report
    """

    def __init__(self, budget: BudgetConfig, storage_path: Optional[Path] = None):
        self.budget = budget
        self._records: list[CostRecord] = []
        self._storage_path = storage_path

        if storage_path is not None:
            self.load()

    # ── Recording ────────────────────────────────────────────────────────

    def record_call(
        self,
        api_name: str,
        tokens_in: int,
        tokens_out: int,
        model: str,
        task_id: Optional[str] = None,
    ) -> CostRecord:
        """Record a single API call and return the cost record.

        Cost is calculated using the budget config (simplified: we use
        a default rate since we don't have per-API price info here).
        In production, you'd look up the APIConfig for per-model pricing.
        """
        # Simplified cost calculation: 0.001 per 1K input, 0.002 per 1K output
        # (reasonable defaults for most providers)
        cost_input = (tokens_in / 1000.0) * 0.001
        cost_output = (tokens_out / 1000.0) * 0.002
        cost_usd = round(cost_input + cost_output, 8)

        record = CostRecord.now(
            api_name=api_name,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            task_id=task_id,
        )
        self._records.append(record)
        self.save()
        return record

    def record_call_with_price(
        self,
        api_name: str,
        tokens_in: int,
        tokens_out: int,
        model: str,
        price_per_1k_input: float,
        price_per_1k_output: float,
        task_id: Optional[str] = None,
    ) -> CostRecord:
        """Record a call with explicit per-model pricing."""
        cost_input = (tokens_in / 1000.0) * price_per_1k_input
        cost_output = (tokens_out / 1000.0) * price_per_1k_output
        cost_usd = round(cost_input + cost_output, 8)

        record = CostRecord.now(
            api_name=api_name,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            task_id=task_id,
        )
        self._records.append(record)
        self.save()
        return record

    # ── Budget queries ───────────────────────────────────────────────────

    def get_monthly_spent(self) -> float:
        """Get total cost for the current calendar month."""
        now = datetime.now(timezone.utc)
        month_start = now.strftime("%Y-%m-01")
        total = 0.0
        for r in self._records:
            if r.timestamp >= month_start:
                total += r.cost_usd
        return round(total, 6)

    def get_remaining_budget(self) -> float:
        """Get remaining budget for the current month."""
        return round(max(0.0, self.budget.monthly_budget_usd - self.get_monthly_spent()), 6)

    def is_over_budget(self) -> bool:
        """Check if monthly budget has been exceeded."""
        return self.get_monthly_spent() >= self.budget.monthly_budget_usd

    def is_near_limit(self, warn_threshold_pct: Optional[float] = None) -> bool:
        """Check if spending is near the warning threshold."""
        threshold = warn_threshold_pct if warn_threshold_pct is not None else self.budget.warn_threshold_pct
        if self.budget.monthly_budget_usd <= 0:
            return False
        pct_used = (self.get_monthly_spent() / self.budget.monthly_budget_usd) * 100
        return pct_used >= threshold

    # ── Statistics ───────────────────────────────────────────────────────

    def get_stats(self) -> CostStats:
        """Get comprehensive cost statistics."""
        stats = CostStats()
        stats.monthly_budget = self.budget.monthly_budget_usd
        stats.total_calls = len(self._records)

        for r in self._records:
            stats.total_cost_usd += r.cost_usd
            stats.total_tokens_in += r.tokens_in
            stats.total_tokens_out += r.tokens_out

            # By API
            stats.by_api[r.api_name] = stats.by_api.get(r.api_name, 0.0) + r.cost_usd

            # By date
            date_key = r.timestamp[:10]  # YYYY-MM-DD
            stats.by_date[date_key] = stats.by_date.get(date_key, 0.0) + r.cost_usd

            # By task
            if r.task_id:
                stats.by_task[r.task_id] = stats.by_task.get(r.task_id, 0.0) + r.cost_usd

        stats.total_cost_usd = round(stats.total_cost_usd, 6)
        stats.by_api = {k: round(v, 6) for k, v in stats.by_api.items()}
        stats.by_date = {k: round(v, 6) for k, v in stats.by_date.items()}
        stats.by_task = {k: round(v, 6) for k, v in stats.by_task.items()}

        stats.remaining_budget = self.get_remaining_budget()
        if stats.monthly_budget > 0:
            stats.budget_used_pct = round((stats.total_cost_usd / stats.monthly_budget) * 100, 2)

        return stats

    # ── Export ───────────────────────────────────────────────────────────

    def export_report(self, format: str = "json") -> str:
        """Export cost report in the specified format."""
        stats = self.get_stats()
        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "stats": stats.to_dict(),
            "records": [r.to_dict() for r in self._records],
        }
        if format == "json":
            return json.dumps(data, indent=2, ensure_ascii=False)
        raise ValueError(f"Unsupported export format: {format}")

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist all cost records to disk (workspace/costs.json)."""
        if self._storage_path is None:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._records]
        with open(self._storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """Load cost records from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return

        with open(self._storage_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._records = [CostRecord.from_dict(d) for d in data]

    @property
    def records(self) -> list[CostRecord]:
        """Return a copy of all cost records."""
        return list(self._records)

    def clear(self) -> None:
        """Clear all records and the persisted file."""
        self._records.clear()
        if self._storage_path is not None and self._storage_path.exists():
            self._storage_path.unlink()
