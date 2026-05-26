"""
Cognitive Budget System — cost-aware brain upgrade control.

Evaluates every brain-routing decision through three gates:
1. Monthly budget → hard cap (force downgrade on overrun)
2. Per-task budget → soft cap (recommend downgrade)
3. Expected gain → cost-benefit analysis (only upgrade if worth it)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

from ..brains.base import BrainType, TaskComplexity

if TYPE_CHECKING:
    from ..api.cost_tracker import CostTracker
    from ..config import OctopusConfig


# ── Brain success rates (empirical estimates) ─────────────────────────────
_BRAIN_SUCCESS_RATES: dict[BrainType, float] = {
    BrainType.CHEAP: 0.70,      # Rule matching — fast but brittle
    BrainType.SKILL: 0.85,      # Pre-compiled skill — reliable within domain
    BrainType.ACTION: 0.80,     # Tool execution — depends on tool quality
    BrainType.MEMORY: 0.75,     # Retrieval — recall precision varies
    BrainType.PLANNING: 0.70,   # Decomposition — complex tasks may miss steps
    BrainType.FRONTIER: 0.90,   # LLM — highest but most expensive
    BrainType.WORLD: 0.95,      # State query — deterministic when data exists
}

# ── Estimated token consumption per brain type ────────────────────────────
_TOKEN_ESTIMATES: dict[BrainType, int] = {
    BrainType.CHEAP: 0,         # Rule engine, no tokens
    BrainType.SKILL: 100,       # Lightweight skill execution
    BrainType.ACTION: 150,      # Tool call + result parsing
    BrainType.MEMORY: 200,      # Embedding + retrieval
    BrainType.PLANNING: 500,    # Task decomposition + reasoning
    BrainType.FRONTIER: 2000,   # Full LLM reasoning
    BrainType.WORLD: 0,         # In-memory state, negligible tokens
}

# ── Estimated latency per brain type (ms) ─────────────────────────────────
_LATENCY_ESTIMATES: dict[BrainType, int] = {
    BrainType.CHEAP: 5,
    BrainType.SKILL: 20,
    BrainType.ACTION: 100,
    BrainType.MEMORY: 50,
    BrainType.PLANNING: 500,
    BrainType.FRONTIER: 2000,
    BrainType.WORLD: 1,
}

# ── Downgrade chain: higher → lower cost brains ───────────────────────────
_DOWNGRADE_PATH: dict[BrainType, BrainType] = {
    BrainType.WORLD: BrainType.PLANNING,
    BrainType.FRONTIER: BrainType.PLANNING,
    BrainType.PLANNING: BrainType.SKILL,
    BrainType.SKILL: BrainType.CHEAP,
    BrainType.ACTION: BrainType.SKILL,
    BrainType.MEMORY: BrainType.CHEAP,
    BrainType.CHEAP: BrainType.CHEAP,  # Terminal — can't go lower
}

# Default USD per token (≈ $1.50 / 1M tokens)
_DEFAULT_TOKEN_PRICE = 0.0000015


@dataclass
class BudgetDecision:
    """Result of a cognitive budget check."""

    allowed: bool
    reason: str
    suggested_brain: str       # Downgrade target if not allowed
    expected_gain: float       # Cost-benefit delta (positive = upgrade justified)
    token_limit: int           # Max tokens allowed for this task


class CognitiveBudget:
    """Cognitive budget system: evaluates the cost-effectiveness of every brain upgrade.

    Three-phase check:
    1. **Hard cap** — monthly budget exhausted → force CHEAP
    2. **Soft cap** — per-task budget insufficient → recommend downgrade
    3. **Gain analysis** — (success boost × task value) − token cost − latency cost
       Only allow upgrade when expected_gain > 0

    Usage::

        budget = CognitiveBudget(cost_tracker, config)
        decision = budget.can_afford(BrainType.FRONTIER, TaskComplexity.MODERATE)
        if decision.allowed:
            router.use_brain(BrainType.FRONTIER)
        else:
            router.use_brain(BrainType[decision.suggested_brain.upper()])
    """

    def __init__(self, cost_tracker: CostTracker, config: OctopusConfig):
        self._cost_tracker = cost_tracker
        self._config = config

    # ── Main API ───────────────────────────────────────────────────────

    def can_afford(
        self,
        target_brain_type: Union[BrainType, str],
        task_complexity: Union[TaskComplexity, float, int],
    ) -> BudgetDecision:
        """Check whether the target brain is affordable for a task.

        Args:
            target_brain_type: The brain we want to route to.
            task_complexity: TaskComplexity enum, float, or int (1-5).

        Returns:
            BudgetDecision with allowed flag, reason, and downgrade suggestion.
        """
        bt = self._normalize_brain(target_brain_type)
        complexity_value = self._normalize_complexity(task_complexity)

        # ── Gate 1: Monthly budget hard cap ────────────────────────────
        if self._cost_tracker.is_over_budget():
            return BudgetDecision(
                allowed=False,
                reason="月度预算已超支，强制使用最低成本脑",
                suggested_brain=self.get_recommended_downgrade(bt),
                expected_gain=0.0,
                token_limit=self.estimate_token_cost(BrainType.CHEAP),
            )

        # ── Gate 2: Per-task budget check ──────────────────────────────
        estimated_cost = self._estimate_cost_usd(bt)
        max_task_cost = self._config.budget.max_per_task_usd

        if estimated_cost > max_task_cost:
            downgrade = self.get_recommended_downgrade(bt)
            overage = estimated_cost - max_task_cost
            return BudgetDecision(
                allowed=False,
                reason=f"单任务估算成本 ${estimated_cost:.4f} 超过上限 ${max_task_cost:.4f}，超出 ${overage:.4f}",
                suggested_brain=downgrade,
                expected_gain=-overage,
                token_limit=self.estimate_token_cost(downgrade),
            )

        # ── Gate 3: Expected gain analysis ─────────────────────────────
        expected_gain = self._compute_expected_gain(bt, complexity_value)

        if expected_gain <= 0:
            downgrade = self.get_recommended_downgrade(bt)
            return BudgetDecision(
                allowed=False,
                reason=f"预期增益 {expected_gain:.4f} ≤ 0，升级不划算",
                suggested_brain=downgrade,
                expected_gain=expected_gain,
                token_limit=self.estimate_token_cost(downgrade),
            )

        return BudgetDecision(
            allowed=True,
            reason="",
            suggested_brain=bt.value,
            expected_gain=expected_gain,
            token_limit=self.estimate_token_cost(bt),
        )

    # ── Downgrade recommendation ───────────────────────────────────────

    def get_recommended_downgrade(self, brain_type: Union[BrainType, str] = BrainType.FRONTIER) -> str:
        """Return the next lower-cost brain in the downgrade chain.

        Chain: WORLD → FRONTIER → PLANNING → SKILL → CHEAP (terminal)
        """
        bt = self._normalize_brain(brain_type)
        return _DOWNGRADE_PATH.get(bt, BrainType.CHEAP).value

    # ── Token estimation ───────────────────────────────────────────────

    def estimate_token_cost(
        self,
        brain_type: Union[BrainType, str],
        task_length: int = 1,
    ) -> int:
        """Estimate total token consumption for a brain type.

        Args:
            brain_type: Target brain.
            task_length: Multiplier for repeated calls (default 1).

        Returns:
            Estimated token count (0 for rule-based brains).
        """
        bt = self._normalize_brain(brain_type)
        return _TOKEN_ESTIMATES.get(bt, 100) * max(1, task_length)

    # ── Helpers ────────────────────────────────────────────────────────

    def _compute_expected_gain(self, target: BrainType, complexity: float) -> float:
        """Compute cost-benefit delta for routing to target brain.

        Formula:
            gain = (success_boost × task_value) − token_cost − latency_cost

        Where:
            - success_boost = target success rate − baseline (CHEAP) success rate
            - task_value = complexity × 0.1
            - token_cost = estimated tokens × price per token
            - latency_cost = estimated latency_ms × 0.001
        """
        target_success = _BRAIN_SUCCESS_RATES.get(target, 0.7)
        baseline_success = _BRAIN_SUCCESS_RATES.get(BrainType.CHEAP, 0.7)
        success_boost = target_success - baseline_success

        task_value = complexity * 0.1

        token_cost = self._estimate_cost_usd(target)
        latency_cost = _LATENCY_ESTIMATES.get(target, 100) * 0.001

        return (success_boost * task_value) - token_cost - latency_cost

    def _estimate_cost_usd(self, brain_type: BrainType) -> float:
        """Convert estimated tokens to USD using default token price."""
        tokens = _TOKEN_ESTIMATES.get(brain_type, 100)
        return round(tokens * _DEFAULT_TOKEN_PRICE, 6)

    @staticmethod
    def _normalize_brain(bt: Union[BrainType, str]) -> BrainType:
        """Accept both BrainType enum and string values."""
        if isinstance(bt, BrainType):
            return bt
        return BrainType(bt.lower())

    @staticmethod
    def _normalize_complexity(cx: Union[TaskComplexity, float, int]) -> float:
        """Accept TaskComplexity enum, float, or int (1-5)."""
        if isinstance(cx, TaskComplexity):
            return float(cx.value)
        return float(cx)
