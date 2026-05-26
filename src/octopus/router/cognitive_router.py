"""
Cognitive Router — the central decision engine of Octopus.

Receives user requests, scores them across 9 dimensions using
heuristic rules, computes a weighted final score, and selects the
optimal brain for execution.

Routing Formula (7 of 9 dimensions in weighted sum):
    FinalScore = α·Complexity + β·Novelty + γ·Risk + δ·RealtimeNeed
                 - ε·SkillConfidence - ζ·LocalCapability - η·BudgetRemaining

Routing Intervals:
    Score < T1 (3.0):  Cheap Brain or Skill Brain
    T1 ≤ Score < T2 (6.0): Planning Brain + Skill Brain + local mid model
    T2 ≤ Score < T3 (9.0): Hybrid mode (Planning + Frontier)
    Score ≥ T3:  Frontier Brain
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..brains.base import BrainRequest, BrainType, TaskComplexity, TaskRisk
from ..config import OctopusConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword rules for heuristic dimension scoring
# ---------------------------------------------------------------------------

# Complexity boost: code-like or math-like patterns
_CODE_PATTERNS = re.compile(
    r"\b(def|class|function|import|from|return|async|await|"
    r"SELECT|INSERT|UPDATE|DELETE|CREATE TABLE|"
    r"<[a-zA-Z]+\s*[/>]|</?[a-zA-Z]+>|"
    r"const|let|var|export|require)\b",
    re.IGNORECASE,
)
_MATH_PATTERNS = re.compile(
    r"[∑∫∏√∞∂∇]|\\frac|\\sum|\\int|\\lim|\\cdot|\\alpha|\\beta|"
    r"derivative|integral|equation|algorithm|proof",
    re.IGNORECASE,
)

# Risk keywords (ordered from critical to moderate)
_HIGH_RISK_PATTERNS = re.compile(
    r"\b(delete|drop|truncate|format|wipe|purge|destroy|"
    r"financial|money|bank|payment|transfer|transaction|"
    r"legal|lawyer|attorney|court|lawsuit|"
    r"medical|diagnosis|prescription|surgery|health|disease|"
    r"password|credential|secret|token|api.?key|"
    r"sudo|root|admin|chmod|chown)\b",
    re.IGNORECASE,
)

# Realtime need indicators
_REALTIME_PATTERNS = re.compile(
    r"\b(current|latest|today|now|recent|news|weather|stock|price|"
    r"real.time|live|breaking|up.to.date)\b",
    re.IGNORECASE,
)

# Tool dependency indicators
_TOOL_PATTERNS = re.compile(
    r"\b(browse|search|fetch|download|upload|scrape|"
    r"execute|run|shell|terminal|"
    r"deploy|commit|push|merge|"
    r"email|send|notify|schedule)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# RouterDecision
# ---------------------------------------------------------------------------


@dataclass
class RouterDecision:
    """Output of the Cognitive Router after analyzing a task.

    Attributes:
        selected_brain: The brain chosen to handle this task.
        final_score: Weighted score computed from 7 dimensions.
        dimension_scores: Individual 0-10 scores for all 9 dimensions.
        reasoning: Human-readable explanation of the decision.
        estimated_cost: Estimated USD cost for the selected brain.
        timestamp: When the decision was made.
        escalated: Whether the decision was escalated due to special rules.
        escalation_reason: Why escalation occurred (if any).
    """

    selected_brain: BrainType
    final_score: float
    dimension_scores: dict[str, float]
    reasoning: str
    estimated_cost: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    escalated: bool = False
    escalation_reason: str = ""


# ---------------------------------------------------------------------------
# CognitiveRouter
# ---------------------------------------------------------------------------


class CognitiveRouter:
    """Central decision engine that routes tasks to the optimal brain.

    Uses heuristic rules (Phase 1 — no LLM dependency) to score
    9 dimensions, then computes a weighted final score to select
    the brain.

    Parameters:
        config: OctopusConfig instance with router_weights and thresholds.
        memory_manager: Optional memory manager for novelty scoring.
        skill_library: Optional skill registry for skill-confidence scoring.
        budget_tracker: Optional budget tracker for cost-aware routing.
        log_path: Path to the routing log JSONL file.
    """

    def __init__(
        self,
        config: Optional[OctopusConfig] = None,
        memory_manager: Any = None,
        skill_library: Any = None,
        budget_tracker: Any = None,
        log_path: Optional[Path] = None,
    ) -> None:
        self.config = config or OctopusConfig.default()
        self.memory_manager = memory_manager
        self.skill_library = skill_library
        self.budget_tracker = budget_tracker

        # Default log path inside workspace
        self.log_path = log_path or (
            self.config.workspace_dir / "routing_log.jsonl"
        )

        # ---- pre-compute budget when no tracker is available ----
        self._default_budget_remaining: float = 10.0  # assume full

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, task: str, **kwargs: Any) -> RouterDecision:
        """Score a raw task string and return a routing decision.

        This is the high-level entry point: it builds a BrainRequest,
        scores dimensions, computes the final score, and selects a brain.

        Args:
            task: Raw user input string.
            **kwargs: Optional overrides forwarded to BrainRequest fields
                      (complexity, risk, novelty_score, metadata, etc.)

        Returns:
            A RouterDecision with the selected brain and reasoning.
        """
        # Build a minimal BrainRequest from the task string
        request = BrainRequest(
            task_id=kwargs.pop("task_id", f"task_{datetime.now(timezone.utc).timestamp():.0f}"),
            user_input=task,
            compiled_context=kwargs.pop("compiled_context", ""),
            relevant_memories=kwargs.pop("relevant_memories", []),
            relevant_skills=kwargs.pop("relevant_skills", []),
            complexity=kwargs.pop("complexity", TaskComplexity.SIMPLE),
            risk=kwargs.pop("risk", TaskRisk.NONE),
            novelty_score=kwargs.pop("novelty_score", 0.0),
            max_tokens=kwargs.pop("max_tokens", 4096),
            budget_usd=kwargs.pop("budget_usd", 0.10),
            timeout_seconds=kwargs.pop("timeout_seconds", 30),
            allowed_tools=kwargs.pop("allowed_tools", []),
            metadata=kwargs.pop("metadata", {}),
        )

        # Keep any remaining kwargs in metadata
        if kwargs:
            request.metadata.update(kwargs)

        return self.decide(request)

    def score(self, request: BrainRequest) -> dict[str, float]:
        """Compute all 9 dimension scores for a request.

        Args:
            request: The BrainRequest to score.

        Returns:
            Dict mapping dimension name → 0-10 score.
        """
        scores: dict[str, float] = {}

        scores["complexity"] = self._score_complexity(request)
        scores["novelty"] = self._score_novelty(request)
        scores["risk"] = self._score_risk(request)
        scores["realtime_need"] = self._score_realtime(request)
        scores["skill_confidence"] = self._score_skill_confidence(request)
        scores["budget_remaining"] = self._score_budget_remaining(request)
        scores["user_preference"] = self._score_user_preference(request)
        scores["local_capability"] = self._score_local_capability(request)
        scores["tool_dependency"] = self._score_tool_dependency(request)

        return scores

    def compute_final_score(self, dimension_scores: dict[str, float]) -> float:
        """Apply the weighted routing formula to dimension scores.

        Formula (from spec):
            FinalScore = α·Complexity + β·Novelty + γ·Risk + δ·RealtimeNeed
                         - ε·SkillConfidence - ζ·LocalCapability - η·BudgetRemaining

        Args:
            dimension_scores: Dict of all 9 dimension scores.

        Returns:
            Weighted final score (float, not clamped).
        """
        w = self.config.router_weights

        raw = (
            w["alpha"] * dimension_scores.get("complexity", 0)
            + w["beta"] * dimension_scores.get("novelty", 0)
            + w["gamma"] * dimension_scores.get("risk", 0)
            + w["delta"] * dimension_scores.get("realtime_need", 0)
            - w["epsilon"] * dimension_scores.get("skill_confidence", 0)
            - w["zeta"] * dimension_scores.get("local_capability", 0)
            - w["eta"] * dimension_scores.get("budget_remaining", 0)
        )

        return round(raw, 4)

    def select_brain(self, final_score: float, dimension_scores: dict[str, float]) -> tuple[BrainType, str]:
        """Select the brain based on final score and thresholds.

        Args:
            final_score: Weighted final score.
            dimension_scores: All 9 dimension scores (for context-aware fallback).

        Returns:
            Tuple of (selected BrainType, reasoning string).
        """
        t = self.config.router_thresholds
        t1, t2, t3 = t["t1"], t["t2"], t["t3"]

        # --- budget exhaustion override ---
        budget_remaining = dimension_scores.get("budget_remaining", 10)
        if budget_remaining <= 1.0:
            return BrainType.CHEAP, (
                f"Budget critically low ({budget_remaining}/10). "
                "Forcing Cheap Brain regardless of task complexity."
            )

        # --- skill confidence override ---
        skill_conf = dimension_scores.get("skill_confidence", 0)
        if skill_conf >= 7.0 and final_score < t3:
            return BrainType.SKILL, (
                f"High skill confidence ({skill_conf}/10) with score {final_score} < T3. "
                "Delegating to Skill Brain for efficient skill execution."
            )

        # --- standard threshold routing ---
        if final_score < t1:
            # Below T1: Cheap or Skill depending on skill match
            if skill_conf >= 4.0:
                target_brain = BrainType.SKILL
                target_reason = (
                    f"Score {final_score} < T1({t1}) with moderate skill confidence ({skill_conf}). "
                    "Routing to Skill Brain."
                )
            else:
                target_brain = BrainType.CHEAP
                target_reason = (
                    f"Score {final_score} < T1({t1}) with low skill confidence ({skill_conf}). "
                    "Routing to Cheap Brain (rule-based)."
                )

        elif final_score < t2:
            target_brain = BrainType.PLANNING
            target_reason = (
                f"Score {final_score} in [{t1}, {t2}). "
                "Routing to Planning Brain with Skill Brain support + local mid-model."
            )

        elif final_score < t3:
            target_brain = BrainType.PLANNING
            target_reason = (
                f"Score {final_score} in [{t2}, {t3}). "
                "Hybrid mode: Planning Brain backed by Frontier for key segments."
            )

        else:
            target_brain = BrainType.FRONTIER
            target_reason = (
                f"Score {final_score} ≥ T3({t3}). "
                "High-complexity / high-risk task. Routing to Frontier Brain."
            )

        # --- cognitive budget check ---
        if self.budget_tracker is not None:
            decision = self._check_budget_for_brain(target_brain, final_score, dimension_scores, target_reason)
            if decision is not None:
                return decision

        return target_brain, target_reason

    def decide(self, request: BrainRequest) -> RouterDecision:
        """Full pipeline: score → final score → select brain → build decision.

        Args:
            request: The BrainRequest to route.

        Returns:
            A RouterDecision with all scoring and routing details.
        """
        # 1. Score all 9 dimensions
        dimension_scores = self.score(request)

        # 2. Compute weighted final score
        final_score = self.compute_final_score(dimension_scores)

        # 3. Select brain
        selected_brain, reasoning = self.select_brain(final_score, dimension_scores)

        # 4. Estimate cost
        estimated_cost = self._estimate_cost(selected_brain, request)

        # 5. Check escalation
        budget_remaining = dimension_scores.get("budget_remaining", 10)
        escalated = budget_remaining <= 1.0
        escalation_reason = "Budget exhausted — forced Cheap Brain" if escalated else ""

        decision = RouterDecision(
            selected_brain=selected_brain,
            final_score=final_score,
            dimension_scores=dimension_scores,
            reasoning=reasoning,
            estimated_cost=estimated_cost,
            escalated=escalated,
            escalation_reason=escalation_reason,
        )

        # 6. Log
        self.log_decision(decision, request)

        return decision

    def log_decision(self, decision: RouterDecision, request: Optional[BrainRequest] = None) -> None:
        """Write the routing decision to a JSONL log file.

        Args:
            decision: The RouterDecision to log.
            request: Optional BrainRequest for context in the log.
        """
        entry: dict[str, Any] = {
            "timestamp": decision.timestamp.isoformat(),
            "selected_brain": decision.selected_brain.value,
            "final_score": decision.final_score,
            "dimension_scores": decision.dimension_scores,
            "reasoning": decision.reasoning,
            "estimated_cost": decision.estimated_cost,
            "escalated": decision.escalated,
            "escalation_reason": decision.escalation_reason,
        }

        if request is not None:
            entry["task_id"] = request.task_id
            entry["user_input"] = request.user_input[:500]  # truncate long inputs

        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write routing log: %s", exc)

    # ------------------------------------------------------------------
    # Dimension scorers (heuristic, Phase 1 — no LLM)
    # ------------------------------------------------------------------

    def _score_complexity(self, request: BrainRequest) -> float:
        """Score task complexity (0-10) from input text heuristics.

        Factors:
            - Text length
            - Code/math keyword presence
            - Analysis/explanation keywords
            - Explicit complexity from request metadata
        """
        text = request.user_input or ""
        score = 1.0

        # Length-based (tuned to avoid short-text inflation)
        length = len(text)
        if length < 30:
            score += 0.5
        elif length < 200:
            score += 1.5
        elif length < 800:
            score += 3.0
        else:
            score += 5.0

        # Code / math signals
        code_hits = len(_CODE_PATTERNS.findall(text))
        math_hits = len(_MATH_PATTERNS.findall(text))
        score += min(code_hits * 0.8, 3.0)
        score += min(math_hits * 1.0, 2.0)

        # Greeting / casual detection — suppress complexity for simple interactions
        if re.search(r"\b(hi|hello|hey|good morning|good evening|thanks|thank you|bye|ok|okay)\b", text, re.IGNORECASE):
            score -= 1.0

        # Deep reasoning keywords (exclude "how"/"why" for short texts to avoid false positives)
        reasoning_re = (
            r"\b(explain|analyze|evaluate|summarize|compare|contrast)\b"
            if length < 30
            else r"\b(explain|analyze|evaluate|summarize|compare|contrast|why|how)\b"
        )
        if re.search(reasoning_re, text, re.IGNORECASE):
            score += 1.0

        # Explicit complexity from request
        if request.complexity == TaskComplexity.HIGHLY_COMPLEX:
            score += 3.0
        elif request.complexity == TaskComplexity.COMPLEX:
            score += 2.0
        elif request.complexity == TaskComplexity.TRIVIAL:
            score -= 1.0

        return max(1.0, min(10.0, round(score, 1)))

    def _score_novelty(self, request: BrainRequest) -> float:
        """Score novelty (0-10): 0 = seen before, 10 = completely new.

        Uses:
            - Built-in novelty_score from BrainRequest
            - Memory similarity via memory_manager
            - Fallback: assume novel if no memory available
        """
        # If BrainRequest already has a novelty_score (0-1), convert to 0-10
        if request.novelty_score > 0:
            return round(request.novelty_score * 10.0, 1)

        # Try memory-based similarity
        if self.memory_manager is not None:
            try:
                # Simple keyword-based check: how many similar past tasks exist
                similar = getattr(self.memory_manager, "search_similar", None)
                if callable(similar):
                    results = similar(request.user_input, top_k=5)
                    if results:
                        avg_similarity = sum(r.get("score", 0) for r in results) / len(results)
                        # High similarity → low novelty
                        novelty = (1.0 - avg_similarity) * 10.0
                        return max(0.0, min(10.0, round(novelty, 1)))
            except Exception:
                logger.debug("Memory similarity search failed; using default novelty", exc_info=True)

        # Fallback: no memory → assume novel
        return 8.0

    def _score_risk(self, request: BrainRequest) -> float:
        """Score risk (0-10) from content patterns and explicit risk level.

        Factors:
            - Destructive keywords (delete, drop, format, …)
            - Sensitive domains (financial, legal, medical)
            - Credential exposure
            - Explicit TaskRisk from request
        """
        text = request.user_input or ""
        score = 0.0

        # Keyword-based risk detection
        high_risk_hits = len(_HIGH_RISK_PATTERNS.findall(text))
        score += min(high_risk_hits * 2.0, 8.0)

        # Explicit risk from request
        if request.risk == TaskRisk.CRITICAL:
            score += 4.0
        elif request.risk == TaskRisk.HIGH:
            score += 3.0
        elif request.risk == TaskRisk.MEDIUM:
            score += 1.5
        elif request.risk == TaskRisk.LOW:
            score += 0.5

        return max(0.0, min(10.0, round(score, 1)))

    def _score_realtime(self, request: BrainRequest) -> float:
        """Score real-time need (0-10) from temporal keywords.

        Higher score means the task needs up-to-date external information.
        """
        text = request.user_input or ""
        hits = len(_REALTIME_PATTERNS.findall(text))
        score = min(hits * 3.0, 10.0)

        # Check metadata for explicit real-time flag
        if request.metadata.get("requires_realtime", False):
            score = max(score, 7.0)

        return round(score, 1)

    def _score_skill_confidence(self, request: BrainRequest) -> float:
        """Score skill-confidence (0-10) based on skill library matches.

        Higher score means a known skill can handle this task reliably.
        """
        confidence = 0.0

        # Check relevant_skills already attached to the request
        if request.relevant_skills:
            confidence += min(len(request.relevant_skills) * 2.0, 6.0)

        # Query skill library
        if self.skill_library is not None:
            try:
                searcher = getattr(self.skill_library, "search", None)
                if callable(searcher):
                    results = searcher(request.user_input, top_k=5)
                    if results:
                        avg_conf = sum(
                            r.get("confidence", r.get("score", 0.5)) for r in results
                        ) / len(results)
                        confidence += avg_conf * 5.0
            except Exception:
                logger.debug("Skill library search failed; using request skills only", exc_info=True)

        return max(0.0, min(10.0, round(confidence, 1)))

    def _score_budget_remaining(self, request: BrainRequest) -> float:
        """Score budget remaining (0-10): 0 = depleted, 10 = full.

        Uses budget_tracker if available; otherwise assumes full budget.
        """
        if self.budget_tracker is not None:
            try:
                remaining_pct = getattr(self.budget_tracker, "remaining_percentage", None)
                if callable(remaining_pct):
                    return round(remaining_pct() * 10.0, 1)
                remaining = getattr(self.budget_tracker, "remaining", None)
                total = getattr(self.budget_tracker, "total_budget", None)
                if remaining is not None and total is not None and total > 0:
                    return round((remaining / total) * 10.0, 1)
            except Exception:
                logger.debug("Budget tracker query failed; using default budget", exc_info=True)

        # Check metadata for budget override
        if "budget_remaining" in request.metadata:
            return float(request.metadata["budget_remaining"])

        return self._default_budget_remaining

    def _score_user_preference(self, request: BrainRequest) -> float:
        """Score user preference (0-10): whether user specified a model/brain.

        Higher score means user explicitly requested a specific brain or model.
        """
        score = 0.0

        # Check metadata for preferred model/brain
        if request.metadata.get("preferred_model"):
            score = 8.0
        elif request.metadata.get("preferred_brain"):
            score = 7.0

        # Check for brain-type mentions in user input
        text = request.user_input.lower() if request.user_input else ""
        if any(b in text for b in ("use gpt", "use claude", "use gemini", "use deepseek", "frontier")):
            score = max(score, 7.0)

        return score

    def _score_local_capability(self, request: BrainRequest) -> float:
        """Score local capability (0-10): available local compute resources.

        Higher score means more can be done locally without cloud API calls.
        """
        score = 5.0  # Default: moderate local capability

        # Check metadata for explicit capability hint
        local_gpu = request.metadata.get("local_gpu", False)
        local_model = request.metadata.get("local_model", None)
        local_ram_gb = request.metadata.get("local_ram_gb", 0)

        if local_gpu:
            score += 3.0
        if local_model:
            score += 2.0
        if local_ram_gb >= 32:
            score += 1.0
        elif local_ram_gb >= 16:
            score += 0.5

        return max(0.0, min(10.0, round(score, 1)))

    def _score_tool_dependency(self, request: BrainRequest) -> float:
        """Score tool dependency (0-10): extent to which cloud/external tools are needed.

        Higher score means more tool/API calls are required (favors cloud brains).
        """
        text = request.user_input or ""
        hits = len(_TOOL_PATTERNS.findall(text))
        score = min(hits * 2.5, 10.0)

        # If allowed_tools are pre-populated, that's a strong signal
        if request.allowed_tools:
            score = max(score, min(len(request.allowed_tools) * 2.0, 9.0))

        return round(score, 1)

    # ------------------------------------------------------------------
    # Cognitive budget gate
    # ------------------------------------------------------------------

    def _check_budget_for_brain(
        self,
        target_brain: BrainType,
        final_score: float,
        dim_scores: dict[str, float],
        base_reason: str,
    ) -> Optional[tuple[BrainType, str]]:
        """Check whether the remaining budget allows using the target brain.

        If the estimated cost of the target brain exceeds the remaining
        monthly budget, this method walks down the downgrade chain
        (FRONTIER → PLANNING → SKILL → CHEAP) until it finds a brain
        whose estimated cost fits within the remaining budget.

        Args:
            target_brain: The brain tentatively selected by threshold routing.
            final_score: Weighted final score (for reasoning context).
            dim_scores: All 9 dimension scores (for reasoning context).
            base_reason: Original routing reason from threshold-based selection.

        Returns:
            (downgraded_brain, reason) if a downgrade was required, or None
            if the target brain fits within the remaining budget.
        """
        if self.budget_tracker is None:
            return None

        # ── Estimate cost for the target brain ──
        cost_map: dict[BrainType, tuple[int, float]] = {
            BrainType.FRONTIER: (4096, 0.015),   # (tokens, price_per_1k USD)
            BrainType.PLANNING: (2000, 0.002),
            BrainType.SKILL: (500, 0.0005),
            BrainType.CHEAP: (200, 0.0),
        }

        tokens, rate_per_1k = cost_map.get(target_brain, (1500, 0.001))
        estimated_cost = round((tokens / 1000.0) * rate_per_1k, 6)

        # ── Get current budget state ──
        try:
            remaining = self.budget_tracker.get_remaining_budget()
            monthly = self.budget_tracker.budget.monthly_budget_usd
        except Exception:
            logger.debug("Budget tracker query failed; skipping budget check", exc_info=True)
            return None

        # Budget is sufficient — no downgrade needed
        if remaining >= estimated_cost:
            return None

        # ── Walk the downgrade chain ──
        downgrade_chain: list[BrainType] = [
            BrainType.FRONTIER,
            BrainType.PLANNING,
            BrainType.SKILL,
            BrainType.CHEAP,
        ]

        try:
            idx = downgrade_chain.index(target_brain)
        except ValueError:
            # Unknown brain type (e.g. MEMORY, ACTION, WORLD) —
            # not in the chain; skip budget check gracefully
            return None

        for lower_idx in range(idx + 1, len(downgrade_chain)):
            lower_brain = downgrade_chain[lower_idx]
            lower_tokens, lower_rate = cost_map.get(lower_brain, (500, 0.001))
            lower_cost = round((lower_tokens / 1000.0) * lower_rate, 6)

            if remaining >= lower_cost:
                budget_reason = (
                    f"{base_reason} | Budget-aware downgrade: "
                    f"{target_brain.value} → {lower_brain.value} "
                    f"(est. ${estimated_cost:.4f} exceeds remaining ${remaining:.4f}, "
                    f"downgraded to ${lower_cost:.4f})"
                )
                logger.info(
                    "Cognitive budget downgrade: %s → %s (score=%.2f, remaining=$%.4f)",
                    target_brain.value,
                    lower_brain.value,
                    final_score,
                    remaining,
                )
                return lower_brain, budget_reason

        # ── Even CHEAP barely fits — force it anyway ──
        _, cheap_rate = cost_map[BrainType.CHEAP]
        cheap_cost = round((200 / 1000.0) * cheap_rate, 6)
        budget_reason = (
            f"{base_reason} | Budget-aware downgrade: "
            f"{target_brain.value} → CHEAP "
            f"(budget nearly depleted: ${remaining:.4f} remaining of ${monthly:.2f} monthly)"
        )
        logger.warning(
            "Cognitive budget: forced CHEAP — only $%.4f remaining (score=%.2f)",
            remaining,
            final_score,
        )
        return BrainType.CHEAP, budget_reason

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_cost(self, brain: BrainType, request: BrainRequest) -> float:
        """Estimate USD cost for processing this request on the given brain.

        Args:
            brain: The selected brain type.
            request: The BrainRequest.

        Returns:
            Estimated cost in USD.
        """
        # Approximate per-brain token budgets
        token_budgets = {
            BrainType.CHEAP: 200,
            BrainType.SKILL: 500,
            BrainType.MEMORY: 1000,
            BrainType.PLANNING: 2000,
            BrainType.ACTION: 1500,
            BrainType.WORLD: 500,
            BrainType.FRONTIER: 4096,
        }
        tokens = token_budgets.get(brain, 1000)

        # Approximate pricing tiers (per 1K tokens, mixed input/output avg)
        price_per_1k = {
            BrainType.CHEAP: 0.0,
            BrainType.SKILL: 0.0005,
            BrainType.MEMORY: 0.001,
            BrainType.PLANNING: 0.002,
            BrainType.ACTION: 0.001,
            BrainType.WORLD: 0.0005,
            BrainType.FRONTIER: 0.015,
        }
        rate = price_per_1k.get(brain, 0.001)

        return round((tokens / 1000) * rate, 6)
