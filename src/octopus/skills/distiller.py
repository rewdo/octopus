"""
Skill Distiller — Extract reusable skills from task execution traces.

Phase 1: Rule-based pattern mining. Finds repeated action sequences across
successful task traces and promotes them to registered Skills.

Provides:
    - TaskTrace: A single task execution record
    - SkillCandidate: A distilled skill proposal with confidence scoring
    - SkillDistiller: The distillation engine
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .skill_engine import SkillRegistry, Skill  # noqa: F401


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskTrace:
    """A single task execution record.

    Captures every step taken, the brain that handled it, cost, and outcome.
    """

    task_id: str
    user_input: str
    brain_used: str
    success: bool
    steps: list[dict[str, Any]]  # [{action, params, result, duration}, ...]
    cost_usd: float = 0.0
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def action_sequence(self) -> tuple[str, ...]:
        """Extract the ordered list of action names from steps."""
        return tuple(s.get("action", "unknown") for s in self.steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class SkillCandidate:
    """A skill proposal distilled from task traces.

    Confidence is purely frequency-based; no LLM involvement in Phase 1.
    """

    name: str
    description: str
    category: str
    steps: list[dict[str, Any]]  # merged step templates from source traces
    source_task_id: str
    confidence: float  # 0.0 – 1.0, based on pattern frequency
    estimated_cost_saving: float = 0.0

    # Internal tracking (not serialised to Skill)
    _source_trace_ids: list[str] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------

def _ngrams(seq: tuple[str, ...], n: int):
    """Yield all contiguous n-grams from a sequence."""
    for i in range(len(seq) - n + 1):
        yield seq[i : i + n]


def _all_subseqs(seq: tuple[str, ...], min_len: int = 2):
    """Yield every contiguous subsequence of length >= min_len."""
    max_len = len(seq)
    for n in range(min_len, max_len + 1):
        yield from _ngrams(seq, n)


# ---------------------------------------------------------------------------
# SkillDistiller
# ---------------------------------------------------------------------------

class SkillDistiller:
    """Mine repeated action patterns from task traces and promote them to Skills.

    Usage::

        distiller = SkillDistiller(registry)

        # Record traces as tasks complete
        distiller.record_trace(TaskTrace(...))

        # Analyse and get candidates
        candidates: list[SkillCandidate] = distiller.analyze()

        # Promote the good ones
        for c in candidates:
            if c.confidence >= 0.5:
                distiller.promote_to_skill(c)

        print(distiller.get_stats())
    """

    # Minimum traces that must share a pattern before it's considered a skill
    MIN_TRACES = 2
    # Minimum sequence length (steps) for a candidate
    MIN_SEQ_LEN = 2
    # Maximum sequence length to explore (avoids combinatorial explosion)
    MAX_SEQ_LEN = 10

    def __init__(self, skill_registry: "Optional[SkillRegistry]" = None):
        self._traces: list[TaskTrace] = []
        self._registry = skill_registry
        self._promoted: set[str] = set()  # candidate names already promoted

    # ---- Recording ----

    def record_trace(self, trace: TaskTrace) -> None:
        """Record a task execution trace for later analysis.

        Args:
            trace: A completed TaskTrace. Stored in-memory; call frequently.
        """
        self._traces.append(trace)

    # ---- Analysis ----

    def analyze(self) -> list[SkillCandidate]:
        """Analyse all recorded traces and distill repeating patterns.

        Algorithm (Phase 1, rule-based):
            1. Keep only successful traces (success=True).
            2. Extract the action name sequence from each trace's steps.
            3. Find every contiguous sub-sequence that appears in ≥2 traces.
            4. Deduplicate: when a longer sequence subsumes a shorter one,
               keep the longer sequence (more specific = more useful).
            5. Score confidence: traces_with_pattern / total_successful_traces.
            6. Estimate cost saving from trace costs that use this pattern.

        Returns:
            List of SkillCandidate sorted by confidence descending.
        """
        successful = [t for t in self._traces if t.success]
        if len(successful) < self.MIN_TRACES:
            return []

        total = len(successful)

        # ── Step 2–3: collect all sub-sequences & count their trace occurrences ──
        # pattern → set of trace_ids that contain it
        pattern_traces: dict[tuple[str, ...], set[str]] = defaultdict(set)

        for trace in successful:
            seq = trace.action_sequence
            for sub in _all_subseqs(seq, self.MIN_SEQ_LEN):
                if len(sub) > self.MAX_SEQ_LEN:
                    continue
                pattern_traces[sub].add(trace.task_id)

        # Keep only patterns that appear in ≥2 traces
        frequent = {
            pat: tids
            for pat, tids in pattern_traces.items()
            if len(tids) >= self.MIN_TRACES
        }
        if not frequent:
            return []

        # ── Step 4: deduplicate — prefer longer sequences ──
        # Sort by length descending; greedily mark subsequences as covered.
        sorted_patterns = sorted(frequent.keys(), key=len, reverse=True)
        covered: set[tuple[str, ...]] = set()
        keep: dict[tuple[str, ...], set[str]] = {}

        for pat in sorted_patterns:
            if pat in covered:
                continue
            keep[pat] = frequent[pat]
            # Mark all strict sub-sequences of this pattern as covered
            plen = len(pat)
            for other in list(frequent.keys()):
                if other in covered or other == pat:
                    continue
                # If 'other' is a subsequence of 'pat', cover it
                if len(other) < plen:
                    for i in range(plen - len(other) + 1):
                        if pat[i : i + len(other)] == other:
                            covered.add(other)
                            break

        # ── Step 5–6: build candidates ──
        candidates: list[SkillCandidate] = []

        for pat, trace_ids in keep.items():
            freq = len(trace_ids)
            confidence = freq / total  # 0..1

            # Merge step templates from all source traces for this pattern
            merged_steps = self._merge_steps(pat, trace_ids, successful)

            # Estimated cost saving: sum of costs of traces that used this pattern
            pattern_traces_list = [t for t in successful if t.task_id in trace_ids]
            cost_saving = sum(t.cost_usd for t in pattern_traces_list)

            # Generate a human-readable name from the action sequence
            name = self._derive_name(pat)

            candidate = SkillCandidate(
                name=name,
                description=f"Auto-distilled skill: {' → '.join(pat)}",
                category=self._infer_category(pat),
                steps=merged_steps,
                source_task_id=list(trace_ids)[0],
                confidence=round(confidence, 4),
                estimated_cost_saving=round(cost_saving, 6),
                _source_trace_ids=list(trace_ids),
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    # ---- Promotion ----

    def promote_to_skill(self, candidate: SkillCandidate) -> bool:
        """Convert a SkillCandidate into a registered Skill.

        Creates a full ``Skill`` object with ``SkillStep`` entries and
        registers it with the ``SkillRegistry`` (if one was provided).

        Args:
            candidate: A SkillCandidate returned by ``analyze()``.

        Returns:
            True if the skill was registered successfully, False otherwise.
        """
        if self._registry is None:
            return False

        # Import at call time to avoid circular imports at module level
        from .skill_engine import Skill, SkillStep  # noqa: F811

        # Convert raw step dicts → SkillStep objects
        skill_steps: list[SkillStep] = []
        for s in candidate.steps:
            skill_steps.append(
                SkillStep(
                    action=s.get("action", "unknown"),
                    params=s.get("params", {}),
                    on_error=s.get("on_error", "skip"),
                    description=s.get("description", ""),
                )
            )

        skill = Skill(
            name=candidate.name,
            description=candidate.description,
            version="1.0.0",
            category=candidate.category,
            steps=skill_steps,
            author="distiller",
            tags=["auto-distilled", candidate.category],
            cost_estimate=round(candidate.estimated_cost_saving / max(len(candidate._source_trace_ids), 1), 6),
            success_rate=candidate.confidence,
            metadata={
                "source_trace_ids": candidate._source_trace_ids,
                "confidence": candidate.confidence,
            },
        )

        self._registry.register(skill)
        self._promoted.add(candidate.name)
        return True

    # ---- Stats ----

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics for the distiller.

        Returns:
            dict with keys: total_traces, successful, failed, candidates,
            promoted, total_cost_usd, total_tokens.
        """
        total = len(self._traces)
        successful = sum(1 for t in self._traces if t.success)
        failed = total - successful
        total_cost = sum(t.cost_usd for t in self._traces)
        total_tokens = sum(t.tokens_used for t in self._traces)

        # Count current candidates (re-run analyse cheaply)
        candidates = len(self.analyze())

        return {
            "total_traces": total,
            "successful": successful,
            "failed": failed,
            "candidates": candidates,
            "promoted": len(self._promoted),
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
        }

    # ---- Internal helpers ----

    @staticmethod
    def _derive_name(action_seq: tuple[str, ...]) -> str:
        """Generate a unique skill name from an action sequence."""
        base = "_".join(action_seq[:3])  # first 3 actions at most
        suffix = uuid.uuid4().hex[:6]
        return f"distill_{base}_{suffix}"

    @staticmethod
    def _infer_category(action_seq: tuple[str, ...]) -> str:
        """Guess a category from the actions in the sequence."""
        action_set = set(action_seq)
        if action_set & {"shell", "exec", "run", "deploy"}:
            return "devops"
        if action_set & {"file_read", "file_write", "file_search", "file_convert"}:
            return "file_operations"
        if action_set & {"web_search", "web_fetch", "web_scrape", "browser"}:
            return "web_tools"
        if action_set & {"call_llm", "generate", "summarize"}:
            return "text_processing"
        if action_set & {"code_generate", "code_review", "code_test"}:
            return "code_generation"
        return "general"

    @staticmethod
    def _merge_steps(
        pattern: tuple[str, ...],
        trace_ids: set[str],
        traces: list[TaskTrace],
    ) -> list[dict[str, Any]]:
        """Merge step templates from all traces that share this pattern.

        Picks the most common params across traces for each step position.
        """
        # Collect step instances by position in the pattern
        by_position: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for trace in traces:
            if trace.task_id not in trace_ids:
                continue
            seq = trace.action_sequence
            # Find the first occurrence of this pattern in the trace
            for i in range(len(seq) - len(pattern) + 1):
                if seq[i : i + len(pattern)] == pattern:
                    for offset, step in enumerate(trace.steps[i : i + len(pattern)]):
                        by_position[offset].append(step)
                    break  # use first match only

        merged: list[dict[str, Any]] = []
        for pos in sorted(by_position.keys()):
            instances = by_position[pos]
            # Use the first instance as template (simple merge)
            template = dict(instances[0])
            # Keep only action and description; params are too variable
            merged.append({
                "action": template.get("action", "unknown"),
                "params": template.get("params", {}),
                "description": template.get("description", f"Step {pos + 1} of distilled skill"),
            })

        return merged

    def __len__(self) -> int:
        return len(self._traces)

    def __repr__(self) -> str:
        return (
            f"SkillDistiller(traces={len(self._traces)}, "
            f"promoted={len(self._promoted)}, "
            f"has_registry={self._registry is not None})"
        )
