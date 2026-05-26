"""
Skill Distiller — Extract reusable skills from task execution traces.

Supports two modes:
  - Rule-based (Phase 1): Pattern mining from action sequences
  - LLM-enhanced (Phase 2): Semantic skill extraction via LLM prompt

Provides:
    - TaskTrace: A single task execution record
    - SkillCandidate: A distilled skill proposal with confidence scoring
    - SkillDistiller: The distillation engine
"""

from __future__ import annotations

import asyncio
import json
import re
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
    """A single task execution record."""

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
    """A skill proposal distilled from task traces."""

    name: str
    description: str
    category: str
    steps: list[dict[str, Any]]  # merged step templates from source traces
    source_task_id: str
    confidence: float  # 0.0 – 1.0
    estimated_cost_saving: float = 0.0

    _source_trace_ids: list[str] = field(default_factory=list, repr=False)


# ---------------------------------------------------------------------------
# N-gram helpers
# ---------------------------------------------------------------------------

def _ngrams(seq: tuple[str, ...], n: int):
    for i in range(len(seq) - n + 1):
        yield seq[i : i + n]


def _all_subseqs(seq: tuple[str, ...], min_len: int = 2):
    max_len = len(seq)
    for n in range(min_len, max_len + 1):
        yield from _ngrams(seq, n)


# ---------------------------------------------------------------------------
# LLM prompt template
# ---------------------------------------------------------------------------

_DISTILL_PROMPT = """You are a skill extraction analyst. Given task execution traces,
identify reusable skill patterns—sequences of actions that repeat across tasks.

For each skill pattern you find, output a JSON object with these fields:
  - "name": short snake_case name (e.g. "file_backup_workflow")
  - "description": one sentence describing what the skill does
  - "category": one of [devops, file_operations, web_tools, text_processing, code_generation, general]
  - "steps": array of {{"action": "...", "params": {{...}}, "description": "..."}}
  - "confidence": float 0.0-1.0, how confident you are this is a real reusable pattern
  - "estimated_cost_saving": float, USD saved if this skill is reused instead of re-executed

Rules:
- Only extract patterns that appear in ≥2 traces.
- Prefer specific, actionable patterns over vague ones.
- Confidence should reflect how consistent and generalizable the pattern is.

Output ONLY a JSON array of patterns, nothing else. No markdown, no explanation.

Example output:
[{{"name": "fetch_and_summarize", "description": "Fetch a webpage and summarize its content", "category": "web_tools", "steps": [{{"action": "web_fetch", "params": {{}}, "description": "Fetch target URL"}}, {{"action": "call_llm", "params": {{"prompt": "summarize"}}, "description": "Summarize fetched content"}}], "confidence": 0.85, "estimated_cost_saving": 0.005}}]

Traces:
{traces_json}
"""


# ---------------------------------------------------------------------------
# SkillDistiller
# ---------------------------------------------------------------------------

class SkillDistiller:
    """Mine repeated action patterns from task traces and promote them to Skills.

    Two-phase analysis:
      - Rule mode (default): pure pattern matching, no API cost
      - LLM mode: semantic extraction via language model, enabled when
        set_api_manager() is called and ≥6 traces are recorded

    Usage::

        distiller = SkillDistiller(registry)
        distiller.set_api_manager(api_manager)  # optional, enables LLM mode
        distiller.record_trace(TaskTrace(...))
        candidates = distiller.analyze()
        for c in candidates:
            if c.confidence >= 0.5:
                distiller.promote_to_skill(c)
        print(distiller.get_stats())
    """

    MIN_TRACES = 2
    MIN_SEQ_LEN = 2
    MAX_SEQ_LEN = 10
    LLM_TRACE_THRESHOLD = 6  # need ≥ this many traces to justify LLM cost

    def __init__(self, skill_registry: "Optional[SkillRegistry]" = None):
        self._traces: list[TaskTrace] = []
        self._registry = skill_registry
        self._promoted: set[str] = set()
        self._api_manager: Any = None

    # ---- API manager ----

    def set_api_manager(self, api_manager: Any) -> None:
        """Attach an APIManager instance to enable LLM-enhanced distillation.

        Without this, analyze() always falls back to rule-based mode.
        """
        self._api_manager = api_manager

    # ---- Recording ----

    def record_trace(self, trace: TaskTrace) -> None:
        """Record a task execution trace for later analysis."""
        self._traces.append(trace)

    # ---- Analysis (dispatcher) ----

    def analyze(self) -> list[SkillCandidate]:
        """Analyse all recorded traces and distill repeating patterns.

        Dispatches to LLM mode if api_manager is set and ≥LLM_TRACE_THRESHOLD
        successful traces exist; otherwise falls back to rule-based analysis.

        Returns:
            List of SkillCandidate sorted by confidence descending.
        """
        successful = [t for t in self._traces if t.success]
        if len(successful) < self.MIN_TRACES:
            return []

        if self._api_manager is not None and len(successful) >= self.LLM_TRACE_THRESHOLD:
            try:
                return self._analyze_llm(successful)
            except Exception:
                pass  # fallback to rule mode on any LLM error
        return self._analyze_rule(successful)

    # ---- Rule-based analysis ----

    def _analyze_rule(self, successful: list[TaskTrace]) -> list[SkillCandidate]:
        """Rule-based pattern mining: find repeated action sequences.

        Algorithm:
            1. Extract action name sequences from each trace's steps.
            2. Find contiguous sub-sequences appearing in ≥2 traces.
            3. Deduplicate: keep longer sequences over subsumed shorter ones.
            4. Score confidence: traces_with_pattern / total_successful_traces.
            5. Estimate cost saving from trace costs.
        """
        total = len(successful)

        # Collect all sub-sequences & count their trace occurrences
        pattern_traces: dict[tuple[str, ...], set[str]] = defaultdict(set)

        for trace in successful:
            seq = trace.action_sequence
            for sub in _all_subseqs(seq, self.MIN_SEQ_LEN):
                if len(sub) > self.MAX_SEQ_LEN:
                    continue
                pattern_traces[sub].add(trace.task_id)

        frequent = {
            pat: tids
            for pat, tids in pattern_traces.items()
            if len(tids) >= self.MIN_TRACES
        }
        if not frequent:
            return []

        # Deduplicate: prefer longer sequences, greedily mark subsumed ones
        sorted_patterns = sorted(frequent.keys(), key=len, reverse=True)
        covered: set[tuple[str, ...]] = set()
        keep: dict[tuple[str, ...], set[str]] = {}

        for pat in sorted_patterns:
            if pat in covered:
                continue
            keep[pat] = frequent[pat]
            plen = len(pat)
            for other in list(frequent.keys()):
                if other in covered or other == pat:
                    continue
                if len(other) < plen:
                    for i in range(plen - len(other) + 1):
                        if pat[i : i + len(other)] == other:
                            covered.add(other)
                            break

        # Build candidates
        candidates: list[SkillCandidate] = []
        for pat, trace_ids in keep.items():
            freq = len(trace_ids)
            confidence = freq / total
            merged_steps = self._merge_steps(pat, trace_ids, successful)
            pattern_traces_list = [t for t in successful if t.task_id in trace_ids]
            cost_saving = sum(t.cost_usd for t in pattern_traces_list)

            candidate = SkillCandidate(
                name=self._derive_name(pat),
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

    # ---- LLM-based analysis ----

    def _analyze_llm(self, successful: list[TaskTrace]) -> list[SkillCandidate]:
        """LLM-enhanced skill extraction from task traces.

        Builds a prompt with all successful traces, sends to the cheapest
        available API, and parses the JSON response into SkillCandidate list.
        """
        # Serialize traces to a compact JSON representation
        traces_data = []
        for t in successful:
            traces_data.append({
                "task_id": t.task_id,
                "user_input": t.user_input[:200],  # truncate long inputs
                "brain_used": t.brain_used,
                "steps": [
                    {"action": s.get("action", "?"), "params": s.get("params", {}),
                     "result_summary": str(s.get("result", ""))[:100]}
                    for s in t.steps
                ],
            })

        prompt = _DISTILL_PROMPT.format(
            traces_json=json.dumps(traces_data, ensure_ascii=False, indent=2)
        )

        messages = [{"role": "user", "content": prompt}]

        # Call LLM via api_manager (wrapping async in sync)
        raw = _run_async(
            self._api_manager.call_with_retry(
                self._api_manager.get_cheapest().name,
                messages,
                temperature=0.2,
                max_tokens=2048,
            )
        )

        # Parse response
        content = raw["choices"][0]["message"]["content"]
        # Strip markdown code fences if present
        json_str = re.sub(r"^```(?:json)?\s*", "", content.strip())
        json_str = re.sub(r"\s*```$", "", json_str)
        patterns = json.loads(json_str)

        if not isinstance(patterns, list):
            return []

        candidates = []
        total = len(successful)
        for p in patterns:
            if not isinstance(p, dict):
                continue
            candidate = SkillCandidate(
                name=p.get("name", f"distill_llm_{uuid.uuid4().hex[:6]}"),
                description=p.get("description", ""),
                category=p.get("category", "general"),
                steps=p.get("steps", []),
                source_task_id=successful[0].task_id,
                confidence=min(float(p.get("confidence", 0.5)), 1.0),
                estimated_cost_saving=float(p.get("estimated_cost_saving", 0.0)),
                _source_trace_ids=[t.task_id for t in successful[:3]],
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        return candidates

    # ---- Promotion ----

    def promote_to_skill(self, candidate: SkillCandidate) -> bool:
        """Convert a SkillCandidate into a registered Skill."""
        if self._registry is None:
            return False

        from .skill_engine import Skill, SkillStep  # noqa: F811

        skill_steps: list[Any] = []
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
            cost_estimate=round(
                candidate.estimated_cost_saving
                / max(len(candidate._source_trace_ids), 1), 6
            ),
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
        """Return summary statistics for the distiller."""
        total = len(self._traces)
        successful = sum(1 for t in self._traces if t.success)
        failed = total - successful
        total_cost = sum(t.cost_usd for t in self._traces)
        total_tokens = sum(t.tokens_used for t in self._traces)
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
        base = "_".join(action_seq[:3])
        suffix = uuid.uuid4().hex[:6]
        return f"distill_{base}_{suffix}"

    @staticmethod
    def _infer_category(action_seq: tuple[str, ...]) -> str:
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
        by_position: dict[int, list[dict[str, Any]]] = defaultdict(list)

        for trace in traces:
            if trace.task_id not in trace_ids:
                continue
            seq = trace.action_sequence
            for i in range(len(seq) - len(pattern) + 1):
                if seq[i : i + len(pattern)] == pattern:
                    for offset, step in enumerate(trace.steps[i : i + len(pattern)]):
                        by_position[offset].append(step)
                    break

        merged: list[dict[str, Any]] = []
        for pos in sorted(by_position.keys()):
            instances = by_position[pos]
            template = dict(instances[0])
            merged.append({
                "action": template.get("action", "unknown"),
                "params": template.get("params", {}),
                "description": template.get(
                    "description", f"Step {pos + 1} of distilled skill"
                ),
            })

        return merged

    def __len__(self) -> int:
        return len(self._traces)

    def __repr__(self) -> str:
        mode = "llm" if self._api_manager is not None else "rule"
        return (
            f"SkillDistiller(traces={len(self._traces)}, "
            f"promoted={len(self._promoted)}, "
            f"mode={mode}, "
            f"has_registry={self._registry is not None})"
        )


# ---------------------------------------------------------------------------
# Utility: run an async coroutine from sync context
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Execute an async coroutine synchronously.

    Handles both running event loops and fresh ones.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already in an event loop — create a new one in a thread (fallback)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()
