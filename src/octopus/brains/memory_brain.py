"""
Memory Brain — long-term memory retrieval and reasoning.

MemoryBrain handles queries about past events, stored facts, user preferences,
and historical context. It searches across all memory layers (graph, episodic,
semantic) and can optionally use an LLM to synthesize answers from retrieved
memories.

Query patterns it handles:
    - "what was the bug I fixed last week?"
    - "recall my preferences"
    - "之前做过什么"
    - "remember when we deployed the API?"
    - "what projects am I working on?"
    - "how did I solve this error before?"
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from .base import (
    BaseBrain,
    BrainRequest,
    BrainResponse,
    BrainType,
)


# ── Memory-related trigger keywords ──────────────────────────────────────

_MEMORY_KEYWORDS_EN = {
    "remember", "recall", "history", "past", "previous",
    "before", "recently", "earlier", "prior", "forget",
    "memory", "memories", "last time", "used to",
    "what did i", "what was", "when did", "how did",
    "preference", "preferences", "favorite", "always",
    "never", "mistake", "mistakes", "lesson", "lessons",
    "learned", "experience", "experiences",
}

_MEMORY_KEYWORDS_ZH = {
    "回忆", "之前", "以前", "历史", "记得", "忘记",
    "上次", "最近", "做过", "经历过", "教训",
    "偏好", "习惯", "喜欢", "总是", "从不",
    "记录", "备忘", "过去",
}


def _normalize_query(text: str) -> str:
    """Normalize query text for keyword matching."""
    return text.lower().strip()


# ── Memory Brain ────────────────────────────────────────────────────────────


class MemoryBrain(BaseBrain):
    """Answers queries about past events, facts, and preferences.

    Retrieves from:
        - MemoryGraph (keyword + semantic search)
        - EpisodicMemory (time-based event queries)
        - SemanticMemory (facts, preferences)

    Optionally uses an external LLM (via request.metadata["api_manager"])
    to synthesize a natural-language answer from retrieved memories.
    """

    brain_type = BrainType.MEMORY

    # Default time windows for contextual queries
    DEFAULT_RECENT_DAYS = 7
    DEFAULT_MEDIUM_DAYS = 30

    def __init__(
        self,
        memory_graph: Any = None,  # MemoryGraph
        episodic: Any = None,      # EpisodicMemory
        semantic: Any = None,      # SemanticMemory
        config: Any = None,
    ):
        """Initialize the memory brain.

        Args:
            memory_graph: Shared MemoryGraph for graph-based search.
            episodic: EpisodicMemory for timeline-based event queries.
            semantic: SemanticMemory for fact and preference queries.
            config: Optional configuration object.
        """
        super().__init__(config)
        self._graph = memory_graph
        self._episodic = episodic
        self._semantic = semantic

    # ── Brain protocol ──────────────────────────────────────────────────

    def can_handle(self, request: BrainRequest) -> bool:
        """Check if this request contains memory-related keywords.

        Args:
            request: The brain request to evaluate.

        Returns:
            True if the query appears to be about memory/recall/history.
        """
        query = _normalize_query(request.user_input)

        # Check English keywords
        for kw in _MEMORY_KEYWORDS_EN:
            if kw in query:
                return True

        # Check Chinese keywords
        for kw in _MEMORY_KEYWORDS_ZH:
            if kw in query:
                return True

        # Also check if it's an explicit memory query by syntax
        # Example: question starting with "what was" or "when did"
        if query.startswith("what was") or query.startswith("when did"):
            return True

        return False

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process a memory-related query.

        Pipeline:
            1. Extract time-range clues from the query
            2. Search MemoryGraph by keywords
            3. Query episodic memory for time-range events
            4. Query semantic memory for matching facts/preferences
            5. Optionally synthesize via LLM, or format a structured response

        Args:
            request: BrainRequest with user_input and optional context.

        Returns:
            BrainResponse with retrieved memories and/or synthesized answer.
        """
        query = request.user_input
        time_range = self._extract_time_range(query)

        # ── 1. MemoryGraph keyword search ──
        graph_results: list[dict[str, Any]] = []
        if self._graph is not None:
            hits = self._graph.search(query, top_k=10)
            graph_results = [
                {
                    "node_id": n.node_id,
                    "type": n.node_type,
                    "content": n.content,
                    "timestamp": n.timestamp.isoformat() if n.timestamp else None,
                    "importance": n.importance,
                    "source": n.source,
                    "metadata": n.metadata,
                }
                for n in hits
            ]

        # ── 2. Episodic memory query ──
        episodic_results: list[dict[str, Any]] = []
        if self._episodic is not None:
            events = self._episodic.query_events(
                time_range=time_range,
                limit=20,
            )
            episodic_results = [
                {
                    "node_id": e.node_id,
                    "type": "Event",
                    "content": e.content,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "importance": e.importance,
                    "source": e.source,
                    "event_type": (e.metadata or {}).get("event_type", ""),
                }
                for e in events
            ]

        # ── 3. Semantic memory query ──
        semantic_results: list[dict[str, Any]] = []
        if self._semantic is not None:
            # Query facts by text
            facts = self._semantic.query_fact_text(query, limit=10)
            semantic_results.extend([
                {
                    "node_id": f.node_id,
                    "type": "Fact",
                    "content": f.content,
                    "timestamp": f.timestamp.isoformat() if f.timestamp else None,
                    "importance": f.importance,
                    "source": f.source,
                    "metadata": f.metadata,
                }
                for f in facts
            ])

            # Also retrieve preferences
            prefs = self._semantic.get_preferences()
            if prefs:
                semantic_results.append({
                    "type": "Preferences",
                    "content": f"User preferences: {prefs}",
                    "metadata": {"preferences": prefs},
                })

        # ── 4. Deduplicate and merge ──
        all_results = self._merge_and_deduplicate(
            graph_results + episodic_results + semantic_results,
        )

        # ── 5. Synthesize answer ──
        if all_results:
            content = self._format_response(query, all_results)
            confidence = min(1.0, len(all_results) * 0.1 + 0.2)
        else:
            content = (
                f"I searched my memory but found no matching results for: \"{query}\".\n\n"
                f"Time range searched: {self._describe_time_range(time_range)}.\n"
                f"Try using different keywords or check if the memory was recorded."
            )
            confidence = 0.05

        # ── 6. Optionally enhance with LLM ──
        api_manager = request.metadata.get("api_manager") if request.metadata else None
        if api_manager is not None and all_results:
            try:
                llm_answer = await self._llm_synthesize(
                    api_manager, query, all_results, request.max_tokens,
                )
                if llm_answer:
                    content = llm_answer
            except Exception:
                pass  # Silent fallback to formatted response

        # ── 7. Build response ──
        self._total_calls += 1

        return BrainResponse(
            success=len(all_results) > 0,
            content=content,
            brain_type=BrainType.MEMORY,
            confidence=confidence,
            tokens_used=self._estimate_tokens(content),
            structured_output={
                "results": all_results[:20],
                "total_found": len(all_results),
                "query": query,
                "time_range": {
                    "start": time_range[0].isoformat() if time_range else None,
                    "end": time_range[1].isoformat() if time_range else None,
                } if time_range else None,
                "memory_counts": {
                    "graph": len(graph_results),
                    "episodic": len(episodic_results),
                    "semantic": len(semantic_results),
                },
            },
        )

    # ── Time-range extraction ───────────────────────────────────────────

    def _extract_time_range(
        self,
        query: str,
    ) -> Optional[tuple[datetime, datetime]]:
        """Extract a time-range hint from the query text.

        Recognizes patterns like:
            - "last week", "this week", "last month"
            - "yesterday", "today"
            - "in the past 3 days"
            - "上周", "昨天", "最近三天"

        Args:
            query: User's natural-language query.

        Returns:
            (start, end) datetime tuple, or None if no time range detected.
        """
        import re

        now = datetime.now()
        query_lower = query.lower()

        # ── English patterns ──
        # "last <N> days/weeks/months"
        match = re.search(r"last\s+(\d+)\s+(day|days|week|weeks|month|months)", query_lower)
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            if "day" in unit:
                delta = timedelta(days=num)
            elif "week" in unit:
                delta = timedelta(weeks=num)
            else:
                delta = timedelta(days=num * 30)
            return (now - delta, now)

        # "past <N> days/weeks/months"
        match = re.search(r"(?:in\s+the\s+)?past\s+(\d+)\s+(day|days|week|weeks|month|months)", query_lower)
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            if "day" in unit:
                delta = timedelta(days=num)
            elif "week" in unit:
                delta = timedelta(weeks=num)
            else:
                delta = timedelta(days=num * 30)
            return (now - delta, now)

        # "last week" (singular, no number)
        if "last week" in query_lower:
            return (now - timedelta(weeks=1), now)

        # "last month" (singular)
        if "last month" in query_lower:
            return (now - timedelta(days=30), now)

        # "yesterday"
        if "yesterday" in query_lower:
            yesterday_start = datetime(now.year, now.month, now.day) - timedelta(days=1)
            yesterday_end = yesterday_start + timedelta(days=1)
            return (yesterday_start, yesterday_end)

        # "today"
        if "today" in query_lower:
            today_start = datetime(now.year, now.month, now.day)
            return (today_start, now)

        # "this week"
        if "this week" in query_lower:
            monday = now - timedelta(days=now.weekday())
            return (datetime(monday.year, monday.month, monday.day), now)

        # ── Chinese patterns ──
        # "上周"
        if "上周" in query:
            return (now - timedelta(weeks=1), now)

        # "昨天"
        if "昨天" in query:
            yesterday_start = datetime(now.year, now.month, now.day) - timedelta(days=1)
            yesterday_end = yesterday_start + timedelta(days=1)
            return (yesterday_start, yesterday_end)

        # "今天"
        if "今天" in query:
            today_start = datetime(now.year, now.month, now.day)
            return (today_start, now)

        # "最近<N>天"
        match = re.search(r"最近\s*(\d+)\s*天", query)
        if match:
            num = int(match.group(1))
            return (now - timedelta(days=num), now)

        # "最近<N>周"
        match = re.search(r"最近\s*(\d+)\s*周", query)
        if match:
            num = int(match.group(1))
            return (now - timedelta(weeks=num), now)

        # "最近" (generic: default to 7 days)
        if "最近" in query:
            return (now - timedelta(days=self.DEFAULT_RECENT_DAYS), now)

        # ── Smart defaults ──
        # If query mentions memory recall without explicit time, default to 30 days
        recall_keywords = {"remember", "recall", "history", "past", "previous", "回忆", "之前", "以前"}
        query_words = set(query_lower.split())
        if recall_keywords & query_words:
            return (now - timedelta(days=self.DEFAULT_MEDIUM_DAYS), now)

        return None

    def _describe_time_range(
        self,
        time_range: Optional[tuple[datetime, datetime]],
    ) -> str:
        """Human-readable description of a time range."""
        if time_range is None:
            return "all time"
        start, end = time_range
        days = (end - start).days
        if days <= 0:
            return "today"
        if days == 1:
            return "yesterday"
        if days <= 7:
            return f"last {days} days"
        if days <= 31:
            return f"last {days // 7} week(s)"
        return f"last {days} days"

    # ── Result processing ───────────────────────────────────────────────

    def _merge_and_deduplicate(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge and deduplicate results from multiple memory sources.

        Deduplicates by node_id where applicable, preserving the highest-
        importance copy of each entry. Sorts by importance descending.

        Args:
            results: List of result dicts from various sources.

        Returns:
            Deduplicated, sorted result list.
        """
        seen: dict[str, dict[str, Any]] = {}

        for item in results:
            nid = item.get("node_id")
            if nid:
                # Deduplicate by node_id, keep higher importance
                if nid in seen:
                    if item.get("importance", 0) > seen[nid].get("importance", 0):
                        seen[nid] = item
                else:
                    seen[nid] = item
            else:
                # Items without node_id (e.g. Preferences) are always kept
                seen[f"_anon_{len(seen)}"] = item

        # Sort: higher importance first, then by timestamp (newer first)
        sorted_results = sorted(
            seen.values(),
            key=lambda x: (
                x.get("importance", 0),
                x.get("timestamp", ""),
            ),
            reverse=True,
        )
        return sorted_results

    def _format_response(
        self,
        query: str,
        results: list[dict[str, Any]],
    ) -> str:
        """Format retrieved memories into a human-readable response.

        Args:
            query: The original user query.
            results: Merged, deduplicated memory results.

        Returns:
            Formatted text response.
        """
        if not results:
            return f"No matching memories found for: \"{query}\"."

        # Group by type
        by_type: dict[str, list[dict[str, Any]]] = {}
        for item in results:
            t = item.get("type", "Other")
            by_type.setdefault(t, []).append(item)

        lines = [f"Found {len(results)} relevant memories for: \"{query}\"\n"]

        # Order sections by priority
        section_order = ["Preferences", "Fact", "Event", "Preference", "Other"]

        for section in section_order:
            items = by_type.pop(section, [])
            if not items:
                continue

            if section == "Preferences":
                lines.append("### Preferences")
            elif section == "Fact":
                lines.append(f"### Facts ({len(items)})")
            elif section == "Event":
                lines.append(f"### Past Events ({len(items)})")
            else:
                lines.append(f"### {section} ({len(items)})")

            for item in items[:10]:  # Cap at 10 per section
                content = item.get("content", "(no content)")
                ts = item.get("timestamp")
                importance = item.get("importance", 0)
                source = item.get("source", "")

                # Format timestamp if present
                ts_str = ""
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        ts_str = f" [{dt.strftime('%Y-%m-%d %H:%M')}]"
                    except (ValueError, TypeError):
                        pass

                # Format metadata hints
                imp_str = f" ★{importance:.1f}" if importance > 0.5 else ""

                # Truncate long content
                display = content if len(content) <= 200 else content[:197] + "..."

                lines.append(f"  - {display}{ts_str}{imp_str}")

            lines.append("")

        # Remaining sections (fallback)
        for section, items in by_type.items():
            if not items:
                continue
            lines.append(f"### {section} ({len(items)})")
            for item in items[:5]:
                content = item.get("content", "(no content)")
                display = content if len(content) <= 200 else content[:197] + "..."
                lines.append(f"  - {display}")
            lines.append("")

        return "\n".join(lines).strip()

    async def _llm_synthesize(
        self,
        api_manager: Any,
        query: str,
        results: list[dict[str, Any]],
        max_tokens: int,
    ) -> Optional[str]:
        """Use an LLM to synthesize a natural-language answer from memories.

        Args:
            api_manager: API manager with LLM access (e.g. call_api method).
            query: User's original question.
            results: Retrieved memory entries.
            max_tokens: Token budget for the LLM response.

        Returns:
            Synthesized answer string, or None if LLM call fails.
        """
        # Build a prompt with retrieved memories as context
        memory_context_parts = []
        for item in results[:15]:  # Limit context to avoid token blowup
            content = item.get("content", "")
            ts = item.get("timestamp", "unknown")
            mem_type = item.get("type", "memory")
            memory_context_parts.append(
                f"[{mem_type}] Time: {ts}\n{content}\n"
            )
        memory_context = "\n".join(memory_context_parts)

        prompt = (
            "You are a memory retrieval assistant. The user asked a question about "
            "their past activities, preferences, or stored facts.\n\n"
            f"USER QUESTION: {query}\n\n"
            f"RETRIEVED MEMORIES:\n{memory_context}\n\n"
            "Based ONLY on the retrieved memories above, answer the user's question. "
            "If the memories don't fully answer it, say so honestly. "
            "Be concise and direct. Do not make up information not in the memories."
        )

        try:
            response = await api_manager.call_api(
                api_name="openai",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=min(max_tokens, 1024),
                temperature=0.3,
            )
            if response and response.get("choices"):
                return response["choices"][0]["message"]["content"]
        except Exception:
            pass

        return None

    # ── Utility ─────────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Rough token count estimate.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count (>= 1).
        """
        if not text:
            return 0
        # ~4 chars per token for English, ~1.5 for CJK
        cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        ascii_chars = len(text) - cjk
        return max(1, int(cjk / 1.5 + ascii_chars / 4))
