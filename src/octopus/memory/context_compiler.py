"""
Context Compiler — assembles relevant context for each brain request.

The Context Compiler follows a 4-step pipeline:
  1. Task Analysis — extract goals, constraints, entities from user input
  2. Multi-dimensional Retrieval — pull relevant nodes from MemoryGraph
  3. Structured Organization — must keep + summary + forbid
  4. Prompt Injection — compile into a token-budgeted context block

Key principle: Memory ≠ Context. Memory is stored; context is compiled
on-the-fly for each request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from octopus.brains.base import BrainRequest
from octopus.memory.memory_graph import MemoryGraph, MemoryNode, NodeType


# ── Task Analysis ──────────────────────────────────────────────────────────

@dataclass
class TaskAnalysis:
    """Result of analyzing a user request to determine what context is needed.

    Attributes:
        goal: The primary objective inferred from the request.
        constraints: Explicit or implicit constraints (e.g. deadline, budget).
        entities: Named entities extracted (projects, people, tools).
        keywords: Key terms for memory retrieval.
        required_context_types: Which node types are most relevant.
        estimated_complexity: Rough complexity estimate (1-5).
    """

    goal: str = ""
    constraints: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    required_context_types: list[str] = field(default_factory=list)
    estimated_complexity: int = 1


# ── Context Compiler ───────────────────────────────────────────────────────

class ContextCompiler:
    """Assembles relevant memory context for brain processing.

    The compiler takes a BrainRequest + MemoryGraph and produces a
    structured context string that fits within a token budget.

    Output format:
        [Current Task]
        Goal: ...
        Constraints: ...

        [Key Context (must retain)]
        ...

        [Related History (summary)]
        ...

        [Available Skills]
        ...
    """

    # Node type → section mapping for organization
    TYPE_PRIORITY: dict[str, int] = {
        NodeType.PREFERENCE: 0,  # Most important — personal preferences
        NodeType.FACT: 1,
        NodeType.EVENT: 2,
        NodeType.ENTITY: 3,
        NodeType.PROJECT: 4,
        NodeType.SKILL: 5,
        NodeType.TOOL: 6,
        NodeType.USER: 7,
    }

    # Keywords that signal high importance for must-retain section
    HIGH_PRIORITY_KEYWORDS: set[str] = {
        "preference", "prefer", "always", "never", "must",
        "critical", "important", "deadline", "rule", "policy",
    }

    def compile(
        self,
        request: BrainRequest,
        memory_graph: MemoryGraph,
        token_budget: int = 2000,
    ) -> str:
        """Compile context for a brain request.

        This is the main entry point. It runs the full 4-step pipeline:
        analyze → retrieve → organize → format.

        Args:
            request: The BrainRequest needing context.
            memory_graph: Source of memories to pull from.
            token_budget: Maximum tokens for the compiled context.

        Returns:
            Formatted context string ready for injection into a prompt.
        """
        # Step 1: Analyze the task
        task = self.analyze_task(request.user_input)

        # Step 2: Retrieve relevant memories
        relevant = self.retrieve_relevant(task, memory_graph)

        # Step 3 & 4: Organize and format within budget
        compiled = self.organize(task, relevant, token_budget)

        return compiled

    def analyze_task(self, user_input: str) -> TaskAnalysis:
        """Analyze a user request to determine what context is needed.

        Uses regex-based heuristics (Phase 1). Future: LLM-based analysis.

        Args:
            user_input: Raw user input string.

        Returns:
            TaskAnalysis with extracted goal, constraints, entities, etc.
        """
        analysis = TaskAnalysis()
        text = user_input.strip()

        if not text:
            return analysis

        # ── Goal ──
        analysis.goal = text[:200]  # Truncated as goal summary

        # ── Constraints ──
        constraint_patterns = [
            (r"by\s+(tomorrow|next\s+\w+|monday|tuesday|wednesday|thursday|friday|saturday|sunday)", "deadline"),
            (r"under\s+\$?(\d+)", "budget"),
            (r"in\s+(\d+)\s*(seconds|minutes|hours|days)", "time_limit"),
            (r"using\s+(.+)", "tool_requirement"),
        ]
        for pattern, ctype in constraint_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                analysis.constraints.append(f"{ctype}: {match.group(0)}")

        # ── Entities ──
        analysis.entities = self._extract_entities(text)

        # ── Keywords ──
        # Extract meaningful words (skip stopwords)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "so", "if", "then", "than",
            "this", "that", "these", "those", "can", "will", "would",
            "could", "should", "do", "does", "did", "has", "have", "had",
            "what", "when", "where", "who", "why", "how", "which",
            "please", "just", "now", "also", "very", "really",
        }
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        analysis.keywords = [w for w in words if w not in stopwords][:20]

        # ── Required context types ──
        type_keywords: dict[str, list[str]] = {
            NodeType.PREFERENCE: ["prefer", "like", "favorite", "always", "never", "style"],
            NodeType.FACT: ["what", "who", "when", "where", "fact", "know"],
            NodeType.EVENT: ["happened", "did", "before", "last", "recent", "history"],
            NodeType.SKILL: ["skill", "workflow", "process", "procedure", "how to"],
            NodeType.PROJECT: ["project", "repo", "code", "build", "deploy"],
            NodeType.TOOL: ["tool", "command", "run", "execute", "script"],
        }
        text_lower = text.lower()
        for ntype, trigger_words in type_keywords.items():
            if any(tw in text_lower for tw in trigger_words):
                analysis.required_context_types.append(ntype)

        # Default: get preferences and facts
        if not analysis.required_context_types:
            analysis.required_context_types = [NodeType.PREFERENCE, NodeType.FACT, NodeType.EVENT]

        # ── Complexity ──
        analysis.estimated_complexity = self._estimate_complexity(text)

        return analysis

    def retrieve_relevant(
        self,
        task: TaskAnalysis,
        memory_graph: MemoryGraph,
    ) -> list[MemoryNode]:
        """Retrieve relevant memories from the graph for a given task.

        Uses keyword search + entity matching + type filtering.

        Args:
            task: Analyzed task with keywords and entities.
            memory_graph: MemoryGraph to query.

        Returns:
            List of relevant MemoryNode objects, deduplicated.
        """
        results: dict[str, MemoryNode] = {}  # node_id → node (dedup)

        # ── Search by keywords ──
        if task.keywords:
            for kw in task.keywords[:5]:
                hits = memory_graph.search(kw, top_k=10)
                for hit in hits:
                    results[hit.node_id] = hit

        # ── Search by entities ──
        for entity in task.entities:
            hits = memory_graph.search(entity, top_k=5)
            for hit in hits:
                results[hit.node_id] = hit

        # ── Fetch preferences (always relevant) ──
        if NodeType.PREFERENCE in task.required_context_types or not task.required_context_types:
            prefs = memory_graph.query_nodes(node_type=NodeType.PREFERENCE, limit=20)
            for pref in prefs:
                results[pref.node_id] = pref

        # ── Fetch facts ──
        if NodeType.FACT in task.required_context_types:
            facts = memory_graph.query_nodes(node_type=NodeType.FACT, limit=20)
            for fact in facts:
                results[fact.node_id] = fact

        # ── Fetch recent events ──
        if NodeType.EVENT in task.required_context_types:
            events = memory_graph.get_timeline(node_type=NodeType.EVENT, limit=10)
            for event in events:
                results[event.node_id] = event

        # Sort by importance
        sorted_results = sorted(
            results.values(),
            key=lambda n: (self.TYPE_PRIORITY.get(n.node_type, 99), -n.importance),
        )
        return sorted_results

    def organize(
        self,
        task: TaskAnalysis,
        relevant_memories: list[MemoryNode],
        token_budget: int = 2000,
    ) -> str:
        """Organize retrieved memories into a structured context block.

        Separates into three sections:
          - Must retain (high-priority preferences, rules)
          - Summary (relevant history, facts)
          - Available (skills, tools)

        Args:
            task: Task analysis for the header.
            relevant_memories: Retrieved MemoryNode objects.
            token_budget: Maximum token count for the output.

        Returns:
            Formatted context string.
        """
        parts: list[str] = []

        # ── Header: Current Task ──
        header = self._format_header(task)
        parts.append(header)

        # Separate memories into categories
        must_retain: list[MemoryNode] = []
        summary: list[MemoryNode] = []
        available: list[MemoryNode] = []

        for mem in relevant_memories:
            # Check for high-priority signals
            content_lower = mem.content.lower()
            is_high_priority = any(
                kw in content_lower for kw in self.HIGH_PRIORITY_KEYWORDS
            ) or mem.node_type in (NodeType.PREFERENCE,)

            if is_high_priority:
                must_retain.append(mem)
            elif mem.node_type in (NodeType.SKILL, NodeType.TOOL):
                available.append(mem)
            else:
                summary.append(mem)

        # ── Budget allocation ──
        budget_remaining = token_budget - self.estimate_tokens(header) - 50  # reserve for section headers

        # Allocate: 40% must-retain, 40% summary, 20% available
        retain_budget = int(budget_remaining * 0.4)
        summary_budget = int(budget_remaining * 0.4)
        avail_budget = int(budget_remaining * 0.2)

        # ── [Key Context (must retain)] ──
        if must_retain:
            section = self._format_section(
                "Key Context (must retain)",
                must_retain,
                retain_budget,
                max_items=10,
            )
            parts.append(section)

        # ── [Related History (summary)] ──
        if summary:
            section = self._format_section(
                "Related History (summary)",
                summary,
                summary_budget,
                max_items=15,
            )
            parts.append(section)

        # ── [Available Skills] ──
        if available:
            section = self._format_section(
                "Available Skills",
                available,
                avail_budget,
                max_items=10,
            )
            parts.append(section)

        return "\n".join(parts)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple heuristics.

        Rules of thumb:
          - English text: ~4 characters per token
          - CJK characters: ~1.5 characters per token
          - Mixed: counts each separately

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count (always >= 1).
        """
        if not text:
            return 0

        # Count CJK characters (Unicode ranges)
        cjk_count = 0
        ascii_count = 0
        for char in text:
            cp = ord(char)
            # CJK Unified Ideographs, CJK Extension A, etc.
            if (
                (0x4E00 <= cp <= 0x9FFF)  # CJK Unified
                or (0x3400 <= cp <= 0x4DBF)  # CJK Extension A
                or (0x3000 <= cp <= 0x303F)  # CJK Symbols/Punctuation
                or (0xFF00 <= cp <= 0xFFEF)  # Halfwidth/Fullwidth
                or (0x2E80 <= cp <= 0x2EFF)  # CJK Radicals
            ):
                cjk_count += 1
            elif cp > 127:
                # Other non-ASCII (emoji, other scripts)
                ascii_count += 2  # count as ~2 chars worth
            else:
                ascii_count += 1

        estimated = int(cjk_count / 1.5 + ascii_count / 4)
        return max(1, estimated)

    # ── Private helpers ─────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities using simple heuristics.

        Looks for:
          - Capitalized words that aren't sentence-start
          - Path-like patterns
          - Email-like patterns
          - @mentions

        Args:
            text: Input text.

        Returns:
            List of extracted entity strings.
        """
        entities: list[str] = []

        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b", text)
        entities.extend(capitalized)

        # File paths
        paths = re.findall(r"(?:[A-Za-z]:\\[^\s,;]+|[~/][^\s,;]+/[^\s,;]+)", text)
        entities.extend(paths)

        # @mentions
        mentions = re.findall(r"@(\w+)", text)
        entities.extend(mentions)

        # Deduplicate
        seen = set()
        result = []
        for e in entities:
            if e.lower() not in seen:
                seen.add(e.lower())
                result.append(e)

        return result[:10]

    def _estimate_complexity(self, text: str) -> int:
        """Heuristic complexity estimation.

        Args:
            text: User input text.

        Returns:
            Complexity score 1-5.
        """
        complexity = 1

        # Length-based
        if len(text) > 200:
            complexity += 1

        # Multi-sentence
        sentences = re.split(r"[.!?]+", text)
        if len([s for s in sentences if s.strip()]) > 2:
            complexity += 1

        # Presence of complex indicators
        complex_signals = [
            "design", "architecture", "implement", "deploy",
            "migrate", "refactor", "optimize", "analyze",
        ]
        text_lower = text.lower()
        signal_count = sum(1 for sig in complex_signals if sig in text_lower)
        if signal_count >= 3:
            complexity += 2
        elif signal_count >= 1:
            complexity += 1

        return min(5, max(1, complexity))

    def _format_header(self, task: TaskAnalysis) -> str:
        """Format the task analysis as a header section.

        Args:
            task: Analyzed task.

        Returns:
            Formatted header string.
        """
        lines = ["[Current Task]"]
        lines.append(f"Goal: {task.goal}")
        if task.constraints:
            lines.append(f"Constraints: {'; '.join(task.constraints)}")
        if task.entities:
            lines.append(f"Entities: {', '.join(task.entities)}")
        lines.append("")
        return "\n".join(lines)

    def _format_section(
        self,
        title: str,
        memories: list[MemoryNode],
        token_budget: int,
        max_items: int = 10,
    ) -> str:
        """Format a section of memories within a token budget.

        Args:
            title: Section title.
            memories: MemoryNode objects to include.
            token_budget: Maximum tokens for this section.
            max_items: Maximum number of items.

        Returns:
            Formatted section string.
        """
        lines = [f"[{title}]"]

        used_tokens = 0
        count = 0
        for mem in memories:
            if count >= max_items:
                break

            # Format: "  - {content} [{source}, {importance:.2f}]"
            line = f"  - {mem.content[:200]}"
            meta_parts = []
            if mem.source and mem.source != "unknown":
                meta_parts.append(mem.source)
            if mem.importance > 0:
                meta_parts.append(f"score={mem.importance:.2f}")
            if meta_parts:
                line += f" [{', '.join(meta_parts)}]"

            line_tokens = self.estimate_tokens(line)
            if used_tokens + line_tokens > token_budget - 10:
                break

            lines.append(line)
            used_tokens += line_tokens
            count += 1

        if count == 0:
            lines.append("  (none)")

        lines.append("")
        return "\n".join(lines)
