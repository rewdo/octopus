"""
Four-layer memory management for Octopus.

L1 — Working Memory:    Transient session state (dict, minutes).
L2 — Episodic Memory:   Timeline of events (graph + time axis, long-term).
L3 — Semantic Memory:   Facts, preferences, structured knowledge (vector + property graph).
L4 — Procedural Memory: Skill definitions and workflows (permanent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from octopus.memory.memory_graph import MemoryGraph, MemoryNode, NodeType, EdgeType


# ── L1: Working Memory ─────────────────────────────────────────────────────

class WorkingMemory:
    """Transient, in-process memory for the current session (minutes).

    Think of it as the agent's "scratchpad" — holds what is immediately
    relevant to the current task.  Cleared between sessions.

    Size-limited: once max_items is exceeded, oldest items are evicted.
    """

    def __init__(self, max_items: int = 50) -> None:
        """Initialize working memory.

        Args:
            max_items: Maximum number of items before eviction begins.
        """
        self._store: dict[str, Any] = {}
        self._order: list[str] = []  # tracks insertion order for eviction
        self._max_items = max_items

    def add(self, key: str, value: Any) -> None:
        """Add or update an item in working memory.

        If max_items is reached, the oldest item is evicted.

        Args:
            key: Unique key for this memory item.
            value: Arbitrary value to store.
        """
        if key in self._store:
            # Update: remove from order and re-append at end
            self._order.remove(key)
        elif len(self._store) >= self._max_items:
            # Evict oldest
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)

        self._store[key] = value
        self._order.append(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve an item from working memory.

        Args:
            key: The key to retrieve.
            default: Default value if key not found.

        Returns:
            Stored value or default.
        """
        return self._store.get(key, default)

    def remove(self, key: str) -> bool:
        """Remove an item from working memory.

        Returns:
            True if the item existed and was removed.
        """
        if key not in self._store:
            return False
        self._store.pop(key, None)
        self._order.remove(key)
        return True

    def clear(self) -> None:
        """Remove all items from working memory."""
        self._store.clear()
        self._order.clear()

    def get_context(self, max_items: int = 50) -> list[dict[str, Any]]:
        """Get the current working context as a list of key-value pairs.

        Args:
            max_items: Maximum number of items to return.

        Returns:
            List of {"key": ..., "value": ...} dicts, most recent first.
        """
        items = []
        for key in reversed(self._order[-max_items:]):
            items.append({"key": key, "value": self._store[key]})
        return items

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current working memory state.

        Returns:
            Dict with all key-value pairs and metadata.
        """
        return {
            "items": dict(self._store),
            "size": len(self._store),
            "max_items": self._max_items,
            "snapshot_time": datetime.now().isoformat(),
        }

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._store)}/{self._max_items} items)"


# ── L2: Episodic Memory ────────────────────────────────────────────────────

class EpisodicMemory:
    """Long-term memory of events and experiences (timeline-based).

    Records what happened, when, and in what context. Backed by the
    MemoryGraph for relational queries and persistence.

    Each event is stored as an Event-type node in the memory graph.
    """

    def __init__(self, memory_graph: Optional[MemoryGraph] = None) -> None:
        """Initialize episodic memory.

        Args:
            memory_graph: Shared MemoryGraph instance. Created if not provided.
        """
        self._graph = memory_graph or MemoryGraph()

    @property
    def graph(self) -> MemoryGraph:
        """Access the underlying memory graph."""
        return self._graph

    def record_event(
        self,
        event_type: str,
        description: str,
        metadata: Optional[dict[str, Any]] = None,
        importance: float = 0.5,
        source: str = "system",
    ) -> str:
        """Record an event in episodic memory.

        Args:
            event_type: What kind of event (e.g. 'user_query', 'task_completed', 'error').
            description: Human-readable description of what happened.
            metadata: Optional structured data about the event.
            importance: Importance score 0..1.
            source: Where the event originated.

        Returns:
            Node id of the recorded event.
        """
        node_id = self._graph.add_node(
            node_type=NodeType.EVENT,
            properties={
                "content": f"[{event_type}] {description}",
                "timestamp": datetime.now(),
                "importance": importance,
                "source": source,
                "metadata": {
                    **(metadata or {}),
                    "event_type": event_type,
                    "recorded_at": datetime.now().isoformat(),
                },
            },
        )
        return node_id

    def query_events(
        self,
        time_range: Optional[tuple[datetime, datetime]] = None,
        event_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryNode]:
        """Query events by time range and/or event type.

        Args:
            time_range: (start, end) datetime tuple.
            event_type: Filter by event type stored in metadata.
            limit: Maximum number of events to return.

        Returns:
            List of MemoryNode objects matching the criteria.
        """
        events = self._graph.query_nodes(
            node_type=NodeType.EVENT,
            time_range=time_range,
            limit=limit * 2,  # Over-fetch to account for event_type filter
        )

        if event_type:
            events = [
                e for e in events
                if (e.metadata or {}).get("event_type") == event_type
            ]

        return events[:limit]

    def get_recent(self, n: int = 20) -> list[MemoryNode]:
        """Get the most recent N events.

        Args:
            n: Number of recent events.

        Returns:
            List of MemoryNode objects, newest first.
        """
        return self._graph.get_timeline(node_type=NodeType.EVENT, limit=n)

    def __len__(self) -> int:
        return len(self._graph._type_index.get(NodeType.EVENT, set()))


# ── L3: Semantic Memory ────────────────────────────────────────────────────

class SemanticMemory:
    """Long-term storage of facts, preferences, and structured knowledge.

    Backed by the MemoryGraph. Facts are stored as (subject, predicate, object)
    triples. Preferences are key-value pairs associated with users.

    Future: vector embeddings for similarity-based retrieval.
    """

    def __init__(self, memory_graph: Optional[MemoryGraph] = None) -> None:
        """Initialize semantic memory.

        Args:
            memory_graph: Shared MemoryGraph instance for persistence.
        """
        self._graph = memory_graph or MemoryGraph()

    @property
    def graph(self) -> MemoryGraph:
        """Access the underlying memory graph."""
        return self._graph

    def store_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "inference",
    ) -> str:
        """Store a factual triple (subject-predicate-object).

        Args:
            subject: Entity the fact is about.
            predicate: Relationship or property name.
            obj: Value or target entity.
            confidence: How certain we are about this fact (0..1).
            source: Origin of this fact.

        Returns:
            Node id of the fact.
        """
        content = f"{subject} {predicate} {obj}"
        node_id = self._graph.add_node(
            node_type=NodeType.FACT,
            properties={
                "content": content,
                "timestamp": datetime.now(),
                "importance": confidence,
                "source": source,
                "metadata": {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence,
                },
            },
        )
        return node_id

    def store_preference(
        self,
        key: str,
        value: Any,
        user_id: str = "default",
    ) -> str:
        """Store a user preference.

        Args:
            key: Preference key (e.g. 'preferred_model').
            value: Preference value.
            user_id: Identifier for the user.

        Returns:
            Node id of the preference node.
        """
        content = f"User '{user_id}' prefers '{key}' = '{value}'"
        node_id = self._graph.add_node(
            node_type=NodeType.PREFERENCE,
            properties={
                "content": content,
                "timestamp": datetime.now(),
                "importance": 0.5,
                "source": "user_input",
                "metadata": {
                    "user_id": user_id,
                    "key": key,
                    "value": value if isinstance(value, (str, int, float, bool)) else json.dumps(value, ensure_ascii=False),
                },
            },
        )
        return node_id

    def query_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryNode]:
        """Query stored facts by subject and/or predicate.

        Args:
            subject: Filter facts about this subject.
            predicate: Filter facts with this predicate.
            limit: Maximum number of results.

        Returns:
            List of MemoryNode objects matching the criteria.
        """
        all_facts = self._graph.query_nodes(node_type=NodeType.FACT, limit=limit * 5)

        results = []
        for fact in all_facts:
            meta = fact.metadata or {}
            if subject and meta.get("subject") != subject:
                continue
            if predicate and meta.get("predicate") != predicate:
                continue
            results.append(fact)
            if len(results) >= limit:
                break

        return results

    def query_fact_text(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search facts by text content.

        Args:
            query: Search query string.
            limit: Maximum number of results.

        Returns:
            List of MemoryNode objects matching the query.
        """
        return self._graph.search(query, top_k=limit)

    def get_preferences(self, user_id: str = "default") -> dict[str, Any]:
        """Get all stored preferences for a user.

        Args:
            user_id: User identifier.

        Returns:
            Dict of preference key → value.
        """
        all_prefs = self._graph.query_nodes(
            node_type=NodeType.PREFERENCE,
            limit=1000,
        )

        result: dict[str, Any] = {}
        for pref in all_prefs:
            meta = pref.metadata or {}
            if meta.get("user_id") == user_id:
                result[meta.get("key", "")] = meta.get("value")

        return result

    def __len__(self) -> int:
        all_facts = len(self._graph._type_index.get(NodeType.FACT, set()))
        all_prefs = len(self._graph._type_index.get(NodeType.PREFERENCE, set()))
        return all_facts + all_prefs


# ── L4: Procedural Memory ──────────────────────────────────────────────────

@dataclass
class SkillDefinition:
    """Definition of a reusable skill or workflow."""

    name: str
    description: str = ""
    category: str = "general"
    version: str = "1.0"
    triggers: list[str] = field(default_factory=list)  # Keywords that trigger this skill
    steps: list[dict[str, Any]] = field(default_factory=list)  # Ordered execution steps
    required_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "triggers": self.triggers,
            "steps": self.steps,
            "required_tools": self.required_tools,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillDefinition":
        """Deserialize from dict."""
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            category=data.get("category", "general"),
            version=data.get("version", "1.0"),
            triggers=data.get("triggers", []),
            steps=data.get("steps", []),
            required_tools=data.get("required_tools", []),
            metadata=data.get("metadata", {}),
        )


class ProceduralMemory:
    """Permanent storage of skills, workflows, and procedural knowledge.

    Skills are defined as reusable, parameterized workflows that the
    Skill Brain can execute.

    This is the most stable layer — skills are rarely deleted, only versioned.
    """

    def __init__(self, memory_graph: Optional[MemoryGraph] = None) -> None:
        """Initialize procedural memory.

        Args:
            memory_graph: Optional MemoryGraph for linking skills to context.
        """
        self._skills: dict[str, SkillDefinition] = {}
        self._graph = memory_graph or MemoryGraph()

    @property
    def graph(self) -> MemoryGraph:
        """Access the underlying memory graph."""
        return self._graph

    def register_skill(
        self,
        skill_name: str,
        skill_def: dict[str, Any],
    ) -> str:
        """Register a new skill or update an existing one.

        Args:
            skill_name: Unique skill name.
            skill_def: Skill definition dict (supports both dict and SkillDefinition fields).

        Returns:
            Node id of the skill in the memory graph.
        """
        if isinstance(skill_def, dict):
            skill = SkillDefinition(name=skill_name, **skill_def)
        else:
            skill = SkillDefinition(name=skill_name)

        self._skills[skill_name] = skill

        # Also store in graph for relational queries
        node_id = self._graph.add_node(
            node_type=NodeType.SKILL,
            properties={
                "content": f"Skill: {skill_name} — {skill.description}",
                "timestamp": datetime.now(),
                "importance": 0.7,
                "source": "system",
                "metadata": skill.to_dict(),
            },
        )
        return node_id

    def get_skill(self, name: str) -> Optional[dict[str, Any]]:
        """Retrieve a skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill definition dict, or None if not found.
        """
        skill = self._skills.get(name)
        if skill:
            return skill.to_dict()
        return None

    def list_skills(self, category: Optional[str] = None) -> list[dict[str, Any]]:
        """List all registered skills, optionally filtered by category.

        Args:
            category: Filter by skill category.

        Returns:
            List of skill definition dicts.
        """
        result = []
        for skill in self._skills.values():
            if category and skill.category != category:
                continue
            result.append(skill.to_dict())
        return result

    def remove_skill(self, name: str) -> bool:
        """Remove a skill from procedural memory.

        Args:
            name: Skill name.

        Returns:
            True if the skill existed and was removed.
        """
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        return f"ProceduralMemory({len(self._skills)} skills)"
