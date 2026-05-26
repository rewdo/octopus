"""
Memory Graph — lightweight graph-based memory store using NetworkX.

Phase 1: In-memory NetworkX with JSON persistence.
Phase 2: Optional Neo4j backend (not implemented yet).

Node types: User, Project, Tool, Skill, Event, Fact, Preference
Edge types: USES, DEPENDS_ON, OCCURRED_AT, OPTIMIZED_FROM, RELATED_TO, PREFERS

Each memory node carries: timestamp, importance_score (0-1), source, optional embedding.
"""

from __future__ import annotations

import json
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import networkx as nx


# ── Node & Edge Type Constants ──────────────────────────────────────────────

class NodeType:
    """Standard node type labels for consistency."""

    USER = "User"
    PROJECT = "Project"
    TOOL = "Tool"
    SKILL = "Skill"
    EVENT = "Event"
    FACT = "Fact"
    PREFERENCE = "Preference"
    ENTITY = "Entity"

    ALL = {USER, PROJECT, TOOL, SKILL, EVENT, FACT, PREFERENCE, ENTITY}


class EdgeType:
    """Standard edge type labels for consistency."""

    USES = "USES"
    DEPENDS_ON = "DEPENDS_ON"
    OCCURRED_AT = "OCCURRED_AT"
    OPTIMIZED_FROM = "OPTIMIZED_FROM"
    RELATED_TO = "RELATED_TO"
    PREFERS = "PREFERS"
    BELONGS_TO = "BELONGS_TO"
    LEARNED_FROM = "LEARNED_FROM"

    ALL = {
        USES, DEPENDS_ON, OCCURRED_AT, OPTIMIZED_FROM,
        RELATED_TO, PREFERS, BELONGS_TO, LEARNED_FROM,
    }


# ── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class MemoryNode:
    """A single node in the memory graph representing one unit of memory.

    Attributes:
        node_id: Unique identifier (auto-generated if not provided).
        node_type: Category label (User, Event, Fact, Preference, etc.).
        content: Human-readable content / description.
        timestamp: When this memory was recorded.
        importance: Score 0..1 indicating how important this memory is.
        source: Where this memory originated (e.g. 'user_input', 'inference', 'skill_execution').
        metadata: Arbitrary structured metadata (e.g. key-value pairs).
        embedding: Optional vector embedding for semantic search (Phase 1: optional).
    """

    node_id: str = field(default_factory=lambda: _make_id())
    node_type: str = NodeType.EVENT
    content: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for JSON persistence)."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "source": self.source,
            "metadata": self.metadata,
            # embedding is intentionally excluded from JSON; use pickle for full fidelity
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryNode":
        """Deserialize from a plain dict."""
        return cls(
            node_id=data["node_id"],
            node_type=data.get("node_type", NodeType.EVENT),
            content=data.get("content", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            importance=data.get("importance", 0.5),
            source=data.get("source", "unknown"),
            metadata=data.get("metadata", {}),
        )


def _make_id() -> str:
    """Generate a short unique node id."""
    return uuid.uuid4().hex[:12]


# ── Memory Graph ────────────────────────────────────────────────────────────

class MemoryGraph:
    """Lightweight graph-based memory store backed by NetworkX.

    Supports:
      - CRUD for typed memory nodes with metadata
      - Typed edges between nodes
      - Keyword-based search (Phase 1) with optional embedding fallback
      - Time-range and type-filtered queries
      - Automatic importance scoring
      - Garbage collection of low-value memories
      - JSON persistence (pickle for full fidelity with embeddings)
    """

    def __init__(self, graph_backend: str = "networkx") -> None:
        """Initialize the memory graph.

        Args:
            graph_backend: Backend identifier ('networkx' only in Phase 1).
        """
        self._graph: nx.DiGraph = nx.DiGraph()
        self._backend = graph_backend

        # Indexes for fast lookup
        self._node_index: dict[str, MemoryNode] = {}
        self._type_index: dict[str, set[str]] = {t: set() for t in NodeType.ALL}

    # ── Node Operations ─────────────────────────────────────────────────

    def add_node(
        self,
        node_type: str,
        node_id: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a memory node to the graph.

        Args:
            node_type: Category label (e.g. 'Event', 'Fact').
            node_id: Unique id; auto-generated if None.
            properties: Content, timestamp, importance, source, etc.

        Returns:
            The node_id of the newly created node.

        Raises:
            ValueError: If node_type is not a recognized type.
        """
        if node_type not in NodeType.ALL:
            raise ValueError(
                f"Unknown node_type '{node_type}'. Must be one of {sorted(NodeType.ALL)}"
            )

        properties = properties or {}
        node = MemoryNode(
            node_id=node_id or _make_id(),
            node_type=node_type,
            content=properties.get("content", ""),
            timestamp=properties.get("timestamp", datetime.now()),
            importance=properties.get("importance", 0.5),
            source=properties.get("source", "unknown"),
            metadata=properties.get("metadata", {}),
            embedding=properties.get("embedding"),
        )

        self._graph.add_node(node.node_id, node_type=node.node_type, data=node)
        self._node_index[node.node_id] = node
        self._type_index[node_type].add(node.node_id)
        return node.node_id

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieve a single node by id."""
        return self._node_index.get(node_id)

    def remove_node(self, node_id: str) -> bool:
        """Remove a node and its associated edges.

        Returns:
            True if the node existed and was removed.
        """
        if node_id not in self._node_index:
            return False
        node = self._node_index[node_id]
        self._graph.remove_node(node_id)
        self._node_index.pop(node_id, None)
        self._type_index.get(node.node_type, set()).discard(node_id)
        return True

    # ── Edge Operations ─────────────────────────────────────────────────

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        properties: Optional[dict[str, Any]] = None,
    ) -> str:
        """Add a typed edge between two existing nodes.

        Args:
            from_id: Source node id.
            to_id: Target node id.
            edge_type: Relationship type (e.g. 'USES', 'RELATED_TO').
            properties: Optional edge metadata.

        Returns:
            Edge key as "<from_id>--[<edge_type>]--><to_id>".

        Raises:
            ValueError: If either node does not exist.
        """
        if from_id not in self._node_index:
            raise ValueError(f"Source node '{from_id}' not found in graph")
        if to_id not in self._node_index:
            raise ValueError(f"Target node '{to_id}' not found in graph")
        if edge_type not in EdgeType.ALL:
            raise ValueError(
                f"Unknown edge_type '{edge_type}'. Must be one of {sorted(EdgeType.ALL)}"
            )

        props = properties or {}
        self._graph.add_edge(from_id, to_id, type=edge_type, **props)
        return f"{from_id}--[{edge_type}]-->{to_id}"

    # ── Query Operations ────────────────────────────────────────────────

    def query_nodes(
        self,
        node_type: Optional[str] = None,
        keywords: Optional[list[str]] = None,
        time_range: Optional[tuple[datetime, datetime]] = None,
        limit: int = 50,
    ) -> list[MemoryNode]:
        """Query nodes with optional type, keyword, and time-range filters.

        Args:
            node_type: Filter by node type (or None for all types).
            keywords: Filter nodes whose content contains ANY keyword (case-insensitive).
            time_range: (start, end) tuple; only nodes within this window.
            limit: Maximum number of results.

        Returns:
            List of matching MemoryNode objects, newest first.
        """
        # Determine candidate node ids
        if node_type and node_type in NodeType.ALL:
            candidates = list(self._type_index.get(node_type, set()))
        else:
            candidates = list(self._node_index.keys())

        results: list[MemoryNode] = []
        for nid in candidates:
            node = self._node_index.get(nid)
            if node is None:
                continue

            # Keyword filter
            if keywords:
                content_lower = node.content.lower()
                if not any(kw.lower() in content_lower for kw in keywords):
                    continue

            # Time range filter
            if time_range:
                start, end = time_range
                if not (start <= node.timestamp <= end):
                    continue

            results.append(node)

        # Sort by timestamp descending (newest first)
        results.sort(key=lambda n: n.timestamp, reverse=True)
        return results[:limit]

    def query_related(
        self,
        node_id: str,
        edge_types: Optional[list[str]] = None,
        depth: int = 2,
        limit: int = 50,
    ) -> list[MemoryNode]:
        """Find nodes related to a given node via graph traversal.

        Args:
            node_id: Starting node id.
            edge_types: Only follow these edge types (None = all).
            depth: Maximum traversal depth (BFS).
            limit: Maximum number of results.

        Returns:
            List of related MemoryNode objects, ordered by proximity.
        """
        if node_id not in self._node_index:
            return []

        visited: set[str] = {node_id}
        results: list[MemoryNode] = []
        frontier: list[str] = [node_id]
        current_depth = 0

        while frontier and current_depth < depth and len(results) < limit:
            next_frontier: list[str] = []
            for current in frontier:
                for neighbor in self._graph.neighbors(current):
                    if neighbor in visited:
                        continue
                    # Edge type filter
                    if edge_types:
                        edge_data = self._graph.get_edge_data(current, neighbor)
                        if edge_data and edge_data.get("type") not in edge_types:
                            continue
                    visited.add(neighbor)
                    node = self._node_index.get(neighbor)
                    if node:
                        results.append(node)
                        if len(results) >= limit:
                            break
                    next_frontier.append(neighbor)
                if len(results) >= limit:
                    break
            frontier = next_frontier
            current_depth += 1

        return results

    def get_timeline(
        self,
        node_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[MemoryNode]:
        """Get memories sorted by timestamp (newest first).

        Args:
            node_type: Optional type filter.
            limit: Maximum number of results.

        Returns:
            List of MemoryNode objects in descending timestamp order.
        """
        if node_type and node_type in NodeType.ALL:
            nodes = [
                self._node_index[nid]
                for nid in self._type_index.get(node_type, set())
                if nid in self._node_index
            ]
        else:
            nodes = list(self._node_index.values())

        nodes.sort(key=lambda n: n.timestamp, reverse=True)
        return nodes[:limit]

    # ── Search ──────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> list[MemoryNode]:
        """Search memories by keyword matching (Phase 1 fallback).

        Phase 2: will use embedding similarity when available.

        Args:
            query_text: Search query string.
            top_k: Max number of results.

        Returns:
            List of matching MemoryNode objects, scored by relevance.
        """
        query_words = query_text.lower().split()
        if not query_words:
            return []

        scored: list[tuple[MemoryNode, float]] = []

        for node in self._node_index.values():
            content_lower = node.content.lower()
            score = 0.0

            # Exact phrase match bonus
            if query_text.lower() in content_lower:
                score += 3.0

            # Word-level matches
            for word in query_words:
                if word in content_lower:
                    score += 1.0

            # Metadata keyword match
            meta_str = json.dumps(node.metadata).lower()
            for word in query_words:
                if word in meta_str:
                    score += 0.5

            if score > 0:
                # Boost by importance
                score *= (1.0 + node.importance)
                scored.append((node, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in scored[:top_k]]

    # ── Importance Scoring ──────────────────────────────────────────────

    def calculate_importance(self, memory: MemoryNode) -> float:
        """Auto-score a memory's importance based on heuristics.

        Scoring factors:
          - Recency: newer memories get a mild boost (decays over 30 days).
          - Content length: longer content often signals richer information.
          - Connections: more edges → more structural importance.
          - Source: user_input and inference are weighted higher.
          - Keyword signals: presence of task/goal/error/learned indicators.

        Args:
            memory: The MemoryNode to score.

        Returns:
            Importance score 0..1.
        """
        score = 0.0

        # ── Recency (decay over 30 days) ──
        age_days = (datetime.now() - memory.timestamp).total_seconds() / 86400
        recency = max(0.0, 1.0 - age_days / 30.0)
        score += recency * 0.2

        # ── Content richness ──
        content_len = len(memory.content)
        if content_len > 200:
            score += 0.2
        elif content_len > 50:
            score += 0.1

        # ── Graph connectivity ──
        if memory.node_id in self._graph:
            degree = self._graph.degree(memory.node_id)
            if degree > 5:
                score += 0.2
            elif degree > 2:
                score += 0.1

        # ── Source weight ──
        source_weights: dict[str, float] = {
            "user_input": 0.15,
            "inference": 0.12,
            "skill_execution": 0.08,
            "system": 0.05,
        }
        score += source_weights.get(memory.source, 0.05)

        # ── Keyword signals ──
        content_lower = memory.content.lower()
        signal_keywords = ["important", "critical", "goal", "deadline", "error", "learned", "preference"]
        for kw in signal_keywords:
            if kw in content_lower:
                score += 0.05
                break  # only count once

        return min(1.0, max(0.0, score))

    # ── Garbage Collection ──────────────────────────────────────────────

    def garbage_collect(self, threshold: float = 0.3) -> int:
        """Remove memories with importance below threshold.

        Skips nodes that are highly connected (>3 edges) even if importance is low.

        Args:
            threshold: Minimum importance to keep.

        Returns:
            Number of nodes removed.
        """
        to_remove: list[str] = []

        for node_id, node in self._node_index.items():
            # Recalculate importance
            importance = self.calculate_importance(node)

            if importance < threshold:
                # Protect highly connected nodes
                degree = self._graph.degree(node_id) if node_id in self._graph else 0
                if degree <= 3:
                    to_remove.append(node_id)

        removed = 0
        for node_id in to_remove:
            if self.remove_node(node_id):
                removed += 1

        return removed

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: str | Path, full: bool = False) -> None:
        """Save the memory graph to disk.

        Args:
            path: File path (JSON or pickle based on full flag).
            full: If True, use pickle (preserves embeddings).
                  If False, use JSON (human-readable, no embeddings).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if full:
            with open(path.with_suffix(".pkl"), "wb") as f:
                pickle.dump(self._graph, f)
        else:
            data = {
                "nodes": [node.to_dict() for node in self._node_index.values()],
                "edges": [
                    {
                        "from": u, "to": v,
                        "type": d.get("type", ""),
                        "properties": {k: v for k, v in d.items() if k != "type"},
                    }
                    for u, v, d in self._graph.edges(data=True)
                ],
                "backend": self._backend,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: str | Path) -> None:
        """Load the memory graph from disk.

        Automatically detects JSON vs pickle format.

        Args:
            path: File path. If JSON, loads nodes/edges. If pickle, full restore.
        """
        path = Path(path)

        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                self._graph = pickle.load(f)
            # Rebuild indexes
            self._node_index.clear()
            self._type_index = {t: set() for t in NodeType.ALL}
            for nid, ndata in self._graph.nodes(data=True):
                node = ndata.get("data")
                if isinstance(node, MemoryNode):
                    self._node_index[nid] = node
                    self._type_index.get(node.node_type, set()).add(nid)
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._graph = nx.DiGraph()
            self._node_index.clear()
            self._type_index = {t: set() for t in NodeType.ALL}
            self._backend = data.get("backend", "networkx")

            for nd in data.get("nodes", []):
                node = MemoryNode.from_dict(nd)
                self._graph.add_node(node.node_id, node_type=node.node_type, data=node)
                self._node_index[node.node_id] = node
                self._type_index.get(node.node_type, set()).add(node.node_id)

            for ed in data.get("edges", []):
                props = ed.get("properties", {})
                props["type"] = ed.get("type", "")
                self._graph.add_edge(ed["from"], ed["to"], **props)

    # ── Stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the memory graph.

        Returns:
            Dict with node count, edge count, type distribution, etc.
        """
        type_counts: dict[str, int] = {}
        for ntype, nids in self._type_index.items():
            count = len(nids)
            if count > 0:
                type_counts[ntype] = count

        importance_values = [n.importance for n in self._node_index.values()]
        avg_importance = sum(importance_values) / len(importance_values) if importance_values else 0.0

        return {
            "total_nodes": len(self._node_index),
            "total_edges": self._graph.number_of_edges(),
            "type_distribution": type_counts,
            "avg_importance": round(avg_importance, 3),
            "backend": self._backend,
        }

    def __len__(self) -> int:
        return len(self._node_index)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._node_index
