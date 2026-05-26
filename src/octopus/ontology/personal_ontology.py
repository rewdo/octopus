"""
Personal Ontology — build individual user world-models from memory.

Phase 3: Extracts preferences, projects, decisions, and skill affinities
from the MemoryGraph.  Each interaction updates the user profile so the
system can offer relevance-weighted recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

from octopus.memory.memory_graph import NodeType, EdgeType

if TYPE_CHECKING:
    from octopus.memory.memory_graph import MemoryGraph

logger = logging.getLogger(__name__)


# ── User Profile ────────────────────────────────────────────────────────────

@dataclass
class UserProfile:
    """A user's distilled world-model extracted from memory.

    Attributes:
        user_id: Unique identifier for this user.
        preferences: Key-value preferences ('language', 'model', etc.).
        project_structure: {project_name: {path, type, last_used}}.
        code_habits: Observed coding patterns.
        decision_history: Chronological record of significant decisions.
        collaboration_style: 'autonomous' / 'guided' / 'collaborative'.
        skill_affinity: {skill_name: usage_count}.
        last_updated: When this profile was last refreshed.
    """

    user_id: str = ""
    preferences: dict[str, Any] = field(default_factory=dict)
    project_structure: dict[str, dict[str, Any]] = field(default_factory=dict)
    code_habits: list[str] = field(default_factory=list)
    decision_history: list[dict[str, Any]] = field(default_factory=list)
    collaboration_style: str = "collaborative"
    skill_affinity: dict[str, int] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "user_id": self.user_id,
            "preferences": self.preferences,
            "project_structure": self.project_structure,
            "code_habits": self.code_habits,
            "decision_history": self.decision_history,
            "collaboration_style": self.collaboration_style,
            "skill_affinity": self.skill_affinity,
            "last_updated": self.last_updated.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        """Deserialize from a plain dict."""
        lu = data.get("last_updated")
        return cls(
            user_id=data.get("user_id", ""),
            preferences=data.get("preferences", {}),
            project_structure=data.get("project_structure", {}),
            code_habits=data.get("code_habits", []),
            decision_history=data.get("decision_history", []),
            collaboration_style=data.get("collaboration_style", "collaborative"),
            skill_affinity=data.get("skill_affinity", {}),
            last_updated=datetime.fromisoformat(lu) if lu else datetime.now(timezone.utc),
        )


# ── Personal Ontology ───────────────────────────────────────────────────────

class PersonalOntology:
    """Build and maintain individual user world-models from the memory graph.

    Usage::

        onto = PersonalOntology(memory_graph)
        profile = onto.build_profile("user_42")
        onto.update_from_interaction("user_42", "fix bug", decision, outcome)
        recs = onto.get_recommendations("user_42", "deploy service")
    """

    def __init__(self, memory_graph: Optional["MemoryGraph"] = None) -> None:
        """Initialise with an optional shared memory graph.

        Args:
            memory_graph: An existing MemoryGraph instance; if None,
                          a fresh internal graph is created.
        """
        self._graph: Optional["MemoryGraph"] = memory_graph
        self._profiles: dict[str, UserProfile] = {}

    # ── Profile Construction ────────────────────────────────────────────

    def build_profile(
        self,
        user_id: str,
        memory_graph: Optional["MemoryGraph"] = None,
    ) -> UserProfile:
        """Extract a UserProfile from the memory graph.

        Walks the graph for Preference, Project, Event, and Skill nodes
        connected to the given user, then aggregates them into a structured
        profile.

        Args:
            user_id: The user identifier to build a profile for.
            memory_graph: Override the internal graph (e.g. a fresh load).

        Returns:
            A populated UserProfile.
        """
        graph = memory_graph or self._graph
        profile = UserProfile(user_id=user_id)

        if graph is None:
            logger.debug("No memory graph available; returning empty profile for %s", user_id)
            self._profiles[user_id] = profile
            return profile

        # ─ 1. Extract preferences ─
        pref_nodes = graph.query_nodes(
            node_type=NodeType.PREFERENCE,
            keywords=[user_id],
            limit=50,
        )
        for node in pref_nodes:
            key = node.metadata.get("preference_key", node.content[:40])
            profile.preferences[key] = node.metadata.get("preference_value", node.content)

        # ─ 2. Extract project structure ─
        proj_nodes = graph.query_nodes(
            node_type=NodeType.PROJECT,
            keywords=[user_id],
            limit=50,
        )
        for node in proj_nodes:
            pname = node.metadata.get("project_name", node.content[:40])
            profile.project_structure[pname] = {
                "path": node.metadata.get("path", ""),
                "type": node.metadata.get("project_type", "unknown"),
                "last_used": node.timestamp.isoformat(),
            }

        # ─ 3. Extract decision history (decisions & significant events) ─
        event_nodes = graph.query_nodes(
            node_type=NodeType.EVENT,
            keywords=[user_id, "decision", "decide"],
            limit=100,
        )
        for node in event_nodes:
            profile.decision_history.append({
                "date": node.timestamp.isoformat(),
                "decision": node.content,
                "context": node.metadata.get("context", ""),
            })
        # Sort chronologically (oldest first for timeline)
        profile.decision_history.sort(key=lambda d: d["date"])

        # ─ 4. Extract skill affinity ─
        skill_nodes = graph.query_nodes(
            node_type=NodeType.SKILL,
            keywords=[user_id],
            limit=100,
        )
        for node in skill_nodes:
            skill_name = node.metadata.get("skill_name", node.content[:30])
            count = node.metadata.get("usage_count", 0)
            if isinstance(count, int) and count > 0:
                profile.skill_affinity[skill_name] = count
            else:
                profile.skill_affinity[skill_name] = profile.skill_affinity.get(skill_name, 0) + 1

        # ─ 5. Extract code habits ─
        fact_nodes = graph.query_nodes(
            node_type=NodeType.FACT,
            keywords=[user_id, "habit", "prefer", "style"],
            limit=50,
        )
        for node in fact_nodes:
            habit = node.metadata.get("habit", node.content[:60])
            if habit and habit not in profile.code_habits:
                profile.code_habits.append(habit)

        profile.last_updated = datetime.now(timezone.utc)
        self._profiles[user_id] = profile
        logger.info(
            "Built profile for %s: %d prefs, %d projects, %d decisions, %d skills",
            user_id,
            len(profile.preferences),
            len(profile.project_structure),
            len(profile.decision_history),
            len(profile.skill_affinity),
        )
        return profile

    # ── Interaction Update ──────────────────────────────────────────────

    def update_from_interaction(
        self,
        user_id: str,
        task: str,
        decision: dict[str, Any],
        outcome: dict[str, Any],
    ) -> UserProfile:
        """Update a user's ontology after an interaction completes.

        Appends the decision to history, adjusts skill usage counts,
        and refreshes the last-updated timestamp.  If no profile exists
        yet, one is created.

        Args:
            user_id: The user involved in the interaction.
            task: Description of what was tackled.
            decision: Dict with keys like 'action', 'confidence', 'rationale'.
            outcome: Dict with keys like 'success', 'feedback', 'time_taken'.

        Returns:
            The updated (or newly created) UserProfile.
        """
        profile = self._profiles.get(user_id)
        if profile is None:
            profile = UserProfile(user_id=user_id)
            self._profiles[user_id] = profile

        # Record the decision
        profile.decision_history.append({
            "date": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "decision": decision,
            "outcome": outcome,
        })

        # Update skill affinity if a skill name was mentioned
        skill = decision.get("skill")
        if skill:
            profile.skill_affinity[skill] = profile.skill_affinity.get(skill, 0) + 1

        profile.last_updated = datetime.now(timezone.utc)
        logger.debug("Updated ontology for %s after task: %s", user_id, task)
        return profile

    # ── Recommendations ─────────────────────────────────────────────────

    def get_recommendations(
        self,
        user_id: str,
        current_task: str,
    ) -> list[str]:
        """Generate context-aware recommendations based on the user's ontology.

        Looks at preferences, skill affinities, past decisions, and code
        habits to suggest relevant approaches.

        Args:
            user_id: The user seeking recommendations.
            current_task: What they're trying to do now.

        Returns:
            A list of recommendation strings (may be empty).
        """
        profile = self._profiles.get(user_id)
        if profile is None:
            logger.debug("No profile for %s; try build_profile() first", user_id)
            return []

        recs: list[str] = []

        # Preference-based tips
        if profile.preferences:
            lang = profile.preferences.get("language") or profile.preferences.get("preferred_language")
            if lang:
                recs.append(f"User prefers {lang} — default to it unless overridden.")
            model = profile.preferences.get("model") or profile.preferences.get("preferred_model")
            if model:
                recs.append(f"Use model {model} for coding tasks.")

        # Skill affinity: suggest most-used skills
        if profile.skill_affinity:
            top_skills = sorted(
                profile.skill_affinity.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            names = [s for s, _ in top_skills]
            recs.append(f"Frequent skills: {', '.join(names)}.")

        # Code habits
        if profile.code_habits:
            recs.append(f"Code style: {'; '.join(profile.code_habits[:5])}.")

        # Past similar decisions
        task_lower = current_task.lower()
        relevant_decisions = [
            d for d in profile.decision_history[-20:]
            if any(w in str(d.get("task", "")).lower() for w in task_lower.split())
        ]
        if relevant_decisions:
            last = relevant_decisions[-1]
            recs.append(
                f"Similar past task '{last.get('task', 'unknown')}' "
                f"with outcome: {last.get('outcome', {}).get('success', 'unknown')}"
            )

        # Collaboration style hint
        recs.append(f"Collaboration style: {profile.collaboration_style}.")

        return recs

    # ── Export / Import ─────────────────────────────────────────────────

    def export_profile(self, user_id: str) -> dict[str, Any]:
        """Export a user profile as a serializable dict.

        Returns an empty dict if no profile exists for the given user_id.
        """
        profile = self._profiles.get(user_id)
        if profile is None:
            return {}
        return profile.to_dict()

    def import_profile(self, data: dict[str, Any]) -> UserProfile:
        """Import a user profile from a dict (e.g. from disk or API).

        Overwrites any existing profile with the same user_id.
        """
        profile = UserProfile.from_dict(data)
        self._profiles[profile.user_id] = profile
        return profile

    # ── Accessors ──────────────────────────────────────────────────────

    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Return the cached profile, or None if not yet built."""
        return self._profiles.get(user_id)

    @property
    def profiles(self) -> dict[str, UserProfile]:
        """All currently loaded profiles (read-only view)."""
        return dict(self._profiles)
