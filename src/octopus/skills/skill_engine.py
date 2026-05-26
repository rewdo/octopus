"""
Skill Engine — Skill definition, registration, discovery, and matching.

Provides:
    - Skill: Serializable skill definition with metadata and steps
    - SkillStep: Individual action step within a skill
    - SkillRegistry: Load, register, find, and match skills
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# SkillStep — a single workflow step
# ---------------------------------------------------------------------------

@dataclass
class SkillStep:
    """A single step in a skill workflow.

    Attributes:
        action: The action type to execute (e.g., 'call_llm', 'extract_regex').
        params: Parameters for the action (supports {input_text} template vars).
        on_error: Error strategy — 'skip' (ignore), 'retry' (retry up to N),
                  or 'abort' (stop the workflow).
        depends_on: Optional list of step action names this step depends on.
        description: Human-readable description of this step.
    """

    action: str
    params: dict[str, Any] = field(default_factory=dict)
    on_error: str = "abort"
    depends_on: list[str] = field(default_factory=list)
    description: str = ""

    _VALID_ERROR_STRATEGIES = {"skip", "retry", "abort"}

    def __post_init__(self):
        if self.on_error not in self._VALID_ERROR_STRATEGIES:
            raise ValueError(
                f"on_error must be one of {self._VALID_ERROR_STRATEGIES}, "
                f"got '{self.on_error}'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "action": self.action,
            "params": self.params,
            "on_error": self.on_error,
            "depends_on": self.depends_on,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillStep":
        """Deserialize from a plain dict."""
        return cls(
            action=data.get("action", ""),
            params=data.get("params", {}),
            on_error=data.get("on_error", "abort"),
            depends_on=data.get("depends_on", []),
            description=data.get("description", ""),
        )


# ---------------------------------------------------------------------------
# Skill — a complete skill definition
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    """A pre-compiled, versioned skill definition.

    Each skill bundles metadata with a DAG of steps that together accomplish
    a well-defined task (e.g., text summarization, file format conversion).

    Attributes:
        name: Unique machine-readable identifier (e.g., 'text_summarize').
        description: Human-readable description of what this skill does.
        version: SemVer version string.
        category: Grouping category (e.g., 'text_processing').
        steps: Ordered list of SkillStep definitions.
        author: Who created this skill.
        tags: Searchable tags for discovery.
        cost_estimate: Estimated USD cost per invocation.
        success_rate: Historical success rate (0.0–1.0).
        dependencies: Optional list of pip/npm packages required.
        input_schema: Optional JSON Schema for validation.
        output_schema: Optional JSON Schema for validation.
        metadata: Arbitrary extra metadata.
    """

    name: str
    description: str
    version: str = "1.0.0"
    category: str = "general"
    steps: list[SkillStep] = field(default_factory=list)
    author: str = "octopus"
    tags: list[str] = field(default_factory=list)
    cost_estimate: float = 0.0
    success_rate: float = 0.95
    dependencies: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("Skill name cannot be empty")
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError(f"success_rate must be 0.0–1.0, got {self.success_rate}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-compatible)."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "category": self.category,
            "steps": [s.to_dict() for s in self.steps],
            "author": self.author,
            "tags": self.tags,
            "cost_estimate": self.cost_estimate,
            "success_rate": self.success_rate,
            "dependencies": self.dependencies,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skill":
        """Deserialize from a plain dict."""
        steps_data = data.get("steps", [])
        steps = [SkillStep.from_dict(s) for s in steps_data]
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            category=data.get("category", "general"),
            steps=steps,
            author=data.get("author", "octopus"),
            tags=data.get("tags", []),
            cost_estimate=data.get("cost_estimate", 0.0),
            success_rate=data.get("success_rate", 0.95),
            dependencies=data.get("dependencies", []),
            input_schema=data.get("input_schema", {}),
            output_schema=data.get("output_schema", {}),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Skill":
        """Load a skill definition from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save skill definition to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"Skill(name='{self.name}', version={self.version}, "
            f"category='{self.category}', steps={len(self.steps)})"
        )


# ---------------------------------------------------------------------------
# SkillRegistry — skill management and discovery
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Central registry for loading, querying, and matching skills.

    Usage::

        registry = SkillRegistry()
        registry.load_from_dir("skills/text_processing/")
        registry.load_from_dir("skills/code_generation/")

        # Find skills matching a task description
        matches = registry.find("summarize some long text")

        # Match by keyword
        matches = registry.match_keywords("code review python")

        # List all skills
        for skill in registry.list_all():
            print(skill.name, skill.version)
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._by_category: dict[str, list[str]] = {}
        self._by_tag: dict[str, list[str]] = {}

    # ---- Registration ----

    def register(self, skill: Skill) -> None:
        """Register a skill, overwriting any existing skill with the same name."""
        # Remove from old category/tag indices first
        self._unindex(skill.name)

        self._skills[skill.name] = skill

        # Category index
        cat = skill.category or "general"
        self._by_category.setdefault(cat, []).append(skill.name)

        # Tag index
        for tag in skill.tags:
            self._by_tag.setdefault(tag, []).append(skill.name)

    def unregister(self, name: str) -> bool:
        """Remove a skill by name. Returns True if it existed."""
        if name not in self._skills:
            return False
        self._unindex(name)
        del self._skills[name]
        return True

    def _unindex(self, name: str) -> None:
        """Remove a skill from all indices without removing from _skills."""
        for cat_list in self._by_category.values():
            if name in cat_list:
                cat_list.remove(name)
        for tag_list in self._by_tag.values():
            if name in tag_list:
                tag_list.remove(name)

    # ---- Lookup ----

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name, or None if not found."""
        return self._skills.get(name)

    def has(self, name: str) -> bool:
        """Check if a skill is registered."""
        return name in self._skills

    def has_any(self, names: list[str]) -> bool:
        """Check if any of the given skill names are registered."""
        return any(name in self._skills for name in names)

    def count(self) -> int:
        """Total number of registered skills."""
        return len(self._skills)

    # ---- Listing ----

    def list_all(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def list_by_category(self, category: str) -> list[Skill]:
        """Return skills in a specific category."""
        names = self._by_category.get(category, [])
        return [self._skills[n] for n in names if n in self._skills]

    def list_by_tag(self, tag: str) -> list[Skill]:
        """Return skills matching a specific tag."""
        names = self._by_tag.get(tag, [])
        return [self._skills[n] for n in names if n in self._skills]

    def list_categories(self) -> list[str]:
        """Return all known categories."""
        return sorted(self._by_category.keys())

    def list_tags(self) -> list[str]:
        """Return all known tags."""
        return sorted(self._by_tag.keys())

    # ---- Discovery / Matching ----

    def find(self, task_description: str) -> list[Skill]:
        """Find skills matching a natural language task description.

        Uses keyword matching against skill name, description, and tags.
        Results are scored and sorted by relevance.

        Args:
            task_description: Natural language description of the task.

        Returns:
            List of matching skills, best match first.
        """
        query_lower = task_description.lower()
        scored: list[tuple[float, Skill]] = []

        for skill in self._skills.values():
            score = 0.0

            # Name match (weighted most heavily)
            name_parts = skill.name.lower().replace("_", " ").split()
            for part in name_parts:
                if part in query_lower:
                    score += 3.0

            # Description match
            desc_lower = skill.description.lower()
            desc_words = set(desc_lower.split())
            query_words = set(query_lower.split())
            common = desc_words & query_words
            if common:
                score += len(common) * 1.5

            # Tag match
            for tag in skill.tags:
                if tag.lower() in query_lower:
                    score += 2.0

            # Category match
            cat_lower = skill.category.lower()
            if cat_lower in query_lower:
                score += 1.5
            cat_parts = cat_lower.replace("_", " ").split()
            for part in cat_parts:
                if part in query_lower:
                    score += 1.0

            if score > 0:
                scored.append((score, skill))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored]

    def match_keywords(self, query: str) -> list[Skill]:
        """Match skills by keywords against name, tags, and description.

        More targeted than find() — better for explicit keyword search.

        Args:
            query: Space-separated keywords.

        Returns:
            Matched skills in relevance order.
        """
        keywords = [kw.lower() for kw in query.split() if kw]
        if not keywords:
            return []

        scored: list[tuple[float, Skill]] = []

        for skill in self._skills.values():
            score = 0.0
            # Name
            name_lower = skill.name.lower().replace("_", " ")
            for kw in keywords:
                if kw in name_lower:
                    score += 4.0
            # Tags
            for kw in keywords:
                for tag in skill.tags:
                    if kw in tag.lower():
                        score += 3.0
            # Description
            desc_lower = skill.description.lower()
            for kw in keywords:
                if kw in desc_lower:
                    score += 1.0
            # Category
            cat_lower = skill.category.lower().replace("_", " ")
            for kw in keywords:
                if kw in cat_lower:
                    score += 1.5

            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored]

    def search(
        self,
        query: str = "",
        category: str = "",
        tag: str = "",
        min_success_rate: float = 0.0,
        limit: int = 20,
    ) -> list[Skill]:
        """Advanced search with multiple filters.

        Args:
            query: Natural language or keyword query.
            category: Filter by category.
            tag: Filter by tag.
            min_success_rate: Minimum success rate.
            limit: Maximum results.

        Returns:
            Filtered and ranked skills.
        """
        # Start with all skills or category-filtered
        if category:
            candidates = self.list_by_category(category)
        else:
            candidates = list(self._skills.values())

        # Tag filter
        if tag:
            tagged_names = set(self._by_tag.get(tag, []))
            candidates = [s for s in candidates if s.name in tagged_names]

        # Success rate filter
        if min_success_rate > 0:
            candidates = [s for s in candidates if s.success_rate >= min_success_rate]

        # Query matching
        if query:
            if re.search(r"\s", query):
                # Multi-word → use find() scoring
                matched = set(s.name for s in self.find(query))
                candidates = [s for s in candidates if s.name in matched]
            else:
                # Single word → keyword match
                matched = set(s.name for s in self.match_keywords(query))
                candidates = [s for s in candidates if s.name in matched]

        return candidates[:limit]

    # ---- Bulk loading ----

    def load_from_dir(self, path: str | Path, recursive: bool = True) -> int:
        """Load all JSON skill files from a directory.

        Args:
            path: Directory path containing .json skill files.
            recursive: Whether to scan subdirectories.

        Returns:
            Number of skills loaded.

        Raises:
            FileNotFoundError: If the path doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Skill directory not found: {path}")

        loaded = 0
        pattern = "**/*.json" if recursive else "*.json"

        for file_path in path.glob(pattern):
            if file_path.is_file():
                try:
                    skill = Skill.from_file(file_path)
                    self.register(skill)
                    loaded += 1
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # Skip malformed files but don't crash
                    import logging
                    logging.warning(f"Failed to load skill from {file_path}: {e}")
                    continue

        return loaded

    def load_from_list(self, skills: list[dict[str, Any]]) -> int:
        """Load skills from a list of dicts.

        Args:
            skills: List of skill dicts (same format as JSON files).

        Returns:
            Number of skills loaded.
        """
        loaded = 0
        for data in skills:
            try:
                self.register(Skill.from_dict(data))
                loaded += 1
            except (KeyError, ValueError) as e:
                import logging
                logging.warning(f"Failed to load skill '{data.get('name', '?')}': {e}")
                continue
        return loaded

    # ---- Export ----

    def export_all(self, path: str | Path) -> int:
        """Export all registered skills as JSON files to a directory.

        Args:
            path: Target directory.

        Returns:
            Number of files exported.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        count = 0
        for skill in self._skills.values():
            file_path = path / f"{skill.name}.json"
            skill.save(file_path)
            count += 1
        return count

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return self.has(name)

    def __repr__(self) -> str:
        return (
            f"SkillRegistry(skills={len(self._skills)}, "
            f"categories={len(self._by_category)}, "
            f"tags={len(self._by_tag)})"
        )
