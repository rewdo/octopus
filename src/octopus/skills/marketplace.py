"""
Skill Marketplace — publish, discover, install, and rate skills.

Provides a local marketplace layer on top of SkillRegistry:
    - Publish skills from the registry to a shared marketplace directory
    - Search by keyword, category, or tag
    - Install/uninstall skills between marketplace and registry
    - Rate skills and view popular rankings
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .skill_engine import Skill, SkillRegistry


# ---------------------------------------------------------------------------
# SkillListing — marketplace entry with public metadata
# ---------------------------------------------------------------------------

@dataclass
class SkillListing:
    """Public metadata for a skill listed in the marketplace."""

    name: str
    version: str
    author: str
    description: str
    category: str
    tags: list[str]
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0

    @classmethod
    def from_skill(cls, skill: Skill, downloads: int = 0,
                   rating: float = 0.0, rating_count: int = 0) -> "SkillListing":
        return cls(
            name=skill.name,
            version=skill.version,
            author=skill.author,
            description=skill.description,
            category=skill.category,
            tags=list(skill.tags),
            downloads=downloads,
            rating=rating,
            rating_count=rating_count,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "downloads": self.downloads,
            "rating": self.rating,
            "rating_count": self.rating_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillListing":
        return cls(**{k: data.get(k, v.default if v.default is not v else v.default_factory())
                       for k, v in cls.__dataclass_fields__.items() if k in data
                       or data.get(k) is not None})


# ---------------------------------------------------------------------------
# SkillMarketplace — local marketplace manager
# ---------------------------------------------------------------------------

class SkillMarketplace:
    """Local skill marketplace: publish, discover, install, and rate skills.

    Skills are stored as paired JSON files under ``storage_dir``:
        - ``{name}.skill.json`` — the full Skill definition
        - ``{name}.listing.json`` — the SkillListing metadata (ratings, downloads)

    Usage::

        registry = SkillRegistry()
        registry.load_from_dir("skills/")

        market = SkillMarketplace(registry, storage_dir="marketplace/")
        market.publish("text_summarize", author="alice")
        results = market.search("summarize")
        market.install("text_summarize")
    """

    def __init__(self, registry: SkillRegistry,
                 storage_dir: str | Path | None = None):
        self._registry = registry
        self._storage_dir = Path(storage_dir or "marketplace")
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._installed: set[str] = set()  # Track installed skill names

    # ---- Publish ----

    def publish(self, skill_name: str, author: str | None = None) -> SkillListing:
        """Publish a skill from the local registry to the marketplace.

        Args:
            skill_name: Name of the skill in the registry.
            author: Override the skill's author field.

        Returns:
            The SkillListing created in the marketplace.

        Raises:
            ValueError: If the skill is not in the registry.
        """
        skill = self._registry.get(skill_name)
        if skill is None:
            raise ValueError(f"Skill '{skill_name}' not found in registry")

        if author is not None:
            skill.author = author

        # Check for existing listing to preserve downloads/ratings
        listing = self._load_listing(skill_name)
        if listing is None:
            listing = SkillListing.from_skill(skill)
        else:
            # Update metadata but keep social stats
            listing.version = skill.version
            listing.description = skill.description
            listing.category = skill.category
            listing.tags = list(skill.tags)
            listing.author = skill.author

        # Write files
        skill.save(self._skill_path(skill_name))
        self._save_listing(skill_name, listing)

        return listing

    # ---- Install / Uninstall ----

    def install(self, skill_name: str) -> bool:
        """Install a skill from the marketplace into the local registry.

        Returns True on success, False if the skill is not in the marketplace.
        Does nothing and returns True if already installed.
        """
        if self._registry.has(skill_name):
            self._installed.add(skill_name)
            return True

        path = self._skill_path(skill_name)
        if not path.exists():
            return False

        skill = Skill.from_file(path)
        self._registry.register(skill)
        self._installed.add(skill_name)

        # Increment download counter
        listing = self._load_listing(skill_name)
        if listing:
            listing.downloads += 1
            self._save_listing(skill_name, listing)

        return True

    def uninstall(self, skill_name: str) -> bool:
        """Uninstall a skill from the local registry (keeps marketplace listing)."""
        self._installed.discard(skill_name)
        return self._registry.unregister(skill_name)

    def list_installed(self) -> list[str]:
        """Return names of all installed marketplace skills."""
        return [s.name for s in self._registry.list_all()
                if s.name in self._installed]

    # ---- Search ----

    def search(self, query: str = "", category: str | None = None,
               tag: str | None = None) -> list[SkillListing]:
        """Search marketplace skills by keyword, category, and/or tag.

        Matching is case-insensitive against name, description, tags, and category.
        Results are sorted by relevance score.
        """
        scored: list[tuple[float, SkillListing]] = []
        query_lower = query.lower() if query else ""

        for listing in self._list_all_listings():
            if category and listing.category != category:
                continue
            if tag and tag not in listing.tags:
                continue

            if not query_lower:
                scored.append((0.0, listing))
                continue

            score = 0.0
            name_parts = listing.name.lower().replace("_", " ").split()
            for part in name_parts:
                if part in query_lower:
                    score += 3.0

            desc_words = set(listing.description.lower().split())
            query_words = set(query_lower.split())
            score += len(desc_words & query_words) * 1.5

            for t in listing.tags:
                if t.lower() in query_lower:
                    score += 2.0

            cat_parts = listing.category.lower().replace("_", " ").split()
            for part in cat_parts:
                if part in query_lower:
                    score += 1.0

            if score > 0 or not query_lower:
                scored.append((score, listing))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s[1] for s in scored]

    # ---- Social ----

    def rate(self, skill_name: str, rating: float) -> bool:
        """Rate a skill (1.0–5.0). Updates the running average.

        Returns False if the skill is not in the marketplace.
        """
        listing = self._load_listing(skill_name)
        if listing is None:
            return False

        if not 1.0 <= rating <= 5.0:
            raise ValueError(f"Rating must be 1.0–5.0, got {rating}")

        total = listing.rating * listing.rating_count + rating
        listing.rating_count += 1
        listing.rating = round(total / listing.rating_count, 2)
        self._save_listing(skill_name, listing)
        return True

    def get_popular(self, limit: int = 10) -> list[SkillListing]:
        """Return the most popular skills ranked by downloads (descending)."""
        listings = self._list_all_listings()
        listings.sort(key=lambda l: l.downloads, reverse=True)
        return listings[:limit]

    # ---- Internal helpers ----

    def _skill_path(self, name: str) -> Path:
        return self._storage_dir / f"{name}.skill.json"

    def _listing_path(self, name: str) -> Path:
        return self._storage_dir / f"{name}.listing.json"

    def _save_listing(self, name: str, listing: SkillListing) -> None:
        with open(self._listing_path(name), "w", encoding="utf-8") as f:
            json.dump(listing.to_dict(), f, indent=2, ensure_ascii=False)

    def _load_listing(self, name: str) -> Optional[SkillListing]:
        path = self._listing_path(name)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return SkillListing.from_dict(data)

    def _list_all_listings(self) -> list[SkillListing]:
        listings: list[SkillListing] = []
        for fname in os.listdir(self._storage_dir):
            if fname.endswith(".listing.json"):
                name = fname[:-len(".listing.json")]
                listing = self._load_listing(name)
                if listing:
                    listings.append(listing)
        return listings

    def __repr__(self) -> str:
        return (f"SkillMarketplace(storage='{self._storage_dir}', "
                f"listings={sum(1 for _ in self._storage_dir.glob('*.listing.json'))})")
