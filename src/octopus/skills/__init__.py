"""
Octopus skills package — pre-compiled skill bundles and engine.

Provides:
    - Skill, SkillStep: Core skill definition types
    - SkillRegistry: Load, register, find, match, and search skills
    - 30+ pre-built skills across 6 categories
"""

from .skill_engine import Skill, SkillRegistry, SkillStep

__all__ = [
    "Skill",
    "SkillStep",
    "SkillRegistry",
]
