"""
Octopus skills package — pre-compiled skill bundles and engine.

Provides:
    - Skill, SkillStep: Core skill definition types
    - SkillRegistry: Load, register, find, match, and search skills
    - TaskTrace, SkillCandidate, SkillDistiller: Auto-distillation from traces
    - 30+ pre-built skills across 6 categories
"""

from .distiller import SkillCandidate, SkillDistiller, TaskTrace
from .skill_engine import Skill, SkillRegistry, SkillStep

__all__ = [
    "Skill",
    "SkillStep",
    "SkillRegistry",
    "TaskTrace",
    "SkillCandidate",
    "SkillDistiller",
]
