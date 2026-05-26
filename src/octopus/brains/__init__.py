"""
Octopus multi-brain architecture package.

Provides the seven specialized brains:
    - CheapBrain: Ultra-low-cost intent classification + entity extraction
    - SkillBrain: Pre-compiled skill workflow execution
    - MemoryBrain (future): Long-term memory retrieval and reasoning
    - PlanningBrain (future): Task decomposition and planning
    - ActionBrain: Tool execution engine (shell, file, web, code)
    - WorldBrain (future): World state maintenance
    - FrontierBrain (future): Cloud LLM reasoning
"""

from .base import (
    BaseBrain,
    BrainRequest,
    BrainResponse,
    BrainType,
    TaskComplexity,
    TaskRisk,
)
from .cheap_brain import CheapBrain
from .skill_brain import SkillBrain, SkillRegistry
from .action_brain import ActionBrain

__all__ = [
    # Base types
    "BaseBrain",
    "BrainRequest",
    "BrainResponse",
    "BrainType",
    "TaskComplexity",
    "TaskRisk",
    # Brains
    "CheapBrain",
    "SkillBrain",
    "ActionBrain",
    # Engine
    "SkillRegistry",
]
