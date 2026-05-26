"""
Octopus multi-brain architecture package.

Provides the seven specialized brains:
    - CheapBrain: Ultra-low-cost intent classification + entity extraction
    - SkillBrain: Pre-compiled skill workflow execution
    - MemoryBrain: Long-term memory retrieval and reasoning
    - PlanningBrain: Task decomposition and DAG-based planning
    - ActionBrain: Tool execution engine (shell, file, web, code)
    - WorldBrain (future): World state maintenance
    - FrontierBrain: Cloud LLM reasoning
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
from .frontier_brain import FrontierBrain
from .memory_brain import MemoryBrain
from .planning_brain import PlanningBrain, SubTask, Plan
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
    "FrontierBrain",
    "MemoryBrain",
    "PlanningBrain",
    "SkillBrain",
    "ActionBrain",
    # Engine
    "SkillRegistry",
    # Planning types
    "SubTask",
    "Plan",
]
