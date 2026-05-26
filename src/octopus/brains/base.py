"""
Base brain interface and shared types for the multi-brain architecture.

Every brain in Octopus inherits from BaseBrain and follows a common
request/response protocol. This enables the Cognitive Router to treat
all brains uniformly while each brain specializes in its domain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class BrainType(Enum):
    """The seven specialized brains in the Octopus architecture."""

    CHEAP = "cheap"         # Rule-based + tiny local model
    SKILL = "skill"         # Pre-compiled skill execution
    MEMORY = "memory"       # Long-term memory retrieval & reasoning
    PLANNING = "planning"   # Task decomposition & planning
    ACTION = "action"       # Tool execution (shell, browser, API)
    WORLD = "world"         # World state maintenance
    FRONTIER = "frontier"   # Cloud LLM for high-value reasoning


class TaskComplexity(Enum):
    """Estimated complexity of a task."""

    TRIVIAL = 1     # Format conversion, simple lookup
    SIMPLE = 2      # Single-step operation
    MODERATE = 3    # Multi-step with some reasoning
    COMPLEX = 4     # Requires planning and tool use
    HIGHLY_COMPLEX = 5  # Creative reasoning, cross-domain synthesis


class TaskRisk(Enum):
    """Risk level of a task."""

    NONE = 0        # No risk (informational queries)
    LOW = 1         # Minor risk (basic operations)
    MEDIUM = 2      # Moderate risk (data modification)
    HIGH = 3        # High risk (financial, security)
    CRITICAL = 4    # Critical risk (legal, medical, irreversible)


@dataclass
class BrainRequest:
    """Standardized request sent to any brain."""

    # Task identification
    task_id: str
    user_input: str

    # Context (compiled by Context Compiler, NOT raw history)
    compiled_context: str = ""
    relevant_memories: list[dict[str, Any]] = field(default_factory=list)
    relevant_skills: list[str] = field(default_factory=list)

    # Task metadata
    complexity: TaskComplexity = TaskComplexity.SIMPLE
    risk: TaskRisk = TaskRisk.NONE
    novelty_score: float = 0.0  # 0.0 (seen before) → 1.0 (completely new)

    # Constraints
    max_tokens: int = 4096
    budget_usd: float = 0.10
    timeout_seconds: int = 30

    # Tool access (Action Brain only)
    allowed_tools: list[str] = field(default_factory=list)

    # Arbitrary metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainResponse:
    """Standardized response from any brain."""

    # Core output
    success: bool
    content: str
    brain_type: BrainType

    # Metrics
    tokens_used: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    confidence: float = 1.0  # 0.0 → 1.0

    # Routing feedback
    suggested_next_brain: Optional[BrainType] = None
    should_escalate: bool = False
    escalation_reason: str = ""

    # Structured output
    structured_output: Any = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseBrain(ABC):
    """Abstract base class for all Octopus brains."""

    def __init__(self, config: Any = None):
        self.config = config
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_calls = 0

    @property
    @abstractmethod
    def brain_type(self) -> BrainType:
        """Which brain type this is."""
        ...

    @abstractmethod
    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process a task request and return a response."""
        ...

    @abstractmethod
    def can_handle(self, request: BrainRequest) -> bool:
        """Quick check: can this brain handle this request?"""
        ...

    @property
    def stats(self) -> dict[str, Any]:
        """Return usage statistics for this brain."""
        return {
            "brain_type": self.brain_type.value,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "total_calls": self._total_calls,
            "avg_tokens_per_call": (
                self._total_tokens // self._total_calls if self._total_calls > 0 else 0
            ),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_tokens = 0
        self._total_cost = 0.0
        self._total_calls = 0
