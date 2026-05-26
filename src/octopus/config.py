"""
Global configuration system for Octopus.

Supports YAML/JSON config files, environment variable overrides,
and multi-level API definitions with per-token pricing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """A single API endpoint configuration with pricing."""

    name: str = Field(..., description="Human-readable name (e.g. 'DeepSeek V4')")
    provider: str = Field(..., description="Provider name (openai, anthropic, deepseek, ...)")
    base_url: str = Field(..., description="API base URL")
    api_key: str = Field(default="", description="API key (or env var name prefixed with $)")
    model: str = Field(..., description="Model identifier")
    price_per_1k_input: float = Field(default=0.0, description="USD per 1K input tokens")
    price_per_1k_output: float = Field(default=0.0, description="USD per 1K output tokens")
    max_tokens: int = Field(default=8192, description="Max tokens per request")
    priority: int = Field(default=0, description="Lower = preferred when costs are equal")

    def resolve_api_key(self) -> str:
        """Resolve API key, supporting $ENV_VAR references."""
        if self.api_key.startswith("$"):
            env_var = self.api_key[1:]
            return os.environ.get(env_var, "")
        return self.api_key


class BudgetConfig(BaseModel):
    """Token budget and cost control settings."""

    monthly_budget_usd: float = Field(default=10.0, description="Monthly budget in USD")
    max_per_task_usd: float = Field(default=0.10, description="Max USD per single task")
    warn_threshold_pct: float = Field(default=80.0, description="Warn when budget % used")
    track_costs: bool = Field(default=True, description="Enable cost tracking")


class BrainConfig(BaseModel):
    """Configuration for each brain's execution backend."""

    cheap: str = Field(default="local_rule", description="Backend for Cheap Brain")
    skill: str = Field(default="local_engine", description="Backend for Skill Brain")
    planning: str = Field(default="api_mid", description="Backend for Planning Brain")
    frontier: str = Field(default="api_high", description="Backend for Frontier Brain")


class MemoryConfig(BaseModel):
    """Memory system configuration."""

    graph_backend: str = Field(default="networkx", description="Graph backend (networkx, neo4j)")
    vector_backend: str = Field(default="chromadb", description="Vector store backend")
    vector_dimensions: int = Field(default=1024, description="Embedding dimensions")
    working_memory_size: int = Field(default=50, description="Max items in working memory")
    importance_threshold: float = Field(default=0.3, description="Min importance to enter long-term memory")
    gc_interval_hours: int = Field(default=24, description="Garbage collection interval")


class OctopusConfig(BaseModel):
    """Root configuration for the Octopus agent."""

    # Core
    config_version: str = Field(default="1.0", description="Config schema version")
    workspace_dir: Path = Field(default=Path("./octopus-workspace"), description="Workspace directory")

    # API & Budget
    apis: list[APIConfig] = Field(default_factory=list, description="Configured API endpoints")
    budget: BudgetConfig = Field(default_factory=BudgetConfig)

    # Brains
    brains: BrainConfig = Field(default_factory=BrainConfig)

    # Memory
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Router thresholds (T1, T2, T3 from spec)
    router_thresholds: dict[str, float] = Field(
        default={
            "t1": 3.0,   # Below T1 → Cheap/Skill Brain
            "t2": 6.0,   # T1-T2 → Planning + Skill + Local
            "t3": 9.0,   # T2-T3 → Hybrid (local + compressed cloud)
                          # Above T3 → Frontier Brain
        }
    )

    # Router weights for the 9-dimension scoring formula
    router_weights: dict[str, float] = Field(
        default={
            "alpha": 1.0,    # Complexity
            "beta": 0.8,     # Novelty
            "gamma": 1.2,    # Risk
            "delta": 0.5,    # Realtime need
            "epsilon": 0.6,  # Skill confidence (negative)
            "zeta": 0.4,     # Local capability (negative)
            "eta": 0.3,      # Budget remaining (negative)
        }
    )

    @classmethod
    def from_file(cls, path: str | Path) -> "OctopusConfig":
        """Load configuration from a YAML or JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def default(cls) -> "OctopusConfig":
        """Create a default configuration with sensible defaults."""
        return cls()

    def save(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, allow_unicode=True)
