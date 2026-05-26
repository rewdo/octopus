"""
Octopus Self-Healing System.

Provides automatic error recovery for the Octopus multi-brain agent:
    - SelfHealer: Core recovery orchestrator
    - RetryPolicy: Configurable exponential backoff retry strategy
    - Checkpoint: Serializable state snapshots for task recovery
"""

from .self_healer import SelfHealer, RetryPolicy, Checkpoint, HealAction

__all__ = ["SelfHealer", "RetryPolicy", "Checkpoint", "HealAction"]
