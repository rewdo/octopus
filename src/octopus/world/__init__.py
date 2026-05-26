"""World state management module for Octopus.

Maintains a live snapshot of the agent's environment: filesystem,
environment variables, git status, database connections, and more.
"""

from .world_state import WorldBrain, WorldState

__all__ = ["WorldBrain", "WorldState"]
