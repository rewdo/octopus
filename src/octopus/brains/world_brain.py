"""
World Brain — maintains the agent's environment state snapshot.

Tracks current working directory, environment variables, file system
state, git branch, and other contextual information. Provides a
consistent view of "the current world" for other brains.

Phase 1: Key-value in-memory state with JSON snapshot.
Phase 2: Full world model engine with file watching and tool-call integration.
"""

from __future__ import annotations

import os
import platform
from typing import Any, Optional

from octopus.brains.base import BaseBrain, BrainRequest, BrainResponse, BrainType


class WorldBrain(BaseBrain):
    """Maintains and queries the agent's current environment state.

    Provides a simple key-value state store plus pre-populated
    system information (OS, Python version, CWD, etc.).
    """

    # Keywords that indicate a world/state query
    _STATE_KEYWORDS = frozenset({
        "env", "environment", "state", "status", "snapshot", "diff",
        "current directory", "current branch", "system", "cwd", "pwd",
        "what is the state", "what's the state", "show environment",
        "check status", "what is my", "what's my",
        "环境", "状态", "当前", "快照", "差异", "系统",
    })

    def __init__(
        self,
        world_state: Any = None,  # optional WorldState from world.world_state
        config: Any = None,
    ):
        super().__init__(config)
        self._state: dict[str, Any] = {}

        # Populate initial system info
        self._state["os"] = platform.system()
        self._state["os_release"] = platform.release()
        self._state["python_version"] = platform.python_version()
        self._state["cwd"] = os.getcwd()
        self._state["hostname"] = platform.node()

        # Merge external WorldState if provided
        if world_state is not None:
            try:
                snap = world_state.snapshot()
                self._state.update(snap)
            except Exception:
                pass

    @property
    def brain_type(self) -> BrainType:
        return BrainType.WORLD

    def can_handle(self, request: BrainRequest) -> bool:
        """Check if the request is a world/state/environment query."""
        text = request.user_input.lower()
        return any(kw in text for kw in self._STATE_KEYWORDS)

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process a world-state query.

        Returns the current environment snapshot as structured output,
        with a human-readable summary as the content.
        """
        # Check for specific key queries
        text = request.user_input.lower()
        requested_key: Optional[str] = None

        # Detect specific key requests
        key_map = {
            "cwd": ["cwd", "current directory", "pwd", "working directory", "当前目录", "工作目录"],
            "os": ["os", "operating system", "操作系统"],
            "python": ["python version", "python"],
            "hostname": ["hostname", "computer name", "主机名"],
            "branch": ["git branch", "branch", "current branch", "分支"],
        }

        for key, patterns in key_map.items():
            if any(p in text for p in patterns):
                requested_key = key
                break

        if requested_key and requested_key in self._state:
            return BrainResponse(
                success=True,
                content=f"{requested_key}: {self._state[requested_key]}",
                brain_type=BrainType.WORLD,
                confidence=1.0,
                structured_output={requested_key: self._state[requested_key]},
            )

        # Full snapshot
        # Build a readable summary
        lines = ["Current Environment:"]
        for key in sorted(self._state.keys()):
            lines.append(f"  {key}: {self._state[key]}")

        return BrainResponse(
            success=True,
            content="\n".join(lines),
            brain_type=BrainType.WORLD,
            confidence=1.0,
            structured_output=dict(self._state),
        )

    def set(self, key: str, value: Any) -> None:
        """Update a specific state key."""
        self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific state key."""
        return self._state.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        """Return a full copy of the current state."""
        return dict(self._state)
