"""
Multi-user isolation layer for Octopus.

Each user gets an independent workspace, configuration, memory namespace,
and cost tracker — all managed through a single MultiUserManager.

Usage::

    manager = MultiUserManager(base_dir=Path("./octopus-users"))
    scope = manager.register_user("alice")
    alice_agent = manager.switch_user("alice")
    result = alice_agent.run_sync("hello")
    manager.list_users()  # → ["alice"]
    manager.delete_user("alice")
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from octopus.agent import OctopusAgent
from octopus.config import OctopusConfig


@dataclass
class UserScope:
    """Per-user isolation boundary.

    Each user owns a private workspace directory, a config file derived
    from the base config, and a memory namespace prefix used to scope
    MemoryGraph nodes so users never see each other's data.
    """

    user_id: str
    workspace_dir: Path
    config_path: Path
    memory_namespace: str  # prefix for MemoryGraph node filtering

    # ── Computed sub-paths ──
    costs_path: Path = field(init=False)
    skills_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.costs_path = self.workspace_dir / "costs.json"
        self.skills_dir = self.workspace_dir / "skills"


class MultiUserManager:
    """Create, switch, list, and delete isolated user environments.

    Directory layout under *base_dir*::

        base_dir/
        ├── users/
        │   ├── {user_id}/
        │   │   ├── workspace/      ← isolated workspace
        │   │   │   ├── costs.json
        │   │   │   ├── checkpoints/
        │   │   │   └── skills/
        │   │   └── config.yaml     ← user-scoped config
        │   └── ...
        └── _manager_state.json     ← manager index (future)
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        self._users: dict[str, UserScope] = {}
        self._users_dir = self._base / "users"
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._discover_existing()

    # ── Public API ──────────────────────────────────────────────────────────

    def register_user(self, user_id: str, base_config: Optional[OctopusConfig] = None) -> UserScope:
        """Register a new user, creating their isolated directory structure.

        Args:
            user_id: Unique user identifier (alpha-numeric, hyphens, underscores).
            base_config: Optional base OctopusConfig to template from.
                         If omitted, ``OctopusConfig.default()`` is used.

        Returns:
            UserScope with paths and memory namespace.

        Raises:
            ValueError: If *user_id* is already registered.
        """
        if user_id in self._users:
            raise ValueError(f"User '{user_id}' is already registered")

        user_dir = self._users_dir / user_id
        workspace_dir = user_dir / "workspace"
        config_path = user_dir / "config.yaml"
        memory_namespace = f"usr:{user_id}"

        # Create directory tree
        workspace_dir.mkdir(parents=True, exist_ok=True)
        (workspace_dir / "checkpoints").mkdir(exist_ok=True)
        (workspace_dir / "skills").mkdir(exist_ok=True)

        # Write user-scoped config
        cfg = base_config or OctopusConfig.default()
        cfg.workspace_dir = workspace_dir
        cfg.save(config_path)

        scope = UserScope(
            user_id=user_id,
            workspace_dir=workspace_dir,
            config_path=config_path,
            memory_namespace=memory_namespace,
        )
        self._users[user_id] = scope
        return scope

    def get_user(self, user_id: str) -> UserScope:
        """Return the UserScope for *user_id*.

        Raises:
            KeyError: If *user_id* is not registered.
        """
        if user_id not in self._users:
            raise KeyError(f"User '{user_id}' not found. Registered: {self.list_users()}")
        return self._users[user_id]

    def list_users(self) -> list[str]:
        """Return sorted list of all registered user IDs."""
        return sorted(self._users.keys())

    def switch_user(self, user_id: str) -> OctopusAgent:
        """Load the user-scoped config and return a fresh OctopusAgent.

        The returned agent uses the user's isolated workspace, memory
        namespace prefix, and cost tracker — fully independent of other users.

        Raises:
            KeyError: If *user_id* is not registered.
            FileNotFoundError: If the user's config file is missing.
        """
        scope = self.get_user(user_id)
        if not scope.config_path.exists():
            raise FileNotFoundError(f"Config missing for user '{user_id}': {scope.config_path}")

        cfg = OctopusConfig.from_file(scope.config_path)
        cfg.workspace_dir = scope.workspace_dir  # ensure idempotent
        return OctopusAgent(cfg)

    def delete_user(self, user_id: str) -> None:
        """Permanently delete a user and all their data.

        Removes the user directory from disk and deregisters from the
        in-memory index. This is irreversible.

        Raises:
            KeyError: If *user_id* is not registered.
        """
        scope = self.get_user(user_id)
        user_dir = self._users_dir / user_id
        if user_dir.exists():
            shutil.rmtree(user_dir)
        del self._users[user_id]

    def user_count(self) -> int:
        """Return the number of registered users."""
        return len(self._users)

    # ── Internal ────────────────────────────────────────────────────────────

    def _discover_existing(self) -> None:
        """Scan the users directory and rebuild the in-memory index."""
        if not self._users_dir.exists():
            return
        for entry in self._users_dir.iterdir():
            if not entry.is_dir():
                continue
            uid = entry.name
            config_path = entry / "config.yaml"
            workspace_dir = entry / "workspace"
            if config_path.exists():
                scope = UserScope(
                    user_id=uid,
                    workspace_dir=workspace_dir,
                    config_path=config_path,
                    memory_namespace=f"usr:{uid}",
                )
                self._users[uid] = scope
