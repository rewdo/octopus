"""World state management: live snapshot of the agent's environment.

Maintains a key-value state map, tracks file changes, captures environment
variables, and auto-updates after every tool call.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from octopus.brains.base import BaseBrain, BrainRequest, BrainResponse, BrainType


class WorldState:
    """Maintains the agent's view of the "current world".

    Tracks:
    - Arbitrary key-value state
    - Filesystem snapshots (watched files)
    - Environment variables
    - System info
    - Tool call results (auto-update)
    """

    def __init__(self):
        # Generic state store
        self._state: dict[str, Any] = {}

        # Watched files: path → (mtime, sha256_hash)
        self._watched_files: dict[str, tuple[float, str]] = {}

        # Last snapshot for diffing
        self._last_snapshot: Optional[dict] = None

        # Initialize with system info
        self._init_system_info()

    # ── Generic state ────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the world state."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in the world state."""
        self._state[key] = value

    def delete(self, key: str) -> None:
        """Delete a key from the world state."""
        self._state.pop(key, None)

    def snapshot(self) -> dict:
        """Take a full snapshot of the current world state.

        Returns a dict containing:
        - state: The key-value state map
        - system: System info (OS, hostname, etc.)
        - watched_files: Current state of watched files
        - env: Subset of environment variables
        """
        snap = {
            "state": deepcopy(self._state),
            "system": self._state.get("_system", {}),
            "watched_files": deepcopy(self._watched_files),
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._last_snapshot = snap
        return snap

    def diff(self, previous_snapshot: Optional[dict] = None) -> dict:
        """Compare current state against a previous snapshot.

        Returns dict with 'added', 'removed', 'changed' keys.
        If previous_snapshot is None, uses the last stored snapshot.
        """
        if previous_snapshot is None:
            previous_snapshot = self._last_snapshot

        current = self.snapshot()
        if previous_snapshot is None:
            return {"added": current["state"], "removed": {}, "changed": {}}

        prev_state = previous_snapshot.get("state", {})

        added = {}
        removed = {}
        changed = {}

        all_keys = set(prev_state.keys()) | set(current["state"].keys())
        for key in all_keys:
            in_prev = key in prev_state
            in_curr = key in current["state"]
            if in_curr and not in_prev:
                added[key] = current["state"][key]
            elif in_prev and not in_curr:
                removed[key] = prev_state[key]
            elif prev_state.get(key) != current["state"].get(key):
                changed[key] = {
                    "old": prev_state.get(key),
                    "new": current["state"].get(key),
                }

        return {"added": added, "removed": removed, "changed": changed}

    # ── File watching ────────────────────────────────────────────────────

    def watch_file(self, path: str | Path) -> None:
        """Start watching a file for changes.

        Records the file's mtime and SHA-256 hash.
        """
        path = str(Path(path).resolve())
        if os.path.isfile(path):
            stat = os.stat(path)
            mtime = stat.st_mtime
            try:
                with open(path, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()
            except (OSError, PermissionError):
                content_hash = ""
            self._watched_files[path] = (mtime, content_hash)
        else:
            self._watched_files[path] = (0.0, "FILE_NOT_FOUND")

    def check_file_changes(self) -> dict[str, str]:
        """Check all watched files for changes.

        Returns dict of path → change type: 'modified', 'deleted', 'created', 'unchanged'.
        """
        changes: dict[str, str] = {}
        for path, (old_mtime, old_hash) in list(self._watched_files.items()):
            if not os.path.isfile(path):
                if old_hash != "FILE_NOT_FOUND":
                    changes[path] = "deleted"
                    self._watched_files[path] = (0.0, "FILE_NOT_FOUND")
                else:
                    changes[path] = "unchanged"
            else:
                stat = os.stat(path)
                new_mtime = stat.st_mtime
                if new_mtime != old_mtime:
                    try:
                        with open(path, "rb") as f:
                            new_hash = hashlib.sha256(f.read()).hexdigest()
                    except (OSError, PermissionError):
                        new_hash = ""
                    if new_hash != old_hash:
                        changes[path] = "modified"
                    else:
                        changes[path] = "unchanged"
                    self._watched_files[path] = (new_mtime, new_hash)
                else:
                    changes[path] = "unchanged"

        return changes

    # ── Environment ──────────────────────────────────────────────────────

    @staticmethod
    def get_env(var_name: str, default: str = "") -> str:
        """Get an environment variable."""
        return os.environ.get(var_name, default)

    def set_env(self, var_name: str, value: str) -> None:
        """Set an environment variable (process-level)."""
        os.environ[var_name] = value
        self._state.setdefault("_env_overrides", {})
        self._state["_env_overrides"][var_name] = value

    # ── Tool call update ─────────────────────────────────────────────────

    def update_from_tool_call(self, tool_name: str, params: dict, result: Any) -> None:
        """Update world state after a tool call completes.

        This is the core hook that keeps the world model fresh.
        Subclasses or plugins can extend this for tool-specific logic.
        """
        # Record in call history
        call_history: list[dict] = self._state.setdefault("_tool_call_history", [])
        call_entry = {
            "tool": tool_name,
            "params": params,
            "result_summary": str(result)[:500],  # truncate for sanity
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        call_history.append(call_entry)

        # Keep only last 100 calls
        if len(call_history) > 100:
            self._state["_tool_call_history"] = call_history[-100:]

        # Track call count
        call_counts: dict[str, int] = self._state.setdefault("_tool_call_counts", {})
        call_counts[tool_name] = call_counts.get(tool_name, 0) + 1

        # Tool-specific updates
        if tool_name == "write_file" or tool_name == "edit":
            path = params.get("path") or params.get("file")
            if path:
                self.watch_file(path)

        if tool_name == "exec":
            cwd = params.get("cwd") or params.get("workdir")
            if cwd:
                self.set("_last_cwd", cwd)

    # ── Initialization helpers ───────────────────────────────────────────

    def _init_system_info(self) -> None:
        """Initialize system information in the world state."""
        self._state["_system"] = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "cwd": os.getcwd(),
        }

    # ── Convenience ──────────────────────────────────────────────────────

    def get_git_branch(self, repo_path: str = ".") -> str:
        """Get the current git branch for a repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return ""

    def get_db_connections(self) -> list[dict]:
        """Get active database connections from state.

        Plugins can store connection info under '_db_connections'.
        """
        return self._state.get("_db_connections", [])


class WorldBrain(BaseBrain):
    """Brain that maintains and queries the world state.

    Handles:
    - State queries (get/set/list operations)
    - Environment variable lookup
    - File change detection
    - Tool call result recording
    """

    brain_type = BrainType.WORLD

    def __init__(self, config: Any = None):
        super().__init__(config)
        self.state = WorldState()

    async def process(self, request: BrainRequest) -> BrainResponse:
        """Process a world-state query or update.

        Request metadata can specify action:
        - "snapshot": return full state snapshot
        - "diff": return diff since last snapshot
        - "get:<key>": return specific key
        - "set:<key>:<value>": set a key
        - Otherwise: return basic state summary
        """
        action = request.metadata.get("action", "summary")
        content = ""
        structured = None

        try:
            if action == "snapshot":
                structured = self.state.snapshot()
                content = f"World snapshot: {len(structured.get('state', {}))} keys"
            elif action == "diff":
                diff_result = self.state.diff()
                structured = diff_result
                added = len(diff_result.get("added", {}))
                changed = len(diff_result.get("changed", {}))
                removed = len(diff_result.get("removed", {}))
                content = f"Diff: +{added} added, ~{changed} changed, -{removed} removed"
            elif action.startswith("get:"):
                key = action[4:]
                value = self.state.get(key)
                structured = {key: value}
                content = str(value)
            elif action.startswith("set:"):
                parts = action[4:].split(":", 1)
                key = parts[0]
                value = parts[1] if len(parts) > 1 else ""
                self.state.set(key, value)
                structured = {key: value}
                content = f"Set '{key}' = '{value}'"
            elif action == "env":
                var_name = request.metadata.get("var", "")
                value = self.state.get_env(var_name)
                structured = {var_name: value}
                content = str(value)
            else:
                # Summary
                keys = list(self.state._state.keys())
                watched = len(self.state._watched_files)
                calls = len(self.state.get("_tool_call_history", []))
                content = (
                    f"WorldBrain: {len(keys)} state keys, "
                    f"{watched} watched files, {calls} tool calls recorded"
                )
                structured = {
                    "state_key_count": len(keys),
                    "watched_files": watched,
                    "tool_calls": calls,
                    "keys": keys,
                }

            self._total_calls += 1

            return BrainResponse(
                success=True,
                content=content,
                brain_type=self.brain_type,
                structured_output=structured,
            )
        except Exception as e:
            return BrainResponse(
                success=False,
                content=str(e),
                brain_type=self.brain_type,
                errors=[str(e)],
            )

    def can_handle(self, request: BrainRequest) -> bool:
        """WorldBrain handles state queries, env lookups, and world information."""
        action = request.metadata.get("action", "")
        state_keywords = [
            "snapshot", "diff", "get:", "set:", "env",
            "state", "world", "environment", "git branch",
            "file changed", "current directory",
        ]
        lower_input = request.user_input.lower()
        return any(kw in action or kw in lower_input for kw in state_keywords)
