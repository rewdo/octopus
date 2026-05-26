"""
SelfHealer — automatic error recovery and retry engine.

Provides:
    - RetryPolicy: Configurable exponential backoff with jitter
    - Checkpoint: Serializable state snapshots saved to disk
    - SelfHealer: Coordinates retry/fallback/abort decisions

Integration point (future): OctopusAgent.run() — wrap LLM calls with
execute_with_retry, save checkpoints between stages, and call recover()
on unexpected failures.
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── RetryPolicy ─────────────────────────────────────────────────────────────


@dataclass
class RetryPolicy:
    """Exponential backoff retry configuration.

    Default: 3 retries, starting at 1s, doubling each attempt, capped at 30s.
    Jitter (±25%) is applied to prevent thundering-herd on shared resources.
    """

    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    backoff_factor: float = 2.0
    max_delay: float = 30.0
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        OSError,
    )
    jitter: float = 0.25  # ±25% random jitter

    def compute_delay(self, attempt: int) -> float:
        """Compute delay for the given *attempt* (0-indexed).

        delay = base_delay * (backoff_factor ** attempt), clamped to max_delay,
        with ±jitter multiplicative randomisation.
        """
        raw = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(raw, self.max_delay)
        if self.jitter > 0:
            factor = 1.0 + random.uniform(-self.jitter, self.jitter)
            delay *= factor
        return delay

    def is_retryable(self, exc: Exception) -> bool:
        """Return True if *exc* matches one of the retryable exception types."""
        return isinstance(exc, self.retryable_exceptions)


# ── HealAction ──────────────────────────────────────────────────────────────


class HealAction(str, enum.Enum):
    """Possible recovery actions after a failure."""

    RETRY = "retry"          # try again (possibly with backoff)
    FALLBACK = "fallback"    # degrade to cheaper / local brain
    ALERT = "alert"          # escalate to human
    ABORT = "abort"          # cannot recover; stop the task


# ── AttemptRecord ───────────────────────────────────────────────────────────


@dataclass
class _AttemptRecord:
    attempt: int
    delay: float
    outcome: Optional[Any] = None
    error: Optional[Exception] = None
    start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end: Optional[datetime] = None


# ── Checkpoint ──────────────────────────────────────────────────────────────


@dataclass
class Checkpoint:
    """A serializable snapshot of task state at a given stage.

    Checkpoints are written to disk so that an interrupted task can be
    resumed from the last good point instead of starting over.
    """

    task_id: str
    stage: str  # "route" | "compile" | "execute" | "verify" | …
    state: dict[str, Any]  # arbitrary serializable state
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "stage": self.stage,
            "state": self.state,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        return cls(
            task_id=data["task_id"],
            stage=data["stage"],
            state=data["state"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


# ── SelfHealer ──────────────────────────────────────────────────────────────


class SelfHealer:
    """Coordinates retry logic, checkpoints, and recovery decisions.

    Usage::

        healer = SelfHealer(checkpoint_dir="./checkpoints")

        # ----- Retry wrapper -----
        result = await healer.execute_with_retry(my_llm_call, prompt, model="gpt")

        # ----- Checkpoints -----
        healer.save_checkpoint("task-1", "route", {"intent": "summarize"})
        cp = healer.load_checkpoint("task-1")      # resume after crash
        healer.clear_checkpoint("task-1")

        # ----- Recovery decision -----
        decision = healer.recover("task-1", error)
        # decision == {"action": "retry", "reason": "...", "fallback_brain": None}
    """

    def __init__(self, checkpoint_dir: str | Path | None = None):
        """Initialise the self-healer.

        Args:
            checkpoint_dir: Directory for on-disk checkpoints.
                            Defaults to ``./checkpoints`` (relative to CWD).
        """
        self._checkpoints: dict[str, Checkpoint] = {}
        self._retry_policy = RetryPolicy()
        self._checkpoint_dir: Path = (
            Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        )
        self._attempts: dict[str, list[_AttemptRecord]] = {}

    # ── Retry execution ─────────────────────────────────────────────────

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        task_id: str = "",
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> Any:
        """Execute *fn* with exponential-backoff retry.

        Args:
            fn: Async or sync callable.
            *args: Positional args forwarded to *fn*.
            task_id: Optional task identifier for logging and tracking.
            retry_policy: Override the default RetryPolicy.
            **kwargs: Keyword args forwarded to *fn*.

        Returns:
            The return value of *fn* on success.

        Raises:
            RuntimeError: When all retries are exhausted.
                The exception carries ``__attempts__`` and ``__last_error__``
                attributes for programmatic inspection.
        """
        policy = retry_policy or self._retry_policy
        call_label = f"{task_id}:{fn.__name__}" if task_id else fn.__name__
        last_error: Optional[Exception] = None
        records: list[_AttemptRecord] = []

        for attempt in range(policy.max_retries + 1):
            delay = policy.compute_delay(attempt)
            rec = _AttemptRecord(attempt=attempt, delay=delay)
            records.append(rec)

            if attempt > 0:
                logger.info(
                    "Retry %s/%s for %r (delay %.2fs)",
                    attempt,
                    policy.max_retries,
                    call_label,
                    delay,
                )
                await asyncio.sleep(delay)

            try:
                # Support both async and sync callables transparently.
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
                rec.outcome = result
                rec.end = datetime.now(timezone.utc)
                self._attempts.setdefault(task_id, []).extend(records)
                return result
            except Exception as exc:
                rec.error = exc
                rec.end = datetime.now(timezone.utc)
                last_error = exc

                if not policy.is_retryable(exc):
                    logger.warning(
                        "Non-retryable exception for %r: %s", call_label, exc
                    )
                    break  # don't retry; exit immediately

                logger.warning(
                    "Attempt %s/%s for %r failed: %s",
                    attempt + 1,
                    policy.max_retries + 1,
                    call_label,
                    exc,
                )

        # All retries exhausted.
        self._attempts.setdefault(task_id, []).extend(records)
        msg = (
            f"All {policy.max_retries + 1} attempt(s) for {call_label!r} "
            f"exhausted. Last error: {last_error}"
        )
        logger.error(msg)
        err = RuntimeError(msg)
        err.__attempts__ = records  # type: ignore[attr-defined]
        err.__last_error__ = last_error  # type: ignore[attr-defined]
        raise err

    # ── Checkpoints ─────────────────────────────────────────────────────

    def save_checkpoint(self, task_id: str, stage: str, state: dict[str, Any]) -> Checkpoint:
        """Persist a checkpoint to memory and disk.

        Args:
            task_id: Unique task identifier.
            stage: Pipeline stage name ("route", "compile", "execute", "verify", …).
            state: Arbitrary JSON-serialisable state snapshot.

        Returns:
            The created :class:`Checkpoint`.
        """
        cp = Checkpoint(task_id=task_id, stage=stage, state=state)
        self._checkpoints[task_id] = cp
        self._write_checkpoint(cp)
        logger.debug("Checkpoint saved: task=%s stage=%s", task_id, stage)
        return cp

    def load_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from memory, falling back to disk.

        Returns ``None`` if no checkpoint exists for *task_id*.
        """
        if task_id in self._checkpoints:
            return self._checkpoints[task_id]
        return self._read_checkpoint(task_id)

    def clear_checkpoint(self, task_id: str) -> None:
        """Remove a checkpoint from memory and disk."""
        self._checkpoints.pop(task_id, None)
        path = self._checkpoint_path(task_id)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    # ── Recovery ────────────────────────────────────────────────────────

    def recover(self, task_id: str, error: Exception) -> dict[str, Any]:
        """Decide how to recover from *error* for a given task.

        Returns a dict::

            {
                "action": "retry" | "fallback" | "abort" | "alert",
                "reason": "...",
                "fallback_brain": "cheap" | "skill" | None,
                "suggested_delay": float | None,
            }
        """
        error_type = type(error).__name__
        error_msg = str(error).lower()

        # ── Model / LLM timeout → degrade to a cheaper brain ──
        timeout_keywords = ("timeout", "timed out", "deadline exceeded")
        if isinstance(error, TimeoutError) or any(k in error_msg for k in timeout_keywords):
            # Determine if we have a checkpoint to resume from
            cp = self.load_checkpoint(task_id)
            cp_info = f" (resume from stage={cp.stage})" if cp else ""
            return {
                "action": HealAction.FALLBACK.value,
                "reason": (
                    f"Model timeout ({error_type}). "
                    f"Suggest fallback to local/cheap brain{cp_info}."
                ),
                "fallback_brain": "cheap",
                "suggested_delay": 2.0,
            }

        # ── Tool-call failure → switch to alternative tool ──
        tool_fail_keywords = ("tool", "tool call", "action failed", "function_call")
        if any(k in error_msg for k in tool_fail_keywords):
            return {
                "action": HealAction.FALLBACK.value,
                "reason": (
                    f"Tool execution failure ({error_type}). "
                    "Suggest switching to a fallback tool or skill."
                ),
                "fallback_brain": "skill",
                "suggested_delay": None,
            }

        # ── Rate limit / quota → back off and retry ──
        rate_keywords = ("rate limit", "quota exceeded", "too many requests", "429")
        if any(k in error_msg for k in rate_keywords):
            return {
                "action": HealAction.RETRY.value,
                "reason": (
                    f"Rate limit / quota hit ({error_type}). "
                    f"Will retry with extended backoff."
                ),
                "fallback_brain": None,
                "suggested_delay": self._retry_policy.max_delay,
            }

        # ── Content filtering / safety → abort ──
        safety_keywords = ("content filter", "safety", "flagged", "policy violation")
        if any(k in error_msg for k in safety_keywords):
            return {
                "action": HealAction.ABORT.value,
                "reason": (
                    f"Content safety trigger ({error_type}). "
                    "Cannot recover — aborting task."
                ),
                "fallback_brain": None,
                "suggested_delay": None,
            }

        # ── Connection errors → retry if retryable ──
        if self._retry_policy.is_retryable(error):
            cp = self.load_checkpoint(task_id)
            if cp:
                self.save_checkpoint(task_id, cp.stage, cp.state)
            return {
                "action": HealAction.RETRY.value,
                "reason": (
                    f"Transient error ({error_type}): {error}. "
                    "Retrying from last checkpoint."
                ),
                "fallback_brain": None,
                "suggested_delay": self._retry_policy.base_delay,
            }

        # ── Unknown / fatal error → save checkpoint, suggest abort ──
        cp = self.load_checkpoint(task_id)
        if cp:
            self.save_checkpoint(task_id, cp.stage, cp.state)
        return {
            "action": HealAction.ALERT.value,
            "reason": (
                f"Unrecognised error ({error_type}): {error}. "
                "Checkpoint saved. Escalate to human or review logs."
            ),
            "fallback_brain": None,
            "suggested_delay": None,
        }

    # ── Internal helpers ────────────────────────────────────────────────

    def _checkpoint_path(self, task_id: str) -> Path:
        """Filesystem path for *task_id* checkpoint JSON."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)
        return self._checkpoint_dir / f"{safe}.checkpoint.json"

    def _write_checkpoint(self, cp: Checkpoint) -> None:
        """Persist *cp* as JSON to disk."""
        try:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = self._checkpoint_path(cp.task_id)
            path.write_text(json.dumps(cp.to_dict(), indent=2, ensure_ascii=False))
        except OSError as exc:
            logger.error("Failed to write checkpoint for %s: %s", cp.task_id, exc)

    def _read_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """Read a checkpoint JSON from disk, if it exists."""
        path = self._checkpoint_path(task_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Corrupt checkpoint for %s: %s", task_id, exc)
            return None

    # ── Inspection ──────────────────────────────────────────────────────

    @property
    def checkpoint_dir(self) -> Path:
        """The directory where checkpoints are persisted."""
        return self._checkpoint_dir

    def list_checkpoints(self) -> dict[str, Checkpoint]:
        """Return a shallow copy of all in-memory checkpoints."""
        return dict(self._checkpoints)

    def get_attempts(self, task_id: str) -> list[_AttemptRecord]:
        """Return the retry-attempt log for *task_id*."""
        return self._attempts.get(task_id, [])
