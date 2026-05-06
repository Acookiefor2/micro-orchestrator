"""
state.py
--------
Thread-safe global mission state shared by all agents throughout the pipeline.
"""

from __future__ import annotations

import threading
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CompletedTask:
    """An immutable record of a single completed sub-task."""
    task_id: str
    agent_role: str
    description: str
    output: str
    status: TaskStatus
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        preview = self.output[:80].replace("\n", " ")
        return (
            f"CompletedTask(id={self.task_id!r}, role={self.agent_role!r}, "
            f"status={self.status.value}, output_preview={preview!r}...)"
        )


class MissionState:
    """
    Centralised, thread-safe state object passed through the entire pipeline.

    All agents share one instance so they can read the accumulated context
    produced by previous workers without coupling themselves to each other.

    Attributes
    ----------
    goal : str
        The original user goal, set once at mission start and never mutated.
    context : dict[str, Any]
        Free-form key/value store for agents to publish and consume shared data.
        Keys should be namespaced by convention, e.g. ``"coder.script"``.
    history : list[CompletedTask]
        Append-only list of every completed sub-task, in execution order.
    critic_feedback : str | None
        The last critique returned by the Critic agent on a failed review.
    retry_count : int
        How many full pipeline retries have been attempted.
    """

    def __init__(self, goal: str) -> None:
        self._lock = threading.RLock()
        self._goal = goal
        self._context: dict[str, Any] = {}
        self._history: list[CompletedTask] = []
        self._critic_feedback: str | None = None
        self._retry_count: int = 0

        logger.info("MissionState initialised for goal: %r", goal[:120])

    # ------------------------------------------------------------------
    # Read-only access (no lock needed for immutable scalar)
    # ------------------------------------------------------------------

    @property
    def goal(self) -> str:
        return self._goal

    @property
    def retry_count(self) -> int:
        with self._lock:
            return self._retry_count

    @property
    def critic_feedback(self) -> str | None:
        with self._lock:
            return self._critic_feedback

    # ------------------------------------------------------------------
    # History (append-only)
    # ------------------------------------------------------------------

    def record_task(
        self,
        task_id: str,
        agent_role: str,
        description: str,
        output: str,
        status: TaskStatus,
        duration_seconds: float,
    ) -> CompletedTask:
        """Append a completed task record and return it."""
        entry = CompletedTask(
            task_id=task_id,
            agent_role=agent_role,
            description=description,
            output=output,
            status=status,
            duration_seconds=duration_seconds,
        )
        with self._lock:
            self._history.append(entry)
        logger.debug("Recorded task: %r", entry)
        return entry

    def get_history(self) -> list[CompletedTask]:
        """Return a shallow copy of the history list (safe for iteration)."""
        with self._lock:
            return list(self._history)

    def get_history_as_text(self) -> str:
        """
        Render the full task history as a human-readable string so agents can
        include it verbatim in their prompts as prior context.
        """
        with self._lock:
            if not self._history:
                return "(no prior tasks completed)"
            parts: list[str] = []
            for i, task in enumerate(self._history, 1):
                parts.append(
                    f"[Task {i} | {task.agent_role} | {task.status.value}]\n"
                    f"Description: {task.description}\n"
                    f"Output:\n{task.output}\n"
                )
            return "\n---\n".join(parts)

    # ------------------------------------------------------------------
    # Context store
    # ------------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        """Write a value to the shared context store."""
        with self._lock:
            self._context[key] = value
        logger.debug("Context updated: key=%r", key)

    def get(self, key: str, default: Any = None) -> Any:
        """Read a value from the shared context store."""
        with self._lock:
            return self._context.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        """Return a deep-copy-safe snapshot of the entire context dict."""
        with self._lock:
            return dict(self._context)

    # ------------------------------------------------------------------
    # Critic / retry
    # ------------------------------------------------------------------

    def set_critic_feedback(self, feedback: str) -> None:
        with self._lock:
            self._critic_feedback = feedback
            self._retry_count += 1
        logger.warning(
            "Critic rejected output (retry #%d). Feedback: %s",
            self._retry_count,
            feedback[:200],
        )

    def clear_critic_feedback(self) -> None:
        with self._lock:
            self._critic_feedback = None

    def reset_for_retry(self) -> None:
        """
        Clear history and context so the pipeline can run fresh while keeping
        the goal, retry count, and latest critic feedback intact.
        """
        with self._lock:
            self._history.clear()
            self._context.clear()
        logger.info("MissionState reset for retry #%d.", self._retry_count)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"MissionState(goal={self._goal[:60]!r}, "
                f"tasks_completed={len(self._history)}, "
                f"retries={self._retry_count})"
            )
