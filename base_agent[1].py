"""
agents/base_agent.py
--------------------
Abstract base class for all agents in the micro-orchestrator framework.

Every concrete agent must implement:
  - ``ROLE``          : a short identifier used in logs and state records.
  - ``system_prompt`` : property returning the agent's persona/instructions.
  - ``execute()``     : the main entry-point called by the Orchestrator.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

from state import MissionState, TaskStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration constants – adjust to match your local Ollama setup
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
OLLAMA_API_KEY: str = "ollama"          # Ollama ignores this; required by openai client
WORKER_MODEL: str = "llama3:8b"         # fast model used by specialist workers
MANAGER_MODEL: str = "llama3:8b"        # can swap to a larger model if available

MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 2.0         # seconds; doubles each retry (exponential)
REQUEST_TIMEOUT: float = 120.0          # seconds per LLM call


def build_ollama_client() -> OpenAI:
    """Return an OpenAI-compatible client pointed at the local Ollama server."""
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY,
        timeout=REQUEST_TIMEOUT,
    )


# ---------------------------------------------------------------------------
# Base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Subclasses declare a ``ROLE`` class attribute and implement
    ``system_prompt`` + ``execute()``.  The base class handles:

    * Building and caching the Ollama client.
    * A generic ``_call_llm()`` helper with exponential-backoff retries.
    * Uniform debug/error logging.
    * Writing results back to the shared ``MissionState``.
    """

    ROLE: str = "base"          # overridden by every subclass
    MODEL: str = WORKER_MODEL   # subclasses may override

    def __init__(self) -> None:
        self._client: OpenAI = build_ollama_client()
        self._logger = logging.getLogger(f"{__name__}.{self.ROLE}")
        self._logger.debug("%s agent initialised (model=%s).", self.ROLE, self.MODEL)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system-level instruction string."""

    @abstractmethod
    def execute(self, task_id: str, description: str, state: MissionState) -> str:
        """
        Execute a single sub-task and return the raw text output.

        Parameters
        ----------
        task_id : str
            Unique identifier for this sub-task (used in state records).
        description : str
            Plain-English description of what this agent must do.
        state : MissionState
            Shared state object; read prior outputs, write new context.

        Returns
        -------
        str
            The agent's final text output for this sub-task.
        """

    # ------------------------------------------------------------------
    # LLM call helper (with retry logic)
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        extra_body: dict[str, Any] | None = None,
    ) -> str:
        """
        Send a chat completion request to Ollama with exponential-backoff retry.

        Parameters
        ----------
        messages :
            Standard OpenAI message list (role + content dicts).
        temperature :
            Sampling temperature (lower = more deterministic).
        max_tokens :
            Maximum tokens to generate.
        extra_body :
            Additional kwargs forwarded to the API (e.g. ``response_format``).

        Returns
        -------
        str
            The assistant's reply text, stripped of leading/trailing whitespace.

        Raises
        ------
        RuntimeError
            If all retries are exhausted without a successful response.
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt < MAX_RETRIES:
            attempt += 1
            backoff = RETRY_BACKOFF_BASE ** (attempt - 1)

            try:
                self._logger.debug(
                    "[%s] LLM call attempt %d/%d …", self.ROLE, attempt, MAX_RETRIES
                )
                kwargs: dict[str, Any] = {
                    "model": self.MODEL,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if extra_body:
                    kwargs["extra_body"] = extra_body

                response = self._client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                content = content.strip()

                self._logger.debug(
                    "[%s] LLM responded (%d chars).", self.ROLE, len(content)
                )
                return content

            except (APIConnectionError, APITimeoutError) as exc:
                last_error = exc
                self._logger.warning(
                    "[%s] Connection/timeout error on attempt %d: %s. "
                    "Retrying in %.1fs …",
                    self.ROLE, attempt, exc, backoff,
                )
            except APIStatusError as exc:
                last_error = exc
                self._logger.warning(
                    "[%s] API status error %d on attempt %d: %s. "
                    "Retrying in %.1fs …",
                    self.ROLE, exc.status_code, attempt, exc.message, backoff,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._logger.error(
                    "[%s] Unexpected error on attempt %d: %r. Retrying in %.1fs …",
                    self.ROLE, attempt, exc, backoff,
                )

            if attempt < MAX_RETRIES:
                time.sleep(backoff)

        raise RuntimeError(
            f"[{self.ROLE}] LLM call failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Convenience: run and record in one shot
    # ------------------------------------------------------------------

    def run_and_record(
        self, task_id: str, description: str, state: MissionState
    ) -> CompletedTaskResult:
        """
        Execute the sub-task, time it, record it in state, and return a
        lightweight result object with the output and status.
        """
        self._logger.info("[%s] Starting task %r …", self.ROLE, task_id)
        start = time.monotonic()
        status = TaskStatus.FAILED
        output = ""

        try:
            output = self.execute(task_id, description, state)
            status = TaskStatus.COMPLETED
            self._logger.info(
                "[%s] Task %r completed (%.2fs).", self.ROLE, task_id,
                time.monotonic() - start,
            )
        except Exception as exc:  # noqa: BLE001
            output = f"ERROR: {exc}"
            self._logger.error(
                "[%s] Task %r FAILED: %s", self.ROLE, task_id, exc
            )
        finally:
            duration = time.monotonic() - start
            state.record_task(
                task_id=task_id,
                agent_role=self.ROLE,
                description=description,
                output=output,
                status=status,
                duration_seconds=duration,
            )

        return CompletedTaskResult(
            task_id=task_id,
            output=output,
            status=status,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(role={self.ROLE!r}, model={self.MODEL!r})"


# ---------------------------------------------------------------------------
# Thin result wrapper (avoids exposing full CompletedTask to callers)
# ---------------------------------------------------------------------------

class CompletedTaskResult:
    """Lightweight result returned by ``run_and_record``."""

    __slots__ = ("task_id", "output", "status")

    def __init__(self, task_id: str, output: str, status: TaskStatus) -> None:
        self.task_id = task_id
        self.output = output
        self.status = status

    @property
    def succeeded(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    def __repr__(self) -> str:
        return (
            f"CompletedTaskResult(task_id={self.task_id!r}, "
            f"status={self.status.value}, succeeded={self.succeeded})"
        )
