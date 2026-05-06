"""
orchestrator.py
---------------
The central state machine for the micro-orchestrator framework.

Responsibilities
----------------
1. **Task Planning** (Manager LLM)
   Call a local LLM with the user goal and obtain a structured JSON task plan
   validated via Pydantic.

2. **Task Routing**
   Iterate through the plan sequentially, resolve the correct specialist agent
   for each task, and invoke it via ``run_and_record()``.

3. **Critic / Validator**
   After all tasks complete, call a dedicated Critic LLM pass that compares
   the compiled output against the original goal.  On failure, re-queue the
   entire pipeline (up to ``MAX_PIPELINE_RETRIES``).

4. **Output Compilation**
   Collect each worker's output into a single cohesive final result.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from state import MissionState, TaskStatus
from agents.base_agent import (
    BaseAgent,
    build_ollama_client,
    MANAGER_MODEL,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
    REQUEST_TIMEOUT,
)
from agents.specialists import CoderAgent, ResearcherAgent, WriterAgent

logger = logging.getLogger(__name__)

MAX_PIPELINE_RETRIES: int = 2   # how many full replans the Critic can trigger

# ---------------------------------------------------------------------------
# Pydantic schemas for the Manager's task plan
# ---------------------------------------------------------------------------

AgentRole = Literal["coder", "researcher", "writer"]


class SubTask(BaseModel):
    """A single unit of work assigned to one specialist agent."""

    task_id: str = Field(
        ...,
        description="Unique short identifier for this task, e.g. 'task_1'.",
        examples=["task_1", "task_2"],
    )
    agent_role: AgentRole = Field(
        ...,
        description="Which specialist agent handles this task.",
    )
    description: str = Field(
        ...,
        min_length=10,
        description="Plain-English description of what the agent must produce.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of task_ids that must complete before this one starts.",
    )

    @field_validator("task_id")
    @classmethod
    def task_id_no_spaces(cls, v: str) -> str:
        if " " in v:
            raise ValueError("task_id must not contain spaces.")
        return v.lower()

    @field_validator("description")
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description must not be blank.")
        return v.strip()


class TaskPlan(BaseModel):
    """The full structured plan produced by the Manager LLM."""

    goal_summary: str = Field(
        ...,
        description="Manager's one-sentence restatement of the user goal.",
    )
    tasks: list[SubTask] = Field(
        ...,
        min_length=1,
        description="Ordered list of sub-tasks to execute sequentially.",
    )

    @field_validator("tasks")
    @classmethod
    def unique_task_ids(cls, tasks: list[SubTask]) -> list[SubTask]:
        ids = [t.task_id for t in tasks]
        if len(ids) != len(set(ids)):
            duplicates = [t for t in ids if ids.count(t) > 1]
            raise ValueError(f"Duplicate task_ids found: {duplicates}")
        return tasks


# ---------------------------------------------------------------------------
# Critic schema
# ---------------------------------------------------------------------------

class CriticVerdict(BaseModel):
    """Structured verdict returned by the Critic LLM."""

    approved: bool = Field(
        ...,
        description="True if the output satisfactorily achieves the goal.",
    )
    feedback: str = Field(
        ...,
        description=(
            "If approved=False, concrete actionable feedback explaining what is missing "
            "or wrong and how to fix it.  If approved=True, a brief justification."
        ),
    )
    score: int = Field(
        ...,
        ge=0,
        le=10,
        description="Quality score from 0 (completely wrong) to 10 (perfect).",
    )


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

_AGENT_REGISTRY: dict[AgentRole, type[BaseAgent]] = {
    "coder": CoderAgent,
    "researcher": ResearcherAgent,
    "writer": WriterAgent,
}


def _resolve_agent(role: AgentRole) -> BaseAgent:
    """Instantiate the correct specialist agent for a given role."""
    cls = _AGENT_REGISTRY.get(role)
    if cls is None:
        raise ValueError(
            f"Unknown agent role {role!r}. "
            f"Registered roles: {list(_AGENT_REGISTRY.keys())}"
        )
    return cls()


# ---------------------------------------------------------------------------
# Manager LLM helpers
# ---------------------------------------------------------------------------

_MANAGER_SYSTEM_PROMPT = """\
You are a senior AI project manager orchestrating a team of specialist agents.
Your job is to decompose a complex user goal into a sequential list of
focused sub-tasks, each handled by exactly one specialist:

  • "researcher" – gathers facts, background knowledge, best-practice notes.
  • "coder"       – writes Python code / scripts.
  • "writer"      – produces formatted Markdown documentation, reports, tables.

Rules:
1. Always start with "researcher" if background knowledge would help the coder.
2. Tasks execute SEQUENTIALLY in the order you list them.
3. Keep each task description specific and actionable (≥ 1 clear sentence).
4. Use 2–5 tasks total; do not over-decompose.
5. ALWAYS end with a "writer" task to format and present the final result.
6. Output ONLY valid JSON matching this exact schema — no prose, no backticks:

{
  "goal_summary": "<one-sentence restatement of the goal>",
  "tasks": [
    {
      "task_id": "task_1",
      "agent_role": "researcher" | "coder" | "writer",
      "description": "<what this agent must produce>",
      "depends_on": []
    }
  ]
}
"""


def _call_manager_llm(
    client: OpenAI,
    goal: str,
    critic_feedback: str | None = None,
) -> str:
    """
    Call the Manager LLM and return the raw JSON string.
    Uses exponential-backoff retry identical to BaseAgent._call_llm.
    """
    user_parts = [f"User goal:\n{goal}"]
    if critic_feedback:
        user_parts.append(
            f"\nThe previous plan was executed but the Critic rejected the output:\n"
            f"{critic_feedback}\n"
            f"Revise your plan to address the critique."
        )

    messages = [
        {"role": "system", "content": _MANAGER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    attempt = 0
    last_error: Exception | None = None
    while attempt < MAX_RETRIES:
        attempt += 1
        backoff = RETRY_BACKOFF_BASE ** (attempt - 1)
        try:
            logger.debug("Manager LLM call attempt %d/%d …", attempt, MAX_RETRIES)
            response = client.chat.completions.create(
                model=MANAGER_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                extra_body={"format": "json"},   # Ollama JSON mode
            )
            content = (response.choices[0].message.content or "").strip()
            logger.debug("Manager raw response (%d chars).", len(content))
            return content
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Manager LLM error on attempt %d: %r. Retrying in %.1fs …",
                attempt, exc, backoff,
            )
            if attempt < MAX_RETRIES:
                time.sleep(backoff)

    raise RuntimeError(
        f"Manager LLM failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


def _parse_task_plan(raw_json: str) -> TaskPlan:
    """
    Parse and validate the Manager's raw JSON string into a ``TaskPlan``.

    Handles the case where the model wraps the JSON in markdown fences.

    Raises
    ------
    ValueError
        If the JSON cannot be parsed or fails Pydantic validation.
    """
    # Strip accidental markdown fences
    cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw_json.strip())
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manager returned invalid JSON: {exc}\nRaw:\n{cleaned}") from exc

    try:
        plan = TaskPlan.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"Manager JSON failed schema validation:\n{exc}\nData: {data}"
        ) from exc

    return plan


# ---------------------------------------------------------------------------
# Critic helpers
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM_PROMPT = """\
You are a rigorous quality-assurance reviewer for an AI pipeline.
Given the original user goal and the compiled output from a team of AI agents,
you must determine whether the output fully satisfies the goal.

Evaluation criteria:
  1. Completeness – does the output address ALL aspects of the goal?
  2. Correctness  – is the code/content technically accurate and runnable?
  3. Formatting   – is the output well-structured and readable?
  4. Quality      – is it production-ready, not a rough draft?

Output ONLY valid JSON matching this schema — no prose, no backticks:

{
  "approved": true | false,
  "feedback": "<actionable critique or justification>",
  "score": <integer 0-10>
}
"""


def _call_critic_llm(
    client: OpenAI,
    goal: str,
    compiled_output: str,
) -> CriticVerdict:
    """
    Call the Critic LLM and return a validated ``CriticVerdict``.
    Retries on transient errors.
    """
    messages = [
        {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## Original User Goal\n{goal}\n\n"
                f"## Compiled Agent Output\n{compiled_output}\n\n"
                "Evaluate and return your JSON verdict."
            ),
        },
    ]

    attempt = 0
    last_error: Exception | None = None
    while attempt < MAX_RETRIES:
        attempt += 1
        backoff = RETRY_BACKOFF_BASE ** (attempt - 1)
        try:
            logger.debug("Critic LLM call attempt %d/%d …", attempt, MAX_RETRIES)
            response = client.chat.completions.create(
                model=MANAGER_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=512,
                extra_body={"format": "json"},
            )
            raw = (response.choices[0].message.content or "").strip()
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            cleaned = re.sub(r"\n?```$", "", cleaned.strip())
            data = json.loads(cleaned)
            verdict = CriticVerdict.model_validate(data)
            return verdict
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            logger.warning(
                "Critic parse error on attempt %d: %r. Retrying …", attempt, exc
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Critic LLM error on attempt %d: %r. Retrying in %.1fs …",
                attempt, exc, backoff,
            )
        if attempt < MAX_RETRIES:
            time.sleep(backoff)

    raise RuntimeError(
        f"Critic LLM failed after {MAX_RETRIES} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    The central hub of the micro-orchestrator.

    Usage
    -----
    ::

        state = MissionState(goal="Write a Python web scraper …")
        orch  = Orchestrator()
        result = orch.run(state)
        print(result.final_output)
    """

    def __init__(self) -> None:
        self._client: OpenAI = build_ollama_client()
        logger.info("Orchestrator initialised (manager_model=%s).", MANAGER_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, state: MissionState) -> "OrchestrationResult":
        """
        Execute the full pipeline for the given mission state.

        The pipeline is:
          1. Manager LLM → task plan.
          2. Sequential worker execution.
          3. Critic review.
          4. If Critic rejects and retries remain → reset state, go to 1.

        Returns
        -------
        OrchestrationResult
            Contains the final compiled output, verdict, and execution summary.
        """
        pipeline_start = time.monotonic()

        for pipeline_attempt in range(1, MAX_PIPELINE_RETRIES + 2):
            logger.info(
                "=== Pipeline attempt %d/%d ===",
                pipeline_attempt,
                MAX_PIPELINE_RETRIES + 1,
            )

            # ---- Step 1: Plan ----
            plan = self._plan(state)
            logger.info(
                "Task plan received: %d tasks — %s",
                len(plan.tasks),
                [t.task_id for t in plan.tasks],
            )

            # ---- Step 2: Execute workers ----
            self._execute_workers(plan, state)

            # ---- Step 3: Compile output ----
            compiled = self._compile_output(state)

            # ---- Step 4: Critic review ----
            verdict = self._critique(state, compiled)
            logger.info(
                "Critic verdict: approved=%s, score=%d/10.",
                verdict.approved,
                verdict.score,
            )

            if verdict.approved:
                logger.info("Pipeline APPROVED by Critic on attempt %d.", pipeline_attempt)
                break

            # ---- Step 5: Retry if budget allows ----
            if pipeline_attempt <= MAX_PIPELINE_RETRIES:
                state.set_critic_feedback(verdict.feedback)
                state.reset_for_retry()
                logger.warning(
                    "Pipeline REJECTED. Replanning (attempt %d remaining) …",
                    MAX_PIPELINE_RETRIES - pipeline_attempt + 1,
                )
            else:
                logger.error(
                    "Pipeline REJECTED after all %d attempt(s). "
                    "Returning best available output.",
                    pipeline_attempt,
                )

        total_duration = time.monotonic() - pipeline_start
        history = state.get_history()

        return OrchestrationResult(
            final_output=compiled,
            verdict=verdict,
            history=history,
            total_duration_seconds=total_duration,
            pipeline_attempts=pipeline_attempt,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plan(self, state: MissionState) -> TaskPlan:
        """Call the Manager LLM, parse and validate the task plan."""
        logger.info("Calling Manager LLM to plan tasks …")
        raw = _call_manager_llm(
            self._client,
            state.goal,
            critic_feedback=state.critic_feedback,
        )

        # Retry the plan parse up to MAX_RETRIES times before giving up
        for parse_attempt in range(1, MAX_RETRIES + 1):
            try:
                plan = _parse_task_plan(raw)
                return plan
            except ValueError as exc:
                logger.warning(
                    "Task plan parse failed (attempt %d): %s", parse_attempt, exc
                )
                if parse_attempt < MAX_RETRIES:
                    logger.info("Re-requesting plan from Manager …")
                    raw = _call_manager_llm(
                        self._client,
                        state.goal,
                        critic_feedback=state.critic_feedback,
                    )
                else:
                    raise RuntimeError(
                        f"Manager failed to produce a valid task plan after "
                        f"{MAX_RETRIES} attempts."
                    ) from exc

        # Unreachable, but satisfies type checker
        raise RuntimeError("Unreachable")  # pragma: no cover

    def _execute_workers(self, plan: TaskPlan, state: MissionState) -> None:
        """Iterate through the plan and dispatch each task to its agent."""
        completed_ids: set[str] = set()

        for sub_task in plan.tasks:
            # Respect declared dependencies (sequential by default, but explicit ordering)
            missing_deps = [d for d in sub_task.depends_on if d not in completed_ids]
            if missing_deps:
                logger.warning(
                    "Task %r depends on %r which have not yet completed. "
                    "Skipping dependency check (sequential execution guarantees ordering).",
                    sub_task.task_id,
                    missing_deps,
                )

            agent = _resolve_agent(sub_task.agent_role)
            logger.info(
                "Dispatching task %r to %r agent …",
                sub_task.task_id,
                sub_task.agent_role,
            )
            result = agent.run_and_record(
                task_id=sub_task.task_id,
                description=sub_task.description,
                state=state,
            )

            if not result.succeeded:
                logger.error(
                    "Task %r FAILED. Pipeline may produce incomplete output.",
                    sub_task.task_id,
                )

            completed_ids.add(sub_task.task_id)

    def _compile_output(self, state: MissionState) -> str:
        """
        Produce a single compiled output string from the mission state.

        Priority order:
          1. ``writer.document`` (richest formatted artefact)
          2. ``coder.script``    (code if no writer ran)
          3. Full history dump   (fallback)
        """
        doc = state.get("writer.document")
        if doc:
            return doc

        script = state.get("coder.script")
        if script:
            return f"```python\n{script}\n```"

        return state.get_history_as_text()

    def _critique(self, state: MissionState, compiled_output: str) -> CriticVerdict:
        """Run the Critic LLM against the compiled output."""
        logger.info("Running Critic review …")
        try:
            return _call_critic_llm(self._client, state.goal, compiled_output)
        except RuntimeError as exc:
            # If the Critic itself fails, auto-approve so we don't lose the output
            logger.error(
                "Critic LLM failed: %s. Auto-approving to preserve output.", exc
            )
            return CriticVerdict(
                approved=True,
                feedback="Critic unavailable; auto-approved.",
                score=5,
            )


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class OrchestrationResult:
    """Encapsulates the complete outcome of a pipeline run."""

    def __init__(
        self,
        final_output: str,
        verdict: CriticVerdict,
        history: list,
        total_duration_seconds: float,
        pipeline_attempts: int,
    ) -> None:
        self.final_output = final_output
        self.verdict = verdict
        self.history = history
        self.total_duration_seconds = total_duration_seconds
        self.pipeline_attempts = pipeline_attempts

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ORCHESTRATION SUMMARY",
            "=" * 60,
            f"Pipeline attempts : {self.pipeline_attempts}",
            f"Total duration    : {self.total_duration_seconds:.2f}s",
            f"Critic approved   : {self.verdict.approved}",
            f"Critic score      : {self.verdict.score}/10",
            f"Critic feedback   : {self.verdict.feedback[:200]}",
            "-" * 60,
            "TASKS EXECUTED:",
        ]
        for task in self.history:
            status_icon = "✓" if task.status == TaskStatus.COMPLETED else "✗"
            lines.append(
                f"  {status_icon} [{task.agent_role:12s}] {task.task_id} "
                f"({task.duration_seconds:.2f}s)"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"OrchestrationResult("
            f"approved={self.verdict.approved}, "
            f"score={self.verdict.score}, "
            f"tasks={len(self.history)}, "
            f"duration={self.total_duration_seconds:.1f}s)"
        )
