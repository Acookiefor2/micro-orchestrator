"""
agents/specialists.py
---------------------
Concrete specialist agents for the micro-orchestrator framework.

Each agent has:
  - A tightly scoped ``system_prompt`` that focuses the model on one role.
  - An ``execute()`` implementation that builds a rich context-aware prompt
    using prior task history from ``MissionState`` before calling the LLM.
  - Post-processing that writes relevant artefacts back to shared state so
    downstream agents can reference them.

Agents
------
CoderAgent      – Writes Python code; stores script in state["coder.script"].
ResearcherAgent – Synthesises information and research notes.
WriterAgent     – Produces formatted documents, reports, and markdown content.
"""

from __future__ import annotations

import logging
import re

from state import MissionState
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CoderAgent
# ---------------------------------------------------------------------------

class CoderAgent(BaseAgent):
    """
    Writes clean, runnable Python code for a given specification.

    Outputs the script verbatim (no surrounding prose).  After execution the
    extracted code is also written to ``state["coder.script"]`` so downstream
    agents (e.g., a WriterAgent producing a README) can embed it directly.
    """

    ROLE = "coder"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert Python software engineer. "
            "Your sole job is to write clean, well-commented, production-ready Python code. "
            "Always include all necessary imports at the top of the file. "
            "Add a short docstring at the module level explaining what the script does. "
            "Use clear variable names and follow PEP 8. "
            "Output ONLY the Python code, starting with the module docstring or imports. "
            "Do NOT include any explanation, markdown fences, or prose outside the code. "
            "If you need to show example output, put it inside a Python comment block at the end."
        )

    def execute(self, task_id: str, description: str, state: MissionState) -> str:
        prior_context = state.get_history_as_text()
        research_notes = state.get("researcher.notes", "")

        user_content_parts = [
            f"# Mission Goal\n{state.goal}\n",
            f"# Your Sub-Task\n{description}\n",
        ]

        if research_notes:
            user_content_parts.append(
                f"# Research Notes (from ResearcherAgent)\n{research_notes}\n"
            )

        if prior_context and prior_context != "(no prior tasks completed)":
            user_content_parts.append(
                f"# Prior Completed Tasks (for context)\n{prior_context}\n"
            )

        user_content_parts.append(
            "Write the complete Python script now. "
            "Output raw Python only – no markdown, no backticks, no commentary outside comments."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]

        raw_output = self._call_llm(messages, temperature=0.15, max_tokens=2048)

        # Strip accidental markdown fences that some models emit regardless
        cleaned = _strip_code_fences(raw_output)

        # Persist the extracted script for downstream agents
        state.set("coder.script", cleaned)
        logger.debug("[coder] Script stored in state (%d chars).", len(cleaned))

        return cleaned


# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------

class ResearcherAgent(BaseAgent):
    """
    Synthesises research notes, background knowledge, and structured guidance
    for a given topic.

    Writes bullet-pointed, structured notes to ``state["researcher.notes"]``
    so the CoderAgent (or WriterAgent) can incorporate them.
    """

    ROLE = "researcher"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert technical researcher and analyst. "
            "Your job is to gather, synthesise, and structure relevant information "
            "on any topic you are given. "
            "Produce clear, concise, structured notes in Markdown bullet format. "
            "Organise your output into logical sections with level-2 headings (##). "
            "Focus on accuracy, completeness, and practical applicability. "
            "If the topic involves code, include key library names, API patterns, "
            "and best-practice recommendations a developer would need. "
            "Do NOT generate code yourself – that is the Coder's job."
        )

    def execute(self, task_id: str, description: str, state: MissionState) -> str:
        prior_context = state.get_history_as_text()

        user_content_parts = [
            f"# Mission Goal\n{state.goal}\n",
            f"# Your Research Sub-Task\n{description}\n",
        ]

        critic_feedback = state.critic_feedback
        if critic_feedback:
            user_content_parts.append(
                f"# Critic Feedback (from a previous failed attempt)\n"
                f"{critic_feedback}\n"
                f"Take this into account to make your research more thorough.\n"
            )

        if prior_context and prior_context != "(no prior tasks completed)":
            user_content_parts.append(
                f"# Prior Completed Tasks (for context)\n{prior_context}\n"
            )

        user_content_parts.append(
            "Produce comprehensive, structured research notes in Markdown now."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]

        output = self._call_llm(messages, temperature=0.3, max_tokens=2048)

        state.set("researcher.notes", output)
        logger.debug("[researcher] Notes stored in state (%d chars).", len(output))

        return output


# ---------------------------------------------------------------------------
# WriterAgent
# ---------------------------------------------------------------------------

class WriterAgent(BaseAgent):
    """
    Produces polished written output: Markdown reports, README files,
    summaries, formatted tables, and explanatory documents.

    Automatically incorporates the latest code from ``state["coder.script"]``
    and research notes from ``state["researcher.notes"]`` when available.
    The final document is also written to ``state["writer.document"]``.
    """

    ROLE = "writer"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert technical writer. "
            "Your job is to produce clear, well-structured, publication-ready documents "
            "in Markdown format. "
            "Use appropriate headings, bullet lists, tables, and code blocks. "
            "Maintain a professional yet approachable tone. "
            "When embedding code, always wrap it in a fenced code block with the correct "
            "language identifier (```python, ```bash, etc.). "
            "Structure the document logically: introduction → details → examples → conclusion. "
            "Do NOT add unnecessary filler phrases. Be precise and informative."
        )

    def execute(self, task_id: str, description: str, state: MissionState) -> str:
        prior_context = state.get_history_as_text()
        coder_script = state.get("coder.script", "")
        research_notes = state.get("researcher.notes", "")

        user_content_parts = [
            f"# Mission Goal\n{state.goal}\n",
            f"# Your Writing Sub-Task\n{description}\n",
        ]

        if research_notes:
            user_content_parts.append(
                f"# Research Notes (incorporate as background)\n{research_notes}\n"
            )

        if coder_script:
            user_content_parts.append(
                f"# Python Script (produced by CoderAgent — embed in document)\n"
                f"```python\n{coder_script}\n```\n"
            )

        if prior_context and prior_context != "(no prior tasks completed)":
            user_content_parts.append(
                f"# Prior Completed Tasks (for reference)\n{prior_context}\n"
            )

        critic_feedback = state.critic_feedback
        if critic_feedback:
            user_content_parts.append(
                f"# Critic Feedback (address these issues in your document)\n"
                f"{critic_feedback}\n"
            )

        user_content_parts.append(
            "Produce the complete Markdown document now. Start with a level-1 heading."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]

        output = self._call_llm(messages, temperature=0.4, max_tokens=3000)

        state.set("writer.document", output)
        logger.debug("[writer] Document stored in state (%d chars).", len(output))

        return output


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    """
    Remove leading/trailing markdown code fences that models sometimes emit
    despite being instructed not to.

    Handles patterns like:
        ```python
        ... code ...
        ```
    or:
        ```
        ... code ...
        ```
    """
    # Match an optional language identifier after the opening fence
    pattern = re.compile(
        r"^```[a-zA-Z]*\n(.*?)```\s*$",
        re.DOTALL,
    )
    match = pattern.match(text.strip())
    if match:
        return match.group(1).rstrip()
    return text
