"""
main.py
-------
CLI entry point for the micro-orchestrator framework.

Usage
-----
Run the built-in demo goal (stock-price scraper):

    python main.py

Run a custom goal:

    python main.py --goal "Write a Python script that downloads NASA APOD images."

Save the final output to a file:

    python main.py --output result.md

Verbose logging:

    python main.py --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging configuration
# Must be done BEFORE importing project modules so all loggers pick it up.
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    date_fmt = "%H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=date_fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Quieten the very chatty openai / httpx transport loggers unless verbose
    if not verbose:
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Project imports (after logging is configured)
# ---------------------------------------------------------------------------

from state import MissionState          # noqa: E402
from orchestrator import Orchestrator   # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo goal
# ---------------------------------------------------------------------------

DEMO_GOAL: str = textwrap.dedent("""\
    Write a Python script that scrapes a dummy webpage for stock prices
    and formats the results into a Markdown table.

    Requirements:
    - Use the `requests` library to fetch the page and `BeautifulSoup` (bs4)
      to parse the HTML.
    - Target this public dummy/test URL for scraping practice:
      https://quotes.toscrape.com  (use it as a stand-in; adapt the selectors
      to extract whatever structured data is present and label it as "stock data"
      for demonstration purposes).
    - Parse at least 5 rows of data (quote text, author, tags).
    - Format the extracted data as a Markdown table with aligned columns.
    - Print the Markdown table to stdout.
    - The script must be fully self-contained (no custom modules required beyond
      requests and bs4).
    - Include a short README section in the final document explaining how to
      install dependencies and run the script.
""")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner(goal: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print("  micro-orchestrator  ·  Hub-and-Spoke Multi-Agent Pipeline")
    print("═" * width)
    print("\n📋  MISSION GOAL:\n")
    for line in goal.strip().splitlines():
        print(f"   {line}")
    print("\n" + "─" * width + "\n")


def _print_final_output(output: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print("  FINAL OUTPUT")
    print("═" * width + "\n")
    print(output)
    print("\n" + "═" * width)


def _save_output(output: str, path: Path) -> None:
    path.write_text(output, encoding="utf-8")
    logger.info("Final output saved to %s", path)
    print(f"\n💾  Output saved to: {path.resolve()}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="micro-orchestrator",
        description="Local-first multi-agent orchestration framework (Ollama backend).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --goal "Build a fibonacci generator in Python."
              python main.py --goal "Explain gradient descent" --output notes.md
              python main.py --verbose
        """),
    )
    parser.add_argument(
        "--goal",
        type=str,
        default=None,
        help="Custom mission goal. Defaults to the built-in stock scraper demo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the final output to FILE (in addition to printing it).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner (useful when piping output).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    _configure_logging(verbose=args.verbose)

    goal = args.goal or DEMO_GOAL

    if not args.no_banner:
        _print_banner(goal)

    # ---- Initialise state and orchestrator ----
    state = MissionState(goal=goal)
    orchestrator = Orchestrator()

    # ---- Run the pipeline ----
    logger.info("Starting orchestration pipeline …")
    wall_start = time.monotonic()

    try:
        result = orchestrator.run(state)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal orchestration error: %s", exc)
        print(f"\n❌  Fatal error: {exc}", file=sys.stderr)
        return 1

    wall_elapsed = time.monotonic() - wall_start

    # ---- Print summary ----
    print("\n" + result.summary())
    print(f"\n⏱   Wall-clock time: {wall_elapsed:.2f}s")

    # ---- Print / save final output ----
    _print_final_output(result.final_output)

    if args.output:
        _save_output(result.final_output, args.output)

    # Exit with non-zero code if Critic rejected (useful for CI pipelines)
    if not result.verdict.approved:
        logger.warning("Critic did not approve the final output (score=%d/10).", result.verdict.score)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
