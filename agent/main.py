from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from agent.budget import BudgetManager
from agent.loop import run_agent


def _setup_logging() -> None:
    level = getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def main() -> None:
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="constrained-agent",
        description="Resource-Constrained Agentic Planning Loop",
    )
    parser.add_argument("task", nargs="?", help="Task to execute (or set AGENT_TASK env var)")
    parser.add_argument("--json", dest="output_json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    task = args.task or os.environ.get("AGENT_TASK", "").strip()
    if not task:
        parser.error("Provide a task as a positional argument or set AGENT_TASK env var.")

    budget = BudgetManager()
    state = run_agent(task, budget)

    if args.output_json:
        print(json.dumps({
            "task": state.task,
            "stopped_reason": state.stopped_reason,
            "final_answer": state.final_answer,
            "budget": budget.status(),
            "steps": [s.model_dump() for s in state.steps],
        }, indent=2))
    else:
        sep = "=" * 64
        s = budget.status()
        print(f"\n{sep}")
        print(f"Task   : {state.task}")
        print(f"Status : {state.stopped_reason}")
        print(f"Budget : {s['calls_used']}/{s['calls_max']} calls, ${s['cost_used']:.5f}/${s['cost_max']:.2f}")
        print(sep)
        print(f"\n{state.final_answer}\n")


if __name__ == "__main__":
    main()
