from __future__ import annotations

import logging
from typing import Callable, Optional

from agent.budget import BudgetExceededError, BudgetManager
from agent.llm import call_llm
from agent.prompts import build_system_prompt, build_user_message
from agent.schemas import AgentState, StepRecord
from agent.tools import execute_tool

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_NO_PROGRESS = 3
_EMPTY_OUTPUTS = frozenset({"", "No results found for this query.", "(no output)"})


def run_agent(
    task: str,
    budget: BudgetManager,
    on_event: Optional[Callable[[dict], None]] = None,
) -> AgentState:
    def emit(event: dict) -> None:
        if on_event:
            on_event(event)

    state = AgentState(task=task)
    consecutive_no_progress = 0

    logger.info("starting | task=%r | budget=%d calls/$%.2f", task, budget.max_calls, budget.max_cost)
    emit({"type": "start", "task": task, "budget": budget.status()})

    while True:
        system_prompt = build_system_prompt(budget)
        user_message = build_user_message(task, [s.model_dump() for s in state.steps])

        try:
            action, _, _ = call_llm(system_prompt=system_prompt, user_message=user_message, budget=budget)
        except BudgetExceededError as exc:
            logger.warning("budget exceeded: %s", exc.reason)
            state.stopped_reason = "budget_exceeded"
            state.final_answer = _partial_report(state, str(exc))
            emit({"type": "done", "stopped_reason": state.stopped_reason, "final_answer": state.final_answer, "budget": budget.status()})
            return state
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            state.stopped_reason = "llm_error"
            state.final_answer = f"Agent stopped due to LLM error: {exc}"
            emit({"type": "error", "message": state.final_answer, "budget": budget.status()})
            return state

        emit({"type": "budget", **budget.status()})

        step_num = len(state.steps) + 1
        logger.info("step %d | action=%s | thought=%s", step_num, action.action, action.thought[:100])

        if action.action == "finish":
            state.final_answer = action.action_input.get("answer", "(no answer provided)")
            state.stopped_reason = "completed"
            logger.info("task completed")
            emit({"type": "done", "stopped_reason": state.stopped_reason, "final_answer": state.final_answer, "budget": budget.status()})
            return state

        same_tool_successes = sum(
            1 for s in state.steps
            if s.action == action.action and s.made_progress
        )
        if same_tool_successes >= 6:
            last_good = next(
                (s.observation for s in reversed(state.steps) if s.made_progress), ""
            )
            logger.warning("loop guardrail: %s called %d times with results, forcing finish", action.action, same_tool_successes)
            state.final_answer = f"Based on research findings:\n\n{last_good[:1000]}"
            state.stopped_reason = "completed"
            emit({"type": "done", "stopped_reason": state.stopped_reason, "final_answer": state.final_answer, "budget": budget.status()})
            return state

        result = execute_tool(action.action, action.action_input)

        if result.timed_out:
            observation = f"[TIMEOUT] Tool '{action.action}' timed out."
            tool_ok = False
        elif not result.success:
            observation = f"[ERROR] {result.error}"
            tool_ok = False
        elif result.output.strip() in _EMPTY_OUTPUTS:
            observation = result.output or "(empty result)"
            tool_ok = False
        else:
            observation = result.output
            tool_ok = True

        made_progress = tool_ok or action.made_progress

        if not made_progress:
            consecutive_no_progress += 1
            logger.info("no-progress streak %d/%d | replan: %s", consecutive_no_progress, MAX_CONSECUTIVE_NO_PROGRESS, action.replan_reason)
        else:
            consecutive_no_progress = 0

        logger.info("observation (tool_ok=%s): %s", tool_ok, observation[:200])

        step = StepRecord(
            step_num=step_num,
            thought=action.thought,
            action=action.action,
            action_input=action.action_input,
            observation=observation,
            made_progress=made_progress,
            replan_reason=action.replan_reason if not made_progress else None,
        )
        state.steps.append(step)
        emit({"type": "step", **step.model_dump()})

        if consecutive_no_progress >= MAX_CONSECUTIVE_NO_PROGRESS:
            logger.warning("agent stuck after %d consecutive no-progress steps", MAX_CONSECUTIVE_NO_PROGRESS)
            state.stopped_reason = "stuck"
            state.final_answer = _partial_report(state, "Agent could not make progress after replanning.")
            emit({"type": "done", "stopped_reason": state.stopped_reason, "final_answer": state.final_answer, "budget": budget.status()})
            return state


def _partial_report(state: AgentState, stop_reason: str) -> str:
    lines = [f"Stopped: {stop_reason}", "", "Completed steps:"]
    for step in state.steps:
        obs_preview = step.observation[:120].replace("\n", " ")
        lines.append(f"  {step.step_num}. {step.action}({list(step.action_input.keys())}) -> {obs_preview}")
    if not state.steps:
        lines.append("  (none)")
    lines += ["", "Partial results:"]
    if state.steps:
        lines.append(f"  Last observation: {state.steps[-1].observation[:500]}")
    else:
        lines.append("  No observations collected.")
    return "\n".join(lines)
