from __future__ import annotations

from agent.budget import BudgetManager
from agent.tools import TOOL_REGISTRY


def build_system_prompt(budget: BudgetManager) -> str:
    tools_lines = "\n".join(
        f"  - {name}: {info['description']}"
        for name, info in TOOL_REGISTRY.items()
    )
    remaining_calls = budget.max_calls - budget.calls_used
    remaining_cost = budget.max_cost - budget.cost_used

    return f"""You are a resource-constrained ReAct agent. Reason step-by-step and call tools.

## Available Tools
{tools_lines}
  - finish: Deliver the final answer. Input: {{"answer": "<complete answer>"}}

## Budget Remaining
- LLM calls left : {remaining_calls} / {budget.max_calls}
- Cost remaining : ${remaining_cost:.4f} / ${budget.max_cost:.2f}

## Response Format
Respond with ONLY a valid JSON object — no prose, no markdown outside the JSON:

{{
  "thought": "one or two sentences of reasoning",
  "made_progress": true,
  "replan_reason": null,
  "action": "web_search" | "code_exec" | "calculator" | "finish",
  "action_input": {{...}}
}}

## Critical JSON Rules
- Your entire response MUST be one valid JSON object.
- String values MUST use \\n for newlines — never embed literal line breaks inside a JSON string.
- Use single quotes inside Python code strings to avoid escaping: 'hello' not "hello"
- Example of correct code_exec format:
  "action_input": {"code": "def merge_sort(arr):\\n    if len(arr)<=1: return arr\\n    m=len(arr)//2\\n    return sorted(merge_sort(arr[:m])+merge_sort(arr[m:]))\\nprint(merge_sort([3,1,2]))"}

## Rules
1. If a web_search or tool call returned useful text, YOU MUST use that information to answer — do NOT call another tool just to "verify".
2. After receiving ONE good search result, synthesize the answer immediately and call action="finish".
3. Never call calculator or code_exec unless the task explicitly requires a calculation or running code. code_exec runs standard Python only — do NOT import requests, numpy, pandas or any third-party package. Use urllib.request for HTTP or do pure computation.
4. Never repeat a tool call with the same or similar input as a previous step.
5. If the previous step returned no useful information, set "made_progress": false and explain a different plan in "replan_reason".
6. For finish: action_input must contain key "answer" with the complete response.
7. If you have seen 2 or more web_search results already, you MUST call action="finish" next — no more searching.
"""


def build_user_message(task: str, history: list) -> str:
    parts = [f"Task: {task}\n"]

    if not history:
        parts.append("No steps taken yet. Start reasoning and pick your first action.")
        return "\n".join(parts)

    parts.append("Steps taken so far:")
    for step in history:
        obs = step["observation"]
        if len(obs) > 800:
            obs = obs[:800] + "\n...[truncated]"
        parts += [
            f"\n--- Step {step['step_num']} ---",
            f"Thought      : {step['thought']}",
            f"Action       : {step['action']}",
            f"Action input : {step['action_input']}",
            f"Observation  : {obs}",
            f"Made progress: {step['made_progress']}",
        ]
        if step.get("replan_reason"):
            parts.append(f"Replan reason: {step['replan_reason']}")

    parts.append("\nWhat is your next thought and action?")
    return "\n".join(parts)
