from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ToolResult(BaseModel):
    success: bool
    output: str
    error: Optional[str] = None
    timed_out: bool = False


class AgentAction(BaseModel):
    thought: str
    made_progress: bool = True
    replan_reason: Optional[str] = None
    action: str
    action_input: Dict[str, Any] = {}


class StepRecord(BaseModel):
    step_num: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    made_progress: bool
    replan_reason: Optional[str] = None


class AgentState(BaseModel):
    task: str
    steps: List[StepRecord] = []
    final_answer: Optional[str] = None
    stopped_reason: Optional[str] = None
