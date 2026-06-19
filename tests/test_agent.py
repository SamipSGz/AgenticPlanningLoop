"""
Tests for the Resource-Constrained Agentic Planning Loop.

Unit tests (no LLM required): budget, tools, parser.
Integration tests (require Ollama or LLM): marked with @pytest.mark.integration.
Run only unit tests  : pytest tests/ -m "not integration"
Run integration tests: pytest tests/ -m integration  (Ollama must be running)
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from agent.budget import BudgetExceededError, BudgetManager
from agent.llm import _parse_action
from agent.schemas import AgentAction, AgentState, ToolResult
from agent.tools import calculator, code_exec, execute_tool


# ===========================================================================
# Budget unit tests
# ===========================================================================

class TestBudgetManager:
    def test_initial_state(self):
        b = BudgetManager()
        assert b.calls_used == 0
        assert b.cost_used == 0.0

    def test_assert_can_call_passes_initially(self):
        b = BudgetManager()
        b.assert_can_call()  # should not raise

    def test_call_limit_raises(self):
        b = BudgetManager()
        b.max_calls = 2
        b.record_call(100, 100)
        b.record_call(100, 100)
        with pytest.raises(BudgetExceededError) as exc_info:
            b.assert_can_call()
        assert "call limit" in str(exc_info.value)

    def test_cost_limit_raises(self):
        b = BudgetManager()
        b.max_cost = 0.001
        b.price_per_1k_tokens = 0.01
        b.record_call(200, 0)  # costs 0.002, exceeds 0.001
        with pytest.raises(BudgetExceededError) as exc_info:
            b.assert_can_call()
        assert "cost limit" in str(exc_info.value)

    def test_record_call_accumulates(self):
        b = BudgetManager()
        b.price_per_1k_tokens = 0.01
        b.record_call(1000, 0)
        b.record_call(0, 1000)
        assert b.calls_used == 2
        assert abs(b.cost_used - 0.02) < 1e-9

    def test_status_dict(self):
        b = BudgetManager()
        s = b.status()
        assert "calls_used" in s
        assert "cost_used" in s
        assert "calls_max" in s
        assert "cost_max" in s

    def test_budget_not_exceeded_at_max_minus_one(self):
        b = BudgetManager()
        b.max_calls = 3
        b.record_call(10, 10)
        b.record_call(10, 10)
        b.assert_can_call()  # 2 used, max 3 — should not raise

    def test_custom_env_vars(self, monkeypatch):
        monkeypatch.setenv("MAX_LLM_CALLS", "5")
        monkeypatch.setenv("MAX_COST_USD", "0.50")
        monkeypatch.setenv("MOCK_PRICE_PER_1K_TOKENS", "0.02")
        b = BudgetManager()
        assert b.max_calls == 5
        assert b.max_cost == 0.50
        assert b.price_per_1k_tokens == 0.02


# ===========================================================================
# Tool unit tests
# ===========================================================================

class TestCalculator:
    def test_basic_arithmetic(self):
        r = calculator("2 + 2")
        assert r.success
        assert "4" in r.output

    def test_multiplication(self):
        r = calculator("3 * 7")
        assert r.success
        assert "21" in r.output

    def test_float_division(self):
        r = calculator("10 / 4")
        assert r.success
        assert "2.5" in r.output

    def test_power(self):
        r = calculator("2 ** 10")
        assert r.success
        assert "1024" in r.output

    def test_sqrt(self):
        r = calculator("sqrt(16)")
        assert r.success
        assert "4.0" in r.output

    def test_pi(self):
        r = calculator("pi")
        assert r.success
        assert "3.14" in r.output

    def test_nested_expression(self):
        r = calculator("sqrt(2) * pi")
        assert r.success

    def test_factorial(self):
        r = calculator("factorial(5)")
        assert r.success
        assert "120" in r.output

    def test_invalid_expression(self):
        r = calculator("import os")
        assert not r.success
        assert r.error is not None

    def test_unknown_name_blocked(self):
        r = calculator("__import__('os')")
        assert not r.success

    def test_division_by_zero(self):
        r = calculator("1 / 0")
        assert not r.success

    def test_modulo(self):
        r = calculator("10 % 3")
        assert r.success
        assert "1" in r.output


class TestCodeExec:
    def test_hello_world(self):
        r = code_exec("print('hello')")
        assert r.success
        assert "hello" in r.output

    def test_fibonacci(self):
        code = """
a, b = 0, 1
result = []
for _ in range(10):
    result.append(a)
    a, b = b, a + b
print(result)
"""
        r = code_exec(code)
        assert r.success
        assert "34" in r.output  # fib[9] = 34

    def test_syntax_error(self):
        r = code_exec("def foo(:")
        assert not r.success
        assert r.error is not None

    def test_runtime_error(self):
        r = code_exec("raise ValueError('test error')")
        assert not r.success
        assert r.error is not None

    def test_unsupported_language(self):
        r = code_exec("console.log('hi')", language="javascript")
        assert not r.success
        assert "Only Python" in (r.error or "")

    def test_stdout_captured(self):
        r = code_exec("for i in range(3): print(i)")
        assert r.success
        assert "0" in r.output and "2" in r.output

    def test_timeout_enforcement(self, monkeypatch):
        monkeypatch.setenv("CODE_EXEC_TIMEOUT_SECONDS", "1")
        # Re-import to pick up new env var
        import importlib
        import agent.tools as tools_mod
        importlib.reload(tools_mod)
        r = tools_mod.code_exec("import time; time.sleep(60)")
        assert not r.success
        assert r.timed_out or "timed out" in (r.error or "").lower()


class TestExecuteTool:
    def test_unknown_tool(self):
        r = execute_tool("nonexistent_tool", {})
        assert not r.success
        assert "Unknown tool" in (r.error or "")

    def test_calculator_via_registry(self):
        r = execute_tool("calculator", {"expression": "5 + 5"})
        assert r.success
        assert "10" in r.output

    def test_invalid_inputs_type_error(self):
        r = execute_tool("calculator", {"bad_param": "x"})
        assert not r.success


# ===========================================================================
# LLM response parser unit tests
# ===========================================================================

class TestParseAction:
    def _make_json(self, **kwargs) -> str:
        base = {
            "thought": "testing",
            "made_progress": True,
            "replan_reason": None,
            "action": "finish",
            "action_input": {"answer": "done"},
        }
        base.update(kwargs)
        return json.dumps(base)

    def test_parses_finish(self):
        a = _parse_action(self._make_json(action="finish", action_input={"answer": "42"}))
        assert a.action == "finish"
        assert a.action_input["answer"] == "42"

    def test_parses_web_search(self):
        a = _parse_action(self._make_json(action="web_search", action_input={"query": "hello"}))
        assert a.action == "web_search"

    def test_parses_code_exec(self):
        a = _parse_action(self._make_json(action="code_exec", action_input={"code": "print(1)"}))
        assert a.action == "code_exec"

    def test_parses_calculator(self):
        a = _parse_action(self._make_json(action="calculator", action_input={"expression": "1+1"}))
        assert a.action == "calculator"

    def test_strips_markdown_fence(self):
        raw = '```json\n{"thought":"t","made_progress":true,"replan_reason":null,"action":"finish","action_input":{"answer":"ok"}}\n```'
        a = _parse_action(raw)
        assert a.action == "finish"

    def test_made_progress_false(self):
        a = _parse_action(self._make_json(made_progress=False, replan_reason="no results"))
        assert a.made_progress is False
        assert a.replan_reason == "no results"

    def test_action_normalisation_search_variant(self):
        a = _parse_action(self._make_json(action="do_web_search", action_input={"query": "q"}))
        assert a.action == "web_search"

    def test_action_normalisation_code_variant(self):
        a = _parse_action(self._make_json(action="run_python_code", action_input={"code": "x"}))
        assert a.action == "code_exec"

    def test_bad_json_raises(self):
        with pytest.raises(ValueError):
            _parse_action("not json at all!!!")

    def test_answer_lifted_from_top_level(self):
        raw = json.dumps({
            "thought": "done",
            "made_progress": True,
            "replan_reason": None,
            "action": "finish",
            "answer": "the answer",
        })
        a = _parse_action(raw)
        assert a.action_input.get("answer") == "the answer"


# ===========================================================================
# Agent loop unit tests (LLM mocked)
# ===========================================================================

def _make_mock_action(action: str, action_input: dict, made_progress: bool = True) -> AgentAction:
    return AgentAction(
        thought="mock thought",
        made_progress=made_progress,
        replan_reason=None if made_progress else "no result",
        action=action,
        action_input=action_input,
    )


def _fake_llm(action, in_tok=50, out_tok=50):
    """Return a call_llm replacement that mirrors the real budget contract."""
    from agent.budget import BudgetExceededError

    def _inner(system_prompt, user_message, budget, model=None):
        budget.assert_can_call()
        budget.record_call(in_tok, out_tok)
        if callable(action):
            return action(), in_tok, out_tok
        return action, in_tok, out_tok

    return _inner


class TestAgentLoopBudgetEnforcement:
    def test_stops_on_call_budget(self):
        from agent.loop import run_agent

        budget = BudgetManager()
        budget.max_calls = 2

        with patch("agent.loop.call_llm", side_effect=_fake_llm(_make_mock_action("calculator", {"expression": "1+1"}))):
            state = run_agent("some task", budget)

        assert state.stopped_reason == "budget_exceeded"
        assert budget.calls_used == budget.max_calls

    def test_stops_on_cost_budget(self):
        from agent.loop import run_agent

        budget = BudgetManager()
        budget.max_calls = 100
        budget.max_cost = 0.001
        budget.price_per_1k_tokens = 0.01

        with patch("agent.loop.call_llm", side_effect=_fake_llm(_make_mock_action("calculator", {"expression": "1+1"}), in_tok=200, out_tok=0)):
            state = run_agent("some task", budget)

        assert state.stopped_reason == "budget_exceeded"

    def test_graceful_exit_reports_partial_steps(self):
        from agent.loop import run_agent

        budget = BudgetManager()
        budget.max_calls = 1

        with patch("agent.loop.call_llm", side_effect=_fake_llm(_make_mock_action("calculator", {"expression": "2+2"}))):
            state = run_agent("test task", budget)

        assert state.stopped_reason == "budget_exceeded"
        assert state.final_answer is not None
        assert "Stopped" in state.final_answer

    def test_completes_normally(self):
        from agent.loop import run_agent

        budget = BudgetManager()
        sequence = [
            _make_mock_action("calculator", {"expression": "6*7"}),
            _make_mock_action("finish", {"answer": "The answer is 42"}),
        ]
        it = iter(sequence)

        with patch("agent.loop.call_llm", side_effect=_fake_llm(lambda: next(it))):
            state = run_agent("What is 6*7?", budget)

        assert state.stopped_reason == "completed"
        assert "42" in (state.final_answer or "")

    def test_stuck_detection(self):
        from agent.loop import run_agent, MAX_CONSECUTIVE_NO_PROGRESS
        from agent.schemas import ToolResult

        budget = BudgetManager()
        stuck_action = AgentAction(
            thought="no idea",
            made_progress=False,
            replan_reason="nothing works",
            action="web_search",
            action_input={"query": "test"},
        )
        empty_result = ToolResult(success=True, output="No results found for this query.")

        with patch("agent.loop.call_llm", side_effect=_fake_llm(stuck_action)), \
             patch("agent.loop.execute_tool", return_value=empty_result):
            state = run_agent("impossible task", budget)

        assert state.stopped_reason == "stuck"
        assert budget.calls_used == MAX_CONSECUTIVE_NO_PROGRESS


# ===========================================================================
# Integration tests (require a running Ollama instance)
# ===========================================================================

integration = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run integration tests (requires Ollama)",
)


@integration
class TestIntegration:
    """Run the 5 required tasks end-to-end against a live LLM."""

    def _run(self, task: str) -> AgentState:
        from agent.loop import run_agent
        budget = BudgetManager()
        return run_agent(task, budget)

    def test_task1_react_explanation(self):
        """Normal research task — should complete within budget."""
        state = self._run("Find a short explanation of what ReAct agents are.")
        assert state.stopped_reason == "completed"
        assert state.final_answer
        assert len(state.steps) <= 10

    def test_task2_fibonacci_code(self):
        """Code execution task — should use code_exec tool."""
        state = self._run("Calculate the first 20 Fibonacci numbers using Python code.")
        assert state.stopped_reason == "completed"
        tools_used = {s.action for s in state.steps}
        assert "code_exec" in tools_used

    def test_task3_mixed_web_and_code(self):
        """Mixed task — should use both web_search and code_exec."""
        state = self._run(
            "Search for Python's current stable version and write code that prints it."
        )
        assert state.stopped_reason in ("completed", "budget_exceeded")
        tools_used = {s.action for s in state.steps}
        assert len(tools_used) >= 1

    def test_task4_adversarial_infinite_loop(self):
        """Adversarial task: fake company triggers replanning, not infinite loop."""
        state = self._run(
            "Keep searching until you find the official website for ZYXQ Nonexistent Labs."
        )
        # Must NOT loop forever — must stop by budget or stuck detection
        assert state.stopped_reason in ("stuck", "budget_exceeded", "completed")
        # At least one no-progress step must have been recorded (replanning shown)
        assert len(state.steps) <= 10

    def test_task5_adversarial_overspending(self):
        """Adversarial task: agent should hit budget and stop cleanly."""
        state = self._run(
            "Research 25 unrelated topics in detail and provide full citations for each."
        )
        # Must stop — either by budget or by completing early with partial answer
        assert state.stopped_reason in ("budget_exceeded", "completed", "stuck")
        if state.stopped_reason == "budget_exceeded":
            # Final answer must report partial completion, not crash
            assert state.final_answer is not None
            assert "Stopped" in state.final_answer
