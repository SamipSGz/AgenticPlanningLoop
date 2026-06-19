# Resource-Constrained Agentic Planning Loop

A ReAct agent that operates under hard resource limits: **10 LLM calls** and **$0.20** per task. Hitting either limit stops execution immediately and returns a partial-completion report.

**Live demo**: https://agenticplanningloop-production.up.railway.app

---

## Quick Start

### Docker (single command)

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER and API keys

docker build -t constrained-agent .
docker run --env-file .env constrained-agent "Find a short explanation of what ReAct agents are."

# JSON output
docker run --env-file .env constrained-agent --json "Calculate the first 20 Fibonacci numbers."
```

### Local (uv)

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env
export $(grep -v '^#' .env | xargs)
constrained-agent "Find a short explanation of what ReAct agents are."
```

### Web UI

```bash
export $(grep -v '^#' .env | xargs)
uv run uvicorn api:app --reload
# open http://localhost:8000
```

### Tests

```bash
pytest tests/ -m "not integration" -v
RUN_INTEGRATION=1 pytest tests/ -m integration -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama`, `anthropic`, or `groq` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name for Ollama |
| `ANTHROPIC_API_KEY` | - | Required when provider is anthropic |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model ID |
| `GROQ_API_KEY` | - | Required when provider is groq |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model ID |
| `TAVILY_API_KEY` | - | If unset, DuckDuckGo is used (free) |
| `MAX_LLM_CALLS` | `10` | Hard call-count budget |
| `MAX_COST_USD` | `0.20` | Hard monetary budget in USD |
| `MOCK_PRICE_PER_1K_TOKENS` | `0.01` | Simulated token cost for free models |
| `TOOL_TIMEOUT_SECONDS` | `8` | Timeout for web search and calculator |
| `CODE_EXEC_TIMEOUT_SECONDS` | `10` | Timeout for code execution |
| `LLM_TIMEOUT_SECONDS` | `120` | Timeout for LLM HTTP requests |
| `LOG_LEVEL` | `INFO` | Set to DEBUG for full traces |
| `AGENT_TASK` | - | Alternative to CLI positional argument |

---

## Architecture Overview

The system is a single-process Python application built around a stateless ReAct loop. A `BudgetManager` holds call and cost counters; every LLM invocation passes through `assert_can_call()` before the request and `record_call()` after. Three tools (web search, code execution, calculator) each run in subprocesses or threads with explicit timeouts and return a typed `ToolResult`. The loop collects `StepRecord` objects in an `AgentState` Pydantic model and either returns on `finish` or produces a partial-completion report on budget exhaustion or stuck detection. A FastAPI server exposes the agent over Server-Sent Events for the live UI.

---

## Planning Loop

ReAct (Reason + Act) was chosen because the Thought → Action → Observation → Reflect cycle maps directly onto the assignment requirements: it makes reasoning visible at every step, naturally surfaces the "Am I making progress?" reflection point, and keeps control flow simple enough to insert hard budget checks between each phase. The biggest weakness is prompt sensitivity: if the model does not strictly follow the JSON output format, the parser must recover. This is handled here with a three-stage fallback (direct parse, newline sanitizer, regex field extractor), but a structured-output API would be more reliable and is the production-grade alternative.

---

## Schema Design

All data is typed with Pydantic v2 models:

- `AgentAction` - parsed LLM output: `thought`, `made_progress`, `replan_reason`, `action`, `action_input`
- `ToolResult` - every tool returns: `success`, `output`, `error`, `timed_out`
- `StepRecord` - one record per executed step, appended to `AgentState.steps`
- `AgentState` - full run state: task, steps, final answer, stopped reason

The LLM sees a flat JSON dictionary and the loop deserialises it into `AgentAction` immediately. Tool observations are truncated to 400 characters before embedding in the next prompt to keep token usage bounded.

---

## Prompt Strategy

The system prompt is rebuilt at every LLM call so budget remaining is always current. It includes:

1. **Tool catalogue** with exact input schemas - reduces hallucinated tool names
2. **Budget remaining** (calls and cost) - makes the agent plan conservatively near the limit
3. **JSON-only output rule** - no prose outside the JSON object
4. **No-repeat rule** - the agent must not repeat an action and input pair that already returned no result, which directly addresses the infinite-loop adversarial case
5. **Progress and replan fields** - `made_progress` and `replan_reason` are required fields in every response, forcing the model to evaluate progress on each step

---

## Failure Modes

**Observed during testing**: Groq's llama-3.3-70b model embeds literal newlines and unescaped double quotes inside JSON string values when generating Python code. This breaks `json.loads` outright. The fix is a three-level parser: first `json.loads`, then a character-walking sanitizer that escapes literal newlines inside strings, then a regex extractor that reads each field individually handling escaped characters manually. A real production fix would use the model's function-calling API which handles escaping at the API layer.

**Also observed**: The model sometimes adds extra keys (like `answer` or `input`) alongside `code` in `action_input`. Fixed by inspecting the tool function signature and dropping unknown kwargs before calling.

---

## Future Work

The agent has no memory across tasks. The most useful addition would be a query-hash cache (SQLite) for web search results, so repeated searches within a budget-constrained run return instantly without spending a call. This would also make the adversarial overspending test harder to trigger since cached results cost nothing.
