# Resource-Constrained Agentic Planning Loop

A minimal ReAct agent that operates under hard resource limits: **10 LLM calls** and **$0.20** per task. Budget enforcement is real — hitting either limit stops execution immediately and returns a partial-completion report.

## Running the Project

### Option 1 — Docker (recommended, single command)

```bash
# 1. Copy and fill in your environment
cp .env.example .env
# Edit .env: set LLM_PROVIDER, OLLAMA_URL, and optionally TAVILY_API_KEY

# 2. Build
docker build -t constrained-agent .

# 3. Run a task
docker run --env-file .env constrained-agent "Find a short explanation of what ReAct agents are."

# JSON output
docker run --env-file .env constrained-agent --json "Calculate the first 20 Fibonacci numbers using Python code."
```

**Ollama note**: if Ollama runs on your host machine, use `OLLAMA_URL=http://host.docker.internal:11434` in your `.env`.

### Option 2 — Local Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # edit as needed
export $(grep -v '^#' .env | xargs)
constrained-agent "Find a short explanation of what ReAct agents are."
```

### Running Tests

```bash
# Unit tests only (no LLM required)
pytest tests/ -m "not integration" -v

# Integration tests (requires Ollama running locally)
RUN_INTEGRATION=1 pytest tests/ -m integration -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` or `anthropic` |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model to use with Ollama |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model ID |
| `TAVILY_API_KEY` | — | If unset, DuckDuckGo is used (free, no key) |
| `MAX_LLM_CALLS` | `10` | Hard call-count budget |
| `MAX_COST_USD` | `0.20` | Hard monetary budget in USD |
| `MOCK_PRICE_PER_1K_TOKENS` | `0.01` | Simulated cost applied to every LLM call |
| `TOOL_TIMEOUT_SECONDS` | `8` | Timeout for web search and calculator |
| `CODE_EXEC_TIMEOUT_SECONDS` | `10` | Timeout for code execution subprocess |
| `LLM_TIMEOUT_SECONDS` | `120` | Timeout for LLM HTTP request |
| `LOG_LEVEL` | `INFO` | `DEBUG` for full traces |
| `AGENT_TASK` | — | Alternative to CLI positional argument |

## Architecture Overview

The system is a single-process Python application structured as a thin wrapper around a stateless ReAct loop. A `BudgetManager` holds call and cost counters; every LLM invocation passes through it — `assert_can_call()` before the HTTP request, `record_call()` after. Three tools (web search, code execution, calculator) each run with explicit timeouts and return a typed `ToolResult`. The loop collects `StepRecord` objects in an `AgentState` Pydantic model and either returns normally on `finish` or triggers a partial-completion report on budget exhaustion or stuck detection.

## Planning Loop

ReAct (Reason + Act) was chosen because its Thought → Action → Observation → Reflect cycle maps directly onto the assignment's requirements: it makes the reasoning visible at every step, naturally surfaces the "Am I making progress?" reflection point, and keeps the control flow simple enough to add hard budget checks between each phase. Its biggest weakness is prompt sensitivity: if the model does not strictly follow the JSON schema, the parser must recover — handled here with a two-stage parse (full JSON, then brace-extraction fallback), but a more capable model or structured-output API would be more reliable.

## Schema Design

All data passed between the loop, LLM, and tools is typed with Pydantic v2 models:

- `AgentAction` — the parsed LLM response: `thought`, `made_progress`, `replan_reason`, `action`, `action_input`.
- `ToolResult` — every tool returns `success`, `output`, `error`, `timed_out`.
- `StepRecord` — one record per executed step, stored in `AgentState.steps`.
- `AgentState` — the complete run: task, steps, final answer, stopped reason.

The LLM sees a flat JSON dictionary; the loop deserialises it into `AgentAction` immediately. This means the LLM never receives raw Python objects and the loop never reads raw strings past the parser. Tool results are truncated to 800 characters before being embedded in the next prompt to keep token usage bounded.

## Prompt Strategy

The system prompt is rebuilt at the start of every LLM call so it always reflects the current budget remaining. It includes:

1. **Tool catalogue** with exact input schemas — reduces hallucinated tool names.
2. **Budget remaining** (calls and cost) — makes the agent plan conservatively near the limit.
3. **Strict JSON-only instruction** — no prose outside the JSON object; the parser enforces this.
4. **Explicit no-repeat rule** — the agent is told not to repeat an action+input pair that previously returned no useful result, directly targeting the infinite-loop adversarial case.
5. **Progress/replan fields** — `made_progress` and `replan_reason` are required fields in the schema, not optional add-ons, so the model is forced to evaluate progress on every step.

The user message includes the full step history (truncated observations) so the model has context for deciding whether it is making progress.

## Failure Modes

**Observed**: Ollama occasionally prefixes the JSON with a natural-language sentence (e.g., "Here is my response:"). The primary `json.loads()` call fails; the fallback scans for the first `{` … last `}` brace pair and re-parses. This recovered correctly in all test runs, but a model that interleaves prose mid-JSON would still break. A structured-output API (Ollama's `format: "json"` field is used, but not all models honour it perfectly) or a tool-calling API would eliminate this failure mode entirely.

## Future Work

The agent currently has no memory across tasks — each run starts from scratch. The most impactful addition would be a persistent scratchpad (e.g., a lightweight SQLite store) that caches web-search results by query hash. This would prevent re-fetching the same URL across tasks, reduce both call count and cost, and make the adversarial overspending test even harder to trigger.
