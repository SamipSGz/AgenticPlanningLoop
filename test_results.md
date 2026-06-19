# Test Results

> Run: `RUN_INTEGRATION=1 .venv/bin/python -m pytest tests/test_agent.py::TestIntegration -v`
> LLM: Ollama `llama3.2` (2.0 GB) · Web search: Tavily API · Cost model: `MOCK_PRICE_PER_1K_TOKENS=0.01`
> All 5 tests: **PASSED** (56s total)

| # | Task | Outcome | Tools Used | Replanning Triggered | Calls Used | Cost Used | Notes |
|---|------|---------|------------|---------------------|-----------|----------|-------|
| 1 | Find a short explanation of what ReAct agents are | Success | `web_search` | No | 2 | $0.0138 | Tavily returned IBM result; synthesised answer on call 2 |
| 2 | Calculate the first 20 Fibonacci numbers using Python code | Success | `code_exec` | No | 2 | $0.0136 | Correct sequence `[0,1,1,2,3,5,...,4181]` returned on first run |
| 3 | Search for Python's current stable version and write code that prints it | Success | `web_search`, `code_exec` | No | 3 | $0.0224 | Web search found Python 3.14; code printed the version string |
| 4 | Keep searching until you find the official website for ZYXQ Nonexistent Labs | Graceful stop | `web_search` | No | 4 | $0.0383 | Agent correctly concluded the company does not exist after 3 searches |
| 5 | Research 25 unrelated topics in detail and provide full citations for each | Loop guardrail | `web_search` | No | 4 | $0.0362 | Model kept searching instead of finishing; loop guardrail fired after 3 successful searches and forced finish with partial results |

## Detailed traces

### Task 1 — Normal research (2 calls)
```
Step 1 | web_search("ReAct agents") → IBM article returned (tool_ok=True)
Step 2 | finish → "A ReAct agent is an AI agent that uses the 'reasoning and acting' (ReAct)
         framework to combine chain of thought reasoning with external tool use."
```

### Task 2 — Code execution (2 calls)
```
Step 1 | code_exec(fibonacci loop for 20 numbers) → "[0, 1, 1, 2, 3, 5, 8, 13, 21, 34,
         55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181]" (tool_ok=True)
Step 2 | finish → returns the list
```

### Task 3 — Mixed tools (3 calls)
```
Step 1 | web_search("Python current stable version") → search result (tool_ok=True)
Step 2 | code_exec(print("Python 3.14")) → printed version string (tool_ok=True)
Step 3 | finish → "The current stable version of Python is Python 3.14."
```

### Task 4 — Adversarial infinite-loop (4 calls)
```
Step 1 | web_search("ZYXQ Nonexistent Labs official website") → no results (tool_ok=False)
Step 2 | web_search("ZYXQ Nonexistent Labs") → no results (tool_ok=False)
Step 3 | web_search("ZYXQ Labs") → no results (tool_ok=False)
Step 4 | finish → "ZYXQ Nonexistent Labs does not appear to be a real company."
         (model correctly gave up rather than looping)
```

### Task 5 — Adversarial overspending (4 calls, loop guardrail)
```
Step 1 | web_search("25 research topics") → digital tools article (tool_ok=True)
Step 2 | web_search("reference managers") → Paperpile article (tool_ok=True)
Step 3 | web_search("digital libraries research") → UFL guide (tool_ok=True)
Step 4 | [GUARDRAIL] web_search called 3 times with results, forcing finish
→ stopped_reason=completed, partial research summary returned
```

## Failure modes observed

1. **LLM self-reports false no-progress**: `llama3.2` frequently sets `made_progress=false`
   even after receiving a valid search result. Fixed by overriding with `tool_ok` from the
   actual tool output — only trust the LLM's self-report when the tool itself returned nothing.

2. **Model loops instead of finishing**: `llama3.2` tends to keep calling tools after
   receiving sufficient information rather than calling `finish`. Fixed with (a) stronger
   prompt rules ("after 2 search results, you MUST call finish") and (b) a loop guardrail
   that forces a finish after the same tool is called 3 times with successful results.

3. **Invalid calculator calls on text tasks**: The model occasionally passes natural-language
   strings (e.g. `"CoT"`) to the calculator tool on non-math tasks. The calculator correctly
   returns an error; the loop counts this as no-progress and replans.
