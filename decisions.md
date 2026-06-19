# Engineering Decisions

- "I considered a multi-agent architecture with a separate planner and executor but chose a single ReAct loop because the assignment rewards clear, auditable control flow and budget enforcement over agent sophistication — a simpler loop is easier to instrument and prove correct."

- "I considered relying on provider-reported token costs (e.g., Anthropic's usage object) but chose a simulated per-token price model (`MOCK_PRICE_PER_1K_TOKENS`) that applies to every backend — including free local Ollama models — because the assignment explicitly requires the monetary budget enforcer to be visibly working regardless of which LLM is used."

- "I considered retrying failed tool calls automatically with exponential back-off but chose explicit reflection and replanning instead, because blind retries can create the exact infinite-loop behaviour the assignment tests against; detecting zero progress and changing strategy is more robust and directly satisfies the replanning requirement."

- "I considered using `eval()` for the calculator tool but chose a custom AST-walking evaluator because `eval()` would allow arbitrary code execution from LLM-generated input, which is a security risk the assignment's no-bare-`except`-pass constraint is also signalling care about."

- "I considered a fixed hard-coded budget of 10 calls / $0.20 but chose environment-variable configuration (`MAX_LLM_CALLS`, `MAX_COST_USD`) because the assignment says 'do not hardcode', and configurable limits make the same agent usable across different test scenarios without code changes."
