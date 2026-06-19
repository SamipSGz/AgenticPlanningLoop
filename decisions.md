# Engineering Decisions

"I considered a multi-agent setup but chose a single ReAct loop because it is simpler to instrument, easier to prove correct, and budget enforcement is straightforward in one place."

"I considered using real provider token costs but chose a simulated price model because free local models like Ollama have no real cost, and the assignment requires the monetary budget limit to be demonstrable regardless of which LLM is used."

"I considered automatic retries on failed tool calls but chose reflection and replanning instead, because blind retries cause the exact infinite loop the assignment tests against."

"I considered using eval() for the calculator but chose a custom AST evaluator because eval() allows arbitrary code execution from LLM output, which is a security risk."

"I considered hardcoding the 10 call and 0.20 dollar limits but chose environment variables because the assignment says do not hardcode, and configurable limits make testing easier."
