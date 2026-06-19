from __future__ import annotations

import json
import logging
import os
from typing import Optional, Tuple

import httpx

from agent.budget import BudgetExceededError, BudgetManager
from agent.schemas import AgentAction

logger = logging.getLogger(__name__)

_VALID_ACTIONS = frozenset({"web_search", "code_exec", "calculator", "finish"})


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def call_llm(
    system_prompt: str,
    user_message: str,
    budget: BudgetManager,
    model: Optional[str] = None,
) -> Tuple[AgentAction, int, int]:
    budget.assert_can_call()
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider == "anthropic":
        return _call_anthropic(system_prompt, user_message, budget, model)
    if provider == "groq":
        return _call_openai_compat(system_prompt, user_message, budget, model)
    return _call_ollama(system_prompt, user_message, budget, model)


def _call_ollama(
    system_prompt: str,
    user_message: str,
    budget: BudgetManager,
    model: Optional[str],
) -> Tuple[AgentAction, int, int]:
    ollama_model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))

    try:
        response = httpx.post(
            f"{ollama_url}/api/chat",
            json={
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.1},
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Ollama request timed out after {timeout}s") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama HTTP {exc.response.status_code}: {exc.response.text[:400]}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {ollama_url}: {exc}") from exc

    data = response.json()
    content: str = data["message"]["content"]
    input_tokens = data.get("prompt_eval_count") or _estimate_tokens(system_prompt + user_message)
    output_tokens = data.get("eval_count") or _estimate_tokens(content)

    cost = budget.record_call(input_tokens, output_tokens)
    logger.info("call #%d | %d/%d tokens | $%.5f | total $%.5f", budget.calls_used, input_tokens, output_tokens, cost, budget.cost_used)

    return _parse_action(content), input_tokens, output_tokens


def _call_anthropic(
    system_prompt: str,
    user_message: str,
    budget: BudgetManager,
    model: Optional[str],
) -> Tuple[AgentAction, int, int]:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic package not installed") from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)
    model_name = model or os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))

    try:
        resp = client.messages.create(
            model=model_name,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
            timeout=timeout,
        )
    except anthropic.APITimeoutError as exc:
        raise RuntimeError("Anthropic request timed out") from exc
    except anthropic.APIStatusError as exc:
        raise RuntimeError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

    content = resp.content[0].text
    input_tokens = resp.usage.input_tokens
    output_tokens = resp.usage.output_tokens

    cost = budget.record_call(input_tokens, output_tokens)
    logger.info("call #%d | %d/%d tokens | $%.5f | total $%.5f", budget.calls_used, input_tokens, output_tokens, cost, budget.cost_used)

    return _parse_action(content), input_tokens, output_tokens


def _call_openai_compat(
    system_prompt: str,
    user_message: str,
    budget: BudgetManager,
    model: Optional[str],
) -> Tuple[AgentAction, int, int]:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    base_urls = {"groq": "https://api.groq.com/openai/v1"}
    base_url = os.environ.get("OPENAI_BASE_URL") or base_urls.get(provider, "https://api.groq.com/openai/v1")

    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(f"{provider.upper()}_API_KEY is not set")

    default_models = {"groq": "llama-3.3-70b-versatile"}
    model_name = model or os.environ.get("GROQ_MODEL") or default_models.get(provider, "llama-3.3-70b-versatile")
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))

    try:
        response = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.1,
                "max_tokens": 4096,
                "response_format": {"type": "json_object"},
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"Groq request timed out after {timeout}s") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Groq HTTP {exc.response.status_code}: {exc.response.text[:400]}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Cannot reach {base_url}: {exc}") from exc

    data = response.json()
    content: str = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    input_tokens = usage.get("prompt_tokens") or _estimate_tokens(system_prompt + user_message)
    output_tokens = usage.get("completion_tokens") or _estimate_tokens(content)

    cost = budget.record_call(input_tokens, output_tokens)
    logger.info("call #%d | %d/%d tokens | $%.5f | total $%.5f", budget.calls_used, input_tokens, output_tokens, cost, budget.cost_used)

    return _parse_action(content), input_tokens, output_tokens


def _regex_extract(text: str) -> dict:
    """Last-resort field extraction when JSON parsing completely fails."""
    import re
    data: dict = {}

    for field in ("thought", "action", "replan_reason"):
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            data[field] = m.group(1)

    m = re.search(r'"made_progress"\s*:\s*(true|false)', text)
    if m:
        data["made_progress"] = m.group(1) == "true"

    def _extract_string_value(src: str, key: str) -> str | None:
        """Extract a JSON string value by key, handling escaped chars and multiline."""
        pattern = rf'"{re.escape(key)}"\s*:\s*"'
        m = re.search(pattern, src)
        if not m:
            return None
        i = m.end()
        result = []
        while i < len(src):
            ch = src[i]
            if ch == "\\" and i + 1 < len(src):
                result.append(src[i:i+2])
                i += 2
            elif ch == '"':
                break
            else:
                result.append(ch)
                i += 1
        return "".join(result)

    action = data.get("action", "")
    if action == "code_exec":
        val = _extract_string_value(text, "code")
        if val is not None:
            data["action_input"] = {"code": val}
    elif action == "web_search":
        val = _extract_string_value(text, "query")
        if val is not None:
            data["action_input"] = {"query": val}
    elif action == "calculator":
        val = _extract_string_value(text, "expression")
        if val is not None:
            data["action_input"] = {"expression": val}
    elif action == "finish":
        val = _extract_string_value(text, "answer")
        if val is not None:
            data["action_input"] = {"answer": val}

    return data


def _sanitize_json(text: str) -> str:
    """Replace literal newlines inside JSON string values with \\n."""
    # Replace unescaped literal newlines inside quoted strings
    result = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\":
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _parse_action(content: str) -> AgentAction:
    text = content.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    data: dict = {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        sanitized = _sanitize_json(text)
        start, end = sanitized.find("{"), sanitized.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(sanitized[start:end])
            except json.JSONDecodeError:
                data = _regex_extract(text)
                if not data.get("action"):
                    raise ValueError(f"Cannot parse LLM response as JSON.\nRaw: {text[:400]}")
        else:
            data = _regex_extract(text)
            if not data.get("action"):
                raise ValueError(f"No JSON object found in LLM response: {text[:300]}")

    raw_action = str(data.get("action", "finish")).lower()
    if raw_action in _VALID_ACTIONS:
        action = raw_action
    elif "search" in raw_action:
        action = "web_search"
    elif any(k in raw_action for k in ("code", "exec", "run", "python")):
        action = "code_exec"
    elif any(k in raw_action for k in ("calc", "math", "compute")):
        action = "calculator"
    else:
        action = "finish"

    action_input: dict = data.get("action_input", data.get("input", {}))
    if not isinstance(action_input, dict):
        action_input = {}

    if action == "finish" and "answer" not in action_input:
        answer = data.get("answer") or data.get("result") or data.get("response")
        if answer:
            action_input["answer"] = str(answer)

    return AgentAction(
        thought=str(data.get("thought", "(no thought)")),
        made_progress=bool(data.get("made_progress", True)),
        replan_reason=data.get("replan_reason") or None,
        action=action,
        action_input=action_input,
    )
