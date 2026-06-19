from __future__ import annotations

import ast
import math
import operator
import os
import subprocess
import threading
from typing import Any, Dict

import httpx

from agent.schemas import ToolResult

TOOL_TIMEOUT = int(os.environ.get("TOOL_TIMEOUT_SECONDS", "8"))
CODE_EXEC_TIMEOUT = int(os.environ.get("CODE_EXEC_TIMEOUT_SECONDS", "10"))


def web_search(query: str, max_results: int = 3) -> ToolResult:
    tavily_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if tavily_key:
        return _tavily_search(query, max_results, tavily_key)
    return _duckduckgo_search(query, max_results)


def _tavily_search(query: str, max_results: int, api_key: str) -> ToolResult:
    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query, "max_results": max_results},
            timeout=TOOL_TIMEOUT,
        )
        response.raise_for_status()
    except httpx.TimeoutException:
        return ToolResult(success=False, output="", error="Tavily search timed out", timed_out=True)
    except httpx.HTTPStatusError as exc:
        return ToolResult(success=False, output="", error=f"Tavily HTTP {exc.response.status_code}: {exc.response.text[:200]}")
    except httpx.RequestError as exc:
        return ToolResult(success=False, output="", error=f"Tavily request error: {exc}")

    data = response.json()
    results = data.get("results", [])
    if not results:
        return ToolResult(success=True, output="No results found for this query.")
    lines = []
    for r in results[:max_results]:
        lines.append(f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nSnippet: {r.get('content', 'N/A')}\n")
    return ToolResult(success=True, output="\n".join(lines))


def _duckduckgo_search(query: str, max_results: int) -> ToolResult:
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult(success=False, output="", error="ddgs (or duckduckgo_search) not installed and no TAVILY_API_KEY set")

    result_container: list = []
    exc_container: list = []

    def _run() -> None:
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    result_container.append(r)
        except Exception as exc:  # noqa: BLE001
            exc_container.append(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=TOOL_TIMEOUT)

    if thread.is_alive():
        return ToolResult(success=False, output="", error="DuckDuckGo search timed out", timed_out=True)
    if exc_container:
        return ToolResult(success=False, output="", error=f"DuckDuckGo error: {exc_container[0]}")
    if not result_container:
        return ToolResult(success=True, output="No results found for this query.")

    lines = []
    for r in result_container:
        lines.append(f"Title: {r.get('title', 'N/A')}\nURL: {r.get('href', 'N/A')}\nSnippet: {r.get('body', 'N/A')}\n")
    return ToolResult(success=True, output="\n".join(lines))


def code_exec(code: str, language: str = "python") -> ToolResult:
    import tempfile

    if language.lower() not in ("python", "python3"):
        return ToolResult(success=False, output="", error=f"Only Python is supported, got: {language!r}")

    # Decode \\n / \\t escape sequences that LLMs embed in JSON strings
    try:
        code = code.encode("raw_unicode_escape").decode("unicode_escape")
    except (UnicodeDecodeError, ValueError):
        pass

    # If still a single line with no real newlines, it may use semicolons — that's valid Python
    # Write to file so indentation/escaping is never an issue


    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=CODE_EXEC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error=f"Code execution timed out after {CODE_EXEC_TIMEOUT}s", timed_out=True)
    except FileNotFoundError:
        return ToolResult(success=False, output="", error="python3 not found in PATH")
    except OSError as exc:
        return ToolResult(success=False, output="", error=f"OS error running python3: {exc}")
    finally:
        import os as _os
        _os.unlink(tmp_path)

    if proc.returncode != 0:
        return ToolResult(
            success=False,
            output=proc.stdout[:2000] if proc.stdout else "",
            error=f"Exit {proc.returncode}: {proc.stderr[:500]}",
        )
    return ToolResult(success=True, output=proc.stdout[:4000] if proc.stdout else "(no output)")


_SAFE_BINOPS: Dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}

_SAFE_UNOPS: Dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES: Dict[str, Any] = {
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "exp": math.exp, "abs": abs, "round": round,
    "ceil": math.ceil, "floor": math.floor,
    "factorial": math.factorial, "gcd": math.gcd,
}


def _safe_eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        fn = _SAFE_BINOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        return fn(_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        fn = _SAFE_UNOPS.get(type(node.op))
        if fn is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return fn(_safe_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            return _SAFE_NAMES[node.id]  # type: ignore[return-value]
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.Call):
        func = _safe_eval_node(node.func)
        if not callable(func):
            raise ValueError(f"Not callable: {func}")
        return func(*[_safe_eval_node(a) for a in node.args])  # type: ignore[operator]
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def calculator(expression: str) -> ToolResult:
    result_box: list = []
    exc_box: list = []

    def _compute() -> None:
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result_box.append(_safe_eval_node(tree.body))
        except Exception as exc:  # noqa: BLE001
            exc_box.append(exc)

    thread = threading.Thread(target=_compute, daemon=True)
    thread.start()
    thread.join(timeout=TOOL_TIMEOUT)

    if thread.is_alive():
        return ToolResult(success=False, output="", error="Calculator timed out", timed_out=True)
    if exc_box:
        return ToolResult(success=False, output="", error=f"Calculation error: {exc_box[0]}")
    return ToolResult(success=True, output=f"{expression} = {result_box[0]}")


TOOL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "web_search": {
        "fn": web_search,
        "description": 'Search the web. Input: {"query": "<search query>", "max_results": 3}',
    },
    "code_exec": {
        "fn": code_exec,
        "description": 'Execute Python code and return stdout. Input: {"code": "<python source>"}',
    },
    "calculator": {
        "fn": calculator,
        "description": 'Evaluate a math expression (supports +,-,*,/,**,%, sqrt, log, sin, cos, pi, e, factorial…). Input: {"expression": "sqrt(2) * pi"}',
    },
}


def execute_tool(name: str, inputs: Dict[str, Any]) -> ToolResult:
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return ToolResult(success=False, output="", error=f"Unknown tool: {name!r}. Available: {list(TOOL_REGISTRY)}")
    try:
        return entry["fn"](**inputs)
    except TypeError as exc:
        return ToolResult(success=False, output="", error=f"Invalid inputs for {name!r}: {exc}")
