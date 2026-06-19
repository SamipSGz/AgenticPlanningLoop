from __future__ import annotations

import asyncio
import json
import os

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, StreamingResponse

from agent.budget import BudgetManager
from agent.loop import run_agent

app = FastAPI(title="Constrained Agent")

_STATIC = os.path.join(os.path.dirname(__file__), "static", "index.html")


@app.get("/", response_class=HTMLResponse)
async def index():
    with open(_STATIC) as f:
        return f.read()


@app.get("/health")
async def health():
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    llm_provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    checks: dict = {"status": "ok", "llm_provider": llm_provider}

    if llm_provider == "ollama":
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{ollama_url}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
                checks["ollama"] = "ok"
                checks["ollama_models"] = models
        except Exception as exc:
            checks["ollama"] = f"unreachable: {exc}"
            checks["status"] = "degraded"
    elif llm_provider == "anthropic":
        checks["anthropic_key"] = "set" if os.environ.get("ANTHROPIC_API_KEY") else "missing"
        if not os.environ.get("ANTHROPIC_API_KEY"):
            checks["status"] = "degraded"

    checks["tavily_key"] = "set" if os.environ.get("TAVILY_API_KEY") else "unset (using DuckDuckGo)"
    return checks


@app.get("/run-stream")
async def run_stream(task: str = Query(..., min_length=1)):
    async def event_stream():
        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def on_event(event: dict) -> None:
            loop.call_soon_threadsafe(q.put_nowait, event)

        async def _run():
            budget = BudgetManager()
            await asyncio.to_thread(run_agent, task, budget, on_event)
            loop.call_soon_threadsafe(q.put_nowait, None)

        asyncio.create_task(_run())

        while True:
            event = await q.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
