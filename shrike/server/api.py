"""FastAPI serving layer: POST /v1/completions (SSE streaming or JSON),
GET /health, GET /metrics.

Run:  python -m shrike.server.api --model models_cache/qwen2.5-0.5b-instruct
"""

from __future__ import annotations

import argparse
import json
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from shrike.engine.engine import LLMEngine
from shrike.engine.request import SamplingParams
from shrike.server.async_engine import AsyncEngine

_engine_config: dict = {}
async_engine: AsyncEngine | None = None


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    ignore_eos: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global async_engine
    async_engine = AsyncEngine(LLMEngine(**_engine_config))
    async_engine.start()
    yield
    await async_engine.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return async_engine.engine.metrics()


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    params = SamplingParams(
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        ignore_eos=req.ignore_eos,
    )
    try:
        req_id, queue = await async_engine.submit(req.prompt, params)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=422)

    if req.stream:
        async def sse():
            while True:
                ev = await queue.get()
                if ev.text:
                    yield f"data: {json.dumps({'text': ev.text})}\n\n"
                if ev.finished:
                    yield f"data: {json.dumps({'finish_reason': ev.finish_reason})}\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(sse(), media_type="text/event-stream")

    chunks: list[str] = []
    finish_reason = None
    num_tokens = 0
    while True:
        ev = await queue.get()
        num_tokens += 1
        chunks.append(ev.text)
        if ev.finished:
            finish_reason = ev.finish_reason
            break
    return JSONResponse(
        {"id": req_id, "text": "".join(chunks), "finish_reason": finish_reason, "num_tokens": num_tokens}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models_cache/qwen2.5-0.5b-instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens-per-step", type=int, default=512)
    parser.add_argument("--max-running", type=int, default=256)
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--attention-backend", choices=["einsum", "triton"], default="einsum")
    args = parser.parse_args()
    _engine_config.update(
        model_dir=args.model,
        max_tokens_per_step=args.max_tokens_per_step,
        max_running=args.max_running,
        enable_prefix_caching=not args.no_prefix_caching,
        attention_backend=args.attention_backend,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
