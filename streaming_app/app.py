import asyncio
import uuid
from time import time
from typing import AsyncGenerator, List, Optional
from fastapi import FastAPI
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from ray import serve

# ---------- OpenAI-compatible streaming models (text completions) ----------

class CompletionStreamChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None

class CompletionStreamChunk(BaseModel):
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: List[CompletionStreamChoice]

def make_chunk(model: str, index: int, text: str, *, finished: bool = False) -> CompletionStreamChunk:
    return CompletionStreamChunk(
        id=f"cmpl-{uuid.uuid4().hex[:24]}",
        created=int(time()),
        model=model,
        choices=[
            CompletionStreamChoice(
                index=index,
                text=text,
                finish_reason="stop" if finished else None,
            )
        ],
    )

# ---------- FastAPI app & Ray Serve deployments ----------

def app():
    app = FastAPI()

    @app.get("/streaming")
    async def streaming(message: str, num_tokens: int, tpot: float, ttft: float):
        handle = serve.get_deployment_handle("GrandChildDeployment", "app1")

        async def _generate(message: str, num_tokens: int, tpot: float, ttft: float) -> AsyncGenerator[str, None]:
            async for sse_line in handle.options(stream=True).streaming.remote(message, num_tokens, tpot, ttft):
                # child already yields properly formatted SSE lines
                yield sse_line
        return StreamingResponse(_generate(message, num_tokens, tpot, ttft), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"})

    @app.get("/streaming-direct")
    async def streaming_direct(message: str, num_tokens: int, tpot: float, ttft: float):
        async def _generate(message: str, num_tokens: int, tpot: float, ttft: float) -> AsyncGenerator[str, None]:
            model_name = "dsv3"
            # Sleep for TTFT before first token
            await asyncio.sleep(ttft)
            
            for i in range(num_tokens):
                # For first token, no additional sleep (TTFT already handled above)
                # For subsequent tokens, sleep for TPOT
                if i > 0:
                    await asyncio.sleep(tpot)
                
                chunk = make_chunk(model_name, index=i, text=f"{message} {i}", finished=(i == num_tokens - 1))
                yield f"data: {chunk.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(_generate(message, num_tokens, tpot, ttft), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "Connection": "keep-alive"})

    return app


@serve.deployment(max_ongoing_requests=10000)
class GrandChildDeployment:
    async def streaming(self, message: str, num_tokens: int, tpot: float, ttft: float):
        model_name = "dsv3"
        await asyncio.sleep(ttft)
        for i in range(num_tokens):
            await asyncio.sleep(tpot)
            chunk = make_chunk(model_name, index=i, text=f"{message} {i}", finished=(i == num_tokens - 1))
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"


@serve.deployment(max_ongoing_requests=10000)
@serve.ingress(app)
class MyDeployment:
    def __init__(self, child):
        self.child = child

app = MyDeployment.bind(GrandChildDeployment.bind())
