"""REST API around the agent.

    uvicorn api:app --reload
    then open http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from agent import ask, build_agent


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] startup: building agent (loads Chroma index + Ollama model)...", flush=True)
    app.state.agent = build_agent()    # build once at startup, not per request
    print("[api] startup: agent ready", flush=True)
    yield
    print("[api] shutdown", flush=True)


app = FastAPI(title="Clinic Assistant", lifespan=lifespan)


# Plain `def`, not `async def`: the agent call is blocking, so FastAPI runs it
# in a threadpool instead of blocking the event loop.
@app.post("/ask", response_model=AskResponse)
def ask_endpoint(body: AskRequest, request: Request) -> AskResponse:
    print(f"\n[api] 1. POST /ask received: {body.question!r}", flush=True)
    answer = ask(request.app.state.agent, body.question)
    print(f"[api] 6. returning response (tools_used={answer.tools_used})", flush=True)
    return AskResponse(answer=answer.text, tools_used=answer.tools_used)


@app.get("/health")
def health():
    return {"status": "ok"}
