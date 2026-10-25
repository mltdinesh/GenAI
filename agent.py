"""The agent: an LLM in a loop that can call our two tools.

Try it from the command line:
    python agent.py "Can I cancel my appointment tomorrow morning?"
"""
import re
import sys
from dataclasses import dataclass, field
from datetime import date

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError

import config
from ingest import load_index
from tools import find_appointment_slots, make_search_tool

SYSTEM_PROMPT = """You are the front-desk assistant for Lakeside Family Clinic.
Today's date is {today}.

Rules:
- For any question about clinic rules or procedures, call search_clinic_policies
  and answer ONLY from what it returns.
- Cite the source of each fact in square brackets, e.g. [appointments.md > Cancellations].
- If the policies do not cover the question, say you don't know and suggest calling the clinic.
- For available times, call find_appointment_slots. You cannot book or cancel appointments.
- Never give medical advice or a diagnosis. For symptoms, tell the patient to see a doctor,
  and for an emergency, to contact emergency services.
"""

# Simple input guard: strip obvious personal identifiers before text reaches the model.
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONE = re.compile(r"\+?\d[\d\s-]{7,}\d")


def redact(text: str) -> str:
    return _PHONE.sub("[PHONE]", _EMAIL.sub("[EMAIL]", text))


@dataclass
class Answer:
    text: str
    tools_used: list[str] = field(default_factory=list)


def build_agent(model=None, vector_store=None):
    model = model or ChatOllama(model=config.CHAT_MODEL, base_url=config.OLLAMA_URL, temperature=0)
    tools = [make_search_tool(vector_store or load_index()), find_appointment_slots]
    return create_agent(
        model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat()),
    )


def ask(agent, question: str) -> Answer:
    clean = redact(question)
    print(f"[agent] 2. after redaction: {clean!r}", flush=True)
    print(f"[agent] 3. invoking agent loop (model={config.CHAT_MODEL}, "
          f"recursion_limit={config.MAX_STEPS})", flush=True)
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": clean}]},
            config={"recursion_limit": config.MAX_STEPS},   # stops a looping agent
        )
    except GraphRecursionError:
        print("[agent] !! step limit hit, giving up", flush=True)
        return Answer("Sorry, I couldn't work that out. Please call the clinic.")

    messages = result["messages"]
    print(f"[agent] 4. agent loop finished, {len(messages)} messages in conversation:", flush=True)
    for i, m in enumerate(messages):
        if m.type == "ai" and m.tool_calls:
            for call in m.tool_calls:
                print(f"    [{i}] ai   -> wants tool {call['name']}({call['args']})", flush=True)
        elif m.type == "tool":
            print(f"    [{i}] tool <- {m.name} returned {len(m.content)} chars", flush=True)
        else:
            print(f"    [{i}] {m.type:<5}: {m.content[:80]!r}", flush=True)

    tools_used = [m.name for m in messages if m.type == "tool"]
    print(f"[agent] 5. final answer: {messages[-1].content[:120]!r}", flush=True)
    return Answer(text=messages[-1].content, tools_used=tools_used)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How late can I cancel an appointment?"
    answer = ask(build_agent(), question)
    print(answer.text)
    print(f"\n(tools used: {answer.tools_used or 'none'})")
