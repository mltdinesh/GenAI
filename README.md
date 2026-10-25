# Clinic Assistant — RAG + tool-calling agent (practice project)

A front-desk assistant for a **fictional** clinic. It answers policy questions
from the clinic's documents with citations, and looks up free appointment
slots. Everything runs locally on Ollama, so no data leaves the machine.

All clinic data in `docs/` and the appointment schedule are made up.

## Setup

```bash
ollama pull llama3.1            # chat model (must support tool calling)
ollama pull nomic-embed-text    # embedding model

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python ingest.py                                       # build the vector index (re-run after editing docs/)
python agent.py "How late can I cancel an appointment?"
python eval.py                                         # run the golden-question evaluation
uvicorn api:app --reload                               # then open http://localhost:8000/docs
```

Using a different model: `CHAT_MODEL=qwen2.5 python agent.py "..."` (any Ollama model with tool support).

## How it works

```
ingest.py   docs/*.md ─► split on "##" headings ─► split to 500 chars (50 overlap)
                      ─► embed (nomic-embed-text) ─► Chroma (.chroma/)

agent.py    question ─► redact email/phone ─► agent loop (llama3.1, temperature 0)
                          ├─ search_clinic_policies  → top 3 chunks with [file > section]
                          └─ find_appointment_slots  → fake schedule (read-only)
                      ─► answer with citations

api.py      POST /ask  {question}  →  {answer, tools_used}
eval.py     7 golden questions: retrieval hit rate + answer checks
```

| File | What it does |
|---|---|
| `config.py` | Models, chunk size, top-k, max agent steps |
| `ingest.py` | Load, chunk, embed and store the documents |
| `tools.py` | The two tools; Pydantic validates the slot tool's arguments |
| `agent.py` | System prompt, PII redaction, the agent, step limit |
| `api.py` | FastAPI endpoint |
| `eval.py` | Evaluation against golden questions |

## Guardrails in the code

- **Grounding:** the prompt says to answer only from retrieved policy text and cite it.
- **"I don't know":** questions the documents don't cover should get a referral to the clinic, not a guess (checked by the parking question in `eval.py`).
- **Argument validation:** an unknown department or a past date is rejected by Pydantic. The error goes back to the model so it can correct itself.
- **Least privilege:** tools are read-only; there is no booking or cancelling tool.
- **Step limit:** `recursion_limit` stops a looping agent.
- **PII redaction:** emails and phone numbers are masked before reaching the model.
- **No medical advice:** the prompt sends symptoms to a doctor and emergencies to emergency services.

## Things to try

1. Run `eval.py`, note the score.
2. Change `CHUNK_SIZE` to 100, re-run `ingest.py` and `eval.py`. What breaks?
3. Remove the "say you don't know" rule from the prompt and ask about parking.
4. Ask "What times are free for dentistry tomorrow?" and see how it handles an invalid department.
5. Swap `llama3.1` for another model and compare the eval score.
