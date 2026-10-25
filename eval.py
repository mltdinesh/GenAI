"""Small evaluation: does retrieval find the right document, and does the
agent's answer contain the facts it should?

    python eval.py

Simple keyword checks, not a full framework like RAGAS, but enough to catch
regressions when you change the chunk size, the prompt or the model.
"""
from datetime import date, timedelta

from agent import ask, build_agent
from ingest import load_index


def next_weekday() -> str:
    day = date.today() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.isoformat()


# question, expected source document (or None), words the answer must contain, tool that must be used
GOLDEN = [
    ("How much notice do I need to give to cancel an appointment?",
     "appointments.md", ["24"], "search_clinic_policies"),
    ("Can I see a cardiologist without a referral?",
     "appointments.md", ["referral"], "search_clinic_policies"),
    ("How long does a prescription refill take?",
     "prescriptions.md", ["3"], "search_clinic_policies"),
    ("How long does it take to get a full copy of my medical record?",
     "medical_records.md", ["30"], "search_clinic_policies"),
    ("Can my first dermatology visit be a video call?",
     "appointments.md", ["person"], "search_clinic_policies"),
    (f"What General Medicine times are free on {next_weekday()}?",
     None, ["09:00"], "find_appointment_slots"),
    # Not covered by the documents: the agent should admit it, not invent an answer.
    ("Does the clinic offer free parking?",
     None, ["call"], "search_clinic_policies"),
]


def eval_retrieval(store) -> int:
    print("== Retrieval: is the expected document in the top results? ==")
    passed = total = 0
    for question, source, _, _ in GOLDEN:
        if source is None:
            continue
        total += 1
        found = [d.metadata["source"] for d in store.similarity_search(question, k=3)]
        ok = source in found
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {question}  -> {found}")
    print(f"  {passed}/{total}\n")
    return passed == total


def eval_answers(agent) -> bool:
    print("== Answers: right tool, expected facts ==")
    passed = 0
    for question, _, must_contain, tool in GOLDEN:
        answer = ask(agent, question)
        text = answer.text.lower()
        problems = []
        if tool not in answer.tools_used:
            problems.append(f"did not call {tool}")
        missing = [w for w in must_contain if w.lower() not in text]
        if missing:
            problems.append(f"missing {missing}")
        ok = not problems
        passed += ok
        print(f"  {'PASS' if ok else 'FAIL'}  {question}")
        if not ok:
            print(f"        {'; '.join(problems)}\n        answer: {answer.text[:200]!r}")
    print(f"  {passed}/{len(GOLDEN)}")
    return passed == len(GOLDEN)


if __name__ == "__main__":
    store = load_index()
    retrieval_ok = eval_retrieval(store)
    answers_ok = eval_answers(build_agent(vector_store=store))
    raise SystemExit(0 if retrieval_ok and answers_ok else 1)
