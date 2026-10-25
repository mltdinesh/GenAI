"""The two tools the agent can call.

The model never runs these itself. It only asks for a tool by name with
arguments; LangChain validates the arguments against the schema and our
code runs the function.
"""
from datetime import date
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

import config

Department = Literal["General Medicine", "Paediatrics", "Cardiology", "Dermatology"]

# Fake schedule standing in for a real booking system. Read-only on purpose:
# the agent can look up slots but has no tool that books or cancels anything.
FAKE_SLOTS: dict[str, list[str]] = {
    "General Medicine": ["09:00", "09:30", "11:00", "14:30", "16:00"],
    "Paediatrics": ["10:00", "13:30", "15:00"],
    "Cardiology": ["08:30", "12:00"],
    "Dermatology": ["11:30", "15:30"],
}


def make_search_tool(vector_store):
    """Build the retrieval tool around a vector store (lets tests pass a fake one)."""

    @tool
    def search_clinic_policies(query: str) -> str:
        """Search the clinic's policy documents (appointments, cancellations,
        telehealth, prescription refills, medical records, privacy).
        Use this for any question about clinic rules or procedures."""
        print(f"    [tool] search_clinic_policies(query={query!r})", flush=True)
        docs = vector_store.similarity_search(query, k=config.TOP_K)
        for d in docs:
            print(f"    [tool]   hit: {d.metadata['source']} > {d.metadata['section']}", flush=True)
        if not docs:
            return "No relevant policy found."
        return "\n\n".join(
            f"[{d.metadata['source']} > {d.metadata['section']}]\n{d.page_content}"
            for d in docs
        )

    return search_clinic_policies


class SlotQuery(BaseModel):
    department: Department = Field(description="Clinic department")
    day: str = Field(description="Date in YYYY-MM-DD format")

    @field_validator("day")
    @classmethod
    def must_be_valid_future_date(cls, value: str) -> str:
        parsed = date.fromisoformat(value)          # raises ValueError if malformed
        if parsed < date.today():
            raise ValueError("date must be today or later")
        return value


@tool(args_schema=SlotQuery)
def find_appointment_slots(department: str, day: str) -> str:
    """Look up open appointment times for a department on a given date.
    Read-only: it does not book anything."""
    print(f"    [tool] find_appointment_slots(department={department!r}, day={day!r})", flush=True)
    weekday = date.fromisoformat(day).weekday()
    if weekday >= 5:
        return f"The clinic is closed on {day} (weekend)."
    slots = FAKE_SLOTS[department]
    return f"Open {department} slots on {day}: {', '.join(slots)}"
