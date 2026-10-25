"""All settings in one place. Override any of them with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / ".chroma"
COLLECTION = "clinic_policies"

# Ollama models (run `ollama list` to see what you have)
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1")          # must support tool calling
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Retrieval
CHUNK_SIZE = 500        # characters
CHUNK_OVERLAP = 50
TOP_K = 3

# Agent guardrail: maximum graph steps before the agent is stopped
MAX_STEPS = 10
