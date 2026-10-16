"""
config.py
=========
Central configuration for the AI-Powered Adaptive Learning Chatbot.

Keeping every tunable value in one place guarantees that ``ingest.py``
(which writes the vector store) and ``rag_engine.py`` (which reads it)
always agree on paths, embedding model and collection name.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"            # Drop your PDF study material here
CHROMA_DIR = BASE_DIR / "chroma_db"     # Persistent ChromaDB directory
COLLECTION_NAME = "learning_materials"  # Chroma collection name

# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------------------------------------------------------
# Text splitting (as per project specification)
# --------------------------------------------------------------------------
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
RETRIEVER_K = 4  # Number of chunks retrieved per question

# --------------------------------------------------------------------------
# LLM (served locally by Ollama)
# --------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"
AVAILABLE_MODELS = ["llama3", "mistral"]
LLM_TEMPERATURE = 0.3  # Low temperature -> more factual, grounded answers

# --------------------------------------------------------------------------
# Adaptive learning levels shown in the UI
# --------------------------------------------------------------------------
STUDENT_LEVELS = ["Beginner", "Intermediate", "Advanced"]
