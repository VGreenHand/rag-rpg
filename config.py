import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

# --- ChromaDB ---
CHROMA_PATH = str(BASE_DIR / "chroma_db")
COLLECTION_SKILLS = "character_skills"
COLLECTION_MEMORY = "my_rag_memory"
COLLECTION_DIALOGUE = "dialogue_memory"
COLLECTION_PLOT_STATE = "plot_state"

# --- Embedding ---
MODEL_NAME = "shibing624/text2vec-base-chinese"

# --- Server ---
API_HOST = "127.0.0.1"
API_PORT = 8765
API_KEY = os.environ.get("RAG_RPG_API_KEY", "rag-rpg-local")

# --- Dialogue Processing ---
DIALOGUE_DIR = BASE_DIR / "dialogues"
MAX_CONTEXT_TURNS = 5
TOP_K_RESULTS = 3
MIN_RELEVANCE = 0.25
MAX_CONSTRAINT_CHARS = 800

# --- Deduplication ---
DEDUP_SIMILARITY_THRESHOLD = 0.92
SUMMARY_MAX_CHARS = 300
SKILL_PROFICIENCY_PATTERN = r'熟练度\s*(\d+)/100'

# --- Constraint Engine ---
CONSTRAINT_COOLDOWN_TURNS = 3
MAX_ACTIVE_CONSTRAINTS = 5

# --- Batch Ingest ---
BATCH_FILE = BASE_DIR / "new_batch.txt"
