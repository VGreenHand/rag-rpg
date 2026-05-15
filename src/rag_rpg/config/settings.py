import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    base_dir: Path = Path(__file__).resolve().parent.parent.parent.parent
    default_profile: str = "default"
    chroma_path: str = ""
    collection_skills: str = "character_skills"
    collection_memory: str = "my_rag_memory"
    collection_dialogue: str = "dialogue_memory"
    collection_plot_state: str = "plot_state"
    model_name: str = "shibing624/text2vec-base-chinese"
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_key: str = ""
    embedding_service_host: str = "127.0.0.1"
    embedding_service_port: int = 8766
    dialogue_dir: str = ""
    max_context_turns: int = 5
    top_k_results: int = 3
    min_relevance: float = 0.25
    max_constraint_chars: int = 800
    dedup_similarity_threshold: float = 0.92
    skill_proficiency_pattern: str = r"熟练度[\s：:]*(\d+)/100"
    constraint_cooldown_turns: int = 3
    max_active_constraints: int = 5
    batch_file: str = ""


def default_settings() -> Settings:
    s = Settings()
    s.chroma_path = str(s.base_dir / "chroma_db")
    s.api_key = os.environ.get("RAG_RPG_API_KEY", "rag-rpg-local")
    s.dialogue_dir = str(s.base_dir / "dialogues")
    s.batch_file = str(s.base_dir / "new_batch.txt")
    return s
