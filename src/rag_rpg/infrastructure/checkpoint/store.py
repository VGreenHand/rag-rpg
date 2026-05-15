import json
from pathlib import Path
from typing import Optional

from rag_rpg.core.interfaces import CheckpointStore


class CheckpointStoreImpl(CheckpointStore):

    def __init__(self, file_path: str):
        self._file = Path(file_path)

    def save(self, state: dict) -> None:
        self._file.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load(self) -> Optional[dict]:
        if not self._file.exists():
            return None
        try:
            with open(self._file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def clear(self) -> None:
        if self._file.exists():
            self._file.unlink(missing_ok=True)
