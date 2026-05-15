from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DialogueTurn:
    speaker: str
    name: str
    content: str
    turn: int = 0
    timestamp: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class SearchResult:
    collection: str
    id: str
    document: str
    metadata: dict
    score: float


class DialogueRepository(ABC):

    @abstractmethod
    def save_turn(self, turn: DialogueTurn, embedding: list[float]) -> str:
        ...

    @abstractmethod
    def search(self, query_embedding: list[float], k: int = 3) -> list[SearchResult]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class EmbeddingService(ABC):

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        ...


class CheckpointStore(ABC):

    @abstractmethod
    def save(self, state: dict) -> None:
        ...

    @abstractmethod
    def load(self) -> Optional[dict]:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...
