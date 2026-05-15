import uuid
from typing import Optional

from rag_rpg.core.interfaces import DialogueRepository, DialogueTurn, SearchResult
from rag_rpg.infrastructure.chroma.client import ChromaClient


class ChromaDialogueRepository(DialogueRepository):

    def __init__(self, chroma_client: ChromaClient, collection_name: str):
        self._chroma = chroma_client
        self._collection_name = collection_name

    def save_turn(self, turn: DialogueTurn, embedding: list[float]) -> str:
        col = self._chroma.get_collection(self._collection_name)
        doc_id = str(uuid.uuid4())
        col.add(
            embeddings=[embedding],
            documents=[turn.content],
            metadatas=[{
                "speaker": turn.speaker,
                "name": turn.name,
                "turn": turn.turn,
                "session_id": turn.session_id or "default",
            }],
            ids=[doc_id],
        )
        return doc_id

    def search(self, query_embedding: list[float], k: int = 3) -> list[SearchResult]:
        col = self._chroma.get_collection(self._collection_name)
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append(SearchResult(
                collection=self._collection_name,
                id=results["ids"][0][i],
                document=results["documents"][0][i],
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                score=round(1.0 - results["distances"][0][i], 4),
            ))
        return output

    def count(self) -> int:
        return self._chroma.count(self._collection_name)
