"""
ChromaDB 统一客户端
"""
import logging
from typing import Optional

import chromadb

logger = logging.getLogger("rag-rpg.infra.chroma")


class ChromaClient:
    def __init__(self, path: str):
        self._path = path
        self._client: Optional[chromadb.PersistentClient] = None
        self._collections: dict[str, chromadb.Collection] = {}

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self._path)
        return self._client

    def get_collection(self, name: str) -> chromadb.Collection:
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    def count(self, name: str) -> int:
        try:
            return self.get_collection(name).count()
        except Exception:
            return 0
