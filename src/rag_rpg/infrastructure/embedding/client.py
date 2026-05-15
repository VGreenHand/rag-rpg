import json
import logging
from typing import Optional

import urllib.request
import urllib.error

from rag_rpg.core.interfaces import EmbeddingService

logger = logging.getLogger("rag-rpg.infra.embedding")


class EmbeddingClientService(EmbeddingService):

    def __init__(self, service_url: str):
        self._service_url = service_url.rstrip("/")

    def encode(self, texts: list[str]) -> list[list[float]]:
        data = json.dumps({"texts": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._service_url}/embed",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            logger.error(f"嵌入服务连接失败 ({self._service_url}): {e}")
            raise ConnectionError(
                f"无法连接到嵌入服务 {self._service_url}，请确保 embedding_service.py 已启动"
            ) from e
        return result["embeddings"]
