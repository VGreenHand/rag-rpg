"""
嵌入服务 HTTP 客户端：替换 SentenceTransformer 的直接调用，
通过 HTTP 请求独立模型服务获取嵌入向量。

用法:
    from embedding_client import get_embedding_client
    client = get_embedding_client()
    emb = client.encode("文本").tolist()       # 单文本 → numpy 数组
    embs = client.encode(["a", "b"]).tolist()  # 批量 → 2D numpy 数组
"""
import json
import logging
import urllib.request
import urllib.error

import numpy as np

from config import EMBEDDING_SERVICE_URL

logger = logging.getLogger("rag-rpg.embedding-client")


class EmbeddingClient:
    """SentenceTransformer 的 HTTP 替代品，接口兼容 .encode()"""

    def __init__(self, service_url: str = EMBEDDING_SERVICE_URL):
        self._service_url = service_url.rstrip("/")

    def encode(self, texts):
        """兼容 SentenceTransformer.encode() 接口

        参数:
            texts: str | list[str] — 待编码文本
        返回:
            np.ndarray — 与 SentenceTransformer.encode() 返回值类型一致
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

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

        embeddings = result["embeddings"]
        if single:
            return np.array(embeddings[0], dtype=np.float32)
        return np.array(embeddings, dtype=np.float32)


_embedding_client_instance: EmbeddingClient = None


def get_embedding_client() -> EmbeddingClient:
    """获取 EmbeddingClient 单例（轻量无状态，无需重复构造）"""
    global _embedding_client_instance
    if _embedding_client_instance is None:
        _embedding_client_instance = EmbeddingClient()
    return _embedding_client_instance
