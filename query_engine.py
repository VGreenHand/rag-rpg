"""
上下文感知查询引擎：分析当前对话状态，生成多策略查询，在向量库中检索相关信息
v2.0: 超时保护 + 降级查询 + 部分失败容忍
"""
import re
import logging
from typing import Optional, Union

from config import (
    CHROMA_PATH, MODEL_NAME, COLLECTION_SKILLS,
    COLLECTION_DIALOGUE, COLLECTION_MEMORY,
    MAX_CONTEXT_TURNS, TOP_K_RESULTS, MIN_RELEVANCE,
    get_chroma_path, DEFAULT_PROFILE,
)
from rag_rpg.infrastructure.chroma.client import ChromaClient
from checkpoint_manager import TimeoutError as CkpTimeoutError
from embedding_client import get_embedding_client
from rag_rpg.common.utils import SafeTimer, SingletonFactory

logger = logging.getLogger("rag-rpg.query")

QUERY_TIMEOUT = 6.0
ENCODE_TIMEOUT = 8.0
COLLECTION_GET_TIMEOUT = 4.0


class QueryEngine:
    def __init__(self, profile: str = DEFAULT_PROFILE):
        self._profile = profile
        self.model = get_embedding_client()
        self._chroma = ChromaClient(path=get_chroma_path(profile))
        self._stats: dict[str, int] = {"timeouts": 0, "degraded": 0, "errors": 0}

    def _get_collection(self, name: str):
        try:
            return SafeTimer.run(
                lambda: self._chroma.get_collection(name),
                COLLECTION_GET_TIMEOUT,
            )
        except CkpTimeoutError:
            self._stats["errors"] += 1
            raise

    def build_queries(self, dialogue_context: list[dict]) -> list[str]:
        recent = dialogue_context[-MAX_CONTEXT_TURNS:]
        user_msgs = [m["content"] for m in recent if m.get("speaker") == "user"]
        ai_msgs = [m["content"] for m in recent if m.get("speaker") == "ai"]
        all_text = " ".join([m["content"] for m in recent])

        queries = []

        if user_msgs:
            last_user = user_msgs[-1][:200]
            queries.append(last_user)
            summary = self._summarize_for_query(last_user)
            if summary and summary != last_user[:50]:
                queries.append(summary)

        if ai_msgs:
            last_ai = ai_msgs[-1][:200]
            queries.append(last_ai)

        combined = f"当前场景: {all_text[:300]}"
        queries.append(combined)

        actions = self._extract_actions(all_text)
        if actions:
            queries.append(f"剧情事件: {actions}")

        return queries

    def _summarize_for_query(self, text: str) -> str:
        text = re.sub(r'[\*_~]', '', text)
        text = re.sub(r'[，。！？、；：""（）\s]+', ' ', text).strip()
        segments = re.split(r'[，。！？\n]', text)
        meaningful = [s.strip() for s in segments if len(s.strip()) >= 6]
        return " ".join(meaningful[:3]) if meaningful else text[:100]

    def _extract_actions(self, text: str) -> str:
        actions = re.findall(
            r'(?:使用|施展|释放|发动|拔出|挥动|冲向|进入|探索|打开|检查|观察|'
            r'感到|发现|听见|看见)(?:了)?[\u4e00-\u9fff]{2,20}',
            text
        )
        return " ".join(actions[:5])

    def search(self, query: str, collections: list[str] = None,
               k: int = None, fail_fast: bool = False) -> list[dict]:
        if collections is None:
            collections = [COLLECTION_SKILLS, COLLECTION_DIALOGUE, COLLECTION_MEMORY]
        if k is None:
            k = TOP_K_RESULTS

        try:
            query_vec = SafeTimer.run(
                self.model.encode, ENCODE_TIMEOUT, query
            ).tolist()
        except CkpTimeoutError:
            self._stats["errors"] += 1
            logger.warning("查询编码超时，返回空结果")
            return []

        all_results = []
        degraded_collections = []

        for col_name in collections:
            try:
                col = self._get_collection(col_name)
                results = SafeTimer.run(
                    lambda: col.query(
                        query_embeddings=[query_vec],
                        n_results=k,
                    ),
                    QUERY_TIMEOUT,
                )
                for i in range(len(results["ids"][0])):
                    dist = results["distances"][0][i]
                    if dist < MIN_RELEVANCE:
                        all_results.append({
                            "collection": col_name,
                            "id": results["ids"][0][i],
                            "document": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "score": round(1.0 - dist, 4),
                        })
            except CkpTimeoutError:
                self._stats["timeouts"] += 1
                degraded_collections.append(col_name)
                logger.warning(f"集合查询超时: {col_name}，降级跳过")
                if fail_fast:
                    raise
            except Exception as e:
                self._stats["errors"] += 1
                degraded_collections.append(col_name)
                logger.debug(f"集合查询异常: {col_name}: {e}")
                if fail_fast:
                    raise

        if degraded_collections:
            self._stats["degraded"] += 1

        return all_results

    def multi_search(self, dialogue_context: list[dict],
                     collections: list[str] = None,
                     k: int = None,
                     fail_fast: bool = False) -> dict:
        queries = self.build_queries(dialogue_context)
        if not queries:
            return {
                "results": [], "total_hits": 0,
                "target_collections": [], "queries_used": 0,
                "degraded": False,
            }

        if collections is None:
            collections = [COLLECTION_SKILLS, COLLECTION_DIALOGUE, COLLECTION_MEMORY]
        if k is None:
            k = TOP_K_RESULTS

        seen_ids = set()
        merged = []
        degraded = False

        for query in queries:
            try:
                hits = self.search(query, collections, k, fail_fast=fail_fast)
                for hit in hits:
                    dedup_key = f"{hit['collection']}:{hit['id']}"
                    if dedup_key not in seen_ids:
                        seen_ids.add(dedup_key)
                        merged.append(hit)
            except CkpTimeoutError:
                degraded = True
                logger.warning(f"多查询中单查询超时，跳过: {query[:50]}...")
            except Exception as e:
                degraded = True
                logger.error(f"多查询中单查询失败: {e}")

        merged.sort(key=lambda h: h["score"], reverse=True)
        merged = merged[:k * 2]

        return {
            "results": merged,
            "total_hits": len(merged),
            "target_collections": collections,
            "queries_used": len(queries),
            "degraded": degraded,
            "stats": dict(self._stats),
        }

    def format_for_llm(self, search_results: dict) -> str:
        results = search_results.get("results", [])
        if not results:
            return ""

        blocks = []
        for i, r in enumerate(results, 1):
            col = r["collection"]
            meta = r.get("metadata", {})
            etype = meta.get("type", meta.get("entry_key", "info"))
            block = (
                f"[{i}] [{col}] [{etype}] "
                f"(相关度: {r['score']:.2f})\n{r['document'][:300]}"
            )
            blocks.append(block)

        degraded_note = ""
        if search_results.get("degraded"):
            degraded_note = " [部分结果因超时降级]"

        header = (
            f"[RAG-RPG 记忆检索] 共命中 {len(results)} 条相关记忆，"
            f"涵盖 {search_results.get('total_hits', 0)} 条去重结果。{degraded_note}\n"
            f"{'─' * 50}\n"
        )
        return header + "\n\n".join(blocks)

    def get_health(self) -> dict:
        return {
            "collections_loaded": len(self._chroma._collections),
            "stats": dict(self._stats),
        }


get_query_engine = SingletonFactory(QueryEngine)
