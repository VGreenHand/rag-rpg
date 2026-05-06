"""
上下文感知查询引擎：分析当前对话状态，生成多策略查询，在向量库中检索相关信息
"""
import re
from typing import Optional, Union

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PATH, MODEL_NAME, COLLECTION_SKILLS,
    COLLECTION_DIALOGUE, COLLECTION_MEMORY,
    MAX_CONTEXT_TURNS, TOP_K_RESULTS, MIN_RELEVANCE,
)


class QueryEngine:
    """分析对话上下文并在多个向量集合中执行语义检索"""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self._collections: dict[str, object] = {}

    def _get_collection(self, name: str):
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        return self._collections[name]

    def build_queries(self, dialogue_context: list[dict]) -> list[str]:
        """从对话上下文中构建多个查询变体以覆盖不同检索角度"""
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
        """提取文本核心意图用于生成精简查询"""
        text = re.sub(r'[\*_~]', '', text)
        text = re.sub(r'[，。！？、；：""（）\s]+', ' ', text).strip()
        segments = re.split(r'[，。！？\n]', text)
        meaningful = [s.strip() for s in segments if len(s.strip()) >= 6]
        return " ".join(meaningful[:3]) if meaningful else text[:100]

    def _extract_actions(self, text: str) -> str:
        """提取文本中的动作描述"""
        actions = re.findall(
            r'(?:使用|施展|释放|发动|拔出|挥动|冲向|进入|探索|打开|检查|观察|'
            r'感到|发现|听见|看见)(?:了)?[\u4e00-\u9fff]{2,20}',
            text
        )
        return " ".join(actions[:5])

    def search(self, query: str, collections: list[str] = None,
               k: int = None) -> list[dict]:
        """单查询检索"""
        if collections is None:
            collections = [COLLECTION_SKILLS, COLLECTION_DIALOGUE, COLLECTION_MEMORY]
        if k is None:
            k = TOP_K_RESULTS

        query_vec = self.model.encode(query).tolist()
        all_results = []

        for col_name in collections:
            try:
                col = self._get_collection(col_name)
                results = col.query(
                    query_embeddings=[query_vec],
                    n_results=k
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
            except Exception:
                continue

        return all_results

    def multi_search(self, dialogue_context: list[dict],
                     collections: list[str] = None,
                     k: int = None) -> dict:
        """多查询融合检索：生成多个查询分别检索后合并去重排序"""
        queries = self.build_queries(dialogue_context)
        if not queries:
            return {"results": [], "total_hits": 0, "target_collections": []}

        if collections is None:
            collections = [COLLECTION_SKILLS, COLLECTION_DIALOGUE, COLLECTION_MEMORY]
        if k is None:
            k = TOP_K_RESULTS

        seen_ids = set()
        merged = []

        for query in queries:
            hits = self.search(query, collections, k)
            for hit in hits:
                dedup_key = f"{hit['collection']}:{hit['id']}"
                if dedup_key not in seen_ids:
                    seen_ids.add(dedup_key)
                    merged.append(hit)

        merged.sort(key=lambda h: h["score"], reverse=True)
        merged = merged[:k * 2]

        return {
            "results": merged,
            "total_hits": len(merged),
            "target_collections": collections,
            "queries_used": len(queries),
        }

    def format_for_llm(self, search_results: dict) -> str:
        """将检索结果转化为大模型可读的格式化文本"""
        results = search_results.get("results", [])
        if not results:
            return ""

        blocks = []
        for i, r in enumerate(results, 1):
            col = r["collection"]
            meta = r.get("metadata", {})
            etype = meta.get("type", meta.get("entry_key", "info"))
            block = f"[{i}] [{col}] [{etype}] (相关度: {r['score']:.2f})\n{r['document'][:300]}"
            blocks.append(block)

        header = (
            f"[RAG-RPG 记忆检索] 共命中 {len(results)} 条相关记忆，"
            f"涵盖 {search_results.get('total_hits', 0)} 条去重结果。\n"
            f"{'─' * 50}\n"
        )
        return header + "\n\n".join(blocks)


_query_engine_instance: Optional[QueryEngine] = None


def get_query_engine() -> QueryEngine:
    global _query_engine_instance
    if _query_engine_instance is None:
        _query_engine_instance = QueryEngine()
    return _query_engine_instance
