"""
对话处理管道：接收原始对话 → 清洗 → 提取关键信息 → 写入TXT → 向量入库
支持实时单条和批量处理两种模式
"""
import re
import uuid
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PATH, COLLECTION_DIALOGUE, COLLECTION_SKILLS,
    MODEL_NAME, DIALOGUE_DIR, BATCH_FILE, COLLECTION_MEMORY,
)


class DialoguePipeline:
    """对话处理管道：串联清洗/提取/TXT写入/ChromaDB入库全流程"""

    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.dialogue_col = self.client.get_or_create_collection(
            name=COLLECTION_DIALOGUE,
            metadata={"hnsw:space": "cosine"}
        )
        self._known_terms: Optional[list] = None

    def _clean_text(self, text: str) -> str:
        """清洗文本：移除HTML标签、星号动作描写标记、多余空白"""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\*{1,3}', '', text)
        text = re.sub(r'[_\*~]{1,2}', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'^\s*[-–—]\s*', '', text, flags=re.MULTILINE)
        return text.strip()

    def _extract_key_terms(self, text: str) -> list[str]:
        """从文本中提取已知技能/角色/地名等关键术语"""
        found = set()
        for term in self._load_known_terms():
            if term in text:
                found.add(term)
        return sorted(found, key=lambda t: len(t), reverse=True)

    def _load_known_terms(self) -> list[str]:
        """从 character_skills 和 my_rag_memory 加载已知术语"""
        if self._known_terms is not None:
            return self._known_terms
        terms = set()
        for col_name in (COLLECTION_SKILLS, COLLECTION_MEMORY):
            try:
                col = self.client.get_collection(name=col_name)
                results = col.get()
                for meta in results.get("metadatas", []):
                    ek = meta.get("entry_key", "")
                    if ek:
                        terms.add(ek)
                for doc in results.get("documents", []):
                    tag_match = re.search(r'^(技能|机制|设定|剧情|背景)[：:]', doc)
                    if tag_match:
                        terms.add(doc[:60].strip())
            except Exception:
                pass
        self._known_terms = list(terms)
        return self._known_terms

    def _write_txt(self, speaker: str, name: str, content: str,
                   turn: int, key_terms: list[str]) -> str:
        """将清洗后的对话写入按日期命名的TXT文件，返回文件路径"""
        today = datetime.now().strftime("%Y-%m-%d")
        DIALOGUE_DIR.mkdir(parents=True, exist_ok=True)
        filepath = DIALOGUE_DIR / f"dialogue_{today}.txt"

        timestamp = datetime.now().strftime("%H:%M:%S")
        role_tag = "👤 USER" if speaker == "user" else "🤖 AI"
        terms_str = ", ".join(key_terms[:8]) if key_terms else "无"

        block = (
            f"\n{'─' * 60}\n"
            f"[{timestamp}] Turn #{turn} | {role_tag} | {name}\n"
            f"[关键术语] {terms_str}\n"
            f"{'─' * 60}\n"
            f"{content}\n"
        )

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(block)
        return str(filepath)

    def _generate_batch_file(self, content: str, key_terms: list[str],
                             tag: str = "dialogue") -> str:
        """生成符合 ingest_new.py 格式的批处理条目写入 batch 文件"""
        for term in key_terms[:3]:
            if term and len(term) <= 200:
                line = f"[{tag.upper()}] 对话提及: {term} | {content[:100]}\n"
                with open(BATCH_FILE, "a", encoding="utf-8") as f:
                    f.write(line)
        return str(BATCH_FILE)

    def _embed_and_store(self, clean_text: str, speaker: str,
                         name: str, turn: int, key_terms: list[str]) -> str:
        """将清洗后的对话文本向量化并存入 ChromaDB"""
        embedding = self.model.encode(clean_text).tolist()
        doc_id = str(uuid.uuid4())
        self.dialogue_col.add(
            embeddings=[embedding],
            documents=[clean_text],
            metadatas=[{
                "speaker": speaker,
                "name": name,
                "turn": turn,
                "key_terms": ",".join(key_terms[:10]),
                "timestamp": datetime.now().isoformat(),
                "source": "realtime"
            }],
            ids=[doc_id]
        )
        return doc_id

    def process_turn(self, speaker: str, name: str, content: str,
                     turn: int) -> dict:
        """处理单轮对话的完整管道，返回处理摘要"""
        raw_len = len(content)
        clean_text = self._clean_text(content)
        key_terms = self._extract_key_terms(clean_text)

        txt_path = self._write_txt(speaker, name, clean_text, turn, key_terms)
        doc_id = self._embed_and_store(clean_text, speaker, name, turn, key_terms)

        if key_terms:
            self._generate_batch_file(clean_text, key_terms)

        return {
            "status": "ok",
            "doc_id": doc_id,
            "txt_path": txt_path,
            "cleaned_length": len(clean_text),
            "raw_length": raw_len,
            "key_terms_found": key_terms,
            "turn": turn,
        }

    def process_batch_txt(self, file_path: str = None) -> dict:
        """批量处理 [TYPE] 标记的TXT文件，将条目导入向量库"""
        src = file_path or str(BATCH_FILE)
        if not os.path.exists(src):
            return {"status": "empty", "message": f"文件 {src} 不存在"}

        pattern = re.compile(r'^\[(.+?)\]\s*(.+)')
        chunks, types_list = [], []

        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = pattern.match(line)
                if m:
                    types_list.append(m.group(1).strip().lower())
                    chunks.append(m.group(2).strip())

        if not chunks:
            return {"status": "empty", "message": "文件无有效条目"}

        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {"type": t, "source": os.path.basename(src)} for t in types_list
        ]

        collection = self.client.get_or_create_collection(name=COLLECTION_MEMORY)
        collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

        if file_path:
            with open(src, "w", encoding="utf-8") as f:
                f.write("")

        return {
            "status": "ok",
            "ingested": len(chunks),
            "types": list(set(types_list)),
            "collection": COLLECTION_MEMORY,
            "total_count": collection.count(),
        }

    def get_stats(self) -> dict:
        """获取管道运行统计"""
        stats = {}
        for col_name in [COLLECTION_DIALOGUE, COLLECTION_SKILLS, COLLECTION_MEMORY]:
            try:
                col = self.client.get_collection(name=col_name)
                stats[col_name] = col.count()
            except Exception:
                stats[col_name] = 0

        dialogue_files = list(DIALOGUE_DIR.glob("*.txt")) if DIALOGUE_DIR.exists() else []
        stats["dialogue_txt_files"] = len(dialogue_files)

        return stats


_pipeline_instance: Optional[DialoguePipeline] = None


def get_pipeline() -> DialoguePipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DialoguePipeline()
    return _pipeline_instance
