"""
对话处理管道：接收原始对话 → 清洗 → 提取关键信息 → 写入TXT → 向量入库
v3.0: 记忆去重 + 智能压缩 + 技能熟练度覆盖更新
"""
import re
import uuid
import os
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PATH, COLLECTION_DIALOGUE, COLLECTION_SKILLS,
    MODEL_NAME, DIALOGUE_DIR, BATCH_FILE, COLLECTION_MEMORY,
    DEDUP_SIMILARITY_THRESHOLD, SUMMARY_MAX_CHARS,
    SKILL_PROFICIENCY_PATTERN,
    get_chroma_path, get_dialogue_dir, get_batch_file, DEFAULT_PROFILE,
)
from checkpoint_manager import get_checkpoint, TimeoutError as CkpTimeoutError

logger = logging.getLogger("rag-rpg.pipeline")

EMBED_TIMEOUT = 15.0
CHROMA_TIMEOUT = 8.0
FILE_IO_TIMEOUT = 5.0
TERM_LOAD_LIMIT = 2000


class SafeTimer:
    @staticmethod
    def run(func: Callable, timeout: float, *args, **kwargs):
        result_holder = []
        exc_holder = []

        def target():
            try:
                result_holder.append(func(*args, **kwargs))
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise CkpTimeoutError(f"操作超时 ({timeout}s)")
        if exc_holder:
            raise exc_holder[0]
        return result_holder[0] if result_holder else None


class DialoguePipeline:
    def __init__(self, profile: str = DEFAULT_PROFILE):
        self._profile = profile
        self.model = SentenceTransformer(MODEL_NAME)
        self.client = chromadb.PersistentClient(path=get_chroma_path(profile))
        self.dialogue_col = self.client.get_or_create_collection(
            name=COLLECTION_DIALOGUE,
            metadata={"hnsw:space": "cosine"}
        )
        self.skill_col = self.client.get_or_create_collection(
            name=COLLECTION_SKILLS,
            metadata={"hnsw:space": "cosine"}
        )
        self._known_terms: Optional[list] = None
        self._terms_loading = False
        self._terms_lock = threading.Lock()
        self._on_progress: Optional[Callable] = None

    def set_progress_callback(self, cb: Callable):
        self._on_progress = cb

    def _report(self, message: str):
        logger.info(message)
        if self._on_progress:
            try:
                self._on_progress(message)
            except Exception:
                pass

    # ─── 文本清洗 ────────────────────────────────

    def _clean_text(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\*{1,3}', '', text)
        text = re.sub(r'[_\*~]{1,2}', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'^\s*[-–—]\s*', '', text, flags=re.MULTILINE)
        return text.strip()

    # ─── 功能1: 智能摘要 ──────────────────────────

    def _summarize_text(self, text: str) -> str:
        if len(text) <= SUMMARY_MAX_CHARS:
            return text
        sentences = re.split(r'(?<=[。！？])', text)
        kept = []
        current_len = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if current_len + len(s) > SUMMARY_MAX_CHARS:
                break
            kept.append(s)
            current_len += len(s)
        result = "".join(kept)
        if len(result) < len(text) * 0.3:
            return text[:SUMMARY_MAX_CHARS]
        return result

    # ─── 功能1: 记忆去重 ──────────────────────────

    def _is_duplicate(self, text_embedding: list[float],
                      threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> bool:
        try:
            results = SafeTimer.run(
                lambda: self.dialogue_col.query(
                    query_embeddings=[text_embedding],
                    n_results=1
                ),
                CHROMA_TIMEOUT
            )
            if results and results["distances"] and results["distances"][0]:
                min_dist = results["distances"][0][0]
                similarity = 1.0 - min_dist
                if similarity >= threshold:
                    return True
            return False
        except Exception:
            return False

    # ─── 关键术语提取 ─────────────────────────────

    def _extract_key_terms(self, text: str) -> list[str]:
        if not text:
            return []
        found = set()
        terms = self._load_known_terms()
        for term in terms:
            if term in text:
                found.add(term)
        return sorted(found, key=lambda t: len(t), reverse=True)

    def _load_known_terms(self) -> list[str]:
        if self._known_terms is not None:
            return self._known_terms
        if self._terms_loading:
            return []
        with self._terms_lock:
            if self._known_terms is not None:
                return self._known_terms
            self._terms_loading = True
        try:
            self._known_terms = self._load_terms_safe()
        finally:
            self._terms_loading = False
        return self._known_terms

    def _load_terms_safe(self) -> list[str]:
        terms = set()
        for col_name in (COLLECTION_SKILLS, COLLECTION_MEMORY):
            try:
                col = self.client.get_collection(name=col_name)
                count = col.count()
                if count == 0:
                    continue
                results = SafeTimer.run(col.get, CHROMA_TIMEOUT, limit=TERM_LOAD_LIMIT)
                for meta in results.get("metadatas", []):
                    ek = meta.get("entry_key", "")
                    if ek:
                        terms.add(ek)
                for doc in results.get("documents", []):
                    tag_match = re.search(r'^(技能|机制|设定|剧情|背景)[：:]', doc)
                    if tag_match:
                        terms.add(doc[:60].strip())
            except CkpTimeoutError:
                logger.warning(f"术语加载超时: {col_name}")
            except Exception as e:
                logger.debug(f"跳过术语加载 {col_name}: {e}")
        return list(terms)

    def invalidate_terms_cache(self):
        with self._terms_lock:
            self._known_terms = None

    # ─── 功能2: 技能熟练度检测与覆盖更新 ────────────

    def _detect_skill_proficiency(self, text: str,
                                  key_terms: list[str]) -> Optional[dict]:
        match = re.search(SKILL_PROFICIENCY_PATTERN, text)
        if not match:
            return None
        proficiency = int(match.group(1))
        entry_key = None
        for term in key_terms:
            try:
                col = self.skill_col
                results = col.get(where={"entry_key": term})
                if results["ids"]:
                    entry_key = term
                    break
            except Exception:
                continue
        if not entry_key:
            return None
        return {"entry_key": entry_key, "proficiency": proficiency}

    def _update_skill_proficiency(self, entry_key: str,
                                  proficiency: int) -> bool:
        try:
            results = self.skill_col.get(where={"entry_key": entry_key})
            if not results["ids"]:
                return False
            old_id = results["ids"][0]
            old_doc = results["documents"][0]
            old_meta = results["metadatas"][0]
            new_doc = re.sub(
                SKILL_PROFICIENCY_PATTERN,
                f"熟练度 {proficiency}/100",
                old_doc
            )
            new_emb = SafeTimer.run(
                self.model.encode, EMBED_TIMEOUT, new_doc
            ).tolist()
            SafeTimer.run(
                lambda: self.skill_col.update(
                    ids=[old_id],
                    documents=[new_doc],
                    embeddings=[new_emb]
                ),
                CHROMA_TIMEOUT
            )
            logger.info(f"熟练度更新: {entry_key} → {proficiency}/100")
            self.invalidate_terms_cache()
            return True
        except Exception as e:
            logger.error(f"熟练度更新失败: {e}")
            return False

    # ─── TXT写入 ─────────────────────────────────

    def _write_txt(self, speaker: str, name: str, content: str,
                   turn: int, key_terms: list[str]) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        dialogue_dir = get_dialogue_dir(self._profile)
        dialogue_dir.mkdir(parents=True, exist_ok=True)
        filepath = dialogue_dir / f"dialogue_{today}.txt"

        timestamp = datetime.now().strftime("%H:%M:%S")
        role_tag = "USER" if speaker == "user" else "AI"
        terms_str = ", ".join(key_terms[:8]) if key_terms else "-"

        block = (
            f"\n{'─' * 60}\n"
            f"[{timestamp}] Turn #{turn} | {role_tag} | {name}\n"
            f"[关键术语] {terms_str}\n"
            f"{'─' * 60}\n"
            f"{content}\n"
        )

        def _write():
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(block)
        SafeTimer.run(_write, FILE_IO_TIMEOUT)
        return str(filepath)

    def _generate_batch_file(self, content: str, key_terms: list[str],
                             tag: str = "dialogue") -> str:
        for term in key_terms[:3]:
            if term and len(term) <= 200:
                line = f"[{tag.upper()}] 对话提及: {term} | {content[:100]}\n"

                def _append():
                    batch_file = get_batch_file(self._profile)
                    with open(batch_file, "a", encoding="utf-8") as f:
                        f.write(line)
                try:
                    SafeTimer.run(_append, FILE_IO_TIMEOUT)
                except CkpTimeoutError:
                    pass
        return str(get_batch_file(self._profile))

    # ─── 向量入库（含去重检查） ────────────────────

    def _embed_and_store(self, clean_text: str, speaker: str,
                         name: str, turn: int, key_terms: list[str]) -> dict:
        embedding = SafeTimer.run(
            self.model.encode, EMBED_TIMEOUT, clean_text
        ).tolist()

        duplicate = self._is_duplicate(embedding)
        if duplicate:
            logger.info(f"去重跳过 Turn#{turn} (相似度>{DEDUP_SIMILARITY_THRESHOLD})")
            return {"doc_id": "duplicate_skipped", "duplicate": True}

        doc_id = str(uuid.uuid4())

        def _add():
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
        SafeTimer.run(_add, CHROMA_TIMEOUT)
        return {"doc_id": doc_id, "duplicate": False}

    # ─── 核心处理流程 ─────────────────────────────

    def process_turn(self, speaker: str, name: str, content: str,
                     turn: int) -> dict:
        raw_len = len(content)

        try:
            clean_text = self._clean_text(content)
        except Exception:
            clean_text = content[:500]

        clean_text = self._summarize_text(clean_text)

        if clean_text:
            key_terms = self._extract_key_terms(clean_text)
        else:
            key_terms = []

        skill_update = self._detect_skill_proficiency(clean_text, key_terms)
        if skill_update:
            self._update_skill_proficiency(
                skill_update["entry_key"],
                skill_update["proficiency"]
            )
            self._report(
                f"熟练度更新: {skill_update['entry_key']} "
                f"→ {skill_update['proficiency']}/100"
            )

        txt_path = ""
        try:
            txt_path = self._write_txt(speaker, name, clean_text, turn, key_terms)
        except CkpTimeoutError:
            logger.warning(f"TXT写入超时 Turn#{turn}")
        except Exception as e:
            logger.error(f"TXT写入失败 Turn#{turn}: {e}")

        store_result = {"doc_id": "", "duplicate": False}
        try:
            store_result = self._embed_and_store(
                clean_text, speaker, name, turn, key_terms
            )
        except CkpTimeoutError:
            logger.error(f"向量入库超时 Turn#{turn}")
        except Exception as e:
            logger.error(f"向量入库失败 Turn#{turn}: {e}")

        try:
            if key_terms and not store_result.get("duplicate"):
                self._generate_batch_file(clean_text, key_terms)
        except Exception:
            pass

        status = "ok"
        if store_result.get("duplicate"):
            status = "duplicate_skipped"
        elif not store_result.get("doc_id"):
            status = "store_failed"

        self._report(
            f"Turn#{turn} | {speaker} | "
            f"clean={len(clean_text)} terms={len(key_terms)} "
            f"status={status}"
        )

        return {
            "status": status,
            "doc_id": store_result.get("doc_id", ""),
            "txt_path": txt_path,
            "cleaned_length": len(clean_text),
            "raw_length": raw_len,
            "key_terms_found": key_terms,
            "turn": turn,
            "skill_updated": bool(skill_update),
        }

    # ─── 断点续执行的批处理 ─────────────────────

    def process_batch_txt(self, file_path: str = None,
                          resume: bool = False) -> dict:
        src = file_path or str(get_batch_file(self._profile))
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

        cp = get_checkpoint(self._profile)
        if resume and cp.can_resume():
            progress = cp.get_progress()
            start_idx = progress["stats"]["total_processed"]
            self._report(f"断点续执行: 从 #{start_idx} 开始")
        else:
            step_defs = [
                {"id": f"batch_{i}", "name": f"入库条目 #{i}"}
                for i in range(len(chunks))
            ]
            cp.init_task(
                task_type="batch_ingest",
                task_params={"file": src, "total_items": len(chunks)},
                step_definitions=step_defs,
            )
            start_idx = 0

        batch_size = 50
        succeeded = 0
        failed = 0

        for i in range(start_idx, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            batch_chunks = chunks[i:batch_end]
            batch_types = types_list[i:batch_end]

            try:
                embeddings = SafeTimer.run(
                    self.model.encode, EMBED_TIMEOUT * 2,
                    batch_chunks
                ).tolist()
            except CkpTimeoutError:
                self._report(f"批量编码超时 offset={i}")
                for j in range(i, batch_end):
                    cp.mark_step_failed(j, "批量编码超时")
                    failed += 1
                continue

            ids = [str(uuid.uuid4()) for _ in batch_chunks]
            metadatas = [
                {"type": t, "source": os.path.basename(src)}
                for t in batch_types
            ]

            def _add_batch():
                collection = self.client.get_or_create_collection(
                    name=COLLECTION_MEMORY
                )
                collection.add(
                    embeddings=embeddings,
                    documents=batch_chunks,
                    metadatas=metadatas,
                    ids=ids
                )

            try:
                SafeTimer.run(_add_batch, CHROMA_TIMEOUT * 2)
                for j in range(i, batch_end):
                    cp.mark_step_success(j)
                succeeded += len(batch_chunks)
                self._report(f"批量入库 {i+1}-{batch_end}/{len(chunks)} 完成")
            except CkpTimeoutError:
                for j in range(i, batch_end):
                    cp.mark_step_failed(j, "ChromaDB写入超时")
                failed += len(batch_chunks)
            except Exception as e:
                for j in range(i, batch_end):
                    cp.mark_step_failed(j, str(e))
                failed += len(batch_chunks)

        status = "completed" if failed == 0 else "partial"
        cp.complete_task(status)

        if file_path and failed == 0:
            try:
                with open(src, "w", encoding="utf-8") as f:
                    f.write("")
            except Exception:
                pass

        collection = self.client.get_or_create_collection(name=COLLECTION_MEMORY)
        return {
            "status": status,
            "ingested": succeeded,
            "failed": failed,
            "types": list(set(types_list)),
            "collection": COLLECTION_MEMORY,
            "total_count": collection.count(),
            "resume_available": cp.can_resume(),
        }

    def get_stats(self) -> dict:
        stats = {}
        for col_name in [COLLECTION_DIALOGUE, COLLECTION_SKILLS, COLLECTION_MEMORY]:
            try:
                col = self.client.get_collection(name=col_name)
                stats[col_name] = col.count()
            except Exception:
                stats[col_name] = 0
        dialogue_dir = get_dialogue_dir(self._profile)
        dialogue_files = (
            list(dialogue_dir.glob("*.txt")) if dialogue_dir.exists() else []
        )
        stats["dialogue_txt_files"] = len(dialogue_files)
        return stats


_pipeline_instances: dict[str, DialoguePipeline] = {}


def get_pipeline(profile: str = DEFAULT_PROFILE) -> DialoguePipeline:
    if profile not in _pipeline_instances:
        _pipeline_instances[profile] = DialoguePipeline(profile)
    return _pipeline_instances[profile]
