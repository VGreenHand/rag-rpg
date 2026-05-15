"""
对话处理管道：接收原始对话 → 清洗 → 提取关键信息 → 写入TXT → 向量入库
v3.1: 向量库只存核心对话内容（过滤叙事描写），TXT 保留完整原文
"""
import re
import uuid
import os
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable

from config import (
    CHROMA_PATH, COLLECTION_DIALOGUE, COLLECTION_SKILLS,
    MODEL_NAME, DIALOGUE_DIR, BATCH_FILE, COLLECTION_MEMORY,
    DEDUP_SIMILARITY_THRESHOLD,
    SKILL_PROFICIENCY_PATTERN,
    get_chroma_path, get_dialogue_dir, get_batch_file, DEFAULT_PROFILE,
)
from rag_rpg.infrastructure.chroma.client import ChromaClient
from checkpoint_manager import get_checkpoint, TimeoutError as CkpTimeoutError
from embedding_client import get_embedding_client
from rag_rpg.common.utils import SafeTimer, SingletonFactory

logger = logging.getLogger("rag-rpg.pipeline")

EMBED_TIMEOUT = 15.0
CHROMA_TIMEOUT = 8.0
FILE_IO_TIMEOUT = 5.0
TERM_LOAD_LIMIT = 2000
SUMMARY_MAX_CHARS = 800


class DialoguePipeline:
    def __init__(self, profile: str = DEFAULT_PROFILE):
        self._profile = profile
        self.model = get_embedding_client()
        self._chroma = ChromaClient(path=get_chroma_path(profile))
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

    def _extract_core_content(self, text: str) -> str:
        """从对话文本中提取核心信息（用于向量存储），
        优先保留「」内对话内容（信息密度最高），过滤（）内叙事描写。"""
        if not text:
            return ""
        if len(text) <= 120:
            return text

        parts = []
        depth = 0
        current = []
        for ch in text:
            if ch == '「':
                if depth > 0:
                    current.append(ch)
                depth += 1
            elif ch == '」':
                depth -= 1
                if depth == 0:
                    candidate = "".join(current).strip()
                    if len(candidate) > 2:
                        parts.append(candidate)
                    current = []
                else:
                    current.append(ch)
            elif depth > 0:
                current.append(ch)

        if not parts:
            return text[:SUMMARY_MAX_CHARS]

        if current:
            candidate = "".join(current).strip()
            if len(candidate) > 2:
                parts.append(candidate)

        result = "；".join(p for p in parts if len(p) > 2)
        if not result:
            return text[:SUMMARY_MAX_CHARS]

        if len(result) <= SUMMARY_MAX_CHARS:
            return result

        sentences = re.split(r'(?<=[。！？；])', result)
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
        return "".join(kept) if kept else result[:SUMMARY_MAX_CHARS]

    # ─── 功能1: 记忆去重 ──────────────────────────

    def _is_duplicate(self, text_embedding: list[float],
                      threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> bool:
        try:
            results = SafeTimer.run(
                lambda: self._chroma.get_collection(COLLECTION_DIALOGUE).query(
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
                col = self._chroma.client.get_collection(name=col_name)
                count = col.count()
                if count == 0 and col_name == COLLECTION_SKILLS and self._profile != DEFAULT_PROFILE:
                    try:
                        fallback_chroma = ChromaClient(path=get_chroma_path(DEFAULT_PROFILE))
                        col = fallback_chroma.client.get_collection(name=col_name)
                        count = col.count()
                    except Exception:
                        pass
                if count == 0:
                    continue
                results = SafeTimer.run(col.get, CHROMA_TIMEOUT, limit=TERM_LOAD_LIMIT)
                if results is None:
                    continue
                for meta in results.get("metadatas", []):
                    ek = meta.get("entry_key", "")
                    if ek:
                        terms.add(ek)
                        bare = ek.replace('\u00b7', '').replace('\u2022', '').replace('·', '')
                        if bare != ek:
                            terms.add(bare)
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

    def _detect_all_proficiencies(self, text: str,
                                  key_terms: list[str]) -> list[dict]:
        """从文本中找出所有「熟练度 N/100」，为每个技能生成单独的更新任务。
        使用原始文本（txt_ready）而非核心提取后的文本，因为熟练度行通常在「」之外。"""
        if not text:
            return []
        updates = []
        for match in re.finditer(SKILL_PROFICIENCY_PATTERN, text):
            proficiency = int(match.group(1))
            match_pos = match.start()
            preceding = text[max(0, match_pos - 60):match_pos]
            matched_term = None
            for term in sorted(key_terms, key=len, reverse=True):
                if term in preceding:
                    matched_term = term
                    break
            if not matched_term:
                continue
            entry_key = self._match_entry_key([matched_term])
            if entry_key:
                updates.append({"entry_key": entry_key, "proficiency": proficiency})
        return updates

    def _match_entry_key(self, key_terms: list[str]) -> Optional[str]:
        col = self._chroma.get_collection(COLLECTION_SKILLS)
        for term in key_terms:
            try:
                results = col.get(where={"entry_key": term})
                if results and results.get("ids"):
                    return term
            except Exception:
                continue
        db_data = None
        for term in key_terms:
            try:
                if db_data is None:
                    db_data = col.get()
                stripped_term = self._strip_dot(term)
                for i in range(len(db_data["ids"])):
                    ek = db_data["metadatas"][i].get("entry_key", "")
                    if self._strip_dot(ek) == stripped_term:
                        return ek
            except Exception:
                continue
        if self._profile != DEFAULT_PROFILE:
            try:
                fallback_chroma = ChromaClient(path=get_chroma_path(DEFAULT_PROFILE))
                fallback_col = fallback_chroma.client.get_collection(name=COLLECTION_SKILLS)
                for term in key_terms:
                    try:
                        results = fallback_col.get(where={"entry_key": term})
                        if results and results.get("ids"):
                            return term
                    except Exception:
                        continue
                fb_data = fallback_col.get()
                stripped_term = self._strip_dot(key_terms[0]) if key_terms else ""
                for i in range(len(fb_data["ids"])):
                    ek = fb_data["metadatas"][i].get("entry_key", "")
                    if self._strip_dot(ek) == stripped_term:
                        return ek
            except Exception:
                pass
        return None

    @staticmethod
    def _strip_dot(s: str) -> str:
        return s.replace('\u00b7', '').replace('\u2022', '').replace('·', '')

    def _update_skill_proficiency(self, entry_key: str,
                                  proficiency: int) -> bool:
        for col, label in [(self._chroma.get_collection(COLLECTION_SKILLS), self._profile), (None, "default")]:
            try:
                if col is None and self._profile != DEFAULT_PROFILE:
                    fallback_chroma = ChromaClient(path=get_chroma_path(DEFAULT_PROFILE))
                    col = fallback_chroma.client.get_collection(name=COLLECTION_SKILLS)
                elif col is None:
                    break
                results = col.get(where={"entry_key": entry_key})
                if not results or not results.get("ids"):
                    if label == "default":
                        return False
                    continue
                old_id = results["ids"][0]
                old_doc = results["documents"][0]
                new_doc = re.sub(
                    SKILL_PROFICIENCY_PATTERN,
                    f"熟练度 {proficiency}/100",
                    old_doc
                )
                new_emb = SafeTimer.run(
                    self.model.encode, EMBED_TIMEOUT, new_doc
                ).tolist()
                SafeTimer.run(
                    lambda c=col: c.update(
                        ids=[old_id],
                        documents=[new_doc],
                        embeddings=[new_emb]
                    ),
                    CHROMA_TIMEOUT
                )
                logger.info(f"熟练度更新: {entry_key} → {proficiency}/100 ({label})")
                self.invalidate_terms_cache()
                return True
            except Exception as e:
                logger.error(f"熟练度更新失败 ({label}): {e}")
                return False
        return False

    # ─── TXT写入 ─────────────────────────────────

    def _write_txt(self, speaker: str, name: str, content: str,
                   turn: int, key_terms: list[str],
                   session_id: str = None) -> str:
        today = datetime.now().strftime("%Y-%m-%d")
        dialogue_dir = get_dialogue_dir(self._profile)

        if session_id:
            dialogue_dir = dialogue_dir / today
            dialogue_dir.mkdir(parents=True, exist_ok=True)
            safe_sid = "".join(c if c.isalnum() or c in "_-" else "_" for c in session_id)
            filepath = dialogue_dir / f"session_{safe_sid}.txt"
        else:
            dialogue_dir.mkdir(parents=True, exist_ok=True)
            filepath = dialogue_dir / f"dialogue_{today}.txt"

        timestamp = datetime.now().strftime("%H:%M:%S")
        role_tag = "USER" if speaker == "user" else "AI"
        terms_str = ", ".join(key_terms[:8]) if key_terms else "-"

        is_new_file = not filepath.exists()
        if is_new_file and session_id:
            init_block = (
                f"{'═' * 60}\n"
                f"会话开始 | ID: {session_id}\n"
                f"角色: {name if speaker == 'ai' else '用户'}\n"
                f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"{'═' * 60}\n\n"
            )
            def _write_init():
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(init_block)
            SafeTimer.run(_write_init, FILE_IO_TIMEOUT)

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
            self._chroma.get_collection(COLLECTION_DIALOGUE).add(
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
                     turn: int, session_id: str | None = None) -> dict:
        raw_len = len(content)

        try:
            clean_text = self._clean_text(content)
        except Exception:
            clean_text = content[:500]

        txt_ready = clean_text

        clean_text = self._extract_core_content(clean_text)

        key_terms = self._extract_key_terms(txt_ready)

        skill_updates = self._detect_all_proficiencies(txt_ready, key_terms)
        for update in skill_updates:
            self._update_skill_proficiency(
                update["entry_key"],
                update["proficiency"]
            )
            self._report(
                f"熟练度更新: {update['entry_key']} "
                f"→ {update['proficiency']}/100"
            )

        txt_path = ""
        try:
            txt_path = self._write_txt(speaker, name, txt_ready, turn, key_terms, session_id)
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
            "skill_updated": len(skill_updates) > 0,
            "skill_updates": len(skill_updates),
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
                collection = self._chroma.client.get_or_create_collection(
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

        collection = self._chroma.client.get_or_create_collection(name=COLLECTION_MEMORY)
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
                col = self._chroma.client.get_collection(name=col_name)
                stats[col_name] = col.count()
            except Exception:
                stats[col_name] = 0
        dialogue_dir = get_dialogue_dir(self._profile)
        dialogue_files = (
            list(dialogue_dir.glob("*.txt")) if dialogue_dir.exists() else []
        )
        stats["dialogue_txt_files"] = len(dialogue_files)
        return stats


get_pipeline = SingletonFactory(DialoguePipeline)
