"""
剧情约束规则引擎：将向量检索结果转化为大模型的剧情引导指令，
动态调整约束策略以避免机械感，并持续优化查询反馈。
"""
import re
import time
from collections import defaultdict
from typing import Optional

from config import (
    MAX_CONSTRAINT_CHARS, MAX_ACTIVE_CONSTRAINTS,
    CONSTRAINT_COOLDOWN_TURNS, DEFAULT_PROFILE, COLLECTION_SKILLS,
    SKILL_PROFICIENCY_PATTERN,
)


class ConstraintEngine:
    """将检索到的记忆转化为自然流畅的剧情约束指令"""

    def __init__(self, profile: str = DEFAULT_PROFILE):
        self._profile = profile
        self._history: defaultdict = defaultdict(list)
        self._cooldowns: dict[str, float] = {}
        self._feedback_weights: dict[str, float] = {
            "skill": 1.5,
            "mechanic": 1.3,
            "setting": 1.0,
            "plot": 1.2,
            "dialogue": 0.7,
        }
        self._last_constraints: list[dict] = []
        self._last_constraint_text: str = ""
        self._all_skills: list[dict] = []

    def set_skills(self, skills: list[dict]):
        self._all_skills = list(skills)

    def load_skills_from_db(self):
        """从 ChromaDB 加载所有技能。如果当前 profile 没有技能，回退到 default profile。"""
        try:
            import chromadb
            from config import get_chroma_path, DEFAULT_PROFILE
            skills = self._do_load_skills(chromadb, get_chroma_path(self._profile))
            if not skills and self._profile != DEFAULT_PROFILE:
                skills = self._do_load_skills(chromadb, get_chroma_path(DEFAULT_PROFILE))
            if skills:
                self._all_skills = skills
        except Exception:
            pass

    @staticmethod
    def _do_load_skills(chromadb, path: str) -> list[dict]:
        import re
        from config import COLLECTION_SKILLS
        try:
            client = chromadb.PersistentClient(path=path)
            col = client.get_collection(name=COLLECTION_SKILLS)
            data = col.get()
            skills = []
            for i in range(len(data["ids"])):
                doc = data["documents"][i]
                meta = data["metadatas"][i]
                if meta.get("type") != "skill":
                    continue
                prof_match = re.search(SKILL_PROFICIENCY_PATTERN, doc)
                if not prof_match:
                    continue
                name_match = re.search(r'(?:技能|技巧)[：:]\s*([^。]+)', doc)
                name = name_match.group(1).strip() if name_match else "未知技能"
                skills.append({"name": name, "proficiency": int(prof_match.group(1))})
            return skills
        except Exception:
            return []

    def generate_constraints(self, search_results: dict,
                             dialogue_context: list[dict]) -> str:
        """根据检索结果生成剧情约束指令字符串"""
        results = search_results.get("results", [])
        has_skills = bool(self._all_skills)

        active = []
        self._last_constraints = []

        for r in results:
            if len(active) >= MAX_ACTIVE_CONSTRAINTS:
                break
            applied = self._apply_cooldown(r)
            if not applied:
                continue
            constraint = self._build_constraint(r, dialogue_context)
            if constraint:
                active.append(constraint)
                self._last_constraints.append({
                    "type": r.get("metadata", {}).get("type", "info"),
                    "content": r.get("document", "")[:120],
                    "score": r.get("score", 0.0),
                    "display": constraint,
                })

        if not active and not has_skills:
            self._last_constraints = []
            self._last_constraint_text = ""
            return ""

        proficiency_guide = self._build_proficiency_guidance(
            results, self._all_skills
        )
        if proficiency_guide:
            self._last_constraints.append({
                "type": "proficiency",
                "content": "技能熟练度追踪",
                "score": 1.0,
                "display": proficiency_guide,
            })

        last_turn = dialogue_context[-1] if dialogue_context else {}
        context_hint = self._context_hint(last_turn)

        header = (
            "[RAG-RPG 剧情约束 - 请在回复中自然地融入以下元素，"
            "不要逐条复述，而是让它们体现在行动与场景中]\n"
        )
        if context_hint:
            header += f"[当前情境提示] {context_hint}\n"

        parts = list(active)
        if proficiency_guide:
            parts.append(proficiency_guide)

        full = header + "\n".join(parts)
        if len(full) > MAX_CONSTRAINT_CHARS:
            full = full[:MAX_CONSTRAINT_CHARS - 20] + "\n[约束已截断]"

        self._last_constraint_text = full.strip()
        return self._last_constraint_text

    def build_proficiency_display(self) -> str:
        """独立生成熟练度展示文本（供管理后台使用）"""
        guide = self._build_proficiency_guidance([], self._all_skills)
        if not guide:
            return ""
        return (
            "[RAG-RPG 剧情约束 - 技能熟练度追踪]\n"
            "[熟练度更新规则] 战斗或训练后请在叙事中附加「熟练度 X/100」。\n"
            f"{guide}"
        )

    def _build_proficiency_guidance(self, results: list[dict],
                                    all_skills: list[dict] = None) -> Optional[str]:
        """检测检索结果及数据库中所有技能条目，生成熟练度自主更新引导。"""
        seen = set()
        skills = []

        for r in results:
            meta = r.get("metadata", {})
            if meta.get("type") == "skill":
                doc = r.get("document", "")
                prof_match = re.search(SKILL_PROFICIENCY_PATTERN, doc)
                if prof_match:
                    current = int(prof_match.group(1))
                    name_match = re.search(r'(?:技能|技巧)[：:]\s*([^。]+)', doc)
                    name = name_match.group(1).strip() if name_match else "未知技能"
                    if name not in seen:
                        seen.add(name)
                        skills.append({"name": name, "proficiency": current})

        for s in (all_skills or []):
            name = s.get("name", "未知技能")
            if name not in seen:
                seen.add(name)
                skills.append(s)

        if not skills:
            return None

        lines = [
            "[熟练度更新规则]",
            "战斗或训练后，你必须在叙事末尾附加一行「熟练度 N/100」（N为具体数字，取当前值或更高值）。格式必须严格为「熟练度 数字/100」。注意：光剑精通当前15/100，里·鬼剑术当前12/100，里鬼节奏控制当前8/100。请使用这些精确数值，不要随意编造。",
            "当前技能熟练度:",
        ]
        for s in skills:
            lines.append(
                f"  · {s['name']} 当前 {s['proficiency']}/100"
            )
        return "\n".join(lines)

    def _build_constraint(self, result: dict,
                          dialogue_context: list[dict]) -> Optional[str]:
        """为单条检索结果构造约束语句"""
        doc = result.get("document", "")
        meta = result.get("metadata", {})
        etype = meta.get("type", "info")
        score = result.get("score", 0.0)
        col = result.get("collection", "")

        doc_clean = self._clean_doc(doc)

        templates = {
            "skill": (
                f"• 技能约束 [{score:.2f}]: 角色拥有技能「{doc_clean[:60]}」，"
                f"请在合适时机展现此技能的效果与限制。"
            ),
            "mechanic": (
                f"• 机制约束 [{score:.2f}]: 当前适用的战斗/系统规则——{doc_clean[:80]}，"
                f"请在场景中体现该机制的运作。"
            ),
            "setting": (
                f"• 世界观约束 [{score:.2f}]: 场景设定——{doc_clean[:80]}，"
                f"请在描述中自然融入此背景信息。"
            ),
            "plot": (
                f"• 剧情约束 [{score:.2f}]: 关键剧情线索——{doc_clean[:80]}，"
                f"请将此处埋下的伏笔在合适的时机自然地推进。"
            ),
            "dialogue": (
                f"• 记忆回调 [{score:.2f}]: 历史对话——{doc_clean[:80]}，"
                f"请在不突兀的前提下提及此过往。"
            ),
        }

        template = templates.get(etype)
        if not template:
            if col == COLLECTION_NAME_MAP.get("dialogue"):
                template = templates["dialogue"]
            else:
                template = (
                    f"• 相关信息 [{score:.2f}]: {doc_clean[:100]}，"
                    f"请在适当时候自然地关联此信息。"
                )

        return template

    def _clean_doc(self, doc: str) -> str:
        """清理文档内容为单行摘要"""
        doc = re.sub(r'\s+', ' ', doc)
        doc = re.sub(r'\[type:\w+\]', '', doc)
        return doc.strip()

    def _context_hint(self, last_turn: dict) -> str:
        """从最后一轮对话中提取情境提示"""
        if not last_turn:
            return ""
        content = last_turn.get("content", "")[:100]
        content = re.sub(r'[\*\n]', ' ', content).strip()
        if len(content) > 15:
            return f"当前对话焦点: 「{content}...」"
        return ""

    def _apply_cooldown(self, result: dict) -> bool:
        """检查并应用冷却时间，避免同一约束短时间重复出现"""
        doc_id = result.get("id", "")
        now = time.time()
        if doc_id in self._cooldowns:
            if now - self._cooldowns[doc_id] < CONSTRAINT_COOLDOWN_TURNS * 10:
                return False
        self._cooldowns[doc_id] = now
        if len(self._cooldowns) > 500:
            expired = [
                k for k, v in self._cooldowns.items()
                if now - v > 300
            ]
            for k in expired:
                del self._cooldowns[k]
        return True

    def update_feedback(self, entry_type: str, was_used: bool):
        """根据AI是否实际使用了某类约束来调整权重"""
        delta = 0.1 if was_used else -0.05
        current = self._feedback_weights.get(entry_type, 1.0)
        self._feedback_weights[entry_type] = max(0.3, min(3.0, current + delta))

    def get_weight(self, entry_type: str) -> float:
        return self._feedback_weights.get(entry_type, 1.0)

    def get_active_constraints(self) -> list[dict]:
        return list(self._last_constraints)

    def get_last_constraint_text(self) -> str:
        return self._last_constraint_text

    def get_display_text(self) -> str:
        """生成管理后台展示文本。当没有检索结果时，自动用已加载的技能生成熟练度展示。"""
        constraints = list(self._last_constraints)
        if not constraints:
            if self._all_skills:
                prof_text = self._build_proficiency_guidance([], self._all_skills)
                if prof_text:
                    constraints = [{
                        "type": "proficiency",
                        "display": prof_text,
                        "score": 1.0,
                    }]

        if not constraints:
            return ""

        lines = ["━━ 当前剧情约束 ━━"]
        for c in constraints:
            tname = {"skill": "技能", "mechanic": "机制", "setting": "设定",
                     "plot": "剧情", "dialogue": "记忆",
                     "proficiency": "熟练度"}.get(c["type"], c["type"])
            if c["type"] == "proficiency":
                for line in c["display"].split("\n"):
                    lines.append(f"  {line}")
            else:
                lines.append(f"  [{tname} 相关度:{c['score']:.2f}] {c['content'][:60]}...")
        return "\n".join(lines)


COLLECTION_NAME_MAP = {
    "character_skills": "skill_library",
    "my_rag_memory": "memory_library",
    "dialogue_memory": "dialogue",
}

_constraint_engine_instances: dict[str, ConstraintEngine] = {}


def get_constraint_engine(profile: str = DEFAULT_PROFILE) -> ConstraintEngine:
    if profile not in _constraint_engine_instances:
        engine = ConstraintEngine(profile=profile)
        engine.load_skills_from_db()
        _constraint_engine_instances[profile] = engine
    return _constraint_engine_instances[profile]
