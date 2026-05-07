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
    CONSTRAINT_COOLDOWN_TURNS,
)


class ConstraintEngine:
    """将检索到的记忆转化为自然流畅的剧情约束指令"""

    def __init__(self):
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

    def generate_constraints(self, search_results: dict,
                             dialogue_context: list[dict]) -> str:
        """根据检索结果生成剧情约束指令字符串"""
        results = search_results.get("results", [])
        if not results:
            self._last_constraints = []
            return ""

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

        if not active:
            self._last_constraints = []
            return ""

        last_turn = dialogue_context[-1] if dialogue_context else {}
        context_hint = self._context_hint(last_turn)

        header = (
            "[RAG-RPG 剧情约束 - 请在回复中自然地融入以下元素，"
            "不要逐条复述，而是让它们体现在行动与场景中]\n"
        )
        if context_hint:
            header += f"[当前情境提示] {context_hint}\n"

        full = header + "\n".join(active)
        if len(full) > MAX_CONSTRAINT_CHARS:
            full = full[:MAX_CONSTRAINT_CHARS - 20] + "\n[约束已截断]"

        return full.strip()

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
        # 限制冷却记录数量
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

    def get_display_text(self) -> str:
        if not self._last_constraints:
            return ""
        lines = ["━━ 当前剧情约束 ━━"]
        for c in self._last_constraints:
            tname = {"skill": "技能", "mechanic": "机制", "setting": "设定",
                     "plot": "剧情", "dialogue": "记忆"}.get(c["type"], c["type"])
            lines.append(f"  [{tname} 相关度:{c['score']:.2f}] {c['content'][:60]}...")
        return "\n".join(lines)


COLLECTION_NAME_MAP = {
    "character_skills": "skill_library",
    "my_rag_memory": "memory_library",
    "dialogue_memory": "dialogue",
}

_constraint_engine_instance: Optional[ConstraintEngine] = None


def get_constraint_engine() -> ConstraintEngine:
    global _constraint_engine_instance
    if _constraint_engine_instance is None:
        _constraint_engine_instance = ConstraintEngine()
    return _constraint_engine_instance
