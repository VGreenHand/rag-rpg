"""
全量数据清理 + 验证脚本
1. 清空所有 ChromaDB 集合
2. 删除所有对话 TXT 和 batch 文件
3. 验证所有数据已清除
4. 验证系统可正常接收新数据
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer
from config import (
    CHROMA_PATH, COLLECTION_SKILLS, COLLECTION_MEMORY,
    COLLECTION_DIALOGUE, COLLECTION_PLOT_STATE,
    DIALOGUE_DIR, MODEL_NAME,
)
from pipeline import get_pipeline

CHROMA_DIR = Path(CHROMA_PATH)
DIALOGUES_DIR = Path(DIALOGUE_DIR)
BATCH_FILE = Path(__file__).parent.parent / "new_batch.txt"
ALL_COLLECTIONS = [COLLECTION_SKILLS, COLLECTION_MEMORY, COLLECTION_DIALOGUE, COLLECTION_PLOT_STATE]

print("=" * 60)
print("  RAG-RPG 全量数据清理")
print("=" * 60)

# ═══════════════════════════════════════════════
# 第一步：清空 ChromaDB 所有集合
# ═══════════════════════════════════════════════
print("\n[1/4] 清空 ChromaDB 集合...")

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
before_stats = {}

for col_name in ALL_COLLECTIONS:
    try:
        col = client.get_collection(col_name)
        count = col.count()
        before_stats[col_name] = count
        if count > 0:
            ids = col.get()["ids"]
            col.delete(ids=ids)
            print(f"  {col_name}: {count} 条 → 已清空")
        else:
            print(f"  {col_name}: 0 条 (无需清理)")
    except Exception:
        print(f"  {col_name}: 集合不存在 (跳过)")

# ═══════════════════════════════════════════════
# 第二步：删除所有 TXT 对话记录和 batch 文件
# ═══════════════════════════════════════════════
print("\n[2/4] 清除 TXT 对话记录...")

txt_files = list(DIALOGUES_DIR.glob("dialogue_*.txt")) + list(DIALOGUES_DIR.rglob("session_*.txt"))
for f in txt_files:
    f.unlink()
    print(f"  已删除: {f.name}")

if BATCH_FILE.exists():
    BATCH_FILE.unlink()
    print(f"  已删除: new_batch.txt")
else:
    print(f"  new_batch.txt 不存在 (跳过)")

if not txt_files and not BATCH_FILE.exists():
    print("  无文件需清理")

# ═══════════════════════════════════════════════
# 第三步：验证
# ═══════════════════════════════════════════════
print("\n[3/4] 验证清理结果...")

all_clear = True

for col_name in ALL_COLLECTIONS:
    try:
        col = client.get_collection(col_name)
        count = col.count()
        if count > 0:
            print(f"  ❌ {col_name}: 仍存在 {count} 条 (清理失败)")
            all_clear = False
        else:
            print(f"  ✅ {col_name}: 0 条 (已清空)")
    except Exception:
        print(f"  ✅ {col_name}: 不存在")

remaining_txt = list(DIALOGUES_DIR.glob("dialogue_*.txt")) + list(DIALOGUES_DIR.rglob("session_*.txt"))
if remaining_txt:
    for f in remaining_txt:
        print(f"  ❌ TXT: {f.name} 未删除")
    all_clear = False
else:
    print(f"  ✅ TXT 记录: 已全部清除")

if BATCH_FILE.exists():
    print(f"  ❌ new_batch.txt 未删除")
    all_clear = False
else:
    print(f"  ✅ new_batch.txt: 不存在")

if all_clear:
    print("\n  🎉 所有数据已清理完毕，数据库恢复至初始状态。")
else:
    print("\n  ⚠️ 部分数据清理失败，请检查上方错误信息。")

# ═══════════════════════════════════════════════
# 第四步：验证系统可正常接收新数据
# ═══════════════════════════════════════════════
print("\n[4/4] 验证系统可接收新数据...")

pipeline = get_pipeline()
result = pipeline.process_turn(
    speaker="user",
    name="测试用户",
    content="这是一条清理后的测试消息，用于验证系统是否正常工作。",
    turn=1,
    session_id="cleanup_verify_test",
)

print(f"  处理状态: {result['status']}")
print(f"  doc_id: {result['doc_id'][:36]}...")
print(f"  清洗后长度: {result['cleaned_length']}")
print(f"  关键术语: {result['key_terms_found']}")

col = client.get_collection(COLLECTION_DIALOGUE)
print(f"  dialogue_memory 条目数: {col.count()}")

results = col.get(include=["documents", "metadatas"], limit=5)
for doc, meta in zip(results["documents"], results["metadatas"]):
    print(f"    已存储: [{meta['speaker']}] {doc[:60]}...")

# 清理测试数据
ids_to_remove = results["ids"]
if ids_to_remove:
    col.delete(ids=ids_to_remove)

# 删除测试生成的 TXT
test_txt = Path(result["txt_path"])
if test_txt.exists():
    test_txt.unlink()

print(f"\n  测试数据已清理。")
print(f"\n  ✅ 系统可正常接收和处理新数据。")

# ═══════════════════════════════════════════════
# 清理总结
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("  清理总结")
print("=" * 60)
for col_name, count in before_stats.items():
    print(f"  {col_name}: {count} 条记录已删除")
print(f"  TXT 文件: {len(txt_files)} 个已删除")
print(f"  所有表已清空，系统状态: 初始")
print("=" * 60)
