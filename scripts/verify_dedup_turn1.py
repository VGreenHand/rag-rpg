import chromadb
import json
from sentence_transformers import SentenceTransformer
from collections import Counter

MODEL_NAME = "shibing624/text2vec-base-chinese"
DEDUP_SIMILARITY_THRESHOLD = 0.92

print("=" * 60)
print("  Turn #1 去重验证脚本")
print("=" * 60)

client = chromadb.PersistentClient(path="./chroma_db")

print("\n[1] 集合列表:")
all_collections = client.list_collections()
for col_obj in all_collections:
    col_name = col_obj if isinstance(col_obj, str) else col_obj.name
    col = client.get_collection(col_name)
    print(f"    {col_name} → {col.count()} 条")

print("\n[2] dialogue_memory 集合诊断")

col = client.get_collection("dialogue_memory")
total_origin = col.count()
print(f"    总条目数: {total_origin}")

results = col.get(include=["documents", "metadatas"])
if not results["ids"]:
    print("    ⚠️ 集合为空，无数据可验证。")
    exit(0)

# 按 turn 和 name 分组统计
turn_counter = Counter()
turn_name_counter = Counter()
for meta in results["metadatas"]:
    t = meta.get("turn")
    n = meta.get("name", "")
    turn_counter[t] += 1
    turn_name_counter[(t, n)] += 1

print("\n    按 Turn 统计:")
for t in sorted(turn_counter):
    print(f"      Turn #{t}: {turn_counter[t]} 条")

print("\n    按 (Turn, Name) 统计:")
for (t, n), c in sorted(turn_name_counter.items()):
    marker = " ⚠️ 重复!" if c > 1 else ""
    print(f"      (Turn#{t}, {n[:20]}): {c} 条{marker}")

# ─── 重点检查: Turn #1 + 铁砧·短句 ───
print("\n[3] Turn #1 铁砧·短句 去重验证")

items_turn1 = [
    (doc, meta) for doc, meta in zip(results["documents"], results["metadatas"])
    if meta.get("turn") == 1 and "铁砧" in str(meta.get("name", ""))
]

print(f"    ChromaDB 中 Turn#1 铁砧条目数: {len(items_turn1)}")

if len(items_turn1) == 0:
    print("    ⚠️ 未找到 Turn#1 铁砧的条目（可能在另一个集合中）")
    # 尝试在所有集合中查找
    for col_obj in all_collections:
        cname = col_obj if isinstance(col_obj, str) else col_obj.name
        ccol = client.get_collection(cname)
        cres = ccol.get(include=["documents", "metadatas"])
        for doc, meta in zip(cres["documents"], cres["metadatas"]):
            if meta.get("turn") == 1 and "铁砧" in str(meta.get("name", "")):
                print(f"    找到于集合 '{cname}': {doc[:80]}")

elif len(items_turn1) == 1:
    doc, meta = items_turn1[0]
    print(f"    ✅ ChromaDB 去重成功: Turn#1 铁砧仅 1 条")
    print(f"       存储内容: {doc[:100]}...")
    print(f"       关键术语: {meta.get('key_terms', '-')}")

elif len(items_turn1) > 1:
    print(f"    ❌ ChromaDB 去重失败: Turn#1 铁砧存在 {len(items_turn1)} 条重复")
    for i, (doc, meta) in enumerate(items_turn1):
        print(f"       [{i}] {doc[:100]}...")

# ─── 相似度交叉验证 ───
print("\n[4] 相似度交叉验证")

if len(items_turn1) >= 2:
    model = SentenceTransformer(MODEL_NAME)
    docs = [doc for doc, _ in items_turn1]
    embs = [model.encode(d).tolist() for d in docs]
    for i in range(len(docs)):
        for j in range(i+1, len(docs)):
            dot = sum(a*b for a,b in zip(embs[i], embs[j]))
            cos_sim = dot / ((sum(a*a for a in embs[i])**0.5) * (sum(b*b for b in embs[j])**0.5))
            print(f"    条目[{i}] vs 条目[{j}]: cosine_sim={cos_sim:.4f}")
            if cos_sim >= DEDUP_SIMILARITY_THRESHOLD:
                print(f"      → 相似度 >= {DEDUP_SIMILARITY_THRESHOLD}，应被去重但未被拦截")
            else:
                print(f"      → 相似度 < {DEDUP_SIMILARITY_THRESHOLD}，内容不同，不应去重")
else:
    print("    仅 1 条或 0 条，无法做交叉验证")

# ─── 全局去重率报告 ───
print("\n[5] 全局去重率报告")
print(f"    去重前预期记录数（含重复）: TXT 中 11 个块")
print(f"    去重后 ChromaDB 实际记录数: {total_origin} 条")

# 检查是否有 Turn 重复但内容不同的情况
print("\n[6] Turn 号重复但内容不同的条目（即同一轮次但不同会话）")
seen_turns = {}
for doc, meta in zip(results["documents"], results["metadatas"]):
    t = meta.get("turn")
    n = meta.get("name", "")
    if t in seen_turns:
        prev_doc, prev_name = seen_turns[t]
        if prev_doc[:60] != doc[:60]:
            print(f"    Turn#{t}: [{prev_name}] vs [{n}] — 内容不同，两个会话")
            print(f"       会话A: {prev_doc[:80]}...")
            print(f"       会话B: {doc[:80]}...")
    else:
        seen_turns[t] = (doc, n)

print("\n" + "=" * 60)
print("  验证完成")
print("=" * 60)
