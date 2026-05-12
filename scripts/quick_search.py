"""
交互式语义检索工具：输入自然语言问题，返回向量库中最相关的记忆。
搜索范围覆盖所有集合（技能库 + 对话记忆 + 自定义记忆）。

用法:
  python scripts/quick_search.py                          # 默认 profile
  python scripts/quick_search.py --profile char_1         # 指定 profile
"""
import argparse
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CHROMA_PATH, get_chroma_path

MODEL_NAME = "shibing624/text2vec-base-chinese"
COLLECTIONS = ["character_skills", "dialogue_memory", "my_rag_memory"]
TOP_K = 3

parser = argparse.ArgumentParser(description="RAG-RPG 记忆搜索工具")
parser.add_argument("--profile", type=str, default="default",
                    help="要查询的 profile 名称（默认: default）")
args = parser.parse_args()

chroma_dir = CHROMA_PATH if args.profile == "default" else get_chroma_path(args.profile)
profile_label = args.profile

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path=chroma_dir)

all_count = 0
for col_name in COLLECTIONS:
    try:
        col = client.get_collection(name=col_name)
        all_count += col.count()
    except Exception:
        pass

print("RAG-RPG 记忆搜索工具")
print("-" * 50)
print(f"  Profile: {profile_label}")
print(f"  ChromaDB: {chroma_dir}")
print("  技能库(character_skills)    对话记忆(dialogue_memory)  自定义记忆(my_rag_memory)")
for col_name in COLLECTIONS:
    try:
        col = client.get_collection(name=col_name)
        print(f"    {col.count():>4} 条", end="")
    except Exception:
        print(f"    无法访问", end="")
print(f"\n  总计: {all_count} 条记忆")
print("-" * 50)
print("输入问题即可查询（输入 q 退出）\n")

while True:
    query = input("? 你想查询什么？ > ").strip()
    if query.lower() == 'q':
        break
    if not query:
        continue

    query_vec = model.encode(query).tolist()
    all_results = []

    for col_name in COLLECTIONS:
        try:
            col = client.get_collection(name=col_name)
            results = col.query(query_embeddings=[query_vec], n_results=TOP_K)
            for i in range(len(results['ids'][0])):
                dist = results['distances'][0][i]
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                all_results.append((col_name, dist, doc, meta))
        except Exception:
            continue

    all_results.sort(key=lambda x: x[1])

    if not all_results:
        print(f"\n  未找到相关信息。\n")
        continue

    print(f"\n与「{query}」最相关的 {TOP_K} 条记忆：")
    for col_name, dist, doc, meta in all_results[:TOP_K]:
        tag = meta.get('type', meta.get('entry_key', col_name))
        speaker = meta.get('speaker', '')
        turn = meta.get('turn', '')
        extra = f"[Turn#{turn} {speaker}]" if speaker else ""
        print(f"  [{tag}] {extra}(相关度:{1-dist:.4f})")
        print(f"    {doc[:500]}\n")
