"""
交互式语义检索工具：输入自然语言问题，返回向量库中最相关的记忆。
"""
import chromadb
from sentence_transformers import SentenceTransformer

# ==================== 配置区 ====================
COLLECTION_NAME = "character_skills"
MODEL_NAME = "shibing624/text2vec-base-chinese"
TOP_K = 3  # 每次返回的最相关条目数
# ===============================================

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name=COLLECTION_NAME)

print(f"记忆库共 {collection.count()} 条，输入问题即可查询（输入 q 退出）\n")

while True:
    query = input("🔍 你想查询什么记忆？ > ").strip()
    if query.lower() == 'q':
        break
    if not query:
        continue

    query_vec = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=TOP_K
    )

    print(f"\n与「{query}」最相关的 {TOP_K} 条记忆：")
    for i, (doc, dist, meta) in enumerate(
        zip(results['documents'][0], results['distances'][0], results['metadatas'][0]), 1
    ):
        tag = meta.get('type', 'unknown')
        print(f"  {i}. [{tag}] 距离:{dist:.4f}\n     {doc[:120]}\n")