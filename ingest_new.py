"""
从 txt 文件中读取带标记的文本行（如 [SKILL] 拔刀斩），
提取内容并追加到 Chroma 向量库。
"""
import re
import uuid
import chromadb
from sentence_transformers import SentenceTransformer

# ==================== 配置区 ====================
INPUT_FILE = "new_batch.txt"              # 存放标记行（一行一个标记）
COLLECTION_NAME = "my_rag_memory"         # 与初始入库保持一致
MODEL_NAME = "shibing624/text2vec-base-chinese"
# ===============================================

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# 正则匹配行首的 [TYPE] 和后面的内容
pattern = re.compile(r'^\[(.+?)\]\s*(.+)')

chunks = []
types = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            chunk_type = match.group(1).strip()
            chunk_text = match.group(2).strip()
            chunks.append(chunk_text)
            types.append(chunk_type)
            print(f"✔ 解析成功: [{chunk_type}] {chunk_text[:60]}...")
        else:
            print(f"⚠ 跳过无法识别的行: {line[:60]}...")

if not chunks:
    print("❌ 没有找到任何有效标记，请检查文件格式。")
    exit()

print(f"\n正在为 {len(chunks)} 条新记忆生成向量...")
embeddings = model.encode(chunks, show_progress_bar=True).tolist()

ids = [str(uuid.uuid4()) for _ in chunks]
metadatas = [{"type": t.lower(), "source": INPUT_FILE} for t in types]

collection.add(
    embeddings=embeddings,
    documents=chunks,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ 成功追加 {len(chunks)} 条新记忆！")
print(f"集合 '{COLLECTION_NAME}' 当前总条目数: {collection.count()}")