"""
从 SillyTavern 导出的角色 JSON 中提取 World Info 条目，
进行 embedding 后存入 Chroma 向量数据库。
"""
import json
import chromadb
from sentence_transformers import SentenceTransformer
import re
import os

# ==================== 配置区（按需修改） ====================
JSON_FILE = "data/CharacterInfo/Characterdesign.json"  # 角色设定 JSON 文件
COLLECTION_NAME = "character_skills"      # Chroma 集合名称
MODEL_NAME = "shibing624/text2vec-base-chinese"
# ===========================================================

print("正在加载 embedding 模型...")
model = SentenceTransformer(MODEL_NAME)

# 连接 Chroma（数据持久化到 ./chroma_db）
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name=COLLECTION_NAME)
print(f"集合 '{COLLECTION_NAME}' 当前条目数: {collection.count()}")

# 清空现有数据（全量覆盖）
if collection.count() > 0:
    print("⚠️ 检测到现有数据，正在清空...")
    existing_ids = collection.get()['ids']
    if existing_ids:
        collection.delete(ids=existing_ids)
    print("✅ 已清空现有数据。")

# 读取 JSON
if not os.path.exists(JSON_FILE):
    raise FileNotFoundError(f"找不到 {JSON_FILE}，请将文件放入当前目录。")

with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 从 Characterdesign.json 中提取 entries
entries = data.get("entries", {})

if not entries:
    raise ValueError("entries 为空，请确认角色卡包含 World Info 条目。")

print(f"找到 {len(entries)} 个世界条目。")

# 提取文本块并清理 [type:xxx] 标签
type_pattern = re.compile(r'\s*\[type:(.+?)\]\s*$')
documents = []
metadatas = []
ids = []

for key, entry in entries.items():
    content = entry.get("content", "").strip()
    if not content:
        continue

    # 使用 entry["key"][0] 作为 entry_key 和 ID
    key_list = entry.get("key", [])
    if not key_list or not isinstance(key_list, list):
        print(f"⚠️ 条目 {key} 缺少有效的 key 字段，跳过")
        continue

    entry_key = key_list[0]  # 取第一个关键词作为 entry_key

    match = type_pattern.search(content)
    entry_type = match.group(1).strip() if match else "unknown"
    # 移除标签，保持文本干净
    content_clean = type_pattern.sub("", content).strip()

    documents.append(content_clean)
    metadatas.append({
        "source": JSON_FILE,
        "entry_key": entry_key,
        "type": entry_type,
        "uid": entry.get("uid", key)
    })
    ids.append(entry_key)

# embedding 并入库
if documents:
    print(f"正在为 {len(documents)} 条文本生成向量...")
    embeddings = model.encode(documents, show_progress_bar=True).tolist()
    collection.add(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✅ 成功入库 {len(documents)} 条记忆！")
else:
    print("⚠️ 没有有效条目，未入库。")

print(f"集合 '{COLLECTION_NAME}' 当前总条目数: {collection.count()}")
