"""
验证 ChromaDB 中的 ids 是否与 Characterdesign.json 中的条目 keys 一致。
"""
import json
import chromadb

# ==================== 配置区 ====================
COLLECTION_NAME = "character_skills"
JSON_FILE = "CharacterInfo/Characterdesign.json"
# ===============================================

# 1. 读取 Characterdesign.json
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

entries = data.get("entries", {})
json_keys = set(entries.keys())
print(f"🔹 Characterdesign.json 中共 {len(json_keys)} 个条目")
print(f"   JSON 条目 keys: {sorted(json_keys, key=int)}")

# 2. 连接 ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name=COLLECTION_NAME)
chroma_count = collection.count()
print(f"\n🔸 ChromaDB 集合 '{COLLECTION_NAME}' 中共 {chroma_count} 个条目")

# 3. 获取 ChromaDB 中所有数据
all_data = collection.get()
chroma_ids = set(all_data["ids"])
print(f"   ChromaDB ids: {sorted(chroma_ids, key=int)}")

# 4. 对比验证
print("\n" + "=" * 50)
print("📊 验证结果")
print("=" * 50)

# 检查 ChromaDB 中每个条目的 metadata.entry_key 是否与 id 匹配
id_key_mismatch = False
for i, (cid, meta) in enumerate(zip(all_data["ids"], all_data["metadatas"])):
    entry_key = meta.get("entry_key", "N/A")
    uid = meta.get("uid", "N/A")
    content_preview = all_data["documents"][i][:80].replace("\n", " ")
    status = "✅" if cid == entry_key else "❌"
    if cid != entry_key:
        id_key_mismatch = True
    print(f"   {status} id={cid:>4} | entry_key={entry_key:>4} | uid={uid:>4} | {content_preview}...")

print()

# 检查 keys 集合是否一致
if json_keys == chroma_ids:
    print("✅ 完全一致：JSON 条目 keys 与 ChromaDB ids 完全匹配！")
else:
    only_in_json = json_keys - chroma_ids
    only_in_chroma = chroma_ids - json_keys
    print("❌ 不一致：")
    if only_in_json:
        print(f"   📄 仅在 JSON 中，不在 ChromaDB: {sorted(only_in_json, key=int)}")
    if only_in_chroma:
        print(f"   🗄️ 仅在 ChromaDB 中，不在 JSON: {sorted(only_in_chroma, key=int)}")

if id_key_mismatch:
    print("❌ 存在 id 与 entry_key 不匹配的条目！")
else:
    print("✅ 所有条目的 id 与 entry_key 均一致！")

print(f"\n📌 集合总条目数: {chroma_count}")
