import chromadb
from sentence_transformers import SentenceTransformer

MODEL_NAME = "shibing624/text2vec-base-chinese"
COLLECTION_NAME = "character_skills"

model = SentenceTransformer(MODEL_NAME)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name=COLLECTION_NAME)

# 配置：你要更新的技能 entry_key 和新内容
ENTRY_KEY = "skill_li_guijianshu"   # 在 World Info 中设置的 Key（里·鬼剑术）
NEW_CONTENT = "技能：里·鬼剑术。剑魂的核心基础技，通过高速连续斩击形成稳定输出循环。当前熟练度 12/100。限制：依赖节奏与站位，被打断时输出大降。[type:skill]"

# 根据 entry_key 查找旧条目
results = collection.get(where={"entry_key": ENTRY_KEY})
if results['ids']:
    old_id = results['ids'][0]
    # 生成新向量
    new_emb = model.encode(NEW_CONTENT).tolist()
    collection.update(
        ids=[old_id],
        documents=[NEW_CONTENT],
        embeddings=[new_emb]
    )
    print(f"✅ 已更新技能 {ENTRY_KEY}，当前熟练度已写入。")
else:
    print("❌ 未找到该条目，请检查 entry_key。")