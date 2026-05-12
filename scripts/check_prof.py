import chromadb, re, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHROMA_PATH, COLLECTION_SKILLS

col = chromadb.PersistentClient(path=CHROMA_PATH).get_collection(name=COLLECTION_SKILLS)
data = col.get()
for i in range(len(data['ids'])):
    doc = data['documents'][i]
    m = re.search(r'熟练度[\s：:]*(\d+)/100', doc)
    if m:
        print(f"{data['metadatas'][i]['entry_key']}: 熟练度 {m.group(1)}/100")
