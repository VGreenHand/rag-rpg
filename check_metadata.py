import chromadb
import json

client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection('character_skills')
results = collection.get()

print(f"Total entries: {len(results['ids'])}\n")
print("All entry_keys:")
for id, meta in zip(results['ids'], results['metadatas']):
    print(f"  ID: {id[:8]}... | entry_key: '{meta['entry_key']}' | type: {meta['type']}")
