"""
Pre-ingestion conflict checker
"""
import json
from Backend.mongodb_client import MongoDBClient

# Initialize MongoDB
mongo_client = MongoDBClient()
collection = mongo_client.collection

# Check existing
existing_topics = collection.distinct("topic", {"source": "knowledge_base"})
existing_modules = collection.distinct("module_name", {"source": "knowledge_base"})

print(f"📊 Current topics in DB: {len(existing_topics)}")
print(f"📦 Current modules: {existing_modules}")

# Load new files
files = [
    r"C:\Users\newbr\OneDrive\Desktop\AISHINEBE_CLAUDE\Parsed_Module5_KB.json"
]

new_topics = []
new_module_counts = {}

for file_path in files:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        
        module_name = docs[0]["module_name"]
        new_module_counts[module_name] = len(docs)
        new_topics.extend([d["topic"] for d in docs])
        print(f"✅ Loaded {file_path}: {len(docs)} documents")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Check duplicates
duplicates = set(existing_topics) & set(new_topics)

print("\n" + "="*50)
if duplicates:
    print(f"⚠️  {len(duplicates)} duplicate topics found")
    for topic in list(duplicates)[:5]:
        print(f"   - {topic}")
else:
    print(f"✅ No conflicts - Safe to ingest {len(new_topics)} documents")
    print(f"\n📦 New modules:")
    for module, count in new_module_counts.items():
        print(f"   - {module}: {count} docs")
print("="*50)