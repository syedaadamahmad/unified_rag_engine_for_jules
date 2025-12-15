# Check what's actually stored in MongoDB for the failing prompts
# from Backend.mongodb_client import MongoDBClient
# client = MongoDBClient()

# # Check "Why AI for Students"
# doc1 = client.db.module_vectors.find_one({"topic": "Why AI for Students"})
# print("Why AI for Students presentation_data keys:", doc1.get('presentation_data', {}).keys() if doc1 else "NOT FOUND")

# # Check "AI in Maths Mastery"  
# doc2 = client.db.module_vectors.find_one({"topic": "AI in Maths Mastery"})
# print("AI in Maths Mastery presentation_data keys:", doc2.get('presentation_data', {}).keys() if doc2 else "NOT FOUND")















# # test_kb_structure.py
# from Backend.mongodb_client import MongoDBClient

# client = MongoDBClient()

# # Check if old presentation topics exist in KB
# topics = ["Future Careers Powered by AI", "Why AI for Students", "AI in Science Labs"]

# for topic in topics:
#     doc = client.collection.find_one({"topic": topic, "source": "knowledge_base"})
#     if doc:
#         print(f"✅ Found: {topic}")
#         print(f"   Has 'presentation_data': {'presentation_data' in doc}")
#         print(f"   Has 'is_presentation': {'is_presentation' in doc}")
#         print(f"   Content length: {len(doc.get('content', ''))}")
#         print()
#     else:
#         print(f"❌ Not found: {topic}\n")

# client.close()














# test_chunk_size.py
from Backend.mongodb_client import MongoDBClient

client = MongoDBClient()

doc = client.collection.find_one({"topic": "Future Careers Powered by AI"})

if doc:
    content = doc.get('content', '')
    print(f"Content length: {len(content)} chars")
    print(f"Content preview (first 1000 chars):")
    print(content[:1000])
    print(f"\n...truncated at 1000 chars")
    print(f"\nNumber of careers in content: {content.count('Example:')}")
else:
    print("Document not found")

client.close()