from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "second_brain"
)

# Ask question
query = input("Ask a Question: ")

# Convert query to embedding
query_embedding = model.encode(
    query
).tolist()

# Search ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

print("\nQuestion:")
print(query)

print("\nMost Relevant Results:\n")

for i, (doc, meta) in enumerate(
    zip(
        results["documents"][0],
        results["metadatas"][0]
    )
):

    print("=" * 60)
    print(f"Result {i+1}")

    print("\nSource PDF:")
    print(meta["source"])

    print("\nContent:")
    print(doc)

    print("=" * 60)