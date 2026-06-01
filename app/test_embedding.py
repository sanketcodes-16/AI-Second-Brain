from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Inheritance allows one class to acquire properties from another class."

embedding = model.encode(text)

print("\nOriginal Text:")
print(text)

print("\nEmbedding Length:")
print(len(embedding))

print("\nFirst 10 Values:")
print(embedding[:10])