from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Read PDF
reader = PdfReader("data/pdfs/sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

print("\nChunks Created:\n")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}")
    print("-"*50)
    print(chunk)
    
chunks = splitter.split_text(text)

print(f"Total Chunks: {len(chunks)}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection("second_brain")
except:
    pass

collection = client.get_or_create_collection(
    name="second_brain"
)

# Store chunks
for i, chunk in enumerate(chunks):

    embedding = model.encode(chunk).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk]
    )

print("Embeddings stored successfully!")