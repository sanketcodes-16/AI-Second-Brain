import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection (to avoid duplicates)
try:
    client.delete_collection("second_brain")
    print("Old collection deleted.")
except:
    print("No old collection found.")

# Create fresh collection
collection = client.get_or_create_collection(
    name="second_brain"
)

# Folder containing PDFs
pdf_folder = "data/pdfs"

# Process all PDFs
for pdf_file in os.listdir(pdf_folder):

    if not pdf_file.endswith(".pdf"):
        continue

    print("\n" + "=" * 60)
    print(f"Processing PDF: {pdf_file}")
    print("=" * 60)

    pdf_path = os.path.join(pdf_folder, pdf_file)

    # Read PDF
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text + "\n"

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    print(f"Total Chunks Created: {len(chunks)}")

    # Show chunks (for debugging)
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}")
        print("-" * 50)
        print(chunk[:200])

    # Store chunks in ChromaDB
    for i, chunk in enumerate(chunks):

        embedding = model.encode(chunk).tolist()

        collection.add(
            ids=[f"{pdf_file}_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[
                {
                    "source": pdf_file
                }
            ]
        )

print("\nEmbeddings stored successfully!")
print("All PDFs processed successfully.")