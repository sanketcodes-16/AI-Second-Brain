from dotenv import load_dotenv
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "second_brain"
)

# Groq
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

while True:

    question = input(
        "\nAsk a Question (type exit to quit): "
    )

    if question.lower() == "exit":
        break

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    context = "\n".join(
        results["documents"][0]
    )

    sources = set()

    for meta in results["metadatas"][0]:
        sources.add(meta["source"])

    prompt = f"""
You are a PDF assistant.

Rules:
1. Answer only from the context.
2. If answer is not present, say:
   'Information not found in uploaded PDFs.'

Context:
{context}

Question:
{question}
"""

    response = llm.invoke(prompt)

    print("\nAnswer:\n")
    print(response.content)

    print("\nSources Used:")

    for source in sources:
        print(f"- {source}")