import uuid
import tempfile

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="AI Second Brain",
    page_icon="🧠",
    layout="wide"
)

# Title
st.title("🧠 AI Second Brain")
st.caption("Multi-PDF RAG System powered by Groq")


# --------------------------------------------------
# PDF Processing Function
# --------------------------------------------------
def process_uploaded_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(
            uploaded_file.getvalue()
        )

        pdf_path = tmp_file.name

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        extracted_text = page.extract_text()

        if extracted_text:
            text += extracted_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks


# --------------------------------------------------
# Load Models Once
# --------------------------------------------------
@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        "second_brain"
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )

    return embedding_model, collection, llm


embedding_model, collection, llm = load_models()


# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    try:
        data = collection.get()
        pdf_names = set()

        for meta in data["metadatas"]:
            if meta and "source" in meta:
                pdf_names.add(
                    meta["source"]
                )
        st.session_state.uploaded_files = list(
            sorted(pdf_names)
        )
    except:
        st.session_state.uploaded_files = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:

    st.header("Controls")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.rerun()
    
    # --------------------------
    # Export Chat
    # --------------------------

    chat_text = ""

    for msg in st.session_state.messages:

        role = (
            "User"
            if msg["role"] == "user"
            else "Assistant"
        )

        chat_text += (
            f"{role}:\n"
            f"{msg['content']}\n\n"
        )

        chat_text += (
            "=" * 50 +
            "\n\n"
        )

    st.download_button(
        label="⬇ Download Chat",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )
    
    st.divider()
    
    st.metric(
        "Total Chunks",
        
        collection.count()
    )

    st.subheader("Uploaded PDFs")
    
    selected_pdf = st.selectbox(
        "Search Inside PDF",
        ["All PDFs"] +
        st.session_state.uploaded_files
    )

    for file_name in st.session_state.uploaded_files:
        col1, col2 = st.columns([8, 1])
        with col1:
            st.success(file_name)
        with col2:
            if st.button(
                "❌",
                key=f"delete_{file_name}"
            ):
                # Get all chunks belonging to this PDF
                results = collection.get(
                    where={
                        "source": file_name
                    }
                )
                # Delete those chunks
                if results["ids"]:

                    collection.delete(
                        ids=results["ids"]
                    )
                # Remove from sidebar list
                if file_name in st.session_state.uploaded_files:
                    st.session_state.uploaded_files.remove(
                        file_name
                    )
                st.rerun()

# --------------------------------------------------
# Process Uploaded PDF
# --------------------------------------------------
if uploaded_file:

    if uploaded_file.name not in st.session_state.uploaded_files:

        with st.spinner(
            "Processing PDF..."
        ):

            chunks = process_uploaded_pdf(
                uploaded_file
            )

            for chunk in chunks:

                embedding = embedding_model.encode(
                    chunk
                ).tolist()

                collection.add(
                    ids=[
                        f"{uploaded_file.name}_{uuid.uuid4()}"
                    ],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[
                        {
                            "source": uploaded_file.name
                        }
                    ]
                )

        st.session_state.uploaded_files.append(
            uploaded_file.name
        )

        st.success(
            f"{uploaded_file.name} uploaded successfully!"
        )


# --------------------------------------------------
# Show Previous Messages
# --------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "confidence" in message:

            st.info(
                f"Confidence Score: {message['confidence']}%"
            )

        if "sources" in message:

            st.markdown(
                "### 📄 Sources"
            )

            for source in message["sources"]:

                st.success(source)

        # --------------------------
        # Retrieved Context Viewer
        # --------------------------

        if "context" in message:

            with st.expander(
                "🔍 Retrieved Context"
            ):

                st.text(
                    message["context"]
                )


# --------------------------------------------------
# Chat Input
# --------------------------------------------------
question = st.chat_input(
    "Ask anything from your PDFs..."
)


# --------------------------------------------------
# Chat Logic
# --------------------------------------------------
if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner(
        "Searching documents..."
    ):

        query_embedding = embedding_model.encode(
            question
        ).tolist()

        if selected_pdf == "All PDFs":

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )

        else:

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={
                    "source": selected_pdf
                }
            )
            
        print("\nDistances:")
        print(results["distances"])

        best_distance = results["distances"][0][0]

        confidence = max(
            0,
            round(
                (1 - best_distance / 2) * 100,
                2
            )
        )

        context = "\n".join(
            results["documents"][0]
        )

        sources = set()

        for meta in results["metadatas"][0]:
            sources.add(
                meta["source"]
            )

        prompt = f"""
You are a document assistant.

STRICT RULES:

1. Answer ONLY using information present in the context.
2. Do NOT add information.
3. Do NOT infer information.
4. If the answer is not explicitly present, reply exactly:

Information not found in uploaded PDFs.

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        answer = response.content

    not_found = (
        "Information not found in uploaded PDFs."
        in answer
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

        if not not_found:

            st.info(
                f"Confidence Score: {confidence}%"
            )

            st.markdown("### 📄 Sources")

            for source in sorted(sources):
                st.success(source)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "context": context
    }

    if not not_found:

        assistant_message["sources"] = list(
            sorted(sources)
        )

        assistant_message["confidence"] = (
            confidence
        )

    st.session_state.messages.append(
        assistant_message
    )
