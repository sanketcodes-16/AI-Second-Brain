# 🧠 AI Second Brain

## 📌 Overview

AI Second Brain is a Multi-PDF Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions in natural language.

The system retrieves the most relevant information using vector embeddings and generates accurate answers using a Large Language Model (LLM).

This project demonstrates the complete RAG pipeline used in modern Generative AI applications.

---

# 🚀 Features

## 📄 Document Management

* Upload PDF documents
* Upload multiple PDFs simultaneously
* Persistent document storage
* Automatic PDF indexing
* Delete individual PDFs
* Total chunk statistics

## 🔍 Semantic Search

* Embedding-based retrieval
* ChromaDB vector database
* Top-K similarity search
* PDF-specific filtering
* Multi-document retrieval

## 🤖 AI Features

* Retrieval-Augmented Generation (RAG)
* Groq Llama 3.3 70B integration
* Context-aware answering
* Hallucination reduction
* Source attribution
* Retrieval similarity score

## 💬 User Experience

* Modern Streamlit chat interface
* Retrieved Context Viewer
* Export chat history
* Clear chat functionality
* Interactive PDF selection

---

# 🏗️ System Architecture

The following architecture demonstrates the complete RAG workflow.

![Architecture](images/architecture.png)

```text
User Question
      │
      ▼
PDF Upload
      │
      ▼
Text Extraction
      │
      ▼
Chunking
      │
      ▼
Embeddings
(all-MiniLM-L6-v2)
      │
      ▼
ChromaDB
(Vector Database)
      │
      ▼
Semantic Retrieval
      │
      ▼
Groq Llama 3.3 70B
      │
      ▼
Answer Generation
```

---

# 📸 Screenshots

## Main Chat Interface

Ask questions directly from uploaded PDFs.

![Chat Interface](images/chat_interface.png)

---

## PDF Management

Manage multiple PDFs, filter searches, and delete documents individually.

![PDF Management](images/pdf_management.png)

---

## Retrieved Context Viewer

View the exact chunks retrieved from ChromaDB before answer generation.

![Retrieved Context](images/retrieved_context.png)

---

## Export Chat

Download the entire conversation history.

![Chat Export](images/export_chat.png)

---

# ⚙️ Tech Stack

## Frontend

* Streamlit

## Large Language Model

* Groq
* Llama 3.3 70B Versatile

## Vector Database

* ChromaDB

## Embedding Model

* Sentence Transformers
* all-MiniLM-L6-v2

## PDF Processing

* PyPDF

## Text Splitting

* LangChain Text Splitters

---

# 🔄 Project Workflow

## Step 1: Upload PDF

Users upload one or more PDF documents.

## Step 2: Extract Text

Text is extracted from all pages using PyPDF.

## Step 3: Chunking

Documents are split into smaller chunks.

```python
chunk_size = 500
chunk_overlap = 100
```

## Step 4: Generate Embeddings

Each chunk is converted into vector embeddings using:

```python
all-MiniLM-L6-v2
```

## Step 5: Store in ChromaDB

Embeddings and metadata are stored in a vector database.

## Step 6: User Query

The user asks a question.

## Step 7: Semantic Retrieval

Relevant chunks are retrieved based on vector similarity.

## Step 8: Answer Generation

Retrieved context is sent to Groq Llama 3.3 70B.

## Step 9: Response Display

The application displays:

* Answer
* Source PDFs
* Retrieval Similarity Score
* Retrieved Context

---

# 🎯 Key Concepts Demonstrated

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Vector Embeddings
* Semantic Search
* Vector Databases
* Prompt Engineering
* Large Language Models (LLMs)
* Information Retrieval
* Multi-document Question Answering
* Generative AI Application Development

---

# 📂 Project Structure

```text
AI-Second-Brain
│
├── app/
│   ├── streamlit_app.py
│   ├── store_embeddings.py
│   ├── search_memory.py
│   ├── rag_chat.py
│   └── ...
│
├── images/
│   ├── architecture.png
│   ├── chat_interface.png
│   ├── pdf_management.png
│   ├── retrieved_context.png
│   └── export_chat.png
│
├── data/
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🛠️ Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Second-Brain.git
cd AI-Second-Brain
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

## Run Application

```bash
streamlit run app/streamlit_app.py
```

---

# 🔮 Future Improvements

* Conversation Memory
* Hybrid Search (Keyword + Semantic)
* OCR Support for Scanned PDFs
* User Authentication
* Cloud Deployment
* PDF Summarization
* Citation-Based Answers
* Reranking Models
* Multi-User Support

---

# 📈 Learning Outcomes

Through this project, I learned:

* Building end-to-end RAG systems
* Working with Vector Databases
* Generating Embeddings
* Semantic Search Implementation
* Streamlit Application Development
* LLM Integration using Groq
* Document Retrieval Pipelines
* Prompt Engineering

---

# 👨‍💻 Author

**Sanket More**

AI Second Brain – Multi-PDF RAG Assistant built using Streamlit, ChromaDB, Groq, and Sentence Transformers.