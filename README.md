# 📚 AI Document Assistant (RAG)

An AI-powered **Retrieval-Augmented Generation (RAG)** application built with **Python**, **Streamlit**, **LangChain**, **ChromaDB**, and **Groq Llama 3.3**. The application enables users to upload documents, perform semantic search, and receive accurate, context-aware answers based solely on the uploaded content.

---

## 🚀 Features

* 📄 Upload PDF, TXT, PNG, JPG, and JPEG documents
* 🖼️ OCR support for image documents using Tesseract OCR
* ✂️ Automatic text chunking
* 🧠 Semantic search with HuggingFace embeddings
* 🗄️ ChromaDB vector database for document storage
* 🤖 AI-powered responses using Groq Llama 3.3 70B
* 💬 Interactive Streamlit chat interface
* 📑 Displays source references for generated answers

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Groq (Llama 3.3 70B Versatile)**
* **ChromaDB**
* **HuggingFace Embeddings**
* **Sentence Transformers**
* **Tesseract OCR**
* **PyPDF**

---

## 📂 Project Structure

```text
AI-Document-Assistant/
│
├── app.py                  # Streamlit Application
├── document_processor.py   # Document Loading, OCR & Chunking
├── vector_store.py         # ChromaDB & Embeddings
├── rag_chain.py            # Retrieval-Augmented Generation Pipeline
├── requirements.txt
├── README.md
├── .env
└── chroma_db/
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/AI-Document-Assistant.git

cd AI-Document-Assistant

python -m venv venv
```

Activate the virtual environment.

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🔄 Workflow

1. Upload a document.
2. Extract text from PDFs, text files, or images.
3. Perform OCR on image files.
4. Split the extracted text into chunks.
5. Generate embeddings using HuggingFace.
6. Store embeddings in ChromaDB.
7. Ask questions about the uploaded document.
8. Retrieve relevant document chunks using semantic search.
9. Generate context-aware answers using Groq Llama 3.3.
10. Display the answer with supporting document references.

---

## 📦 Supported File Types

* PDF
* TXT
* PNG
* JPG
* JPEG

---

## 👨‍💻 Author

**Vijay Bhadane**

Master's Student in Computer Science

AI/ML • Data Science • Generative AI • Retrieval-Augmented Generation (RAG)

---
