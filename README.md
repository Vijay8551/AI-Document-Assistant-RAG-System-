📚 AI Document Assistant
Chat with your documents — powered by RAG

Python Streamlit LangChain Groq ChromaDB License: MIT

Upload a PDF, TXT, or image — then ask questions about it in plain English.
Answers are grounded strictly in your document, with source citations on every response.

✨ Features
📄 PDF — page-level ingestion via PyPDF
📝 TXT — plain text file support
🖼️ Images (PNG / JPG / JPEG) — text extraction via Tesseract OCR with in-app preview
🔒 Grounded answers — LLM only responds from document context; says "I cannot find this information" otherwise
📌 Source citations — every answer shows the page number, document type, and a 200-character snippet of the source chunk
💬 Persistent chat history — full session memory rendered in a clean chat UI
🧹 Safe temp file handling — uploaded files are cleaned up immediately after processing
🖥️ UI Overview
┌─────────────────────┬──────────────────────────────────────────┐
│      SIDEBAR        │            MAIN CHAT AREA                │
│                     │                                          │
│  📤 Upload Document │  📚 AI Document Assistant                │
│  [ file uploader ]  │  Groq + HuggingFace + ChromaDB           │
│                     │                                          │
│  🖼️ Image preview   │  ┌──────────────────────────────────┐    │
│  (if image)         │  │ 👤 User: What is this doc about? │    │
│                     │  └──────────────────────────────────┘    │
│  🚀 Process button  │  ┌──────────────────────────────────┐    │
│  ✅ Success message │  │ 🤖 Assistant: This document...   │    │
│                     │  │  📎 Sources ▼                    │    │
│  📝 OCR Preview     │  │   Source 1 | Page: 2 | Type: text│    │
│  (if image)         │  └──────────────────────────────────┘    │
│                     │                                          │
│  🧠 Models info     │  [ Ask something about the document... ] │
└─────────────────────┴──────────────────────────────────────────┘
🏗️ RAG Pipeline
User uploads file
        │
        ▼
┌───────────────────┐
│   1. INGESTION    │  document_processor.py
│  .pdf → PyPDF     │
│  .txt → TextLoader│
│  .png/.jpg/.jpeg  │
│    → Tesseract OCR│
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   2. CHUNKING     │  chunk_documents()
│  size  = 1000     │  RecursiveCharacterTextSplitter
│  overlap = 200    │  separators: \n\n · \n · ". " · " "
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  3. EMBEDDING     │  vector_store.py
│  all-MiniLM-L6-v2 │  HuggingFaceEmbeddings (CPU)
│  normalize=True   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  4. VECTOR STORE  │  ChromaDB → ./chroma_db
│  Chroma.from_docs │  persisted locally
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  5. RETRIEVAL     │  rag_chain.py
│  similarity search│  top k = 4 chunks
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  6. GENERATION    │  ChatGroq
│  llama-3.3-70b    │  temperature=0, max_tokens=1024
│  GROUNDED_PROMPT  │  → answer + sources returned
└───────────────────┘
🛠️ Tech Stack
Layer	Technology
Frontend	Streamlit (wide layout, sidebar + chat)
Orchestration	LangChain
LLM	Groq — llama-3.3-70b-versatile
Embeddings	HuggingFace — sentence-transformers/all-MiniLM-L6-v2
Vector DB	ChromaDB (local, persisted to ./chroma_db)
PDF Parsing	pypdf via PyPDFLoader
Image OCR	Tesseract + pytesseract + Pillow
Config	python-dotenv
📁 Project Structure
ai-document-assistant/
│
├── app.py                  # Streamlit UI — sidebar, chat, session state
├── document_processor.py   # Ingestion (PDF/TXT/Image OCR) + chunking
├── vector_store.py         # HuggingFace embeddings + ChromaDB
├── rag_chain.py            # RAGChain class, grounded prompt, Groq LLM
│
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .gitignore              # Excludes .env, chroma_db/, __pycache__
├── LICENSE
└── README.md
🚀 Getting Started
1. Clone the repository
git clone https://github.com/your-username/ai-document-assistant.git
cd ai-document-assistant
2. Install Python dependencies
pip install -r requirements.txt
3. Install Tesseract OCR
Tesseract is required for image file support.

Windows: Download the installer from UB-Mannheim and install to the default path: C:\Program Files\Tesseract-OCR\

macOS:

brew install tesseract
Linux (Ubuntu / Debian):

sudo apt-get install tesseract-ocr
⚠️ macOS / Linux users: After installing, comment out the hardcoded path in document_processor.py:

# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
4. Configure environment variables
cp .env.example .env
Open .env and add your key:

GROQ_API_KEY=your_groq_api_key_here
Get a free API key at console.groq.com.

5. Run the app
streamlit run app.py
Visit http://localhost:8501 in your browser.

🔑 Environment Variables
Variable	Required	Default	Description
GROQ_API_KEY	✅ Yes	—	Groq API key for LLM inference
GROQ_MODEL	❌ No	llama-3.3-70b-versatile	Groq model name
HF_EMBED_MODEL	❌ No	sentence-transformers/all-MiniLM-L6-v2	Embedding model
📖 How to Use
Open the app at http://localhost:8501
Use the sidebar to upload a file — PDF, TXT, or image
For images, a thumbnail preview appears immediately on selection
Click 🚀 Process Document to index the file
For images, expand 📝 OCR Text Preview to verify the extracted text
Type your question in the chat box at the bottom
The assistant replies with a grounded answer and an expandable 📎 Sources section showing the exact chunks used
🔍 Error Handling
Scenario	Behaviour
Unsupported file type	ValueError raised in load_document() → shown in sidebar
Image with no readable text	ValueError("No text extracted from image.")
Tesseract not installed / crash	RuntimeError("OCR failed: ...") → shown in sidebar
Document yields zero chunks	st.error("No content extracted.") — indexing aborted
Question asked before upload	st.warning("⚠️ Upload and process a document first.")
Unexpected errors (API, etc.)	Caught by broad except Exception → shown in main area
Temp file cleanup	finally: os.unlink(tmp_path) — always runs, even on crash
