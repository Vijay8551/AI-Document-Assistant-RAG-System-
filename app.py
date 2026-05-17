import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from document_processor import (
    load_document,
    chunk_documents,
    SUPPORTED_EXTS,
)

from vector_store import create_vector_store
from rag_chain import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 AI Document Assistant")
st.caption("Groq + HuggingFace + ChromaDB")

# ================= SESSION =================

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ================= SIDEBAR =================

with st.sidebar:

    st.header("📤 Upload Document")

    uploaded = st.file_uploader(
        "Choose a file",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTS],
    )

    if uploaded:

        # Image Preview
        if uploaded.type.startswith("image/"):
            st.image(
                uploaded,
                caption="Preview",
                use_container_width=True
            )

        if st.button("🚀 Process Document"):

            with st.spinner("Processing document..."):

                ext = os.path.splitext(uploaded.name)[1]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=ext
                ) as tmp:

                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                try:

                    # Load document
                    docs = load_document(tmp_path)

                    # Chunking
                    chunks = chunk_documents(docs)

                    if len(chunks) == 0:
                        st.error("No content extracted.")
                    else:

                        # Create vector DB
                        vectorstore = create_vector_store(chunks)

                        # Build RAG chain
                        st.session_state.qa_chain = build_rag_chain(
                            vectorstore
                        )

                        st.success(
                            f"✅ Indexed {len(chunks)} chunks!"
                        )

                        # OCR Preview
                        if ext.lower() in [".png", ".jpg", ".jpeg"]:

                            with st.expander("📝 OCR Text Preview"):
                                st.text(docs[0].page_content[:2000])

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

                finally:
                    os.unlink(tmp_path)

    st.divider()

    st.markdown("### 🧠 Models")

    st.code(
        "LLM: llama-3.3-70b-versatile\n"
        "Embeddings: all-MiniLM-L6-v2"
    )

# ================= CHAT HISTORY =================

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if "sources" in msg:

            with st.expander("📎 Sources"):

                for i, src in enumerate(msg["sources"], 1):

                    st.markdown(
                        f"**Source {i}** "
                        f"| Page: `{src['page']}` "
                        f"| Type: `{src['type']}`"
                    )

                    st.caption(src["snippet"])

# ================= CHAT INPUT =================

question = st.chat_input(
    "Ask something about the uploaded document..."
)

if question:

    if st.session_state.qa_chain is None:

        st.warning("⚠️ Upload and process a document first.")

    else:

        # USER MESSAGE
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        # ASSISTANT RESPONSE
        with st.chat_message("assistant"):

            with st.spinner("⚡ Thinking..."):

                result = ask_question(
                    st.session_state.qa_chain,
                    question
                )

            st.markdown(result["answer"])

            with st.expander("📎 Sources"):

                for i, src in enumerate(result["sources"], 1):

                    st.markdown(
                        f"**Source {i}** "
                        f"| Page: `{src['page']}` "
                        f"| Type: `{src['type']}`"
                    )

                    st.caption(src["snippet"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"]
            })