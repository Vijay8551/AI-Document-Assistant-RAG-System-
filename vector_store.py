from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

EMBED_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )


def create_vector_store(
    chunks,
    persist_dir="./chroma_db"
):

    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    vectorstore.persist()

    return vectorstore


def load_vector_store(
    persist_dir="./chroma_db"
):

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )