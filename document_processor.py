# document_processor.py

import os
from PIL import Image
import pytesseract

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_core.documents import Document


# =========================================================
# TESSERACT OCR PATH (WINDOWS)
# =========================================================
# Install Tesseract OCR first:
# https://github.com/UB-Mannheim/tesseract/wiki
#
# Default installation path:
# C:\Program Files\Tesseract-OCR\tesseract.exe
# =========================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# =========================================================

SUPPORTED_EXTS = {
    ".pdf",
    ".txt",
    ".png",
    ".jpg",
    ".jpeg",
}


# =========================================================
# IMAGE OCR
# =========================================================

def extract_text_from_image(file_path: str):

    try:

        image = Image.open(file_path)

        # Convert RGBA → RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # OCR
        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        return text.strip()

    except Exception as e:

        raise RuntimeError(
            f"OCR failed: {str(e)}"
        )


# =========================================================
# DOCUMENT LOADER
# =========================================================

def load_document(file_path: str):

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in SUPPORTED_EXTS:

        raise ValueError(
            f"Unsupported file type: {ext}"
        )

    # =====================================================
    # PDF
    # =====================================================

    if ext == ".pdf":

        return PyPDFLoader(file_path).load()

    # =====================================================
    # TXT
    # =====================================================

    if ext == ".txt":

        return TextLoader(
            file_path,
            encoding="utf-8"
        ).load()

    # =====================================================
    # IMAGE OCR
    # =====================================================

    if ext in [".png", ".jpg", ".jpeg"]:

        text = extract_text_from_image(file_path)

        if not text:

            raise ValueError(
                "No text extracted from image."
            )

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path,
                    "type": "image-ocr"
                }
            )
        ]


# =========================================================
# DOCUMENT CHUNKING
# =========================================================

def chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    return splitter.split_documents(documents)