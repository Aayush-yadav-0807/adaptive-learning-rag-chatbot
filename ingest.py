"""
ingest.py
=========
Standalone ingestion script.

Pipeline
--------
1. Discover every PDF inside the ``data/`` folder.
2. Load each PDF page by page (``PyPDFLoader``).
3. Split the text into overlapping chunks
   (``RecursiveCharacterTextSplitter``: size 800, overlap 150).
4. Embed the chunks with HuggingFace ``all-MiniLM-L6-v2``.
5. Persist the vectors in a local ChromaDB directory.

Usage
-----
    python ingest.py             # rebuild the vector store from scratch
    python ingest.py --append    # add to the existing store (may duplicate)
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")

# Maximum number of chunks sent to ChromaDB in a single call
BATCH_SIZE = 256


def find_pdfs(data_dir: Path) -> List[Path]:
    """Return a sorted list of PDF files inside ``data_dir``.

    Raises:
        FileNotFoundError: if the folder is missing or contains no PDFs.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"Data folder not found: '{data_dir}'. "
            "Create it and place your PDF study material inside."
        )

    pdf_files = sorted(
        p for p in data_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in '{data_dir}'. "
            "Add at least one .pdf file and run the script again."
        )
    return pdf_files


def load_pdf(pdf_path: Path) -> List[Document]:
    """Load a single PDF into a list of page-level ``Document`` objects.

    A corrupted or unreadable PDF is logged and skipped instead of
    crashing the whole ingestion run.
    """
    try:
        pages = PyPDFLoader(str(pdf_path)).load()
    except Exception as exc:  # noqa: BLE001 - we want to survive any parser error
        logger.warning("Skipping '%s' (could not be read): %s", pdf_path.name, exc)
        return []

    # Keep only pages that actually contain text (scanned images give "")
    pages = [p for p in pages if p.page_content and p.page_content.strip()]
    if not pages:
        logger.warning(
            "'%s' has no extractable text (scanned PDF?). Skipping.", pdf_path.name
        )
        return []

    # Store a clean, short file name for display in the UI
    for page in pages:
        page.metadata["source"] = pdf_path.name

    logger.info("Loaded %-35s -> %d pages", pdf_path.name, len(pages))
    return pages


def split_documents(documents: List[Document]) -> List[Document]:
    """Split page-level documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Split %d pages into %d chunks (size=%d, overlap=%d)",
        len(documents), len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP,
    )
    return chunks


def build_vector_store(chunks: List[Document], reset: bool = True) -> int:
    """Embed ``chunks`` and persist them in ChromaDB.

    Args:
        chunks: Chunked documents to store.
        reset: If True, delete any existing vector store first so that
            re-running the script never creates duplicate entries.

    Returns:
        The number of vectors stored in the collection afterwards.
    """
    if reset and config.CHROMA_DIR.exists():
        try:
            shutil.rmtree(config.CHROMA_DIR)
            logger.info("Removed old vector store at '%s'", config.CHROMA_DIR)
        except PermissionError as exc:
            raise PermissionError(
                "Cannot delete the old vector store. Stop the Streamlit app "
                f"(it keeps files open) and try again. Details: {exc}"
            ) from exc

    logger.info("Loading embedding model '%s' ...", config.EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    total = len(chunks)
    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        vector_store.add_documents(batch)
        logger.info("Embedded %d / %d chunks", min(start + BATCH_SIZE, total), total)

    return vector_store._collection.count()  # noqa: SLF001


def main() -> int:
    """Entry point. Returns a process exit code (0 = success)."""
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector store from PDFs in the data/ folder."
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Add to the existing vector store instead of rebuilding it.",
    )
    args = parser.parse_args()

    try:
        pdf_files = find_pdfs(config.DATA_DIR)
        logger.info("Found %d PDF file(s) in '%s'", len(pdf_files), config.DATA_DIR)

        documents: List[Document] = []
        for pdf in pdf_files:
            documents.extend(load_pdf(pdf))

        if not documents:
            logger.error("No readable text could be extracted from any PDF.")
            return 1

        chunks = split_documents(documents)
        count = build_vector_store(chunks, reset=not args.append)

    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except PermissionError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during ingestion: %s", exc)
        return 1

    logger.info("Done. Vector store now holds %d chunks at '%s'.", count, config.CHROMA_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
