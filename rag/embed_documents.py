import io
import json
from contextlib import redirect_stdout, redirect_stderr

import chromadb
from sentence_transformers import SentenceTransformer


"""
Document embedding script for the prototype knowledge base.

This module prepares the local document collection for retrieval. It reads the source docuemtns, generates vector embeddings and strores them in a persistent ChromaDB collection.

Purpose in the prototype:
- Create the retrieval backbone for RAG and MCP pipelines
- Transform textural knowledge into vector representations
- Assign metadata such as title and sensitivity flags
- Ensure that all retrieval operations rely on the same document base
"""


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "documents"
VECTORSTORE_PATH = "vectorstore"


def build_embedding_text(doc: dict) -> str:
    """
    Use both title and content for embeddings.

    Args:
    - doc (dict): Single documents from the JSON knowledge base

    Returns:
    - str: Combined text used forr the embedding
    """
    title = doc.get("title", "").strip()
    content = doc.get("content", "").strip()
    return f"{title}\n\n{content}".strip()


def is_sensitive_document(doc: dict) -> bool:
    """
    Identify documents that should be treated as sensitive.

    Args:
    - doc (dict): Single document from the JSON knowledge base

    Returns:
    - bool: True if the document is considered sensitive
    """
    sensitive_ids = {"doc36", "doc37"}
    sensitive_title_keywords = {"secret code", "system identifier"}

    doc_id = str(doc.get("id", "")).strip().lower()
    title = str(doc.get("title", "")).strip().lower()

    if doc_id in sensitive_ids:
        return True

    return any(keyword in title for keyword in sensitive_title_keywords)


def main() -> None:
    """
    Build and store embeddings for the full document collection.

    Workflow:
    1. Load the embedding model
    2. Create or reset the persistent ChromaDB collection
    3. Read the JSON documents collection
    4. Generate embeddings for each doucment
    5. Store document content, embeddings and metadata in the vector store

    This script is executed once before retrieval experiments so that RAG and MCP pipelines operate on the same indexed knowledge base.
    """

    # Suppress verbose movel loading output to keep command line clean
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=VECTORSTORE_PATH)

    # Rebuild the collection from scratch to ensure a clean and reproducible state
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(COLLECTION_NAME)

    with open("data/documents.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    for doc in docs:
        text_for_embedding = build_embedding_text(doc)
        embedding = model.encode(text_for_embedding, show_progress_bar=False).tolist()

        collection.add(
            ids=[doc["id"]],
            documents=[doc["content"]],
            embeddings=[embedding],
            metadatas=[
                {
                    "title": doc["title"],
                    "sensitive": is_sensitive_document(doc),
                    "embedding_text": text_for_embedding,
                }
            ],
        )

    print("Documents embedded and stored.")
    print("Stored documents:", collection.count())


if __name__ == "__main__":
    main()