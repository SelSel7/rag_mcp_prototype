import io
import logging
import os
import re
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer
from transformers.utils import logging as transformers_logging

"""
Retrieval component for the prototype

This module provides the shared retrieval logic used by RAG and MCP pipelines. It loads the embedding model, accesses the persistent vector store and returns the most relevant documents for a given query.

Purpose in the prototype:
- Provide a common retrieval backend for RAG and MCP
- Convert user queries into vector embeddings
- Retrieve semantically similar documents from ChromaDB
- Filter sensitive documents before they reach the gneration step
"""


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTORSTORE_PATH = "vectorstore"
COLLECTION_NAME = "documents"

_model = None
_client = None
_collection = None


SENSITIVE_QUERY_PATTERNS = [
    r"\bsecret code\b",
    r"\bsystem identifier\b",
    r"\bhidden identifier\b",
    r"\brestricted information\b",
]


def _silence_embedding_stack() -> None:
    """
    Suppress verbose logging from the embedding stack.

    This keeps terminal output clean and avoids unnecessary warnings during model loading and embedding generation.
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    warnings.filterwarnings("ignore")
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    transformers_logging.set_verbosity_error()


def get_model() -> SentenceTransformer:
    """
    Load the embedding model lazily.

    The model is initialized only once and then reused across retrieval calls. This reduces repeated loading overhead during experiments.

    Returns:
    - SentenceTranformer: Embedding model instance
    """
    global _model
    if _model is None:
        _silence_embedding_stack()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def get_client():
    """
    Create or return the persistent ChromaDB client.

    Returns:
    - chromadb.PersistentClient: Client connected to the local vector store
    """
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=VECTORSTORE_PATH)
    return _client


def get_collection():
    """
    Access the document collection stored in ChromaDB.

    Returns:
    - ChromaDB collection: Persistent collection of embedded documents
    """
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(COLLECTION_NAME)
    return _collection


def build_query_embedding(query: str) -> list[float]:
    """
    Convert a user query into an embedding vector.

    Args:
    - query (str): Input question or query string

    Returns:
    - list[float]: Vector representation of the query
    """
    model = get_model()
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return model.encode(query, show_progress_bar=False).tolist()


def is_sensitive_query(query: str) -> bool:
    """
    Check wheather query targets senstive information

    Args:
    - query (str): Input question

    Returns:
    - bool: True if the query is considered sensitive
    """
    lowered = query.lower()
    return any(re.search(pattern, lowered) for pattern in SENSITIVE_QUERY_PATTERNS)


def retrieve(
    query: str,
    k: int = 3,
    allow_sensitive: bool = False,
    candidate_pool_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Retrieve the top-k most relevant documents for the vector store.

    Workflow:
    1. Embed the input query
    2. Search the persistent ChromaDB collection
    3. Rank and collect the most relevant documents
    4. Filter out sensitive documents if access is not allowed

    Args:
    - query (str): Input question or query string
    - k (int): Number of documents to return
    - allow_sensitive (bool): Wheather sensitive documents may be returned
    - candicate_pool_size (int | None): Number of initial candidates considered before filtering and truncation

    Returns:
    - tuple:
        - retrieved_docs (list[dict[str, Any]]): Retrieved non-sensitive documents
        - blocked_sensitive_doic_ids (list[str]): Sensitive documents IDs that were filtered
    """
    if candidate_pool_size is None:
        candidate_pool_size = max(k * 4, 10)

    collection = get_collection()
    query_embedding = build_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_pool_size,
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved_docs: list[dict[str, Any]] = []
    blocked_sensitive_doc_ids: list[str] = []

    for doc_id, doc_text, metadata, distance in zip(ids, documents, metadatas, distances):
        sensitive = bool(metadata.get("sensitive", False))

        if sensitive and not allow_sensitive:
            blocked_sensitive_doc_ids.append(doc_id)
            continue

        retrieved_docs.append(
            {
                "id": doc_id,
                "title": metadata.get("title", "Unknown Title"),
                "content": doc_text,
                "sensitive": sensitive,
                "score": float(distance),
            }
        )

        if len(retrieved_docs) >= k:
            break

    return retrieved_docs, blocked_sensitive_doc_ids