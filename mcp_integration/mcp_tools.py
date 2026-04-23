from typing import Any

from config import TOP_K
from rag.retrieve import retrieve


def retrieve_documents(query: str, k: int = TOP_K, allow_sensitive: bool = False) -> dict[str, Any]:
    """
    Retrieval backend function for the MCP tool interface.

    This function serves as the bridge between the MCP server layer and the underlying retrieval component of the prototype.
    It wrapes the core retrieval logic and formats the output into a structured response that can be returned through the MCP protocol.

    Purpose in the prototype:
    - Expose the retrieval system as a callable MCP tool
    - Separate business logic (retrieval) from protocol logic (MCP server)
    - Enable modular interaction between the language model and external data

    Args:
    - query (str): User question or query string
    - k (int): Number of documents to retrieve (Top-K)
    - allow_sensitive (bool): Whether sensitive documents may be returned

    Returns:
    - dict[str, Any]:
        - documments: List of retrieved documents
        - blocked_sensitive_doc_ids: List of filtered sensitive docuement IDs
        - document_count: Number of returned documents
    """
    docs, blocked_sensitive_doc_ids = retrieve(
        query=query,
        k=k,
        allow_sensitive=allow_sensitive,
    )

    return {
        "documents": docs,
        "blocked_sensitive_doc_ids": blocked_sensitive_doc_ids,
        "document_count": len(docs),
    }