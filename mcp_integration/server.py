import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from mcp_integration.mcp_tools import retrieve_documents

"""
MCP server implementation for the retrieval component.

This module exposes the retrieval functionality of the prototype as a structured MCP tool. It acts as the protocol layer between the language model and the underlying retrieval system.

Purpose in the prototype:
- Provide a standardized interface for external tool access
- Enable structured communication between the LLM and retrieval system
- Separate protocol logic (MCP) from business logic (retrieval)

The server communicates via stdio and is stared as a subprocess by the client.
"""

# Initialize the MCP server instance
mcp_server = FastMCP("rag-retrieval-server")


@mcp_server.tool()
def retrieve_documents_tool(
    query: str,
    k: int = 3,
    allow_sensitive: bool = False,
) -> dict[str, Any]:
    
    """
    MCP-exposed retrieval tool.

    This function wrapes the retrieval backend and makes it accessible through the MCP protocol. It allows the language model to request relevant documents in a structured and controlled way

    Args:
    - query (str): User question or query string
    - k (int): Number of documents to retriev3e (Top-K)
    - allow_sensitive (bool): Whether sensitive documents may be returned

    Returns:
    - dict[str, Any]:
        - documents: Retrieved documents
        - blocked_sensitive_doc_ids: Filtered sensitive documents
        - documents_count: Number of returned documents
    """

    return retrieve_documents(
        query=query,
        k=k,
        allow_sensitive=allow_sensitive,
    )


if __name__ == "__main__":
    try:
        mcp_server.run(transport="stdio")
    except Exception as e:
        # stderr is okay for debugging stdio servers
        print(f"SERVER CRASH: {e}", file=sys.stderr, flush=True)
        raise