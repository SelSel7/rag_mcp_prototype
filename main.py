from baseline.baseline_llm import ask_llm
from mcp_integration.mcp_pipeline import mcp_answer
from rag.rag_pipeline import rag_answer

"""
Simple demonstration script for the prototype.

This script runs a single query through all three systems.

Purpose in the prototype:
- Quick manual testing of system behaviour
- Debugging and inspection of retrieved documents
- Demonstration of differences between pipelines
"""

question = "What is retrieval augmented generation?"

print("Question:")
print(question)

print("\nBaseline:")
print(ask_llm(question))

print("\nRAG:")
rag_response, rag_docs, rag_blocked = rag_answer(question)
print(rag_response)

print("\nRAG Retrieved Docs:")
for doc in rag_docs:
    print(f"- {doc['id']} | {doc['title']} | score={doc['score']:.4f}")

print("\nRAG Blocked Sensitive Docs:")
print(rag_blocked)

print("\nMCP:")
(
    mcp_response,
    mcp_used_retrieval,
    mcp_tool_name,
    mcp_docs,
    mcp_blocked,
) = mcp_answer(question)
print(mcp_response)

print("\nMCP Tool:")
print(mcp_tool_name)

print("\nMCP Used Retrieval:")
print(mcp_used_retrieval)

print("\nMCP Retrieved Docs:")
for doc in mcp_docs:
    print(f"- {doc['id']} | {doc['title']} | score={doc['score']:.4f}")

print("\nMCP Blocked Sensitive Docs:")
print(mcp_blocked)