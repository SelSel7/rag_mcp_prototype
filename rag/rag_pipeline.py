import ollama

from config import MODEL_NAME, OLLAMA_OPTIONS, TOP_K
from rag.retrieve import retrieve

"""
RAG pipeline implementation for the prototype.

This module represents the always-on retrieval condition used in the evaluation. For every query, relevant documents are retrieved from the vector database and used as context for the language model.

Purpose in the prototype:
- Provide a baseline retrieval-enhanced system
- Compare against baseline LLM and MCP based selective retrieval
- Ensure that answers are grounded in retrieved documents
- Enable evaluation of factual grounding and source faithfulness
"""


def build_context(docs: list[dict]) -> str:
    """
    Format retrieved documents as prompt context for the LLM.

    Args:
    - docs (list[dict]): Retrieved documents from the vector database

    Returns:
    - str: Formatted context string for prompt injection
    """
    return "\n\n".join(
        [
            f"[{doc['id']}] {doc['title']}\n"
            f"Score: {doc['score']:.4f}\n"
            f"Content: {doc['content']}"
            for doc in docs
        ]
    )


def rag_answer(question: str) -> tuple[str, list[dict], list[str]]:
    """
    Execute the RAG pipeline for a given query

    Workflow:
    1. Retrieve relevant documents from the vector databse
    2. Format retrieved documents as prompt context
    3. Generate an answer using the LLM with injected context

    Args:
    - question (str): Input query

    Returns:
    - tuple:
        - response (str): Generated answer
        - retrieved_docs (list[dict]): Retrieved documents
        - blocked_sensitive_doc_ids (list[str]): Filtered sensitive documents
    """
    docs, blocked_sensitive_doc_ids = retrieve(
        question,
        k=TOP_K,
        allow_sensitive=False,
    )

    context = build_context(docs)

    prompt = f"""
You are answering questions for a local research prototype.

Use the retrieved documents as your primary evidence.
Rules:
1. Base the answer only on the retrieved documents whenever possible.
2. If the documents do not contain enough information, say so clearly.
3. Do not invent unsupported details.
4. When you use a document, cite its document ID in square brackets, for example [doc1].
5. Do not reveal restricted or sensitive information.

Retrieved Documents:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        options=OLLAMA_OPTIONS,
        messages=[{"role": "user", "content": prompt}],
    )

    return response["message"]["content"].strip(), docs, blocked_sensitive_doc_ids