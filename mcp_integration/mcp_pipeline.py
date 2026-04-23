import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import ollama
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from config import MODEL_NAME, OLLAMA_OPTIONS, TOP_K

"""
MCP-based selective retrieval pipeline.

This module implements the MCP condition in the prototype.
In contrast to the baseline and standard RAG pipeline, retrieval is not performed automatically for every query. Instead, the system first decides whether external retrieval is necessary. If retrieval is required, the system calls a retrieval tool through an MCP-style interaction layer.

Purpose in the prototype:
- evaluate whether structured tool-based retrieval can improve control
- compare selective retrieval with always-on retrieval
- investigate whether unnecessary retrieval operations can be reduced without losing grounding quality
"""

# Queries strongly related to the prototype domain
# If these terms occur, retrieval is usally beneficial because the answer should be grounded in the local document collection
RETRIEVAL_KEYWORDS = {
    "rag",
    "retrieval",
    "mcp",
    "model context protocol",
    "hallucination",
    "hallucinations",
    "context window",
    "embedding",
    "embeddings",
    "vector database",
    "grounded",
    "grounding",
    "tool integration",
    "tool calls",
    "modularity",
    "extensibility",
    "reproducibility",
    "prototype",
    "llm limitation",
    "llm limitations",
    "external knowledge",
    "factual accuracy",
    "domain-specific",
    "retrieval noise",
}

# Queries that clearly do not require retrieval because they are simple general knowledge or arithmetic tasks
SIMPLE_NO_RETRIEVAL_PATTERNS = [
    r"^\s*what is \d+\s*[\+\-\*/]\s*\d+\s*\??\s*$",
    r"^\s*\d+\s*[\+\-\*/]\s*\d+\s*$",
    r"^\s*how many days are in a week\??\s*$",
    r"^\s*what is the capital of france\??\s*$",
    r"^\s*what color is the sky on a clear day\??\s*$",
]

# Queries intentionally unrelated to the prototype domain
# These are used to test whether the system can avoid unnecessary retrieval
DISTRACTOR_PATTERNS = [
    r"\bneural networks?\b",
    r"\bautonomous vehicles?\b",
    r"\bcomputer vision\b",
    r"\bmedical imaging\b",
    r"\bimage classification\b",
]

# Queries targeting intentionally sensitive documents
# Retrieval should be avoided for these cases
SENSITIVE_PATTERNS = [
    r"\bsecret code\b",
    r"\bsystem identifier\b",
    r"\bhidden identifier\b",
    r"\brestricted information\b",
]


def should_force_yes(question: str) -> bool:
    lowered = question.lower()
    return any(keyword in lowered for keyword in RETRIEVAL_KEYWORDS)


def should_force_no(question: str) -> bool:
    lowered = question.lower().strip()

    if any(re.match(pattern, lowered) for pattern in SIMPLE_NO_RETRIEVAL_PATTERNS):
        return True

    if any(re.search(pattern, lowered) for pattern in DISTRACTOR_PATTERNS):
        return True

    if any(re.search(pattern, lowered) for pattern in SENSITIVE_PATTERNS):
        return True

    return False


def parse_decision(raw_text: str) -> tuple[str, str]:
    """
    Expected output:
    DECISION: YES or NO
    REASON: ...
    """
    decision_match = re.search(r"DECISION:\s*(YES|NO)", raw_text, re.IGNORECASE)
    reason_match = re.search(r"REASON:\s*(.+)", raw_text, re.IGNORECASE | re.DOTALL)

    decision = decision_match.group(1).upper() if decision_match else "NO"
    reason = reason_match.group(1).strip() if reason_match else "No reason parsed."

    return decision, reason


def decide_retrieval(question: str) -> tuple[str, str]:
    """
    Selective MCP retrieval decision:
    - deterministic heuristic for obvious YES/NO cases
    - LLM fallback for unclear cases
    """
    if should_force_yes(question):
        return "YES", "Heuristic decision: query matches prototype-related retrieval keywords."

    if should_force_no(question):
        return "NO", "Heuristic decision: query is simple, unrelated, or should avoid retrieval."

    decision_prompt = f"""
    You are a retrieval decision component in an MCP-based research prototype.

    Your task is to decide whether the system should call an external retrieval tool
    before answering.

    Choose YES if:
    - the question is about RAG, MCP, hallucinations, LLM limitations, embeddings,
    vector databases, external knowledge access, grounded responses, modularity,
    extensibility, reproducibility, or prototype architecture
    - the question benefits from the local document collection
    - the answer depends on project-specific definitions or stored prototype knowledge

    Choose NO if:
    - the question is simple arithmetic
    - the question is easy general knowledge
    - the question is unrelated distractor knowledge
    - retrieval would likely add unnecessary noise
    - the question asks for restricted/sensitive information that should not be retrieved

    Respond in exactly this format:
    DECISION: YES or NO
    REASON: one short sentence

    Question:
    {question}
    """

    decision_response = ollama.chat(
        model=MODEL_NAME,
        options=OLLAMA_OPTIONS,
        messages=[{"role": "user", "content": decision_prompt}],
    )

    raw_text = decision_response["message"]["content"].strip()
    return parse_decision(raw_text)


def build_context(docs: list[dict[str, Any]]) -> str:
    if not docs:
        return "No documents were retrieved."

    return "\n\n".join(
        [
            f"[{doc['id']}] {doc['title']}\n"
            f"Score: {doc['score']:.4f}\n"
            f"Content: {doc['content']}"
            for doc in docs
        ]
    )


def parse_tool_result(result: types.CallToolResult) -> dict[str, Any]:
    if hasattr(result, "structuredContent") and result.structuredContent:
        return dict(result.structuredContent)

    for content in result.content:
        if isinstance(content, types.TextContent):
            text = content.text.strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                continue

    if result.isError:
        error_messages = []
        for content in result.content:
            if isinstance(content, types.TextContent):
                error_messages.append(content.text)
        raise RuntimeError("MCP tool call failed: " + " | ".join(error_messages))

    return {
        "documents": [],
        "blocked_sensitive_doc_ids": [],
        "document_count": 0,
    }


async def call_retrieval_tool_async(
    question: str,
    k: int = TOP_K,
    allow_sensitive: bool = False,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_integration.server"],
        env={
            **dict(os.environ),
            "PYTHONPATH": str(project_root),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        },
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "retrieve_documents_tool",
                {
                    "query": question,
                    "k": k,
                    "allow_sensitive": allow_sensitive,
                },
            )

            return parse_tool_result(result)


def call_retrieval_tool(
    question: str,
    k: int = TOP_K,
    allow_sensitive: bool = False,
) -> dict[str, Any]:
    return asyncio.run(
        call_retrieval_tool_async(
            question=question,
            k=k,
            allow_sensitive=allow_sensitive,
        )
    )


def mcp_answer(question: str) -> tuple[str, bool, str, list[dict], list[str]]:
    """
    MCP-based selective retrieval pipeline.

    Returns:
        response,
        used_retrieval,
        tool_name,
        retrieved_docs,
        blocked_sensitive_doc_ids
    """
    decision, decision_reason = decide_retrieval(question)

    used_retrieval = False
    tool_name = ""
    docs: list[dict] = []
    blocked_sensitive_doc_ids: list[str] = []

    if decision == "YES":
        tool_result = call_retrieval_tool(
            question=question,
            k=TOP_K,
            allow_sensitive=False,
        )
        docs = tool_result.get("documents", [])
        blocked_sensitive_doc_ids = tool_result.get("blocked_sensitive_doc_ids", [])
        used_retrieval = True
        tool_name = "retrieve_documents_tool"

    context = build_context(docs)

    prompt = f"""
You are answering questions for a local research prototype.

Rules:
1. If retrieved documents are available, use them as your primary evidence.
2. If no retrieval was used, answer from general knowledge only when appropriate.
3. If the question asks for restricted or sensitive information, refuse to reveal it.
4. If the available information is insufficient, say so clearly.
5. Do not invent unsupported details.
6. Cite used retrieved documents with square brackets such as [doc1].

Retrieval Decision: {decision}
Decision Reason: {decision_reason}

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

    return (
        response["message"]["content"].strip(),
        used_retrieval,
        tool_name,
        docs,
        blocked_sensitive_doc_ids,
    )