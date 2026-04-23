import ollama

from config import MODEL_NAME, OLLAMA_OPTIONS


def ask_llm(question: str) -> str:
    """
    Baseline LLM pipeline (no retrieval).

    This function represents the baseline condition in the evaluation. This model generates an answer solely based on its knowledge without access to external documents.

    Purpose:
    - Provide a reference point for comparision with RAG and MCP pipelines
    - Evaluate how the model behaves without grounding

    Args:
    - question (str): Input question for the experiment

    Returns:
    - str: Generated answer from the LLM
    """
    response = ollama.chat(
        model=MODEL_NAME,
        options=OLLAMA_OPTIONS,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer the question clearly and concisely.\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )

    # Extract only the generated message content
    return response["message"]["content"].strip()


if __name__ == "__main__":
    # Simple test run for debugging purposes
    question = "What is retrieval augmented generation?"
    print(ask_llm(question))