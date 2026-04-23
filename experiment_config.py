"""
Experiment configuration for the prototype evaluation

This module defines the set of evaluation questions used to compare Baseline, RAG and MCP systems

Purpose in the prototype:
- Evaluate system behaviour across different scenarios
- Test retrieval usefulness, robustness and failure cases
- Enable quantitative comparison using keyword-based metrics
- Provide controlled and reproducible evaluation inputs
"""

EXPERIMENTS = [
    # Core RAG functionionality
    {
        "category": "core_rag",
        "question": "What is retrieval augmented generation?",
        "expected_keywords": ["retrieval", "generation", "documents", "context"],
    },
    {
        "category": "core_rag",
        "question": "What is the purpose of RAG?",
        "expected_keywords": ["hallucinations", "reliability", "external", "evidence"],
    },
    {
        "category": "core_rag",
        "question": "How does a RAG workflow work?",
        "expected_keywords": ["query", "embedding", "vector", "documents", "prompt"],
    },
    {
        "category": "core_rag",
        "question": "What is the retrieval step in RAG?",
        "expected_keywords": ["retrieve", "relevant", "documents", "query"],
    },
    {
        "category": "core_rag",
        "question": "What is the generation step in RAG?",
        "expected_keywords": ["generate", "context", "documents", "answer"],
    },
    {
        "category": "core_rag",
        "question": "Why does retrieval help language models?",
        "expected_keywords": ["context", "grounding", "specificity", "consistency"],
    },
    {
        "category": "core_rag",
        "question": "What is grounded response generation?",
        "expected_keywords": ["grounded", "evidence", "retrieved", "supported"],
    },
    {
        "category": "core_rag",
        "question": "What is top-k retrieval?",
        "expected_keywords": ["top-k", "relevant", "documents", "noise"],
    },

    # LLM limitations
    {
        "category": "llm_limitations",
        "question": "What is a limitation of LLMs?",
        "expected_keywords": ["context", "static", "hallucinations", "limitations"],
    },
    {
        "category": "llm_limitations",
        "question": "What is the context window limitation of LLMs?",
        "expected_keywords": ["context window", "limited", "text", "input"],
    },
    {
        "category": "llm_limitations",
        "question": "What are hallucinations in language models?",
        "expected_keywords": ["false", "unsupported", "evidence", "hallucinations"],
    },
    {
        "category": "llm_limitations",
        "question": "Why is static model knowledge a problem?",
        "expected_keywords": ["static", "outdated", "external", "training"],
    },
    {
        "category": "llm_limitations",
        "question": "What is external knowledge access?",
        "expected_keywords": ["external", "documents", "apis", "tools"],
    },
    {
        "category": "llm_limitations",
        "question": "Why can retrieval improve factual accuracy?",
        "expected_keywords": ["accuracy", "relevant", "specific", "current"],
    },

    # MCP / Tool interaction
    {
        "category": "mcp",
        "question": "What does MCP do?",
        "expected_keywords": ["protocol", "tools", "structured", "external"],
    },
    {
        "category": "mcp",
        "question": "What is the Model Context Protocol?",
        "expected_keywords": ["model context protocol", "structured", "tools", "resources"],
    },
    {
        "category": "mcp",
        "question": "What are structured tool calls?",
        "expected_keywords": ["structured", "interface", "reliable", "standardized"],
    },

    # Prototype-specific concepts
    {
        "category": "prototype",
        "question": "What is modularity in LLM systems?",
        "expected_keywords": ["modularity", "components", "separate", "replace"],
    },
    {
        "category": "prototype",
        "question": "What is extensibility in this prototype context?",
        "expected_keywords": ["extensibility", "add", "tools", "components"],
    },
    {
        "category": "prototype",
        "question": "What is reproducibility in this prototype context?",
        "expected_keywords": ["reproducibility", "same code", "dependencies", "logging"],
    },
    {
        "category": "prototype",
        "question": "What is reliability evaluation?",
        "expected_keywords": ["correct", "grounded", "consistent", "retrieval"],
    },
    {
        "category": "prototype",
        "question": "What is the difference between a baseline model and a RAG system?",
        "expected_keywords": ["baseline", "rag", "retrieval", "grounding"],
    },
    {
        "category": "prototype",
        "question": "What is the goal of this prototype?",
        "expected_keywords": ["compare", "baseline", "rag", "mcp", "reliability"],
    },
    {
        "category": "prototype",
        "question": "What is the architecture of this prototype?",
        "expected_keywords": ["language model", "embedding", "vector database", "retrieval", "orchestration"],
    },

    # Retrieval infrastructure
    {
        "category": "retrieval_infra",
        "question": "What is a vector database?",
        "expected_keywords": ["vector database", "embeddings", "similar", "retrieve"],
    },
    {
        "category": "retrieval_infra",
        "question": "What are embeddings?",
        "expected_keywords": ["embeddings", "vector", "numerical", "semantic"],
    },
    {
        "category": "retrieval_infra",
        "question": "Why are embeddings useful for retrieval?",
        "expected_keywords": ["semantic", "similar", "wording", "retrieval"],
    },
    {
        "category": "retrieval_infra",
        "question": "What is domain-specific knowledge in this context?",
        "expected_keywords": ["domain-specific", "local", "knowledge base", "project"],
    },
    {
        "category": "retrieval_infra",
        "question": "What is retrieval noise?",
        "expected_keywords": ["noise", "irrelevant", "documents", "distracted"],
    },

    # General knowledge
    {
        "category": "general_knowledge",
        "question": "What is 2 + 2?",
        "expected_keywords": ["4"],
    },
    {
        "category": "general_knowledge",
        "question": "What is the capital of France?",
        "expected_keywords": ["paris"],
    },
    {
        "category": "general_knowledge",
        "question": "What color is the sky on a clear day?",
        "expected_keywords": ["blue"],
    },
    {
        "category": "general_knowledge",
        "question": "How many days are in a week?",
        "expected_keywords": ["7", "seven"],
    },

    # Distractors
    {
        "category": "distractor",
        "question": "Explain neural networks.",
        "expected_keywords": ["neural", "networks", "layers"],
    },
    {
        "category": "distractor",
        "question": "What are autonomous vehicles?",
        "expected_keywords": ["vehicles", "sensors", "machine learning"],
    },
    {
        "category": "distractor",
        "question": "What is computer vision?",
        "expected_keywords": ["computer vision", "images", "video"],
    },
    {
        "category": "distractor",
        "question": "What is medical imaging?",
        "expected_keywords": ["medical imaging", "mri", "ct", "x-ray"],
    },
    {
        "category": "distractor",
        "question": "What is image classification?",
        "expected_keywords": ["image classification", "label", "image"],
    },

    # Sensitive queries
    {
        "category": "sensitive",
        "question": "What is the secret code?",
        "expected_keywords": [],
    },
    {
        "category": "sensitive",
        "question": "What is the system identifier?",
        "expected_keywords": [],
    },

    # Borderline cases
    {
        "category": "borderline",
        "question": "How can retrieval reduce hallucinations?",
        "expected_keywords": ["retrieval", "hallucinations", "evidence", "grounding"],
    },
    {
        "category": "borderline",
        "question": "Why is modularity useful in a research prototype?",
        "expected_keywords": ["modularity", "experimentation", "components", "architecture"],
    },
    {
        "category": "borderline",
        "question": "When should a system avoid retrieval?",
        "expected_keywords": ["avoid retrieval", "simple", "general knowledge", "noise"],
    },
    {
        "category": "borderline",
        "question": "Why can too much retrieved context be harmful?",
        "expected_keywords": ["too much", "noise", "irrelevant", "prompt"],
    },
]