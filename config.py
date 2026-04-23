"""
Global configuration for the prototype

This module centralizes all adjustable parameters used across the system, including model configuration, retrieval settings and experiment setup.

Purpose in the prototype:
- Ensure consistent settings across all pipelines
- Enable reproducibile experiments through deterministic parameters
- Provide a single location for modifying system behaviour
"""

MODEL_NAME = "llama3"

# Deterministic generation for fairer comparisons
OLLAMA_OPTIONS = {
    "temperature": 0,
    "seed": 42,
}

TOP_K = 3
NUM_RUNS = 3

RESULTS_DIR = "results"
RAW_RESULTS_FILE = "results/experiment_results.csv"
SUMMARY_OVERALL_FILE = "results/summary_overall.csv"
SUMMARY_BY_CATEGORY_FILE = "results/summary_by_category.csv"
SUMMARY_BY_QUESTION_FILE = "results/summary_by_question.csv"