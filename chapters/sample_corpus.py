"""
Built-in sample corpus for chapter exercises.

This keeps the repository self-contained and avoids mandatory external
text files for first-run tutorials.

`source` here is metadata for retrieval traceability only.
It is NOT a filesystem dependency.
"""

from __future__ import annotations

SAMPLE_CORPUS: list[dict[str, str]] = [
    {
        "subject": "ai",
        "source": "builtin://ai/rag_notes",
        "ext": "md",
        "text": (
            "In a RAG system, the vector database is responsible for candidate recall. "
            "The key metrics are retrieval quality, latency, and throughput. "
            "To keep production behavior stable, common optimizations include keeping chunk overlap, "
            "using two-stage retrieval with reranking, caching frequent queries, and applying metadata filters."
        ),
    },
    {
        "subject": "bio",
        "source": "builtin://bio/drug_discovery",
        "ext": "txt",
        "text": (
            "Typical AI applications in drug discovery include molecular generation, "
            "molecular property prediction, and target identification. "
            "During early-stage screening, machine learning can estimate toxicity and solubility. "
            "In practical retrieval systems, high-recall candidate generation is usually "
            "followed by task-aware reranking."
        ),
    },
]
