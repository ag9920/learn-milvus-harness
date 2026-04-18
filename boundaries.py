from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

from core import ask, index_documents, load_embedding_fn
from chapters.sample_corpus import SAMPLE_CORPUS


@dataclass
class HarnessConfig:
    db: str
    collection: str
    input_dir: str | None
    chunk_size: int
    overlap: int
    topk: int
    candidate_multiplier: int
    alpha: float


@dataclass
class HarnessRuntime:
    client: MilvusClient
    embedding_fn: Any
    config: HarnessConfig

    def resolve_corpus(self) -> tuple[Path | None, list[dict[str, str]] | None]:
        if self.config.input_dir:
            return Path(self.config.input_dir), None
        return None, SAMPLE_CORPUS

    def index(self, rebuild: bool) -> Any:
        input_dir, documents = self.resolve_corpus()
        return index_documents(
            client=self.client,
            collection_name=self.config.collection,
            embedding_fn=self.embedding_fn,
            input_dir=input_dir,
            rebuild=rebuild,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            documents=documents,
        )

    def ask(self, question: str, filter_expr: str | None) -> list[dict[str, Any]]:
        return ask(
            client=self.client,
            collection_name=self.config.collection,
            embedding_fn=self.embedding_fn,
            question=question,
            topk=self.config.topk,
            filter_expr=filter_expr,
            candidate_multiplier=self.config.candidate_multiplier,
            alpha=self.config.alpha,
        )


def build_runtime(config: HarnessConfig) -> HarnessRuntime:
    return HarnessRuntime(
        client=MilvusClient(config.db),
        embedding_fn=load_embedding_fn(),
        config=config,
    )
