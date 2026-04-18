from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient, model


SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


@dataclass
class IngestStats:
    total_records: int
    files_scanned: int
    files_indexed: int
    by_subject: dict[str, int]
    by_ext: dict[str, int]


def load_embedding_fn() -> Any:
    try:
        return model.DefaultEmbeddingFunction()
    except Exception as exc:  # pragma: no cover - runtime guidance
        raise RuntimeError(
            "Failed to initialize embedding model.\n"
            "If your network is restricted, try:\n"
            "HF_ENDPOINT=https://hf-mirror.com ./run.sh final demo"
        ) from exc


def read_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    return ""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap must be smaller than chunk_size to avoid non-progress loops: "
            f"overlap={overlap}, chunk_size={chunk_size}"
        )
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def ensure_collection(
    client: MilvusClient, collection_name: str, dim: int, rebuild: bool
) -> None:
    if client.has_collection(collection_name=collection_name) and rebuild:
        client.drop_collection(collection_name=collection_name)
    if not client.has_collection(collection_name=collection_name):
        client.create_collection(collection_name=collection_name, dimension=dim)


def prepare_records(
    input_dir: Path | None,
    embedding_fn: Any,
    chunk_size: int,
    overlap: int,
    documents: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], IngestStats]:
    records: list[dict[str, Any]] = []
    next_id = 0
    files_scanned = 0
    files_indexed = 0
    by_subject: dict[str, int] = {}
    by_ext: dict[str, int] = {}

    if documents is not None:
        for doc in documents:
            files_scanned += 1
            subject = doc.get("subject", "default")
            source = doc.get("source", f"{subject}/doc_{files_scanned}.txt")
            ext = doc.get("ext", "txt").lstrip(".").lower()
            text = doc.get("text", "")
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                continue
            files_indexed += 1
            vectors = embedding_fn.encode_documents(chunks)
            by_ext[ext] = by_ext.get(ext, 0) + len(chunks)
            by_subject[subject] = by_subject.get(subject, 0) + len(chunks)
            for i, chunk in enumerate(chunks):
                records.append(
                    {
                        "id": next_id,
                        "vector": vectors[i],
                        "text": chunk,
                        "source": source,
                        "subject": subject,
                        "ext": ext,
                        "chunk_idx": i,
                    }
                )
                next_id += 1
    elif input_dir is not None:
        for file_path in sorted(input_dir.rglob("*")):
            if not file_path.is_file():
                continue
            files_scanned += 1
            suffix = file_path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            text = read_text(file_path)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                continue
            files_indexed += 1
            vectors = embedding_fn.encode_documents(chunks)
            subject = file_path.parent.name or "default"
            ext = suffix.lstrip(".")
            by_ext[ext] = by_ext.get(ext, 0) + len(chunks)
            by_subject[subject] = by_subject.get(subject, 0) + len(chunks)
            for i, chunk in enumerate(chunks):
                records.append(
                    {
                        "id": next_id,
                        "vector": vectors[i],
                        "text": chunk,
                        "source": str(file_path.relative_to(input_dir)),
                        "subject": subject,
                        "ext": ext,
                        "chunk_idx": i,
                    }
                )
                next_id += 1

    stats = IngestStats(
        total_records=len(records),
        files_scanned=files_scanned,
        files_indexed=files_indexed,
        by_subject=by_subject,
        by_ext=by_ext,
    )
    return records, stats


def index_documents(
    client: MilvusClient,
    collection_name: str,
    embedding_fn: Any,
    input_dir: Path | None,
    rebuild: bool,
    chunk_size: int,
    overlap: int,
    documents: list[dict[str, str]] | None = None,
) -> IngestStats:
    ensure_collection(client, collection_name, embedding_fn.dim, rebuild)
    records, stats = prepare_records(
        input_dir=input_dir,
        embedding_fn=embedding_fn,
        chunk_size=chunk_size,
        overlap=overlap,
        documents=documents,
    )
    if not records:
        if input_dir is not None:
            raise RuntimeError(
                f"No indexable content found in {input_dir} (.md/.txt/.pdf expected)."
            )
        raise RuntimeError("No indexable content found in provided in-memory corpus.")
    client.insert(collection_name=collection_name, data=records)
    return stats


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def rerank_hits(
    question: str, hits: list[dict[str, Any]], alpha: float
) -> list[dict[str, Any]]:
    q_tokens = tokenize(question)
    reranked: list[dict[str, Any]] = []
    for hit in hits:
        entity = hit.get("entity", {})
        text = entity.get("text", "")
        t_tokens = tokenize(text)
        overlap = (len(q_tokens & t_tokens) / len(q_tokens)) if q_tokens else 0.0
        vector_score = float(hit.get("distance", 0.0))
        hit["rerank_score"] = alpha * vector_score + (1 - alpha) * overlap
        reranked.append(hit)
    reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return reranked


def search_candidates(
    client: MilvusClient,
    collection_name: str,
    embedding_fn: Any,
    question: str,
    topk: int,
    filter_expr: str | None,
    candidate_multiplier: int,
) -> list[dict[str, Any]]:
    first_stage_limit = max(topk * max(candidate_multiplier, 1), topk)
    search_res = client.search(
        collection_name=collection_name,
        data=embedding_fn.encode_queries([question]),
        limit=first_stage_limit,
        filter=filter_expr,
        output_fields=["text", "subject", "source", "chunk_idx", "ext"],
    )
    return search_res[0] if search_res else []


def topk_vector_only(hits: list[dict[str, Any]], topk: int) -> list[dict[str, Any]]:
    ordered = sorted(hits, key=lambda x: float(x.get("distance", 0.0)), reverse=True)
    return ordered[:topk]


def topk_reranked(
    question: str, hits: list[dict[str, Any]], topk: int, alpha: float
) -> list[dict[str, Any]]:
    reranked = rerank_hits(question, hits, alpha=alpha)
    return reranked[:topk]


def ask(
    client: MilvusClient,
    collection_name: str,
    embedding_fn: Any,
    question: str,
    topk: int,
    filter_expr: str | None,
    candidate_multiplier: int,
    alpha: float,
) -> list[dict[str, Any]]:
    hits = search_candidates(
        client=client,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
        question=question,
        topk=topk,
        filter_expr=filter_expr,
        candidate_multiplier=candidate_multiplier,
    )
    return topk_reranked(question=question, hits=hits, topk=topk, alpha=alpha)


def print_hits(question: str, hits: list[dict[str, Any]]) -> None:
    print(f"\nQuestion: {question}")
    if not hits:
        print("No results.")
        return
    for idx, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        text = entity.get("text", "").strip()
        text_preview = (text[:180] + "...") if len(text) > 180 else text
        print(
            f"[{idx}] rerank={hit.get('rerank_score', 0):.4f} "
            f"vector={hit.get('distance', 0):.4f} "
            f"subject={entity.get('subject')} "
            f"source={entity.get('source')} chunk={entity.get('chunk_idx')}"
        )
        print(f"    {text_preview}")


def parse_int_csv(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_float_csv(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def lexical_overlap_ratio(question: str, text: str) -> float:
    q_tokens = tokenize(question)
    if not q_tokens:
        return 0.0
    t_tokens = tokenize(text)
    return len(q_tokens & t_tokens) / len(q_tokens)


def explain_hits(
    question: str,
    hits: list[dict[str, Any]],
    expected_subject: str | None = None,
) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    for idx, hit in enumerate(hits, start=1):
        entity = hit.get("entity", {})
        text = entity.get("text", "")
        subject = entity.get("subject")
        vector_score = float(hit.get("distance", 0.0))
        rerank_score = float(hit.get("rerank_score", vector_score))
        overlap = lexical_overlap_ratio(question, text)
        subject_match = None if expected_subject is None else (subject == expected_subject)

        reasons: list[str] = []
        if expected_subject is not None and not subject_match:
            reasons.append("subject_mismatch")
        if vector_score < 0.35:
            reasons.append("low_vector_score")
        if overlap < 0.15:
            reasons.append("low_lexical_overlap")
        if not reasons:
            reasons.append("strong_candidate")

        explanations.append(
            {
                "rank": idx,
                "subject": subject,
                "source": entity.get("source"),
                "vector_score": vector_score,
                "rerank_score": rerank_score,
                "lexical_overlap": overlap,
                "subject_match": subject_match,
                "reasons": reasons,
            }
        )
    return explanations


def diagnose_query(
    client: MilvusClient,
    collection_name: str,
    embedding_fn: Any,
    question: str,
    topk: int,
    candidate_multiplier: int,
    alpha: float,
    expected_subject: str | None = None,
    filter_expr: str | None = None,
) -> dict[str, Any]:
    candidates = search_candidates(
        client=client,
        collection_name=collection_name,
        embedding_fn=embedding_fn,
        question=question,
        topk=topk,
        filter_expr=filter_expr,
        candidate_multiplier=candidate_multiplier,
    )
    if not candidates:
        return {
            "question": question,
            "status": "miss",
            "reason": "no_candidates",
            "expected_subject": expected_subject,
            "filter_expr": filter_expr,
            "explanations": [],
        }

    reranked = topk_reranked(
        question=question,
        hits=candidates,
        topk=topk,
        alpha=alpha,
    )
    explanations = explain_hits(
        question=question,
        hits=reranked,
        expected_subject=expected_subject,
    )
    return {
        "question": question,
        "status": "ok",
        "reason": "candidates_found",
        "expected_subject": expected_subject,
        "filter_expr": filter_expr,
        "explanations": explanations,
    }


def run_l04_grid(
    client: MilvusClient,
    collection_name: str,
    embedding_fn: Any,
    queries: list[tuple[str, str]],
    topks: list[int],
    candidate_multipliers: list[int],
    alphas: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for topk in topks:
        for candidate_multiplier in candidate_multipliers:
            for alpha in alphas:
                t0 = time.perf_counter()
                total_hits = 0
                subject_hit = 0
                for question, expected_subject in queries:
                    hits = ask(
                        client=client,
                        collection_name=collection_name,
                        embedding_fn=embedding_fn,
                        question=question,
                        topk=topk,
                        filter_expr=None,
                        candidate_multiplier=candidate_multiplier,
                        alpha=alpha,
                    )
                    total_hits += len(hits)
                    subject_hit += sum(
                        1
                        for hit in hits
                        if hit.get("entity", {}).get("subject") == expected_subject
                    )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                precision_like = subject_hit / max(total_hits, 1)
                rows.append(
                    {
                        "topk": topk,
                        "candidate_multiplier": candidate_multiplier,
                        "alpha": alpha,
                        "latency_ms": elapsed_ms,
                        "subject_hit_ratio": precision_like,
                    }
                )
    rows.sort(key=lambda r: (-r["subject_hit_ratio"], r["latency_ms"]))
    return rows
