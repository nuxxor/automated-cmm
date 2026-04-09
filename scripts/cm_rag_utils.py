#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from qwen3_ft_utils import resolve_path


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]*")
WHITESPACE_RE = re.compile(r"\s+")
METADATA_FILENAMES = {"project.json", "metadata.json", "sources.json"}


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def infer_project_name(input_dir: Path) -> str:
    return input_dir.name.replace("-", "_").replace(" ", "_").lower()


def flatten_json_strings(payload: Any) -> list[str]:
    rows: list[str] = []
    if isinstance(payload, str):
        rows.append(payload)
    elif isinstance(payload, dict):
        for key, value in payload.items():
            nested = flatten_json_strings(value)
            if nested:
                rows.extend([f"{key}: {row}" for row in nested])
    elif isinstance(payload, list):
        for item in payload:
            rows.extend(flatten_json_strings(item))
    return rows


def read_source_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text())
        return "\n".join(flatten_json_strings(payload))
    return path.read_text()


def load_project_metadata(project_dir: Path, *, fallback_name: str) -> dict[str, Any]:
    metadata_path = project_dir / "project.json"
    if not metadata_path.exists():
        return {
            "project": fallback_name,
            "display_name": fallback_name,
            "aliases": [],
            "notes": "No project.json metadata file found.",
        }
    payload = json.loads(metadata_path.read_text())
    aliases = payload.get("aliases", [])
    if isinstance(aliases, str):
        aliases = [aliases]
    return {
        "project": payload.get("project", fallback_name),
        "display_name": payload.get("display_name", payload.get("project", fallback_name)),
        "aliases": aliases,
        "canonical_urls": payload.get("canonical_urls", []),
        "source_notes": payload.get("source_notes", ""),
        "owner": payload.get("owner", ""),
    }


def split_into_paragraphs(text: str) -> list[str]:
    paragraphs = [normalize_whitespace(part) for part in re.split(r"\n\s*\n", text)]
    return [part for part in paragraphs if part]


def chunk_paragraphs(paragraphs: list[str], *, chunk_words: int, overlap_words: int) -> list[str]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    overlap_words = max(0, overlap_words)

    for paragraph in paragraphs:
        paragraph_words = paragraph.split()
        paragraph_len = len(paragraph_words)
        if paragraph_len > chunk_words:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_words = 0
            start = 0
            step = max(1, chunk_words - overlap_words)
            while start < paragraph_len:
                piece = paragraph_words[start : start + chunk_words]
                chunks.append(" ".join(piece).strip())
                start += step
            continue

        if current and current_words + paragraph_len > chunk_words:
            chunks.append(" ".join(current).strip())
            if overlap_words > 0:
                tail_words = " ".join(current).split()[-overlap_words:]
                current = [" ".join(tail_words)] if tail_words else []
                current_words = len(tail_words)
            else:
                current = []
                current_words = 0

        current.append(paragraph)
        current_words += paragraph_len

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def build_rag_index(
    input_dir: str | Path,
    *,
    project_name: str | None = None,
    chunk_words: int = 160,
    overlap_words: int = 32,
    include_suffixes: tuple[str, ...] = (".md", ".txt", ".json"),
) -> dict[str, Any]:
    root = resolve_path(input_dir)
    project = project_name or infer_project_name(root)
    metadata = load_project_metadata(root, fallback_name=project)
    project = metadata.get("project") or project
    chunks: list[dict[str, Any]] = []
    doc_freqs: Counter[str] = Counter()

    for source_path in sorted(root.rglob("*")):
        if not source_path.is_file() or source_path.suffix.lower() not in include_suffixes:
            continue
        if source_path.name in METADATA_FILENAMES:
            continue
        text = normalize_whitespace(read_source_text(source_path))
        if not text:
            continue
        paragraphs = split_into_paragraphs(text)
        for chunk_idx, chunk_text in enumerate(
            chunk_paragraphs(paragraphs, chunk_words=chunk_words, overlap_words=overlap_words)
        ):
            terms = tokenize(chunk_text)
            if not terms:
                continue
            term_freqs = Counter(terms)
            doc_freqs.update(term_freqs.keys())
            chunks.append(
                {
                    "id": f"{project}-{len(chunks):04d}",
                    "project": project,
                    "source_path": str(source_path.relative_to(root)),
                    "chunk_index": chunk_idx,
                    "text": chunk_text,
                    "term_freqs": dict(term_freqs),
                    "length": len(terms),
                }
            )

    if not chunks:
        raise SystemExit(f"No supported documents found under {root}")

    avg_chunk_length = sum(chunk["length"] for chunk in chunks) / len(chunks)
    return {
        "project_name": project,
        "project_metadata": metadata,
        "source_root": str(root),
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "avg_chunk_length": avg_chunk_length,
        "doc_count": len(chunks),
        "doc_freqs": dict(doc_freqs),
        "chunks": chunks,
    }


def build_multi_project_rag_index(
    input_root: str | Path,
    *,
    chunk_words: int = 160,
    overlap_words: int = 32,
    include_suffixes: tuple[str, ...] = (".md", ".txt", ".json"),
) -> dict[str, Any]:
    root = resolve_path(input_root)
    project_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not project_dirs:
        raise SystemExit(f"No project directories found under {root}")

    combined_chunks: list[dict[str, Any]] = []
    combined_doc_freqs: Counter[str] = Counter()
    projects: list[str] = []
    project_metadata: dict[str, dict[str, Any]] = {}

    for project_dir in project_dirs:
        project_index = build_rag_index(
            project_dir,
            project_name=infer_project_name(project_dir),
            chunk_words=chunk_words,
            overlap_words=overlap_words,
            include_suffixes=include_suffixes,
        )
        projects.append(project_index["project_name"])
        project_metadata[project_index["project_name"]] = project_index.get("project_metadata", {})
        for chunk in project_index["chunks"]:
            chunk["source_path"] = f"{project_index['project_name']}/{chunk['source_path']}"
            combined_chunks.append(chunk)
            combined_doc_freqs.update(chunk["term_freqs"].keys())

    avg_chunk_length = sum(chunk["length"] for chunk in combined_chunks) / len(combined_chunks)
    return {
        "project_name": "multi_project",
        "projects": projects,
        "project_metadata": project_metadata,
        "source_root": str(root),
        "chunk_words": chunk_words,
        "overlap_words": overlap_words,
        "avg_chunk_length": avg_chunk_length,
        "doc_count": len(combined_chunks),
        "doc_freqs": dict(combined_doc_freqs),
        "chunks": combined_chunks,
    }


def score_chunk(index: dict[str, Any], query_terms: list[str], chunk: dict[str, Any]) -> float:
    if not query_terms:
        return 0.0
    term_freqs = chunk["term_freqs"]
    length = max(1, int(chunk["length"]))
    avgdl = max(1.0, float(index["avg_chunk_length"]))
    total_docs = max(1, int(index["doc_count"]))
    k1 = 1.5
    b = 0.75
    score = 0.0
    for term in query_terms:
        freq = int(term_freqs.get(term, 0))
        if freq <= 0:
            continue
        doc_freq = int(index["doc_freqs"].get(term, 0))
        idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        denom = freq + k1 * (1 - b + b * (length / avgdl))
        score += idf * ((freq * (k1 + 1)) / denom)
    return round(score, 6)


def project_matches(index: dict[str, Any], chunk_project: str, requested_project: str | None) -> bool:
    if not requested_project:
        return True
    requested = requested_project.lower()
    if chunk_project.lower() == requested:
        return True
    metadata = index.get("project_metadata", {}).get(chunk_project, {})
    aliases = [str(alias).lower() for alias in metadata.get("aliases", [])]
    display_name = str(metadata.get("display_name", "")).lower()
    return requested in aliases or requested == display_name


def search_index(
    index: dict[str, Any],
    query: str,
    *,
    top_k: int = 5,
    project_name: str | None = None,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    hits: list[dict[str, Any]] = []
    for chunk in index["chunks"]:
        if not project_matches(index, chunk["project"], project_name):
            continue
        score = score_chunk(index, query_terms, chunk)
        if score <= min_score:
            continue
        hits.append(
            {
                "id": chunk["id"],
                "project": chunk["project"],
                "source_path": chunk["source_path"],
                "chunk_index": chunk["chunk_index"],
                "score": score,
                "text": chunk["text"],
            }
        )
    hits.sort(key=lambda row: (-row["score"], row["source_path"], row["chunk_index"]))
    return hits[:top_k]


def render_retrieved_context(hits: list[dict[str, Any]]) -> str:
    blocks = render_retrieved_context_blocks(hits)
    return "\n\n".join(blocks).strip()


def render_retrieved_context_blocks(hits: list[dict[str, Any]]) -> list[str]:
    blocks: list[str] = []
    for idx, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{idx}] source={hit['source_path']} score={hit['score']:.3f}\n{hit['text']}"
        )
    return blocks


def render_retrieved_context_limited(hits: list[dict[str, Any]], *, max_chars: int | None = None) -> str:
    if not max_chars:
        return render_retrieved_context(hits)
    selected: list[str] = []
    total = 0
    for block in render_retrieved_context_blocks(hits):
        block_len = len(block)
        separator_len = 2 if selected else 0
        if selected and total + separator_len + block_len > max_chars:
            break
        if not selected and block_len > max_chars:
            selected.append(block[:max_chars].rstrip())
            break
        selected.append(block)
        total += separator_len + block_len
    return "\n\n".join(selected).strip()
