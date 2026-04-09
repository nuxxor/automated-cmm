#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cm_rag_utils import (
    build_multi_project_rag_index,
    build_rag_index,
    render_retrieved_context_limited,
    search_index,
)
from eval_project_rag_benchmark import score_case
from qwen3_ft_utils import load_json, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Config-driven local RAG pipeline runner.")
    parser.add_argument("--config", default="configs/rag_default.json")
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--query")
    parser.add_argument("--project-name")
    parser.add_argument("--output-json", help="Optional output path for query payload.")
    return parser.parse_args()


def build_index(config: dict) -> dict:
    chunking = config.get("chunking", {})
    if config.get("multi_project_root", False):
        index = build_multi_project_rag_index(
            config["knowledge_root"],
            chunk_words=int(chunking.get("chunk_words", 160)),
            overlap_words=int(chunking.get("overlap_words", 32)),
        )
    else:
        index = build_rag_index(
            config["knowledge_root"],
            project_name=config.get("project_name"),
            chunk_words=int(chunking.get("chunk_words", 160)),
            overlap_words=int(chunking.get("overlap_words", 32)),
        )
    output_path = resolve_path(config["index_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    return index


def load_or_build_index(config: dict, *, force_build: bool) -> dict:
    index_path = resolve_path(config["index_output"])
    if force_build or not index_path.exists():
        return build_index(config)
    return load_json(index_path)


def evaluate_index(config: dict, index: dict) -> dict:
    retrieval = config.get("retrieval", {})
    benchmark = load_json(config["benchmark_path"])
    top_k = int(retrieval.get("top_k", 3))
    min_score = float(retrieval.get("min_score", 0.0))

    case_results = []
    for case in benchmark["cases"]:
        hits = search_index(index, case["query"], top_k=top_k, project_name=case["project"], min_score=min_score)
        case_results.append(score_case(case, hits))

    total_score = round(sum(case["score"] for case in case_results), 4)
    report = {
        "config_path": str(resolve_path(config.get("_config_path", "configs/rag_default.json"))),
        "index_path": str(resolve_path(config["index_output"])),
        "benchmark_path": str(resolve_path(config["benchmark_path"])),
        "case_count": len(case_results),
        "top_k": top_k,
        "min_score": min_score,
        "score": total_score,
        "average_score": round(total_score / max(1, len(case_results)), 4),
        "pass_rate_at_8": round(sum(1 for case in case_results if case["score"] >= 8.0) / max(1, len(case_results)), 4),
        "case_results": case_results,
    }
    output_path = resolve_path(config["report_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def run_query(config: dict, index: dict, query: str, project_name: str | None, output_json: str | None) -> dict:
    retrieval = config.get("retrieval", {})
    hits = search_index(
        index,
        query,
        top_k=int(retrieval.get("top_k", 3)),
        project_name=project_name,
        min_score=float(retrieval.get("min_score", 0.0)),
    )
    payload = {
        "query": query,
        "project_name": project_name,
        "hits": hits,
        "rendered_context": render_retrieved_context_limited(
            hits,
            max_chars=int(retrieval.get("max_context_chars", 6000)),
        ),
    }
    if output_json:
        output_path = resolve_path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    config["_config_path"] = str(resolve_path(args.config))
    index = load_or_build_index(config, force_build=args.build_index)

    payload: dict[str, object] = {
        "index": {
            "path": str(resolve_path(config["index_output"])),
            "doc_count": index["doc_count"],
            "projects": index.get("projects", [index.get("project_name")]),
        }
    }
    if args.eval:
        payload["eval"] = evaluate_index(config, index)
    if args.query:
        payload["query"] = run_query(config, index, args.query, args.project_name, args.output_json)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
