#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cm_rag_utils import search_index
from qwen3_ft_utils import load_json, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality on a local project-doc benchmark.")
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def score_case(case: dict, hits: list[dict]) -> dict:
    hit_sources = [hit["source_path"] for hit in hits]
    joined_text = " ".join(hit["text"] for hit in hits).lower()

    expected_sources = case.get("expected_sources", [])
    expected_keywords = [keyword.lower() for keyword in case.get("expected_keywords", [])]

    matched_sources = [
        source
        for source in expected_sources
        if any(source in candidate_source for candidate_source in hit_sources)
    ]
    matched_keywords = [keyword for keyword in expected_keywords if keyword in joined_text]

    source_score = 0.0
    if expected_sources:
        source_score = 4.0 * (len(matched_sources) / len(expected_sources))
    else:
        source_score = 4.0
    top1_bonus = 0.0
    if expected_sources and hit_sources and any(source in hit_sources[0] for source in expected_sources):
        top1_bonus = 2.0
    keyword_score = 0.0
    if expected_keywords:
        keyword_score = 4.0 * (len(matched_keywords) / len(expected_keywords))
    score = round(source_score + top1_bonus + keyword_score, 4)

    return {
        "id": case["id"],
        "project": case["project"],
        "query": case["query"],
        "score": score,
        "matched_sources": matched_sources,
        "matched_keywords": matched_keywords,
        "top_hit_source": hit_sources[0] if hit_sources else None,
        "hits": hits,
    }


def main() -> None:
    args = parse_args()
    index = load_json(args.index_json)
    benchmark = load_json(args.benchmark_json)

    case_results = []
    for case in benchmark["cases"]:
        hits = search_index(
            index,
            case["query"],
            top_k=args.top_k,
            project_name=case["project"],
            min_score=args.min_score,
        )
        case_results.append(score_case(case, hits))

    total_score = round(sum(case["score"] for case in case_results), 4)
    pass_count = sum(1 for case in case_results if case["score"] >= 8.0)
    source_hit_count = sum(1 for case in case_results if case["matched_sources"])
    top1_hit_count = sum(
        1
        for case, source_case in zip(case_results, benchmark["cases"], strict=True)
        if source_case.get("expected_sources")
        and case["top_hit_source"]
        and any(source in case["top_hit_source"] for source in source_case["expected_sources"])
    )

    report = {
        "index_path": str(resolve_path(args.index_json)),
        "benchmark_path": str(resolve_path(args.benchmark_json)),
        "case_count": len(case_results),
        "top_k": args.top_k,
        "min_score": args.min_score,
        "score": total_score,
        "average_score": round(total_score / max(1, len(case_results)), 4),
        "pass_rate_at_8": round(pass_count / max(1, len(case_results)), 4),
        "source_hit_rate": round(source_hit_count / max(1, len(case_results)), 4),
        "top1_source_hit_rate": round(top1_hit_count / max(1, len(case_results)), 4),
        "case_results": case_results,
    }
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"score": report["score"], "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
