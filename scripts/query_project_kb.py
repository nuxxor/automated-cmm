#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cm_rag_utils import render_retrieved_context_limited, search_index
from qwen3_ft_utils import load_json, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a local project-doc retrieval index.")
    parser.add_argument("--index-json", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--project-name")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-context-chars", type=int)
    parser.add_argument("--output-json")
    parser.add_argument("--render-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = load_json(args.index_json)
    hits = search_index(
        index,
        args.query,
        top_k=args.top_k,
        project_name=args.project_name,
        min_score=args.min_score,
    )
    payload = {
        "index_path": str(resolve_path(args.index_json)),
        "query": args.query,
        "project_name": args.project_name or index.get("project_name"),
        "top_k": args.top_k,
        "min_score": args.min_score,
        "max_context_chars": args.max_context_chars,
        "hits": hits,
    }
    if args.render_context:
        payload["rendered_context"] = render_retrieved_context_limited(hits, max_chars=args.max_context_chars)

    if args.output_json:
        output_path = resolve_path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
