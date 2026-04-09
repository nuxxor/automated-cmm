#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from cm_rag_utils import build_multi_project_rag_index, build_rag_index
from qwen3_ft_utils import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk local project docs into a lightweight retrieval index.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--project-name")
    parser.add_argument("--chunk-words", type=int, default=160)
    parser.add_argument("--overlap-words", type=int, default=32)
    parser.add_argument("--multi-project-root", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.multi_project_root:
        index = build_multi_project_rag_index(
            args.input_dir,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
    else:
        index = build_rag_index(
            args.input_dir,
            project_name=args.project_name,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
    output_path = resolve_path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "project_name": index["project_name"],
                "doc_count": index["doc_count"],
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
