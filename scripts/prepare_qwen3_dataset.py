#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from statistics import mean
from typing import Any

from qwen3_ft_utils import load_config, load_json, resolve_path, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare prompt-completion files for Qwen3 Unsloth SFT.")
    parser.add_argument(
        "--config",
        default="configs/qwen3_14b_unsloth.json",
        help="Relative or absolute path to the training config JSON.",
    )
    parser.add_argument("--train-json", help="Override source train JSON.")
    parser.add_argument("--val-json", help="Override source val JSON.")
    parser.add_argument("--outdir", help="Override prepared dataset directory.")
    return parser.parse_args()


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def normalize_example(example: dict[str, Any]) -> dict[str, Any]:
    messages = example["finetune_chat"]["messages"]
    if len(messages) < 3:
        raise ValueError(f"Example {example['id']} has fewer than 3 chat messages.")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Example {example['id']} does not end with an assistant message.")

    prompt = messages[:-1]
    completion = [messages[-1]]
    return {
        "id": example["id"],
        "timestamp": example["timestamp"],
        "category": example["category"],
        "secondary_categories": example.get("secondary_categories", []),
        "quality_score": example["quality_score"],
        "prompt": prompt,
        "completion": completion,
        "messages": messages,
        "assistant_text": completion[0]["content"],
        "prompt_preview": prompt[-1]["content"][:280] if prompt else "",
        "source_message_ids": example.get("source_message_ids", []),
    }


def summarize(split_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    categories = Counter(row["category"] for row in rows)
    prompt_tokens = [sum(whitespace_tokens(message["content"]) for message in row["prompt"]) for row in rows]
    completion_tokens = [whitespace_tokens(row["assistant_text"]) for row in rows]
    return {
        "split": split_name,
        "rows": len(rows),
        "category_counts": dict(categories),
        "avg_prompt_tokens": round(mean(prompt_tokens), 2) if prompt_tokens else 0,
        "avg_completion_tokens": round(mean(completion_tokens), 2) if completion_tokens else 0,
        "max_prompt_tokens": max(prompt_tokens) if prompt_tokens else 0,
        "max_completion_tokens": max(completion_tokens) if completion_tokens else 0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    train_json = resolve_path(args.train_json or config["raw_data"]["train_json"])
    val_json = resolve_path(args.val_json or config["raw_data"]["val_json"])
    outdir = resolve_path(args.outdir or config["prepared_data"]["dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    train_examples = [normalize_example(example) for example in load_json(train_json)]
    val_examples = [normalize_example(example) for example in load_json(val_json)]

    train_prompt_path = outdir / config["prepared_data"]["train_prompt_completion"]
    val_prompt_path = outdir / config["prepared_data"]["val_prompt_completion"]
    train_conv_path = outdir / config["prepared_data"]["train_conversations"]
    val_conv_path = outdir / config["prepared_data"]["val_conversations"]
    manifest_path = outdir / config["prepared_data"]["manifest"]

    train_prompt_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "quality_score": row["quality_score"],
            "prompt": row["prompt"],
            "completion": row["completion"],
        }
        for row in train_examples
    ]
    val_prompt_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "quality_score": row["quality_score"],
            "prompt": row["prompt"],
            "completion": row["completion"],
        }
        for row in val_examples
    ]

    train_conv_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "quality_score": row["quality_score"],
            "messages": row["messages"],
        }
        for row in train_examples
    ]
    val_conv_rows = [
        {
            "id": row["id"],
            "category": row["category"],
            "quality_score": row["quality_score"],
            "messages": row["messages"],
        }
        for row in val_examples
    ]

    manifest = {
        "config": config["_config_path"],
        "source_files": {
            "train_json": str(train_json),
            "val_json": str(val_json),
        },
        "prepared_files": {
            "train_prompt_completion": str(train_prompt_path),
            "val_prompt_completion": str(val_prompt_path),
            "train_conversations": str(train_conv_path),
            "val_conversations": str(val_conv_path),
        },
        "splits": [
            summarize("train", train_examples),
            summarize("val", val_examples),
        ],
    }

    write_jsonl(train_prompt_path, train_prompt_rows)
    write_jsonl(val_prompt_path, val_prompt_rows)
    write_jsonl(train_conv_path, train_conv_rows)
    write_jsonl(val_conv_path, val_conv_rows)
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
