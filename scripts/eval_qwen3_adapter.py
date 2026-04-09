#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from qwen3_ft_utils import (
    apply_chat_template,
    load_config,
    load_jsonl,
    load_model_and_tokenizer,
    prepare_model_for_inference,
    resolve_path,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample predictions from a trained adapter or base model.")
    parser.add_argument("--config", default="configs/qwen3_14b_unsloth.json")
    parser.add_argument("--model-path", help="Adapter directory or model path to load.")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--output", help="JSONL file for predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    prepared_dir = resolve_path(config["prepared_data"]["dir"])
    split_file = (
        prepared_dir / config["prepared_data"]["val_prompt_completion"]
        if args.split == "val"
        else prepared_dir / config["prepared_data"]["train_prompt_completion"]
    )
    records = load_jsonl(split_file)

    model_path = resolve_path(args.model_path or (resolve_path(config["paths"]["run_dir"]) / "adapter"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    _, backend, model, tokenizer = load_model_and_tokenizer(
        str(model_path),
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=True,
        full_finetuning=False,
    )
    prepare_model_for_inference(backend, model)

    random.seed(args.seed)
    sample_rows = random.sample(records, min(args.samples, len(records)))

    import torch

    device = model.device
    generations = []
    for row in sample_rows:
        rendered = apply_chat_template(
            tokenizer,
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        encoded = tokenizer(rendered, return_tensors="pt").to(device)
        output_ids = model.generate(
            **encoded,
            max_new_tokens=config["generation"]["max_new_tokens"],
            temperature=config["generation"]["temperature"],
            top_p=config["generation"]["top_p"],
            top_k=config["generation"]["top_k"],
            do_sample=config["generation"]["do_sample"],
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output_ids[0][encoded["input_ids"].shape[1] :]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        generations.append(
            {
                "id": row["id"],
                "category": row["category"],
                "prompt": row["prompt"],
                "expected": row["completion"][0]["content"],
                "predicted": prediction,
            }
        )

    output_path = resolve_path(
        args.output or f"reports/eval_{args.split}_{Path(str(model_path)).name}.jsonl"
    )
    write_jsonl(output_path, generations)


if __name__ == "__main__":
    main()
