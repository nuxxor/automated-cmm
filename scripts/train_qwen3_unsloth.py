#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Any

from qwen3_ft_utils import (
    apply_chat_template,
    attach_lora_adapter,
    filter_supported_kwargs,
    get_unsloth_backend,
    load_config,
    load_model_and_tokenizer,
    model_has_lora_adapters,
    resolve_path,
    torch_precision_flags,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-14B on the curated CM dataset with Unsloth.")
    parser.add_argument("--config", default="configs/qwen3_14b_unsloth.json")
    parser.add_argument("--train-file", help="Prepared prompt-completion train JSONL.")
    parser.add_argument("--val-file", help="Prepared prompt-completion validation JSONL.")
    parser.add_argument("--output-dir", help="Override run output directory.")
    parser.add_argument("--resume-from-checkpoint", help="Resume training from a checkpoint.")
    parser.add_argument("--max-steps", type=int, help="Optional max_steps override for smoke runs.")
    return parser.parse_args()


def build_sft_config_kwargs(
    config: dict[str, Any],
    output_dir: Path,
    sft_signature_params: dict[str, inspect.Parameter],
    tokenizer: Any,
) -> dict[str, Any]:
    training = config["training"]
    bf16, fp16 = torch_precision_flags()

    sft_kwargs = {
        "output_dir": str(output_dir),
        "max_length": training["max_seq_length"],
        "max_seq_length": training["max_seq_length"],
        "per_device_train_batch_size": training["per_device_train_batch_size"],
        "per_device_eval_batch_size": training["per_device_eval_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "num_train_epochs": training["num_train_epochs"],
        "learning_rate": training["learning_rate"],
        "warmup_ratio": training["warmup_ratio"],
        "weight_decay": training["weight_decay"],
        "max_grad_norm": training["max_grad_norm"],
        "logging_steps": training["logging_steps"],
        "eval_steps": training["eval_steps"],
        "save_steps": training["save_steps"],
        "save_total_limit": training["save_total_limit"],
        "save_strategy": "steps",
        "report_to": "none",
        "bf16": bf16,
        "fp16": fp16,
        "packing": training["packing"],
        "seed": training["seed"],
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "optim": "adamw_torch_fused",
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
    }

    # TrainingArguments / SFTConfig renamed this across versions.
    if "evaluation_strategy" in sft_signature_params:
        sft_kwargs["evaluation_strategy"] = "steps"
    if "eval_strategy" in sft_signature_params:
        sft_kwargs["eval_strategy"] = "steps"

    return sft_kwargs


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    prepared_dir = resolve_path(config["prepared_data"]["dir"])
    train_file = resolve_path(
        args.train_file or prepared_dir / config["prepared_data"]["train_conversations"]
    )
    val_file = resolve_path(
        args.val_file or prepared_dir / config["prepared_data"]["val_conversations"]
    )
    output_dir = resolve_path(args.output_dir or config["paths"]["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(
            "Prepared dataset files are missing. Run scripts/prepare_qwen3_dataset.py first."
        )

    get_unsloth_backend()

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth.chat_templates import train_on_responses_only

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "validation": str(val_file)},
    )

    base_model_path = resolve_path(config["base_model"])
    existing_adapter = (base_model_path / "adapter_config.json").exists()

    _, backend, model, tokenizer = load_model_and_tokenizer(
        config["base_model"],
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=True,
        full_finetuning=False,
        is_trainable=True if existing_adapter else None,
    )
    if model_has_lora_adapters(model):
        print("Training will continue on the already-loaded LoRA adapter.")
    else:
        model = attach_lora_adapter(
            backend,
            model,
            r=config["training"]["lora_r"],
            lora_alpha=config["training"]["lora_alpha"],
            lora_dropout=config["training"]["lora_dropout"],
            target_modules=config["training"]["target_modules"],
            gradient_checkpointing=config["training"]["gradient_checkpointing"],
        )

    if tokenizer.eos_token in (None, "", "<EOS_TOKEN>") and tokenizer.eos_token_id is not None:
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
    if tokenizer.eos_token in (None, "", "<EOS_TOKEN>"):
        tokenizer.eos_token = "<|im_end|>"
    if tokenizer.pad_token is None or tokenizer.pad_token == "<|vision_pad|>":
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model, "config", None) is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"

    signature = inspect.signature(SFTConfig)
    sft_kwargs = build_sft_config_kwargs(config, output_dir, signature.parameters, tokenizer)
    if args.max_steps is not None:
        sft_kwargs["max_steps"] = args.max_steps
    sft_args = SFTConfig(**filter_supported_kwargs(SFTConfig, sft_kwargs))

    def formatting_func(example: dict[str, Any]) -> list[str]:
        messages = example["messages"]
        if messages and isinstance(messages[0], list):
            return [
                apply_chat_template(
                    tokenizer,
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
                for conversation in messages
            ]
        return [
            apply_chat_template(
                tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        ]

    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "formatting_func": formatting_func,
    }
    trainer_signature = inspect.signature(SFTTrainer)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)
    trainer = train_on_responses_only(
        trainer,
        tokenizer=tokenizer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = dict(train_result.metrics)
    metrics["train_samples"] = len(dataset["train"])
    metrics["eval_samples"] = len(dataset["validation"])
    metrics["base_model"] = config["base_model"]
    metrics["prepared_train_file"] = str(train_file)
    metrics["prepared_val_file"] = str(val_file)

    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "resolved_config.json", config)


if __name__ == "__main__":
    main()
