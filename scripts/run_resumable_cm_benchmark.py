#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cm_autoresearch import score_evaluation
from qwen3_ft_utils import (
    choose_reference_free_cm_response,
    cleanup_cm_response_with_trace,
    describe_generation,
    generate_response,
    load_config,
    load_json,
    load_model_and_tokenizer,
    load_text,
    prepare_model_for_inference,
    repair_cm_response_for_prompt,
    render_prompt_with_slots,
    resolve_model_path_from_config,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a resumable CM benchmark with periodic checkpoints.")
    parser.add_argument("--config", default="configs/qwen3_14b_deploy_default.json")
    parser.add_argument("--model-path", help="Adapter directory or model path to load.")
    parser.add_argument("--system-prompt", default="prompts/cm_deployment_prompt_v5.txt")
    parser.add_argument("--primer-json", help="Optional JSON list of few-shot chat messages to insert before the user turn.")
    parser.add_argument("--candidate-json", default="runtime/cm_runtime_candidate_default.json")
    parser.add_argument("--secondary-candidate-json", help="Optional second runtime candidate for router-style eval.")
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-json", help="Optional checkpoint path. Defaults to <output>.checkpoint.json")
    parser.add_argument("--save-every", type=int, default=12)
    parser.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint and start from scratch.")
    return parser.parse_args()


def summarize_cleanup(outputs: list[dict]) -> dict[str, object]:
    rule_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    changed_ids: list[str] = []
    for row in outputs:
        if row["cleanup_changed"]:
            changed_ids.append(row["id"])
        for item in row["cleanup_trace"]:
            rule_counts[item["rule"]] = rule_counts.get(item["rule"], 0) + 1
            category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
    return {
        "changed_cases": len(changed_ids),
        "changed_case_ids": changed_ids,
        "rule_counts": dict(sorted(rule_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
    }


def summarize_scores_by_category(scored: dict) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in scored["case_results"]:
        buckets.setdefault(row["category"], []).append(row["score"])
    return {key: round(sum(values) / len(values), 4) for key, values in sorted(buckets.items())}


def resolve_candidate_bundle(config: dict, system_prompt_path: str, candidate_json_path: str | None) -> dict:
    generation = dict(config["generation"])
    cleanup_profile = None
    system_prompt = load_text(system_prompt_path).strip()
    candidate_path = None
    if candidate_json_path:
        candidate_path = resolve_path(candidate_json_path)
        candidate = load_json(candidate_json_path)
        generation.update(candidate.get("generation", {}))
        cleanup_profile = candidate.get("cleanup_profile")
        system_prompt = render_prompt_with_slots(system_prompt, candidate.get("prompt_slots", {}))
    return {
        "generation": generation,
        "cleanup_profile": cleanup_profile,
        "system_prompt": system_prompt,
        "candidate_path": str(candidate_path) if candidate_path else None,
        "generation_summary": describe_generation(generation),
    }


def write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    benchmark = load_json(args.benchmark_json)
    primary = resolve_candidate_bundle(config, args.system_prompt, args.candidate_json)
    secondary = (
        resolve_candidate_bundle(config, args.system_prompt, args.secondary_candidate_json)
        if args.secondary_candidate_json
        else None
    )
    primer_messages = []
    if args.primer_json:
        primer_messages = [msg for msg in load_json(args.primer_json) if msg.get("role") != "system"]

    output_path = resolve_path(args.output)
    checkpoint_path = resolve_path(args.checkpoint_json) if args.checkpoint_json else output_path.with_suffix(output_path.suffix + ".checkpoint.json")

    outputs: list[dict] = []
    completed_ids: set[str] = set()
    if checkpoint_path.exists() and not args.fresh:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("benchmark_path") == str(resolve_path(args.benchmark_json)):
            outputs = checkpoint.get("outputs", [])
            completed_ids = {row["id"] for row in outputs}

    model_path = resolve_model_path_from_config(config, args.model_path)
    _, backend, model, tokenizer = load_model_and_tokenizer(
        str(model_path),
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=True,
        full_finetuning=False,
    )
    prepare_model_for_inference(backend, model)

    completed_since_save = 0
    for case in benchmark["cases"]:
        if case["id"] in completed_ids:
            continue

        primary_raw = generate_response(
            model,
            tokenizer,
            [
                {"role": "system", "content": primary["system_prompt"]},
                *primer_messages,
                {"role": "user", "content": case["message"]},
            ],
            primary["generation"],
        )
        primary_cleanup = cleanup_cm_response_with_trace(primary_raw, primary["cleanup_profile"])
        primary_cleanup["cleaned_text"] = repair_cm_response_for_prompt(case["message"], primary_cleanup["cleaned_text"])

        chosen_label = "primary"
        raw = primary_raw
        cleanup_report = primary_cleanup
        cleaned = primary_cleanup["cleaned_text"]
        output_row = {
            "id": case["id"],
            "message": case["message"],
            "primary_raw_response": primary_raw,
            "primary_cleaned_response": primary_cleanup["cleaned_text"],
        }

        if secondary:
            secondary_raw = generate_response(
                model,
                tokenizer,
                [
                    {"role": "system", "content": secondary["system_prompt"]},
                    *primer_messages,
                    {"role": "user", "content": case["message"]},
                ],
                secondary["generation"],
            )
            secondary_cleanup = cleanup_cm_response_with_trace(secondary_raw, secondary["cleanup_profile"])
            secondary_cleanup["cleaned_text"] = repair_cm_response_for_prompt(case["message"], secondary_cleanup["cleaned_text"])
            selection = choose_reference_free_cm_response(
                case["message"],
                [
                    {"label": "primary", "text": primary_cleanup["cleaned_text"]},
                    {"label": "secondary", "text": secondary_cleanup["cleaned_text"]},
                ],
            )
            chosen_label = selection["chosen_label"]
            output_row.update(
                {
                    "secondary_raw_response": secondary_raw,
                    "secondary_cleaned_response": secondary_cleanup["cleaned_text"],
                    "chosen_candidate": chosen_label,
                    "candidate_scores": selection["candidates"],
                }
            )
            if chosen_label == "secondary":
                raw = secondary_raw
                cleanup_report = secondary_cleanup
                cleaned = secondary_cleanup["cleaned_text"]

        output_row.update(
            {
                "raw_response": raw,
                "cleaned_response": cleaned,
                "response": cleaned,
                "cleanup_trace": cleanup_report["fired_rules"],
                "cleanup_changed": cleanup_report["changed"],
                "style_rule_count": cleanup_report["style_rule_count"],
                "safety_rule_count": cleanup_report["safety_rule_count"],
            }
        )
        outputs.append(output_row)
        completed_ids.add(case["id"])
        completed_since_save += 1

        if completed_since_save >= args.save_every:
            write_checkpoint(
                checkpoint_path,
                {
                    "benchmark_path": str(resolve_path(args.benchmark_json)),
                    "output_path": str(output_path),
                    "completed_cases": len(outputs),
                    "total_cases": len(benchmark["cases"]),
                    "outputs": outputs,
                },
            )
            completed_since_save = 0

    raw_scored = score_evaluation(benchmark, outputs, response_field="raw_response")
    cleaned_scored = score_evaluation(benchmark, outputs, response_field="cleaned_response")
    cleanup_case_deltas = []
    for raw_case, cleaned_case in zip(raw_scored["case_results"], cleaned_scored["case_results"], strict=True):
        cleanup_case_deltas.append(
            {
                "id": raw_case["id"],
                "category": raw_case["category"],
                "raw_score": raw_case["score"],
                "cleaned_score": cleaned_case["score"],
                "cleanup_delta": round(cleaned_case["score"] - raw_case["score"], 4),
            }
        )

    report = {
        "benchmark_path": str(resolve_path(args.benchmark_json)),
        "candidate_path": primary["candidate_path"],
        "secondary_candidate_path": secondary["candidate_path"] if secondary else None,
        "score_mode": "both",
        "score": cleaned_scored["score"],
        "primary_score": cleaned_scored["score"],
        "raw_score": raw_scored["score"],
        "cleaned_score": cleaned_scored["score"],
        "raw_category_scores": summarize_scores_by_category(raw_scored),
        "cleaned_category_scores": summarize_scores_by_category(cleaned_scored),
        "cleanup_delta": round(cleaned_scored["score"] - raw_scored["score"], 4),
        "raw_global_penalties": raw_scored["global_penalties"],
        "cleaned_global_penalties": cleaned_scored["global_penalties"],
        "raw_case_results": raw_scored["case_results"],
        "cleaned_case_results": cleaned_scored["case_results"],
        "case_results": cleaned_scored["case_results"],
        "cleanup_case_deltas": cleanup_case_deltas,
        "cleanup_summary": summarize_cleanup(outputs),
        "generation": primary["generation_summary"],
        "secondary_generation": secondary["generation_summary"] if secondary else None,
        "outputs": outputs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(json.dumps({"score": report["score"], "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
