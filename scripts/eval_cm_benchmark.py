#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter

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
    parser = argparse.ArgumentParser(description="Run a benchmark against the current CM runtime profile.")
    parser.add_argument("--config", default="configs/qwen3_14b_deploy_default.json")
    parser.add_argument("--model-path", help="Adapter directory or model path to load.")
    parser.add_argument("--system-prompt", default="prompts/cm_deployment_prompt_v5.txt")
    parser.add_argument("--primer-json", help="Optional JSON list of few-shot chat messages to insert before the user turn.")
    parser.add_argument("--candidate-json", default="runtime/cm_runtime_candidate_default.json")
    parser.add_argument("--secondary-candidate-json", help="Optional second runtime candidate for router-style eval.")
    parser.add_argument("--benchmark-json", required=True)
    parser.add_argument("--output", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score-raw-only", action="store_true")
    mode.add_argument("--score-cleaned-only", action="store_true")
    return parser.parse_args()


def summarize_cleanup(outputs: list[dict]) -> dict[str, object]:
    rule_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    changed_case_ids: list[str] = []

    for output in outputs:
        if output["cleanup_changed"]:
            changed_case_ids.append(output["id"])
        for row in output["cleanup_trace"]:
            rule_counter[row["rule"]] += 1
            category_counter[row["category"]] += 1

    return {
        "changed_cases": len(changed_case_ids),
        "changed_case_ids": changed_case_ids,
        "rule_counts": dict(sorted(rule_counter.items())),
        "category_counts": dict(sorted(category_counter.items())),
    }


def summarize_scores_by_category(scored: dict) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for row in scored["case_results"]:
        buckets.setdefault(row["category"], []).append(row["score"])
    return {key: round(sum(values) / len(values), 4) for key, values in sorted(buckets.items())}


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    primary_candidate = load_json(args.candidate_json)
    benchmark = load_json(args.benchmark_json)

    generation = dict(config["generation"])
    generation.update(primary_candidate.get("generation", {}))
    cleanup_profile = primary_candidate.get("cleanup_profile")
    generation_summary = describe_generation(generation)
    system_prompt = render_prompt_with_slots(load_text(args.system_prompt).strip(), primary_candidate.get("prompt_slots", {}))
    primer_messages = []
    if args.primer_json:
        primer_messages = [msg for msg in load_json(args.primer_json) if msg.get("role") != "system"]

    secondary_generation = None
    secondary_cleanup_profile = None
    secondary_system_prompt = None
    secondary_generation_summary = None
    if args.secondary_candidate_json:
        secondary_candidate = load_json(args.secondary_candidate_json)
        secondary_generation = dict(config["generation"])
        secondary_generation.update(secondary_candidate.get("generation", {}))
        secondary_cleanup_profile = secondary_candidate.get("cleanup_profile")
        secondary_system_prompt = render_prompt_with_slots(
            load_text(args.system_prompt).strip(),
            secondary_candidate.get("prompt_slots", {}),
        )
        secondary_generation_summary = describe_generation(secondary_generation)

    model_path = resolve_model_path_from_config(config, args.model_path)
    _, backend, model, tokenizer = load_model_and_tokenizer(
        str(model_path),
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=True,
        full_finetuning=False,
    )
    prepare_model_for_inference(backend, model)

    outputs = []
    for case in benchmark["cases"]:
        primary_raw = generate_response(
            model,
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                *primer_messages,
                {"role": "user", "content": case["message"]},
            ],
            generation,
        )
        primary_cleanup = cleanup_cm_response_with_trace(primary_raw, cleanup_profile)
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

        if secondary_generation is not None and secondary_cleanup_profile is not None and secondary_system_prompt is not None:
            secondary_raw = generate_response(
                model,
                tokenizer,
                [
                    {"role": "system", "content": secondary_system_prompt},
                    *primer_messages,
                    {"role": "user", "content": case["message"]},
                ],
                secondary_generation,
            )
            secondary_cleanup = cleanup_cm_response_with_trace(secondary_raw, secondary_cleanup_profile)
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

    score_mode = "both"
    primary_scored = cleaned_scored
    if args.score_raw_only:
        score_mode = "raw_only"
        primary_scored = raw_scored
    elif args.score_cleaned_only:
        score_mode = "cleaned_only"
        primary_scored = cleaned_scored

    report = {
        "benchmark_path": str(resolve_path(args.benchmark_json)),
        "candidate_path": str(resolve_path(args.candidate_json)),
        "score_mode": score_mode,
        "score": primary_scored["score"],
        "primary_score": primary_scored["score"],
        "raw_score": raw_scored["score"],
        "cleaned_score": cleaned_scored["score"],
        "raw_category_scores": summarize_scores_by_category(raw_scored),
        "cleaned_category_scores": summarize_scores_by_category(cleaned_scored),
        "cleanup_delta": round(cleaned_scored["score"] - raw_scored["score"], 4),
        "raw_global_penalties": raw_scored["global_penalties"],
        "cleaned_global_penalties": cleaned_scored["global_penalties"],
        "raw_case_results": raw_scored["case_results"],
        "cleaned_case_results": cleaned_scored["case_results"],
        "case_results": primary_scored["case_results"],
        "cleanup_case_deltas": cleanup_case_deltas,
        "cleanup_summary": summarize_cleanup(outputs),
        "generation": generation_summary,
        "secondary_generation": secondary_generation_summary,
        "outputs": outputs,
    }

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"score": report["score"], "output": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
