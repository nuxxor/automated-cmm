#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations

from cm_autoresearch import score_evaluation
from qwen3_ft_utils import (
    cleanup_cm_response_with_trace,
    describe_generation,
    generate_response,
    load_config,
    load_json,
    load_model_and_tokenizer,
    load_text,
    prepare_model_for_inference,
    render_prompt_with_slots,
    resolve_model_path_from_config,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Head-to-head CM preference eval across prompt/runtime variants.")
    parser.add_argument("--config", default="configs/qwen3_14b_deploy_default.json")
    parser.add_argument("--model-path", help="Adapter directory or model path to load.")
    parser.add_argument("--benchmark-json", default="data/benchmarks/cm_style_benchmark_v1.json")
    parser.add_argument("--candidates-json", default="configs/cm_preference_candidates_v1.json")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", help="Optional markdown summary path.")
    parser.add_argument("--case-epsilon", type=float, default=0.05)
    parser.add_argument("--rank-on", choices=["cleaned", "raw"], default="cleaned")
    return parser.parse_args()


def load_candidate_bundle(config: dict, candidate_spec: dict) -> dict:
    generation = dict(config["generation"])
    cleanup_profile = None
    system_prompt = load_text(candidate_spec["system_prompt"]).strip()
    primer_messages = []

    candidate_json_path = candidate_spec.get("candidate_json")
    if candidate_json_path:
        candidate = load_json(candidate_json_path)
        generation.update(candidate.get("generation", {}))
        cleanup_profile = candidate.get("cleanup_profile")
        system_prompt = render_prompt_with_slots(system_prompt, candidate.get("prompt_slots", {}))

    primer_json_path = candidate_spec.get("primer_json")
    if primer_json_path:
        primer_messages = [msg for msg in load_json(primer_json_path) if msg.get("role") != "system"]

    return {
        "id": candidate_spec["id"],
        "label": candidate_spec.get("label", candidate_spec["id"]),
        "system_prompt": system_prompt,
        "generation": generation,
        "cleanup_profile": cleanup_profile,
        "primer_messages": primer_messages,
        "system_prompt_path": str(resolve_path(candidate_spec["system_prompt"])),
        "candidate_json_path": str(resolve_path(candidate_json_path)) if candidate_json_path else None,
        "primer_json_path": str(resolve_path(primer_json_path)) if primer_json_path else None,
    }


def rank_candidates(candidate_reports: list[dict], rank_on: str) -> list[dict]:
    score_key = "cleaned_score" if rank_on == "cleaned" else "raw_score"
    return sorted(candidate_reports, key=lambda item: (-item[score_key], item["label"]))


def build_pairwise(candidate_reports: list[dict], epsilon: float, *, case_key: str, score_key: str) -> dict:
    index = {report["id"]: report for report in candidate_reports}
    pairwise: dict[str, dict] = {}

    for left_id, right_id in combinations(index.keys(), 2):
        left = index[left_id]
        right = index[right_id]
        stats = {
            "scored_on": score_key,
            "left_id": left_id,
            "right_id": right_id,
            "left_label": left["label"],
            "right_label": right["label"],
            "left_case_wins": 0,
            "right_case_wins": 0,
            "ties": 0,
            "left_total_margin": 0.0,
            "right_total_margin": 0.0,
            "per_case": [],
        }
        left_cases = {item["id"]: item for item in left[case_key]}
        right_cases = {item["id"]: item for item in right[case_key]}
        for case_id in left_cases:
            left_case = left_cases[case_id]
            right_case = right_cases[case_id]
            margin = round(left_case["score"] - right_case["score"], 4)
            winner = "tie"
            if margin > epsilon:
                winner = left_id
                stats["left_case_wins"] += 1
                stats["left_total_margin"] += margin
            elif margin < -epsilon:
                winner = right_id
                stats["right_case_wins"] += 1
                stats["right_total_margin"] += abs(margin)
            else:
                stats["ties"] += 1
            stats["per_case"].append(
                {
                    "case_id": case_id,
                    "left_score": left_case["score"],
                    "right_score": right_case["score"],
                    "margin": margin,
                    "winner": winner,
                }
            )
        pairwise[f"{left_id}__vs__{right_id}"] = stats
    return pairwise


def build_markdown(report: dict) -> str:
    lines = [
        "# CM Preference Eval",
        "",
        f"Benchmark: `{report['benchmark_path']}`",
        f"Model path: `{report['model_path']}`",
        f"Ranked on: `{report['rank_on']}`",
        "",
        "## Ranking",
        "",
        "| Rank | Candidate | Raw | Cleaned | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for idx, candidate in enumerate(report["ranking"], 1):
        lines.append(
            f"| {idx} | {candidate['label']} | {candidate['raw_score']:.2f} | {candidate['cleaned_score']:.2f} | {candidate['cleanup_delta']:.2f} |"
        )

    for section_name, pairwise_key in (("Pairwise Cleaned", "pairwise_cleaned"), ("Pairwise Raw", "pairwise_raw")):
        lines.extend(["", f"## {section_name}", ""])
        for _, stats in report[pairwise_key].items():
            lines.append(f"### {stats['left_label']} vs {stats['right_label']}")
            lines.append("")
            lines.append(
                f"- case wins: `{stats['left_case_wins']}` vs `{stats['right_case_wins']}` with `{stats['ties']}` ties"
            )
            lines.append(
                f"- total margin: `{stats['left_total_margin']:.2f}` vs `{stats['right_total_margin']:.2f}`"
            )
            lines.append("")

    lines.extend(["## Candidate Notes", ""])
    for candidate in report["ranking"]:
        lines.append(f"### {candidate['label']}")
        lines.append("")
        lines.append(f"- raw score: `{candidate['raw_score']:.2f}`")
        lines.append(f"- cleaned score: `{candidate['cleaned_score']:.2f}`")
        lines.append(f"- cleanup delta: `{candidate['cleanup_delta']:.2f}`")
        lines.append(f"- prompt: `{candidate['system_prompt_path']}`")
        if candidate["candidate_json_path"]:
            lines.append(f"- runtime: `{candidate['candidate_json_path']}`")
        if candidate["primer_json_path"]:
            lines.append(f"- primer: `{candidate['primer_json_path']}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    benchmark = load_json(args.benchmark_json)
    candidate_specs = load_json(args.candidates_json)["candidates"]

    model_path = resolve_model_path_from_config(config, args.model_path)
    _, backend, model, tokenizer = load_model_and_tokenizer(
        str(model_path),
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=True,
        full_finetuning=False,
    )
    prepare_model_for_inference(backend, model)

    candidate_reports = []
    for candidate_spec in candidate_specs:
        bundle = load_candidate_bundle(config, candidate_spec)
        outputs = []
        for case in benchmark["cases"]:
            raw = generate_response(
                model,
                tokenizer,
                [
                    {"role": "system", "content": bundle["system_prompt"]},
                    *bundle["primer_messages"],
                    {"role": "user", "content": case["message"]},
                ],
                bundle["generation"],
            )
            cleanup_report = cleanup_cm_response_with_trace(raw, bundle["cleanup_profile"])
            outputs.append(
                {
                    "id": case["id"],
                    "message": case["message"],
                    "raw_response": raw,
                    "cleaned_response": cleanup_report["cleaned_text"],
                    "response": cleanup_report["cleaned_text"],
                    "cleanup_trace": cleanup_report["fired_rules"],
                    "cleanup_changed": cleanup_report["changed"],
                }
            )

        raw_scored = score_evaluation(benchmark, outputs, response_field="raw_response")
        cleaned_scored = score_evaluation(benchmark, outputs, response_field="cleaned_response")
        candidate_reports.append(
            {
                "id": bundle["id"],
                "label": bundle["label"],
                "score": cleaned_scored["score"] if args.rank_on == "cleaned" else raw_scored["score"],
                "raw_score": raw_scored["score"],
                "cleaned_score": cleaned_scored["score"],
                "cleanup_delta": round(cleaned_scored["score"] - raw_scored["score"], 4),
                "raw_global_penalties": raw_scored["global_penalties"],
                "cleaned_global_penalties": cleaned_scored["global_penalties"],
                "raw_case_results": raw_scored["case_results"],
                "cleaned_case_results": cleaned_scored["case_results"],
                "outputs": outputs,
                "system_prompt_path": bundle["system_prompt_path"],
                "candidate_json_path": bundle["candidate_json_path"],
                "primer_json_path": bundle["primer_json_path"],
                "generation": describe_generation(bundle["generation"]),
            }
        )

    ranking = rank_candidates(candidate_reports, args.rank_on)
    pairwise_cleaned = build_pairwise(
        candidate_reports,
        args.case_epsilon,
        case_key="cleaned_case_results",
        score_key="cleaned_score",
    )
    pairwise_raw = build_pairwise(
        candidate_reports,
        args.case_epsilon,
        case_key="raw_case_results",
        score_key="raw_score",
    )
    report = {
        "benchmark_path": str(resolve_path(args.benchmark_json)),
        "candidates_path": str(resolve_path(args.candidates_json)),
        "model_path": str(model_path),
        "rank_on": args.rank_on,
        "ranking": ranking,
        "pairwise_cleaned": pairwise_cleaned,
        "pairwise_raw": pairwise_raw,
    }

    output_json = resolve_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output_md:
        output_md = resolve_path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(report))

    print(
        json.dumps(
            {
                "best": ranking[0]["id"],
                "rank_on": args.rank_on,
                "score": ranking[0]["cleaned_score"] if args.rank_on == "cleaned" else ranking[0]["raw_score"],
                "output": str(output_json),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
