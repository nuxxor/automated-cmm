#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from cm_rag_utils import render_retrieved_context_limited, search_index

from qwen3_ft_utils import (
    choose_reference_free_cm_response,
    cleanup_cm_response,
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
    parser = argparse.ArgumentParser(description="Run the CM model with a deployment prompt.")
    parser.add_argument("--config", default="configs/qwen3_14b_deploy_default.json")
    parser.add_argument("--model-path", help="Adapter directory or model path to load.")
    parser.add_argument("--system-prompt", default="prompts/cm_deployment_prompt_v5.txt")
    parser.add_argument("--primer-json", help="Optional JSON list of few-shot chat messages to insert before the user turn.")
    parser.add_argument("--candidate-json", help="Optional JSON file with prompt_slots, generation, and cleanup_profile overrides.")
    parser.add_argument("--secondary-candidate-json", help="Optional second runtime candidate to compare and route against.")
    parser.add_argument("--message", help="Single user message to run.")
    parser.add_argument("--cases-json", help="JSON file containing a list of {id,message} objects.")
    parser.add_argument("--rag-index-json", help="Optional retrieval index JSON for project-specific docs.")
    parser.add_argument("--rag-project-name", help="Optional project name filter for retrieval.")
    parser.add_argument("--rag-top-k", type=int, default=4)
    parser.add_argument("--rag-min-score", type=float, default=0.0)
    parser.add_argument("--rag-max-context-chars", type=int, default=6000)
    parser.add_argument("--show-rag-context", action="store_true")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-cleanup", action="store_true")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--show-effective-config", action="store_true")
    parser.add_argument("--output", help="Optional JSON file for multi-case output.")
    return parser.parse_args()


def resolve_candidate_bundle(
    base_generation: dict[str, object],
    base_prompt: str,
    candidate_json: str | None,
) -> tuple[dict[str, object], dict | None, str, dict | None]:
    generation = dict(base_generation)
    cleanup_profile = None
    system_prompt = base_prompt
    candidate = None
    if candidate_json:
        candidate = load_json(candidate_json)
        generation.update(candidate.get("generation", {}))
        cleanup_profile = candidate.get("cleanup_profile")
        system_prompt = render_prompt_with_slots(base_prompt, candidate.get("prompt_slots", {}))
    return generation, cleanup_profile, system_prompt, candidate


def main() -> None:
    args = parse_args()
    if not args.message and not args.cases_json:
        raise SystemExit("Provide --message or --cases-json.")

    config = load_config(args.config)
    primer_messages = []
    rag_index = None

    model_path = resolve_model_path_from_config(config, args.model_path)
    base_generation = dict(config["generation"])
    base_prompt = load_text(args.system_prompt).strip()
    if args.primer_json:
        primer_messages = [msg for msg in load_json(args.primer_json) if msg.get("role") != "system"]
    if args.rag_index_json:
        rag_index = load_json(args.rag_index_json)
    generation, cleanup_profile, system_prompt, _ = resolve_candidate_bundle(
        base_generation,
        base_prompt,
        args.candidate_json,
    )
    if args.max_new_tokens is not None:
        generation["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        generation["temperature"] = args.temperature
    if args.top_p is not None:
        generation["top_p"] = args.top_p
    if args.top_k is not None:
        generation["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        generation["repetition_penalty"] = args.repetition_penalty
    if args.do_sample:
        generation["do_sample"] = True
    if args.no_sample:
        generation["do_sample"] = False

    if args.show_effective_config:
        secondary_generation = None
        secondary_cleanup = None
        secondary_prompt = None
        if args.secondary_candidate_json:
            secondary_generation, secondary_cleanup, secondary_prompt, _ = resolve_candidate_bundle(
                base_generation,
                base_prompt,
                args.secondary_candidate_json,
            )
        print(
            json.dumps(
                {
                    "model_path": str(model_path),
                    "generation": describe_generation(generation),
                    "cleanup_profile": cleanup_profile,
                    "secondary_generation": describe_generation(secondary_generation) if secondary_generation else None,
                    "secondary_cleanup_profile": secondary_cleanup,
                    "secondary_system_prompt": secondary_prompt,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
    _, backend, model, tokenizer = load_model_and_tokenizer(
        str(model_path),
        max_seq_length=config["training"]["max_seq_length"],
        load_in_4bit=True,
        full_finetuning=False,
    )
    prepare_model_for_inference(backend, model)
    torch.manual_seed(args.seed)

    def build_messages(message: str, *, project_name: str | None = None, prompt_text: str | None = None) -> tuple[list[dict[str, str]], list[dict]]:
        active_prompt = prompt_text or system_prompt
        messages: list[dict[str, str]] = [{"role": "system", "content": active_prompt}]
        rag_hits: list[dict] = []
        if rag_index is not None:
            rag_hits = search_index(
                rag_index,
                message,
                top_k=args.rag_top_k,
                project_name=project_name or args.rag_project_name,
                min_score=args.rag_min_score,
            )
            if rag_hits:
                rag_context = render_retrieved_context_limited(
                    rag_hits,
                    max_chars=args.rag_max_context_chars,
                )
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Project facts from local docs:\n"
                            f"{rag_context}\n\n"
                            "Use this context for factual details. If it does not answer the question, say so plainly instead of guessing."
                        ),
                    }
                )
        messages.extend(primer_messages)
        messages.append({"role": "user", "content": message})
        return messages, rag_hits

    if args.message:
        primary_messages, primary_rag_hits = build_messages(args.message)
        primary_response = generate_response(
            model,
            tokenizer,
            primary_messages,
            generation,
        )
        if not args.no_cleanup:
            primary_response = cleanup_cm_response(primary_response, cleanup_profile)
        primary_response = repair_cm_response_for_prompt(args.message, primary_response)
        response = primary_response
        if args.secondary_candidate_json:
            secondary_generation, secondary_cleanup, secondary_prompt, _ = resolve_candidate_bundle(
                base_generation,
                base_prompt,
                args.secondary_candidate_json,
            )
            secondary_messages, secondary_rag_hits = build_messages(args.message, prompt_text=secondary_prompt)
            secondary_response = generate_response(
                model,
                tokenizer,
                secondary_messages,
                secondary_generation,
            )
            if not args.no_cleanup:
                secondary_response = cleanup_cm_response(secondary_response, secondary_cleanup)
            secondary_response = repair_cm_response_for_prompt(args.message, secondary_response)
            selection = choose_reference_free_cm_response(
                args.message,
                [
                    {"label": "primary", "text": primary_response},
                    {"label": "secondary", "text": secondary_response},
                ],
            )
            response = selection["chosen_text"]
        if args.show_rag_context and rag_index is not None:
            payload = {"response": response, "rag_hits": primary_rag_hits}
            if args.secondary_candidate_json:
                payload["secondary_rag_hits"] = secondary_rag_hits
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        print(response)
        return

    cases = json.loads(Path(resolve_path(args.cases_json)).read_text())
    results = []
    for item in cases:
        message = item["message"]
        project_name = item.get("project")
        primary_messages, primary_rag_hits = build_messages(message, project_name=project_name)
        primary_response = generate_response(
            model,
            tokenizer,
            primary_messages,
            generation,
        )
        if not args.no_cleanup:
            primary_response = cleanup_cm_response(primary_response, cleanup_profile)
        primary_response = repair_cm_response_for_prompt(message, primary_response)
        response = primary_response
        result = {"id": item.get("id"), "message": message, "project": project_name, "response": response}
        if primary_rag_hits:
            result["rag_hits"] = primary_rag_hits
        if args.secondary_candidate_json:
            secondary_generation, secondary_cleanup, secondary_prompt, _ = resolve_candidate_bundle(
                base_generation,
                base_prompt,
                args.secondary_candidate_json,
            )
            secondary_messages, secondary_rag_hits = build_messages(
                message,
                project_name=project_name,
                prompt_text=secondary_prompt,
            )
            secondary_response = generate_response(
                model,
                tokenizer,
                secondary_messages,
                secondary_generation,
            )
            if not args.no_cleanup:
                secondary_response = cleanup_cm_response(secondary_response, secondary_cleanup)
            secondary_response = repair_cm_response_for_prompt(message, secondary_response)
            selection = choose_reference_free_cm_response(
                message,
                [
                    {"label": "primary", "text": primary_response},
                    {"label": "secondary", "text": secondary_response},
                ],
            )
            result.update(
                {
                    "response": selection["chosen_text"],
                    "chosen_candidate": selection["chosen_label"],
                    "candidate_scores": selection["candidates"],
                    "primary_response": primary_response,
                    "secondary_response": secondary_response,
                }
            )
            if secondary_rag_hits:
                result["secondary_rag_hits"] = secondary_rag_hits
        results.append(result)

    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
