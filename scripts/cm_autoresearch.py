#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

from qwen3_ft_utils import (
    cleanup_cm_response,
    generate_response,
    load_config,
    load_json,
    load_model_and_tokenizer,
    load_text,
    prepare_model_for_inference,
    render_prompt_with_slots,
    resolve_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous prompt/decode research loop for the CM model.")
    parser.add_argument("--config", default="configs/cm_autoresearch_v1.json")
    parser.add_argument("--max-iterations", type=int, help="Override config search.max_iterations. 0 means run forever.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing state and start from scratch.")
    return parser.parse_args()


def candidate_signature(candidate: dict[str, Any]) -> str:
    payload = json.dumps(candidate, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def render_prompt(seed_prompt: str, candidate: dict[str, Any]) -> str:
    return render_prompt_with_slots(seed_prompt, candidate["prompt_slots"])


def sentence_count(text: str) -> int:
    parts = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text.strip()) if chunk.strip()]
    return len(parts) if parts else (1 if text.strip() else 0)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
    return parts[0].strip().lower() if parts and parts[0].strip() else text.strip().lower()


def first_three_words(text: str) -> str:
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    return " ".join(words[:3])


def score_case(case: dict[str, Any], response: str, global_cfg: dict[str, Any]) -> dict[str, Any]:
    score = 0.0
    notes: list[str] = []
    matches: dict[str, bool] = {}

    wc = word_count(response)
    sc = sentence_count(response)

    if response.strip():
        score += 0.5
    else:
        notes.append("empty")

    if global_cfg["min_words"] <= wc <= global_cfg["max_words"]:
        score += 1.0
    else:
        penalty = 1.25 if wc < global_cfg["min_words"] else 1.0
        score -= penalty
        notes.append(f"word_count={wc}")

    max_sentences = case.get("max_sentences")
    if max_sentences is None:
        max_sentences = global_cfg["max_sentences"]
    if sc <= max_sentences:
        score += 0.75
    else:
        score -= 0.5 * (sc - max_sentences)
        notes.append(f"sentence_count={sc}")

    for group in case.get("required_groups", []):
        hit = any(re.search(pattern, response, flags=re.IGNORECASE) for pattern in group["patterns"])
        matches[group["label"]] = hit
        if hit:
            score += float(group["weight"])
        else:
            score -= float(group["weight"]) * 1.1
            notes.append(f"missing:{group['label']}")

    for group in case.get("preferred_groups", []):
        hit = any(re.search(pattern, response, flags=re.IGNORECASE) for pattern in group["patterns"])
        matches[group["label"]] = hit
        if hit:
            score += float(group["weight"])

    for pattern in global_cfg.get("forbidden_patterns", []):
        if re.search(pattern, response, flags=re.IGNORECASE):
            score -= 2.0
            notes.append(f"forbidden:{pattern}")

    for pattern in case.get("forbidden_patterns", []):
        if re.search(pattern, response, flags=re.IGNORECASE):
            score -= 2.0
            notes.append(f"forbidden:{pattern}")

    if case.get("expects_question") is True:
        if "?" in response:
            score += 0.75
        else:
            score -= 0.75
            notes.append("missing_question")

    if case.get("disallow_question") is True:
        if "?" in response:
            score -= 0.5
            notes.append("unexpected_question")
        else:
            score += 0.25

    return {
        "score": round(score, 4),
        "word_count": wc,
        "sentence_count": sc,
        "matches": matches,
        "notes": notes,
    }


def score_evaluation(
    benchmark: dict[str, Any],
    outputs: list[dict[str, Any]],
    *,
    response_field: str = "response",
) -> dict[str, Any]:
    global_cfg = benchmark["global"]
    total_score = 0.0
    case_results = []

    phrase_counter: dict[str, int] = {phrase: 0 for phrase in global_cfg.get("exact_phrase_penalties", {})}
    opening_counter: dict[str, int] = {}
    trigram_counter: dict[str, int] = {}

    for case, output in zip(benchmark["cases"], outputs, strict=True):
        response = output[response_field]
        case_result = score_case(case, response, global_cfg)
        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "message": case["message"],
                "evaluated_field": response_field,
                "response": response,
                **case_result,
            }
        )
        total_score += case_result["score"]

        opening = first_sentence(response)
        if opening:
            opening_counter[opening] = opening_counter.get(opening, 0) + 1
        trigram = first_three_words(response)
        if trigram:
            trigram_counter[trigram] = trigram_counter.get(trigram, 0) + 1

        lower = response.lower()
        for phrase in phrase_counter:
            phrase_counter[phrase] += lower.count(phrase.lower())

    penalties = []
    duplicate_opening_penalty = float(global_cfg.get("duplicate_opening_penalty", 0.0))
    for opening, count in opening_counter.items():
        if count > 1:
            penalty = duplicate_opening_penalty * (count - 1)
            total_score -= penalty
            penalties.append({"type": "duplicate_opening", "opening": opening, "count": count, "penalty": round(penalty, 4)})

    for trigram, count in trigram_counter.items():
        if trigram and count > 2:
            penalty = 0.4 * (count - 2)
            total_score -= penalty
            penalties.append({"type": "duplicate_trigram", "opening": trigram, "count": count, "penalty": round(penalty, 4)})

    for phrase, count in phrase_counter.items():
        if count > 1:
            penalty = float(global_cfg["exact_phrase_penalties"][phrase]) * (count - 1)
            total_score -= penalty
            penalties.append({"type": "phrase_repeat", "phrase": phrase, "count": count, "penalty": round(penalty, 4)})

    return {
        "score": round(total_score, 4),
        "case_results": case_results,
        "global_penalties": penalties,
    }


def evaluate_candidate(
    model: Any,
    tokenizer: Any,
    benchmark: dict[str, Any],
    seed_prompt: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = render_prompt(seed_prompt, candidate)
    outputs = []
    for case in benchmark["cases"]:
        raw = generate_response(
            model,
            tokenizer,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case["message"]},
            ],
            candidate["generation"],
        )
        cleaned = cleanup_cm_response(raw, candidate["cleanup_profile"])
        outputs.append({"id": case["id"], "message": case["message"], "raw_response": raw, "response": cleaned})

    scored = score_evaluation(benchmark, outputs)
    scored["outputs"] = outputs
    scored["system_prompt"] = system_prompt
    return scored


def seed_candidate(config: dict[str, Any]) -> dict[str, Any]:
    candidate = {
        "prompt_slots": {slot: "" for slot in config["prompt_bias_slots"]},
        "generation": dict(config["generation_seed"]),
        "cleanup_profile": dict(config["cleanup_seed"]),
    }
    seed_candidate_path = config.get("seed_candidate_path")
    if seed_candidate_path:
        loaded = load_json(seed_candidate_path)
        candidate["prompt_slots"].update(loaded.get("prompt_slots", {}))
        candidate["generation"].update(loaded.get("generation", {}))
        candidate["cleanup_profile"].update(loaded.get("cleanup_profile", {}))
    return candidate


def choose_parent(state: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    leaders = state["leaderboard"]
    if not leaders:
        return state["best_candidate"]
    if rng.random() < 0.55:
        return state["best_candidate"]
    idx = rng.randrange(min(len(leaders), 4))
    return leaders[idx]["candidate"]


def choose_second_parent(state: dict[str, Any], first: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    candidates = [row["candidate"] for row in state["leaderboard"][: min(len(state["leaderboard"]), 6)]]
    alternatives = [candidate for candidate in candidates if candidate != first]
    if not alternatives:
        return first
    return rng.choice(alternatives)


def focus_slots_from_report(report: dict[str, Any], config: dict[str, Any]) -> list[str]:
    threshold = float(config["search"].get("repair_threshold", 4.5))
    hinted: list[str] = []
    hints = config.get("case_slot_hints", {})
    for row in report.get("case_results", []):
        if row["score"] < threshold:
            hinted.extend(hints.get(row["id"], []))
    # preserve order, drop duplicates
    seen = set()
    ordered = []
    for slot in hinted:
        if slot not in seen:
            seen.add(slot)
            ordered.append(slot)
    return ordered


def crossover_candidate(
    primary: dict[str, Any],
    secondary: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
    focus_slots: list[str] | None = None,
) -> dict[str, Any]:
    child = json.loads(json.dumps(primary))

    for slot in config["prompt_bias_slots"]:
        take_secondary = rng.random() < (0.8 if focus_slots and slot in focus_slots else 0.35)
        if take_secondary:
            child["prompt_slots"][slot] = secondary["prompt_slots"].get(slot, "")

    for key in config["generation_space"]:
        if rng.random() < 0.3:
            child["generation"][key] = secondary["generation"].get(key, child["generation"][key])

    for key in config["cleanup_space"]:
        if rng.random() < 0.3:
            child["cleanup_profile"][key] = secondary["cleanup_profile"].get(key, child["cleanup_profile"][key])

    return child


def mutate_candidate(
    candidate: dict[str, Any],
    config: dict[str, Any],
    rng: random.Random,
    focus_slots: list[str] | None = None,
) -> dict[str, Any]:
    child = json.loads(json.dumps(candidate))
    mutation_count = rng.choice(config["search"]["mutations_per_candidate"])
    mutation_types = ["prompt", "generation", "cleanup"]

    for _ in range(mutation_count):
        mutation_type = rng.choices(mutation_types, weights=[0.55, 0.2, 0.25], k=1)[0]
        if mutation_type == "prompt":
            if focus_slots and rng.random() < 0.7:
                slot = rng.choice(focus_slots)
            else:
                slot = rng.choice(list(config["prompt_bias_slots"]))
            choices = config["prompt_bias_slots"][slot]
            current = child["prompt_slots"][slot]
            options = [value for value in choices if value != current]
            if options:
                child["prompt_slots"][slot] = rng.choice(options)
        elif mutation_type == "generation":
            key = rng.choice(list(config["generation_space"]))
            current = child["generation"].get(key)
            options = [value for value in config["generation_space"][key] if value != current]
            if options:
                child["generation"][key] = rng.choice(options)
        else:
            key = rng.choice(list(config["cleanup_space"]))
            current = child["cleanup_profile"][key]
            options = [value for value in config["cleanup_space"][key] if value != current]
            if options:
                child["cleanup_profile"][key] = rng.choice(options)

    return child


def load_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    return json.loads(state_path.read_text())


def write_state(output_dir: Path, state: dict[str, Any], best_report: dict[str, Any]) -> None:
    write_json(output_dir / "state.json", state)
    write_json(output_dir / "best_candidate.json", state["best_candidate"])
    write_json(output_dir / "leaderboard.json", state["leaderboard"])
    write_json(output_dir / "best_report.json", best_report)
    (output_dir / "best_prompt.txt").write_text(best_report["system_prompt"])


def append_history(history_path: Path, row: dict[str, Any]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_leaderboard(existing: list[dict[str, Any]], candidate_row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    by_sig: dict[str, dict[str, Any]] = {row["signature"]: row for row in existing}
    previous = by_sig.get(candidate_row["signature"])
    if previous is None or candidate_row["score"] > previous["score"]:
        by_sig[candidate_row["signature"]] = candidate_row
    ranked = sorted(by_sig.values(), key=lambda row: row["score"], reverse=True)
    return ranked[:top_k]


def initialize_state(
    config: dict[str, Any],
    output_dir: Path,
    benchmark: dict[str, Any],
    seed_prompt: str,
    model: Any,
    tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = seed_candidate(config)
    signature = candidate_signature(candidate)
    report = evaluate_candidate(model, tokenizer, benchmark, seed_prompt, candidate)
    state = {
        "config_path": config["_config_path"],
        "iteration": 0,
        "best_score": report["score"],
        "best_signature": signature,
        "best_candidate": candidate,
        "leaderboard": [
            {
                "signature": signature,
                "score": report["score"],
                "candidate": candidate,
            }
        ],
        "seen_signatures": [signature],
        "started_at": time.time(),
        "updated_at": time.time(),
    }
    iteration_report = {
        "iteration": 0,
        "signature": signature,
        "score": report["score"],
        "improved": True,
        "candidate": candidate,
        **report,
    }
    iterations_dir = output_dir / "iterations"
    iterations_dir.mkdir(parents=True, exist_ok=True)
    write_json(iterations_dir / "iteration_000000.json", iteration_report)
    append_history(
        output_dir / "history.jsonl",
        {
            "iteration": 0,
            "signature": signature,
            "score": report["score"],
            "improved": True,
        },
    )
    write_state(output_dir, state, iteration_report)
    return state, iteration_report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_iterations is not None:
        config["search"]["max_iterations"] = args.max_iterations

    output_dir = resolve_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "autoresearch.pid").write_text(str(os.getpid()))
    benchmark = load_json(config["benchmark_path"])
    seed_prompt = load_text(config["seed_prompt_path"]).strip()
    rng = random.Random(config["search"]["random_seed"])

    model_cfg = load_config(config["model"]["config"])
    model_path = resolve_path(config["model"]["model_path"])
    _, backend, model, tokenizer = load_model_and_tokenizer(
        str(model_path),
        max_seq_length=model_cfg["training"]["max_seq_length"],
        load_in_4bit=True,
        full_finetuning=False,
    )
    prepare_model_for_inference(backend, model)

    state_path = output_dir / "state.json"
    if args.fresh:
        state = None
    else:
        state = load_state(state_path)

    if state is None:
        state, best_report = initialize_state(config, output_dir, benchmark, seed_prompt, model, tokenizer)
        print(f"[baseline] score={state['best_score']:.4f} signature={state['best_signature']}")
    else:
        best_report = json.loads((output_dir / "best_report.json").read_text())
        print(
            f"[resume] iteration={state['iteration']} best={state['best_score']:.4f} signature={state['best_signature']}",
            flush=True,
        )

    seen_signatures = set(state.get("seen_signatures", []))
    max_iterations = int(config["search"]["max_iterations"])
    checkpoint_every = int(config["search"]["checkpoint_every"])
    top_k = int(config["search"]["keep_top_k"])
    epsilon = float(config["search"]["improvement_epsilon"])
    crossover_rate = float(config["search"].get("crossover_rate", 0.0))

    try:
        while True:
            if max_iterations > 0 and state["iteration"] >= max_iterations:
                break

            parent = choose_parent(state, rng)
            focus_slots = focus_slots_from_report(best_report, config)
            child = None
            child_signature = None
            for _ in range(64):
                proposal_base = parent
                if crossover_rate > 0 and len(state["leaderboard"]) > 1 and rng.random() < crossover_rate:
                    second_parent = choose_second_parent(state, parent, rng)
                    proposal_base = crossover_candidate(parent, second_parent, config, rng, focus_slots=focus_slots)

                proposal = mutate_candidate(proposal_base, config, rng, focus_slots=focus_slots)
                signature = candidate_signature(proposal)
                if signature not in seen_signatures:
                    child = proposal
                    child_signature = signature
                    break
            if child is None:
                child = mutate_candidate(seed_candidate(config), config, rng, focus_slots=focus_slots)
                child_signature = candidate_signature(child)

            report = evaluate_candidate(model, tokenizer, benchmark, seed_prompt, child)
            iteration = state["iteration"] + 1
            improved = report["score"] > state["best_score"] + epsilon
            seen_signatures.add(child_signature)

            iteration_report = {
                "iteration": iteration,
                "signature": child_signature,
                "score": report["score"],
                "improved": improved,
                "candidate": child,
                **report,
            }
            write_json(output_dir / "iterations" / f"iteration_{iteration:06d}.json", iteration_report)

            append_history(
                output_dir / "history.jsonl",
                {
                    "iteration": iteration,
                    "signature": child_signature,
                    "score": report["score"],
                    "improved": improved,
                },
            )

            leaderboard_entry = {
                "signature": child_signature,
                "score": report["score"],
                "candidate": child,
            }
            state["leaderboard"] = compact_leaderboard(state["leaderboard"], leaderboard_entry, top_k)
            state["iteration"] = iteration
            state["seen_signatures"] = sorted(seen_signatures)
            state["updated_at"] = time.time()

            if improved:
                state["best_score"] = report["score"]
                state["best_signature"] = child_signature
                state["best_candidate"] = child
                best_report = iteration_report

            if improved or iteration % checkpoint_every == 0:
                write_state(output_dir, state, best_report)

            print(
                f"[iter {iteration:05d}] score={report['score']:.4f} best={state['best_score']:.4f} "
                f"improved={'yes' if improved else 'no'} signature={child_signature}",
                flush=True,
            )

    except KeyboardInterrupt:
        print("[stopped] received interrupt, writing state", flush=True)
    finally:
        write_state(output_dir, state, best_report)


if __name__ == "__main__":
    sys.setrecursionlimit(10_000)
    main()
