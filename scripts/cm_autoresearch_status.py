#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show status for the CM autoresearch loop.")
    parser.add_argument("--config", default="configs/cm_autoresearch_v1.json")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    output_dir = (config_path.parent.parent / config["output_dir"]).resolve()
    state_path = output_dir / "state.json"
    pid_path = output_dir / "autoresearch.pid"
    history_path = output_dir / "history.jsonl"

    print(f"config: {config_path}")
    print(f"output_dir: {output_dir}")

    if pid_path.exists():
        pid = int(pid_path.read_text().strip())
        print(f"pid: {pid} ({'running' if is_alive(pid) else 'stale'})")
    else:
        print("pid: none")

    if not state_path.exists():
        print("state: missing")
        return

    state = load_json(state_path)
    print(f"iteration: {state['iteration']}")
    print(f"best_score: {state['best_score']}")
    print(f"best_signature: {state['best_signature']}")

    non_empty_slots = {k: v for k, v in state["best_candidate"]["prompt_slots"].items() if v}
    if non_empty_slots:
        print("best_prompt_slots:")
        for key, value in non_empty_slots.items():
            print(f"  - {key}: {value}")
    else:
        print("best_prompt_slots: none")

    print(f"generation: {state['best_candidate']['generation']}")
    print(f"cleanup_profile: {state['best_candidate']['cleanup_profile']}")

    if history_path.exists():
        lines = history_path.read_text().strip().splitlines()
        print("recent_history:")
        for line in lines[-5:]:
            print(f"  {line}")


if __name__ == "__main__":
    main()
